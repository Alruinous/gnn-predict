from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch_pruning as tp
from timm.models.beit import Beit
from timm.models.convnext import ConvNeXt
from timm.models.vision_transformer import VisionTransformer

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers import BertConfig

    from gnn_archs.config import MutationConfig

IMAGE_MUTATION_TYPES = {
    "ActivationSwap",
    "ActivationFunctionSwap",
    "AddIntermediateFCLayer",
    "AddNormalizationLayer",
    "AddSEBlock",
    "BEiTAttentionHeadPruning",
    "BEiTBlockReduction",
    "BEiTChannelPruning",
    "BEiTGlobalChannelPruning",
    "BEiTLayerPruning",
    "BEiTModifyAttentionHeads",
    "BEiTModifyDropout",
    "BEiTModifyMLPDimension",
    "ChannelPruning",
    "ChannelPruningByIndex",
    "ConvKernelReplacement",
    "ConvNeXtKernelSizeModification",
    "ConvNeXtModifyDropout",
    "ConvNeXtMLPExpansionRatio",
    "ConvNeXtStageReduction",
    "ConvToDepthwiseSeparable",
    "ConvToGroupedConv",
    "GlobalChannelPruning",
    "ModifyDropout",
    "ReplaceFCStructure",
    "ViTBlockReduction",
    "ViTModifyAttentionHeads",
    "ViTModifyDropout",
    "ViTModifyEmbedDim",
    "ViTModifyMLPDimension",
}
TEXT_MUTATION_TYPES = {
    "ActivationFunctionSwap",
    "AttentionHeadsModification",
    "BertAttentionHeadPruning",
    "BertHiddenSizePruning",
    "BertLayerPruning",
    "DropoutModification",
    "HiddenSizeModification",
    "IntermediateSizeModification",
    "PositionEncodingModification",
    "TransformerLayerReduction",
    "VocabSizeModification",
}


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if reduction_ratio <= 0:
            raise ValueError("reduction_ratio must be positive")

        reduced_channels = max(1, channels // reduction_ratio)
        self.channels = channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scale = self.gate(self.pool(inputs))
        return inputs * scale


def apply_image_mutations(
    model: nn.Module,
    mutations: list[MutationConfig],
    target_output_classes: int,
    example_input_shape: list[int],
) -> nn.Module:
    example_inputs = build_image_example_inputs(example_input_shape)
    mutated_model = model
    for mutation in mutations:
        mutation_type = mutation.type
        params = mutation.params

        if mutation_type == "ActivationSwap":
            mutated_model = swap_activations(
                mutated_model,
                target_name=str(params["target_func_name"]),
                new_name=str(params["new_func_name"]),
            )
            continue
        if mutation_type == "ActivationFunctionSwap":
            mutated_model = swap_activations(
                mutated_model,
                target_name=str(params["target_activation"]),
                new_name=str(params["replace_activation"]),
            )
            continue
        if mutation_type == "AddIntermediateFCLayer":
            mutated_model = add_intermediate_fc_layer(
                mutated_model,
                hidden_size=int(params["hidden_size"]),
                dropout_rate=float(params.get("dropout_rate", 0.0)),
                activation_name=str(params.get("activation", "relu")),
                target_output_classes=target_output_classes,
            )
            continue
        if mutation_type == "ReplaceFCStructure":
            mutated_model = replace_fc_structure(
                mutated_model,
                hidden_layers=[int(value) for value in params["fc_structure"]],
                dropout_rates=[
                    float(value) for value in params.get("dropout_rates", [])
                ],
                activations=[str(value) for value in params.get("activations", [])],
                target_output_classes=target_output_classes,
            )
            continue
        if mutation_type == "ConvKernelReplacement":
            mutated_model = replace_conv_kernel(
                mutated_model,
                layer_name=str(params["layer_name"]),
                new_kernel=[int(value) for value in params["new_kernel"]],
                padding=int(params["padding"]),
            )
            continue
        if mutation_type == "ConvToDepthwiseSeparable":
            mutated_model = replace_with_depthwise_separable_conv(
                mutated_model,
                layer_name=str(params["layer_name"]),
            )
            continue
        if mutation_type == "ConvToGroupedConv":
            mutated_model = replace_with_grouped_conv(
                mutated_model,
                layer_name=str(params["layer_name"]),
                groups=int(params["groups"]),
            )
            continue
        if mutation_type == "AddNormalizationLayer":
            mutated_model = add_normalization_layer(
                mutated_model,
                after_layer_name=str(params["after_layer_name"]),
                norm_type=str(params.get("norm_type", "batch_norm")),
                num_groups=int(params.get("num_groups", 32)),
            )
            continue
        if mutation_type == "AddSEBlock":
            mutated_model = add_se_block(
                mutated_model,
                after_layer_name=str(params["after_layer_name"]),
                reduction_ratio=int(params.get("reduction_ratio", 16)),
            )
            continue
        if mutation_type == "ModifyDropout":
            mutated_model = modify_dropout_rate(
                mutated_model,
                layer_name=str(params["layer_name"]),
                new_probability=float(params["new_p"]),
            )
            continue
        if mutation_type == "ChannelPruning":
            mutated_model = prune_channels_by_ratio(
                mutated_model,
                layer_name=str(params["layer_name"]),
                ratio=float(params["ratio"]),
                example_inputs=example_inputs,
            )
            continue
        if mutation_type == "ChannelPruningByIndex":
            mutated_model = prune_channels_by_index(
                mutated_model,
                layer_name=str(params["layer_name"]),
                prune_indices=[int(value) for value in params["prune_indices"]],
                example_inputs=example_inputs,
            )
            continue
        if mutation_type == "GlobalChannelPruning":
            mutated_model = apply_global_channel_pruning(
                mutated_model,
                pruning_ratio=float(params["pruning_ratio"]),
                pruning_method=str(params.get("pruning_method", "l2")),
                ignored_layer_prefixes=[
                    str(value) for value in params.get("ignored_layers", [])
                ],
                layers_to_target=[
                    str(value) for value in params.get("layers_to_target", [])
                ],
                example_inputs=example_inputs,
            )
            continue
        if mutation_type == "ConvNeXtStageReduction":
            mutated_model = modify_convnext_stage_depths(
                mutated_model,
                target_blocks=[int(value) for value in params["target_blocks"]],
            )
            continue
        if mutation_type == "ConvNeXtMLPExpansionRatio":
            mutated_model = modify_convnext_mlp_expansion_ratio(
                mutated_model,
                expansion_ratio=float(params["expansion_ratio"]),
            )
            continue
        if mutation_type == "ConvNeXtKernelSizeModification":
            mutated_model = modify_convnext_depthwise_kernel_size(
                mutated_model,
                kernel_size=int(params["kernel_size"]),
            )
            continue
        if mutation_type == "ConvNeXtModifyDropout":
            mutated_model = modify_convnext_dropout(
                mutated_model,
                new_probability=float(params["new_p"]),
            )
            continue
        if mutation_type == "ViTBlockReduction":
            mutated_model = reduce_vit_block_count(
                mutated_model,
                num_blocks_to_keep=int(params["num_blocks_to_keep"]),
            )
            continue
        if mutation_type == "ViTModifyAttentionHeads":
            mutated_model = modify_vit_attention_heads(
                mutated_model,
                new_num_heads=int(params["new_num_heads"]),
            )
            continue
        if mutation_type == "ViTModifyMLPDimension":
            mutated_model = modify_vit_mlp_dimension(
                mutated_model,
                new_mlp_dim=int(params["new_mlp_dim"]),
            )
            continue
        if mutation_type == "ViTModifyDropout":
            mutated_model = modify_vit_dropout(
                mutated_model,
                dropout_type=str(params["dropout_type"]),
                new_probability=float(params["new_p"]),
            )
            continue
        if mutation_type == "ViTModifyEmbedDim":
            mutated_model = modify_vit_embed_dim(
                mutated_model,
                new_embed_dim=int(params["new_embed_dim"]),
                new_num_heads=int(params["new_num_heads"]),
                new_mlp_dim=int(params["new_mlp_dim"]),
            )
            continue
        if mutation_type == "BEiTBlockReduction":
            mutated_model = reduce_beit_block_count(
                mutated_model,
                num_blocks_to_keep=int(params["num_blocks_to_keep"]),
            )
            continue
        if mutation_type == "BEiTLayerPruning":
            mutated_model = prune_beit_layers(
                mutated_model,
                target_layers=int(params["target_layers"]),
                removal_strategy=str(params.get("removal_strategy", "remove_last")),
            )
            continue
        if mutation_type == "BEiTModifyMLPDimension":
            mutated_model = modify_beit_mlp_dimension(
                mutated_model,
                new_mlp_dim=int(params["new_mlp_dim"]),
            )
            continue
        if mutation_type == "BEiTModifyDropout":
            mutated_model = modify_beit_dropout(
                mutated_model,
                dropout_type=str(params["dropout_type"]),
                new_probability=float(params["new_p"]),
            )
            continue
        if mutation_type == "BEiTModifyAttentionHeads":
            mutated_model = modify_beit_attention_heads(
                mutated_model,
                new_num_heads=int(params["new_num_heads"]),
            )
            continue
        if mutation_type == "BEiTAttentionHeadPruning":
            mutated_model = prune_beit_attention_heads(
                mutated_model,
                head_pruning_ratio=float(params["head_pruning_ratio"]),
                layer_pruning_ratio=float(params.get("layer_pruning_ratio", 0.0)),
            )
            continue
        if mutation_type == "BEiTChannelPruning":
            mutated_model = apply_beit_channel_pruning(
                mutated_model,
                layer_name=str(params["layer_name"]),
                ratio=float(params["ratio"]),
            )
            continue
        if mutation_type == "BEiTGlobalChannelPruning":
            mutated_model = apply_beit_global_channel_pruning(
                mutated_model,
                pruning_ratio=float(params["pruning_ratio"]),
                pruning_method=str(params.get("pruning_method", "l2")),
                ignored_layer_prefixes=[
                    str(value) for value in params.get("ignored_layers", [])
                ],
                layers_to_target=[
                    str(value) for value in params.get("layers_to_target", [])
                ],
            )
            continue

        raise NotImplementedError(f"unsupported image mutation type: {mutation_type}")

    return mutated_model


def apply_text_config_mutations(
    config: BertConfig, mutations: list[MutationConfig]
) -> BertConfig:
    mutated_config = config

    for mutation in mutations:
        mutation_type = mutation.type
        params = mutation.params

        if mutation_type in {"TransformerLayerReduction", "BertLayerPruning"}:
            target_layers = int(params["target_layers"])
            if target_layers <= 0:
                raise ValueError("target_layers must be positive")
            mutated_config.num_hidden_layers = target_layers
            continue

        if mutation_type == "AttentionHeadsModification":
            mutated_config.num_attention_heads = int(params["num_heads"])
            continue

        if mutation_type == "HiddenSizeModification":
            mutated_config.hidden_size = int(params["hidden_size"])
            continue

        if mutation_type == "IntermediateSizeModification":
            mutated_config.intermediate_size = int(params["intermediate_size"])
            continue

        if mutation_type == "DropoutModification":
            mutated_config.hidden_dropout_prob = float(
                params.get("hidden_dropout_prob", mutated_config.hidden_dropout_prob)
            )
            mutated_config.attention_probs_dropout_prob = float(
                params.get(
                    "attention_probs_dropout_prob",
                    mutated_config.attention_probs_dropout_prob,
                )
            )
            continue

        if mutation_type == "ActivationFunctionSwap":
            mutated_config.hidden_act = str(params.get("new_activation", "gelu"))
            continue

        if mutation_type == "PositionEncodingModification":
            mutated_config.max_position_embeddings = int(
                params["max_position_embeddings"]
            )
            continue

        if mutation_type == "VocabSizeModification":
            mutated_config.vocab_size = int(params["vocab_size"])
            continue

        if mutation_type == "BertAttentionHeadPruning":
            head_ratio = float(params.get("head_pruning_ratio", 0.0))
            if not 0.0 <= head_ratio < 1.0:
                raise ValueError("head_pruning_ratio must be in [0, 1)")
            original_heads = mutated_config.num_attention_heads
            remaining_heads = max(1, int(original_heads * (1.0 - head_ratio)))
            while (
                mutated_config.hidden_size % remaining_heads != 0
                and remaining_heads > 1
            ):
                remaining_heads -= 1
            if mutated_config.hidden_size % remaining_heads != 0:
                raise ValueError(
                    "BertAttentionHeadPruning produced an invalid hidden/head size pair"
                )
            mutated_config.num_attention_heads = remaining_heads
            continue

        if mutation_type == "BertHiddenSizePruning":
            pruning_ratio = float(
                params.get(
                    "hidden_size_pruning_ratio",
                    params.get("pruning_ratio", 0.0),
                )
            )
            if not 0.0 <= pruning_ratio < 1.0:
                raise ValueError("BertHiddenSizePruning ratio must be in [0, 1)")
            new_hidden_size = int(mutated_config.hidden_size * (1.0 - pruning_ratio))
            remainder = new_hidden_size % mutated_config.num_attention_heads
            if remainder:
                lower_hidden_size = new_hidden_size - remainder
                upper_hidden_size = (
                    lower_hidden_size + mutated_config.num_attention_heads
                )
                if (
                    new_hidden_size - lower_hidden_size
                    <= upper_hidden_size - new_hidden_size
                ):
                    new_hidden_size = lower_hidden_size
                else:
                    new_hidden_size = upper_hidden_size
            mutated_config.hidden_size = new_hidden_size
            continue

        raise NotImplementedError(f"unsupported text mutation type: {mutation_type}")

    validate_bert_config(mutated_config)
    return mutated_config


def validate_mutation_types(
    mutations: list[MutationConfig], allowed_types: set[str], model_kind: str
) -> None:
    for mutation in mutations:
        if mutation.type not in allowed_types:
            raise NotImplementedError(
                f"{model_kind} model does not support "
                f"mutation type '{mutation.type}' yet"
            )


def validate_bert_config(config: BertConfig) -> None:
    if config.hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if config.num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    if config.num_attention_heads <= 0:
        raise ValueError("num_attention_heads must be positive")
    if config.intermediate_size <= 0:
        raise ValueError("intermediate_size must be positive")
    if config.vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if config.max_position_embeddings <= 0:
        raise ValueError("max_position_embeddings must be positive")
    if config.hidden_size % config.num_attention_heads != 0:
        raise ValueError(
            "hidden_size must be divisible by num_attention_heads "
            "after applying text mutations"
        )


def reduce_vit_block_count(model: nn.Module, num_blocks_to_keep: int) -> nn.Module:
    vit_model = require_vit_model(model, "ViTBlockReduction")
    if num_blocks_to_keep <= 0:
        raise ValueError("ViTBlockReduction num_blocks_to_keep must be positive")
    if num_blocks_to_keep > len(vit_model.blocks):
        raise ValueError(
            "ViTBlockReduction num_blocks_to_keep "
            f"{num_blocks_to_keep} exceeds current depth {len(vit_model.blocks)}"
        )
    if num_blocks_to_keep == len(vit_model.blocks):
        return model
    return rebuild_vit_model(vit_model, depth=num_blocks_to_keep)


def modify_vit_attention_heads(model: nn.Module, new_num_heads: int) -> nn.Module:
    vit_model = require_vit_model(model, "ViTModifyAttentionHeads")
    if new_num_heads <= 0:
        raise ValueError("ViTModifyAttentionHeads new_num_heads must be positive")
    if vit_model.embed_dim % new_num_heads != 0:
        raise ValueError(
            "ViTModifyAttentionHeads requires embed_dim "
            f"{vit_model.embed_dim} to be divisible by new_num_heads {new_num_heads}"
        )
    return rebuild_vit_model(vit_model, num_heads=new_num_heads)


def modify_vit_mlp_dimension(model: nn.Module, new_mlp_dim: int) -> nn.Module:
    vit_model = require_vit_model(model, "ViTModifyMLPDimension")
    if new_mlp_dim <= 0:
        raise ValueError("ViTModifyMLPDimension new_mlp_dim must be positive")
    return rebuild_vit_model(vit_model, mlp_hidden_dim=new_mlp_dim)


def modify_vit_dropout(
    model: nn.Module,
    dropout_type: str,
    new_probability: float,
) -> nn.Module:
    vit_model = require_vit_model(model, "ViTModifyDropout")
    if not 0.0 <= new_probability <= 1.0:
        raise ValueError("ViTModifyDropout new_probability must be in [0, 1]")

    normalized_type = dropout_type.strip().lower()
    if normalized_type == "attention":
        for block in vit_model.blocks:
            block.attn.attn_drop.p = new_probability
            block.attn.proj_drop.p = new_probability
        return model
    if normalized_type == "mlp":
        for block in vit_model.blocks:
            for attr_name in ("drop1", "drop2", "drop"):
                dropout_layer = getattr(block.mlp, attr_name, None)
                if isinstance(dropout_layer, nn.Dropout):
                    dropout_layer.p = new_probability
        return model
    raise NotImplementedError(
        f"unsupported ViTModifyDropout dropout_type: {dropout_type}"
    )


def modify_vit_embed_dim(
    model: nn.Module,
    new_embed_dim: int,
    new_num_heads: int,
    new_mlp_dim: int,
) -> nn.Module:
    vit_model = require_vit_model(model, "ViTModifyEmbedDim")
    if new_embed_dim <= 0:
        raise ValueError("ViTModifyEmbedDim new_embed_dim must be positive")
    if new_num_heads <= 0:
        raise ValueError("ViTModifyEmbedDim new_num_heads must be positive")
    if new_mlp_dim <= 0:
        raise ValueError("ViTModifyEmbedDim new_mlp_dim must be positive")
    if new_embed_dim % new_num_heads != 0:
        raise ValueError(
            "ViTModifyEmbedDim requires new_embed_dim "
            f"{new_embed_dim} to be divisible by new_num_heads {new_num_heads}"
        )
    return rebuild_vit_model(
        vit_model,
        embed_dim=new_embed_dim,
        num_heads=new_num_heads,
        mlp_hidden_dim=new_mlp_dim,
    )


def rebuild_vit_model(
    model: VisionTransformer,
    *,
    depth: int | None = None,
    embed_dim: int | None = None,
    num_heads: int | None = None,
    mlp_hidden_dim: int | None = None,
) -> VisionTransformer:
    reference_block = model.blocks[0]
    target_depth = depth if depth is not None else len(model.blocks)
    target_embed_dim = embed_dim if embed_dim is not None else model.embed_dim
    target_num_heads = (
        num_heads if num_heads is not None else reference_block.attn.num_heads
    )
    target_mlp_hidden_dim = (
        mlp_hidden_dim
        if mlp_hidden_dim is not None
        else reference_block.mlp.fc1.out_features
    )
    if target_depth <= 0:
        raise ValueError("ViT rebuild depth must be positive")
    if target_embed_dim <= 0:
        raise ValueError("ViT rebuild embed_dim must be positive")
    if target_num_heads <= 0:
        raise ValueError("ViT rebuild num_heads must be positive")
    if target_mlp_hidden_dim <= 0:
        raise ValueError("ViT rebuild mlp_hidden_dim must be positive")
    if target_embed_dim % target_num_heads != 0:
        raise ValueError(
            "ViT rebuild requires embed_dim "
            f"{target_embed_dim} to be divisible by num_heads {target_num_heads}"
        )

    mlp_ratio = target_mlp_hidden_dim / target_embed_dim
    if mlp_ratio <= 0:
        raise ValueError("ViT rebuild mlp_ratio must be positive")

    parameter = next(model.parameters())
    rebuilt_model = VisionTransformer(
        img_size=model.patch_embed.img_size,
        patch_size=model.patch_embed.patch_size,
        in_chans=model.patch_embed.proj.in_channels,
        num_classes=get_vit_num_classes(model),
        global_pool=model.global_pool,
        embed_dim=target_embed_dim,
        depth=target_depth,
        num_heads=target_num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=reference_block.attn.qkv.bias is not None,
        qk_norm=not isinstance(reference_block.attn.q_norm, nn.Identity),
        proj_bias=reference_block.attn.proj.bias is not None,
        class_token=model.cls_token is not None,
        no_embed_class=model.no_embed_class,
        reg_tokens=getattr(model, "num_reg_tokens", 0),
        pre_norm=not isinstance(getattr(model, "norm_pre", nn.Identity()), nn.Identity),
        final_norm=not isinstance(model.norm, nn.Identity),
        fc_norm=not isinstance(getattr(model, "fc_norm", nn.Identity()), nn.Identity),
        dynamic_img_size=getattr(model, "dynamic_img_size", False),
        dynamic_img_pad=bool(getattr(model, "dynamic_img_pad", False)),
        drop_rate=get_dropout_probability(getattr(model, "head_drop", None)),
        pos_drop_rate=get_dropout_probability(getattr(model, "pos_drop", None)),
        attn_drop_rate=get_dropout_probability(reference_block.attn.attn_drop),
        proj_drop_rate=get_dropout_probability(reference_block.attn.proj_drop),
        drop_path_rate=get_vit_drop_path_probability(model),
        norm_layer=type(reference_block.norm1),
        act_layer=type(reference_block.mlp.act),
        device=parameter.device,
        dtype=parameter.dtype,
    )
    sync_vit_dropout_configuration(model, rebuilt_model)
    return rebuilt_model


def require_vit_model(model: nn.Module, mutation_type: str) -> VisionTransformer:
    if not isinstance(model, VisionTransformer):
        raise TypeError(
            f"{mutation_type} requires a timm VisionTransformer, got {type(model)}"
        )
    if not model.blocks:
        raise ValueError(f"{mutation_type} requires at least one transformer block")
    return model


def get_vit_num_classes(model: VisionTransformer) -> int:
    if isinstance(model.head, nn.Linear):
        return model.head.out_features
    if isinstance(model.head, nn.Identity):
        return 0
    raise TypeError(f"unsupported ViT head type: {type(model.head)}")


def get_dropout_probability(module: nn.Module | None) -> float:
    if module is None:
        return 0.0
    if isinstance(module, nn.Dropout):
        return float(module.p)
    return float(getattr(module, "drop_prob", 0.0))


def get_vit_drop_path_probability(model: VisionTransformer) -> float:
    last_block = model.blocks[-1]
    return max(
        get_dropout_probability(getattr(last_block, "drop_path", None)),
        get_dropout_probability(getattr(last_block, "drop_path1", None)),
        get_dropout_probability(getattr(last_block, "drop_path2", None)),
    )


def sync_vit_dropout_configuration(
    source_model: VisionTransformer,
    target_model: VisionTransformer,
) -> None:
    if isinstance(getattr(target_model, "pos_drop", None), nn.Dropout):
        target_model.pos_drop.p = get_dropout_probability(source_model.pos_drop)
    if isinstance(getattr(target_model, "head_drop", None), nn.Dropout):
        target_model.head_drop.p = get_dropout_probability(source_model.head_drop)

    for source_block, target_block in zip(
        source_model.blocks, target_model.blocks, strict=False
    ):
        target_block.attn.attn_drop.p = source_block.attn.attn_drop.p
        target_block.attn.proj_drop.p = source_block.attn.proj_drop.p
        for attr_name in ("drop1", "drop2", "drop"):
            source_dropout = getattr(source_block.mlp, attr_name, None)
            target_dropout = getattr(target_block.mlp, attr_name, None)
            if isinstance(source_dropout, nn.Dropout) and isinstance(
                target_dropout, nn.Dropout
            ):
                target_dropout.p = source_dropout.p


def reduce_beit_block_count(model: nn.Module, num_blocks_to_keep: int) -> nn.Module:
    beit_model = require_beit_model(model, "BEiTBlockReduction")
    if num_blocks_to_keep <= 0:
        raise ValueError("BEiTBlockReduction num_blocks_to_keep must be positive")
    if num_blocks_to_keep > len(beit_model.blocks):
        raise ValueError(
            "BEiTBlockReduction num_blocks_to_keep "
            f"{num_blocks_to_keep} exceeds current depth {len(beit_model.blocks)}"
        )
    if num_blocks_to_keep == len(beit_model.blocks):
        return model
    beit_model.blocks = nn.ModuleList(list(beit_model.blocks[:num_blocks_to_keep]))
    return model


def prune_beit_layers(
    model: nn.Module,
    target_layers: int,
    removal_strategy: str,
) -> nn.Module:
    beit_model = require_beit_model(model, "BEiTLayerPruning")
    original_layer_count = len(beit_model.blocks)
    if target_layers <= 0:
        raise ValueError("BEiTLayerPruning target_layers must be positive")
    if target_layers > original_layer_count:
        raise ValueError(
            "BEiTLayerPruning target_layers "
            f"{target_layers} exceeds current depth {original_layer_count}"
        )
    if target_layers == original_layer_count:
        return model

    blocks = list(beit_model.blocks)
    normalized_strategy = removal_strategy.strip().lower()
    if normalized_strategy == "remove_last":
        kept_blocks = blocks[:target_layers]
    elif normalized_strategy == "remove_first":
        kept_blocks = blocks[-target_layers:]
    elif normalized_strategy == "remove_middle":
        removed_layers = original_layer_count - target_layers
        start_index = removed_layers // 2
        kept_blocks = blocks[start_index : start_index + target_layers]
    else:
        raise NotImplementedError(
            f"unsupported BEiTLayerPruning removal_strategy: {removal_strategy}"
        )

    beit_model.blocks = nn.ModuleList(kept_blocks)
    return model


def modify_beit_mlp_dimension(model: nn.Module, new_mlp_dim: int) -> nn.Module:
    beit_model = require_beit_model(model, "BEiTModifyMLPDimension")
    if new_mlp_dim <= 0:
        raise ValueError("BEiTModifyMLPDimension new_mlp_dim must be positive")

    for block in beit_model.blocks:
        block.mlp.fc1 = resize_linear_layer(
            block.mlp.fc1,
            new_out_features=new_mlp_dim,
        )
        block.mlp.fc2 = resize_linear_layer(
            block.mlp.fc2,
            new_in_features=new_mlp_dim,
        )
    return model


def modify_beit_dropout(
    model: nn.Module,
    dropout_type: str,
    new_probability: float,
) -> nn.Module:
    beit_model = require_beit_model(model, "BEiTModifyDropout")
    if not 0.0 <= new_probability <= 1.0:
        raise ValueError("BEiTModifyDropout new_probability must be in [0, 1]")

    normalized_type = dropout_type.strip().lower()
    if normalized_type == "attention":
        for block in beit_model.blocks:
            block.attn.attn_drop.p = new_probability
            block.attn.proj_drop.p = new_probability
        return model
    if normalized_type == "mlp":
        for block in beit_model.blocks:
            for attr_name in ("drop1", "drop2", "drop"):
                dropout_layer = getattr(block.mlp, attr_name, None)
                if isinstance(dropout_layer, nn.Dropout):
                    dropout_layer.p = new_probability
        return model
    raise NotImplementedError(
        f"unsupported BEiTModifyDropout dropout_type: {dropout_type}"
    )


def modify_beit_attention_heads(model: nn.Module, new_num_heads: int) -> nn.Module:
    beit_model = require_beit_model(model, "BEiTModifyAttentionHeads")
    if new_num_heads <= 0:
        raise ValueError("BEiTModifyAttentionHeads new_num_heads must be positive")

    target_embed_dim = resolve_compatible_embed_dim(beit_model.embed_dim, new_num_heads)
    return resize_beit_embedding(
        model,
        target_embed_dim=target_embed_dim,
        target_num_heads=new_num_heads,
    )


def prune_beit_attention_heads(
    model: nn.Module,
    head_pruning_ratio: float,
    layer_pruning_ratio: float,
) -> nn.Module:
    beit_model = require_beit_model(model, "BEiTAttentionHeadPruning")
    if not 0.0 <= head_pruning_ratio < 1.0:
        raise ValueError(
            "BEiTAttentionHeadPruning head_pruning_ratio must be in [0, 1)"
        )
    if not 0.0 <= layer_pruning_ratio < 1.0:
        raise ValueError(
            "BEiTAttentionHeadPruning layer_pruning_ratio must be in [0, 1)"
        )

    current_num_heads = beit_model.blocks[0].attn.num_heads
    target_num_heads = max(1, int(current_num_heads * (1.0 - head_pruning_ratio)))
    if target_num_heads == current_num_heads and head_pruning_ratio > 0.0:
        target_num_heads = max(1, current_num_heads - 1)
    model = modify_beit_attention_heads(model, target_num_heads)

    if layer_pruning_ratio > 0.0:
        current_depth = len(beit_model.blocks)
        target_layers = max(1, int(current_depth * (1.0 - layer_pruning_ratio)))
        model = prune_beit_layers(
            model,
            target_layers=target_layers,
            removal_strategy="remove_last",
        )
    return model


def apply_beit_channel_pruning(
    model: nn.Module,
    layer_name: str,
    ratio: float,
) -> nn.Module:
    beit_model = require_beit_model(model, "BEiTChannelPruning")
    if not 0.0 < ratio < 1.0:
        raise ValueError("BEiTChannelPruning ratio must be in (0, 1)")

    if layer_name in {"patch_embed.proj", "head"} or layer_name.endswith(
        (".attn.qkv", ".attn.proj")
    ):
        current_embed_dim = beit_model.embed_dim
        prune_count = int(current_embed_dim * ratio)
        if prune_count <= 0:
            raise ValueError(
                "BEiTChannelPruning ratio "
                f"{ratio} prunes zero channels for '{layer_name}'"
            )
        if prune_count >= current_embed_dim:
            raise ValueError(
                "BEiTChannelPruning ratio "
                f"{ratio} would remove every channel from '{layer_name}'"
            )
        raw_target_embed_dim = current_embed_dim - prune_count
        target_embed_dim = resolve_compatible_embed_dim(
            raw_target_embed_dim,
            beit_model.blocks[0].attn.num_heads,
        )
        return resize_beit_embedding(
            model,
            target_embed_dim=target_embed_dim,
            target_num_heads=beit_model.blocks[0].attn.num_heads,
        )

    if layer_name.endswith(".mlp.fc1"):
        target_layer = get_module(beit_model, layer_name)
        if not isinstance(target_layer, nn.Linear):
            raise TypeError(f"layer '{layer_name}' is not an nn.Linear module")

        prune_count = int(target_layer.out_features * ratio)
        if prune_count <= 0:
            raise ValueError(
                "BEiTChannelPruning ratio "
                f"{ratio} prunes zero channels for '{layer_name}'"
            )
        if prune_count >= target_layer.out_features:
            raise ValueError(
                "BEiTChannelPruning ratio "
                f"{ratio} would remove every channel from '{layer_name}'"
            )

        new_mlp_dim = target_layer.out_features - prune_count
        parent_name, _, _ = layer_name.rpartition(".")
        block_name, _, _ = parent_name.rpartition(".")
        block = get_module(beit_model, block_name)
        block.mlp.fc1 = resize_linear_layer(
            block.mlp.fc1,
            new_out_features=new_mlp_dim,
        )
        block.mlp.fc2 = resize_linear_layer(
            block.mlp.fc2,
            new_in_features=new_mlp_dim,
        )
        return model

    raise NotImplementedError(
        f"unsupported BEiTChannelPruning layer_name: {layer_name}"
    )


def apply_beit_global_channel_pruning(
    model: nn.Module,
    pruning_ratio: float,
    pruning_method: str,
    ignored_layer_prefixes: list[str],
    layers_to_target: list[str],
) -> nn.Module:
    del pruning_method, ignored_layer_prefixes, layers_to_target

    beit_model = require_beit_model(model, "BEiTGlobalChannelPruning")
    if not 0.0 < pruning_ratio < 1.0:
        raise ValueError("BEiTGlobalChannelPruning pruning_ratio must be in (0, 1)")

    current_embed_dim = beit_model.embed_dim
    prune_count = int(current_embed_dim * pruning_ratio)
    if prune_count <= 0:
        raise ValueError(
            "BEiTGlobalChannelPruning pruning_ratio prunes zero embedding channels"
        )
    if prune_count >= current_embed_dim:
        raise ValueError(
            "BEiTGlobalChannelPruning pruning_ratio "
            "would remove every embedding channel"
        )

    raw_target_embed_dim = current_embed_dim - prune_count
    target_embed_dim = resolve_compatible_embed_dim(
        raw_target_embed_dim,
        beit_model.blocks[0].attn.num_heads,
    )
    return resize_beit_embedding(
        model,
        target_embed_dim=target_embed_dim,
        target_num_heads=beit_model.blocks[0].attn.num_heads,
    )


def modify_convnext_stage_depths(
    model: nn.Module,
    target_blocks: list[int],
) -> nn.Module:
    convnext_model = require_convnext_model(model, "ConvNeXtStageReduction")
    if len(target_blocks) != len(convnext_model.stages):
        raise ValueError(
            "ConvNeXtStageReduction target_blocks length "
            f"{len(target_blocks)} does not match stage count "
            f"{len(convnext_model.stages)}"
        )

    for stage_index, desired_length in enumerate(target_blocks):
        if desired_length <= 0:
            raise ValueError("ConvNeXtStageReduction target_blocks must be positive")
        stage = convnext_model.stages[stage_index]
        current_blocks = list(stage.blocks.children())
        stage.blocks = resize_sequential_blocks(current_blocks, desired_length)
    return model


def modify_convnext_mlp_expansion_ratio(
    model: nn.Module,
    expansion_ratio: float,
) -> nn.Module:
    convnext_model = require_convnext_model(model, "ConvNeXtMLPExpansionRatio")
    if expansion_ratio <= 0:
        raise ValueError("ConvNeXtMLPExpansionRatio expansion_ratio must be positive")

    for stage in convnext_model.stages:
        for block in stage.blocks:
            input_dim = block.mlp.fc1.in_features
            target_hidden_dim = round(input_dim * expansion_ratio)
            if target_hidden_dim <= 0:
                raise ValueError(
                    "ConvNeXtMLPExpansionRatio target hidden dimension must be positive"
                )
            block.mlp.fc1 = resize_linear_layer(
                block.mlp.fc1,
                new_out_features=target_hidden_dim,
            )
            block.mlp.fc2 = resize_linear_layer(
                block.mlp.fc2,
                new_in_features=target_hidden_dim,
            )
    return model


def modify_convnext_depthwise_kernel_size(
    model: nn.Module,
    kernel_size: int,
) -> nn.Module:
    convnext_model = require_convnext_model(model, "ConvNeXtKernelSizeModification")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            "ConvNeXtKernelSizeModification kernel_size must be a positive odd integer"
        )

    for stage in convnext_model.stages:
        for block in stage.blocks:
            block.conv_dw = resize_conv2d_kernel(block.conv_dw, kernel_size)
    return model


def modify_convnext_dropout(
    model: nn.Module,
    new_probability: float,
) -> nn.Module:
    convnext_model = require_convnext_model(model, "ConvNeXtModifyDropout")
    if not 0.0 <= new_probability <= 1.0:
        raise ValueError("ConvNeXtModifyDropout new_probability must be in [0, 1]")

    updated_layers = 0
    for stage in convnext_model.stages:
        for block in stage.blocks:
            for attr_name in ("drop1", "drop2"):
                dropout_layer = getattr(block.mlp, attr_name, None)
                if isinstance(dropout_layer, nn.Dropout):
                    dropout_layer.p = new_probability
                    updated_layers += 1

    head_dropout = getattr(convnext_model.head, "drop", None)
    if isinstance(head_dropout, nn.Dropout):
        head_dropout.p = new_probability
        updated_layers += 1

    if updated_layers == 0:
        raise ValueError("ConvNeXtModifyDropout did not find any dropout modules")
    return model


def require_beit_model(model: nn.Module, mutation_type: str) -> Beit:
    if not isinstance(model, Beit):
        raise TypeError(
            f"{mutation_type} requires a timm Beit model, got {type(model)}"
        )
    if not model.blocks:
        raise ValueError(f"{mutation_type} requires at least one transformer block")
    return model


def require_convnext_model(model: nn.Module, mutation_type: str) -> ConvNeXt:
    if not isinstance(model, ConvNeXt):
        raise TypeError(
            f"{mutation_type} requires a timm ConvNeXt model, got {type(model)}"
        )
    if not model.stages:
        raise ValueError(f"{mutation_type} requires at least one stage")
    return model


def resolve_compatible_embed_dim(current_embed_dim: int, num_heads: int) -> int:
    if current_embed_dim <= 0:
        raise ValueError("current_embed_dim must be positive")
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    remainder = current_embed_dim % num_heads
    if remainder == 0:
        return current_embed_dim

    lower = current_embed_dim - remainder
    upper = lower + num_heads
    if lower < num_heads:
        return upper
    if current_embed_dim - lower <= upper - current_embed_dim:
        return lower
    return upper


def resize_beit_embedding(
    model: nn.Module,
    *,
    target_embed_dim: int,
    target_num_heads: int,
) -> nn.Module:
    beit_model = require_beit_model(model, "BEiT embedding resize")
    if target_embed_dim <= 0:
        raise ValueError("target_embed_dim must be positive")
    if target_num_heads <= 0:
        raise ValueError("target_num_heads must be positive")
    if target_embed_dim % target_num_heads != 0:
        raise ValueError(
            "BEiT embedding resize requires target_embed_dim "
            f"{target_embed_dim} to be divisible by target_num_heads {target_num_heads}"
        )

    target_head_dim = target_embed_dim // target_num_heads

    beit_model.patch_embed.proj = resize_conv2d_out_channels(
        beit_model.patch_embed.proj,
        new_out_channels=target_embed_dim,
    )
    if hasattr(beit_model.patch_embed, "embed_dim"):
        beit_model.patch_embed.embed_dim = target_embed_dim
    if isinstance(getattr(beit_model.patch_embed, "norm", None), nn.LayerNorm):
        beit_model.patch_embed.norm = resize_layer_norm(
            beit_model.patch_embed.norm,
            target_embed_dim,
        )

    if beit_model.cls_token is not None:
        beit_model.cls_token = nn.Parameter(
            resize_parameter_last_dimension(
                beit_model.cls_token.detach(),
                target_embed_dim,
            )
        )
    if getattr(beit_model, "pos_embed", None) is not None:
        beit_model.pos_embed = nn.Parameter(
            resize_parameter_last_dimension(
                beit_model.pos_embed.detach(),
                target_embed_dim,
            )
        )
    if getattr(beit_model, "rel_pos_bias", None) is not None:
        beit_model.rel_pos_bias.relative_position_bias_table = nn.Parameter(
            resize_beit_attention_bias_table(
                beit_model.rel_pos_bias.relative_position_bias_table.detach(),
                target_num_heads,
            )
        )

    for block in beit_model.blocks:
        attention = block.attn
        attention.qkv = resize_beit_qkv_layer(
            attention.qkv,
            target_embed_dim=target_embed_dim,
        )
        for attr_name in ("q_bias", "k_bias", "v_bias"):
            bias = getattr(attention, attr_name, None)
            if isinstance(bias, nn.Parameter):
                setattr(
                    attention,
                    attr_name,
                    nn.Parameter(resize_vector(bias.detach(), target_embed_dim)),
                )
            elif isinstance(bias, torch.Tensor):
                setattr(
                    attention,
                    attr_name,
                    resize_vector(bias.detach(), target_embed_dim),
                )
        attention.proj = resize_linear_layer(
            attention.proj,
            new_in_features=target_embed_dim,
            new_out_features=target_embed_dim,
        )
        if getattr(attention, "relative_position_bias_table", None) is not None:
            attention.relative_position_bias_table = nn.Parameter(
                resize_beit_attention_bias_table(
                    attention.relative_position_bias_table.detach(),
                    target_num_heads,
                )
            )
        attention.num_heads = target_num_heads
        attention.scale = target_head_dim**-0.5

        block.mlp.fc1 = resize_linear_layer(
            block.mlp.fc1,
            new_in_features=target_embed_dim,
        )
        block.mlp.fc2 = resize_linear_layer(
            block.mlp.fc2,
            new_out_features=target_embed_dim,
        )
        block.norm1 = resize_layer_norm(block.norm1, target_embed_dim)
        block.norm2 = resize_layer_norm(block.norm2, target_embed_dim)

        if getattr(block, "gamma_1", None) is not None:
            block.gamma_1 = nn.Parameter(
                resize_vector(block.gamma_1.detach(), target_embed_dim)
            )
        if getattr(block, "gamma_2", None) is not None:
            block.gamma_2 = nn.Parameter(
                resize_vector(block.gamma_2.detach(), target_embed_dim)
            )

    if isinstance(beit_model.norm, nn.LayerNorm):
        beit_model.norm = resize_layer_norm(beit_model.norm, target_embed_dim)
    if isinstance(getattr(beit_model, "fc_norm", None), nn.LayerNorm):
        beit_model.fc_norm = resize_layer_norm(beit_model.fc_norm, target_embed_dim)
    if isinstance(beit_model.head, nn.Linear):
        beit_model.head = resize_linear_layer(
            beit_model.head,
            new_in_features=target_embed_dim,
        )

    beit_model.embed_dim = target_embed_dim
    if hasattr(beit_model, "num_features"):
        beit_model.num_features = target_embed_dim
    return model


def resize_parameter_last_dimension(
    values: torch.Tensor,
    new_last_dim: int,
) -> torch.Tensor:
    if new_last_dim <= 0:
        raise ValueError("new_last_dim must be positive")
    new_shape = list(values.shape)
    new_shape[-1] = new_last_dim
    resized = values.new_zeros(new_shape)
    shared_dim = min(values.shape[-1], new_last_dim)
    resized[..., :shared_dim] = values[..., :shared_dim]
    return resized


def resize_vector(values: torch.Tensor, new_dim: int) -> torch.Tensor:
    if new_dim <= 0:
        raise ValueError("new_dim must be positive")
    resized = values.new_zeros(new_dim)
    shared_dim = min(values.shape[0], new_dim)
    resized[:shared_dim] = values[:shared_dim]
    return resized


def resize_matrix(
    values: torch.Tensor,
    new_out_dim: int,
    new_in_dim: int,
) -> torch.Tensor:
    if new_out_dim <= 0 or new_in_dim <= 0:
        raise ValueError("matrix dimensions must be positive")
    resized = values.new_zeros((new_out_dim, new_in_dim))
    shared_out_dim = min(values.shape[0], new_out_dim)
    shared_in_dim = min(values.shape[1], new_in_dim)
    resized[:shared_out_dim, :shared_in_dim] = values[:shared_out_dim, :shared_in_dim]
    return resized


def resize_linear_layer(
    layer: nn.Linear,
    *,
    new_in_features: int | None = None,
    new_out_features: int | None = None,
) -> nn.Linear:
    target_in_features = (
        new_in_features if new_in_features is not None else layer.in_features
    )
    target_out_features = (
        new_out_features if new_out_features is not None else layer.out_features
    )
    new_layer = nn.Linear(
        target_in_features,
        target_out_features,
        bias=layer.bias is not None,
        device=layer.weight.device,
        dtype=layer.weight.dtype,
    )
    with torch.no_grad():
        new_layer.weight.copy_(
            resize_matrix(
                layer.weight.detach(),
                target_out_features,
                target_in_features,
            )
        )
        if layer.bias is not None and new_layer.bias is not None:
            new_layer.bias.copy_(
                resize_vector(layer.bias.detach(), target_out_features)
            )
    return new_layer


def resize_beit_qkv_layer(
    layer: nn.Linear,
    *,
    target_embed_dim: int,
) -> nn.Linear:
    new_layer = nn.Linear(
        target_embed_dim,
        target_embed_dim * 3,
        bias=layer.bias is not None,
        device=layer.weight.device,
        dtype=layer.weight.dtype,
    )
    with torch.no_grad():
        q_weight, k_weight, v_weight = torch.chunk(layer.weight.detach(), 3, dim=0)
        resized_q_weight = resize_matrix(q_weight, target_embed_dim, target_embed_dim)
        resized_k_weight = resize_matrix(k_weight, target_embed_dim, target_embed_dim)
        resized_v_weight = resize_matrix(v_weight, target_embed_dim, target_embed_dim)
        new_layer.weight.copy_(
            torch.cat(
                [resized_q_weight, resized_k_weight, resized_v_weight],
                dim=0,
            )
        )
        if layer.bias is not None and new_layer.bias is not None:
            q_bias, k_bias, v_bias = torch.chunk(layer.bias.detach(), 3, dim=0)
            new_layer.bias.copy_(
                torch.cat(
                    [
                        resize_vector(q_bias, target_embed_dim),
                        resize_vector(k_bias, target_embed_dim),
                        resize_vector(v_bias, target_embed_dim),
                    ],
                    dim=0,
                )
            )
    return new_layer


def resize_beit_attention_bias_table(
    values: torch.Tensor,
    new_num_heads: int,
) -> torch.Tensor:
    return resize_matrix(values, values.shape[0], new_num_heads)


def resize_conv2d_out_channels(layer: nn.Conv2d, new_out_channels: int) -> nn.Conv2d:
    if new_out_channels <= 0:
        raise ValueError("new_out_channels must be positive")
    new_layer = nn.Conv2d(
        layer.in_channels,
        new_out_channels,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
        bias=layer.bias is not None,
        padding_mode=layer.padding_mode,
        device=layer.weight.device,
        dtype=layer.weight.dtype,
    )
    with torch.no_grad():
        new_layer.weight.zero_()
        shared_out_channels = min(layer.out_channels, new_out_channels)
        shared_height = min(layer.weight.shape[-2], new_layer.weight.shape[-2])
        shared_width = min(layer.weight.shape[-1], new_layer.weight.shape[-1])
        source_h_start = (layer.weight.shape[-2] - shared_height) // 2
        source_w_start = (layer.weight.shape[-1] - shared_width) // 2
        target_h_start = (new_layer.weight.shape[-2] - shared_height) // 2
        target_w_start = (new_layer.weight.shape[-1] - shared_width) // 2
        new_layer.weight[
            :shared_out_channels,
            :,
            target_h_start : target_h_start + shared_height,
            target_w_start : target_w_start + shared_width,
        ] = layer.weight[
            :shared_out_channels,
            :,
            source_h_start : source_h_start + shared_height,
            source_w_start : source_w_start + shared_width,
        ]
        if layer.bias is not None and new_layer.bias is not None:
            new_layer.bias[:shared_out_channels] = layer.bias[:shared_out_channels]
    return new_layer


def resize_conv2d_kernel(layer: nn.Conv2d, kernel_size: int) -> nn.Conv2d:
    new_layer = nn.Conv2d(
        layer.in_channels,
        layer.out_channels,
        kernel_size=kernel_size,
        stride=layer.stride,
        padding=kernel_size // 2,
        dilation=layer.dilation,
        groups=layer.groups,
        bias=layer.bias is not None,
        padding_mode=layer.padding_mode,
        device=layer.weight.device,
        dtype=layer.weight.dtype,
    )
    with torch.no_grad():
        new_layer.weight.zero_()
        shared_height = min(layer.weight.shape[-2], new_layer.weight.shape[-2])
        shared_width = min(layer.weight.shape[-1], new_layer.weight.shape[-1])
        source_h_start = (layer.weight.shape[-2] - shared_height) // 2
        source_w_start = (layer.weight.shape[-1] - shared_width) // 2
        target_h_start = (new_layer.weight.shape[-2] - shared_height) // 2
        target_w_start = (new_layer.weight.shape[-1] - shared_width) // 2
        new_layer.weight[
            :,
            :,
            target_h_start : target_h_start + shared_height,
            target_w_start : target_w_start + shared_width,
        ] = layer.weight[
            :,
            :,
            source_h_start : source_h_start + shared_height,
            source_w_start : source_w_start + shared_width,
        ]
        if layer.bias is not None and new_layer.bias is not None:
            new_layer.bias.copy_(layer.bias)
    return new_layer


def resize_layer_norm(layer: nn.LayerNorm, normalized_shape: int) -> nn.LayerNorm:
    if normalized_shape <= 0:
        raise ValueError("normalized_shape must be positive")
    new_layer = nn.LayerNorm(
        normalized_shape,
        eps=layer.eps,
        elementwise_affine=layer.elementwise_affine,
        device=layer.weight.device if layer.elementwise_affine else None,
        dtype=layer.weight.dtype if layer.elementwise_affine else None,
    )
    if layer.elementwise_affine:
        with torch.no_grad():
            new_layer.weight.copy_(
                resize_vector(layer.weight.detach(), normalized_shape)
            )
            if layer.bias is not None and new_layer.bias is not None:
                new_layer.bias.copy_(
                    resize_vector(layer.bias.detach(), normalized_shape)
                )
    return new_layer


def resize_sequential_blocks(
    blocks: list[nn.Module],
    desired_length: int,
) -> nn.Sequential:
    if not blocks:
        raise ValueError("expected at least one block to resize")
    if desired_length <= 0:
        raise ValueError("desired_length must be positive")
    if desired_length == len(blocks):
        return nn.Sequential(*blocks)
    if desired_length < len(blocks):
        return nn.Sequential(*blocks[:desired_length])

    expanded_blocks = list(blocks)
    while len(expanded_blocks) < desired_length:
        expanded_blocks.append(copy.deepcopy(blocks[-1]))
    return nn.Sequential(*expanded_blocks)


def swap_activations(model: nn.Module, target_name: str, new_name: str) -> nn.Module:
    activation_class = activation_name_to_types(normalize_activation_name(target_name))
    replacement = build_activation(new_name)
    replace_matching_activations(model, activation_class, replacement)
    return model


def add_intermediate_fc_layer(
    model: nn.Module,
    hidden_size: int,
    dropout_rate: float,
    activation_name: str,
    target_output_classes: int,
) -> nn.Module:
    classifier_name, classifier = find_last_linear_layer(model)
    new_classifier = nn.Sequential(
        nn.Linear(classifier.in_features, hidden_size),
        build_activation(activation_name),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_size, target_output_classes),
    )
    replace_module(model, classifier_name, new_classifier)
    return model


def replace_fc_structure(
    model: nn.Module,
    hidden_layers: list[int],
    dropout_rates: list[float],
    activations: list[str],
    target_output_classes: int,
) -> nn.Module:
    classifier_name, classifier = find_last_linear_layer(model)
    layers: list[nn.Module] = []
    input_features = classifier.in_features

    for index, hidden_size in enumerate(hidden_layers):
        layers.append(nn.Linear(input_features, hidden_size))
        layers.append(build_activation(get_list_item(activations, index, "relu")))

        dropout_rate = get_list_item(dropout_rates, index, None)
        if dropout_rate is not None and dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        input_features = hidden_size

    layers.append(nn.Linear(input_features, target_output_classes))
    replace_module(model, classifier_name, nn.Sequential(*layers))
    return model


def replace_conv_kernel(
    model: nn.Module, layer_name: str, new_kernel: list[int], padding: int
) -> nn.Module:
    if len(new_kernel) != 2:
        raise ValueError("new_kernel must contain two integers")

    target_layer = get_module(model, layer_name)
    if not isinstance(target_layer, nn.Conv2d):
        raise TypeError(f"layer '{layer_name}' is not an nn.Conv2d")

    new_conv = nn.Conv2d(
        in_channels=target_layer.in_channels,
        out_channels=target_layer.out_channels,
        kernel_size=tuple(new_kernel),
        stride=target_layer.stride,
        padding=padding,
        dilation=target_layer.dilation,
        groups=target_layer.groups,
        bias=target_layer.bias is not None,
        padding_mode=target_layer.padding_mode,
    )
    copy_conv_weights(target_layer, new_conv)
    replace_module(model, layer_name, new_conv)
    return model


def replace_with_depthwise_separable_conv(
    model: nn.Module,
    layer_name: str,
) -> nn.Module:
    target_layer = get_module(model, layer_name)
    if not isinstance(target_layer, nn.Conv2d):
        raise TypeError(f"layer '{layer_name}' is not an nn.Conv2d")

    depthwise = nn.Conv2d(
        in_channels=target_layer.in_channels,
        out_channels=target_layer.in_channels,
        kernel_size=target_layer.kernel_size,
        stride=target_layer.stride,
        padding=target_layer.padding,
        dilation=target_layer.dilation,
        groups=target_layer.in_channels,
        bias=False,
        padding_mode=target_layer.padding_mode,
    )
    pointwise = nn.Conv2d(
        in_channels=target_layer.in_channels,
        out_channels=target_layer.out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=target_layer.bias is not None,
        padding_mode=target_layer.padding_mode,
    )
    replace_module(model, layer_name, nn.Sequential(depthwise, pointwise))
    return model


def replace_with_grouped_conv(
    model: nn.Module,
    layer_name: str,
    groups: int,
) -> nn.Module:
    target_layer = get_module(model, layer_name)
    if not isinstance(target_layer, nn.Conv2d):
        raise TypeError(f"layer '{layer_name}' is not an nn.Conv2d")
    if groups <= 0:
        raise ValueError("groups must be positive")
    if target_layer.in_channels % groups != 0:
        raise ValueError(
            f"layer '{layer_name}' input channels {target_layer.in_channels} "
            f"must be divisible by groups={groups}"
        )
    if target_layer.out_channels % groups != 0:
        raise ValueError(
            f"layer '{layer_name}' output channels {target_layer.out_channels} "
            f"must be divisible by groups={groups}"
        )

    new_conv = nn.Conv2d(
        in_channels=target_layer.in_channels,
        out_channels=target_layer.out_channels,
        kernel_size=target_layer.kernel_size,
        stride=target_layer.stride,
        padding=target_layer.padding,
        dilation=target_layer.dilation,
        groups=groups,
        bias=target_layer.bias is not None,
        padding_mode=target_layer.padding_mode,
    )
    copy_grouped_conv_weights(target_layer, new_conv)
    replace_module(model, layer_name, new_conv)
    return model


def add_normalization_layer(
    model: nn.Module,
    after_layer_name: str,
    norm_type: str,
    num_groups: int,
) -> nn.Module:
    parent_module_name, target_index = resolve_sequential_insertion_point(
        model, after_layer_name
    )
    parent_module = get_module(model, parent_module_name)
    assert isinstance(parent_module, nn.Sequential)

    num_channels = infer_output_channels_for_insertion(parent_module, target_index)
    new_layer = build_normalization_layer(
        norm_type=norm_type,
        num_channels=num_channels,
        num_groups=num_groups,
    )
    replace_module(
        model,
        parent_module_name,
        insert_layer_into_sequential(parent_module, target_index, new_layer),
    )
    return model


def add_se_block(
    model: nn.Module,
    after_layer_name: str,
    reduction_ratio: int,
) -> nn.Module:
    parent_module_name, target_index = resolve_sequential_insertion_point(
        model, after_layer_name
    )
    parent_module = get_module(model, parent_module_name)
    assert isinstance(parent_module, nn.Sequential)

    num_channels = infer_output_channels_for_insertion(parent_module, target_index)
    replace_module(
        model,
        parent_module_name,
        insert_layer_into_sequential(
            parent_module,
            target_index,
            SqueezeExcitationBlock(
                channels=num_channels,
                reduction_ratio=reduction_ratio,
            ),
        ),
    )
    return model


def modify_dropout_rate(
    model: nn.Module,
    layer_name: str,
    new_probability: float,
) -> nn.Module:
    if not 0.0 <= new_probability <= 1.0:
        raise ValueError("new_probability must be in [0, 1]")

    target_layer = get_module(model, layer_name)
    if not isinstance(
        target_layer,
        (
            nn.AlphaDropout,
            nn.Dropout,
            nn.Dropout1d,
            nn.Dropout2d,
            nn.Dropout3d,
            nn.FeatureAlphaDropout,
        ),
    ):
        raise TypeError(f"layer '{layer_name}' is not a dropout module")
    target_layer.p = new_probability
    return model


def prune_channels_by_ratio(
    model: nn.Module,
    layer_name: str,
    ratio: float,
    example_inputs: torch.Tensor,
) -> nn.Module:
    if not 0.0 < ratio < 1.0:
        raise ValueError("ChannelPruning ratio must be in (0, 1)")

    target_module = get_module(model, layer_name)
    _, total_channels = get_pruning_target(target_module)
    prune_count = int(total_channels * ratio)
    if prune_count <= 0:
        raise ValueError(
            f"ChannelPruning ratio {ratio} prunes zero channels for '{layer_name}'"
        )
    if prune_count >= total_channels:
        raise ValueError(
            "ChannelPruning ratio "
            f"{ratio} would remove every channel from '{layer_name}'"
        )

    prune_indices = select_low_importance_channel_indices(target_module, prune_count)
    return prune_channels_by_index(
        model,
        layer_name=layer_name,
        prune_indices=prune_indices,
        example_inputs=example_inputs,
    )


def prune_channels_by_index(
    model: nn.Module,
    layer_name: str,
    prune_indices: list[int],
    example_inputs: torch.Tensor,
) -> nn.Module:
    target_module = get_module(model, layer_name)
    pruning_fn, total_channels = get_pruning_target(target_module)
    normalized_indices = normalize_prune_indices(
        prune_indices=prune_indices,
        total_channels=total_channels,
        mutation_type="ChannelPruningByIndex",
    )
    dependency_graph = tp.DependencyGraph().build_dependency(
        model,
        example_inputs=example_inputs,
    )
    pruning_group = dependency_graph.get_pruning_group(
        target_module,
        pruning_fn,
        idxs=normalized_indices,
    )
    if not dependency_graph.check_pruning_group(pruning_group):
        raise ValueError(f"invalid pruning group for layer '{layer_name}'")
    pruning_group.prune()
    return model


def apply_global_channel_pruning(
    model: nn.Module,
    pruning_ratio: float,
    pruning_method: str,
    ignored_layer_prefixes: list[str],
    layers_to_target: list[str],
    example_inputs: torch.Tensor,
) -> nn.Module:
    if not 0.0 < pruning_ratio < 1.0:
        raise ValueError("GlobalChannelPruning pruning_ratio must be in (0, 1)")

    importance, requires_gradients = build_global_importance(pruning_method)
    ignored_layers = collect_global_ignored_layers(
        model=model,
        ignored_layer_prefixes=ignored_layer_prefixes,
        layers_to_target=layers_to_target,
    )

    was_training = model.training
    model.eval()
    if requires_gradients:
        model.zero_grad(set_to_none=True)
        logits = extract_pruning_logits(model(example_inputs))
        if logits.shape[-1] == 1:
            loss = logits.sum()
        else:
            labels = torch.zeros(logits.shape[0], dtype=torch.long)
            loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        global_pruning=True,
        pruning_ratio=pruning_ratio,
        iterative_steps=1,
        ignored_layers=ignored_layers,
    )
    pruner.step()
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return model


def build_image_example_inputs(example_input_shape: list[int]) -> torch.Tensor:
    if len(example_input_shape) != 4:
        raise ValueError("image mutations require a 4D example_input_shape")
    generator = torch.Generator().manual_seed(0)
    return torch.randn(tuple(example_input_shape), generator=generator)


def get_pruning_target(module: nn.Module) -> tuple[Callable[..., Any], int]:
    if isinstance(module, nn.Conv2d):
        return tp.prune_conv_out_channels, module.out_channels
    if isinstance(module, nn.Linear):
        return tp.prune_linear_out_channels, module.out_features
    if isinstance(module, nn.BatchNorm2d):
        return tp.prune_batchnorm_out_channels, module.num_features
    raise TypeError(
        "channel pruning only supports nn.Conv2d, nn.Linear, and nn.BatchNorm2d"
    )


def normalize_prune_indices(
    prune_indices: list[int], total_channels: int, mutation_type: str
) -> list[int]:
    if not prune_indices:
        raise ValueError(f"{mutation_type} requires at least one prune index")

    normalized_indices = sorted(prune_indices)
    if len(set(normalized_indices)) != len(normalized_indices):
        raise ValueError(f"{mutation_type} prune indices must be unique")
    if normalized_indices[0] < 0:
        raise ValueError(f"{mutation_type} prune indices must be non-negative")
    if normalized_indices[-1] >= total_channels:
        raise ValueError(
            f"{mutation_type} prune index {normalized_indices[-1]} is out of range "
            f"for {total_channels} channels"
        )
    if len(normalized_indices) >= total_channels:
        raise ValueError(f"{mutation_type} cannot prune every channel")
    return normalized_indices


def select_low_importance_channel_indices(
    module: nn.Module, prune_count: int
) -> list[int]:
    importance = compute_output_channel_importance(module)
    if prune_count >= importance.numel():
        raise ValueError("prune_count must be smaller than the total channel count")
    return sorted(torch.argsort(importance)[:prune_count].tolist())


def compute_output_channel_importance(module: nn.Module) -> torch.Tensor:
    if isinstance(module, nn.Conv2d):
        return module.weight.detach().abs().flatten(1).sum(dim=1)
    if isinstance(module, nn.Linear):
        return module.weight.detach().abs().flatten(1).sum(dim=1)
    if isinstance(module, nn.BatchNorm2d):
        if not module.affine or module.weight is None:
            raise ValueError("BatchNorm channel pruning requires affine parameters")
        return module.weight.detach().abs()
    raise TypeError(
        "channel pruning only supports nn.Conv2d, nn.Linear, and nn.BatchNorm2d"
    )


def build_global_importance(
    pruning_method: str,
) -> tuple[tp.importance.Importance, bool]:
    normalized_method = pruning_method.strip().lower()
    if normalized_method == "l1":
        return tp.importance.MagnitudeImportance(p=1), False
    if normalized_method in {"l2", "magnitude"}:
        return tp.importance.MagnitudeImportance(p=2), False
    if normalized_method == "random":
        return tp.importance.RandomImportance(), False
    if normalized_method in {"taylor", "gradient"}:
        # torch-pruning 1.6.0 does not expose a separate gradient-only scorer.
        return tp.importance.TaylorImportance(), True
    raise NotImplementedError(
        f"unsupported GlobalChannelPruning pruning_method: {pruning_method}"
    )


def collect_global_ignored_layers(
    model: nn.Module,
    ignored_layer_prefixes: list[str],
    layers_to_target: list[str],
) -> list[nn.Module]:
    classifier_name, classifier = find_last_linear_layer(model)
    ignored_layers: list[nn.Module] = [classifier]

    for name, module in model.named_modules():
        if not is_global_prunable_module(module):
            continue
        if name == classifier_name:
            continue
        if matches_module_prefix(name, ignored_layer_prefixes):
            ignored_layers.append(module)
            continue
        if layers_to_target and not matches_module_prefix(name, layers_to_target):
            ignored_layers.append(module)

    return deduplicate_modules(ignored_layers)


def is_global_prunable_module(module: nn.Module) -> bool:
    return isinstance(module, (nn.Conv2d, nn.Linear))


def matches_module_prefix(module_name: str, prefixes: list[str]) -> bool:
    return any(
        module_name == prefix or module_name.startswith(f"{prefix}.")
        for prefix in prefixes
    )


def deduplicate_modules(modules: list[nn.Module]) -> list[nn.Module]:
    unique_modules: list[nn.Module] = []
    seen_ids: set[int] = set()
    for module in modules:
        module_id = id(module)
        if module_id in seen_ids:
            continue
        seen_ids.add(module_id)
        unique_modules.append(module)
    return unique_modules


def extract_pruning_logits(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, tuple):
        return outputs[0]
    if isinstance(outputs, torch.Tensor):
        return outputs
    raise TypeError(f"unsupported model output type: {type(outputs)}")


def copy_conv_weights(source: nn.Conv2d, target: nn.Conv2d) -> None:
    with torch.no_grad():
        target.weight.zero_()

        shared_height = min(source.weight.shape[-2], target.weight.shape[-2])
        shared_width = min(source.weight.shape[-1], target.weight.shape[-1])

        source_h_start = (source.weight.shape[-2] - shared_height) // 2
        source_w_start = (source.weight.shape[-1] - shared_width) // 2
        target_h_start = (target.weight.shape[-2] - shared_height) // 2
        target_w_start = (target.weight.shape[-1] - shared_width) // 2

        target.weight[
            :,
            :,
            target_h_start : target_h_start + shared_height,
            target_w_start : target_w_start + shared_width,
        ] = source.weight[
            :,
            :,
            source_h_start : source_h_start + shared_height,
            source_w_start : source_w_start + shared_width,
        ]

        if source.bias is not None and target.bias is not None:
            target.bias.copy_(source.bias)


def copy_grouped_conv_weights(source: nn.Conv2d, target: nn.Conv2d) -> None:
    source_weight = source.weight.detach()
    out_channels_per_group = target.out_channels // target.groups
    in_channels_per_group = target.in_channels // target.groups

    with torch.no_grad():
        target.weight.zero_()
        for group_index in range(target.groups):
            out_start = group_index * out_channels_per_group
            out_end = out_start + out_channels_per_group
            in_start = group_index * in_channels_per_group
            in_end = in_start + in_channels_per_group
            target.weight[out_start:out_end] = source_weight[
                out_start:out_end, in_start:in_end
            ]

        if source.bias is not None and target.bias is not None:
            target.bias.copy_(source.bias)


def resolve_sequential_insertion_point(
    model: nn.Module,
    after_layer_name: str,
) -> tuple[str, int]:
    parent_module_name, _, leaf_name = after_layer_name.rpartition(".")
    if not parent_module_name or not leaf_name.isdigit():
        raise ValueError(
            "insertion mutations require an after_layer_name like 'features.9'"
        )

    parent_module = get_module(model, parent_module_name)
    if not isinstance(parent_module, nn.Sequential):
        raise TypeError(
            f"parent module '{parent_module_name}' must be an nn.Sequential"
        )

    target_index = int(leaf_name)
    if not 0 <= target_index < len(parent_module):
        raise IndexError(
            f"layer index {target_index} is out of range for '{parent_module_name}'"
        )
    return parent_module_name, target_index


def insert_layer_into_sequential(
    sequential: nn.Sequential,
    target_index: int,
    new_layer: nn.Module,
) -> nn.Sequential:
    layers = list(sequential.children())
    layers.insert(target_index + 1, new_layer)
    return nn.Sequential(*layers)


def infer_output_channels_for_insertion(
    sequential: nn.Sequential,
    target_index: int,
) -> int:
    search_index = target_index
    while search_index >= 0:
        channels = infer_module_output_channels(sequential[search_index])
        if channels is not None:
            return channels
        search_index -= 1
    raise ValueError(
        f"could not infer output channels near sequential index {target_index}"
    )


def infer_module_output_channels(module: nn.Module) -> int | None:
    if isinstance(module, nn.Conv2d):
        return module.out_channels
    if isinstance(module, nn.BatchNorm2d):
        return module.num_features
    if isinstance(module, nn.GroupNorm):
        return module.num_channels
    if isinstance(module, SqueezeExcitationBlock):
        return module.channels
    if isinstance(module, nn.Sequential):
        for child in reversed(list(module.children())):
            channels = infer_module_output_channels(child)
            if channels is not None:
                return channels
    return None


def build_normalization_layer(
    norm_type: str,
    num_channels: int,
    num_groups: int,
) -> nn.Module:
    normalized_type = norm_type.strip().lower()
    if normalized_type == "batch_norm":
        return nn.BatchNorm2d(num_channels)
    if normalized_type == "group_norm":
        if num_groups <= 0:
            raise ValueError("num_groups must be positive")
        if num_channels % num_groups != 0:
            raise ValueError(
                "num_channels "
                f"{num_channels} must be divisible by num_groups {num_groups}"
            )
        return nn.GroupNorm(num_groups, num_channels)
    raise NotImplementedError(f"unsupported normalization type: {norm_type}")


def activation_name_to_types(name: str) -> tuple[type[nn.Module], ...]:
    mapping: dict[str, tuple[type[nn.Module], ...]] = {
        "elu": (nn.ELU,),
        "gelu": (nn.GELU,),
        "hardswish": (nn.Hardswish,),
        "leaky_relu": (nn.LeakyReLU,),
        "mish": (nn.Mish,),
        "relu": (nn.ReLU,),
        "relu6": (nn.ReLU6,),
        "selu": (nn.SELU,),
        "sigmoid": (nn.Sigmoid,),
        "swish": (nn.SiLU,),
        "silu": (nn.SiLU,),
        "tanh": (nn.Tanh,),
    }
    if name not in mapping:
        raise NotImplementedError(f"unsupported activation name: {name}")
    return mapping[name]


def build_activation(name: str) -> nn.Module:
    normalized_name = normalize_activation_name(name)
    if normalized_name == "elu":
        return nn.ELU()
    if normalized_name == "gelu":
        return nn.GELU()
    if normalized_name == "hardswish":
        return nn.Hardswish()
    if normalized_name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    if normalized_name == "mish":
        return nn.Mish()
    if normalized_name == "prelu":
        return nn.PReLU()
    if normalized_name == "relu":
        return nn.ReLU()
    if normalized_name == "relu6":
        return nn.ReLU6()
    if normalized_name == "selu":
        return nn.SELU()
    if normalized_name == "sigmoid":
        return nn.Sigmoid()
    if normalized_name in {"silu", "swish"}:
        return nn.SiLU()
    if normalized_name == "tanh":
        return nn.Tanh()
    raise NotImplementedError(f"unsupported activation name: {name}")


def replace_matching_activations(
    module: nn.Module,
    activation_types: tuple[type[nn.Module], ...],
    replacement: nn.Module,
) -> None:
    for child_name, child in list(module.named_children()):
        if isinstance(child, activation_types):
            setattr(module, child_name, instantiate_activation_like(replacement))
            continue
        replace_matching_activations(child, activation_types, replacement)


def instantiate_activation_like(module: nn.Module) -> nn.Module:
    return module.__class__(*build_activation_args(module))


def build_activation_args(module: nn.Module) -> tuple:
    if isinstance(module, nn.LeakyReLU):
        return (module.negative_slope,)
    return ()


def find_last_linear_layer(model: nn.Module) -> tuple[str, nn.Linear]:
    linear_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    if not linear_layers:
        raise ValueError("model does not expose an nn.Linear classifier")
    layer_name, layer = linear_layers[-1]
    return layer_name, layer


def get_module(model: nn.Module, module_name: str) -> nn.Module:
    module = model
    for part in module_name.split("."):
        module = (
            module[int(part)]  # type: ignore[index]
            if part.isdigit()
            else getattr(module, part)
        )
    return module


def replace_module(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent_module_name, _, leaf_name = module_name.rpartition(".")
    parent_module = (
        model if not parent_module_name else get_module(model, parent_module_name)
    )
    if leaf_name.isdigit():
        parent_module[int(leaf_name)] = new_module  # type: ignore[index]
    else:
        setattr(parent_module, leaf_name, new_module)


def get_list_item(values: list[object], index: int, default: object) -> object:
    value_list = list(values)
    if index < len(value_list):
        return value_list[index]
    return default


def normalize_activation_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")
