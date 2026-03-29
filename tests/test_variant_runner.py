from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch

from gnn_archs.config import ArchConfig
from gnn_archs.mutations import SqueezeExcitationBlock
from gnn_archs.result import ResultDocument, write_result_document
from gnn_archs.util.variant_expander import expand_arch_config
from gnn_archs.variant_runner import (
    RunContext,
    build_variant_model,
    count_parameters,
    prepare_output_layout,
    run_variant,
)


def build_image_variant(
    mutations: list[dict[str, object]],
    *,
    base_model_name: str = "resnet18",
    variant_name: str = "image_mutation_smoke",
    example_input_shape: list[int] | None = None,
) -> object:
    resolved_input_shape = (
        example_input_shape if example_input_shape is not None else [1, 3, 64, 64]
    )
    config = ArchConfig.model_validate(
        {
            "base_model_groups": [
                {
                    "base_model": {"name": base_model_name, "pretrained": False},
                    "single_variant_define": [
                        {
                            "name": variant_name,
                            "variant_config": {
                                "target_input_channels": 3,
                                "target_output_classes": 4,
                                "example_input_shape": resolved_input_shape,
                                "run_training": False,
                                "run_inference": False,
                                "export_onnx": False,
                            },
                            "mutations": mutations,
                        }
                    ],
                }
            ]
        }
    )
    return expand_arch_config(config)[0]


def build_vgg_variant(mutations: list[dict[str, object]]) -> object:
    return build_image_variant(
        mutations,
        base_model_name="vgg11",
        variant_name="vgg11_image_mutation_smoke",
    )


def build_vit_variant(
    mutations: list[dict[str, object]],
    *,
    variant_name: str = "vit_image_mutation_smoke",
) -> object:
    return build_image_variant(
        mutations,
        base_model_name="vit_tiny_patch16_224",
        variant_name=variant_name,
        example_input_shape=[1, 3, 224, 224],
    )


def build_beit_variant(
    mutations: list[dict[str, object]],
    *,
    variant_name: str = "beit_image_mutation_smoke",
) -> object:
    return build_image_variant(
        mutations,
        base_model_name="beit_base_patch16_224",
        variant_name=variant_name,
        example_input_shape=[1, 3, 224, 224],
    )


def build_convnext_variant(
    mutations: list[dict[str, object]],
    *,
    variant_name: str = "convnext_image_mutation_smoke",
) -> object:
    return build_image_variant(
        mutations,
        base_model_name="convnext_tiny",
        variant_name=variant_name,
        example_input_shape=[1, 3, 224, 224],
    )


def test_image_variant_runner_executes_training_inference_and_onnx(
    tmp_path: Path,
) -> None:
    config = ArchConfig.model_validate(
        {
            "base_model_groups": [
                {
                    "base_model": {"name": "resnet18", "pretrained": False},
                    "single_variant_define": [
                        {
                            "name": "resnet18_runtime_smoke",
                            "variant_config": {
                                "target_input_channels": 3,
                                "target_output_classes": 4,
                                "example_input_shape": [1, 3, 32, 32],
                                "run_training": True,
                                "run_inference": True,
                                "export_onnx": True,
                                "training_batch_sizes": [2],
                                "training_epochs": 1,
                                "use_fake_imagenet": True,
                                "fake_dataset_size": 4,
                            },
                            "mutations": [
                                {
                                    "type": "ActivationSwap",
                                    "params": {
                                        "target_func_name": "relu",
                                        "new_func_name": "gelu",
                                    },
                                },
                                {
                                    "type": "AddIntermediateFCLayer",
                                    "params": {
                                        "hidden_size": 16,
                                        "dropout_rate": 0.1,
                                        "activation": "relu",
                                    },
                                },
                            ],
                        }
                    ],
                }
            ]
        }
    )
    variant = expand_arch_config(config)[0]
    output_layout = prepare_output_layout(tmp_path / "output")
    context = RunContext(
        config_path=tmp_path / "image.yaml",
        output_layout=output_layout,
        device=torch.device("cpu"),
        gpu_node="cpu-test",
        gpu_ids=[0],
        logger=logging.getLogger("test_image_variant_runner"),
    )

    result = run_variant(variant, context)

    assert result.training is not None
    assert result.inference is not None
    assert result.onnx_export is not None
    assert Path(result.onnx_export.path).exists()
    assert result.training.metrics["total_steps"] == 2
    assert result.metadata["model_kind"] == "image"


def test_text_variant_runner_executes_text_pipeline(tmp_path: Path) -> None:
    config = ArchConfig.model_validate(
        {
            "base_model_groups": [
                {
                    "base_model": {"name": "bert-base-uncased", "pretrained": False},
                    "single_variant_define": [
                        {
                            "name": "bert_runtime_smoke",
                            "variant_config": {
                                "target_input_channels": 1,
                                "target_output_classes": 3,
                                "example_input_shape": [2, 8],
                                "run_training": True,
                                "run_inference": True,
                                "export_onnx": False,
                                "training_batch_sizes": [2],
                                "training_epochs": 1,
                                "use_fake_text_dataset": True,
                                "max_sequence_length": 8,
                                "fake_dataset_size": 4,
                            },
                            "mutations": [
                                {
                                    "type": "TransformerLayerReduction",
                                    "params": {"target_layers": 2},
                                },
                                {
                                    "type": "HiddenSizeModification",
                                    "params": {"hidden_size": 64},
                                },
                                {
                                    "type": "AttentionHeadsModification",
                                    "params": {"num_heads": 4},
                                },
                                {
                                    "type": "IntermediateSizeModification",
                                    "params": {"intermediate_size": 128},
                                },
                                {
                                    "type": "DropoutModification",
                                    "params": {
                                        "hidden_dropout_prob": 0.05,
                                        "attention_probs_dropout_prob": 0.05,
                                    },
                                },
                            ],
                        }
                    ],
                }
            ]
        }
    )
    variant = expand_arch_config(config)[0]
    output_layout = prepare_output_layout(tmp_path / "output")
    context = RunContext(
        config_path=tmp_path / "text.yaml",
        output_layout=output_layout,
        device=torch.device("cpu"),
        gpu_node="cpu-test",
        gpu_ids=[0],
        logger=logging.getLogger("test_text_variant_runner"),
    )

    result = run_variant(variant, context)

    assert result.training is not None
    assert result.inference is not None
    assert result.metadata["model_kind"] == "text"
    assert result.metadata["validation_num_outputs"] == 3


def test_result_document_serialization_writes_clean_json(tmp_path: Path) -> None:
    config = ArchConfig.model_validate(
        {
            "base_model_groups": [
                {
                    "base_model": {"name": "resnet18", "pretrained": False},
                    "single_variant_define": [
                        {
                            "name": "resnet18_serialization_smoke",
                            "variant_config": {
                                "target_input_channels": 3,
                                "target_output_classes": 2,
                                "example_input_shape": [1, 3, 32, 32],
                                "run_training": False,
                                "run_inference": False,
                                "export_onnx": False,
                            },
                            "mutations": [],
                        }
                    ],
                }
            ]
        }
    )
    variant = expand_arch_config(config)[0]
    output_layout = prepare_output_layout(tmp_path / "output")
    context = RunContext(
        config_path=tmp_path / "serialization.yaml",
        output_layout=output_layout,
        device=torch.device("cpu"),
        gpu_node="cpu-test",
        gpu_ids=[0],
        logger=logging.getLogger("test_result_document"),
    )

    result = run_variant(variant, context)
    document = ResultDocument(
        config_path=str(context.config_path),
        gpu_node=context.gpu_node,
        gpu_ids=context.gpu_ids,
        variants=[result],
        summary={"variant_count": 1},
    )
    output_path = tmp_path / "result.json"
    write_result_document(output_path, document)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0.0"
    assert payload["variants"][0]["source"] == "single_variant_define"


def test_build_variant_model_supports_channel_pruning() -> None:
    baseline_variant = build_image_variant([])
    pruned_variant = build_image_variant(
        [
            {
                "type": "ChannelPruning",
                "params": {
                    "layer_name": "layer1.0.conv1",
                    "ratio": 0.25,
                },
            }
        ]
    )

    baseline_model = build_variant_model(baseline_variant)
    pruned_model = build_variant_model(pruned_variant)

    baseline_layer = baseline_model.get_submodule("layer1.0.conv1")
    pruned_layer = pruned_model.get_submodule("layer1.0.conv1")

    assert isinstance(baseline_layer, torch.nn.Conv2d)
    assert isinstance(pruned_layer, torch.nn.Conv2d)
    assert baseline_layer.out_channels == 64
    assert pruned_layer.out_channels == 48

    with torch.no_grad():
        logits = pruned_model(torch.randn(1, 3, 64, 64))
    assert logits.shape == (1, 4)


def test_build_variant_model_supports_channel_pruning_by_index() -> None:
    baseline_variant = build_image_variant([])
    pruned_variant = build_image_variant(
        [
            {
                "type": "ChannelPruningByIndex",
                "params": {
                    "layer_name": "layer1.0.conv1",
                    "prune_indices": [0, 1, 2, 3],
                },
            }
        ]
    )

    baseline_model = build_variant_model(baseline_variant)
    pruned_model = build_variant_model(pruned_variant)

    baseline_layer = baseline_model.get_submodule("layer1.0.conv1")
    pruned_layer = pruned_model.get_submodule("layer1.0.conv1")

    assert isinstance(baseline_layer, torch.nn.Conv2d)
    assert isinstance(pruned_layer, torch.nn.Conv2d)
    assert pruned_layer.out_channels == baseline_layer.out_channels - 4

    with torch.no_grad():
        logits = pruned_model(torch.randn(1, 3, 64, 64))
    assert logits.shape == (1, 4)


def test_build_variant_model_supports_global_channel_pruning() -> None:
    baseline_variant = build_image_variant([])
    pruned_variant = build_image_variant(
        [
            {
                "type": "GlobalChannelPruning",
                "params": {
                    "pruning_ratio": 0.1,
                    "pruning_method": "gradient",
                    "layers_to_target": ["layer2"],
                },
            }
        ]
    )

    baseline_model = build_variant_model(baseline_variant)
    pruned_model = build_variant_model(pruned_variant)

    assert count_parameters(pruned_model) < count_parameters(baseline_model)

    with torch.no_grad():
        logits = pruned_model(torch.randn(1, 3, 64, 64))
    assert logits.shape == (1, 4)


def test_build_variant_model_supports_depthwise_separable_conv() -> None:
    baseline_variant = build_vgg_variant([])
    optimized_variant = build_vgg_variant(
        [
            {
                "type": "ConvToDepthwiseSeparable",
                "params": {"layer_name": "features.8"},
            }
        ]
    )

    baseline_model = build_variant_model(baseline_variant)
    optimized_model = build_variant_model(optimized_variant)

    baseline_layer = baseline_model.get_submodule("features.8")
    optimized_layer = optimized_model.get_submodule("features.8")

    assert isinstance(baseline_layer, torch.nn.Conv2d)
    assert isinstance(optimized_layer, torch.nn.Sequential)

    depthwise = optimized_layer[0]
    pointwise = optimized_layer[1]
    assert isinstance(depthwise, torch.nn.Conv2d)
    assert isinstance(pointwise, torch.nn.Conv2d)
    assert depthwise.groups == depthwise.in_channels == depthwise.out_channels
    assert pointwise.kernel_size == (1, 1)
    assert count_parameters(optimized_model) < count_parameters(baseline_model)

    with torch.no_grad():
        logits = optimized_model(torch.randn(1, 3, 64, 64))
    assert logits.shape == (1, 4)


def test_build_variant_model_supports_grouped_conv() -> None:
    grouped_variant = build_vgg_variant(
        [
            {
                "type": "ConvToGroupedConv",
                "params": {
                    "layer_name": "features.8",
                    "groups": 4,
                },
            }
        ]
    )

    grouped_model = build_variant_model(grouped_variant)
    grouped_layer = grouped_model.get_submodule("features.8")

    assert isinstance(grouped_layer, torch.nn.Conv2d)
    assert grouped_layer.groups == 4

    with torch.no_grad():
        logits = grouped_model(torch.randn(1, 3, 64, 64))
    assert logits.shape == (1, 4)


def test_build_variant_model_supports_add_normalization_layer() -> None:
    normalized_variant = build_vgg_variant(
        [
            {
                "type": "AddNormalizationLayer",
                "params": {
                    "after_layer_name": "features.9",
                    "norm_type": "batch_norm",
                },
            }
        ]
    )

    normalized_model = build_variant_model(normalized_variant)

    assert isinstance(normalized_model.features[10], torch.nn.BatchNorm2d)
    assert normalized_model.features[10].num_features == 256

    with torch.no_grad():
        logits = normalized_model(torch.randn(1, 3, 64, 64))
    assert logits.shape == (1, 4)


def test_build_variant_model_supports_add_se_block() -> None:
    se_variant = build_vgg_variant(
        [
            {
                "type": "AddSEBlock",
                "params": {
                    "after_layer_name": "features.5",
                    "reduction_ratio": 16,
                },
            }
        ]
    )

    se_model = build_variant_model(se_variant)

    assert isinstance(se_model.features[6], SqueezeExcitationBlock)
    assert se_model.features[6].channels == 128

    with torch.no_grad():
        logits = se_model(torch.randn(1, 3, 64, 64))
    assert logits.shape == (1, 4)


def test_build_variant_model_supports_modify_dropout() -> None:
    baseline_variant = build_vgg_variant([])
    dropout_variant = build_vgg_variant(
        [
            {
                "type": "ModifyDropout",
                "params": {
                    "layer_name": "head.drop",
                    "new_p": 0.2,
                },
            }
        ]
    )

    baseline_model = build_variant_model(baseline_variant)
    dropout_model = build_variant_model(dropout_variant)

    assert isinstance(baseline_model.head.drop, torch.nn.Dropout)
    assert isinstance(dropout_model.head.drop, torch.nn.Dropout)
    assert baseline_model.head.drop.p != dropout_model.head.drop.p
    assert dropout_model.head.drop.p == 0.2

    with torch.no_grad():
        logits = dropout_model(torch.randn(1, 3, 64, 64))
    assert logits.shape == (1, 4)


def test_build_variant_model_supports_vit_block_reduction() -> None:
    baseline_variant = build_vit_variant([])
    reduced_variant = build_vit_variant(
        [
            {
                "type": "ViTBlockReduction",
                "params": {"num_blocks_to_keep": 6},
            }
        ],
        variant_name="vit_block_reduction_smoke",
    )

    baseline_model = build_variant_model(baseline_variant)
    reduced_model = build_variant_model(reduced_variant)

    assert len(baseline_model.blocks) == 12
    assert len(reduced_model.blocks) == 6
    assert count_parameters(reduced_model) < count_parameters(baseline_model)

    with torch.no_grad():
        logits = reduced_model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 4)


def test_build_variant_model_supports_vit_attention_heads_and_mlp_dimension() -> None:
    vit_variant = build_vit_variant(
        [
            {
                "type": "ViTModifyAttentionHeads",
                "params": {"new_num_heads": 6},
            },
            {
                "type": "ViTModifyMLPDimension",
                "params": {"new_mlp_dim": 384},
            },
        ],
        variant_name="vit_heads_mlp_smoke",
    )

    vit_model = build_variant_model(vit_variant)
    first_block = vit_model.blocks[0]

    assert first_block.attn.num_heads == 6
    assert first_block.mlp.fc1.out_features == 384
    assert first_block.mlp.fc2.in_features == 384

    with torch.no_grad():
        logits = vit_model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 4)


def test_build_variant_model_supports_vit_embed_dim_mutation() -> None:
    vit_variant = build_vit_variant(
        [
            {
                "type": "ViTModifyEmbedDim",
                "params": {
                    "new_embed_dim": 144,
                    "new_num_heads": 6,
                    "new_mlp_dim": 288,
                },
            }
        ],
        variant_name="vit_embed_dim_smoke",
    )

    vit_model = build_variant_model(vit_variant)
    first_block = vit_model.blocks[0]

    assert vit_model.embed_dim == 144
    assert vit_model.head.in_features == 144
    assert vit_model.patch_embed.proj.out_channels == 144
    assert first_block.attn.num_heads == 6
    assert first_block.mlp.fc1.out_features == 288

    with torch.no_grad():
        logits = vit_model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 4)


def test_build_variant_model_supports_vit_dropout_mutation() -> None:
    vit_variant = build_vit_variant(
        [
            {
                "type": "ViTModifyDropout",
                "params": {
                    "dropout_type": "attention",
                    "new_p": 0.2,
                },
            },
            {
                "type": "ViTModifyDropout",
                "params": {
                    "dropout_type": "mlp",
                    "new_p": 0.3,
                },
            },
        ],
        variant_name="vit_dropout_smoke",
    )

    vit_model = build_variant_model(vit_variant)
    first_block = vit_model.blocks[0]

    assert first_block.attn.attn_drop.p == 0.2
    assert first_block.attn.proj_drop.p == 0.2
    assert first_block.mlp.drop1.p == 0.3
    assert first_block.mlp.drop2.p == 0.3

    with torch.no_grad():
        logits = vit_model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 4)


def test_build_variant_model_rejects_invalid_vit_attention_heads() -> None:
    invalid_variant = build_vit_variant(
        [
            {
                "type": "ViTModifyAttentionHeads",
                "params": {"new_num_heads": 5},
            }
        ],
        variant_name="vit_invalid_heads",
    )

    with pytest.raises(ValueError, match="divisible"):
        build_variant_model(invalid_variant)


def test_build_variant_model_supports_beit_block_mlp_and_dropout_mutations() -> None:
    beit_variant = build_beit_variant(
        [
            {
                "type": "BEiTBlockReduction",
                "params": {"num_blocks_to_keep": 6},
            },
            {
                "type": "BEiTModifyMLPDimension",
                "params": {"new_mlp_dim": 1536},
            },
            {
                "type": "BEiTModifyDropout",
                "params": {
                    "dropout_type": "attention",
                    "new_p": 0.2,
                },
            },
            {
                "type": "BEiTModifyDropout",
                "params": {
                    "dropout_type": "mlp",
                    "new_p": 0.3,
                },
            },
        ],
        variant_name="beit_block_mlp_dropout_smoke",
    )

    beit_model = build_variant_model(beit_variant)
    first_block = beit_model.blocks[0]

    assert len(beit_model.blocks) == 6
    assert first_block.mlp.fc1.out_features == 1536
    assert first_block.attn.attn_drop.p == 0.2
    assert first_block.attn.proj_drop.p == 0.2
    assert first_block.mlp.drop1.p == 0.3
    assert first_block.mlp.drop2.p == 0.3

    with torch.no_grad():
        logits = beit_model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 4)


def test_build_variant_model_supports_beit_attention_head_mutation() -> None:
    beit_variant = build_beit_variant(
        [
            {
                "type": "BEiTModifyAttentionHeads",
                "params": {"new_num_heads": 10},
            }
        ],
        variant_name="beit_attention_heads_smoke",
    )

    beit_model = build_variant_model(beit_variant)
    first_block = beit_model.blocks[0]

    assert beit_model.embed_dim == 770
    assert beit_model.patch_embed.proj.out_channels == 770
    assert first_block.attn.num_heads == 10
    assert first_block.attn.relative_position_bias_table.shape[1] == 10
    assert beit_model.head.in_features == 770

    with torch.no_grad():
        logits = beit_model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 4)


def test_build_variant_model_supports_beit_attention_head_pruning() -> None:
    beit_variant = build_beit_variant(
        [
            {
                "type": "BEiTAttentionHeadPruning",
                "params": {
                    "head_pruning_ratio": 0.25,
                    "layer_pruning_ratio": 0.25,
                },
            }
        ],
        variant_name="beit_attention_head_pruning_smoke",
    )

    beit_model = build_variant_model(beit_variant)
    first_block = beit_model.blocks[0]

    assert len(beit_model.blocks) == 9
    assert beit_model.embed_dim == 765
    assert first_block.attn.num_heads == 9

    with torch.no_grad():
        logits = beit_model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 4)


def test_build_variant_model_supports_beit_layer_and_channel_pruning() -> None:
    baseline_variant = build_beit_variant([], variant_name="beit_baseline")
    layer_pruned_variant = build_beit_variant(
        [
            {
                "type": "BEiTLayerPruning",
                "params": {
                    "target_layers": 5,
                    "removal_strategy": "remove_middle",
                },
            }
        ],
        variant_name="beit_layer_pruning_smoke",
    )
    channel_pruned_variant = build_beit_variant(
        [
            {
                "type": "BEiTChannelPruning",
                "params": {
                    "layer_name": "patch_embed.proj",
                    "ratio": 0.1,
                },
            }
        ],
        variant_name="beit_channel_pruning_smoke",
    )
    global_pruned_variant = build_beit_variant(
        [
            {
                "type": "BEiTGlobalChannelPruning",
                "params": {
                    "pruning_ratio": 0.1,
                    "pruning_method": "l2",
                    "ignored_layers": ["head"],
                },
            }
        ],
        variant_name="beit_global_channel_pruning_smoke",
    )

    baseline_model = build_variant_model(baseline_variant)
    layer_pruned_model = build_variant_model(layer_pruned_variant)
    channel_pruned_model = build_variant_model(channel_pruned_variant)
    global_pruned_model = build_variant_model(global_pruned_variant)

    assert len(layer_pruned_model.blocks) == 5
    assert channel_pruned_model.patch_embed.proj.out_channels == 696
    assert count_parameters(global_pruned_model) < count_parameters(baseline_model)

    with torch.no_grad():
        layer_pruned_logits = layer_pruned_model(torch.randn(1, 3, 224, 224))
        channel_pruned_logits = channel_pruned_model(torch.randn(1, 3, 224, 224))
        global_pruned_logits = global_pruned_model(torch.randn(1, 3, 224, 224))
    assert layer_pruned_logits.shape == (1, 4)
    assert channel_pruned_logits.shape == (1, 4)
    assert global_pruned_logits.shape == (1, 4)


def test_build_variant_model_supports_convnext_stage_reduction() -> None:
    convnext_variant = build_convnext_variant(
        [
            {
                "type": "ConvNeXtStageReduction",
                "params": {
                    "target_blocks": [4, 2, 12, 2],
                },
            }
        ],
        variant_name="convnext_stage_reduction_smoke",
    )

    convnext_model = build_variant_model(convnext_variant)

    assert [len(stage.blocks) for stage in convnext_model.stages] == [4, 2, 12, 2]

    with torch.no_grad():
        logits = convnext_model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 4)


def test_build_variant_model_supports_convnext_mlp_expansion_ratio() -> None:
    convnext_variant = build_convnext_variant(
        [
            {
                "type": "ConvNeXtMLPExpansionRatio",
                "params": {"expansion_ratio": 1.5},
            }
        ],
        variant_name="convnext_mlp_ratio_smoke",
    )

    convnext_model = build_variant_model(convnext_variant)

    assert convnext_model.stages[0].blocks[0].mlp.fc1.out_features == 144
    assert convnext_model.stages[0].blocks[0].mlp.fc2.in_features == 144

    with torch.no_grad():
        logits = convnext_model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 4)


def test_build_variant_model_supports_convnext_kernel_size_modification() -> None:
    convnext_variant = build_convnext_variant(
        [
            {
                "type": "ConvNeXtKernelSizeModification",
                "params": {"kernel_size": 5},
            }
        ],
        variant_name="convnext_kernel_size_smoke",
    )

    convnext_model = build_variant_model(convnext_variant)
    first_block = convnext_model.stages[0].blocks[0]

    assert first_block.conv_dw.kernel_size == (5, 5)
    assert first_block.conv_dw.padding == (2, 2)

    with torch.no_grad():
        logits = convnext_model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 4)


def test_build_variant_model_supports_convnext_dropout_modification() -> None:
    baseline_variant = build_convnext_variant([])
    dropout_variant = build_convnext_variant(
        [
            {
                "type": "ConvNeXtModifyDropout",
                "params": {"new_p": 0.25},
            }
        ],
        variant_name="convnext_dropout_smoke",
    )

    baseline_model = build_variant_model(baseline_variant)
    dropout_model = build_variant_model(dropout_variant)

    assert baseline_model.stages[0].blocks[0].mlp.drop1.p == 0.0
    assert baseline_model.stages[0].blocks[0].mlp.drop2.p == 0.0
    assert baseline_model.head.drop.p == 0.0
    assert dropout_model.stages[0].blocks[0].mlp.drop1.p == 0.25
    assert dropout_model.stages[0].blocks[0].mlp.drop2.p == 0.25
    assert dropout_model.head.drop.p == 0.25

    with torch.no_grad():
        logits = dropout_model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 4)


def test_build_variant_model_rejects_invalid_convnext_dropout_probability() -> None:
    invalid_variant = build_convnext_variant(
        [
            {
                "type": "ConvNeXtModifyDropout",
                "params": {"new_p": 1.5},
            }
        ],
        variant_name="convnext_invalid_dropout_smoke",
    )

    with pytest.raises(
        ValueError, match=r"ConvNeXtModifyDropout new_probability must be in \[0, 1\]"
    ):
        build_variant_model(invalid_variant)
