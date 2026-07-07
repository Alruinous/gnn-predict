from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn
from transformers import (
    Gemma4ForCausalLM,
    Gemma4TextConfig,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    T5ForSequenceClassification,
)

import gnn_archs.causal_lm_builder as causal_lm_builder_module
import gnn_archs.variant_runner as variant_runner_module
from gnn_archs.config import ArchConfig, ResolvedVariantSpec
from gnn_archs.gpt2_builder import Gpt2ForGnnArchsSequenceClassification
from gnn_archs.mutations import SqueezeExcitationBlock
from gnn_archs.result import (
    InferenceResult,
    ResultDocument,
    TimeWindow,
    TrainingResult,
    write_result_document,
)
from gnn_archs.util.onnx_initializer import write_randomized_onnx_model
from gnn_archs.util.variant_expander import expand_arch_config
from gnn_model.data.onnx_graph import build_graph_data_from_onnx
from gnn_archs.variant_runner import (
    RunContext,
    build_example_batch,
    build_training_batch,
    build_variant_model,
    count_parameters,
    derive_config_output_name,
    export_causal_lm_decode_onnx,
    export_onnx_model,
    prepare_output_layout,
    run_decode,
    run_prefill,
    run_inference,
    run_variant,
    train_model,
)


def build_image_variant(
    mutations: list[dict[str, object]],
    *,
    base_model_name: str = "resnet18",
    variant_name: str = "image_mutation_smoke",
    example_input_shape: list[int] | None = None,
) -> ResolvedVariantSpec:
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


def build_text_variant(
    mutations: list[dict[str, object]],
    *,
    base_model_name: str = "bert-base-uncased",
    variant_name: str = "text_mutation_smoke",
    example_input_shape: list[int] | None = None,
) -> ResolvedVariantSpec:
    resolved_input_shape = (
        example_input_shape if example_input_shape is not None else [1, 8]
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
                                "target_input_channels": 1,
                                "target_output_classes": 4,
                                "example_input_shape": resolved_input_shape,
                                "run_training": False,
                                "run_inference": False,
                                "export_onnx": False,
                                "use_fake_text_dataset": True,
                            },
                            "mutations": mutations,
                        }
                    ],
                }
            ]
        }
    )
    return expand_arch_config(config)[0]


def build_gpt2_config_override(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "vocab_size": 32,
        "n_positions": 16,
        "n_embd": 32,
        "n_layer": 2,
        "n_head": 4,
        "n_inner": 64,
        "resid_pdrop": 0.0,
        "embd_pdrop": 0.0,
        "attn_pdrop": 0.0,
    }
    config.update(overrides)
    return config


def build_gpt2_variant(
    *,
    variant_name: str = "gpt2_runtime_smoke",
    variant_config_overrides: dict[str, object] | None = None,
    mutations: list[dict[str, object]] | None = None,
) -> ResolvedVariantSpec:
    variant_config = {
        "target_input_channels": 1,
        "target_output_classes": 3,
        "example_input_shape": [2, 8],
        "run_training": False,
        "run_inference": False,
        "export_onnx": False,
        "batch_size": 2,
        "training_measurement_min_seconds": 1e-9,
        "use_fake_text_dataset": True,
        "gpt2_config": build_gpt2_config_override(),
    }
    if variant_config_overrides is not None:
        variant_config.update(variant_config_overrides)

    config = ArchConfig.model_validate(
        {
            "base_model_groups": [
                {
                    "base_model": {"name": "gpt2", "pretrained": False},
                    "single_variant_define": [
                        {
                            "name": variant_name,
                            "variant_config": variant_config,
                            "mutations": mutations or [],
                        }
                    ],
                }
            ]
        }
    )
    return expand_arch_config(config)[0]


def build_t5_config_override(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "vocab_size": 32,
        "d_model": 32,
        "d_ff": 64,
        "num_layers": 1,
        "num_decoder_layers": 1,
        "num_heads": 4,
        "d_kv": 8,
        "relative_attention_num_buckets": 8,
        "relative_attention_max_distance": 16,
        "dropout_rate": 0.0,
        "classifier_dropout": 0.0,
    }
    config.update(overrides)
    return config


def build_t5_variant(
    *,
    variant_name: str = "t5_runtime_smoke",
    variant_config_overrides: dict[str, object] | None = None,
    mutations: list[dict[str, object]] | None = None,
) -> ResolvedVariantSpec:
    variant_config: dict[str, object] = {
        "target_input_channels": 1,
        "target_output_classes": 3,
        "example_input_shape": [2, 6],
        "run_training": False,
        "run_inference": False,
        "export_onnx": False,
        "batch_size": 2,
        "training_measurement_min_seconds": 1e-9,
        "use_fake_text_dataset": True,
        "t5_config": build_t5_config_override(),
    }
    if variant_config_overrides is not None:
        variant_config.update(variant_config_overrides)

    config = ArchConfig.model_validate(
        {
            "base_model_groups": [
                {
                    "base_model": {"name": "t5", "pretrained": False},
                    "single_variant_define": [
                        {
                            "name": variant_name,
                            "variant_config": variant_config,
                            "mutations": mutations or [],
                        }
                    ],
                }
            ]
        }
    )
    return expand_arch_config(config)[0]


def build_qwen_variant(
    *,
    base_model_name: str = "qwen3",
    variant_name: str = "qwen_runtime_smoke",
    variant_config_overrides: dict[str, object] | None = None,
) -> ResolvedVariantSpec:
    variant_config: dict[str, object] = {
        "example_input_shape": [2, 8],
        "run_training": False,
        "run_inference": False,
        "run_prefill": False,
        "export_onnx": False,
        "batch_size": 2,
        "training_measurement_min_seconds": 1e-9,
        "prefill_measurement_min_seconds": 1e-9,
        "decode_measurement_min_seconds": 1e-9,
        "use_fake_text_dataset": True,
        "qwen3_config": {
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 16,
        },
    }
    if variant_config_overrides is not None:
        variant_config.update(variant_config_overrides)

    config = ArchConfig.model_validate(
        {
            "base_model_groups": [
                {
                    "base_model": {"name": base_model_name, "pretrained": False},
                    "single_variant_define": [
                        {
                            "name": variant_name,
                            "variant_config": variant_config,
                            "mutations": [],
                        }
                    ],
                }
            ]
        }
    )
    return expand_arch_config(config)[0]


def build_gemma4_config_override(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 32,
        "sliding_window": 8,
        "vocab_size_per_layer_input": 32,
        "hidden_size_per_layer_input": 4,
    }
    config.update(overrides)
    return config


def build_gemma4_variant(
    *,
    base_model_name: str = "gemma4",
    variant_name: str = "gemma4_runtime_smoke",
    variant_config_overrides: dict[str, object] | None = None,
) -> ResolvedVariantSpec:
    variant_config: dict[str, object] = {
        "example_input_shape": [2, 8],
        "run_training": False,
        "run_inference": False,
        "run_prefill": False,
        "export_onnx": False,
        "batch_size": 2,
        "training_measurement_min_seconds": 1e-9,
        "prefill_measurement_min_seconds": 1e-9,
        "decode_measurement_min_seconds": 1e-9,
        "use_fake_text_dataset": True,
        "gemma4_config": build_gemma4_config_override(),
    }
    if variant_config_overrides is not None:
        variant_config.update(variant_config_overrides)

    config = ArchConfig.model_validate(
        {
            "base_model_groups": [
                {
                    "base_model": {"name": base_model_name, "pretrained": False},
                    "single_variant_define": [
                        {
                            "name": variant_name,
                            "variant_config": variant_config,
                            "mutations": [],
                        }
                    ],
                }
            ]
        }
    )
    return expand_arch_config(config)[0]


def build_llama_variant(
    *,
    base_model_name: str = "Llama-3.2-1B",
    variant_name: str = "llama_runtime_smoke",
    variant_config_overrides: dict[str, object] | None = None,
    mutations: list[dict[str, object]] | None = None,
) -> ResolvedVariantSpec:
    variant_config: dict[str, object] = {
        "example_input_shape": [2, 8],
        "run_training": False,
        "run_inference": False,
        "run_prefill": False,
        "export_onnx": False,
        "batch_size": 2,
        "training_measurement_min_seconds": 1e-9,
        "prefill_measurement_min_seconds": 1e-9,
        "use_fake_text_dataset": True,
    }
    if variant_config_overrides is not None:
        variant_config.update(variant_config_overrides)

    config = ArchConfig.model_validate(
        {
            "base_model_groups": [
                {
                    "base_model": {"name": base_model_name, "pretrained": True},
                    "single_variant_define": [
                        {
                            "name": variant_name,
                            "variant_config": variant_config,
                            "mutations": mutations or [],
                        }
                    ],
                }
            ]
        }
    )
    return expand_arch_config(config)[0]


def build_tiny_qwen_model() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=None,
        use_cache=True,
    )
    return Qwen3ForCausalLM(config)


def build_tiny_gemma4_model() -> Gemma4ForCausalLM:
    config = Gemma4TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
        vocab_size_per_layer_input=32,
        hidden_size_per_layer_input=4,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        use_cache=True,
    )
    return Gemma4ForCausalLM(config)


def build_tiny_llama_model() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=True,
    )
    return LlamaForCausalLM(config)


def build_recommender_common_config_override(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "sparse_features": [
            {"name": "user_id", "vocab_size": 32, "embed_dim": 4},
            {"name": "item_id", "vocab_size": 64, "embed_dim": 4},
            {"name": "device_type", "vocab_size": 8, "embed_dim": 4},
        ],
        "dense_features": [
            {"name": "user_age_score"},
            {"name": "item_price_score"},
        ],
        "mlp_dims": [16, 8],
        "activation": "relu",
        "dropout": 0.0,
    }
    config.update(overrides)
    return config


def build_recommender_variant(
    *,
    base_model_name: str = "deepfm",
    variant_name: str = "recommender_runtime_smoke",
    variant_config_overrides: dict[str, object] | None = None,
    model_config_overrides: dict[str, object] | None = None,
    mutations: list[dict[str, object]] | None = None,
) -> ResolvedVariantSpec:
    model_specific_config: dict[str, object]
    if base_model_name == "deepfm":
        model_config_field = "deepfm_config"
        model_specific_config = {"fm_feature_names": ["user_id", "item_id"]}
    elif base_model_name == "dcn":
        model_config_field = "dcn_config"
        model_specific_config = {"n_cross_layers": 2}
    elif base_model_name == "dcnv2":
        model_config_field = "dcnv2_config"
        model_specific_config = {
            "n_cross_layers": 2,
            "low_rank": 4,
            "num_experts": 2,
            "model_structure": "parallel",
            "use_low_rank_mixture": True,
        }
    elif base_model_name == "edcn":
        model_config_field = "edcn_config"
        model_specific_config = {
            "n_cross_layers": 2,
            "bridge_type": "hadamard_product",
            "use_regulation_module": True,
            "temperature": 1.0,
        }
    else:
        raise ValueError(f"unsupported recommender model: {base_model_name}")
    if model_config_overrides is not None:
        model_specific_config = {**model_specific_config, **model_config_overrides}
    common_config = build_recommender_common_config_override(**model_specific_config)
    if base_model_name == "edcn":
        common_config.pop("mlp_dims")

    variant_config: dict[str, object] = {
        "target_output_classes": 1,
        "example_input_shape": [2],
        "run_training": False,
        "run_inference": False,
        "export_onnx": False,
        "batch_size": 2,
        "training_measurement_min_seconds": 1e-9,
        "inference_measurement_min_seconds": 1e-9,
        "use_fake_recommender_dataset": True,
        model_config_field: common_config,
    }
    if variant_config_overrides is not None:
        variant_config.update(variant_config_overrides)

    config = ArchConfig.model_validate(
        {
            "base_model_groups": [
                {
                    "base_model": {"name": base_model_name, "pretrained": False},
                    "single_variant_define": [
                        {
                            "name": variant_name,
                            "variant_config": variant_config,
                            "mutations": mutations or [],
                        }
                    ],
                }
            ]
        }
    )
    return expand_arch_config(config)[0]


def require_t5_test_eos_token_id(model: T5ForSequenceClassification) -> int:
    eos_token_id = model.config.eos_token_id
    assert isinstance(eos_token_id, int)
    return eos_token_id


def build_vgg_variant(mutations: list[dict[str, object]]) -> ResolvedVariantSpec:
    return build_image_variant(
        mutations,
        base_model_name="vgg11",
        variant_name="vgg11_image_mutation_smoke",
    )


def build_vit_variant(
    mutations: list[dict[str, object]],
    *,
    variant_name: str = "vit_image_mutation_smoke",
) -> ResolvedVariantSpec:
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
) -> ResolvedVariantSpec:
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
) -> ResolvedVariantSpec:
    return build_image_variant(
        mutations,
        base_model_name="convnext_tiny",
        variant_name=variant_name,
        example_input_shape=[1, 3, 224, 224],
    )


@pytest.mark.parametrize(
    ("base_model_name", "mutation_name", "mutations"),
    [
        ("efficientnet_b0", "baseline", []),
        (
            "efficientnet_b0",
            "fc",
            [
                {
                    "type": "AddIntermediateFCLayer",
                    "params": {
                        "hidden_size": 512,
                        "dropout_rate": 0.1,
                        "activation": "relu",
                    },
                }
            ],
        ),
        (
            "efficientnet_b0",
            "kernel",
            [
                {
                    "type": "ConvKernelReplacement",
                    "params": {
                        "layer_name": "conv_stem",
                        "new_kernel": [5, 5],
                        "padding": 2,
                    },
                }
            ],
        ),
        (
            "efficientnet_b0",
            "activation",
            [
                {
                    "type": "ActivationFunctionSwap",
                    "params": {
                        "target_activation": "silu",
                        "replace_activation": "relu",
                    },
                }
            ],
        ),
        (
            "efficientnet_b0",
            "pruning",
            [
                {
                    "type": "ChannelPruning",
                    "params": {"layer_name": "conv_stem", "ratio": 0.125},
                }
            ],
        ),
        ("swin_tiny_patch4_window7_224", "baseline", []),
        (
            "swin_tiny_patch4_window7_224",
            "fc",
            [
                {
                    "type": "AddIntermediateFCLayer",
                    "params": {
                        "hidden_size": 512,
                        "dropout_rate": 0.1,
                        "activation": "gelu",
                    },
                }
            ],
        ),
        (
            "swin_tiny_patch4_window7_224",
            "kernel",
            [
                {
                    "type": "ConvKernelReplacement",
                    "params": {
                        "layer_name": "patch_embed.proj",
                        "new_kernel": [2, 2],
                        "padding": 0,
                    },
                }
            ],
        ),
        (
            "swin_tiny_patch4_window7_224",
            "activation",
            [
                {
                    "type": "ActivationFunctionSwap",
                    "params": {
                        "target_activation": "gelu",
                        "replace_activation": "relu",
                    },
                }
            ],
        ),
        (
            "swin_tiny_patch4_window7_224",
            "pruning",
            [
                {
                    "type": "ChannelPruning",
                    "params": {
                        "layer_name": "layers.0.blocks.0.mlp.fc1",
                        "ratio": 0.05,
                    },
                }
            ],
        ),
    ],
)
def test_build_variant_model_supports_new_timm_model_configs(
    base_model_name: str,
    mutation_name: str,
    mutations: list[dict[str, object]],
) -> None:
    variant = build_image_variant(
        mutations,
        base_model_name=base_model_name,
        variant_name=f"{base_model_name}_{mutation_name}_smoke",
        example_input_shape=[1, 3, 224, 224],
    )

    model = build_variant_model(variant)

    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 224, 224))
    assert tuple(logits.shape) == (1, 4)


def test_image_variant_runner_executes_training_inference_and_onnx(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "resnet18_variants.yaml"
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
                                "batch_size": 2,
                                "training_measurement_min_seconds": 1e-9,
                                "use_fake_imagenet": True,
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
    output_layout = prepare_output_layout(tmp_path / "output", config_path)
    context = RunContext(
        config_path=config_path,
        output_layout=output_layout,
        device=torch.device("cpu"),
        gpu_node="cpu-test",
        logger=logging.getLogger("test_image_variant_runner"),
    )

    result = run_variant(variant, context)

    assert result.training is not None
    assert result.inference is not None
    assert result.onnx_export is not None
    assert Path(result.onnx_export.path).exists()
    assert Path(result.onnx_export.path).parent == output_layout.onnx_models_dir
    exported_model = onnx.load(result.onnx_export.path)
    assert len(exported_model.graph.initializer) > 0
    assert [value.name for value in exported_model.graph.input] == ["inputs"]
    assert result.training.metrics["total_steps"] == 1
    assert result.metadata["model_kind"] == "image"
    assert result.onnx_export.graph_info["runtime_input_names"] == ["inputs"]
    assert result.onnx_export.graph_info["parameter_input_names"] == []


def test_train_model_records_elapsed_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variant = build_image_variant(
        [],
        variant_name="timed_training_variant",
        example_input_shape=[1, 3, 8, 8],
    ).model_copy(
        update={
            "variant_config": build_image_variant(
                [],
                variant_name="timed_training_config",
                example_input_shape=[1, 3, 8, 8],
            ).variant_config.model_copy(
                update={
                    "run_training": True,
                    "training_measurement_min_seconds": 5.0,
                    "batch_size": 2,
                    "use_fake_imagenet": True,
                }
            )
        }
    )
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 4))
    time_values = iter([100.0, 102.0, 106.0])
    monkeypatch.setattr(variant_runner_module.time, "time", lambda: next(time_values))

    result = train_model(variant, model, torch.device("cpu"))

    assert result.hyperparameters["measurement_min_seconds"] == 5.0
    assert result.metrics["total_steps"] == 2
    assert result.timings.started_at_ts == 100.0
    assert result.timings.ended_at_ts == 106.0


def test_image_variant_runner_can_export_architecture_only_onnx(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "resnet18_variants.yaml"
    config = ArchConfig.model_validate(
        {
            "base_model_groups": [
                {
                    "base_model": {"name": "resnet18", "pretrained": False},
                    "single_variant_define": [
                        {
                            "name": "resnet18_architecture_only",
                            "variant_config": {
                                "target_input_channels": 3,
                                "target_output_classes": 4,
                                "example_input_shape": [1, 3, 32, 32],
                                "run_training": False,
                                "run_inference": False,
                                "export_onnx": True,
                                "onnx_export_mode": "architecture_only",
                            },
                            "mutations": [],
                        }
                    ],
                }
            ]
        }
    )
    variant = expand_arch_config(config)[0]
    output_layout = prepare_output_layout(tmp_path / "output", config_path)
    context = RunContext(
        config_path=config_path,
        output_layout=output_layout,
        device=torch.device("cpu"),
        gpu_node="cpu-test",
        logger=logging.getLogger("test_image_variant_runner_architecture_only"),
    )

    result = run_variant(variant, context)

    assert result.onnx_export is not None
    exported_model = onnx.load(result.onnx_export.path)
    exported_input_names = [value.name for value in exported_model.graph.input]

    assert len(exported_model.graph.initializer) == 0
    assert exported_input_names[0] == "inputs"
    assert len(exported_input_names) > 1
    assert result.onnx_export.graph_info["runtime_input_names"] == ["inputs"]
    parameter_input_names = result.onnx_export.graph_info["parameter_input_names"]
    assert isinstance(parameter_input_names, list)
    assert parameter_input_names
    assert all(name in exported_input_names for name in parameter_input_names)


def test_randomized_architecture_only_onnx_runs_with_runtime_inputs_only(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "resnet18_variants.yaml"
    config = ArchConfig.model_validate(
        {
            "base_model_groups": [
                {
                    "base_model": {"name": "resnet18", "pretrained": False},
                    "single_variant_define": [
                        {
                            "name": "resnet18_randomized_architecture_only",
                            "variant_config": {
                                "target_input_channels": 3,
                                "target_output_classes": 4,
                                "example_input_shape": [1, 3, 32, 32],
                                "run_training": False,
                                "run_inference": False,
                                "export_onnx": True,
                                "onnx_export_mode": "architecture_only",
                            },
                            "mutations": [],
                        }
                    ],
                }
            ]
        }
    )
    variant = expand_arch_config(config)[0]
    output_layout = prepare_output_layout(tmp_path / "output", config_path)
    context = RunContext(
        config_path=config_path,
        output_layout=output_layout,
        device=torch.device("cpu"),
        gpu_node="cpu-test",
        logger=logging.getLogger("test_randomized_architecture_only_onnx"),
    )

    result = run_variant(variant, context)
    assert result.onnx_export is not None

    randomized_path = output_layout.onnx_models_dir / "resnet18_randomized.onnx"
    write_randomized_onnx_model(
        Path(result.onnx_export.path),
        randomized_path,
        seed=0,
    )

    randomized_model = onnx.load(randomized_path)
    assert len(randomized_model.graph.initializer) > 0
    assert [value.name for value in randomized_model.graph.input] == ["inputs"]

    session = ort.InferenceSession(
        str(randomized_path),
        providers=["CPUExecutionProvider"],
    )
    assert [value.name for value in session.get_inputs()] == ["inputs"]
    outputs = session.run(None, {"inputs": torch.randn(1, 3, 32, 32).numpy()})
    assert outputs[0].shape == (1, 4)


def test_build_variant_model_builds_gpt2_model() -> None:
    variant = build_gpt2_variant()

    model = build_variant_model(variant)
    batch = build_example_batch(variant.variant_config, model, is_text_model=True)
    outputs = model(**batch)

    assert isinstance(model, Gpt2ForGnnArchsSequenceClassification)
    assert model.config.n_embd == 32
    assert model.config.pad_token_id == 0
    assert outputs.logits.shape == (2, 3)


@pytest.mark.parametrize(
    ("variant_config_overrides", "match"),
    [
        ({"gpt2_config": None}, "variant_config.gpt2_config"),
        (
            {"gpt2_config": build_gpt2_config_override(n_embd=30, n_head=8)},
            "divisible",
        ),
        (
            {"gpt2_config": build_gpt2_config_override(n_positions=4)},
            "n_positions",
        ),
        (
            {"gpt2_config": build_gpt2_config_override(resid_pdrop=1.0)},
            "resid_pdrop",
        ),
    ],
)
def test_build_variant_model_rejects_invalid_gpt2_config(
    variant_config_overrides: dict[str, object],
    match: str,
) -> None:
    variant = build_gpt2_variant(variant_config_overrides=variant_config_overrides)

    with pytest.raises(ValueError, match=match):
        build_variant_model(variant)


def test_build_variant_model_rejects_gpt2_mutations() -> None:
    variant = build_gpt2_variant(
        mutations=[
            {
                "type": "HiddenSizeModification",
                "params": {"hidden_size": 64},
            }
        ]
    )

    with pytest.raises(ValueError, match="do not support mutations"):
        build_variant_model(variant)


def test_gpt2_variant_runner_executes_text_pipeline(tmp_path: Path) -> None:
    config_path = tmp_path / "gpt2_variants.yaml"
    variant = build_gpt2_variant(
        variant_config_overrides={
            "run_training": True,
            "run_inference": True,
            "pre_inference_cooldown_seconds": 0.0,
            "inference_measurement_min_seconds": 1e-9,
        }
    )
    output_layout = prepare_output_layout(tmp_path / "output", config_path)
    context = RunContext(
        config_path=config_path,
        output_layout=output_layout,
        device=torch.device("cpu"),
        gpu_node="cpu-test",
        logger=logging.getLogger("test_gpt2_variant_runner"),
    )

    result = run_variant(variant, context)

    assert result.training is not None
    assert result.inference is not None
    assert result.metadata["model_kind"] == "text"
    assert result.metadata["validation_num_outputs"] == 3


def test_gpt2_architecture_only_onnx_randomizes_to_runtime_inputs(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "gpt2_variants.yaml"
    variant = build_gpt2_variant(
        variant_name="gpt2_architecture_only",
        variant_config_overrides={
            "export_onnx": True,
            "onnx_export_mode": "architecture_only",
        },
    )
    output_layout = prepare_output_layout(tmp_path / "output", config_path)
    context = RunContext(
        config_path=config_path,
        output_layout=output_layout,
        device=torch.device("cpu"),
        gpu_node="cpu-test",
        logger=logging.getLogger("test_gpt2_architecture_only"),
    )

    result = run_variant(variant, context)
    assert result.onnx_export is not None
    assert result.onnx_export.graph_info["runtime_input_names"] == [
        "input_ids",
        "attention_mask",
    ]
    assert result.onnx_export.graph_info["initializer_names"] == []
    assert result.onnx_export.graph_info["parameter_input_names"]

    randomized_path = output_layout.onnx_models_dir / "gpt2_randomized.onnx"
    write_randomized_onnx_model(
        Path(result.onnx_export.path),
        randomized_path,
        seed=0,
    )

    session = ort.InferenceSession(
        str(randomized_path),
        providers=["CPUExecutionProvider"],
    )
    assert [value.name for value in session.get_inputs()] == [
        "input_ids",
        "attention_mask",
    ]
    outputs = session.run(
        None,
        {
            "input_ids": torch.randint(0, 32, (2, 8), dtype=torch.long).numpy(),
            "attention_mask": torch.ones((2, 8), dtype=torch.long).numpy(),
        },
    )
    assert outputs[0].shape == (2, 3)


def test_qwen_architecture_only_onnx_keeps_full_logits(tmp_path: Path) -> None:
    config_path = tmp_path / "qwen3_variants.yaml"
    variant = build_qwen_variant(
        variant_name="qwen_architecture_only",
        variant_config_overrides={
            "export_onnx": True,
            "onnx_export_mode": "architecture_only",
        },
    )
    model = build_tiny_qwen_model()
    output_layout = prepare_output_layout(tmp_path / "output", config_path)
    context = RunContext(
        config_path=config_path,
        output_layout=output_layout,
        device=torch.device("cpu"),
        gpu_node="cpu-test",
        logger=logging.getLogger("test_qwen_architecture_only"),
    )

    assert model.config.use_cache is True
    assert model.generation_config.use_cache is True
    result = export_onnx_model(variant, model, context, True)

    assert result.graph_info["runtime_input_names"] == [
        "input_ids",
        "attention_mask",
    ]
    assert result.graph_info["output_names"] == [
        "logits",
        "present_0_key",
        "present_0_value",
    ]
    assert result.graph_info["initializer_names"] == []
    exported_model = onnx.load(result.path)
    output_shape = exported_model.graph.output[0].type.tensor_type.shape.dim
    assert [dimension.dim_value for dimension in output_shape] == [2, 8, 32]
    assert Path(result.path).name == "qwen_architecture_only_prefill.onnx"
    assert model.config.use_cache is True
    assert model.generation_config.use_cache is True


def test_gemma4_architecture_only_onnx_keeps_full_logits(tmp_path: Path) -> None:
    config_path = tmp_path / "gemma4_variants.yaml"
    variant = build_gemma4_variant(
        variant_name="gemma4_architecture_only",
        variant_config_overrides={
            "export_onnx": True,
            "onnx_export_mode": "architecture_only",
        },
    )
    model = build_tiny_gemma4_model()
    output_layout = prepare_output_layout(tmp_path / "output", config_path)
    context = RunContext(
        config_path=config_path,
        output_layout=output_layout,
        device=torch.device("cpu"),
        gpu_node="cpu-test",
        logger=logging.getLogger("test_gemma4_architecture_only"),
    )

    assert model.config.use_cache is True
    assert model.generation_config.use_cache is True
    result = export_onnx_model(variant, model, context, True)

    assert result.graph_info["runtime_input_names"] == [
        "input_ids",
        "attention_mask",
    ]
    assert result.graph_info["output_names"] == [
        "logits",
        "present_0_key",
        "present_0_value",
        "present_1_key",
        "present_1_value",
    ]
    assert result.graph_info["initializer_names"] == []
    exported_model = onnx.load(result.path)
    output_shape = exported_model.graph.output[0].type.tensor_type.shape.dim
    assert [dimension.dim_value for dimension in output_shape] == [2, 8, 32]
    assert Path(result.path).name == "gemma4_architecture_only_prefill.onnx"
    assert model.config.use_cache is True
    assert model.generation_config.use_cache is True


def test_build_variant_model_builds_qwen3_model_from_config() -> None:
    variant = build_qwen_variant(
        variant_config_overrides={
            "qwen3_config": {
                "vocab_size": 64,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "max_position_embeddings": 32,
            }
        }
    )

    model = build_variant_model(variant)

    assert isinstance(model, Qwen3ForCausalLM)
    assert model.config.model_type == "qwen3"
    assert model.config.vocab_size == 64
    assert model.config.hidden_size == 32
    assert model.config.num_hidden_layers == 2
    assert model.config.use_cache is True
    assert next(model.parameters()).dtype is torch.float16


def test_build_variant_model_builds_gemma4_model_from_config() -> None:
    variant = build_gemma4_variant(
        variant_config_overrides={
            "gemma4_config": build_gemma4_config_override(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=64,
            )
        }
    )

    model = build_variant_model(variant)

    assert isinstance(model, Gemma4ForCausalLM)
    assert model.config.model_type == "gemma4_text"
    assert model.config.vocab_size == 64
    assert model.config.hidden_size == 32
    assert model.config.num_hidden_layers == 3
    assert model.config.layer_types == [
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    assert model.config.use_cache is True
    assert next(model.parameters()).dtype is torch.float16


def test_build_variant_model_dispatches_llama_to_causal_lm_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variant = build_llama_variant()
    tiny_model = build_tiny_llama_model()
    recorded: dict[str, object] = {}

    def fake_from_pretrained(model_path: Path, **kwargs: object) -> LlamaForCausalLM:
        recorded["model_path"] = model_path
        recorded.update(kwargs)
        return tiny_model

    monkeypatch.setattr(
        causal_lm_builder_module,
        "resolve_causal_lm_model_path",
        lambda model_name: tmp_path,
    )
    monkeypatch.setattr(
        causal_lm_builder_module.AutoModelForCausalLM,
        "from_pretrained",
        fake_from_pretrained,
    )

    model = build_variant_model(variant)

    assert model is tiny_model
    assert recorded["model_path"] == tmp_path
    assert recorded["dtype"] is torch.float16
    assert recorded["local_files_only"] is True
    assert model.config.use_cache is True


def test_build_variant_model_rejects_llama_mutations() -> None:
    variant = build_llama_variant(
        mutations=[
            {
                "type": "HiddenSizeModification",
                "params": {"hidden_size": 64},
            }
        ]
    )

    with pytest.raises(ValueError, match="do not support mutations"):
        build_variant_model(variant)


def test_causal_lm_variant_runner_records_family_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "llama_variants.yaml"
    variant = build_llama_variant()
    output_layout = prepare_output_layout(tmp_path / "output", config_path)
    context = RunContext(
        config_path=config_path,
        output_layout=output_layout,
        device=torch.device("cpu"),
        gpu_node="cpu-test",
        logger=logging.getLogger("test_llama_variant_runner"),
    )

    monkeypatch.setattr(
        variant_runner_module,
        "build_variant_model",
        lambda spec: build_tiny_llama_model(),
    )

    result = run_variant(variant, context)

    assert result.metadata["model_kind"] == "llama"
    assert result.metadata["pretrained_weights_loaded"] is True
    assert result.metadata["validation_num_outputs"] == 32


def test_causal_lm_architecture_only_onnx_keeps_full_logits(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "llama_variants.yaml"
    variant = build_llama_variant(
        variant_name="llama_architecture_only",
        variant_config_overrides={
            "export_onnx": True,
            "onnx_export_mode": "architecture_only",
        },
    )
    model = build_tiny_llama_model()
    model.config._attn_implementation = "sdpa"
    output_layout = prepare_output_layout(tmp_path / "output", config_path)
    context = RunContext(
        config_path=config_path,
        output_layout=output_layout,
        device=torch.device("cpu"),
        gpu_node="cpu-test",
        logger=logging.getLogger("test_llama_architecture_only"),
    )

    assert model.config.use_cache is True
    assert model.generation_config.use_cache is True
    result = export_onnx_model(variant, model, context, True)

    assert result.graph_info["runtime_input_names"] == [
        "input_ids",
        "attention_mask",
    ]
    assert result.graph_info["output_names"] == [
        "logits",
        "present_0_key",
        "present_0_value",
    ]
    assert result.graph_info["initializer_names"] == []
    assert model.config._attn_implementation == "sdpa"
    assert model.config.use_cache is True
    assert model.generation_config.use_cache is True
    exported_model = onnx.load(result.path)
    output_shape = exported_model.graph.output[0].type.tensor_type.shape.dim
    assert [dimension.dim_value for dimension in output_shape] == [2, 8, 32]
    assert Path(result.path).name == "llama_architecture_only_prefill.onnx"


def test_causal_lm_decode_onnx_exports_past_kv_inputs(tmp_path: Path) -> None:
    config_path = tmp_path / "qwen3_variants.yaml"
    variant = build_qwen_variant(
        variant_name="qwen_decode_cached",
        variant_config_overrides={
            "export_onnx": True,
            "onnx_export_mode": "architecture_only",
            "run_decode": True,
            "decode_max_output_length": 3,
        },
    )
    model = build_tiny_qwen_model()
    output_layout = prepare_output_layout(tmp_path / "output", config_path)
    context = RunContext(
        config_path=config_path,
        output_layout=output_layout,
        device=torch.device("cpu"),
        gpu_node="cpu-test",
        logger=logging.getLogger("test_qwen_decode_cached"),
    )

    result = export_causal_lm_decode_onnx(variant, model, context)

    assert Path(result.path).name == "qwen_decode_cached_decode.onnx"
    assert result.graph_info["output_names"] == [
        "logits",
        "present_0_key",
        "present_0_value",
    ]
    runtime_input_names = result.graph_info["runtime_input_names"]
    assert runtime_input_names == [
        "input_ids",
        "attention_mask",
        "past_0_key",
        "past_0_value",
    ]
    onnx_model = onnx.load(result.path)
    graph_input_names = {value.name for value in onnx_model.graph.input}
    assert set(runtime_input_names) <= graph_input_names
    logits_shape = onnx_model.graph.output[0].type.tensor_type.shape.dim
    assert [dimension.dim_value for dimension in logits_shape] == [2, 1, 32]
    past_key_input = next(
        value for value in onnx_model.graph.input if value.name == "past_0_key"
    )
    past_key_shape = [
        dimension.dim_value
        for dimension in past_key_input.type.tensor_type.shape.dim
    ]
    assert past_key_shape == [2, 2, 10, 8]

    data = build_graph_data_from_onnx(
        result.path,
        batch_size=2,
        runtime_input_names=list(runtime_input_names),
        phase="decode",
        gpu_name="v100",
        decode_output_length=3,
    )
    assert data.x is not None
    assert data.x.shape[0] > 0
    assert data.graph_features.shape[0] == 1


def test_build_variant_model_builds_t5_model_with_eos_inputs() -> None:
    variant = build_t5_variant()

    model = build_variant_model(variant)
    batch = build_example_batch(variant.variant_config, model, is_text_model=True)
    outputs = model(**batch)

    assert isinstance(model, T5ForSequenceClassification)
    eos_token_id = require_t5_test_eos_token_id(model)
    assert model.config.model_type == "t5"
    assert batch["input_ids"].shape == (2, 6)
    assert torch.all(batch["input_ids"][:, -1] == eos_token_id)
    assert torch.all((batch["input_ids"] == eos_token_id).sum(dim=1) == 1)
    assert outputs.logits.shape == (2, 3)


@pytest.mark.parametrize(
    ("variant_config_overrides", "match"),
    [
        ({"t5_config": None}, "variant_config.t5_config"),
        (
            {"t5_config": build_t5_config_override(d_model=30, num_heads=8, d_kv=None)},
            "divisible",
        ),
        (
            {"t5_config": build_t5_config_override(feed_forward_proj="gelu")},
            "feed_forward_proj",
        ),
        (
            {"t5_config": build_t5_config_override(dropout_rate=1.0)},
            "dropout_rate",
        ),
    ],
)
def test_build_variant_model_rejects_invalid_t5_config(
    variant_config_overrides: dict[str, object],
    match: str,
) -> None:
    variant = build_t5_variant(variant_config_overrides=variant_config_overrides)

    with pytest.raises(ValueError, match=match):
        build_variant_model(variant)


def test_build_variant_model_rejects_t5_mutations() -> None:
    variant = build_t5_variant(
        mutations=[
            {
                "type": "HiddenSizeModification",
                "params": {"hidden_size": 64},
            }
        ]
    )

    with pytest.raises(ValueError, match="do not support mutations"):
        build_variant_model(variant)


def test_t5_variant_runner_executes_text_pipeline(tmp_path: Path) -> None:
    config_path = tmp_path / "t5_variants.yaml"
    variant = build_t5_variant(
        variant_config_overrides={
            "run_training": True,
            "run_inference": True,
            "pre_inference_cooldown_seconds": 0.0,
            "inference_measurement_min_seconds": 1e-9,
        }
    )
    output_layout = prepare_output_layout(tmp_path / "output", config_path)
    context = RunContext(
        config_path=config_path,
        output_layout=output_layout,
        device=torch.device("cpu"),
        gpu_node="cpu-test",
        logger=logging.getLogger("test_t5_variant_runner"),
    )

    result = run_variant(variant, context)

    assert result.training is not None
    assert result.inference is not None
    assert result.metadata["model_kind"] == "text"
    assert result.metadata["validation_num_outputs"] == 3


def test_t5_architecture_only_onnx_randomizes_to_runtime_inputs(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "t5_variants.yaml"
    variant = build_t5_variant(
        variant_name="t5_architecture_only",
        variant_config_overrides={
            "export_onnx": True,
            "onnx_export_mode": "architecture_only",
        },
    )
    output_layout = prepare_output_layout(tmp_path / "output", config_path)
    context = RunContext(
        config_path=config_path,
        output_layout=output_layout,
        device=torch.device("cpu"),
        gpu_node="cpu-test",
        logger=logging.getLogger("test_t5_architecture_only"),
    )

    result = run_variant(variant, context)
    assert result.onnx_export is not None
    assert result.onnx_export.graph_info["runtime_input_names"] == [
        "input_ids",
        "attention_mask",
    ]
    assert result.onnx_export.graph_info["initializer_names"] == []
    assert result.onnx_export.graph_info["parameter_input_names"]

    randomized_path = output_layout.onnx_models_dir / "t5_randomized.onnx"
    write_randomized_onnx_model(
        Path(result.onnx_export.path),
        randomized_path,
        seed=0,
    )

    session = ort.InferenceSession(
        str(randomized_path),
        providers=["CPUExecutionProvider"],
    )
    assert [value.name for value in session.get_inputs()] == [
        "input_ids",
        "attention_mask",
    ]
    input_ids = torch.randint(2, 32, (2, 6), dtype=torch.long)
    input_ids[:, -1] = 1
    outputs = session.run(
        None,
        {
            "input_ids": input_ids.numpy(),
            "attention_mask": torch.ones((2, 6), dtype=torch.long).numpy(),
        },
    )
    output = outputs[0]
    assert isinstance(output, np.ndarray)
    assert output.shape == (2, 3)


@pytest.mark.parametrize("base_model_name", ["deepfm", "dcn", "dcnv2", "edcn"])
def test_build_variant_model_builds_recommender_model(base_model_name: str) -> None:
    variant = build_recommender_variant(base_model_name=base_model_name)

    model = build_variant_model(variant)
    batch = build_example_batch(variant.variant_config, model, is_text_model=False)
    outputs = model(batch)
    metrics = variant_runner_module.validate_model(
        variant,
        model,
        torch.device("cpu"),
        is_text_model=False,
        is_recommender_model=True,
    )

    assert outputs.shape == (2,)
    assert metrics == {"batch_size": 2, "num_outputs": 1}


@pytest.mark.parametrize("base_model_name", ["deepfm", "dcn", "dcnv2", "edcn"])
def test_recommender_variant_runner_executes_pipeline(
    tmp_path: Path,
    base_model_name: str,
) -> None:
    config_path = tmp_path / f"{base_model_name}_variants.yaml"
    variant = build_recommender_variant(
        base_model_name=base_model_name,
        variant_name=f"{base_model_name}_runtime_smoke",
        variant_config_overrides={
            "run_training": True,
            "run_inference": True,
            "pre_inference_cooldown_seconds": 0.0,
        },
    )
    output_layout = prepare_output_layout(tmp_path / "output", config_path)
    context = RunContext(
        config_path=config_path,
        output_layout=output_layout,
        device=torch.device("cpu"),
        gpu_node="cpu-test",
        logger=logging.getLogger(f"test_{base_model_name}_variant_runner"),
    )

    result = run_variant(variant, context)

    assert result.training is not None
    assert result.inference is not None
    assert result.metadata["model_kind"] == "recommender"
    assert result.metadata["validation_num_outputs"] == 1


@pytest.mark.parametrize("base_model_name", ["deepfm", "dcn", "dcnv2", "edcn"])
def test_recommender_architecture_only_onnx_randomizes_to_runtime_inputs(
    tmp_path: Path,
    base_model_name: str,
) -> None:
    config_path = tmp_path / f"{base_model_name}_variants.yaml"
    variant = build_recommender_variant(
        base_model_name=base_model_name,
        variant_name=f"{base_model_name}_architecture_only",
        variant_config_overrides={
            "export_onnx": True,
            "onnx_export_mode": "architecture_only",
        },
    )
    model = build_variant_model(variant)
    feature_batch = build_example_batch(
        variant.variant_config,
        model,
        is_text_model=False,
    )
    output_layout = prepare_output_layout(tmp_path / "output", config_path)
    context = RunContext(
        config_path=config_path,
        output_layout=output_layout,
        device=torch.device("cpu"),
        gpu_node="cpu-test",
        logger=logging.getLogger(f"test_{base_model_name}_architecture_only"),
    )

    result = run_variant(variant, context)
    assert result.onnx_export is not None
    runtime_input_names = result.onnx_export.graph_info["runtime_input_names"]
    assert runtime_input_names == list(feature_batch)
    assert result.onnx_export.graph_info["initializer_names"] == []
    assert result.onnx_export.graph_info["parameter_input_names"]

    randomized_path = output_layout.onnx_models_dir / f"{base_model_name}_randomized.onnx"
    write_randomized_onnx_model(
        Path(result.onnx_export.path),
        randomized_path,
        seed=0,
    )

    session = ort.InferenceSession(
        str(randomized_path),
        providers=["CPUExecutionProvider"],
    )
    assert [value.name for value in session.get_inputs()] == runtime_input_names
    outputs = session.run(
        None,
        {name: tensor.numpy() for name, tensor in feature_batch.items()},
    )
    assert outputs[0].shape == (2, 1)


@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_recommender_fake_batches_use_feature_shapes(batch_size: int) -> None:
    variant = build_recommender_variant(
        variant_config_overrides={
            "example_input_shape": [batch_size],
            "batch_size": batch_size,
        }
    )
    model = build_variant_model(variant)
    example_batch = build_example_batch(
        variant.variant_config,
        model,
        is_text_model=False,
    )
    training_features, labels = build_training_batch(
        spec=variant,
        model=model,
        batch_size=batch_size,
        generator=torch.Generator().manual_seed(42),
    )

    assert set(example_batch) == {
        "user_id",
        "item_id",
        "device_type",
        "user_age_score",
        "item_price_score",
    }
    assert all(tensor.shape == (batch_size,) for tensor in example_batch.values())
    assert example_batch["user_id"].dtype == torch.long
    assert example_batch["user_age_score"].dtype == torch.float32
    assert isinstance(training_features, dict)
    assert labels.shape == (batch_size,)
    assert labels.dtype == torch.float32


def test_recommender_fake_batch_supports_vector_dense_features() -> None:
    variant = build_recommender_variant(
        model_config_overrides={
            "dense_features": [
                {"name": "user_age_score"},
                {"name": "engagement_vector", "embed_dim": 4},
            ]
        }
    )
    model = build_variant_model(variant)

    example_batch = build_example_batch(
        variant.variant_config,
        model,
        is_text_model=False,
    )

    assert example_batch["user_age_score"].shape == (2,)
    assert example_batch["engagement_vector"].shape == (2, 4)
    assert example_batch["engagement_vector"].dtype == torch.float32


@pytest.mark.parametrize(
        ("base_model_name", "config_field", "match"),
        [
            ("deepfm", "deepfm_config", "variant_config.deepfm_config"),
            ("dcn", "dcn_config", "variant_config.dcn_config"),
            ("dcnv2", "dcnv2_config", "variant_config.dcnv2_config"),
            ("edcn", "edcn_config", "variant_config.edcn_config"),
        ],
)
def test_build_variant_model_rejects_recommender_missing_model_config(
    base_model_name: str,
    config_field: str,
    match: str,
) -> None:
    variant = build_recommender_variant(base_model_name=base_model_name)
    variant_config = variant.variant_config.model_copy(update={config_field: None})
    invalid_variant = variant.model_copy(update={"variant_config": variant_config})

    with pytest.raises(ValueError, match=match):
        build_variant_model(invalid_variant)


def test_build_variant_model_rejects_deepfm_unknown_fm_feature() -> None:
    variant = build_recommender_variant(base_model_name="deepfm")
    deepfm_config = variant.variant_config.deepfm_config
    assert deepfm_config is not None
    variant_config = variant.variant_config.model_copy(
        update={
            "deepfm_config": deepfm_config.model_copy(
                update={"fm_feature_names": ["unknown_feature"]}
            )
        }
    )
    invalid_variant = variant.model_copy(update={"variant_config": variant_config})

    with pytest.raises(ValueError, match="fm_feature_names"):
        build_variant_model(invalid_variant)


def test_build_variant_model_rejects_dcn_missing_cross_layers() -> None:
    variant = build_recommender_variant(base_model_name="dcn")
    dcn_config = variant.variant_config.dcn_config
    assert dcn_config is not None
    variant_config = variant.variant_config.model_copy(
        update={
            "dcn_config": dcn_config.model_copy(update={"n_cross_layers": None})
        }
    )
    invalid_variant = variant.model_copy(update={"variant_config": variant_config})

    with pytest.raises(ValueError, match="n_cross_layers"):
        build_variant_model(invalid_variant)


@pytest.mark.parametrize(
    ("base_model_name", "model_config_overrides", "match"),
    [
        ("dcn", {"n_cross_layers": 0}, "n_cross_layers"),
        ("dcnv2", {"n_cross_layers": 0}, "n_cross_layers"),
        ("dcnv2", {"low_rank": 0}, "low_rank"),
        ("dcnv2", {"num_experts": 0}, "num_experts"),
        ("edcn", {"n_cross_layers": 0}, "n_cross_layers"),
        ("edcn", {"temperature": 0.0}, "temperature"),
    ],
)
def test_recommender_variant_rejects_non_positive_model_params(
    base_model_name: str,
    model_config_overrides: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        build_recommender_variant(
            base_model_name=base_model_name,
            model_config_overrides=model_config_overrides,
        )


def test_recommender_variant_rejects_multiple_model_configs() -> None:
    common_config = build_recommender_common_config_override(
        fm_feature_names=["user_id", "item_id"]
    )
    dcnv2_config = build_recommender_common_config_override(
        n_cross_layers=2,
        low_rank=4,
        num_experts=2,
        model_structure="parallel",
        use_low_rank_mixture=True,
    )

    with pytest.raises(ValueError, match="only one recommender config"):
        ArchConfig.model_validate(
            {
                "base_model_groups": [
                    {
                        "base_model": {"name": "dcnv2", "pretrained": False},
                        "single_variant_define": [
                            {
                                "name": "invalid_multi_recommender",
                                "variant_config": {
                                    "target_output_classes": 1,
                                    "example_input_shape": [2],
                                    "deepfm_config": common_config,
                                    "dcnv2_config": dcnv2_config,
                                },
                                "mutations": [],
                            }
                        ],
                    }
                ]
            }
        )


def test_text_variant_runner_executes_text_pipeline(tmp_path: Path) -> None:
    config_path = tmp_path / "text_variants.yaml"
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
                                "batch_size": 2,
                                "training_measurement_min_seconds": 1e-9,
                                "use_fake_text_dataset": True,
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
    output_layout = prepare_output_layout(tmp_path / "output", config_path)
    context = RunContext(
        config_path=config_path,
        output_layout=output_layout,
        device=torch.device("cpu"),
        gpu_node="cpu-test",
        logger=logging.getLogger("test_text_variant_runner"),
    )

    result = run_variant(variant, context)

    assert result.training is not None
    assert result.inference is not None
    assert result.metadata["model_kind"] == "text"
    assert result.metadata["validation_num_outputs"] == 3


def test_run_variant_waits_between_training_and_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variant = build_image_variant([], variant_name="cooldown_variant").model_copy(
        update={
            "variant_config": build_image_variant(
                [],
                variant_name="cooldown_variant_config",
            ).variant_config.model_copy(
                update={
                    "run_training": True,
                    "run_inference": True,
                    "pre_inference_cooldown_seconds": 3.0,
                }
            )
        }
    )
    config_path = tmp_path / "cooldown_variants.yaml"
    output_layout = prepare_output_layout(tmp_path / "output", config_path)
    context = RunContext(
        config_path=config_path,
        output_layout=output_layout,
        device=torch.device("cpu"),
        gpu_node="cpu-test",
        logger=logging.getLogger("test_variant_cooldown"),
    )
    events: list[str] = []

    monkeypatch.setattr(
        variant_runner_module,
        "build_variant_model",
        lambda spec: torch.nn.Identity(),
    )
    monkeypatch.setattr(
        variant_runner_module,
        "validate_model",
        lambda *args, **kwargs: {"batch_size": 1, "num_outputs": 4},
    )

    def fake_train_model(*args: object, **kwargs: object) -> TrainingResult:
        events.append("train")
        return TrainingResult(
            timings=TimeWindow(
                started_at_ts=10.0,
                ended_at_ts=20.0,
                started_at_text="2026-04-11T00:00:10+00:00",
                ended_at_text="2026-04-11T00:00:20+00:00",
            )
        )

    def fake_run_inference(*args: object, **kwargs: object) -> InferenceResult:
        events.append("infer")
        return InferenceResult(
            metrics={
                "iterations": 5,
                "batch_size": 1,
                "avg_latency_ms": 1000.0,
                "num_outputs": 4,
            },
            timings=TimeWindow(
                started_at_ts=23.0,
                ended_at_ts=28.0,
                started_at_text="2026-04-11T00:00:23+00:00",
                ended_at_text="2026-04-11T00:00:28+00:00",
            ),
        )

    monkeypatch.setattr(variant_runner_module, "train_model", fake_train_model)
    monkeypatch.setattr(variant_runner_module, "run_inference", fake_run_inference)
    monkeypatch.setattr(
        variant_runner_module.time,
        "sleep",
        lambda seconds: events.append(f"sleep:{seconds}"),
    )
    monkeypatch.setattr(
        variant_runner_module,
        "cleanup_workload_boundary",
        lambda device: events.append(f"cleanup:{device}"),
    )

    result = run_variant(variant, context)

    assert events == ["train", "sleep:3.0", "infer", "cleanup:cpu"]
    assert result.timings["training"].started_at_ts == 10.0
    assert result.timings["training"].ended_at_ts == 20.0
    assert result.timings["inference"].started_at_ts == 23.0
    assert result.timings["inference"].ended_at_ts == 28.0


def test_run_inference_measures_until_min_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variant = build_image_variant(
        [],
        variant_name="timed_inference_variant",
        example_input_shape=[1, 3, 16, 16],
    ).model_copy(
        update={
            "variant_config": build_image_variant(
                [],
                variant_name="timed_inference_config",
                example_input_shape=[1, 3, 16, 16],
            ).variant_config.model_copy(
                update={
                    "run_inference": True,
                    "inference_measurement_min_seconds": 5.0,
                }
            )
        }
    )
    forward_calls: list[str] = []
    time_values = iter([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])

    monkeypatch.setattr(
        variant_runner_module,
        "build_example_batch",
        lambda *args, **kwargs: {"inputs": torch.ones(1, 3, 16, 16)},
    )

    def fake_forward_model(*args: object, **kwargs: object) -> torch.Tensor:
        forward_calls.append("forward")
        return torch.ones(1, 4)

    monkeypatch.setattr(variant_runner_module, "forward_model", fake_forward_model)
    monkeypatch.setattr(variant_runner_module.time, "time", lambda: next(time_values))

    result = run_inference(
        variant,
        torch.nn.Identity(),
        torch.device("cpu"),
        False,
    )

    assert len(forward_calls) == 7
    assert result.metrics["iterations"] == 5
    assert result.metrics["avg_latency_ms"] == 1000.0
    assert result.timings.started_at_ts == 100.0
    assert result.timings.ended_at_ts == 105.0


def test_train_model_measures_gemma4_sft() -> None:
    variant = build_gemma4_variant(
        variant_config_overrides={
            "run_training": True,
            "training_measurement_min_seconds": 1e-9,
        }
    )
    model = build_tiny_gemma4_model()

    result = train_model(variant, model, torch.device("cpu"))

    assert result.metrics["total_steps"] >= 1
    assert np.isfinite(result.metrics["final_loss"])


def test_run_prefill_measures_qwen_forward() -> None:
    variant = build_qwen_variant(
        variant_config_overrides={
            "run_prefill": True,
            "prefill_measurement_min_seconds": 1e-9,
        }
    )
    model = build_tiny_qwen_model()

    result = run_prefill(variant, model, torch.device("cpu"))

    assert result.metrics["iterations"] >= 1
    assert result.metrics["batch_size"] == 2
    assert result.metrics["sequence_length"] == 8
    assert result.metrics["num_outputs"] == model.config.vocab_size


def test_run_prefill_measures_gemma4_forward() -> None:
    variant = build_gemma4_variant(
        variant_config_overrides={
            "run_prefill": True,
            "prefill_measurement_min_seconds": 1e-9,
        }
    )
    model = build_tiny_gemma4_model()

    result = run_prefill(variant, model, torch.device("cpu"))

    assert result.metrics["iterations"] >= 1
    assert result.metrics["batch_size"] == 2
    assert result.metrics["sequence_length"] == 8
    assert result.metrics["num_outputs"] == model.config.vocab_size


def test_run_decode_measures_qwen_generation() -> None:
    variant = build_qwen_variant(
        variant_config_overrides={
            "run_decode": True,
            "decode_max_output_length": 4,
            "decode_measurement_min_seconds": 1e-9,
        }
    )
    model = build_tiny_qwen_model()

    result = run_decode(variant, model, torch.device("cpu"))

    assert result.metrics["iterations"] >= 1
    assert result.metrics["batch_size"] == 2
    assert result.metrics["sequence_length"] == 8
    assert result.metrics["decode_max_output_length"] == 4
    assert result.metrics["generated_output_length"] <= 4


def test_run_decode_measures_gemma4_generation() -> None:
    variant = build_gemma4_variant(
        variant_config_overrides={
            "run_decode": True,
            "decode_max_output_length": 4,
            "decode_measurement_min_seconds": 1e-9,
        }
    )
    model = build_tiny_gemma4_model()

    result = run_decode(variant, model, torch.device("cpu"))

    assert result.metrics["iterations"] >= 1
    assert result.metrics["batch_size"] == 2
    assert result.metrics["sequence_length"] == 8
    assert result.metrics["decode_max_output_length"] == 4
    assert result.metrics["generated_output_length"] <= 4


def test_run_prefill_measures_llama_forward() -> None:
    variant = build_llama_variant(
        variant_config_overrides={
            "run_prefill": True,
            "prefill_measurement_min_seconds": 1e-9,
        }
    )
    model = build_tiny_llama_model()

    result = run_prefill(variant, model, torch.device("cpu"))

    assert result.metrics["iterations"] >= 1
    assert result.metrics["batch_size"] == 2
    assert result.metrics["sequence_length"] == 8
    assert result.metrics["num_outputs"] == model.config.vocab_size


def test_cleanup_workload_boundary_clears_cuda_cache_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gc_calls: list[str] = []
    cuda_calls: list[tuple[str, str | None]] = []

    monkeypatch.setattr(
        variant_runner_module.gc,
        "collect",
        lambda: gc_calls.append("gc"),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: cuda_calls.append(("sync", str(device))),
    )
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: cuda_calls.append(("empty_cache", None)),
    )

    variant_runner_module.cleanup_workload_boundary(torch.device("cuda:0"))

    assert gc_calls == ["gc"]
    assert cuda_calls == [("sync", "cuda:0"), ("empty_cache", None)]


def test_build_training_batch_keeps_fake_tensors_on_cpu() -> None:
    image_variant = build_image_variant(
        [],
        variant_name="image_cpu_dataset",
        example_input_shape=[1, 3, 16, 16],
    )
    image_model = build_variant_model(image_variant)
    image_inputs, image_labels = build_training_batch(
        spec=image_variant,
        model=image_model,
        batch_size=2,
        generator=torch.Generator().manual_seed(42),
    )

    assert image_inputs.device.type == "cpu"
    assert image_labels.device.type == "cpu"
    assert image_inputs.shape == (2, 3, 16, 16)
    assert image_labels.shape == (2,)

    text_variant = build_text_variant([], variant_name="text_cpu_batch")
    text_model = build_variant_model(text_variant)
    text_input_ids, text_attention_mask, text_labels = build_training_batch(
        spec=text_variant,
        model=text_model,
        batch_size=2,
        generator=torch.Generator().manual_seed(42),
    )

    assert text_input_ids.device.type == "cpu"
    assert text_attention_mask.device.type == "cpu"
    assert text_labels.device.type == "cpu"
    assert text_input_ids.shape == (2, 8)
    assert text_attention_mask.shape == (2, 8)
    assert text_labels.shape == (2,)

    t5_variant = build_t5_variant(variant_name="t5_cpu_batch")
    t5_model = build_variant_model(t5_variant)
    assert isinstance(t5_model, T5ForSequenceClassification)
    t5_input_ids, t5_attention_mask, t5_labels = build_training_batch(
        spec=t5_variant,
        model=t5_model,
        batch_size=2,
        generator=torch.Generator().manual_seed(42),
    )

    assert t5_input_ids.device.type == "cpu"
    assert t5_attention_mask.device.type == "cpu"
    assert t5_labels.device.type == "cpu"
    eos_token_id = require_t5_test_eos_token_id(t5_model)
    assert torch.all(t5_input_ids[:, -1] == eos_token_id)
    assert torch.all((t5_input_ids == eos_token_id).sum(dim=1) == 1)

    qwen_variant = build_qwen_variant(variant_name="qwen_cpu_batch")
    qwen_model = build_tiny_qwen_model()
    qwen_input_ids, qwen_attention_mask, qwen_labels = build_training_batch(
        spec=qwen_variant,
        model=qwen_model,
        batch_size=2,
        generator=torch.Generator().manual_seed(42),
    )

    assert qwen_input_ids.device.type == "cpu"
    assert qwen_attention_mask.device.type == "cpu"
    assert qwen_labels.device.type == "cpu"
    assert qwen_input_ids.shape == (2, 8)
    assert qwen_attention_mask.shape == (2, 8)
    assert qwen_labels.shape == (2, 8)
    assert torch.equal(qwen_labels, qwen_input_ids)

    llama_variant = build_llama_variant(variant_name="llama_cpu_batch")
    llama_model = build_tiny_llama_model()
    llama_input_ids, llama_attention_mask, llama_labels = build_training_batch(
        spec=llama_variant,
        model=llama_model,
        batch_size=2,
        generator=torch.Generator().manual_seed(42),
    )

    assert llama_input_ids.device.type == "cpu"
    assert llama_attention_mask.device.type == "cpu"
    assert llama_labels.device.type == "cpu"
    assert llama_input_ids.shape == (2, 8)
    assert llama_attention_mask.shape == (2, 8)
    assert llama_labels.shape == (2, 8)
    assert torch.equal(llama_labels, llama_input_ids)


def test_build_example_batch_keeps_runtime_inputs_on_cpu() -> None:
    image_variant = build_image_variant(
        [],
        variant_name="image_cpu_batch",
        example_input_shape=[1, 3, 16, 16],
    )
    image_model = build_variant_model(image_variant)
    image_batch = build_example_batch(
        image_variant.variant_config,
        image_model,
        False,
    )

    assert image_batch["inputs"].device.type == "cpu"

    text_variant = build_text_variant([], variant_name="text_cpu_batch")
    text_model = build_variant_model(text_variant)
    text_batch = build_example_batch(
        text_variant.variant_config,
        text_model,
        True,
    )

    assert text_batch["input_ids"].device.type == "cpu"
    assert text_batch["attention_mask"].device.type == "cpu"

    t5_variant = build_t5_variant(variant_name="t5_cpu_batch")
    t5_model = build_variant_model(t5_variant)
    assert isinstance(t5_model, T5ForSequenceClassification)
    t5_batch = build_example_batch(
        t5_variant.variant_config,
        t5_model,
        True,
    )

    assert t5_batch["input_ids"].device.type == "cpu"
    assert t5_batch["attention_mask"].device.type == "cpu"
    eos_token_id = require_t5_test_eos_token_id(t5_model)
    assert torch.all(t5_batch["input_ids"][:, -1] == eos_token_id)

    qwen_variant = build_qwen_variant(variant_name="qwen_cpu_batch")
    qwen_model = build_tiny_qwen_model()
    qwen_batch = build_example_batch(
        qwen_variant.variant_config,
        qwen_model,
        True,
    )

    assert qwen_batch["input_ids"].device.type == "cpu"
    assert qwen_batch["attention_mask"].device.type == "cpu"
    assert qwen_batch["input_ids"].shape == (2, 8)
    assert qwen_batch["attention_mask"].shape == (2, 8)

    llama_variant = build_llama_variant(variant_name="llama_cpu_batch")
    llama_model = build_tiny_llama_model()
    llama_batch = build_example_batch(
        llama_variant.variant_config,
        llama_model,
        True,
    )

    assert llama_batch["input_ids"].device.type == "cpu"
    assert llama_batch["attention_mask"].device.type == "cpu"
    assert llama_batch["input_ids"].shape == (2, 8)
    assert llama_batch["attention_mask"].shape == (2, 8)


def test_run_variant_cleans_workload_boundary_on_success_and_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variant = build_image_variant([])
    config_path = tmp_path / "cleanup_variants.yaml"
    output_layout = prepare_output_layout(tmp_path / "output", config_path)
    context = RunContext(
        config_path=config_path,
        output_layout=output_layout,
        device=torch.device("cpu"),
        gpu_node="cpu-test",
        logger=logging.getLogger("test_variant_cleanup"),
    )
    cleanup_calls: list[str] = []

    monkeypatch.setattr(
        variant_runner_module,
        "cleanup_workload_boundary",
        lambda device: cleanup_calls.append(str(device)),
    )

    run_variant(variant, context)
    assert cleanup_calls == ["cpu"]

    cleanup_calls.clear()
    monkeypatch.setattr(
        variant_runner_module,
        "validate_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cleanup test")),
    )

    with pytest.raises(RuntimeError, match="cleanup test"):
        run_variant(variant, context)

    assert cleanup_calls == ["cpu"]


def test_build_variant_model_aligns_bert_hidden_size_pruning_to_attention_heads() -> None:
    pruned_variant = build_text_variant(
        [
            {
                "type": "BertHiddenSizePruning",
                "params": {"hidden_size_pruning_ratio": 0.1},
            }
        ],
        variant_name="bert_hidden_size_pruning_smoke",
    )

    pruned_model = build_variant_model(pruned_variant)

    assert pruned_model.config.hidden_size == 696
    assert pruned_model.config.num_attention_heads == 12
    assert pruned_model.config.hidden_size % pruned_model.config.num_attention_heads == 0


def test_build_variant_model_keeps_invalid_text_attention_heads_failing_fast() -> None:
    invalid_variant = build_text_variant(
        [
            {
                "type": "AttentionHeadsModification",
                "params": {"num_heads": 10},
            }
        ],
        variant_name="bert_invalid_attention_heads",
    )

    with pytest.raises(
        ValueError,
        match="hidden_size must be divisible by num_attention_heads",
    ):
        build_variant_model(invalid_variant)


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
    config_path = tmp_path / "serialization_variants.yaml"
    output_layout = prepare_output_layout(tmp_path / "output", config_path)
    context = RunContext(
        config_path=config_path,
        output_layout=output_layout,
        device=torch.device("cpu"),
        gpu_node="cpu-test",
        logger=logging.getLogger("test_result_document"),
    )

    result = run_variant(variant, context)
    document = ResultDocument(
        config_path=str(context.config_path),
        gpu_node=context.gpu_node,
        variants=[result],
        summary={"variant_count": 1},
    )
    output_path = tmp_path / "result.json"
    write_result_document(output_path, document)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "2.0.0"
    assert payload["variants"][0]["source"] == "single_variant_define"
    assert "gpu_ids" not in payload


def test_derive_config_output_name_strips_variants_suffix() -> None:
    assert derive_config_output_name(Path("/tmp/bert_large_variants.yaml")) == (
        "bert_large"
    )
    assert derive_config_output_name(Path("/tmp/runtime.yaml")) == "runtime"


def test_prepare_output_layout_uses_dataset_root_directories(
    tmp_path: Path,
) -> None:
    output_layout = prepare_output_layout(
        tmp_path / "output",
        tmp_path / "bert_large_variants.yaml",
    )

    dataset_root = tmp_path / "output" / "bert_large"
    assert output_layout.root == dataset_root
    assert output_layout.onnx_models_dir == dataset_root / "onnx_models"
    assert output_layout.results_dir == dataset_root / "results"
    assert output_layout.logs_dir == dataset_root / "logs"
    for directory in (
        output_layout.root,
        output_layout.onnx_models_dir,
        output_layout.results_dir,
        output_layout.logs_dir,
    ):
        assert directory.is_dir()


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


def _build_yolo_variant(
    mutations: list[dict[str, object]],
    *,
    base_model_name: str = "yolo11n",
    variant_name: str = "yolo_smoke",
) -> object:
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
                                "target_output_classes": 10,
                                "example_input_shape": [1, 3, 640, 640],
                                "run_training": True,
                                "run_inference": True,
                                "export_onnx": True,
                                "onnx_export_mode": "full",
                                "batch_size": 2,
                                "training_measurement_min_seconds": 1e-9,
                                "use_fake_imagenet": True,
                                "inference_measurement_min_seconds": 0.1,
                            },
                            "mutations": mutations,
                        }
                    ],
                }
            ]
        }
    )
    return expand_arch_config(config)[0]


def test_detection_variant_runner_executes_training_inference_and_onnx(
    tmp_path: Path,
) -> None:
    """YOLO 运行时全链路 smoke test：build → train → infer → export（含 m.export=True）。"""
    config_path = tmp_path / "yolo11n_variants.yaml"
    variant = _build_yolo_variant([], variant_name="yolo11n_runtime_smoke")
    output_layout = prepare_output_layout(tmp_path / "output", config_path)
    context = RunContext(
        config_path=config_path,
        output_layout=output_layout,
        device=torch.device("cpu"),
        gpu_node="cpu-test",
        logger=logging.getLogger("test_detection_variant_runner"),
    )

    result = run_variant(variant, context)

    assert result.training is not None
    assert result.inference is not None
    assert result.onnx_export is not None
    assert Path(result.onnx_export.path).exists()
    assert Path(result.onnx_export.path).parent == output_layout.onnx_models_dir
    exported_model = onnx.load(result.onnx_export.path)
    assert len(exported_model.graph.node) > 0
    assert result.metadata["model_kind"] == "detection"
    assert result.onnx_export.graph_info["runtime_input_names"] == ["images"]
    assert result.training.metrics["total_steps"] == 1


def test_detection_variant_runner_with_activation_mutation(
    tmp_path: Path,
) -> None:
    """YOLO YAML 级 mutation 运行时验证：ActivationOverride 能正确构建和导出。"""
    config_path = tmp_path / "yolo11n_mutation.yaml"
    variant = _build_yolo_variant(
        [{"type": "ActivationOverride", "params": {"activation": "nn.ReLU()"}}],
        variant_name="yolo11n_relu_smoke",
    )
    output_layout = prepare_output_layout(tmp_path / "output", config_path)
    context = RunContext(
        config_path=config_path,
        output_layout=output_layout,
        device=torch.device("cpu"),
        gpu_node="cpu-test",
        logger=logging.getLogger("test_detection_mutation"),
    )

    result = run_variant(variant, context)

    assert result.onnx_export is not None
    assert Path(result.onnx_export.path).exists()
    assert result.metadata["model_kind"] == "detection"


def test_detection_variant_rejects_unknown_mutation_type() -> None:
    """未知 YOLO mutation 类型应抛出 ValueError，而不是静默跳过。"""
    variant = _build_yolo_variant(
        [{"type": "NonExistentMutation", "params": {"foo": "bar"}}],
        variant_name="yolo_invalid_mutation",
    )
    with pytest.raises(ValueError, match=r"未知的 YOLO mutation 类型"):
        build_variant_model(variant)
