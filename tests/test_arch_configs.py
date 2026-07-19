from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest
import yaml

from gnn_archs.config import (
    ArchConfig,
    BaseModelGroup,
    MutationSet,
    ResolvedVariantSpec,
    is_causal_lm_model_name,
    is_detection_model_name,
    is_text_model_name,
)
from gnn_archs.mutations import (
    IMAGE_MUTATION_TYPES,
    TEXT_MUTATION_TYPES,
    apply_text_config_mutations,
)
from gnn_archs.util.variant_expander import expand_arch_config
from gnn_archs.variant_runner import build_bert_config
from gnn_archs.yolo_builder import YOLO_YAML_MUTATION_TYPES

ROOT = Path(__file__).resolve().parents[1]
ARCH_CONFIG_DIR = ROOT / "config" / "arch"
ARCH_CONFIG_PATHS = sorted(ARCH_CONFIG_DIR.glob("*.yaml"), key=lambda path: path.name)
BERT_TEXT_CONFIG_PATHS = sorted(
    ARCH_CONFIG_DIR.glob("bert*_variants*.yaml"), key=lambda path: path.name
)
NEW_IMAGE_CONFIG_VARIANT_COUNTS = {
    "efficientnet_variants.yaml": 288,
    "swin_variants.yaml": 108,
}
GPT2_CONFIG_VARIANT_COUNTS = {
    "gpt2_variants.yaml": 240,
}
T5_CONFIG_VARIANT_COUNTS = {
    "t5_variants.yaml": 240,
}
LLAMA_CONFIG_VARIANT_COUNTS = {
    "llama_variants.yaml": 22,
}
GEMMA_CONFIG_VARIANT_COUNTS = {
    "gemma_variants.yaml": 22,
}
GEMMA4_CONFIG_VARIANT_COUNTS = {
    "gemma4.yaml": 639,
}
CAUSAL_LM_FULL_FLOW_SHAPES = {
    (1, 128),
    (1, 256),
    (1, 384),
    (1, 512),
    (2, 128),
    (2, 256),
    (2, 384),
    (3, 128),
    (3, 256),
    (4, 128),
    (4, 256),
}


def load_arch_config(config_path: Path) -> ArchConfig:
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return ArchConfig.model_validate(raw_config)


def iter_group_mutation_sets(group: BaseModelGroup) -> Iterable[MutationSet]:
    for variant in group.single_variant_define:
        yield MutationSet(name=variant.name, mutations=variant.mutations)
    if group.combinatorial_variant_grid is not None:
        yield from group.combinatorial_variant_grid.mutation_sets
    yield from group.fc_mutation_sets


def iter_text_variant_cases(
    config: ArchConfig,
) -> Iterable[tuple[str, object, object, list[object]]]:
    for group in config.base_model_groups:
        if not is_text_model_name(group.base_model.name):
            continue

        for variant in group.single_variant_define:
            yield (
                variant.name,
                group.base_model,
                variant.variant_config,
                variant.mutations,
            )

        grid = group.combinatorial_variant_grid
        if grid is None:
            continue

        variant_config = grid.base_variant_config_template.model_copy(
            update={
                "target_input_channels": grid.input_channels[0],
                "target_output_classes": grid.output_classes[0],
            }
        )
        for mutation_set in grid.mutation_sets:
            yield (
                mutation_set.name,
                group.base_model,
                variant_config,
                mutation_set.mutations,
            )


@pytest.mark.parametrize(
    "config_path",
    ARCH_CONFIG_PATHS,
    ids=[path.name for path in ARCH_CONFIG_PATHS],
)
def test_arch_configs_validate_and_expand(config_path: Path) -> None:
    config = load_arch_config(config_path)

    variants = expand_arch_config(config)

    assert variants
    assert all(
        isinstance(variant.variant_config.export_graph, bool) for variant in variants
    )


@pytest.mark.parametrize(
    ("config_name", "expected_count"),
    NEW_IMAGE_CONFIG_VARIANT_COUNTS.items(),
)
def test_new_image_arch_configs_expand_to_expected_counts(
    config_name: str, expected_count: int
) -> None:
    config = load_arch_config(ARCH_CONFIG_DIR / config_name)

    variants = expand_arch_config(config)

    assert len(variants) == expected_count


@pytest.mark.parametrize(
    ("config_name", "expected_count"),
    GPT2_CONFIG_VARIANT_COUNTS.items(),
)
def test_gpt2_arch_configs_expand_to_expected_counts(
    config_name: str, expected_count: int
) -> None:
    config = load_arch_config(ARCH_CONFIG_DIR / config_name)

    variants = expand_arch_config(config)
    base_variants = [variant for variant in variants if "_bs" not in variant.name]
    batch_variants = [variant for variant in variants if "_bs" in variant.name]
    sequence_lengths = {
        variant.variant_config.example_input_shape[1] for variant in variants
    }

    assert len(variants) == expected_count
    assert len(base_variants) == 216
    assert len(batch_variants) == 24
    assert sequence_lengths == {128, 256, 512}
    assert all(variant.variant_config.gpt2_config is not None for variant in variants)
    assert all(not variant.mutations for variant in variants)
    assert all(
        variant.variant_config.example_input_shape[0] == 4
        and variant.variant_config.batch_size == 4
        for variant in base_variants
    )
    assert all(
        variant.variant_config.gpt2_config.n_positions
        >= variant.variant_config.example_input_shape[1]
        for variant in variants
        if variant.variant_config.gpt2_config is not None
    )
    assert all(
        variant.variant_config.gpt2_config.n_embd
        % variant.variant_config.gpt2_config.n_head
        == 0
        for variant in variants
        if variant.variant_config.gpt2_config is not None
    )


@pytest.mark.parametrize(
    ("config_name", "expected_count"),
    T5_CONFIG_VARIANT_COUNTS.items(),
)
def test_t5_arch_configs_expand_to_expected_counts(
    config_name: str, expected_count: int
) -> None:
    config = load_arch_config(ARCH_CONFIG_DIR / config_name)

    variants = expand_arch_config(config)
    base_variants = [
        variant for variant in variants if variant.source == "variant_config_grid"
    ]
    batch_variants = [variant for variant in variants if "_bs" in variant.name]
    sequence_lengths = {
        variant.variant_config.example_input_shape[1] for variant in variants
    }
    output_classes = {
        variant.variant_config.target_output_classes for variant in variants
    }
    t5_configs = [variant.variant_config.t5_config for variant in variants]

    assert len(variants) == expected_count
    assert len(base_variants) == 216
    assert len(batch_variants) == 24
    assert len(base_variants) + len(batch_variants) == len(variants)
    assert all(variant.source == "single_variant_define" for variant in batch_variants)
    assert sequence_lengths == {128, 256, 512}
    assert output_classes == {2, 10, 100}
    assert all(t5_config is not None for t5_config in t5_configs)
    assert all(not variant.mutations for variant in variants)
    assert all(
        variant.variant_config.example_input_shape[0] == 4
        and variant.variant_config.batch_size == 4
        for variant in base_variants
    )
    assert {
        variant.variant_config.example_input_shape[0] for variant in batch_variants
    } == {2, 4, 8, 16}
    assert all(
        variant.variant_config.example_input_shape[1] == 512
        and variant.variant_config.target_output_classes == 10
        for variant in batch_variants
    )
    assert all(
        t5_config is not None
        and t5_config.vocab_size == 16384
        and t5_config.relative_attention_max_distance == 512
        for variant in batch_variants
        for t5_config in [variant.variant_config.t5_config]
    )
    assert all(
        t5_config is not None
        and t5_config.d_kv is not None
        and t5_config.d_kv * t5_config.num_heads == t5_config.d_model
        and t5_config.dropout_rate == 0.0
        and t5_config.classifier_dropout == 0.0
        for t5_config in t5_configs
    )
    assert all(
        t5_config is not None
        and t5_config.relative_attention_max_distance
        >= variant.variant_config.example_input_shape[1]
        for variant in variants
        for t5_config in [variant.variant_config.t5_config]
    )


@pytest.mark.parametrize(
    ("config_name", "expected_count"),
    LLAMA_CONFIG_VARIANT_COUNTS.items(),
)
def test_llama_arch_configs_expand_to_expected_counts(
    config_name: str, expected_count: int
) -> None:
    config = load_arch_config(ARCH_CONFIG_DIR / config_name)

    variants = expand_arch_config(config)

    assert len(variants) == expected_count
    assert {variant.base_model.name for variant in variants} == {
        "Llama-3.2-1B",
        "Llama-3.2-3B",
    }
    assert_causal_lm_full_flow_variants(variants)


@pytest.mark.parametrize(
    ("config_name", "expected_count"),
    GEMMA_CONFIG_VARIANT_COUNTS.items(),
)
def test_gemma_arch_configs_expand_to_expected_counts(
    config_name: str, expected_count: int
) -> None:
    config = load_arch_config(ARCH_CONFIG_DIR / config_name)

    variants = expand_arch_config(config)

    assert len(variants) == expected_count
    assert {variant.base_model.name for variant in variants} == {
        "gemma-2-2b",
        "gemma-3-1b-pt",
    }
    assert_causal_lm_full_flow_variants(variants)


@pytest.mark.parametrize(
    ("config_name", "expected_count"),
    GEMMA4_CONFIG_VARIANT_COUNTS.items(),
)
def test_gemma4_arch_configs_expand_to_expected_counts(
    config_name: str, expected_count: int
) -> None:
    config = load_arch_config(ARCH_CONFIG_DIR / config_name)

    variants = expand_arch_config(config)
    full_flow_variants = [
        variant
        for variant in variants
        if variant.variant_config.run_prefill and variant.variant_config.run_decode
    ]
    decode_only_variants = [
        variant
        for variant in variants
        if variant.variant_config.run_decode and not variant.variant_config.run_prefill
    ]
    prefill_only_variants = [
        variant
        for variant in variants
        if variant.variant_config.run_prefill and not variant.variant_config.run_decode
    ]

    assert len(variants) == expected_count
    assert {variant.base_model.name for variant in variants} == {"gemma4"}
    assert all(not variant.base_model.pretrained for variant in variants)
    assert all(is_causal_lm_model_name(variant.base_model.name) for variant in variants)
    assert all(variant.source == "variant_config_grid" for variant in variants)
    assert all(not variant.variant_config.run_inference for variant in variants)
    assert all(variant.variant_config.use_fake_text_dataset for variant in variants)
    assert all(not variant.mutations for variant in variants)
    assert all(variant.variant_config.qwen3_config is None for variant in variants)
    assert all(variant.variant_config.gemma4_config is not None for variant in variants)
    assert all(
        variant.variant_config.target_input_channels is None for variant in variants
    )
    assert all(
        variant.variant_config.target_output_classes is None for variant in variants
    )
    assert all(
        variant.variant_config.batch_size
        == variant.variant_config.example_input_shape[0]
        for variant in variants
    )
    assert all(
        variant.variant_config.example_input_shape[1]
        + (
            variant.variant_config.decode_max_output_length
            if variant.variant_config.run_decode
            else 0
        )
        <= variant.variant_config.gemma4_config.max_position_embeddings
        for variant in variants
        if variant.variant_config.gemma4_config is not None
    )
    assert len(full_flow_variants) == 73
    assert len(decode_only_variants) == 554
    assert len(prefill_only_variants) == 12
    assert all(not variant.variant_config.run_training for variant in variants)
    assert all(
        variant.variant_config.decode_max_output_length > 0
        for variant in full_flow_variants + decode_only_variants
    )
    assert all(
        variant.variant_config.decode_max_output_length == 0
        for variant in prefill_only_variants
    )


def assert_causal_lm_full_flow_variants(
    variants: list[ResolvedVariantSpec],
) -> None:
    assert all(is_causal_lm_model_name(variant.base_model.name) for variant in variants)
    assert all(variant.source == "variant_config_grid" for variant in variants)
    assert all(variant.variant_config.run_training for variant in variants)
    assert all(variant.variant_config.run_prefill for variant in variants)
    assert all(not variant.variant_config.run_inference for variant in variants)
    assert all(variant.variant_config.use_fake_text_dataset for variant in variants)
    assert all(not variant.mutations for variant in variants)
    assert all(
        variant.variant_config.target_input_channels is None for variant in variants
    )
    assert all(
        variant.variant_config.target_output_classes is None for variant in variants
    )
    assert {
        tuple(variant.variant_config.example_input_shape) for variant in variants
    } == CAUSAL_LM_FULL_FLOW_SHAPES
    assert all(
        variant.variant_config.batch_size
        == variant.variant_config.example_input_shape[0]
        for variant in variants
    )
    assert all(
        variant.variant_config.example_input_shape[0]
        * variant.variant_config.example_input_shape[1]
        <= 1024
        for variant in variants
    )


def test_gpt2_batch_sweep_variants_define_batch_in_name_and_config() -> None:
    config = load_arch_config(ARCH_CONFIG_DIR / "gpt2_variants.yaml")

    batch_variants = [
        variant for variant in expand_arch_config(config) if "_bs" in variant.name
    ]
    batch_sizes = {
        int(variant.name.rsplit("_bs", maxsplit=1)[1]) for variant in batch_variants
    }

    assert len(batch_variants) == 24
    assert batch_sizes == {2, 4, 8, 16}
    assert all(
        variant.variant_config.example_input_shape[0]
        == variant.variant_config.batch_size
        for variant in batch_variants
    )
    assert all(
        variant.variant_config.example_input_shape[1] == 512
        and variant.variant_config.target_output_classes == 10
        for variant in batch_variants
    )


@pytest.mark.parametrize(
    "config_path",
    ARCH_CONFIG_PATHS,
    ids=[path.name for path in ARCH_CONFIG_PATHS],
)
def test_arch_config_mutations_match_model_kind(config_path: Path) -> None:
    config = load_arch_config(config_path)

    for group in config.base_model_groups:
        if is_detection_model_name(group.base_model.name):
            allowed_mutation_types = YOLO_YAML_MUTATION_TYPES
        elif is_text_model_name(group.base_model.name):
            allowed_mutation_types = TEXT_MUTATION_TYPES
        else:
            allowed_mutation_types = IMAGE_MUTATION_TYPES
        for mutation_set in iter_group_mutation_sets(group):
            for mutation in mutation_set.mutations:
                assert mutation.type in allowed_mutation_types


@pytest.mark.parametrize(
    "config_path",
    ARCH_CONFIG_PATHS,
    ids=[path.name for path in ARCH_CONFIG_PATHS],
)
def test_arch_configs_define_phase_isolation_for_training_inference_pairs(
    config_path: Path,
) -> None:
    config = load_arch_config(config_path)

    for variant in expand_arch_config(config):
        if not (
            variant.variant_config.run_training and variant.variant_config.run_inference
        ):
            continue


@pytest.mark.parametrize(
    "config_path",
    BERT_TEXT_CONFIG_PATHS,
    ids=[path.name for path in BERT_TEXT_CONFIG_PATHS],
)
def test_bert_text_configs_apply_mutations_without_invalid_hidden_head_pairs(
    config_path: Path,
) -> None:
    config = load_arch_config(config_path)

    for case_name, base_model, variant_config, mutations in iter_text_variant_cases(
        config
    ):
        bert_config = build_bert_config(base_model, variant_config)
        mutated_config = apply_text_config_mutations(bert_config, mutations)

        assert mutated_config.hidden_size % mutated_config.num_attention_heads == 0, (
            case_name
        )


@pytest.mark.parametrize(
    "config_path",
    BERT_TEXT_CONFIG_PATHS,
    ids=[path.name for path in BERT_TEXT_CONFIG_PATHS],
)
def test_bert_pruning_mutations_use_runtime_parameter_names(config_path: Path) -> None:
    config = load_arch_config(config_path)

    for group in config.base_model_groups:
        for mutation_set in iter_group_mutation_sets(group):
            for mutation in mutation_set.mutations:
                if mutation.type == "BertAttentionHeadPruning":
                    assert "head_pruning_ratio" in mutation.params
                    assert "pruning_ratio" not in mutation.params
                if mutation.type == "BertLayerPruning":
                    assert "target_layers" in mutation.params
