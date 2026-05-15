from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest
import yaml

from gnn_archs.config import (
    ArchConfig,
    BaseModelGroup,
    MutationSet,
    is_detection_model_name,
    is_text_model_name,
)
from gnn_archs.mutations import (
    IMAGE_MUTATION_TYPES,
    TEXT_MUTATION_TYPES,
    apply_text_config_mutations,
)
from gnn_archs.yolo_builder import YOLO_YAML_MUTATION_TYPES
from gnn_archs.util.variant_expander import expand_arch_config
from gnn_archs.variant_runner import build_bert_config


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
    "t5_variants.yaml": 16,
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
        variant.variant_config.onnx_export_mode == "architecture_only"
        for variant in variants
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

    assert len(variants) == expected_count
    assert all(variant.variant_config.gpt2_config is not None for variant in variants)
    assert all(not variant.mutations for variant in variants)


@pytest.mark.parametrize(
    ("config_name", "expected_count"),
    T5_CONFIG_VARIANT_COUNTS.items(),
)
def test_t5_arch_configs_expand_to_expected_counts(
    config_name: str, expected_count: int
) -> None:
    config = load_arch_config(ARCH_CONFIG_DIR / config_name)

    variants = expand_arch_config(config)

    assert len(variants) == expected_count
    assert all(variant.variant_config.t5_config is not None for variant in variants)
    assert all(not variant.mutations for variant in variants)


def test_gpt2_batch_sweep_variants_define_batch_in_name_and_config() -> None:
    config = load_arch_config(ARCH_CONFIG_DIR / "gpt2_variants.yaml")

    batch_variants = [
        variant for variant in expand_arch_config(config) if "_bs" in variant.name
    ]
    batch_sizes = {
        int(variant.name.rsplit("_bs", maxsplit=1)[1]) for variant in batch_variants
    }

    assert len(batch_variants) == 24
    assert batch_sizes == {1, 2, 4, 8}
    assert all(
        variant.variant_config.example_input_shape[0]
        == variant.variant_config.training_batch_sizes[0]
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
            variant.variant_config.run_training
            and variant.variant_config.run_inference
        ):
            continue
        assert variant.variant_config.pre_inference_cooldown_seconds == 5.0
        assert variant.variant_config.inference_measurement_min_seconds == 40.0
        assert variant.variant_config.training_measurement_min_seconds == 40.0


@pytest.mark.parametrize(
    "config_path",
    BERT_TEXT_CONFIG_PATHS,
    ids=[path.name for path in BERT_TEXT_CONFIG_PATHS],
)
def test_bert_text_configs_apply_mutations_without_invalid_hidden_head_pairs(
    config_path: Path,
) -> None:
    config = load_arch_config(config_path)

    for case_name, base_model, variant_config, mutations in iter_text_variant_cases(config):
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
