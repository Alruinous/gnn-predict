from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest
import yaml

from gnn_archs.config import ArchConfig, BaseModelGroup, MutationSet, is_text_model_name
from gnn_archs.mutations import (
    IMAGE_MUTATION_TYPES,
    TEXT_MUTATION_TYPES,
    apply_text_config_mutations,
)
from gnn_archs.util.variant_expander import expand_arch_config
from gnn_archs.variant_runner import build_bert_config


ROOT = Path(__file__).resolve().parents[1]
ARCH_CONFIG_DIR = ROOT / "config" / "arch"
ARCH_CONFIG_PATHS = sorted(ARCH_CONFIG_DIR.glob("*.yaml"), key=lambda path: path.name)
BERT_TEXT_CONFIG_PATHS = [
    ARCH_CONFIG_DIR / "bert_variants.yaml",
    ARCH_CONFIG_DIR / "bert_large_variants.yaml",
]


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
        variant.variant_config.onnx_export_mode
        == ("architecture_only" if variant.variant_config.export_onnx else "full")
        for variant in variants
    )


@pytest.mark.parametrize(
    "config_path",
    ARCH_CONFIG_PATHS,
    ids=[path.name for path in ARCH_CONFIG_PATHS],
)
def test_arch_config_mutations_match_model_kind(config_path: Path) -> None:
    config = load_arch_config(config_path)

    for group in config.base_model_groups:
        allowed_mutation_types = (
            TEXT_MUTATION_TYPES
            if is_text_model_name(group.base_model.name)
            else IMAGE_MUTATION_TYPES
        )
        for mutation_set in iter_group_mutation_sets(group):
            for mutation in mutation_set.mutations:
                assert mutation.type in allowed_mutation_types


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
