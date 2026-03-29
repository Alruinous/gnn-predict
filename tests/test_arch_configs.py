from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest
import yaml

from gnn_archs.config import ArchConfig, BaseModelGroup, MutationSet, is_text_model_name
from gnn_archs.mutations import IMAGE_MUTATION_TYPES, TEXT_MUTATION_TYPES
from gnn_archs.util.variant_expander import expand_arch_config


ROOT = Path(__file__).resolve().parents[1]
ARCH_CONFIG_DIR = ROOT / "config" / "arch"
ARCH_CONFIG_PATHS = sorted(ARCH_CONFIG_DIR.glob("*.yaml"), key=lambda path: path.name)


def load_arch_config(config_path: Path) -> ArchConfig:
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return ArchConfig.model_validate(raw_config)


def iter_group_mutation_sets(group: BaseModelGroup) -> Iterable[MutationSet]:
    for variant in group.single_variant_define:
        yield MutationSet(name=variant.name, mutations=variant.mutations)
    if group.combinatorial_variant_grid is not None:
        yield from group.combinatorial_variant_grid.mutation_sets
    yield from group.fc_mutation_sets


@pytest.mark.parametrize(
    "config_path",
    ARCH_CONFIG_PATHS,
    ids=[path.name for path in ARCH_CONFIG_PATHS],
)
def test_arch_configs_validate_and_expand(config_path: Path) -> None:
    config = load_arch_config(config_path)

    variants = expand_arch_config(config)

    assert variants


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
