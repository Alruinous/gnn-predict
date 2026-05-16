from __future__ import annotations

import copy
from itertools import product
from typing import TYPE_CHECKING, Any

from gnn_archs.config import (
    ResolvedVariantSpec,
    VariantConfig,
    is_recommender_model_name,
    is_text_model_name,
)

if TYPE_CHECKING:
    from gnn_archs.config import (
        ArchConfig,
        BaseModelGroup,
        VariantConfigGridAxisValue,
    )


def expand_arch_config(config: ArchConfig) -> list[ResolvedVariantSpec]:
    variants: list[ResolvedVariantSpec] = []
    for group in config.base_model_groups:
        variants.extend(expand_group_variants(group))
    return variants


def expand_group_variants(group: BaseModelGroup) -> list[ResolvedVariantSpec]:
    explicit_variants = build_explicit_variants(group)
    grid_variants = build_grid_variants(group)
    config_grid_variants = build_variant_config_grid_variants(group)
    fc_variants = build_fc_variants(group, grid_variants + config_grid_variants)
    variants = explicit_variants + grid_variants + config_grid_variants + fc_variants
    validate_unique_variant_names(variants)
    total_variants = len(variants)
    return [
        variant.model_copy(update={"group_total_variants_defined": total_variants})
        for variant in variants
    ]


def build_explicit_variants(group: BaseModelGroup) -> list[ResolvedVariantSpec]:
    return [
        ResolvedVariantSpec(
            name=variant.name,
            base_model=group.base_model,
            variant_config=variant.variant_config,
            mutations=variant.mutations,
            source="single_variant_define",
            group_total_variants_defined=0,
        )
        for variant in group.single_variant_define
    ]


def build_grid_variants(group: BaseModelGroup) -> list[ResolvedVariantSpec]:
    grid = group.combinatorial_variant_grid
    if grid is None:
        return []

    variants: list[ResolvedVariantSpec] = []
    for input_channels, output_classes, mutation_set in product(
        grid.input_channels,
        grid.output_classes,
        grid.mutation_sets,
    ):
        variant_name = build_grid_variant_name(
            group.base_model.name,
            input_channels,
            output_classes,
            mutation_set.name,
        )
        variant_config = resolve_variant_config(
            model_name=group.base_model.name,
            template=grid.base_variant_config_template,
            input_channels=input_channels,
            output_classes=output_classes,
        )
        variants.append(
            ResolvedVariantSpec(
                name=variant_name,
                base_model=group.base_model,
                variant_config=variant_config,
                mutations=mutation_set.mutations,
                source="combinatorial_variant_grid",
                group_total_variants_defined=0,
            )
        )

    return variants


def build_variant_config_grid_variants(
    group: BaseModelGroup,
) -> list[ResolvedVariantSpec]:
    grid = group.variant_config_grid
    if grid is None:
        return []

    variants: list[ResolvedVariantSpec] = []
    axes = grid.axes
    for value_combination in product(*[axis.values for axis in axes]):
        axis_values = dict(
            zip((axis.name for axis in axes), value_combination, strict=True)
        )
        variant_name = build_variant_config_grid_name(
            group.base_model.name,
            axis_values,
        )
        variant_config = resolve_variant_config_grid_template(
            grid.base_variant_config_template,
            value_combination,
        )
        variants.append(
            ResolvedVariantSpec(
                name=variant_name,
                base_model=group.base_model,
                variant_config=variant_config,
                mutations=grid.mutations,
                source="variant_config_grid",
                group_total_variants_defined=0,
            )
        )

    return variants


def build_fc_variants(
    group: BaseModelGroup, base_grid_variants: list[ResolvedVariantSpec]
) -> list[ResolvedVariantSpec]:
    if not base_grid_variants or not group.fc_mutation_sets:
        return []

    variants: list[ResolvedVariantSpec] = []
    for base_variant in base_grid_variants:
        for mutation_set in group.fc_mutation_sets:
            if mutation_set.name == "no_fc_mutation" and not mutation_set.mutations:
                continue
            variants.append(
                ResolvedVariantSpec(
                    name=f"{base_variant.name}_fc_{mutation_set.name}",
                    base_model=base_variant.base_model,
                    variant_config=base_variant.variant_config,
                    mutations=base_variant.mutations + mutation_set.mutations,
                    source="fc_mutation_sets",
                    group_total_variants_defined=0,
                )
            )
    return variants


def validate_unique_variant_names(variants: list[ResolvedVariantSpec]) -> None:
    names = [variant.name for variant in variants]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"variant names must be unique: {duplicate_names}")


def build_variant_config_grid_name(
    base_model_name: str,
    axis_values: dict[str, VariantConfigGridAxisValue],
) -> str:
    value_names = [value.name for value in axis_values.values()]
    return "_".join([base_model_name, *value_names])


def resolve_variant_config_grid_template(
    template: VariantConfig,
    values: tuple[VariantConfigGridAxisValue, ...],
) -> VariantConfig:
    raw_config = template.model_dump(mode="python")
    for value in values:
        for field_path, override_value in value.overrides.items():
            apply_field_path_override(raw_config, field_path, override_value)
    return VariantConfig.model_validate(raw_config)


def apply_field_path_override(
    raw_config: dict[str, Any],
    field_path: str,
    override_value: Any,
) -> None:
    path_parts = field_path.split(".")
    target: Any = raw_config
    for path_part in path_parts[:-1]:
        if not isinstance(target, dict) or path_part not in target:
            raise ValueError(
                f"variant_config_grid override path is unknown: {field_path}"
            )
        target = target[path_part]
    final_part = path_parts[-1]
    if not isinstance(target, dict) or final_part not in target:
        raise ValueError(f"variant_config_grid override path is unknown: {field_path}")
    target[final_part] = copy.deepcopy(override_value)


def build_grid_variant_name(
    base_model_name: str,
    input_channels: int,
    output_classes: int,
    mutation_set_name: str,
) -> str:
    return (
        f"{base_model_name}_ic{input_channels}_oc{output_classes}_{mutation_set_name}"
    )


def resolve_variant_config(
    model_name: str,
    template: VariantConfig,
    input_channels: int,
    output_classes: int,
) -> VariantConfig:
    example_input_shape = build_example_input_shape(
        model_name=model_name,
        template=template,
        input_channels=input_channels,
    )
    return template.model_copy(
        update={
            "target_input_channels": input_channels,
            "target_output_classes": output_classes,
            "example_input_shape": example_input_shape,
        }
    )


def build_example_input_shape(
    model_name: str, template: VariantConfig, input_channels: int
) -> list[int]:
    if is_recommender_model_name(model_name):
        batch_size = template.example_input_shape[0]
        return [batch_size]

    if is_text_model_name(model_name):
        batch_size, sequence_length = template.example_input_shape
        return [batch_size, sequence_length]

    batch_size, _channels, height, width = template.example_input_shape
    return [batch_size, input_channels, height, width]
