from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

from gnn_archs.config import ResolvedVariantSpec, is_text_model_name

if TYPE_CHECKING:
    from gnn_archs.config import ArchConfig, BaseModelGroup, VariantConfig


def expand_arch_config(config: ArchConfig) -> list[ResolvedVariantSpec]:
    variants: list[ResolvedVariantSpec] = []
    for group in config.base_model_groups:
        variants.extend(expand_group_variants(group))
    return variants


def expand_group_variants(group: BaseModelGroup) -> list[ResolvedVariantSpec]:
    explicit_variants = build_explicit_variants(group)
    grid_variants = build_grid_variants(group)
    fc_variants = build_fc_variants(group, grid_variants)
    variants = explicit_variants + grid_variants + fc_variants
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


def build_grid_variant_name(
    base_model_name: str,
    input_channels: int,
    output_classes: int,
    mutation_set_name: str,
) -> str:
    return (
        f"{base_model_name}_ic{input_channels}_oc{output_classes}_"
        f"{mutation_set_name}"
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
    if is_text_model_name(model_name):
        batch_size, sequence_length = template.example_input_shape
        return [batch_size, sequence_length]

    batch_size, _channels, height, width = template.example_input_shape
    return [batch_size, input_channels, height, width]
