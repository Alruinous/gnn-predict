from __future__ import annotations

from copy import deepcopy
from typing import Any

ALLOWED_GROUP_KEYS = {
    "base_model",
    "single_variant_define",
    "combinatorial_variant_grid",
    "fc_mutation_sets",
}
GRID_TEMPLATE_KEYS = {
    "run_training",
    "run_inference",
    "run_workload",
    "pre_inference_cooldown_seconds",
    "inference_measurement_min_seconds",
    "export_onnx",
    "onnx_export_mode",
    "batch_size",
    "training_measurement_min_seconds",
    "use_fake_imagenet",
    "use_fake_text_dataset",
    "use_real_text_dataset",
}
ALLOWED_GRID_KEYS = {
    "input_channels",
    "output_classes",
    "base_variant_config_template",
    "mutation_sets",
    *GRID_TEMPLATE_KEYS,
}


def migrate_arch_config_dict(raw_config: dict[str, Any]) -> dict[str, Any]:
    base_model_groups = raw_config.get("base_model_groups")
    if not isinstance(base_model_groups, list) or not base_model_groups:
        raise ValueError("base_model_groups must be a non-empty list")

    return {
        "base_model_groups": [
            migrate_base_model_group(group) for group in base_model_groups
        ]
    }


def migrate_base_model_group(raw_group: dict[str, Any]) -> dict[str, Any]:
    unknown_keys = set(raw_group) - ALLOWED_GROUP_KEYS
    if unknown_keys:
        raise ValueError(f"unsupported base_model_group keys: {sorted(unknown_keys)}")

    migrated_group: dict[str, Any] = {"base_model": deepcopy(raw_group["base_model"])}

    if "single_variant_define" in raw_group:
        migrated_group["single_variant_define"] = [
            migrate_single_variant_definition(variant)
            for variant in raw_group["single_variant_define"]
        ]

    if "combinatorial_variant_grid" in raw_group:
        migrated_group["combinatorial_variant_grid"] = migrate_combinatorial_grid(
            raw_group["combinatorial_variant_grid"]
        )

    if "fc_mutation_sets" in raw_group:
        migrated_group["fc_mutation_sets"] = [
            migrate_mutation_set(mutation_set, mutation_list_key="fc_mutations")
            for mutation_set in raw_group["fc_mutation_sets"]
        ]

    return migrated_group


def migrate_single_variant_definition(raw_variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": raw_variant["name"],
        "variant_config": migrate_variant_config(raw_variant["variant_config"]),
        "mutations": [
            migrate_mutation(mutation) for mutation in raw_variant.get("mutations", [])
        ],
    }


def migrate_combinatorial_grid(raw_grid: dict[str, Any]) -> dict[str, Any]:
    unknown_keys = set(raw_grid) - ALLOWED_GRID_KEYS
    if unknown_keys:
        raise ValueError(
            f"unsupported combinatorial_variant_grid keys: {sorted(unknown_keys)}"
        )

    template = deepcopy(raw_grid.get("base_variant_config_template", {}))
    for key in GRID_TEMPLATE_KEYS:
        if key in raw_grid:
            if key in template and template[key] != raw_grid[key]:
                raise ValueError(
                    f"conflicting value for '{key}' "
                    "between combinatorial grid and template"
                )
            template[key] = raw_grid[key]

    migrated_grid = {
        "input_channels": deepcopy(raw_grid["input_channels"]),
        "output_classes": deepcopy(raw_grid["output_classes"]),
        "base_variant_config_template": migrate_variant_config(template),
        "mutation_sets": [
            migrate_mutation_set(mutation_set)
            for mutation_set in raw_grid.get("mutation_sets", [])
        ],
    }
    if not migrated_grid["mutation_sets"]:
        raise ValueError("combinatorial_variant_grid.mutation_sets must not be empty")
    return migrated_grid


def migrate_variant_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(raw_config)
    if "fake_dataset_size" in config:
        raise ValueError("unsupported variant_config key: fake_dataset_size")

    if "run_workload" in config and "run_inference" not in config:
        config["run_inference"] = config["run_workload"]

    config.pop("run_workload", None)
    config.pop("validate_model", None)

    if "run_training" not in config:
        config["run_training"] = False
    if "run_inference" not in config:
        config["run_inference"] = False
    if "export_onnx" not in config:
        config["export_onnx"] = False
    if "onnx_export_mode" not in config:
        config["onnx_export_mode"] = "full"

    return config


def migrate_mutation_set(
    raw_mutation_set: dict[str, Any], mutation_list_key: str = "mutations"
) -> dict[str, Any]:
    return {
        "name": raw_mutation_set["name"],
        "mutations": [
            migrate_mutation(mutation)
            for mutation in raw_mutation_set.get(mutation_list_key, [])
        ],
    }


def migrate_mutation(raw_mutation: dict[str, Any]) -> dict[str, Any]:
    mutation_type = raw_mutation.get("type")
    if not mutation_type:
        raise ValueError(f"mutation is missing type: {raw_mutation}")

    params = deepcopy(raw_mutation.get("params", {}))
    for key, value in raw_mutation.items():
        if key in {"type", "params"}:
            continue
        if key in params:
            raise ValueError(
                f"mutation '{mutation_type}' duplicates param '{key}' "
                "between top level and params"
            )
        params[key] = deepcopy(value)

    return {"type": mutation_type, "params": params}
