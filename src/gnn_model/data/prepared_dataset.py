from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gnn_model.config import PreparedDataConfig
from gnn_model.data.dataset import (
    GraphDatasetBundle,
    load_graph_split,
)


def load_prepared_graph_datasets(config: PreparedDataConfig) -> GraphDatasetBundle:
    manifest_path = Path(config.manifest_path)
    manifest = load_manifest(manifest_path)
    target_names = require_string_tuple(manifest, "target_names")
    split_files = require_split_files(manifest)
    train_data = load_graph_split(
        manifest_path.parent / split_files["train"],
        target_dim=len(target_names),
    )
    val_data = load_graph_split(
        manifest_path.parent / split_files["val"],
        target_dim=len(target_names),
    )
    test_data = load_graph_split(
        manifest_path.parent / split_files["test"],
        target_dim=len(target_names),
    )
    return GraphDatasetBundle(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        target_names=target_names,
    )


def load_manifest(path: Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"prepared manifest does not exist: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"prepared manifest must be a JSON object: {manifest_path}")
    return payload


def require_string_tuple(manifest: dict[str, Any], field_name: str) -> tuple[str, ...]:
    value = manifest.get(field_name)
    if not isinstance(value, list) or not value:
        raise ValueError(f"prepared manifest {field_name} must be a non-empty list")
    parsed = tuple(item for item in value if isinstance(item, str) and item.strip())
    if len(parsed) != len(value):
        raise ValueError(f"prepared manifest {field_name} must contain only strings")
    return parsed


def require_split_files(manifest: dict[str, Any]) -> dict[str, str]:
    value = manifest.get("split_files")
    if not isinstance(value, dict):
        raise ValueError("prepared manifest split_files must be a mapping")
    required_splits = {"train", "val", "test"}
    missing_splits = sorted(required_splits - set(value))
    if missing_splits:
        joined = ", ".join(missing_splits)
        raise ValueError(f"prepared manifest split_files missing: {joined}")
    split_files = {
        split_name: value[split_name]
        for split_name in required_splits
        if isinstance(value[split_name], str) and value[split_name].strip()
    }
    if set(split_files) != required_splits:
        raise ValueError("prepared manifest split file values must be strings")
    return split_files
