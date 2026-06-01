from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from gnn_model.data.extract import TARGET_FIELDS
from gnn_model.data.constants import (
    EDGE_FEATURE_NAMES,
    GRAPH_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    OP_TYPE_NAMES,
)
from gnn_model.runner import run_experiment
from gnn_model_test_utils import (
    TARGET_NAMES,
    build_synthetic_graph,
    write_split_config,
    write_split_dataset,
    write_target_scalers,
)


def test_run_experiment_trains_and_writes_result(tmp_path: Path) -> None:
    data_dir = write_split_dataset(tmp_path / "scaled")
    scaler_dir = write_target_scalers(tmp_path / "scalers", TARGET_NAMES)
    config_path = write_split_config(
        tmp_path / "split_config.yaml",
        experiment_name="pipeline_smoke",
        data_dir=data_dir,
        scaler_dir=scaler_dir,
    )

    result, result_path = run_experiment(
        config_path=config_path,
        output_dir=tmp_path / "output",
        requested_device="cpu",
    )

    assert result.training.best_epoch == 1
    assert Path(result.training.checkpoint_path).exists()
    assert result_path.exists()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0.0"
    metrics = payload["evaluation"]["metrics"]
    assert metrics["mae"] >= 0
    assert metrics["wape"] >= 0
    assert metrics["max_abs_error"] >= 0
    assert metrics["duration_sec_avg_wape"] >= 0
    assert metrics["duration_sec_avg_max_abs_error"] >= 0
    assert metrics["original_scale_mae"] >= 0
    assert metrics["original_scale_wape"] >= 0
    assert metrics["original_scale_max_abs_error"] >= 0
    assert metrics["original_scale_duration_sec_avg_wape"] >= 0
    assert payload["dataset"]["train_count"] > 0
    assert payload["metadata"]["data_dir"] == str(data_dir.resolve())
    assert payload["metadata"]["scaler_dir"] == str(scaler_dir.resolve())


def test_run_experiment_trains_with_prepared_dataset(tmp_path: Path) -> None:
    manifest_path = write_prepared_manifest(tmp_path)
    config_path = tmp_path / "prepared_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "prepared_pipeline_smoke",
                "data": {
                    "kind": "prepared",
                    "manifest_path": str(manifest_path),
                },
                "model": {
                    "hidden_dim": 32,
                    "num_layers": 2,
                    "num_heads": 4,
                    "dropout_rate": 0.1,
                },
                "training": {
                    "batch_size": 2,
                    "num_epochs": 1,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                },
            }
        ),
        encoding="utf-8",
    )

    result, result_path = run_experiment(
        config_path=config_path,
        output_dir=tmp_path / "output",
        requested_device="cpu",
    )

    assert result.target_names == list(TARGET_FIELDS)
    assert result.dataset.train_count > 0
    assert result_path.exists()


def write_prepared_manifest(path: Path) -> Path:
    prepared_dir = path / "prepared"
    prepared_dir.mkdir()
    split_data = {
        "train": [
            build_synthetic_graph(0, target_dim=len(TARGET_FIELDS)),
            build_synthetic_graph(1, target_dim=len(TARGET_FIELDS)),
        ],
        "val": [build_synthetic_graph(2, target_dim=len(TARGET_FIELDS))],
        "test": [build_synthetic_graph(3, target_dim=len(TARGET_FIELDS))],
    }
    split_files = {
        "train": "train.pt",
        "val": "val.pt",
        "test": "test.pt",
    }
    for split_name, file_name in split_files.items():
        torch.save(split_data[split_name], prepared_dir / file_name)
    manifest_path = prepared_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "4.0.0",
                "target_names": list(TARGET_FIELDS),
                "node_feature_names": list(NODE_FEATURE_NAMES),
                "op_type_names": list(OP_TYPE_NAMES),
                "edge_feature_names": list(EDGE_FEATURE_NAMES),
                "graph_feature_names": list(GRAPH_FEATURE_NAMES),
                "split_files": split_files,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path
