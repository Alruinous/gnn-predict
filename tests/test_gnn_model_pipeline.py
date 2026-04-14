from __future__ import annotations

import json
from pathlib import Path

import yaml

from gnn_model.runner import run_experiment


def write_fake_config(path: Path, experiment_name: str) -> Path:
    config_path = path / f"{experiment_name}.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": experiment_name,
                "data": {
                    "kind": "fake",
                    "dataset_size": 8,
                    "random_seed": 9,
                    "val_ratio": 0.25,
                    "test_ratio": 0.25,
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
    return config_path


def test_run_experiment_trains_and_writes_result(tmp_path: Path) -> None:
    config_path = write_fake_config(tmp_path, "pipeline_smoke")

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
    assert payload["evaluation"]["metrics"]["mae"] >= 0
    assert payload["dataset"]["train_count"] > 0

