from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_gnn_model_cli_runs_end_to_end(tmp_path: Path) -> None:
    config_path = tmp_path / "cli_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "cli_smoke",
                "data": {
                    "kind": "fake",
                    "dataset_size": 8,
                    "random_seed": 4,
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
    output_dir = tmp_path / "cli_output"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gnn_model",
            "--config",
            str(config_path),
            "--output_dir",
            str(output_dir),
            "--device",
            "cpu",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result_files = list((output_dir / "cli_smoke" / "results").glob("*.json"))
    assert len(result_files) == 1
    payload = json.loads(result_files[0].read_text(encoding="utf-8"))
    assert payload["experiment_name"] == "cli_smoke"

