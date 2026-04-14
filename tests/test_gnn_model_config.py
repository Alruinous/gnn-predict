from __future__ import annotations

from pathlib import Path

import yaml

from gnn_model.config import load_experiment_config


def test_gnn_model_config_loads_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "gnn_model.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "config_defaults",
                "data": {"kind": "fake"},
            }
        ),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.experiment_name == "config_defaults"
    assert config.data.kind == "fake"
    assert config.data.dataset_size == 12
    assert config.model.hidden_dim == 64
    assert config.training.num_epochs == 1

