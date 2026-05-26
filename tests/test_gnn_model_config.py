from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from gnn_model.config import SplitDataConfig, load_experiment_config
from gnn_model.data.dataset import SPLIT_FILE_NAMES
from gnn_model_test_utils import TARGET_NAMES


def test_gnn_model_config_loads_split_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "gnn_model.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "config_defaults",
                "data": {
                    "kind": "split",
                    "data_dir": "data/scaled",
                    "target_names": list(TARGET_NAMES),
                    "scaler_dir": "data/scalers",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.experiment_name == "config_defaults"
    assert isinstance(config.data, SplitDataConfig)
    assert config.data.data_dir == "data/scaled"
    assert config.data.target_names == list(TARGET_NAMES)
    assert config.data.split_files == SPLIT_FILE_NAMES
    assert config.model.hidden_dim == 64
    assert config.model.readout_mode == "mean"
    assert config.model.target_readout_modes == {}
    assert config.model.structural_context_mode == "none"
    assert config.training.num_epochs == 1


@pytest.mark.parametrize(
    "data_config",
    [
        {"kind": "split", "data_dir": "", "target_names": list(TARGET_NAMES)},
        {"kind": "split", "data_dir": "data/scaled", "target_names": []},
        {
            "kind": "split",
            "data_dir": "data/scaled",
            "target_names": list(TARGET_NAMES),
            "scaler_dir": "",
        },
    ],
)
def test_gnn_model_config_rejects_invalid_split_data(
    tmp_path: Path,
    data_config: dict[str, object],
) -> None:
    config_path = tmp_path / "gnn_model.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "bad_config",
                "data": data_config,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_experiment_config(config_path)


def test_gnn_model_config_rejects_fake_data(tmp_path: Path) -> None:
    config_path = tmp_path / "gnn_model.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "bad_config",
                "data": {"kind": "fake"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="fake"):
        load_experiment_config(config_path)
