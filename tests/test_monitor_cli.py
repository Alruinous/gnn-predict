from __future__ import annotations

import logging
import runpy
import sys
from pathlib import Path

import pytest

import monitor as root_monitor
from gnn_archs.monitoring import cli as monitoring_cli


def test_monitoring_cli_main_filters_models_and_prints_written_paths(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}
    fake_settings = object()
    written_paths = [Path("/tmp/densenet121.csv"), Path("/tmp/bert-large-cased.csv")]

    def fake_load_monitor_settings(
        config_path: Path,
        *,
        target_names: tuple[str, ...] | None = None,
    ) -> object:
        recorded["config_path"] = config_path
        recorded["target_names"] = target_names
        return fake_settings

    def fake_run_monitoring(settings: object, *, logger: object) -> list[Path]:
        assert settings is fake_settings
        assert logger is monitoring_cli.logger
        return written_paths

    monkeypatch.setattr(
        monitoring_cli,
        "load_monitor_settings",
        fake_load_monitor_settings,
    )
    monkeypatch.setattr(
        monitoring_cli,
        "run_monitoring",
        fake_run_monitoring,
    )

    exit_code = monitoring_cli.main(
        [
            "--config",
            "config/monitor/monitor.yaml",
            "--models",
            "densenet121, bert-large-cased, densenet121,",
        ]
    )

    assert exit_code == 0
    assert recorded["config_path"] == Path("config/monitor/monitor.yaml")
    assert recorded["target_names"] == ("densenet121", "bert-large-cased")
    assert capsys.readouterr().out.splitlines() == [str(path) for path in written_paths]


def test_monitoring_cli_empty_models_argument_uses_all_enabled_targets(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}
    fake_settings = object()

    def fake_load_monitor_settings(
        config_path: Path,
        *,
        target_names: tuple[str, ...] | None = None,
    ) -> object:
        recorded["config_path"] = config_path
        recorded["target_names"] = target_names
        return fake_settings

    def fake_run_monitoring(settings: object, *, logger: object) -> list[Path]:
        assert settings is fake_settings
        assert logger is monitoring_cli.logger
        return [Path("/tmp/all-enabled.csv")]

    monkeypatch.setattr(
        monitoring_cli,
        "load_monitor_settings",
        fake_load_monitor_settings,
    )
    monkeypatch.setattr(
        monitoring_cli,
        "run_monitoring",
        fake_run_monitoring,
    )
    caplog.set_level(logging.INFO, logger=monitoring_cli.__name__)

    exit_code = monitoring_cli.main(
        [
            "--config",
            "config/monitor/monitor.yaml",
            "--models",
            ",,,",
        ]
    )

    assert exit_code == 0
    assert recorded["config_path"] == Path("config/monitor/monitor.yaml")
    assert recorded["target_names"] is None
    assert (
        "Empty --models value provided; reading all enabled targets from "
        "config/monitor/monitor.yaml."
    ) in caplog.text
    assert capsys.readouterr().out.splitlines() == [str(Path("/tmp/all-enabled.csv"))]


def test_monitor_py_delegates_to_package_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    def fake_main(argv: list[str] | None = None) -> int:
        recorded["argv"] = argv
        return 7

    monkeypatch.setattr(monitoring_cli, "main", fake_main)

    exit_code = root_monitor.main(["--config", "cfg.yaml", "--models", "resnet50"])

    assert exit_code == 7
    assert recorded["argv"] == ["--config", "cfg.yaml", "--models", "resnet50"]


def test_python_m_monitoring_entry_uses_cli_main(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}
    fake_settings = object()

    def fake_load_monitor_settings(
        config_path: Path,
        *,
        target_names: tuple[str, ...] | None = None,
    ) -> object:
        recorded["config_path"] = config_path
        recorded["target_names"] = target_names
        return fake_settings

    def fake_run_monitoring(settings: object, *, logger: object) -> list[Path]:
        assert settings is fake_settings
        assert logger is monitoring_cli.logger
        return [Path("/tmp/out.csv")]

    monkeypatch.setattr(
        monitoring_cli,
        "load_monitor_settings",
        fake_load_monitor_settings,
    )
    monkeypatch.setattr(
        monitoring_cli,
        "run_monitoring",
        fake_run_monitoring,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "python",
            "--config",
            "config/monitor/monitor.yaml",
            "--models",
            "densenet121,bert-large-cased",
        ],
    )

    with pytest.raises(SystemExit, match="0"):
        runpy.run_module("gnn_archs.monitoring", run_name="__main__")

    assert recorded["config_path"] == Path("config/monitor/monitor.yaml")
    assert recorded["target_names"] == ("densenet121", "bert-large-cased")
    assert capsys.readouterr().out.splitlines() == [str(Path("/tmp/out.csv"))]
