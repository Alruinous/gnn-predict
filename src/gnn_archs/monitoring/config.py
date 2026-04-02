from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import Field, field_validator, model_validator

from gnn_archs.config import StrictModel


class MonitorDefaults(StrictModel):
    prometheus_url: str
    namespace: str
    cpu_rate_window: str = "2m"
    query_step_seconds: int = 1

    @field_validator("prometheus_url", "namespace", "cpu_rate_window")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("monitor config strings must not be empty")
        return normalized_value

    @field_validator("query_step_seconds")
    @classmethod
    def validate_positive_step(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("query_step_seconds must be positive")
        return value


class MonitorTarget(StrictModel):
    enabled: bool = True
    result_json: str
    node_name: str
    pod_name: str
    output_csv: str | None = None

    @field_validator("result_json", "node_name", "pod_name")
    @classmethod
    def validate_required_strings(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("monitor target strings must not be empty")
        return normalized_value

    @field_validator("output_csv")
    @classmethod
    def validate_optional_output_csv(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("output_csv must not be empty when provided")
        return normalized_value


class MonitorConfig(StrictModel):
    defaults: MonitorDefaults
    targets: dict[str, MonitorTarget] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_targets(self) -> MonitorConfig:
        if not self.targets:
            raise ValueError("targets must not be empty")
        return self


@dataclass(frozen=True)
class ResolvedMonitorTarget:
    name: str
    node_name: str
    pod_name: str
    result_json: Path
    output_csv: Path


@dataclass(frozen=True)
class ResolvedMonitorSettings:
    config_path: Path
    prometheus_url: str
    namespace: str
    cpu_rate_window: str
    query_step_seconds: int
    targets: tuple[ResolvedMonitorTarget, ...]


def load_monitor_settings(config_path: Path) -> ResolvedMonitorSettings:
    resolved_config_path = Path(config_path).resolve()
    with resolved_config_path.open(encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    config = MonitorConfig.model_validate(raw_config)
    enabled_targets: list[ResolvedMonitorTarget] = []
    config_dir = resolved_config_path.parent

    for target_name, target in config.targets.items():
        normalized_name = target_name.strip()
        if not normalized_name:
            raise ValueError("target names must not be empty")
        if not target.enabled:
            continue

        result_json = _resolve_path(config_dir, target.result_json)
        output_csv = (
            _resolve_path(config_dir, target.output_csv)
            if target.output_csv is not None
            else result_json.with_name(f"{result_json.stem}_monitor.csv")
        )
        enabled_targets.append(
            ResolvedMonitorTarget(
                name=normalized_name,
                node_name=target.node_name,
                pod_name=target.pod_name,
                result_json=result_json,
                output_csv=output_csv,
            )
        )

    if not enabled_targets:
        raise ValueError("monitor config must enable at least one target")

    return ResolvedMonitorSettings(
        config_path=resolved_config_path,
        prometheus_url=config.defaults.prometheus_url,
        namespace=config.defaults.namespace,
        cpu_rate_window=config.defaults.cpu_rate_window,
        query_step_seconds=config.defaults.query_step_seconds,
        targets=tuple(enabled_targets),
    )


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()
