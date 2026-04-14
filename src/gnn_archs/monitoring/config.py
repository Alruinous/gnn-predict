from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import Field, field_validator, model_validator

from gnn_archs.config import StrictModel

PROMETHEUS_DURATION_PATTERN = re.compile(r"(\d+)(ms|s|m|h|d|w|y)")
PROMETHEUS_DURATION_SECONDS = {
    "ms": 0.001,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
    "w": 604800.0,
    "y": 31536000.0,
}


class MonitorDefaults(StrictModel):
    prometheus_url: str
    namespace: str
    cpu_rate_window: str = "3s"
    query_step_seconds: int = 1

    @field_validator("prometheus_url", "namespace", "cpu_rate_window")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("monitor config strings must not be empty")
        return normalized_value

    @field_validator("cpu_rate_window")
    @classmethod
    def validate_cpu_rate_window(cls, value: str) -> str:
        parse_prometheus_duration_seconds(value)
        return value

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
    gpu_id: str
    output_csv: str | None = None

    @field_validator("result_json", "node_name", "pod_name", "gpu_id")
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
    gpu_id: str
    result_json: Path
    output_csv: Path


@dataclass(frozen=True)
class ResolvedMonitorSettings:
    config_path: Path
    prometheus_url: str
    namespace: str
    cpu_rate_window: str
    cpu_rate_window_seconds: float
    query_step_seconds: int
    targets: tuple[ResolvedMonitorTarget, ...]


def load_monitor_settings(
    config_path: Path,
    *,
    target_names: Sequence[str] | None = None,
) -> ResolvedMonitorSettings:
    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    config = MonitorConfig.model_validate(raw_config)
    normalized_targets: dict[str, MonitorTarget] = {}
    enabled_target_names: list[str] = []

    for target_name, target in config.targets.items():
        normalized_name = _normalize_target_name(target_name)
        if normalized_name in normalized_targets:
            raise ValueError(
                "monitor config target names must be unique after trimming: "
                f"{normalized_name}"
            )
        normalized_targets[normalized_name] = target
        if target.enabled:
            enabled_target_names.append(normalized_name)

    requested_target_names = _normalize_requested_target_names(target_names)
    selected_target_names = _resolve_selected_target_names(
        requested_target_names,
        normalized_targets=normalized_targets,
        enabled_target_names=enabled_target_names,
    )
    resolved_targets = [
        _resolve_target(name, normalized_targets[name])
        for name in selected_target_names
    ]

    if not resolved_targets:
        raise ValueError("monitor config must enable at least one target")

    return ResolvedMonitorSettings(
        config_path=config_path,
        prometheus_url=config.defaults.prometheus_url,
        namespace=config.defaults.namespace,
        cpu_rate_window=config.defaults.cpu_rate_window,
        cpu_rate_window_seconds=parse_prometheus_duration_seconds(
            config.defaults.cpu_rate_window
        ),
        query_step_seconds=config.defaults.query_step_seconds,
        targets=tuple(resolved_targets),
    )


def _resolve_path(raw_path: str) -> Path:
    return Path(raw_path)


def _normalize_target_name(target_name: str) -> str:
    normalized_name = target_name.strip()
    if not normalized_name:
        raise ValueError("target names must not be empty")
    return normalized_name


def _normalize_requested_target_names(
    target_names: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if target_names is None:
        return None

    return tuple(_normalize_target_name(target_name) for target_name in target_names)


def _resolve_selected_target_names(
    requested_target_names: tuple[str, ...] | None,
    *,
    normalized_targets: dict[str, MonitorTarget],
    enabled_target_names: list[str],
) -> tuple[str, ...]:
    if requested_target_names is None:
        return tuple(enabled_target_names)

    enabled_target_name_set = set(enabled_target_names)
    all_target_name_set = set(normalized_targets)
    unknown_target_names = [
        name for name in requested_target_names if name not in all_target_name_set
    ]
    disabled_target_names = [
        name
        for name in requested_target_names
        if name in all_target_name_set and name not in enabled_target_name_set
    ]
    if unknown_target_names or disabled_target_names:
        problems: list[str] = []
        if unknown_target_names:
            problems.append(f"unknown: {', '.join(unknown_target_names)}")
        if disabled_target_names:
            problems.append(f"disabled: {', '.join(disabled_target_names)}")
        enabled_targets_text = ", ".join(enabled_target_names) or "<none>"
        raise ValueError(
            "requested monitor targets are unavailable "
            f"({'; '.join(problems)}); enabled targets: {enabled_targets_text}"
        )
    return requested_target_names


def _resolve_target(
    target_name: str,
    target: MonitorTarget,
) -> ResolvedMonitorTarget:
    result_json = _resolve_path(target.result_json)
    output_csv = (
        _resolve_path(target.output_csv)
        if target.output_csv is not None
        else result_json.with_name(f"{result_json.stem}_monitor.csv")
    )
    return ResolvedMonitorTarget(
        name=target_name,
        node_name=target.node_name,
        pod_name=target.pod_name,
        gpu_id=target.gpu_id,
        result_json=result_json,
        output_csv=output_csv,
    )


def parse_prometheus_duration_seconds(value: str) -> float:
    total_seconds = 0.0
    position = 0
    for match in PROMETHEUS_DURATION_PATTERN.finditer(value):
        if match.start() != position:
            raise ValueError(f"invalid Prometheus duration: {value}")
        position = match.end()
        total_seconds += int(match.group(1)) * PROMETHEUS_DURATION_SECONDS[
            match.group(2)
        ]

    if position != len(value) or total_seconds <= 0:
        raise ValueError(f"invalid Prometheus duration: {value}")
    return total_seconds
