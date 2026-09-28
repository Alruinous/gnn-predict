from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from gnn_archs.result import ResultDocument

from .prometheus import PrometheusClient
from .queries import (
    GPU_METRIC_DEFINITIONS,
    IX_GPU_METRIC_DEFINITIONS,
    build_container_start_time_query,
    build_gpu_metrics_query,
    build_ix_gpu_metrics_query,
    build_node_cpu_total_query,
    build_node_memory_total_query,
    build_pod_cpu_query,
    build_pod_info_query,
    build_pod_memory_query,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from gnn_archs.result import TimeWindow, VariantResult

    from .config import ResolvedMonitorSettings, ResolvedMonitorTarget
    from .prometheus import PrometheusQueryAPI

CSV_COLUMNS = [
    "target_name",
    "result_json",
    "config_path",
    "variant_name",
    "base_model_name",
    "namespace",
    "node_name",
    "pod_name",
    "gpu_node",
    "gpu_id",
    "gpu_backend",
    "gpu_uuid",
    "phase",
    "started_at_ts",
    "ended_at_ts",
    "duration_sec",
    "deployment_duration_sec_avg",
    "phase_rounds",
    "sample_count",
    "cpu_sample_coverage",
    "memory_sample_coverage",
    "gpu_min_sample_coverage",
    "batch_size",
    "decode_output_length",
    "resolved_gpu_label",
    "resolved_device_label",
    "cpu_cores_avg",
    "cpu_cores_max",
    "cpu_cores_p95",
    "cpu_cores_pct_of_total_avg",
    "memory_gb_avg",
    "memory_gb_max",
    "memory_gb_p95",
    "memory_gb_pct_of_total_avg",
    "container_started_at_ts",
    "memory_baseline_gb",
    "memory_baseline_sample_count",
    "memory_delta_gb_avg",
    "memory_delta_gb_max",
    "memory_delta_gb_p95",
    "gpu_util_percent_avg",
    "gpu_util_percent_max",
    "gpu_util_percent_p95",
    "gpu_sm_active_percent_avg",
    "gpu_sm_active_percent_max",
    "gpu_sm_active_percent_p95",
    "gpu_sm_util_percent_avg",
    "gpu_sm_util_percent_max",
    "gpu_sm_util_percent_p95",
    "gpu_sm_occupancy_percent_avg",
    "gpu_sm_occupancy_percent_max",
    "gpu_sm_occupancy_percent_p95",
    "gpu_mem_used_mb_avg",
    "gpu_mem_used_mb_max",
    "gpu_mem_used_mb_p95",
    "gpu_mem_free_mb_avg",
    "gpu_mem_free_mb_max",
    "gpu_mem_free_mb_p95",
    "gpu_mem_copy_util_percent_avg",
    "gpu_mem_copy_util_percent_max",
    "gpu_mem_copy_util_percent_p95",
    "gpu_dram_active_percent_avg",
    "gpu_dram_active_percent_max",
    "gpu_dram_active_percent_p95",
    "gpu_pcie_tx_mb_per_sec_avg",
    "gpu_pcie_tx_mb_per_sec_max",
    "gpu_pcie_tx_mb_per_sec_p95",
    "gpu_pcie_rx_mb_per_sec_avg",
    "gpu_pcie_rx_mb_per_sec_max",
    "gpu_pcie_rx_mb_per_sec_p95",
    "gpu_power_watts_avg",
    "gpu_power_watts_max",
    "gpu_power_watts_p95",
    "gpu_temp_celsius_avg",
    "gpu_temp_celsius_max",
    "gpu_temp_celsius_p95",
]
MIN_REQUIRED_PHASE_SAMPLES = 1


class MissingMonitorDataError(ValueError):
    """A required metric was absent, rather than malformed or ambiguous."""


def require_int_value(value: object, label: str) -> int:
    assert isinstance(value, int | float | str) and not isinstance(value, bool), label
    if isinstance(value, float):
        assert value.is_integer(), label
    if isinstance(value, str):
        parsed = float(value)
        assert parsed.is_integer(), label
        return int(parsed)
    return int(value)


@dataclass(frozen=True)
class MonitorPhaseRecord:
    target_name: str
    result_json: Path
    config_path: str
    variant_name: str
    base_model_name: str
    namespace: str
    node_name: str
    pod_name: str
    gpu_node: str
    gpu_id: str
    phase: str
    started_at_ts: float
    ended_at_ts: float
    duration_sec: float
    deployment_duration_sec_avg: float
    phase_rounds: int
    batch_size: int
    decode_output_length: int
    gpu_backend: str = "dcgm"
    gpu_uuid: str | None = None


@dataclass(frozen=True)
class MemoryBaseline:
    container_started_at_ts: float | None
    memory_baseline_gb: float | None
    sample_count: int


def extract_phase_records(
    target_name: str,
    result_json: Path,
    *,
    namespace: str,
    node_name: str,
    pod_name: str,
    gpu_id: str,
) -> tuple[ResultDocument, list[MonitorPhaseRecord]]:
    document = load_result_document(result_json)
    records: list[MonitorPhaseRecord] = []

    for variant in document.variants:
        deployment_duration_sec_avg = _deployment_duration_sec(variant)
        if variant.training is not None:
            records.append(
                _build_phase_record(
                    target_name=target_name,
                    result_json=result_json,
                    document=document,
                    variant=variant,
                    namespace=namespace,
                    node_name=node_name,
                    pod_name=pod_name,
                    phase="training",
                    timings=variant.training.timings,
                    deployment_duration_sec_avg=deployment_duration_sec_avg,
                    gpu_id=gpu_id,
                    batch_size=require_int_value(
                        variant.training.hyperparameters["batch_size"],
                        "training batch_size must be int",
                    ),
                    decode_output_length=0,
                )
            )
        if variant.inference is not None:
            records.append(
                _build_phase_record(
                    target_name=target_name,
                    result_json=result_json,
                    document=document,
                    variant=variant,
                    namespace=namespace,
                    node_name=node_name,
                    pod_name=pod_name,
                    phase="inference",
                    timings=variant.inference.timings,
                    deployment_duration_sec_avg=deployment_duration_sec_avg,
                    gpu_id=gpu_id,
                    batch_size=require_int_value(
                        variant.inference.metrics["batch_size"],
                        "inference batch_size must be int",
                    ),
                    decode_output_length=0,
                )
            )
        if variant.prefill is not None:
            records.append(
                _build_phase_record(
                    target_name=target_name,
                    result_json=result_json,
                    document=document,
                    variant=variant,
                    namespace=namespace,
                    node_name=node_name,
                    pod_name=pod_name,
                    phase="prefill",
                    timings=variant.prefill.timings,
                    deployment_duration_sec_avg=deployment_duration_sec_avg,
                    gpu_id=gpu_id,
                    batch_size=require_int_value(
                        variant.prefill.metrics["batch_size"],
                        "prefill batch_size must be int",
                    ),
                    decode_output_length=0,
                )
            )
        if variant.decode is not None:
            records.append(
                _build_phase_record(
                    target_name=target_name,
                    result_json=result_json,
                    document=document,
                    variant=variant,
                    namespace=namespace,
                    node_name=node_name,
                    pod_name=pod_name,
                    phase="decode",
                    timings=variant.decode.timings,
                    deployment_duration_sec_avg=deployment_duration_sec_avg,
                    gpu_id=gpu_id,
                    batch_size=require_int_value(
                        variant.decode.metrics["batch_size"],
                        "decode batch_size must be int",
                    ),
                    decode_output_length=require_int_value(
                        variant.decode.metrics["decode_max_output_length"],
                        "decode output length must be int",
                    ),
                )
            )

    if not records:
        raise ValueError(
            f"no training, inference, prefill, or decode phases found in {result_json}"
        )

    return document, records


def load_result_document(result_json: Path) -> ResultDocument:
    resolved_result_json = Path(result_json)
    if not resolved_result_json.exists():
        raise FileNotFoundError(
            f"result document does not exist: {resolved_result_json}"
        )
    return ResultDocument.model_validate_json(
        resolved_result_json.read_text(encoding="utf-8")
    )


def monitor_target(
    settings: ResolvedMonitorSettings,
    target: ResolvedMonitorTarget,
    client: PrometheusQueryAPI,
    logger: logging.Logger,
) -> pd.DataFrame:
    _, phase_records = extract_phase_records(
        target.name,
        target.result_json,
        namespace=settings.namespace,
        node_name=target.node_name,
        pod_name=target.pod_name,
        gpu_id=target.gpu_id,
    )
    phase_records = [
        replace(record, gpu_backend=target.gpu_backend, gpu_uuid=target.gpu_uuid)
        for record in phase_records
    ]

    reference_timestamp, node_total_cpu, node_total_memory_gb = _resolve_node_context(
        client, target, settings.namespace, phase_records
    )
    memory_baseline = _query_memory_baseline(
        client,
        target.pod_name,
        settings.namespace,
        reference_timestamp,
        settings.memory_baseline_window_seconds,
        settings.query_step_seconds,
        logger,
    )

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for record in phase_records:
        try:
            row = _monitor_phase_record(
                phase_record=record,
                client=client,
                cpu_rate_window=settings.cpu_rate_window,
                cpu_rate_window_seconds=settings.cpu_rate_window_seconds,
                query_step_seconds=settings.query_step_seconds,
                min_phase_coverage_ratio=settings.min_phase_coverage_ratio,
                node_total_cpu=node_total_cpu,
                node_total_memory_gb=node_total_memory_gb,
                memory_baseline=memory_baseline,
                logger=logger,
            )
        except MissingMonitorDataError as exc:
            skipped.append(f"{record.variant_name}/{record.phase}")
            logger.warning(
                "skipping monitor record for %s/%s: %s",
                record.variant_name,
                record.phase,
                exc,
            )
            continue
        rows.append(row)

    if not rows:
        raise MissingMonitorDataError(
            f"no complete monitoring records for {target.name}; "
            f"skipped {len(skipped)} phase(s): {', '.join(skipped)}"
        )
    logger.info(
        "monitor target %s: collected %d/%d phases; skipped %d",
        target.name,
        len(rows),
        len(phase_records),
        len(skipped),
    )
    dataframe = pd.DataFrame(rows, columns=CSV_COLUMNS)
    target.output_csv.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(target.output_csv, index=False, encoding="utf-8")
    return dataframe


def run_monitoring(
    settings: ResolvedMonitorSettings,
    *,
    logger: logging.Logger,
    client_factory: Callable[[str], PrometheusQueryAPI] | None = None,
) -> list[Path]:
    resolved_client_factory = client_factory or (lambda url: PrometheusClient(url))
    client = resolved_client_factory(settings.prometheus_url)
    written_paths: list[Path] = []

    for target in settings.targets:
        monitor_target(settings, target, client, logger)
        written_paths.append(target.output_csv)

    return written_paths


def _build_phase_record(
    *,
    target_name: str,
    result_json: Path,
    document: ResultDocument,
    variant: VariantResult,
    namespace: str,
    node_name: str,
    pod_name: str,
    phase: str,
    timings: TimeWindow,
    deployment_duration_sec_avg: float,
    gpu_id: str,
    batch_size: int,
    decode_output_length: int,
) -> MonitorPhaseRecord:
    started_at_ts = _require_timestamp(
        timings.started_at_ts,
        phase=phase,
        variant_name=variant.name,
        field_name="started_at_ts",
    )
    ended_at_ts = _require_timestamp(
        timings.ended_at_ts,
        phase=phase,
        variant_name=variant.name,
        field_name="ended_at_ts",
    )
    if ended_at_ts < started_at_ts:
        raise ValueError(
            f"{phase} timing end must be >= start for variant {variant.name}: "
            f"{ended_at_ts} < {started_at_ts}"
        )
    if phase == "training":
        training = variant.training
        assert training is not None
        phase_rounds = training.metrics.get("total_steps")
    elif phase == "inference":
        inference = variant.inference
        assert inference is not None
        phase_rounds = inference.metrics.get("iterations")
    elif phase == "prefill":
        prefill = variant.prefill
        assert prefill is not None
        phase_rounds = prefill.metrics.get("iterations")
    else:
        assert phase == "decode", phase
        decode = variant.decode
        assert decode is not None
        phase_rounds = decode.metrics.get("iterations")
    assert isinstance(phase_rounds, int) and not isinstance(phase_rounds, bool), (
        f"{phase} phase_rounds must be int for variant {variant.name}"
    )
    assert phase_rounds > 0, (
        f"{phase} phase_rounds must be positive for variant {variant.name}"
    )

    return MonitorPhaseRecord(
        target_name=target_name,
        result_json=result_json,
        config_path=document.config_path,
        variant_name=variant.name,
        base_model_name=variant.base_model_name,
        namespace=namespace,
        node_name=node_name,
        pod_name=pod_name,
        gpu_node=document.gpu_node,
        gpu_id=gpu_id,
        phase=phase,
        started_at_ts=started_at_ts,
        ended_at_ts=ended_at_ts,
        duration_sec=ended_at_ts - started_at_ts,
        deployment_duration_sec_avg=deployment_duration_sec_avg,
        phase_rounds=phase_rounds,
        batch_size=batch_size,
        decode_output_length=decode_output_length,
    )


def _deployment_duration_sec(variant: VariantResult) -> float:
    model_build_timings = variant.timings.get("model_build")
    if model_build_timings is None:
        raise ValueError(f"model_build timings missing for variant {variant.name}")
    started_at_ts = _require_timestamp(
        model_build_timings.started_at_ts,
        phase="model_build",
        variant_name=variant.name,
        field_name="started_at_ts",
    )
    ended_at_ts = _require_timestamp(
        model_build_timings.ended_at_ts,
        phase="model_build",
        variant_name=variant.name,
        field_name="ended_at_ts",
    )
    if ended_at_ts < started_at_ts:
        raise ValueError(
            f"model_build timing end must be >= start for variant {variant.name}: "
            f"{ended_at_ts} < {started_at_ts}"
        )
    return ended_at_ts - started_at_ts


def _require_timestamp(
    value: float | None,
    *,
    phase: str,
    variant_name: str,
    field_name: str,
) -> float:
    if value is None:
        raise ValueError(
            f"{phase} timings missing {field_name} for variant {variant_name}"
        )
    return float(value)


def _resolve_node_context(
    client: PrometheusQueryAPI,
    target: ResolvedMonitorTarget,
    namespace: str,
    phase_records: list[MonitorPhaseRecord],
) -> tuple[float, float, float]:
    # A single failed scrape at the first phase boundary must not invalidate
    # later variants. Node capacities are stable, but still read near this run.
    candidate_timestamps = dict.fromkeys(
        [record.ended_at_ts for record in phase_records]
        + [record.started_at_ts for record in phase_records]
    )
    last_missing: MissingMonitorDataError | None = None
    for timestamp in candidate_timestamps:
        try:
            _validate_target_pod(client, target, namespace, timestamp)
            node_total_cpu = _query_scalar(
                client,
                build_node_cpu_total_query(target.node_name),
                timestamp,
                description="node CPU total",
            )
            node_total_memory_gb = _query_scalar(
                client,
                build_node_memory_total_query(target.node_name),
                timestamp,
                description="node memory total",
            ) / (1024**3)
        except MissingMonitorDataError as exc:
            last_missing = exc
            continue
        if node_total_cpu <= 0 or node_total_memory_gb <= 0:
            raise ValueError("node CPU and memory totals must be positive")
        return timestamp, node_total_cpu, node_total_memory_gb

    raise MissingMonitorDataError(
        f"no valid Pod and node capacity metrics for {target.name} during its "
        f"phases; last missing metric: {last_missing}"
    )


def _validate_target_pod(
    client: PrometheusQueryAPI,
    target: ResolvedMonitorTarget,
    namespace: str,
    timestamp: float,
) -> None:
    pod_info = client.instant_query(
        build_pod_info_query(target.pod_name, namespace),
        timestamp,
    )
    if len(pod_info) == 0:
        raise MissingMonitorDataError(
            f"no kube_pod_info series for {target.pod_name} in {namespace}"
        )
    if len(pod_info) != 1:
        raise ValueError(
            "expected exactly one kube_pod_info series for "
            f"{target.pod_name} in {namespace}, got {len(pod_info)}"
        )
    node_name = pod_info[0].get("metric", {}).get("node")
    if node_name != target.node_name:
        raise ValueError(
            f"pod {target.pod_name} resolved to node {node_name}, "
            f"expected {target.node_name}"
        )


def _monitor_phase_record(
    *,
    phase_record: MonitorPhaseRecord,
    client: PrometheusQueryAPI,
    cpu_rate_window: str,
    cpu_rate_window_seconds: float,
    query_step_seconds: int,
    min_phase_coverage_ratio: float,
    node_total_cpu: float,
    node_total_memory_gb: float,
    memory_baseline: MemoryBaseline,
    logger: logging.Logger,
) -> dict[str, Any]:
    cpu_query_start_ts = phase_record.started_at_ts + cpu_rate_window_seconds
    if cpu_query_start_ts >= phase_record.ended_at_ts:
        logger.warning(
            f"Phase {phase_record.variant_name}/{phase_record.phase} "
            + f"starts at {phase_record.started_at_ts:.2f} and "
            + f"ends at {phase_record.ended_at_ts:.2f}, "
            + f"which is too short for cpu_rate_window={cpu_rate_window} ",
        )
        cpu_query_start_ts = phase_record.ended_at_ts

    cpu_values = _extract_single_series_values(
        client.range_query(
            build_pod_cpu_query(
                phase_record.pod_name,
                phase_record.namespace,
                cpu_rate_window,
            ),
            cpu_query_start_ts,
            phase_record.ended_at_ts,
            query_step_seconds,
        ),
        description=f"CPU usage for {phase_record.variant_name}/{phase_record.phase}",
        scale_factor=1.0,
    )
    if len(cpu_values) == 0:
        raise MissingMonitorDataError(
            f"No CPU samples returned for {phase_record.variant_name}/"
            f"{phase_record.phase}; check the rate window and scrape interval"
        )
    cpu_coverage = _sample_coverage(
        len(cpu_values),
        cpu_query_start_ts,
        phase_record.ended_at_ts,
        query_step_seconds,
    )
    _require_coverage(cpu_coverage, min_phase_coverage_ratio, phase_record, "CPU usage")

    memory_values = _extract_single_series_values(
        client.range_query(
            build_pod_memory_query(phase_record.pod_name, phase_record.namespace),
            phase_record.started_at_ts,
            phase_record.ended_at_ts,
            query_step_seconds,
        ),
        description=(
            f"memory usage for {phase_record.variant_name}/{phase_record.phase}"
        ),
        scale_factor=1 / (1024**3),
    )
    if len(memory_values) == 0:
        raise MissingMonitorDataError(
            f"No memory samples returned for {phase_record.variant_name}/"
            f"{phase_record.phase}; check the Prometheus scrape configuration"
        )
    memory_coverage = _sample_coverage(
        len(memory_values),
        phase_record.started_at_ts,
        phase_record.ended_at_ts,
        query_step_seconds,
    )
    _require_coverage(
        memory_coverage, min_phase_coverage_ratio, phase_record, "memory usage"
    )

    gpu_definitions = (
        IX_GPU_METRIC_DEFINITIONS
        if phase_record.gpu_backend == "ix"
        else GPU_METRIC_DEFINITIONS
    )
    metric_names = tuple(definition.prometheus_name for definition in gpu_definitions)
    if phase_record.gpu_backend == "ix":
        gpu_query = build_ix_gpu_metrics_query(
            metric_names,
            phase_record.node_name,
            phase_record.gpu_id,
            phase_record.gpu_uuid,
        )
    else:
        gpu_query = build_gpu_metrics_query(
            metric_names,
            phase_record.pod_name,
            phase_record.namespace,
            phase_record.gpu_id,
        )
    gpu_series = client.range_query(
        gpu_query,
        phase_record.started_at_ts,
        phase_record.ended_at_ts,
        query_step_seconds,
    )
    gpu_metrics, resolved_gpu_label, resolved_device_label, gpu_sample_counts = (
        _extract_gpu_metrics(
            gpu_series,
            phase_record=phase_record,
        )
    )
    gpu_min_coverage = min(
        _sample_coverage(
            count,
            phase_record.started_at_ts,
            phase_record.ended_at_ts,
            query_step_seconds,
        )
        for count in gpu_sample_counts
    )
    _require_coverage(
        gpu_min_coverage, min_phase_coverage_ratio, phase_record, "required GPU metrics"
    )

    cpu_summary = _summarize_values(cpu_values, "cpu_cores")
    cpu_summary["cpu_cores_pct_of_total_avg"] = round(
        cpu_summary["cpu_cores_avg"] / node_total_cpu * 100,
        2,
    )

    memory_summary = _summarize_values(memory_values, "memory_gb")
    memory_summary["memory_gb_pct_of_total_avg"] = round(
        memory_summary["memory_gb_avg"] / node_total_memory_gb * 100,
        2,
    )
    memory_delta_summary = _summarize_memory_delta(memory_values, memory_baseline)

    sample_count = min([len(cpu_values), len(memory_values), *gpu_sample_counts])
    if sample_count < MIN_REQUIRED_PHASE_SAMPLES:
        raise MissingMonitorDataError(
            f"{phase_record.variant_name}/{phase_record.phase} has too few samples "
            f"(min={sample_count}, required={MIN_REQUIRED_PHASE_SAMPLES}); "
            "increase the experiment measurement window"
        )

    row = {
        "target_name": phase_record.target_name,
        "result_json": str(phase_record.result_json),
        "config_path": phase_record.config_path,
        "variant_name": phase_record.variant_name,
        "base_model_name": phase_record.base_model_name,
        "namespace": phase_record.namespace,
        "node_name": phase_record.node_name,
        "pod_name": phase_record.pod_name,
        "gpu_node": phase_record.gpu_node,
        "gpu_id": phase_record.gpu_id,
        "gpu_backend": phase_record.gpu_backend,
        "gpu_uuid": phase_record.gpu_uuid,
        "phase": phase_record.phase,
        "started_at_ts": phase_record.started_at_ts,
        "ended_at_ts": phase_record.ended_at_ts,
        "duration_sec": round(phase_record.duration_sec, 6),
        "deployment_duration_sec_avg": round(
            phase_record.deployment_duration_sec_avg,
            6,
        ),
        "phase_rounds": phase_record.phase_rounds,
        "sample_count": sample_count,
        "cpu_sample_coverage": round(cpu_coverage, 3),
        "memory_sample_coverage": round(memory_coverage, 3),
        "gpu_min_sample_coverage": round(gpu_min_coverage, 3),
        "resolved_gpu_label": resolved_gpu_label,
        "resolved_device_label": resolved_device_label,
        "batch_size": phase_record.batch_size,
        "decode_output_length": phase_record.decode_output_length,
        "container_started_at_ts": memory_baseline.container_started_at_ts,
        "memory_baseline_gb": memory_baseline.memory_baseline_gb,
        "memory_baseline_sample_count": memory_baseline.sample_count,
    }
    row.update(cpu_summary)
    row.update(memory_summary)
    row.update(memory_delta_summary)
    row.update(gpu_metrics)
    return row


def _sample_coverage(
    sample_count: int, start_ts: float, end_ts: float, step_seconds: int
) -> float:
    expected_count = math.floor((end_ts - start_ts) / step_seconds + 1e-9) + 1
    return min(sample_count / max(expected_count, 1), 1.0)


def _require_coverage(
    coverage: float,
    minimum: float,
    phase_record: MonitorPhaseRecord,
    description: str,
) -> None:
    if coverage < minimum:
        raise MissingMonitorDataError(
            f"{description} coverage for {phase_record.variant_name}/"
            f"{phase_record.phase} is {coverage:.1%}, below required {minimum:.1%}"
        )


def _query_memory_baseline(
    client: PrometheusQueryAPI,
    pod_name: str,
    namespace: str,
    reference_timestamp: float,
    memory_baseline_window_seconds: int,
    query_step_seconds: int,
    logger: logging.Logger,
) -> MemoryBaseline:
    container_started_at_ts = _query_optional_scalar(
        client,
        build_container_start_time_query(pod_name, namespace),
        reference_timestamp,
        description="container start time",
    )
    if container_started_at_ts is None:
        logger.warning(
            f"No container start time returned for {pod_name} in {namespace}; "
            "memory delta columns will be empty."
        )
        return MemoryBaseline(None, None, 0)

    baseline_values = _extract_optional_single_series_values(
        client.range_query(
            build_pod_memory_query(pod_name, namespace),
            container_started_at_ts,
            container_started_at_ts + memory_baseline_window_seconds,
            query_step_seconds,
        ),
        scale_factor=1 / (1024**3),
    )
    if not baseline_values:
        logger.warning(
            f"No memory baseline samples returned for {pod_name} in {namespace}; "
            "memory delta columns will be empty."
        )
        return MemoryBaseline(container_started_at_ts, None, 0)

    return MemoryBaseline(
        container_started_at_ts=container_started_at_ts,
        memory_baseline_gb=round(min(baseline_values), 3),
        sample_count=len(baseline_values),
    )


def _summarize_memory_delta(
    memory_values: list[float],
    memory_baseline: MemoryBaseline,
) -> dict[str, float | None]:
    if memory_baseline.memory_baseline_gb is None:
        return {
            "memory_delta_gb_avg": None,
            "memory_delta_gb_max": None,
            "memory_delta_gb_p95": None,
        }
    delta_values = [
        max(memory_value - memory_baseline.memory_baseline_gb, 0.0)
        for memory_value in memory_values
    ]
    summary = _summarize_values(delta_values, "memory_delta_gb")
    result: dict[str, float | None] = dict(summary)
    return result


def _extract_gpu_metrics(
    gpu_series: list[dict[str, Any]],
    *,
    phase_record: MonitorPhaseRecord,
) -> tuple[dict[str, Any], str, str, list[int]]:
    metrics_by_name: dict[str, list[dict[str, Any]]] = {}
    for series in gpu_series:
        metric_name = series.get("metric", {}).get("__name__")
        if metric_name is None:
            raise ValueError(
                f"GPU query returned a series without __name__ for "
                f"{phase_record.variant_name}/{phase_record.phase}"
            )
        metrics_by_name.setdefault(metric_name, []).append(series)

    resolved_gpu_label: str | None = None
    resolved_device_label: str | None = None
    resolved_uuid_label: str | None = None
    sample_counts: list[int] = []
    metric_summaries: dict[str, Any] = {}

    definitions = (
        IX_GPU_METRIC_DEFINITIONS
        if phase_record.gpu_backend == "ix"
        else GPU_METRIC_DEFINITIONS
    )
    for definition in definitions:
        series_candidates = metrics_by_name.get(definition.prometheus_name, [])
        if not series_candidates and not definition.required:
            continue
        if not series_candidates and definition.required:
            raise MissingMonitorDataError(
                "no GPU series for "
                f"{definition.prometheus_name} during "
                f"{phase_record.variant_name}/{phase_record.phase}"
            )
        if len(series_candidates) != 1:
            raise ValueError(
                "expected exactly one GPU series for "
                f"{definition.prometheus_name} during "
                f"{phase_record.variant_name}/{phase_record.phase}, "
                f"got {len(series_candidates)}"
            )

        series = series_candidates[0]
        metric_labels = series.get("metric", {})
        gpu_label = metric_labels.get("gpu")
        device_label = (
            metric_labels.get("name")
            if phase_record.gpu_backend == "ix"
            else metric_labels.get("device")
        )
        if not gpu_label or not device_label:
            raise ValueError(
                "GPU series must include gpu and device/name labels for "
                f"{definition.prometheus_name}"
            )
        if gpu_label != phase_record.gpu_id:
            raise ValueError(
                "GPU metrics resolved to an unexpected gpu label for "
                f"{phase_record.variant_name}/{phase_record.phase}: "
                f"{gpu_label} != {phase_record.gpu_id}"
            )
        if phase_record.gpu_backend == "ix":
            if metric_labels.get("node_name") != phase_record.node_name:
                raise ValueError(
                    f"IX GPU series is not on node {phase_record.node_name}"
                )
            if (
                phase_record.gpu_uuid is not None
                and metric_labels.get("uuid") != phase_record.gpu_uuid
            ):
                raise ValueError(
                    f"IX GPU series UUID does not match {phase_record.gpu_uuid}"
                )
            uuid_label = metric_labels.get("uuid")
            if not uuid_label:
                raise ValueError("IX GPU series must include a uuid label")
            if resolved_uuid_label is None:
                resolved_uuid_label = uuid_label
            elif uuid_label != resolved_uuid_label:
                raise ValueError("IX GPU metrics resolved to inconsistent UUIDs")
        if resolved_gpu_label is None:
            resolved_gpu_label = gpu_label
            resolved_device_label = device_label
        elif resolved_gpu_label != gpu_label or resolved_device_label != device_label:
            raise ValueError(
                "GPU metrics resolved to inconsistent labels for "
                f"{phase_record.variant_name}/{phase_record.phase}"
            )

        try:
            values = _extract_series_values(
                series,
                description=(
                    f"{definition.prometheus_name} for "
                    f"{phase_record.variant_name}/{phase_record.phase}"
                ),
                scale_factor=definition.scale_factor,
            )
        except MissingMonitorDataError:
            if not definition.required:
                continue
            raise
        if definition.required:
            sample_counts.append(len(values))
        metric_summaries.update(_summarize_values(values, definition.output_prefix))

    assert resolved_gpu_label is not None
    assert resolved_device_label is not None
    return metric_summaries, resolved_gpu_label, resolved_device_label, sample_counts


def _query_scalar(
    client: PrometheusQueryAPI,
    query: str,
    timestamp: float,
    *,
    description: str,
) -> float:
    result = client.instant_query(query, timestamp)
    if len(result) == 0:
        raise MissingMonitorDataError(f"no series for {description}")
    if len(result) != 1:
        raise ValueError(
            f"expected exactly one series for {description}, got {len(result)}"
        )
    raw_value = result[0].get("value")
    if not isinstance(raw_value, list) or len(raw_value) != 2:
        raise ValueError(f"invalid scalar payload for {description}")
    return float(raw_value[1])


def _query_optional_scalar(
    client: PrometheusQueryAPI,
    query: str,
    timestamp: float,
    *,
    description: str,
) -> float | None:
    result = client.instant_query(query, timestamp)
    if len(result) == 0:
        return None
    if len(result) != 1:
        raise ValueError(
            f"expected at most one series for {description}, got {len(result)}"
        )
    raw_value = result[0].get("value")
    if not isinstance(raw_value, list) or len(raw_value) != 2:
        raise ValueError(f"invalid scalar payload for {description}")
    raw_scalar = raw_value[1]
    if raw_scalar in {"NaN", "nan", "null", None}:
        return None
    return float(raw_scalar)


def _extract_single_series_values(
    result: list[dict[str, Any]],
    *,
    description: str,
    scale_factor: float,
) -> list[float]:
    if len(result) == 0:
        return []
    return _extract_series_values(
        result[0],
        description=description,
        scale_factor=scale_factor,
    )


def _extract_optional_single_series_values(
    result: list[dict[str, Any]],
    *,
    scale_factor: float,
) -> list[float]:
    if len(result) == 0:
        return []
    raw_values = result[0].get("values")
    if not isinstance(raw_values, list):
        raise ValueError("invalid series payload for memory baseline")

    values: list[float] = []
    for sample in raw_values:
        if not isinstance(sample, list | tuple) or len(sample) != 2:
            raise ValueError("invalid sample payload for memory baseline")
        raw_value = sample[1]
        if raw_value in {"NaN", "nan", "null", None}:
            continue
        values.append(float(raw_value) * scale_factor)
    return values


def _extract_series_values(
    series: dict[str, Any],
    *,
    description: str,
    scale_factor: float,
) -> list[float]:
    raw_values = series.get("values")
    if not isinstance(raw_values, list):
        raise ValueError(f"invalid series payload for {description}")

    values: list[float] = []
    for sample in raw_values:
        if not isinstance(sample, list | tuple) or len(sample) != 2:
            raise ValueError(f"invalid sample payload for {description}")
        raw_value = sample[1]
        if raw_value in {"NaN", "nan", "null", None}:
            continue
        values.append(float(raw_value) * scale_factor)

    if not values:
        raise MissingMonitorDataError(f"no samples returned for {description}")
    return values


def _summarize_values(values: list[float], prefix: str) -> dict[str, float]:
    series = pd.Series(values, dtype="float64")
    return {
        f"{prefix}_avg": round(float(series.mean()), 3),
        f"{prefix}_max": round(float(series.max()), 3),
        f"{prefix}_p95": round(float(series.quantile(0.95)), 3),
    }
