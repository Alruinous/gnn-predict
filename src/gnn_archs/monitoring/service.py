from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from gnn_archs.result import ResultDocument

from .prometheus import PrometheusClient
from .queries import (
    GPU_METRIC_DEFINITIONS,
    build_gpu_metrics_query,
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
    "phase",
    "started_at_ts",
    "ended_at_ts",
    "duration_sec",
    "phase_rounds",
    "sample_count",
    "batch_size",
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
    "gpu_util_percent_avg",
    "gpu_util_percent_max",
    "gpu_util_percent_p95",
    "gpu_sm_active_percent_avg",
    "gpu_sm_active_percent_max",
    "gpu_sm_active_percent_p95",
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
    phase_rounds: int
    batch_size: int


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
                    gpu_id=gpu_id,
                    batch_size=int(variant.training.hyperparameters["batch_size"]),
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
                    gpu_id=gpu_id,
                    batch_size=int(variant.inference.metrics["batch_size"])
                )
            )

    if not records:
        raise ValueError(f"no training or inference phases found in {result_json}")

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

    reference_timestamp = phase_records[0].started_at_ts
    _validate_target_pod(client, target, settings.namespace, reference_timestamp)
    node_total_cpu = _query_scalar(
        client,
        build_node_cpu_total_query(target.node_name),
        reference_timestamp,
        description="node CPU total",
    )
    node_total_memory_gb = _query_scalar(
        client,
        build_node_memory_total_query(target.node_name),
        reference_timestamp,
        description="node memory total",
    ) / (1024**3)

    rows = []
    for record in phase_records:
        row = _monitor_phase_record(
            phase_record=record,
            client=client,
            cpu_rate_window=settings.cpu_rate_window,
            cpu_rate_window_seconds=settings.cpu_rate_window_seconds,
            query_step_seconds=settings.query_step_seconds,
            node_total_cpu=node_total_cpu,
            node_total_memory_gb=node_total_memory_gb,
            logger=logger,
        )
        if len(row) == 0:
            logger.warning(
                f"skipping monitor record for {record.variant_name}/"
                + f"{record.phase} due to missing metrics"
            )
            continue
        rows.append(row)

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
    gpu_id: str,
    batch_size: int,
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
        phase_rounds = training.hyperparameters.get("epochs")
    else:
        assert phase == "inference", phase
        inference = variant.inference
        assert inference is not None
        phase_rounds = inference.metrics.get("iterations")
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
        phase_rounds=phase_rounds,
        batch_size=batch_size,
    )


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
    node_total_cpu: float,
    node_total_memory_gb: float,
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
        logger.warning(
            f"No CPU samples returned for {phase_record.variant_name}/"
            + f"{phase_record.phase}; this may indicate an issue with the "
            + "Prometheus query or scrape configuration."
        )
        return {}

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
        logger.warning(
            f"No memory samples returned for {phase_record.variant_name}/"
            + f"{phase_record.phase}; this may indicate an issue with the "
            + "Prometheus query or scrape configuration."
        )
        return {}

    gpu_query = build_gpu_metrics_query(
        tuple(definition.prometheus_name for definition in GPU_METRIC_DEFINITIONS),
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

    sample_count = min([len(cpu_values), len(memory_values), *gpu_sample_counts])
    if sample_count < MIN_REQUIRED_PHASE_SAMPLES:
        raise ValueError(
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
        "phase": phase_record.phase,
        "started_at_ts": phase_record.started_at_ts,
        "ended_at_ts": phase_record.ended_at_ts,
        "duration_sec": round(phase_record.duration_sec, 6),
        "phase_rounds": phase_record.phase_rounds,
        "sample_count": sample_count,
        "resolved_gpu_label": resolved_gpu_label,
        "resolved_device_label": resolved_device_label,
        "batch_size": phase_record.batch_size,
    }
    row.update(cpu_summary)
    row.update(memory_summary)
    row.update(gpu_metrics)
    return row


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
    sample_counts: list[int] = []
    metric_summaries: dict[str, Any] = {}

    for definition in GPU_METRIC_DEFINITIONS:
        series_candidates = metrics_by_name.get(definition.prometheus_name, [])
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
        device_label = metric_labels.get("device")
        if not gpu_label or not device_label:
            raise ValueError(
                "GPU series must include both gpu and device labels for "
                f"{definition.prometheus_name}"
            )
        if gpu_label != phase_record.gpu_id:
            raise ValueError(
                "GPU metrics resolved to an unexpected gpu label for "
                f"{phase_record.variant_name}/{phase_record.phase}: "
                f"{gpu_label} != {phase_record.gpu_id}"
            )
        if resolved_gpu_label is None:
            resolved_gpu_label = gpu_label
            resolved_device_label = device_label
        elif resolved_gpu_label != gpu_label or resolved_device_label != device_label:
            raise ValueError(
                "GPU metrics resolved to inconsistent labels for "
                f"{phase_record.variant_name}/{phase_record.phase}"
            )

        values = _extract_series_values(
            series,
            description=(
                f"{definition.prometheus_name} for "
                f"{phase_record.variant_name}/{phase_record.phase}"
            ),
            scale_factor=definition.scale_factor,
        )
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
    if len(result) != 1:
        raise ValueError(
            f"expected exactly one series for {description}, got {len(result)}"
        )
    raw_value = result[0].get("value")
    if not isinstance(raw_value, list) or len(raw_value) != 2:
        raise ValueError(f"invalid scalar payload for {description}")
    return float(raw_value[1])


def _extract_single_series_values(
    result: list[dict[str, Any]],
    *,
    description: str,
    scale_factor: float,
) -> list[float]:
    if len(result) == 0:
        return []
    # if len(result) != 1:
    #     raise ValueError(
    #         f"expected exactly one series for {description}, got {len(result)}"
    #     )
    return _extract_series_values(
        result[0],
        description=description,
        scale_factor=scale_factor,
    )


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
        raise ValueError(f"no samples returned for {description}")
    return values


def _summarize_values(values: list[float], prefix: str) -> dict[str, float]:
    series = pd.Series(values, dtype="float64")
    return {
        f"{prefix}_avg": round(float(series.mean()), 3),
        f"{prefix}_max": round(float(series.max()), 3),
        f"{prefix}_p95": round(float(series.quantile(0.95)), 3),
    }
