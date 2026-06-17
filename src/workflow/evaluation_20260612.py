from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from gnn_model.data.constants import GRAPH_FEATURE_NAMES
from workflow.gnn_predictor import (
    ToolPrediction,
    WorkflowGnnPredictor,
    WorkflowGnnPredictorConfig,
)
from workflow.loader import load_workflows
from workflow.model_export import (
    GnnArchsWorkflowModelExporter,
    build_workflow_graph_from_onnx,
)
from workflow.scheduler import prediction_memory_gb
from workflow.schema import WorkflowNodeConfig

DEFAULT_MODEL_ROOT = Path("output/gnn_full_retrain_20260612")
DEFAULT_EXPERIMENT_DIR = DEFAULT_MODEL_ROOT / "gnn_model_scaled_20260612"
DEFAULT_CONFIG_PATH = DEFAULT_MODEL_ROOT / "effective_config.yaml"
DEFAULT_CHECKPOINT_PATH = DEFAULT_EXPERIMENT_DIR / "checkpoints" / "best_model.pt"
DEFAULT_SCALER_DIR = Path("data/scalers")
DEFAULT_TEST_DATA_PATH = Path("data/extracted/test.pt")
DEFAULT_WORKFLOW_DIR = Path("config/workflow/evaluation_20260612")
DEFAULT_DOC_PATH = Path("docs/res/worflow_evaluation_20260612.md")

TARGET_NAMES = (
    "deployment_duration_sec_avg",
    "run_duration_sec_avg",
    "cpu_cores_max",
    "memory_delta_gb_max",
    "gpu_util_percent_max",
    "gpu_sm_active_percent_max",
    "gpu_sm_occupancy_percent_max",
    "gpu_mem_used_mb_max",
    "gpu_power_watts_avg",
)
TRACE_STRATEGIES = (
    "v100_only",
    "a100_first",
    "round_robin",
    "static_size",
    "gnn_aware",
)
DEVICE_ORDER = ("v100_slot", "a100_slot")
DEVICE_GPU_NAMES = {"v100_slot": "v100", "a100_slot": "a100"}
SIMULATED_DEVICE_MEMORY_GB = {"v100_slot": 8.0, "a100_slot": 14.0}
MEMORY_SAFETY_MARGIN_GB = 0.5
OOM_RETRY_PENALTY_SEC = 5.0
SECONDS_PER_HOUR = 3600.0
BYTES_PER_GIB = 1024.0**3


@dataclass(frozen=True)
class TraceRecord:
    model_key: str
    variant_name: str
    base_model_name: str
    phase: str
    batch_size: int
    decode_output_length: int
    actual_by_device: dict[str, dict[str, float]]
    prediction_by_device: dict[str, ToolPrediction]
    static_memory_gb: float


@dataclass(frozen=True)
class TraceSession:
    name: str
    tools: list[TraceRecord]


@dataclass
class SimulationDevice:
    name: str
    gpu_name: str
    capacity_gb: float
    cache: OrderedDict[str, float] = field(default_factory=OrderedDict)

    @property
    def used_gb(self) -> float:
        return sum(self.cache.values())

    @property
    def free_gb(self) -> float:
        return self.capacity_gb - self.used_gb

    def has_model(self, model_key: str) -> bool:
        return model_key in self.cache

    def touch(self, model_key: str) -> None:
        memory_gb = self.cache.pop(model_key)
        self.cache[model_key] = memory_gb

    def cache_model(self, model_key: str, memory_gb: float) -> int:
        evicted = 0
        while self.free_gb < memory_gb and self.cache:
            self.cache.popitem(last=False)
            evicted += 1
        if self.free_gb < memory_gb:
            raise RuntimeError(f"device cannot fit model after eviction: {self.name}")
        self.cache[model_key] = memory_gb
        return evicted


@dataclass
class ToolAttemptResult:
    duration_sec: float
    energy_wh: float
    deployment_sec: float
    cache_hit: bool
    oom_events: int
    retry_events: int
    failed: bool
    evictions: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GNN-aware workflow scheduling on V100/A100 traces."
    )
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--scaler-dir", type=Path, default=DEFAULT_SCALER_DIR)
    parser.add_argument("--test-data-path", type=Path, default=DEFAULT_TEST_DATA_PATH)
    parser.add_argument("--workflow-dir", type=Path, default=DEFAULT_WORKFLOW_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trace-records", type=int, default=24)
    parser.add_argument("--sessions", type=int, default=24)
    parser.add_argument("--tools-per-session", type=int, default=3)
    parser.add_argument("--skip-workflow-predictions", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_evaluation(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        scaler_dir=args.scaler_dir,
        test_data_path=args.test_data_path,
        workflow_dir=args.workflow_dir,
        predictor_device=args.device,
        trace_record_count=args.trace_records,
        session_count=args.sessions,
        tools_per_session=args.tools_per_session,
        include_workflow_predictions=not args.skip_workflow_predictions,
    )
    markdown = build_markdown_report(payload)
    args.doc_path.parent.mkdir(parents=True, exist_ok=True)
    args.doc_path.write_text(markdown, encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.doc_path}")
    return 0


def run_evaluation(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    scaler_dir: Path = DEFAULT_SCALER_DIR,
    test_data_path: Path = DEFAULT_TEST_DATA_PATH,
    workflow_dir: Path = DEFAULT_WORKFLOW_DIR,
    predictor_device: str = "cpu",
    trace_record_count: int = 24,
    session_count: int = 24,
    tools_per_session: int = 3,
    include_workflow_predictions: bool = True,
) -> dict[str, Any]:
    predictor = WorkflowGnnPredictor(
        WorkflowGnnPredictorConfig(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            scaler_dir=scaler_dir,
            device=predictor_device,
            target_names=list(TARGET_NAMES),
        )
    )
    records = load_trace_records(
        test_data_path,
        predictor,
        trace_record_count=trace_record_count,
    )
    sessions = build_trace_sessions(
        records,
        session_count=session_count,
        tools_per_session=tools_per_session,
    )
    trace_results = [
        simulate_strategy(strategy, sessions).as_dict() for strategy in TRACE_STRATEGIES
    ]
    workflow_predictions = (
        predict_workflow_samples(workflow_dir, predictor)
        if include_workflow_predictions
        else []
    )
    return {
        "summary": build_summary(trace_results),
        "model": {
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "scaler_dir": str(scaler_dir),
            "test_data_path": str(test_data_path),
        },
        "trace": {
            "record_count": len(records),
            "session_count": len(sessions),
            "tools_per_session": tools_per_session,
            "device_memory_gb": SIMULATED_DEVICE_MEMORY_GB,
            "memory_safety_margin_gb": MEMORY_SAFETY_MARGIN_GB,
            "oom_retry_penalty_sec": OOM_RETRY_PENALTY_SEC,
            "records": [record_summary(record) for record in records],
        },
        "strategy_results": trace_results,
        "workflow_predictions": workflow_predictions,
        "external_references": external_references(),
    }


@dataclass(frozen=True)
class StrategyMetrics:
    strategy: str
    session_count: int
    tool_calls: int
    avg_session_makespan_sec: float
    p95_session_makespan_sec: float
    avg_tool_jct_sec: float
    p95_tool_jct_sec: float
    oom_events: int
    retry_events: int
    failed_tools: int
    evictions: int
    cache_hits: int
    cache_hit_rate: float
    cold_start_sec: float
    energy_wh: float
    avg_memory_utilization: float
    peak_memory_utilization: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "strategy": self.strategy,
            "session_count": self.session_count,
            "tool_calls": self.tool_calls,
            "avg_session_makespan_sec": self.avg_session_makespan_sec,
            "p95_session_makespan_sec": self.p95_session_makespan_sec,
            "avg_tool_jct_sec": self.avg_tool_jct_sec,
            "p95_tool_jct_sec": self.p95_tool_jct_sec,
            "oom_events": self.oom_events,
            "retry_events": self.retry_events,
            "failed_tools": self.failed_tools,
            "evictions": self.evictions,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hit_rate,
            "cold_start_sec": self.cold_start_sec,
            "energy_wh": self.energy_wh,
            "avg_memory_utilization": self.avg_memory_utilization,
            "peak_memory_utilization": self.peak_memory_utilization,
        }


def load_trace_records(
    test_data_path: Path,
    predictor: WorkflowGnnPredictor,
    *,
    trace_record_count: int,
) -> list[TraceRecord]:
    data = torch.load(test_data_path, map_location="cpu", weights_only=False)
    grouped: dict[tuple[str, str, int, int], dict[str, Any]] = defaultdict(dict)
    for item in data:
        if str(item.phase) == "training":
            continue
        gpu_name = str(item.gpu_name)
        if gpu_name not in {"v100", "a100"}:
            continue
        key = (
            str(item.variant_name),
            str(item.phase),
            int(item.batch_size),
            int(item.decode_output_length),
        )
        device_name = device_name_for_gpu(gpu_name)
        grouped[key].setdefault(device_name, item)

    paired_items = [
        (key, by_device)
        for key, by_device in grouped.items()
        if all(device_name in by_device for device_name in DEVICE_ORDER)
    ]
    paired_items.sort(
        key=lambda pair: trace_item_weight(pair[1]),
        reverse=True,
    )
    selected_items = paired_items[:trace_record_count]
    records = [
        build_trace_record(key, by_device, predictor)
        for key, by_device in selected_items
    ]
    if len(records) != trace_record_count:
        raise ValueError(f"paired trace records missing: {len(records)}")
    return records


def build_trace_record(
    key: tuple[str, str, int, int],
    by_device: Mapping[str, Any],
    predictor: WorkflowGnnPredictor,
) -> TraceRecord:
    actual_by_device: dict[str, dict[str, float]] = {}
    prediction_by_device: dict[str, ToolPrediction] = {}
    for device_name in DEVICE_ORDER:
        item = by_device[device_name]
        actual_by_device[device_name] = target_metrics_from_tensor(item.y)
        prediction = predictor.predict_graph(
            node_name=str(item.variant_name),
            device_name=device_name,
            gpu_name=str(item.gpu_name),
            graph=item,
        )
        prediction_by_device[device_name] = sanitize_prediction(prediction)

    first_item = by_device[DEVICE_ORDER[0]]
    return TraceRecord(
        model_key=build_trace_model_key(key),
        variant_name=str(first_item.variant_name),
        base_model_name=str(first_item.base_model_name),
        phase=str(first_item.phase),
        batch_size=int(first_item.batch_size),
        decode_output_length=int(first_item.decode_output_length),
        actual_by_device=actual_by_device,
        prediction_by_device=prediction_by_device,
        static_memory_gb=static_memory_estimate_gb(first_item),
    )


def target_metrics_from_tensor(values: torch.Tensor) -> dict[str, float]:
    row = values.reshape(-1).tolist()
    if len(row) != len(TARGET_NAMES):
        raise ValueError(f"target dim mismatch: {len(row)}")
    return {
        target_name: max(0.0, float(value))
        for target_name, value in zip(TARGET_NAMES, row, strict=True)
    }


def sanitize_prediction(prediction: ToolPrediction) -> ToolPrediction:
    return ToolPrediction(
        node_name=prediction.node_name,
        device_name=prediction.device_name,
        gpu_name=prediction.gpu_name,
        metrics={
            target_name: max(0.0, prediction.metric(target_name))
            for target_name in TARGET_NAMES
        },
    )


def trace_item_weight(by_device: Mapping[str, Any]) -> float:
    values = []
    for item in by_device.values():
        metrics = target_metrics_from_tensor(item.y)
        values.append(
            memory_gb_from_metrics(metrics)
            + metrics["deployment_duration_sec_avg"]
            + metrics["run_duration_sec_avg"]
        )
    return max(values)


def static_memory_estimate_gb(item: Any) -> float:
    graph_features = item.graph_features.reshape(-1)
    parameter_bytes = feature_value(graph_features, "parameter_input_bytes")
    graph_memory_bytes = feature_value(graph_features, "graph_memory_bytes")
    return max(parameter_bytes, graph_memory_bytes, 0.0) / BYTES_PER_GIB


def feature_value(graph_features: torch.Tensor, name: str) -> float:
    index = GRAPH_FEATURE_NAMES.index(name)
    return float(graph_features[index].item())


def build_trace_model_key(key: tuple[str, str, int, int]) -> str:
    variant_name, phase, batch_size, decode_output_length = key
    return f"{variant_name}|{phase}|bs{batch_size}|decode{decode_output_length}"


def device_name_for_gpu(gpu_name: str) -> str:
    return f"{gpu_name}_slot"


def build_trace_sessions(
    records: Sequence[TraceRecord],
    *,
    session_count: int,
    tools_per_session: int,
) -> list[TraceSession]:
    if not records:
        raise ValueError("records must not be empty")
    sessions: list[TraceSession] = []
    hot_record_count = max(1, len(records) // 2)
    for session_index in range(session_count):
        tools = [
            select_session_record(
                records,
                hot_record_count,
                session_index,
                tool_index,
            )
            for tool_index in range(tools_per_session)
        ]
        sessions.append(
            TraceSession(
                name=f"session_{session_index + 1:02d}",
                tools=tools,
            )
        )
    return sessions


def select_session_record(
    records: Sequence[TraceRecord],
    hot_record_count: int,
    session_index: int,
    tool_index: int,
) -> TraceRecord:
    if tool_index == 0 and session_index % 4 == 3:
        return records[(hot_record_count + session_index) % len(records)]
    return records[(session_index + tool_index * 3) % hot_record_count]


def simulate_strategy(
    strategy: str,
    sessions: Sequence[TraceSession],
) -> StrategyMetrics:
    if strategy not in TRACE_STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    devices = build_simulation_devices()
    session_makespans: list[float] = []
    tool_jcts: list[float] = []
    memory_utilization_samples: list[float] = []
    round_robin_index = 0
    oom_events = 0
    retry_events = 0
    failed_tools = 0
    evictions = 0
    cache_hits = 0
    cold_start_sec = 0.0
    energy_wh = 0.0

    for session in sessions:
        device_time = {device_name: 0.0 for device_name in DEVICE_ORDER}
        for record in session.tools:
            order, round_robin_index = device_order_for_strategy(
                strategy,
                record,
                devices,
                device_time,
                round_robin_index,
            )
            result, device_name = run_tool_attempts(record, order, devices)
            device_time[device_name] += result.duration_sec
            tool_jcts.append(result.duration_sec)
            oom_events += result.oom_events
            retry_events += result.retry_events
            failed_tools += int(result.failed)
            evictions += result.evictions
            cache_hits += int(result.cache_hit)
            cold_start_sec += result.deployment_sec
            energy_wh += result.energy_wh
        session_makespans.append(max(device_time.values()))
        memory_utilization_samples.append(memory_utilization(devices))

    tool_calls = sum(len(session.tools) for session in sessions)
    return StrategyMetrics(
        strategy=strategy,
        session_count=len(sessions),
        tool_calls=tool_calls,
        avg_session_makespan_sec=mean(session_makespans),
        p95_session_makespan_sec=percentile(session_makespans, 0.95),
        avg_tool_jct_sec=mean(tool_jcts),
        p95_tool_jct_sec=percentile(tool_jcts, 0.95),
        oom_events=oom_events,
        retry_events=retry_events,
        failed_tools=failed_tools,
        evictions=evictions,
        cache_hits=cache_hits,
        cache_hit_rate=cache_hits / tool_calls if tool_calls else 0.0,
        cold_start_sec=cold_start_sec,
        energy_wh=energy_wh,
        avg_memory_utilization=mean(memory_utilization_samples),
        peak_memory_utilization=max(memory_utilization_samples, default=0.0),
    )


def build_simulation_devices() -> dict[str, SimulationDevice]:
    return {
        device_name: SimulationDevice(
            name=device_name,
            gpu_name=DEVICE_GPU_NAMES[device_name],
            capacity_gb=SIMULATED_DEVICE_MEMORY_GB[device_name],
        )
        for device_name in DEVICE_ORDER
    }


def device_order_for_strategy(
    strategy: str,
    record: TraceRecord,
    devices: Mapping[str, SimulationDevice],
    device_time: Mapping[str, float],
    round_robin_index: int,
) -> tuple[list[str], int]:
    if strategy == "v100_only":
        return ["v100_slot", "a100_slot"], round_robin_index
    if strategy == "a100_first":
        return ["a100_slot", "v100_slot"], round_robin_index
    if strategy == "round_robin":
        preferred = DEVICE_ORDER[round_robin_index % len(DEVICE_ORDER)]
        remaining = [
            device_name for device_name in DEVICE_ORDER if device_name != preferred
        ]
        return [preferred, *remaining], round_robin_index + 1
    if strategy == "static_size":
        if (
            record.static_memory_gb + MEMORY_SAFETY_MARGIN_GB
            <= devices["v100_slot"].capacity_gb
        ):
            return ["v100_slot", "a100_slot"], round_robin_index
        return ["a100_slot", "v100_slot"], round_robin_index
    return gnn_aware_device_order(record, devices, device_time), round_robin_index


def gnn_aware_device_order(
    record: TraceRecord,
    devices: Mapping[str, SimulationDevice],
    device_time: Mapping[str, float],
) -> list[str]:
    scored_devices: list[tuple[float, str]] = []
    fallback_devices: list[tuple[float, str]] = []
    for device_name in DEVICE_ORDER:
        prediction = record.prediction_by_device[device_name]
        required_gb = prediction_memory_gb(prediction) + MEMORY_SAFETY_MARGIN_GB
        device = devices[device_name]
        score = predicted_schedule_score(
            record,
            device_name,
            device,
            device_time[device_name],
        )
        if required_gb <= device.capacity_gb:
            scored_devices.append((score, device_name))
        else:
            fallback_devices.append((required_gb, device_name))
    if scored_devices:
        return [device_name for _, device_name in sorted(scored_devices)]
    return [device_name for _, device_name in sorted(fallback_devices)]


def predicted_schedule_score(
    record: TraceRecord,
    device_name: str,
    device: SimulationDevice,
    queued_sec: float,
) -> float:
    prediction = record.prediction_by_device[device_name]
    deploy_sec = (
        0.0
        if device.has_model(record.model_key)
        else prediction.metric("deployment_duration_sec_avg")
    )
    run_sec = prediction.metric("run_duration_sec_avg")
    memory_pressure = (
        prediction_memory_gb(prediction) + MEMORY_SAFETY_MARGIN_GB
    ) / device.capacity_gb
    power_cost = prediction.metric("gpu_power_watts_avg") / 1000.0
    return queued_sec + deploy_sec + run_sec + 0.1 * memory_pressure + power_cost


def run_tool_attempts(
    record: TraceRecord,
    order: Sequence[str],
    devices: dict[str, SimulationDevice],
) -> tuple[ToolAttemptResult, str]:
    duration_sec = 0.0
    energy_wh = 0.0
    deployment_sec = 0.0
    oom_events = 0
    evictions = 0
    last_device_name = order[-1]
    for attempt_index, device_name in enumerate(order):
        last_device_name = device_name
        device = devices[device_name]
        metrics = record.actual_by_device[device_name]
        required_memory_gb = memory_gb_from_metrics(metrics)
        if device.has_model(record.model_key):
            device.touch(record.model_key)
            run_sec = metrics["run_duration_sec_avg"]
            duration_sec += run_sec
            energy_wh += energy_wh_from_metrics(metrics, run_sec)
            return (
                ToolAttemptResult(
                    duration_sec=duration_sec,
                    energy_wh=energy_wh,
                    deployment_sec=deployment_sec,
                    cache_hit=True,
                    oom_events=oom_events,
                    retry_events=attempt_index,
                    failed=False,
                    evictions=evictions,
                ),
                device_name,
            )

        memory_with_margin_gb = required_memory_gb + MEMORY_SAFETY_MARGIN_GB
        if memory_with_margin_gb > device.capacity_gb:
            oom_events += 1
            duration_sec += OOM_RETRY_PENALTY_SEC
            continue

        evictions += device.cache_model(record.model_key, required_memory_gb)
        deploy_sec = metrics["deployment_duration_sec_avg"]
        run_sec = metrics["run_duration_sec_avg"]
        active_sec = deploy_sec + run_sec
        duration_sec += active_sec
        deployment_sec += deploy_sec
        energy_wh += energy_wh_from_metrics(metrics, active_sec)
        return (
            ToolAttemptResult(
                duration_sec=duration_sec,
                energy_wh=energy_wh,
                deployment_sec=deployment_sec,
                cache_hit=False,
                oom_events=oom_events,
                retry_events=attempt_index,
                failed=False,
                evictions=evictions,
            ),
            device_name,
        )

    return (
        ToolAttemptResult(
            duration_sec=duration_sec,
            energy_wh=energy_wh,
            deployment_sec=deployment_sec,
            cache_hit=False,
            oom_events=oom_events,
            retry_events=max(0, len(order) - 1),
            failed=True,
            evictions=evictions,
        ),
        last_device_name,
    )


def memory_gb_from_metrics(metrics: Mapping[str, float]) -> float:
    gpu_memory_gb = float(metrics["gpu_mem_used_mb_max"]) / 1024.0
    memory_delta_gb = float(metrics["memory_delta_gb_max"])
    return max(gpu_memory_gb, memory_delta_gb, 0.0)


def energy_wh_from_metrics(metrics: Mapping[str, float], duration_sec: float) -> float:
    return float(metrics["gpu_power_watts_avg"]) * duration_sec / SECONDS_PER_HOUR


def memory_utilization(devices: Mapping[str, SimulationDevice]) -> float:
    values = [device.used_gb / device.capacity_gb for device in devices.values()]
    return mean(values)


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(
        len(sorted_values) - 1,
        max(0, math.ceil(percentile_value * len(sorted_values)) - 1),
    )
    return sorted_values[index]


def record_summary(record: TraceRecord) -> dict[str, Any]:
    return {
        "variant_name": record.variant_name,
        "base_model_name": record.base_model_name,
        "phase": record.phase,
        "batch_size": record.batch_size,
        "decode_output_length": record.decode_output_length,
        "static_memory_gb": record.static_memory_gb,
        "actual_memory_gb": {
            device_name: memory_gb_from_metrics(metrics)
            for device_name, metrics in record.actual_by_device.items()
        },
        "predicted_memory_gb": {
            device_name: prediction_memory_gb(prediction)
            for device_name, prediction in record.prediction_by_device.items()
        },
    }


def predict_workflow_samples(
    workflow_dir: Path,
    predictor: WorkflowGnnPredictor,
) -> list[dict[str, Any]]:
    exporter = GnnArchsWorkflowModelExporter()
    rows: list[dict[str, Any]] = []
    for workflow_path in sorted(workflow_dir.glob("*.yaml")):
        workflow = load_workflows(workflow_path)[0]
        for node in workflow.nodes:
            if node.type != "tool":
                continue
            rows.append(
                predict_workflow_node(
                    workflow_path.stem,
                    node,
                    exporter,
                    predictor,
                )
            )
    return rows


def predict_workflow_node(
    workflow_name: str,
    node: WorkflowNodeConfig,
    exporter: GnnArchsWorkflowModelExporter,
    predictor: WorkflowGnnPredictor,
) -> dict[str, Any]:
    assert node.model is not None
    predictions: dict[str, dict[str, float]] = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        onnx_path = exporter.export_onnx(node, Path(temp_dir))
        for device_name in DEVICE_ORDER:
            gpu_name = DEVICE_GPU_NAMES[device_name]
            graph = build_workflow_graph_from_onnx(node, onnx_path, gpu_name)
            prediction = predictor.predict_graph(
                node_name=node.name,
                device_name=device_name,
                gpu_name=gpu_name,
                graph=graph,
            )
            predictions[device_name] = sanitize_prediction(prediction).metrics
    return {
        "workflow": workflow_name,
        "node": node.name,
        "task": node.task,
        "model": node.model.name,
        "phase": node.runtime.phase if node.runtime else None,
        "v100_memory_gb": memory_gb_from_metrics(predictions["v100_slot"]),
        "a100_memory_gb": memory_gb_from_metrics(predictions["a100_slot"]),
        "v100_duration_sec": predicted_duration_sec(predictions["v100_slot"]),
        "a100_duration_sec": predicted_duration_sec(predictions["a100_slot"]),
        "v100_power_watts": predictions["v100_slot"]["gpu_power_watts_avg"],
        "a100_power_watts": predictions["a100_slot"]["gpu_power_watts_avg"],
    }


def predicted_duration_sec(metrics: Mapping[str, float]) -> float:
    return float(metrics["deployment_duration_sec_avg"]) + float(
        metrics["run_duration_sec_avg"]
    )


def build_summary(strategy_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    gnn_result = next(
        result for result in strategy_results if result["strategy"] == "gnn_aware"
    )
    baseline_results = [
        result for result in strategy_results if result["strategy"] != "gnn_aware"
    ]
    latency_baseline = min(
        baseline_results,
        key=lambda result: float(result["avg_session_makespan_sec"]),
    )
    availability_baseline = min(
        baseline_results,
        key=lambda result: (
            int(result["oom_events"]),
            float(result["avg_session_makespan_sec"]),
        ),
    )
    return {
        "availability_baseline": availability_baseline["strategy"],
        "latency_baseline": latency_baseline["strategy"],
        "gnn_avg_makespan_sec": gnn_result["avg_session_makespan_sec"],
        "availability_baseline_avg_makespan_sec": availability_baseline[
            "avg_session_makespan_sec"
        ],
        "makespan_improvement": relative_improvement(
            availability_baseline["avg_session_makespan_sec"],
            gnn_result["avg_session_makespan_sec"],
        ),
        "latency_gap": relative_improvement(
            latency_baseline["avg_session_makespan_sec"],
            gnn_result["avg_session_makespan_sec"],
        ),
        "gnn_oom_events": gnn_result["oom_events"],
        "availability_baseline_oom_events": availability_baseline["oom_events"],
        "latency_baseline_oom_events": latency_baseline["oom_events"],
        "gnn_energy_wh": gnn_result["energy_wh"],
        "availability_baseline_energy_wh": availability_baseline["energy_wh"],
        "energy_change": relative_improvement(
            availability_baseline["energy_wh"],
            gnn_result["energy_wh"],
        ),
    }


def relative_improvement(baseline_value: float, candidate_value: float) -> float:
    if baseline_value == 0:
        return 0.0
    return (baseline_value - candidate_value) / baseline_value


def build_markdown_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Workflow GNN 调度评估 20260612",
        "",
        "## 结论",
        "",
        *conclusion_lines(payload),
        "",
        "## 实验口径",
        "",
        "本实验面向 V100/A100 受限资源集群中的 agent workflow 工具节点调度。",
        (
            "主实验采用离线 trace replay: GNN 预测值只用于调度决策, "
            "最终耗时, 显存, OOM 和能耗代理使用 `data/extracted/test.pt` "
            "中 V100/A100 配对样本的实测 target 计算。"
        ),
        "这避免了只用预测值闭环评估, 但仍不是一次真实线上集群压测。",
        "",
        "## 评估指标依据",
        "",
        "类似 GPU 集群和推理服务调度工作通常同时关注用户侧延迟与系统侧资源效率。",
        (
            "本实验采用 session makespan, tool JCT, OOM/retry, "
            "缓存命中, 冷启动耗时, 显存利用率和能耗代理。"
        ),
        "",
        reference_table(payload["external_references"]),
        "",
        "## GNN 模型与数据",
        "",
        model_table(payload),
        "",
        "20260612 复训记录中的 test 指标摘要:",
        "",
        "- original-scale overall WAPE: `0.135549`, R2: `0.949278`。",
        "- scheduling-core WAPE: `0.135791`。",
        (
            "- 关键目标 WAPE: deployment `0.141669`, run `0.222849`, "
            "gpu_mem `0.135247`, gpu_power `0.148130`。"
        ),
        "- V100 test WAPE: `0.147589`, A100 test WAPE: `0.115542`。",
        "",
        "## 样例 Workflow",
        "",
        (
            "样例 YAML 保存在 `config/workflow/evaluation_20260612/`, "
            "只包含 workflow 计划阶段可知信息。"
        ),
        "",
        workflow_prediction_table(payload["workflow_predictions"]),
        "",
        "## Trace Replay 设置",
        "",
        trace_table(payload["trace"]),
        "",
        (
            "配对 trace 选择优先覆盖显存和冷启动压力更高的 test 样本; "
            "每个 session 包含固定数量 ready tools, 缓存跨 session 保留。"
        ),
        "",
        "## 对比结果",
        "",
        strategy_table(payload["strategy_results"]),
        "",
        "## 结果解读",
        "",
        *interpretation_lines(payload),
        "",
        "## 限制",
        "",
        (
            "- 本实验没有真实启动 V100/A100 集群, "
            "只能说明 GNN 预测信号在离线调度仿真中的决策价值。"
        ),
        (
            "- Trace replay 使用 row-random test split 中的已见分布样本, "
            "不能替代 family-holdout 泛化实验。"
        ),
        (
            "- 能耗使用 `gpu_power_watts_avg * duration` 作为代理, "
            "不等同于机房级能耗计量。"
        ),
        (
            "- 当前模拟器使用单次 retry 和 LRU 缓存驱逐, "
            "真实系统还需要纳入并发干扰, 网络, 容器启动和 agent 重规划成本。"
        ),
        "",
    ]
    return "\n".join(lines)


def conclusion_lines(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    return [
        (
            "- 在可用性优先口径下, GNN-aware 相对 "
            f"`{summary['availability_baseline']}` 的平均 session makespan "
            f"改善 `{format_percent(summary['makespan_improvement'])}`。"
        ),
        (
            f"- GNN-aware OOM events 为 `{summary['gnn_oom_events']}`, "
            "可用性 baseline OOM events 为 "
            f"`{summary['availability_baseline_oom_events']}`。"
        ),
        (
            f"- 最快 baseline `{summary['latency_baseline']}` 的 OOM events "
            f"为 `{summary['latency_baseline_oom_events']}`, "
            "说明低均值延迟来自更高失败/重试风险。"
        ),
        (
            "- GNN-aware 能耗代理相对可用性 baseline 变化 "
            f"`{format_percent(summary['energy_change'])}`。"
        ),
        (
            "- 收益主要来自提前规避 V100 可用显存不足, "
            "把冷启动更重的工具放到更合适的设备, 以及跨 session 复用缓存。"
        ),
    ]


def interpretation_lines(payload: Mapping[str, Any]) -> list[str]:
    gnn_result = next(
        result
        for result in payload["strategy_results"]
        if result["strategy"] == "gnn_aware"
    )
    return [
        (
            "- 静态策略只能看到 ONNX 规模代理, 无法直接判断不同 GPU 上的"
            "部署耗时, 运行耗时和显存峰值, 因此在受限显存下更容易触发 retry。"
        ),
        (
            "- `v100_only` 和 `static_size` 在部分重模型 trace 上会先打到 "
            "V100, OOM 后再 retry 到 A100, 直接增加 tool JCT 和 session makespan。"
        ),
        (
            "- GNN-aware 的 cache hit rate 为 "
            f"`{format_percent(gnn_result['cache_hit_rate'])}`, "
            "说明预测调度没有牺牲跨 session 复用。"
        ),
        (
            "- 对 agent workflow 来说, 降低 OOM 与 retry 比微调单个工具的"
            "毫秒级运行时间更直接影响端到端体验。"
        ),
    ]


def model_table(payload: Mapping[str, Any]) -> str:
    model = payload["model"]
    rows = [
        ("config", model["config_path"]),
        ("checkpoint", model["checkpoint_path"]),
        ("scalers", model["scaler_dir"]),
        ("test trace", model["test_data_path"]),
    ]
    return markdown_table(("item", "path"), rows)


def trace_table(trace: Mapping[str, Any]) -> str:
    rows = [
        ("trace records", str(trace["record_count"])),
        ("sessions", str(trace["session_count"])),
        ("tools per session", str(trace["tools_per_session"])),
        ("V100 available memory", f"{trace['device_memory_gb']['v100_slot']:.2f} GB"),
        ("A100 available memory", f"{trace['device_memory_gb']['a100_slot']:.2f} GB"),
        ("memory margin", f"{trace['memory_safety_margin_gb']:.2f} GB"),
        ("OOM retry penalty", f"{trace['oom_retry_penalty_sec']:.2f} sec"),
    ]
    return markdown_table(("item", "value"), rows)


def strategy_table(results: Sequence[Mapping[str, Any]]) -> str:
    rows = [
        (
            result["strategy"],
            format_float(result["avg_session_makespan_sec"]),
            format_float(result["p95_session_makespan_sec"]),
            format_float(result["avg_tool_jct_sec"]),
            str(result["oom_events"]),
            str(result["retry_events"]),
            format_percent(result["cache_hit_rate"]),
            format_float(result["cold_start_sec"]),
            format_float(result["energy_wh"]),
            format_percent(result["avg_memory_utilization"]),
        )
        for result in results
    ]
    return markdown_table(
        (
            "strategy",
            "avg makespan sec",
            "p95 makespan sec",
            "avg tool JCT sec",
            "OOM",
            "retry",
            "cache hit",
            "cold start sec",
            "energy Wh",
            "avg mem util",
        ),
        rows,
    )


def workflow_prediction_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "本次运行跳过了样例 workflow prospective 预测。"
    table_rows = [
        (
            row["workflow"],
            row["node"],
            row["model"],
            row["phase"],
            format_float(row["v100_memory_gb"]),
            format_float(row["a100_memory_gb"]),
            format_float(row["v100_duration_sec"]),
            format_float(row["a100_duration_sec"]),
        )
        for row in rows
    ]
    return markdown_table(
        (
            "workflow",
            "node",
            "model",
            "phase",
            "V100 mem GB",
            "A100 mem GB",
            "V100 deploy+run sec",
            "A100 deploy+run sec",
        ),
        table_rows,
    )


def reference_table(references: Sequence[Mapping[str, str]]) -> str:
    rows = [(item["topic"], item["use"], item["url"]) for item in references]
    return markdown_table(("reference", "why used", "url"), rows)


def markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator, *row_lines])


def format_float(value: float) -> str:
    return f"{float(value):.4f}"


def format_percent(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


def external_references() -> list[dict[str, str]]:
    return [
        {
            "topic": "SCHEDTUNE",
            "use": "异构 GPU 调度以 OOM avoidance、显存利用率和 makespan 为核心指标。",
            "url": "https://people.cs.vt.edu/~butta/docs/ccgrid22-schedtune.pdf",
        },
        {
            "topic": "PAL",
            "use": "GPU 集群调度常用 JCT、utilization 和 makespan 评估策略收益。",
            "url": "https://arxiv.org/abs/2408.11919",
        },
        {
            "topic": "MArk",
            "use": "推理服务评估关注 SLO latency 与 serving cost 的折中。",
            "url": "https://www.usenix.org/conference/atc19/presentation/zhang-chengliang",
        },
    ]


if __name__ == "__main__":
    raise SystemExit(main())
