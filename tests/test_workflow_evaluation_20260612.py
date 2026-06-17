from __future__ import annotations

from workflow.evaluation_20260612 import (
    TraceRecord,
    TraceSession,
    build_markdown_report,
    simulate_strategy,
)
from workflow.gnn_predictor import ToolPrediction


def build_metrics(
    *,
    deploy_sec: float,
    run_sec: float,
    memory_gb: float,
    power_watts: float = 100.0,
) -> dict[str, float]:
    return {
        "deployment_duration_sec_avg": deploy_sec,
        "run_duration_sec_avg": run_sec,
        "cpu_cores_max": 1.0,
        "memory_delta_gb_max": memory_gb,
        "gpu_util_percent_max": 50.0,
        "gpu_sm_active_percent_max": 40.0,
        "gpu_sm_occupancy_percent_max": 20.0,
        "gpu_mem_used_mb_max": memory_gb * 1024.0,
        "gpu_power_watts_avg": power_watts,
    }


def build_record(
    name: str,
    *,
    v100_memory_gb: float,
    a100_memory_gb: float,
    deploy_sec: float = 3.0,
    run_sec: float = 1.0,
) -> TraceRecord:
    actual_by_device = {
        "v100_slot": build_metrics(
            deploy_sec=deploy_sec,
            run_sec=run_sec,
            memory_gb=v100_memory_gb,
        ),
        "a100_slot": build_metrics(
            deploy_sec=deploy_sec,
            run_sec=run_sec,
            memory_gb=a100_memory_gb,
        ),
    }
    return TraceRecord(
        model_key=name,
        variant_name=name,
        base_model_name=name,
        phase="inference",
        batch_size=1,
        decode_output_length=0,
        actual_by_device=actual_by_device,
        prediction_by_device={
            device_name: ToolPrediction(
                node_name=name,
                device_name=device_name,
                gpu_name=device_name.split("_", maxsplit=1)[0],
                metrics=metrics,
            )
            for device_name, metrics in actual_by_device.items()
        },
        static_memory_gb=0.1,
    )


def test_gnn_aware_avoids_v100_oom_when_a100_fits() -> None:
    record = build_record(
        "heavy_tool",
        v100_memory_gb=10.0,
        a100_memory_gb=10.0,
    )
    sessions = [TraceSession(name="session_01", tools=[record])]

    baseline = simulate_strategy("v100_only", sessions)
    gnn_aware = simulate_strategy("gnn_aware", sessions)

    assert baseline.oom_events == 1
    assert gnn_aware.oom_events == 0
    assert gnn_aware.avg_tool_jct_sec < baseline.avg_tool_jct_sec


def test_simulation_counts_cross_session_cache_hits() -> None:
    record = build_record(
        "shared_tool",
        v100_memory_gb=1.0,
        a100_memory_gb=1.0,
        deploy_sec=4.0,
        run_sec=1.0,
    )
    sessions = [
        TraceSession(name="session_01", tools=[record]),
        TraceSession(name="session_02", tools=[record]),
    ]

    result = simulate_strategy("gnn_aware", sessions)

    assert result.cache_hits == 1
    assert result.cold_start_sec == 4.0
    assert result.avg_tool_jct_sec == 3.0


def test_build_markdown_report_includes_core_sections() -> None:
    payload = {
        "summary": {
            "availability_baseline": "v100_only",
            "latency_baseline": "round_robin",
            "makespan_improvement": 0.25,
            "latency_gap": -0.1,
            "gnn_oom_events": 0,
            "availability_baseline_oom_events": 2,
            "latency_baseline_oom_events": 4,
            "energy_change": 0.1,
        },
        "model": {
            "config_path": "config.yaml",
            "checkpoint_path": "best.pt",
            "scaler_dir": "scalers",
            "test_data_path": "test.pt",
        },
        "trace": {
            "record_count": 1,
            "session_count": 1,
            "tools_per_session": 1,
            "device_memory_gb": {"v100_slot": 8.0, "a100_slot": 14.0},
            "memory_safety_margin_gb": 0.5,
            "oom_retry_penalty_sec": 5.0,
        },
        "strategy_results": [
            {
                "strategy": "v100_only",
                "avg_session_makespan_sec": 5.0,
                "p95_session_makespan_sec": 5.0,
                "avg_tool_jct_sec": 5.0,
                "oom_events": 2,
                "retry_events": 2,
                "cache_hit_rate": 0.0,
                "cold_start_sec": 4.0,
                "energy_wh": 1.0,
                "avg_memory_utilization": 0.2,
            },
            {
                "strategy": "gnn_aware",
                "avg_session_makespan_sec": 4.0,
                "p95_session_makespan_sec": 4.0,
                "avg_tool_jct_sec": 4.0,
                "oom_events": 0,
                "retry_events": 0,
                "cache_hit_rate": 0.0,
                "cold_start_sec": 4.0,
                "energy_wh": 0.9,
                "avg_memory_utilization": 0.2,
            },
        ],
        "workflow_predictions": [],
        "external_references": [
            {"topic": "SCHEDTUNE", "use": "OOM", "url": "https://example.com"}
        ],
    }

    markdown = build_markdown_report(payload)

    assert "# Workflow GNN 调度评估 20260612" in markdown
    assert "## 对比结果" in markdown
    assert "SCHEDTUNE" in markdown
