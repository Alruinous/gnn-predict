from __future__ import annotations

from workflow.gnn_predictor import ToolPrediction
from workflow.scheduler import WorkflowDeviceState, WorkflowScheduler, WorkflowSchedulerConfig


def build_prediction(node_name: str, memory_gb: float) -> ToolPrediction:
    return ToolPrediction(
        node_name=node_name,
        device_name="gpu0",
        gpu_name="v100",
        metrics={
            "deployment_duration_sec_avg": 1.0,
            "run_duration_sec_avg": 1.0,
            "memory_delta_gb_max": memory_gb,
            "gpu_mem_used_mb_max": memory_gb * 1024.0,
            "gpu_power_watts_avg": 100.0,
        },
    )


def test_plan_ready_tools_splits_batches_by_memory() -> None:
    scheduler = WorkflowScheduler(
        WorkflowSchedulerConfig(
            devices=[
                WorkflowDeviceState(
                    name="gpu0",
                    gpu_name="v100",
                    total_memory_gb=32.0,
                    available_memory_gb=4.5,
                )
            ],
            memory_safety_margin_gb=0.5,
        )
    )

    batches = scheduler.plan_ready_tools(
        {
            "tool_a": [build_prediction("tool_a", 2.0)],
            "tool_b": [build_prediction("tool_b", 2.0)],
        }
    )

    assert [[decision.node_name for decision in batch] for batch in batches] == [
        ["tool_a"],
        ["tool_b"],
    ]
