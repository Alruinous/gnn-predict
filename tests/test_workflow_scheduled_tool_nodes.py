from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from torch_geometric.data import Data

from workflow.gnn_predictor import ToolPrediction
from workflow.scheduled_tool_nodes import ScheduledToolNode
from workflow.scheduler import (
    WorkflowDeviceState,
    WorkflowScheduler,
    WorkflowSchedulerConfig,
    WorkflowSchedulingError,
)
from workflow.schema import Workflow, WorkflowNodeConfig


def build_tool_workflow() -> Workflow:
    return Workflow.model_validate(
        {
            "nodes": [
                {"name": "input", "type": "input"},
                {
                    "name": "detector",
                    "type": "tool",
                    "task": "object_detection",
                    "model": {
                        "name": "yolov8n",
                        "parameters": {
                            "input_channels": 3,
                            "output_classes": 80,
                        },
                    },
                    "runtime": {
                        "batch_size": 1,
                        "input_shape": [1, 3, 224, 224],
                        "phase": "inference",
                    },
                },
                {"name": "output", "type": "output"},
            ],
            "edges": [
                {"source": "input", "target": "detector", "attributes": {}},
                {"source": "detector", "target": "output", "attributes": {}},
            ],
        }
    )


class RecordingHandler:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        node: WorkflowNodeConfig,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(context)
        return {"node": node.name}


class FakeExporter:
    def __init__(self) -> None:
        self.export_dirs: list[Path] = []

    def export_onnx(self, node: WorkflowNodeConfig, output_dir: Path) -> Path:
        self.export_dirs.append(output_dir)
        export_path = output_dir / f"{node.name}.onnx"
        export_path.write_bytes(b"fake")
        return export_path


class FakePredictor:
    def __init__(self, memory_gb: float) -> None:
        self.memory_gb = memory_gb

    def predict_graph(
        self,
        *,
        node_name: str,
        device_name: str,
        gpu_name: str,
        graph: Data,
    ) -> ToolPrediction:
        return ToolPrediction(
            node_name=node_name,
            device_name=device_name,
            gpu_name=gpu_name,
            metrics={
                "deployment_duration_sec_avg": 2.0,
                "run_duration_sec_avg": 1.0,
                "memory_delta_gb_max": self.memory_gb,
                "gpu_mem_used_mb_max": self.memory_gb * 1024.0,
                "gpu_power_watts_avg": 100.0,
            },
        )


def build_scheduler(available_memory_gb: float) -> WorkflowScheduler:
    return WorkflowScheduler(
        WorkflowSchedulerConfig(
            devices=[
                WorkflowDeviceState(
                    name="gpu0",
                    gpu_name="v100",
                    total_memory_gb=32.0,
                    available_memory_gb=available_memory_gb,
                )
            ],
            memory_safety_margin_gb=0.5,
        )
    )


def test_scheduled_tool_node_injects_schedule_and_cleans_temp_onnx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "workflow.scheduled_tool_nodes.build_workflow_graph_from_onnx",
        lambda *_args: Data(),
    )
    workflow = build_tool_workflow()
    node = workflow.node_map()["detector"]
    handler = RecordingHandler()
    exporter = FakeExporter()
    tool_node = ScheduledToolNode(
        node=node,
        handler=handler,
        scheduler=build_scheduler(available_memory_gb=8.0),
        predictor=FakePredictor(memory_gb=2.0),
        exporter=exporter,
    )

    result = tool_node.run(task="detect", input_path="/tmp/image.jpg", payload={})

    assert "status: succeeded" in result
    assert handler.calls[0]["workflow_schedule"]["device_name"] == "gpu0"
    assert all(not export_dir.exists() for export_dir in exporter.export_dirs)


def test_scheduled_tool_node_raises_when_prediction_exceeds_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "workflow.scheduled_tool_nodes.build_workflow_graph_from_onnx",
        lambda *_args: Data(),
    )
    workflow = build_tool_workflow()
    node = workflow.node_map()["detector"]
    handler = RecordingHandler()

    tool_node = ScheduledToolNode(
        node=node,
        handler=handler,
        scheduler=build_scheduler(available_memory_gb=1.0),
        predictor=FakePredictor(memory_gb=2.0),
        exporter=FakeExporter(),
    )

    with pytest.raises(WorkflowSchedulingError, match="enough memory"):
        tool_node.run(task="detect")

    assert handler.calls == []
