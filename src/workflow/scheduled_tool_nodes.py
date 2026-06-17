from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from workflow.gnn_predictor import WorkflowPredictionProvider
from workflow.handlers import NodeHandler
from workflow.model_export import (
    GnnArchsWorkflowModelExporter,
    WorkflowModelExporter,
    build_workflow_graph_from_onnx,
)
from workflow.scheduler import ToolScheduleDecision, WorkflowScheduler
from workflow.schema import Workflow, WorkflowNodeConfig, WorkflowNodeResult
from workflow.tool_nodes import BaseToolNode
from workflow.types import WorkflowContext


class ScheduledToolNode(BaseToolNode):
    def __init__(
        self,
        node: WorkflowNodeConfig,
        handler: NodeHandler,
        scheduler: WorkflowScheduler,
        predictor: WorkflowPredictionProvider,
        context: WorkflowContext | None = None,
        exporter: WorkflowModelExporter | None = None,
    ) -> None:
        super().__init__(node=node, handler=handler, context=context)
        self.scheduler = scheduler
        self.predictor = predictor
        self.exporter = exporter or GnnArchsWorkflowModelExporter()

    def run(
        self,
        task: str,
        input_path: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> str:
        started_at = time.time()
        node_input = self.prepare_node_input(task, input_path, payload or {})
        decision = self.schedule()
        node_input["workflow_schedule"] = decision.model_dump(mode="json")
        self.ensure_deployed(decision)
        try:
            metadata = self.handler.run(self.node, node_input)
        except Exception as exc:
            ended_at = time.time()
            node_result = WorkflowNodeResult(
                node_name=self.node.name,
                status="failed",
                started_at=started_at,
                ended_at=ended_at,
                duration_sec=ended_at - started_at,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return self.format_node_output(node_result)

        ended_at = time.time()
        node_result = WorkflowNodeResult(
            node_name=self.node.name,
            status="succeeded",
            started_at=started_at,
            ended_at=ended_at,
            duration_sec=ended_at - started_at,
            metadata={
                "workflow_schedule": decision.model_dump(mode="json"),
                **metadata,
            },
        )
        return self.format_node_output(node_result)

    def schedule(self) -> ToolScheduleDecision:
        predictions = []
        with tempfile.TemporaryDirectory() as temp_dir:
            onnx_path = self.exporter.export_onnx(self.node, Path(temp_dir))
            for device in self.scheduler.devices.values():
                graph = build_workflow_graph_from_onnx(
                    self.node,
                    onnx_path,
                    device.gpu_name,
                )
                predictions.append(
                    self.predictor.predict_graph(
                        node_name=self.node.name,
                        device_name=device.name,
                        gpu_name=device.gpu_name,
                        graph=graph,
                    )
                )
        return self.scheduler.select_device(self.node, predictions)

    def ensure_deployed(self, decision: ToolScheduleDecision) -> None:
        self.scheduler.mark_deployed(self.node, decision)


def build_scheduled_tool_nodes(
    workflow: Workflow,
    handler: NodeHandler,
    scheduler: WorkflowScheduler,
    predictor: WorkflowPredictionProvider,
    context: WorkflowContext | None = None,
    exporter: WorkflowModelExporter | None = None,
) -> list[ScheduledToolNode]:
    tool_nodes: list[ScheduledToolNode] = []
    for node in workflow.nodes:
        if node.type != "tool":
            continue
        tool_nodes.append(
            ScheduledToolNode(
                node=node,
                handler=handler,
                scheduler=scheduler,
                predictor=predictor,
                context=context,
                exporter=exporter,
            )
        )
    return tool_nodes
