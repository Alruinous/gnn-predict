from __future__ import annotations

import pytest

from workflow.agent import build_workflow_agent
from workflow.schema import Workflow, WorkflowNodeConfig, WorkflowNodeResult
from workflow.tool_nodes import BaseToolNode, build_tool_nodes
from workflow.types import WorkflowContext


def build_workflow() -> Workflow:
    return Workflow.model_validate(
        {
            "nodes": [
                {"name": "input", "type": "input"},
                {
                    "name": "main_llm",
                    "type": "agent",
                    "task": "react_agent",
                    "model": {"name": "any-main-model", "parameters": {}},
                    "runtime": {
                        "batch_size": 1,
                        "sequence_length": 128,
                        "phase": "prefill",
                    },
                },
                {
                    "name": "cv_tool",
                    "type": "tool",
                    "task": "object_detection",
                    "description": "Use this detector when an image needs object localization.",
                    "model": {
                        "name": "my-private/cv-model@2026",
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
                {"source": "input", "target": "main_llm", "attributes": {}},
                {"source": "main_llm", "target": "cv_tool", "attributes": {}},
                {"source": "cv_tool", "target": "output", "attributes": {}},
            ],
        }
    )


class RecordingHandler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def run(
        self,
        node: WorkflowNodeConfig,
        context: dict[str, object],
    ) -> dict[str, object]:
        self.calls.append((node.name, context))
        return {
            **context,
            "node": node.name,
            "model": node.model.name if node.model else None,
        }


class ImageToolNode(BaseToolNode):
    def prepare_node_input(
        self,
        task: str,
        input_path: str | None,
        payload: dict[str, object],
    ) -> WorkflowContext:
        node_input = super().prepare_node_input(task, input_path, payload)
        node_input["image_path"] = input_path
        node_input["preprocess"] = "resize"
        return node_input

    def format_node_output(self, node_result: WorkflowNodeResult) -> str:
        return f"image_result={node_result.metadata['image_path']}"


def test_build_tool_nodes_uses_workflow_tool_nodes() -> None:
    workflow = build_workflow()
    handler = RecordingHandler()

    tool_nodes = build_tool_nodes(workflow, handler=handler)
    tools = [tool_node.as_tool() for tool_node in tool_nodes]

    assert [tool.name for tool in tools] == ["cv_tool"]
    assert tools[0].description == (
        "Use this detector when an image needs object localization."
    )


def test_build_tool_nodes_falls_back_to_generated_description() -> None:
    workflow = build_workflow()
    handler = RecordingHandler()
    cv_node = workflow.node_map()["cv_tool"].model_copy(
        update={"description": None}
    )
    workflow = workflow.model_copy(
        update={
            "nodes": [
                cv_node if node.name == cv_node.name else node
                for node in workflow.nodes
            ]
        }
    )

    tool = build_tool_nodes(workflow, handler=handler)[0].as_tool()

    assert "my-private/cv-model@2026" in tool.description
    assert "object_detection" in tool.description


def test_base_tool_node_calls_handler_with_generic_task_context() -> None:
    workflow = build_workflow()
    handler = RecordingHandler()
    tool_node = build_tool_nodes(
        workflow,
        handler=handler,
        context={"request_id": "req-1"},
    )[0]
    tool = tool_node.as_tool()

    result = tool.invoke(
        {
            "task": "describe image",
            "input_path": "/tmp/image.jpg",
            "payload": {"threshold": 0.5},
        }
    )

    assert "status: succeeded" in result
    assert "model: my-private/cv-model@2026" in result
    assert handler.calls[0][0] == "cv_tool"
    assert handler.calls[0][1]["request_id"] == "req-1"
    assert handler.calls[0][1]["task"] == "describe image"
    assert handler.calls[0][1]["input_path"] == "/tmp/image.jpg"
    assert handler.calls[0][1]["payload"] == {"threshold": 0.5}


def test_subclass_tool_node_overrides_input_and_output_hooks() -> None:
    workflow = build_workflow()
    handler = RecordingHandler()
    tool_node = ImageToolNode(workflow.node_map()["cv_tool"], handler=handler)
    tool = tool_node.as_tool()

    result = tool.invoke(
        {
            "task": "classify image",
            "input_path": "/tmp/image.jpg",
            "payload": {},
        }
    )

    assert result == "image_result=/tmp/image.jpg"
    assert handler.calls[0][1]["image_path"] == "/tmp/image.jpg"
    assert handler.calls[0][1]["preprocess"] == "resize"


def test_build_workflow_agent_compiles_without_network(monkeypatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_NAME", "deepseek/deepseek-v4-flash")
    handler = RecordingHandler()

    agent = build_workflow_agent(build_workflow(), handler=handler)

    assert agent is not None


def test_build_workflow_agent_requires_handler_without_custom_tools() -> None:
    with pytest.raises(ValueError, match="handler is required"):
        build_workflow_agent(build_workflow())
