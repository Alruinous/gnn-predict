from __future__ import annotations

import json
import operator
import time
from typing import Annotated, Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from workflow.baseline.runner import BaselineRunner
from workflow.baseline.types import (
    BaselineNodeConfig,
    BaselineTraceEvent,
    BaselineWorkflow,
)


def merge_node_outputs(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    overlap = set(left) & set(right)
    assert not overlap, overlap
    merged = dict(left)
    merged.update(right)
    return merged


class BaselineState(TypedDict):
    inputs: dict[str, str]
    node_outputs: Annotated[dict[str, str], merge_node_outputs]
    trace: Annotated[list[dict[str, Any]], operator.add]
    final_output: str | None


def build_baseline_graph(
    workflow: BaselineWorkflow,
    runner: BaselineRunner,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    dependencies = workflow.dependencies()
    adjacency = workflow.adjacency()
    entry_nodes = [name for name in workflow.node_names() if not dependencies[name]]
    terminal_nodes = [name for name in workflow.node_names() if not adjacency[name]]

    graph = StateGraph(cast(Any, BaselineState))
    for node in workflow.nodes:
        graph.add_node(
            node.name,
            _build_node_action(
                node,
                dependencies[node.name],
                runner,
                node.name in terminal_nodes,
            ),
        )
    assert entry_nodes, workflow.node_names()
    assert terminal_nodes, workflow.node_names()

    for node_name in entry_nodes:
        graph.add_edge(START, node_name)
    for target_name in workflow.node_names():
        source_names = dependencies[target_name]
        if len(source_names) == 1:
            graph.add_edge(source_names[0], target_name)
        elif len(source_names) > 1:
            graph.add_edge(source_names, target_name)
    for node_name in terminal_nodes:
        graph.add_edge(node_name, END)

    return cast(CompiledStateGraph[Any, Any, Any, Any], graph.compile())


def run_baseline_workflow(
    workflow: BaselineWorkflow,
    inputs: dict[str, str],
    runner: BaselineRunner,
) -> BaselineState:
    runner.load_workflow(workflow)
    graph = build_baseline_graph(workflow, runner)
    result = graph.invoke(
        {
            "inputs": inputs,
            "node_outputs": {},
            "trace": [],
            "final_output": None,
        }
    )
    assert isinstance(result, dict), type(result)
    return cast(BaselineState, result)


def _build_node_action(
    node: BaselineNodeConfig,
    dependencies: list[str],
    runner: BaselineRunner,
    is_terminal: bool,
) -> Any:
    def action(state: BaselineState) -> dict[str, Any]:
        started_at = time.perf_counter()
        output = _run_node(node, dependencies, state, runner)
        ended_at = time.perf_counter()
        event = BaselineTraceEvent(
            node_name=node.name,
            node_type=node.type,
            started_at=started_at,
            ended_at=ended_at,
            duration_sec=ended_at - started_at,
            output_text=output,
        )
        update: dict[str, Any] = {
            "node_outputs": {node.name: output},
            "trace": [event.model_dump()],
        }
        if node.type == "output" or is_terminal:
            update["final_output"] = output
        return update

    return action


def _run_node(
    node: BaselineNodeConfig,
    dependencies: list[str],
    state: BaselineState,
    runner: BaselineRunner,
) -> str:
    if node.type == "input":
        if "input_text" in state["inputs"]:
            return state["inputs"]["input_text"]
        return json.dumps(state["inputs"], ensure_ascii=False, sort_keys=True)

    if node.type in ("evaluator", "output"):
        return _dependency_output(dependencies, state)

    assert node.type in ("agent", "tool"), node.type
    assert node.prompt_template is not None
    prompt = node.prompt_template.format(**_prompt_context(dependencies, state))
    return runner.run_node(node, prompt)


def _dependency_output(dependencies: list[str], state: BaselineState) -> str:
    assert dependencies, dependencies
    if len(dependencies) == 1:
        return state["node_outputs"][dependencies[0]]
    return "\n\n".join(state["node_outputs"][source] for source in dependencies)


def _prompt_context(
    dependencies: list[str],
    state: BaselineState,
) -> dict[str, str]:
    outputs = state["node_outputs"]
    for source in dependencies:
        assert source in outputs, source

    context = dict(state["inputs"])
    dependency_outputs = {source: outputs[source] for source in dependencies}
    context.update(dependency_outputs)
    context["node_outputs_json"] = json.dumps(
        dependency_outputs,
        ensure_ascii=False,
        sort_keys=True,
    )
    if dependencies:
        context["previous_output"] = _dependency_output(dependencies, state)
    return context
