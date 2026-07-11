from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from workflow.baseline.execution import execute_baseline_node
from workflow.baseline.runner import BaselineRunner
from workflow.baseline.state import (
    BaselineState,
    BaselineStateUpdate,
    initial_baseline_state,
)
from workflow.baseline.types import (
    BaselineNodeConfig,
    BaselineSessionRequest,
    BaselineWorkflow,
)


def build_bsp_graph(
    workflow: BaselineWorkflow,
    runner: BaselineRunner,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    dependencies = workflow.dependencies()
    terminal_name = workflow.terminal_name()
    graph = StateGraph(cast(Any, BaselineState))

    for node in workflow.nodes:
        graph.add_node(
            node.name,
            _build_node_action(
                node,
                dependencies[node.name],
                runner,
                node.name == terminal_name,
            ),
        )
    for node_name in workflow.entry_names():
        graph.add_edge(START, node_name)
    for target_name, source_names in dependencies.items():
        if len(source_names) == 1:
            graph.add_edge(source_names[0], target_name)
        elif len(source_names) > 1:
            graph.add_edge(source_names, target_name)
    graph.add_edge(terminal_name, END)
    return cast(CompiledStateGraph[Any, Any, Any, Any], graph.compile())


def run_bsp_workflow(
    workflow: BaselineWorkflow,
    request: BaselineSessionRequest,
    runner: BaselineRunner,
) -> BaselineState:
    runner.load_workflow(workflow)
    graph = build_bsp_graph(workflow, runner)
    return _normalize_state(graph.invoke(initial_baseline_state(request)))


def run_bsp_workflow_batch(
    workflow: BaselineWorkflow,
    requests: list[BaselineSessionRequest],
    runner: BaselineRunner,
    *,
    max_concurrency: int,
) -> list[BaselineState]:
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    if not requests:
        return []

    runner.load_workflow(workflow)
    graph = build_bsp_graph(workflow, runner)
    results = graph.batch(
        [initial_baseline_state(request) for request in requests],
        config={"max_concurrency": max_concurrency},
    )
    return [_normalize_state(result) for result in results]


def _build_node_action(
    node: BaselineNodeConfig,
    dependencies: list[str],
    runner: BaselineRunner,
    is_terminal: bool,
) -> Any:
    def action(state: BaselineState) -> BaselineStateUpdate:
        return execute_baseline_node(
            node,
            dependencies,
            state,
            runner,
            is_terminal,
        )

    return action


def _normalize_state(result: object) -> BaselineState:
    assert isinstance(result, dict), type(result)
    return cast(BaselineState, result)
