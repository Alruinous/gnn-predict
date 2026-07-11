from __future__ import annotations

from workflow.baseline.execution import execute_baseline_node
from workflow.baseline.runner import BaselineRunner
from workflow.baseline.state import (
    BaselineState,
    initial_baseline_state,
    merge_node_outputs,
)
from workflow.baseline.types import BaselineSessionRequest, BaselineWorkflow


def run_sequential_workflow(
    workflow: BaselineWorkflow,
    request: BaselineSessionRequest,
    runner: BaselineRunner,
) -> BaselineState:
    runner.load_workflow(workflow)
    state = initial_baseline_state(request)
    dependencies = workflow.dependencies()
    node_map = workflow.node_map()
    terminal_name = workflow.terminal_name()

    for node_name in workflow.topological_order():
        update = execute_baseline_node(
            node_map[node_name],
            dependencies[node_name],
            state,
            runner,
            node_name == terminal_name,
        )
        state["node_outputs"] = merge_node_outputs(
            state["node_outputs"], update["node_outputs"]
        )
        state["trace"].extend(update["trace"])
        if "final_output" in update:
            state["final_output"] = update["final_output"]

    return state
