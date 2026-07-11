from __future__ import annotations

import json
import time

from workflow.baseline.runner import BaselineRunner
from workflow.baseline.state import BaselineState, BaselineStateUpdate
from workflow.baseline.types import BaselineNodeConfig, BaselineTraceEvent


def execute_baseline_node(
    node: BaselineNodeConfig,
    dependencies: list[str],
    state: BaselineState,
    runner: BaselineRunner,
    is_terminal: bool,
) -> BaselineStateUpdate:
    started_at = time.perf_counter()
    output = _run_node(node, dependencies, state, runner)
    ended_at = time.perf_counter()
    event = BaselineTraceEvent(
        session_id=state["session_id"],
        node_name=node.name,
        node_type=node.type,
        started_at=started_at,
        ended_at=ended_at,
        duration_sec=ended_at - started_at,
        output_text=output,
    )
    update = BaselineStateUpdate(
        node_outputs={node.name: output},
        trace=[event.model_dump(mode="python")],
    )
    if is_terminal:
        update["final_output"] = output
    return update


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
