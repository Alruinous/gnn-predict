from __future__ import annotations

from pathlib import Path

from langchain.agents import AgentState
from langchain_core.messages import AIMessage, HumanMessage

from workflow.controller import WorkflowController
from workflow.types import WorkerState

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "config/workflow/runtime_smoke_20260702/fan_in_smoke.yaml"


class EchoAgent:
    def invoke(self, state: AgentState) -> AgentState:
        text = state["messages"][-1].content
        return AgentState(messages=[AIMessage(content=f"echo({text})")])


def echo_agent_factory(config, system_prompt, tools, device):
    return EchoAgent()


def test_workflow_controller_runs_fan_in_smoke_workflow_end_to_end(
    ray_session, wait_until
):
    controller = WorkflowController.from_yaml(
        str(SMOKE_CONFIG), agent_factory=echo_agent_factory
    )
    controller.start_workflow()

    session_ids = ["s1", "s2"]
    for session_id in session_ids:
        controller.submit(
            session_id=session_id,
            item_id=f"{session_id}-item",
            message=AgentState(messages=[HumanMessage(content=f"doc-{session_id}")]),
        )

    assert wait_until(
        lambda: all(controller.is_session_complete(s) for s in session_ids),
        timeout_sec=120,
    )
    for session_id in session_ids:
        results = controller.get_session_results(session_id)
        assert set(results) == {"reducer"}
        text = results["reducer"]["messages"][-1].content
        assert "part A" in text
        assert "part B" in text
        assert f"doc-{session_id}" in text

    status = controller.get_workflow_status()
    assert status.failures == []
    assert all(state == WorkerState.RUNNING for state in status.node_states.values())

    controller.stop_workflow()
    assert wait_until(
        lambda: all(
            state == WorkerState.STOPPED
            for state in controller.get_workflow_status().node_states.values()
        )
    )
