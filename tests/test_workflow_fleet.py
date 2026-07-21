from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from langchain.agents import AgentState
from langchain_core.messages import AIMessage

from workflow.artifacts import SchedulerConfig
from workflow.fleet import WorkflowFleet
from workflow.schema import Workflow
from workflow.types import SessionState
from workflow.worker import message_text


def state(text: str) -> AgentState:
    return AgentState(messages=[AIMessage(content=text)])


def echo_workflow(workflow_name: str) -> Workflow:
    return Workflow.model_validate(
        {
            "workflow_name": workflow_name,
            "nodes": [{"name": "echo", "type": "function", "function": "echo"}],
            "edges": [],
        }
    )


def echo(
    inputs: Mapping[str, object],
    states: Mapping[str, AgentState],
    parameters: Mapping[str, object],
) -> AgentState:
    return state(str(inputs["value"]))


def test_two_workflows_share_one_scheduler_and_run_independently(
    ray_session: None,
    wait_until,
    tmp_path: Path,
) -> None:
    fleet = WorkflowFleet(
        scheduler_config=SchedulerConfig(),
        output_dir=tmp_path / "fleet",
        run_id="shared-run",
    )
    fleet.start()
    try:
        fleet.register_workflow(
            echo_workflow("workflow-a"),
            functions={"echo": echo},
            output_dir=tmp_path / "workflow-a",
        )
        fleet.register_workflow(
            echo_workflow("workflow-b"),
            functions={"echo": echo},
            output_dir=tmp_path / "workflow-b",
        )
        binding_a = fleet._binding("workflow-a")
        binding_b = fleet._binding("workflow-b")
        assert binding_a is not binding_b
        assert binding_a.result_store is not binding_b.result_store

        fleet.submit("workflow-a", "a-s1", {"value": "hello-a"})
        fleet.submit("workflow-b", "b-s1", {"value": "hello-b"})

        assert wait_until(
            lambda: fleet.has_result("workflow-a", "a-s1"), timeout_sec=10.0
        )
        assert wait_until(
            lambda: fleet.has_result("workflow-b", "b-s1"), timeout_sec=10.0
        )
        assert message_text(fleet.get_result("workflow-a", "a-s1")) == "hello-a"
        assert message_text(fleet.get_result("workflow-b", "b-s1")) == "hello-b"

        fleet.drain_workflow("workflow-a")
        fleet.drain_workflow("workflow-b")
    finally:
        fleet.shutdown()

    for name in ("workflow_trace.jsonl", "run_summary.json"):
        assert (tmp_path / "fleet" / name).is_file()
    for workflow_dir in ("workflow-a", "workflow-b"):
        assert (tmp_path / workflow_dir / "session_results.jsonl").is_file()


def test_registration_failure_does_not_affect_already_registered_workflow(
    ray_session: None,
    wait_until,
    tmp_path: Path,
) -> None:
    fleet = WorkflowFleet(
        scheduler_config=SchedulerConfig(),
        output_dir=tmp_path / "fleet",
        run_id="shared-run",
    )
    fleet.start()
    try:
        fleet.register_workflow(
            echo_workflow("workflow-a"),
            functions={"echo": echo},
            output_dir=tmp_path / "workflow-a",
        )

        try:
            fleet.register_workflow(
                echo_workflow("workflow-b"),
                functions={},  # missing "echo" registration -> must raise
                output_dir=tmp_path / "workflow-b",
            )
        except KeyError as error:
            assert "echo" in str(error)
        else:
            raise AssertionError("expected registration to fail")

        assert "workflow-b" not in fleet._bindings

        fleet.submit("workflow-a", "a-s1", {"value": "still-alive"})
        assert wait_until(
            lambda: fleet.has_result("workflow-a", "a-s1"), timeout_sec=10.0
        )
        assert message_text(fleet.get_result("workflow-a", "a-s1")) == "still-alive"

        fleet.drain_workflow("workflow-a")
    finally:
        fleet.shutdown()


def test_draining_one_workflow_does_not_disturb_another(
    ray_session: None,
    wait_until,
    tmp_path: Path,
) -> None:
    fleet = WorkflowFleet(
        scheduler_config=SchedulerConfig(),
        output_dir=tmp_path / "fleet",
        run_id="shared-run",
    )
    fleet.start()
    try:
        fleet.register_workflow(
            echo_workflow("workflow-a"),
            functions={"echo": echo},
            output_dir=tmp_path / "workflow-a",
        )
        fleet.register_workflow(
            echo_workflow("workflow-b"),
            functions={"echo": echo},
            output_dir=tmp_path / "workflow-b",
        )

        fleet.submit("workflow-a", "a-s1", {"value": "a-first"})
        assert wait_until(
            lambda: fleet.has_result("workflow-a", "a-s1"), timeout_sec=10.0
        )

        fleet.drain_workflow("workflow-a")

        assert fleet.get_session_state("workflow-a", "a-s1") == SessionState.COMPLETED
        assert "workflow-a" not in fleet._bindings

        # workflow-b, never drained, must still be fully usable after
        # workflow-a's independent drain (shared scheduler/GPU pool intact).
        fleet.submit("workflow-b", "b-s1", {"value": "b-after-a-drained"})
        assert wait_until(
            lambda: fleet.has_result("workflow-b", "b-s1"), timeout_sec=10.0
        )
        assert (
            message_text(fleet.get_result("workflow-b", "b-s1"))
            == "b-after-a-drained"
        )

        fleet.drain_workflow("workflow-b")
    finally:
        fleet.shutdown()


def test_unknown_workflow_name_is_rejected(ray_session: None, tmp_path: Path) -> None:
    fleet = WorkflowFleet(
        scheduler_config=SchedulerConfig(),
        output_dir=tmp_path / "fleet",
        run_id="shared-run",
    )
    fleet.start()
    try:
        try:
            fleet.submit("missing", "s1", {})
        except KeyError as error:
            assert "missing" in str(error)
        else:
            raise AssertionError("expected a KeyError for an unregistered workflow")
    finally:
        fleet.shutdown()
