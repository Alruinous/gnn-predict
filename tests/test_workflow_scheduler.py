from __future__ import annotations

from typing import Any, Literal

import pytest
from pydantic import ValidationError

from workflow.scheduler import (
    AgentTaskRuntimeReport,
    CancelSessionAction,
    CompleteDecision,
    FunctionTaskRuntimeReport,
    OutputReport,
    SchedulerCore,
    TaskCancelledError,
)
from workflow.schema import Workflow
from workflow.types import NodeTaskState, SessionState


def function_workflow() -> Workflow:
    return Workflow.model_validate(
        {
            "nodes": [
                {
                    "name": "entry",
                    "type": "function",
                    "function": "prepare",
                },
                {
                    "name": "terminal",
                    "type": "function",
                    "function": "finalize",
                },
            ],
            "edges": [{"source": "entry", "target": "terminal"}],
        }
    )


def branched_workflow() -> Workflow:
    return Workflow.model_validate(
        {
            "nodes": [
                {"name": "start", "type": "function", "function": "start"},
                {"name": "left", "type": "function", "function": "left"},
                {"name": "right", "type": "function", "function": "right"},
                {"name": "end", "type": "function", "function": "end"},
            ],
            "edges": [
                {"source": "start", "target": "left"},
                {"source": "start", "target": "right"},
                {"source": "left", "target": "end"},
                {"source": "right", "target": "end"},
            ],
        }
    )


def agent_workflow() -> Workflow:
    return Workflow.model_validate(
        {
            "nodes": [
                {
                    "name": "agent",
                    "type": "agent",
                    "model": {"name": "test-model"},
                    "execution": {
                        "model_path": "/models/test-model",
                        "max_new_tokens": 16,
                        "serving": {
                            "max_model_len": 1024,
                            "max_num_seqs": 1,
                            "max_num_batched_tokens": 1024,
                        },
                    },
                    "prompt_template": "{content}",
                }
            ],
            "edges": [],
        }
    )


def function_report(
    core: SchedulerCore,
    task_id: str,
    *,
    status: Literal["success", "failed"] = "success",
) -> FunctionTaskRuntimeReport:
    task = core.tasks[task_id]
    return FunctionTaskRuntimeReport(
        task_id=task_id,
        session_id=task.session_id,
        node_id=task.node_id,
        input_item_ids=list(task.input_item_ids),
        started_at=10.0,
        finished_at=12.0,
        duration_sec=2.0,
        status=status,
        error_type=None if status == "success" else "RuntimeError",
        error_message=None if status == "success" else "execution failed",
    )


def output_report(
    task_id: str,
    *,
    terminal: bool = False,
) -> OutputReport:
    return OutputReport(
        task_id=task_id,
        output_item_ids=[f"output-{task_id}"],
        persisted_terminal_result=terminal,
    )


def complete_and_finish(core: SchedulerCore, task_id: str) -> None:
    assert core.complete(task_id, function_report(core, task_id)).emit_output
    node_id = core.tasks[task_id].node_id
    core.finish_node(
        task_id,
        output_report(task_id, terminal=node_id == core.workflow.graph.terminal_node),
    )


def test_function_task_completes_only_after_output_emission() -> None:
    core = SchedulerCore(function_workflow())
    core.register_session("s1")
    task_id = core.begin_node("s1", "entry", ["item-1"])

    decision = core.complete(task_id, function_report(core, task_id))

    assert decision == CompleteDecision(emit_output=True)
    assert core.tasks[task_id].state == NodeTaskState.EMITTING
    assert core.sessions["s1"].state == SessionState.ACTIVE
    core.finish_node(task_id, output_report(task_id))
    assert core.tasks[task_id].state == NodeTaskState.COMPLETED


def test_terminal_function_completion_marks_session_completed() -> None:
    core = SchedulerCore(function_workflow())
    core.register_session("s1")
    entry_id = core.begin_node("s1", "entry", ["item-1"])
    complete_and_finish(core, entry_id)
    terminal_id = core.begin_node("s1", "terminal", ["output-entry"])

    complete_and_finish(core, terminal_id)

    assert core.tasks[terminal_id].state == NodeTaskState.COMPLETED
    assert core.sessions["s1"].state == SessionState.COMPLETED
    assert core.drain_complete()


def test_terminal_finish_requires_persisted_result() -> None:
    core = SchedulerCore(function_workflow())
    core.register_session("s1")
    entry_id = core.begin_node("s1", "entry", ["item-1"])
    complete_and_finish(core, entry_id)
    terminal_id = core.begin_node("s1", "terminal", ["output-entry"])
    core.complete(terminal_id, function_report(core, terminal_id))

    with pytest.raises(ValueError, match="persisted"):
        core.finish_node(terminal_id, output_report(terminal_id))

    assert core.tasks[terminal_id].state == NodeTaskState.EMITTING
    assert core.sessions["s1"].state == SessionState.ACTIVE


def test_register_and_begin_reject_invalid_lifecycle_operations() -> None:
    core = SchedulerCore(function_workflow())
    core.register_session("s1")

    with pytest.raises(ValueError, match="already registered"):
        core.register_session("s1")
    with pytest.raises(KeyError, match="unknown session"):
        core.begin_node("missing", "entry", ["item-1"])
    with pytest.raises(KeyError, match="unknown node"):
        core.begin_node("s1", "missing", ["item-1"])

    task_id = core.begin_node("s1", "entry", ["item-1"])
    with pytest.raises(ValueError, match="already begun"):
        core.begin_node("s1", "entry", ["item-2"])
    with pytest.raises(ValueError, match="dependencies"):
        core.begin_node("s1", "terminal", ["item-2"])

    core.fail_node(task_id, "Stopped", "session stopped")
    with pytest.raises(ValueError, match="not active"):
        core.begin_node("s1", "terminal", ["item-2"])


def test_begin_accepts_dependency_that_is_emitting_its_output() -> None:
    core = SchedulerCore(function_workflow())
    core.register_session("s1")
    entry_id = core.begin_node("s1", "entry", ["item-1"])
    core.complete(entry_id, function_report(core, entry_id))

    terminal_id = core.begin_node("s1", "terminal", ["output-entry"])

    assert core.tasks[terminal_id].state == NodeTaskState.RUNNING


def test_agent_acquire_can_be_polled_cancelled_and_requested_again() -> None:
    core = SchedulerCore(agent_workflow())
    core.register_session("s1")
    task_id = core.begin_node("s1", "agent", ["item-1"])

    assert core.tasks[task_id].state == NodeTaskState.ACQUIRING
    acquire_id = core.request_acquire(task_id, input_tokens=64, created_at=3.5)
    pending = core.pending_acquires[acquire_id]
    assert pending.task_id == task_id
    assert pending.input_tokens == 64
    assert pending.created_at == 3.5
    assert core.poll_grant(acquire_id) is None

    with pytest.raises(ValueError, match="pending acquire"):
        core.request_acquire(task_id, input_tokens=64, created_at=4.0)
    core.cancel_acquire(acquire_id)
    assert acquire_id not in core.pending_acquires
    replacement_id = core.request_acquire(task_id, input_tokens=32, created_at=4.0)
    assert replacement_id in core.pending_acquires


def test_acquire_rejects_invalid_task_state_and_identifiers() -> None:
    function_core = SchedulerCore(function_workflow())
    function_core.register_session("s1")
    function_task_id = function_core.begin_node("s1", "entry", ["item-1"])

    with pytest.raises(ValueError, match="agent task"):
        function_core.request_acquire(function_task_id, input_tokens=8, created_at=1.0)
    with pytest.raises(ValueError):
        function_core.request_acquire(function_task_id, input_tokens=0, created_at=1.0)
    with pytest.raises(KeyError, match="unknown acquire"):
        function_core.cancel_acquire("missing")
    with pytest.raises(KeyError, match="unknown acquire"):
        function_core.poll_grant("missing")


def test_complete_rejects_agent_reports_until_resource_lifecycle_exists() -> None:
    core = SchedulerCore(agent_workflow())
    core.register_session("s1")
    task_id = core.begin_node("s1", "agent", ["item-1"])
    acquire_id = core.request_acquire(task_id, input_tokens=64, created_at=1.0)
    report = AgentTaskRuntimeReport(
        acquire_id=acquire_id,
        task_id=task_id,
        session_id="s1",
        node_id="agent",
        input_item_ids=["item-1"],
        model_key="test-model",
        accelerator_id="node/a100:0",
        gpu_kind="a100",
        input_tokens=64,
        max_new_tokens=16,
        output_tokens=8,
        hit_token_limit=False,
        started_at=1.0,
        finished_at=2.0,
        duration_sec=1.0,
        status="success",
    )

    with pytest.raises(NotImplementedError, match="agent"):
        core.complete(task_id, report, acquire_id=acquire_id)


def test_complete_validates_function_report_identity_and_state() -> None:
    core = SchedulerCore(function_workflow())
    core.register_session("s1")
    task_id = core.begin_node("s1", "entry", ["item-1"])
    mismatched = function_report(core, task_id).model_copy(
        update={"session_id": "other"}
    )

    with pytest.raises(ValueError, match="does not match"):
        core.complete(task_id, mismatched)

    core.complete(task_id, function_report(core, task_id))
    with pytest.raises(ValueError, match="running"):
        core.complete(task_id, function_report(core, task_id))
    with pytest.raises(KeyError, match="unknown task"):
        core.finish_node("missing", output_report(task_id))


def test_failed_function_report_fails_task_and_session() -> None:
    core = SchedulerCore(function_workflow())
    core.register_session("s1")
    task_id = core.begin_node("s1", "entry", ["item-1"])

    decision = core.complete(
        task_id,
        function_report(core, task_id, status="failed"),
    )

    assert decision == CompleteDecision(emit_output=False)
    assert core.tasks[task_id].state == NodeTaskState.FAILED
    assert core.sessions["s1"].state == SessionState.FAILED


def test_session_failure_is_isolated_and_queues_one_cancellation_action() -> None:
    core = SchedulerCore(function_workflow())
    core.register_session("failed")
    core.register_session("active")
    failed_task_id = core.begin_node("failed", "entry", ["item-failed"])
    active_task_id = core.begin_node("active", "entry", ["item-active"])

    core.fail_node(failed_task_id, "RuntimeError", "boom")

    assert core.sessions["failed"].state == SessionState.FAILED
    assert core.sessions["active"].state == SessionState.ACTIVE
    assert core.tasks[active_task_id].state == NodeTaskState.RUNNING
    assert core.take_actions() == [CancelSessionAction(session_id="failed")]
    assert core.take_actions() == []


def test_session_failure_cancels_pending_acquiring_and_emitting_tasks() -> None:
    core = SchedulerCore(branched_workflow())
    core.register_session("s1")
    start_id = core.begin_node("s1", "start", ["item-1"])
    complete_and_finish(core, start_id)
    left_id = core.begin_node("s1", "left", ["left-input"])
    right_id = core.begin_node("s1", "right", ["right-input"])
    core.complete(right_id, function_report(core, right_id))
    end_id = core.sessions["s1"].task_ids["end"]

    core.fail_node(end_id, "EmissionError", "cannot persist")

    assert core.tasks[left_id].state == NodeTaskState.RUNNING
    assert core.tasks[right_id].state == NodeTaskState.CANCELLED
    assert core.tasks[end_id].state == NodeTaskState.FAILED
    assert not core.drain_complete()

    decision = core.complete(left_id, function_report(core, left_id))
    assert decision == CompleteDecision(emit_output=False)
    assert core.tasks[left_id].state == NodeTaskState.CANCELLED
    assert core.drain_complete()


def test_late_failed_function_report_cancels_running_task() -> None:
    core = SchedulerCore(branched_workflow())
    core.register_session("s1")
    start_id = core.begin_node("s1", "start", ["item-1"])
    complete_and_finish(core, start_id)
    left_id = core.begin_node("s1", "left", ["left-input"])
    end_id = core.sessions["s1"].task_ids["end"]
    core.fail_node(end_id, "EmissionError", "cannot persist")

    decision = core.complete(
        left_id,
        function_report(core, left_id, status="failed"),
    )

    assert decision == CompleteDecision(emit_output=False)
    assert core.tasks[left_id].state == NodeTaskState.CANCELLED


def test_session_failure_removes_pending_acquire() -> None:
    core = SchedulerCore(agent_workflow())
    core.register_session("s1")
    task_id = core.begin_node("s1", "agent", ["item-1"])
    acquire_id = core.request_acquire(task_id, input_tokens=64, created_at=1.0)

    core.fail_node(task_id, "TimeoutError", "acquire timed out")

    assert acquire_id not in core.pending_acquires
    assert core.tasks[task_id].state == NodeTaskState.FAILED
    assert core.drain_complete()


def test_cancelled_task_uses_narrow_finish_and_fail_errors() -> None:
    core = SchedulerCore(function_workflow())
    core.register_session("s1")
    task_id = core.begin_node("s1", "entry", ["item-1"])
    core.complete(task_id, function_report(core, task_id))
    terminal_id = core.sessions["s1"].task_ids["terminal"]
    core.fail_node(terminal_id, "Stopped", "session stopped")

    with pytest.raises(TaskCancelledError):
        core.finish_node(task_id, output_report(task_id))
    with pytest.raises(TaskCancelledError):
        core.fail_node(task_id, "QueueError", "queue closed")


def test_emission_failure_is_terminal() -> None:
    core = SchedulerCore(function_workflow())
    core.register_session("s1")
    task_id = core.begin_node("s1", "entry", ["item-1"])
    core.complete(task_id, function_report(core, task_id))

    core.fail_node(task_id, "QueueError", "output queue closed")

    assert core.tasks[task_id].state == NodeTaskState.FAILED
    assert core.sessions["s1"].state == SessionState.FAILED
    with pytest.raises(ValueError, match="emitting"):
        core.finish_node(task_id, output_report(task_id))


def test_finish_node_validates_report_and_state() -> None:
    core = SchedulerCore(function_workflow())
    core.register_session("s1")
    task_id = core.begin_node("s1", "entry", ["item-1"])

    with pytest.raises(ValueError, match="emitting"):
        core.finish_node(task_id, output_report(task_id))

    core.complete(task_id, function_report(core, task_id))
    with pytest.raises(ValueError, match="does not match"):
        core.finish_node(
            task_id,
            output_report(task_id).model_copy(update={"task_id": "other"}),
        )


def test_drain_waits_for_active_work_and_emission() -> None:
    core = SchedulerCore(function_workflow())
    assert core.drain_complete()
    core.register_session("s1")
    assert not core.drain_complete()
    task_id = core.begin_node("s1", "entry", ["item-1"])
    assert not core.drain_complete()
    core.complete(task_id, function_report(core, task_id))
    assert not core.drain_complete()


def test_runtime_reports_are_frozen_strict_and_forbid_extra_fields() -> None:
    payload: dict[str, Any] = {
        "task_id": "task-1",
        "session_id": "s1",
        "node_id": "node-1",
        "input_item_ids": ["item-1"],
        "started_at": 1.0,
        "finished_at": 2.0,
        "duration_sec": 1.0,
        "status": "success",
        "error_type": None,
        "error_message": None,
    }
    report = FunctionTaskRuntimeReport.model_validate(payload)

    with pytest.raises(ValidationError, match="frozen"):
        report.status = "failed"
    with pytest.raises(ValidationError, match="Extra inputs"):
        FunctionTaskRuntimeReport.model_validate({**payload, "unexpected": True})
    with pytest.raises(ValidationError):
        FunctionTaskRuntimeReport.model_validate({**payload, "duration_sec": "1.0"})
    with pytest.raises(ValidationError):
        FunctionTaskRuntimeReport.model_validate({**payload, "status": "oom"})


def test_agent_runtime_report_uses_the_plan_contract() -> None:
    payload: dict[str, Any] = {
        "acquire_id": "acquire-1",
        "task_id": "task-1",
        "session_id": "s1",
        "node_id": "agent",
        "input_item_ids": ["item-1"],
        "model_key": "test-model",
        "accelerator_id": "node/a100:0",
        "gpu_kind": "a100",
        "input_tokens": 64,
        "max_new_tokens": 16,
        "output_tokens": 8,
        "hit_token_limit": False,
        "finish_reason": None,
        "queue_time_sec": None,
        "time_to_first_token_sec": None,
        "replica_inflight_at_start": 1,
        "started_at": 1.0,
        "finished_at": 2.0,
        "duration_sec": 1.0,
        "status": "success",
        "engine_failed": False,
        "error_type": None,
    }

    report = AgentTaskRuntimeReport.model_validate(payload)

    assert report.model_dump(mode="python") == payload
    with pytest.raises(ValidationError, match="Extra inputs"):
        AgentTaskRuntimeReport.model_validate({**payload, "error_message": "no"})
    with pytest.raises(ValidationError):
        AgentTaskRuntimeReport.model_validate({**payload, "input_tokens": 0})
