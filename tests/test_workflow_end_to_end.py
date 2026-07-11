from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

import pytest
from langchain.agents import AgentState
from langchain_core.messages import AIMessage

from workflow import controller as controller_module
from workflow.artifacts import SchedulerConfig
from workflow.controller import WorkflowController
from workflow.schema import Workflow
from workflow.worker import message_text


class WaitUntil(Protocol):
    def __call__(
        self,
        predicate: Callable[[], bool],
        timeout_sec: float = 60.0,
        interval_sec: float = 0.2,
    ) -> bool: ...


def state(text: str) -> AgentState:
    return AgentState(messages=[AIMessage(content=text)])


def function_workflow(*, terminal_function: str = "merge") -> Workflow:
    if terminal_function == "block":
        return Workflow.model_validate(
            {
                "nodes": [
                    {"name": "terminal", "type": "function", "function": "block"}
                ],
                "edges": [],
            }
        )
    return Workflow.model_validate(
        {
            "nodes": [
                {
                    "name": "split",
                    "type": "function",
                    "function": "split",
                    "routing": "targeted",
                },
                {"name": "left", "type": "function", "function": "left"},
                {"name": "right", "type": "function", "function": "right"},
                {"name": "merge", "type": "function", "function": "merge"},
            ],
            "edges": [
                {"source": "split", "target": "left"},
                {"source": "split", "target": "right"},
                {"source": "left", "target": "merge"},
                {"source": "right", "target": "merge"},
            ],
        }
    )


def split(
    inputs: Mapping[str, object],
    states: Mapping[str, AgentState],
    parameters: Mapping[str, object],
) -> dict[str, AgentState]:
    value = str(inputs["value"])
    return {"left": state(f"{value}:left"), "right": state(f"{value}:right")}


def left(
    inputs: Mapping[str, object],
    states: Mapping[str, AgentState],
    parameters: Mapping[str, object],
) -> AgentState:
    return state(f"L[{message_text(states['split'])}]")


def right(
    inputs: Mapping[str, object],
    states: Mapping[str, AgentState],
    parameters: Mapping[str, object],
) -> AgentState:
    return state(f"R[{message_text(states['split'])}]")


def merge(
    inputs: Mapping[str, object],
    states: Mapping[str, AgentState],
    parameters: Mapping[str, object],
) -> AgentState:
    return state(
        f"{inputs['value']}|{message_text(states['left'])}|"
        f"{message_text(states['right'])}"
    )


def test_function_only_workflow_drains_without_session_mixing(
    ray_session: None,
    tmp_path: Path,
) -> None:
    controller = WorkflowController(
        function_workflow(),
        functions={"split": split, "left": left, "right": right, "merge": merge},
        scheduler_config=SchedulerConfig(),
        output_dir=tmp_path,
        run_id="function-e2e",
    )
    controller.start()
    with pytest.raises(RuntimeError, match="start"):
        controller.start()

    expected: dict[str, str] = {}
    for index in range(3):
        session_id = f"session-{index}"
        value = f"value-{index}"
        controller.submit(session_id, {"value": value})
        expected[session_id] = f"{value}|L[{value}:left]|R[{value}:right]"
    with pytest.raises(ValueError, match="duplicate"):
        controller.submit("session-0", {"value": "duplicate"})

    controller.drain_and_stop(timeout_sec=30.0)

    assert {
        session_id: message_text(controller.get_result(session_id))
        for session_id in expected
    } == expected
    for name in ("session_results.jsonl", "workflow_trace.jsonl", "run_summary.json"):
        assert (tmp_path / name).is_file()
    with pytest.raises(RuntimeError, match=r"admission|stopped"):
        controller.submit("late", {"value": "late"})


def test_stop_now_is_bounded_and_does_not_write_success_summary(
    ray_session: None,
    wait_until: WaitUntil,
    tmp_path: Path,
) -> None:
    started_path = tmp_path / "started"

    def block(
        inputs: Mapping[str, object],
        states: Mapping[str, AgentState],
        parameters: Mapping[str, object],
    ) -> AgentState:
        started_path.write_text("started", encoding="utf-8")
        time.sleep(30)
        return state("late")

    controller = WorkflowController(
        function_workflow(terminal_function="block"),
        functions={"block": block},
        scheduler_config=SchedulerConfig(),
        output_dir=tmp_path,
        run_id="forced-stop",
    )
    controller.start()
    controller.submit("session-1", {"value": "one"})
    assert wait_until(started_path.exists, timeout_sec=10.0)

    started = time.monotonic()
    controller.stop_now(timeout_sec=0.1)

    assert time.monotonic() - started < 3.0
    assert not (tmp_path / "run_summary.json").exists()
    with pytest.raises(RuntimeError, match=r"admission|stopped"):
        controller.submit("late", {"value": "late"})


def test_session_latency_ends_at_persistence_and_releases_input_reference(
    ray_session: None,
    wait_until: WaitUntil,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def terminal(
        inputs: Mapping[str, object],
        states: Mapping[str, AgentState],
        parameters: Mapping[str, object],
    ) -> AgentState:
        return state(str(inputs["value"]))

    controller = WorkflowController(
        function_workflow(terminal_function="block"),
        functions={"block": terminal},
        scheduler_config=SchedulerConfig(),
        output_dir=tmp_path,
        run_id="session-latency",
    )
    controller.start()
    original_put = controller_module.ray.put

    def delayed_put(value: object) -> object:
        time.sleep(0.12)
        return original_put(value)

    monkeypatch.setattr(controller_module.ray, "put", delayed_put)
    submitted_at = time.perf_counter()
    try:
        controller.submit("session-1", {"value": "one"})
        assert "session-1" in controller._session_input_refs

        result_path = tmp_path / "session_results.jsonl"
        assert wait_until(
            lambda: result_path.exists() and bool(result_path.read_text().strip()),
            timeout_sec=10.0,
            interval_sec=0.01,
        )
        persisted_elapsed = time.perf_counter() - submitted_at
        assert wait_until(
            lambda: "session-1" not in controller._session_input_refs,
            timeout_sec=10.0,
            interval_sec=0.01,
        )

        time.sleep(0.2)
        controller.drain_and_stop(timeout_sec=30.0)
        total_elapsed = time.perf_counter() - submitted_at

        summary = json.loads((tmp_path / "run_summary.json").read_text())
        latency = summary["session_latency_sec"]
        assert set(latency) >= {"p50", "p95", "max"}
        assert latency["p50"] == pytest.approx(latency["max"])
        assert latency["p95"] == pytest.approx(latency["max"])
        assert latency["max"] >= 0.12
        assert latency["max"] <= persisted_elapsed + 0.1
        assert total_elapsed - latency["max"] >= 0.15
    finally:
        if not controller._stopped:
            controller.stop_now(timeout_sec=1.0)
