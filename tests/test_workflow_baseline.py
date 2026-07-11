from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
from typing import Any, cast

import pytest
from langchain.agents import AgentState
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from workflow.baseline.bsp.executor import (
    run_bsp_workflow,
    run_bsp_workflow_batch,
)
from workflow.baseline.runner import StaticHuggingFaceRunner, StubBaselineRunner
from workflow.baseline.sequential.executor import run_sequential_workflow
from workflow.baseline.types import (
    BaselineNodeConfig,
    BaselineSessionRequest,
    BaselineWorkflow,
)

EXECUTION = {"model_path": "/models/stub", "devices": ["cuda:0"]}


def agent_node(name: str, prompt_template: str = "{previous_output}") -> dict[str, Any]:
    return {
        "name": name,
        "type": "agent",
        "prompt_template": prompt_template,
        "execution": EXECUTION,
    }


def branched_workflow() -> BaselineWorkflow:
    return BaselineWorkflow.model_validate(
        {
            "nodes": [
                {"name": "input", "type": "input"},
                agent_node("slow"),
                agent_node("fast"),
                agent_node("next"),
                {"name": "output", "type": "output"},
            ],
            "edges": [
                {"source": "input", "target": "slow"},
                {"source": "input", "target": "fast"},
                {"source": "fast", "target": "next"},
                {"source": "slow", "target": "output"},
                {"source": "next", "target": "output"},
            ],
        }
    )


class OrderedRunner:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def load_workflow(self, workflow: BaselineWorkflow) -> None:
        self.calls.clear()

    def run_node(self, node: BaselineNodeConfig, prompt: str) -> str:
        self.calls.append(node.name)
        return node.name


class BspBarrierRunner:
    def __init__(self) -> None:
        self.slow_started = Event()
        self.fast_finished = Event()
        self.next_started = Event()
        self.release_slow = Event()

    def load_workflow(self, workflow: BaselineWorkflow) -> None:
        pass

    def run_node(self, node: BaselineNodeConfig, prompt: str) -> str:
        if node.name == "slow":
            self.slow_started.set()
            assert self.fast_finished.wait(timeout=2.0)
            assert self.release_slow.wait(timeout=2.0)
        elif node.name == "fast":
            self.fast_finished.set()
        elif node.name == "next":
            self.next_started.set()
        return node.name


class BatchRunner:
    def __init__(self) -> None:
        self.barrier = Barrier(2)
        self.lock = Lock()
        self.active = 0
        self.max_active = 0

    def load_workflow(self, workflow: BaselineWorkflow) -> None:
        pass

    def run_node(self, node: BaselineNodeConfig, prompt: str) -> str:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.barrier.wait(timeout=2.0)
            return prompt
        finally:
            with self.lock:
                self.active -= 1


class FakeAgent:
    def __init__(self) -> None:
        self.lock = Lock()
        self.active = 0
        self.max_active = 0

    def invoke(self, state: AgentState) -> AgentState:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.05)
            return AgentState(messages=[AIMessage(content="ok")])
        finally:
            with self.lock:
                self.active -= 1


def test_sequential_baseline_runs_stable_topological_order() -> None:
    runner = OrderedRunner()

    state = run_sequential_workflow(
        branched_workflow(),
        BaselineSessionRequest(
            session_id="session-sequential",
            inputs={"input_text": "request"},
        ),
        runner,
    )

    assert runner.calls == ["slow", "fast", "next"]
    assert [event["node_name"] for event in state["trace"]] == [
        "input",
        "slow",
        "fast",
        "next",
        "output",
    ]
    assert state["final_output"] == "slow\n\nnext"
    assert {event["session_id"] for event in state["trace"]} == {
        "session-sequential"
    }


def test_bsp_baseline_waits_for_superstep_barrier() -> None:
    runner = BspBarrierRunner()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            run_bsp_workflow,
            branched_workflow(),
            BaselineSessionRequest(
                session_id="session-bsp",
                inputs={"input_text": "request"},
            ),
            runner,
        )
        try:
            assert runner.fast_finished.wait(timeout=2.0)
            assert runner.slow_started.wait(timeout=2.0)
            assert not runner.next_started.wait(timeout=0.1)
        finally:
            runner.release_slow.set()
        state = future.result(timeout=2.0)

    assert runner.next_started.is_set()
    assert state["final_output"] == "slow\n\nnext"
    assert {event["session_id"] for event in state["trace"]} == {"session-bsp"}


def test_bsp_batch_runs_sessions_concurrently_and_preserves_order() -> None:
    workflow = BaselineWorkflow.model_validate(
        {
            "nodes": [agent_node("model", "{value}")],
            "edges": [],
        }
    )
    runner = BatchRunner()

    states = run_bsp_workflow_batch(
        workflow,
        [
            BaselineSessionRequest(session_id="session-a", inputs={"value": "a"}),
            BaselineSessionRequest(session_id="session-b", inputs={"value": "b"}),
        ],
        runner,
        max_concurrency=2,
    )

    assert runner.max_active == 2
    assert [state["session_id"] for state in states] == ["session-a", "session-b"]
    assert [state["final_output"] for state in states] == ["a", "b"]
    assert [state["trace"][0]["session_id"] for state in states] == [
        "session-a",
        "session-b",
    ]


def test_bsp_batch_requires_positive_concurrency() -> None:
    workflow = BaselineWorkflow.model_validate(
        {"nodes": [{"name": "input", "type": "input"}], "edges": []}
    )

    with pytest.raises(ValueError, match="max_concurrency must be positive"):
        run_bsp_workflow_batch(
            workflow,
            [],
            StubBaselineRunner(),
            max_concurrency=0,
        )


def test_static_runner_serializes_calls_to_same_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = BaselineWorkflow.model_validate(
        {"nodes": [agent_node("model", "{value}")], "edges": []}
    )
    node = workflow.nodes[0]
    fake_agent = FakeAgent()
    runner = StaticHuggingFaceRunner()
    monkeypatch.setattr(
        runner,
        "_load_agent",
        lambda node: cast(CompiledStateGraph, fake_agent),
    )
    runner.load_workflow(workflow)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(runner.run_node, node, f"prompt-{index}")
            for index in range(2)
        ]
        outputs = [future.result(timeout=2.0) for future in futures]

    assert outputs == ["ok", "ok"]
    assert fake_agent.max_active == 1


def test_bsp_baseline_accepts_multiple_entries() -> None:
    workflow = BaselineWorkflow.model_validate(
        {
            "nodes": [
                agent_node("left", "{value}"),
                agent_node("right", "{value}"),
                {"name": "output", "type": "output"},
            ],
            "edges": [
                {"source": "left", "target": "output"},
                {"source": "right", "target": "output"},
            ],
        }
    )

    state = run_bsp_workflow(
        workflow,
        BaselineSessionRequest(session_id="multi-entry", inputs={"value": "x"}),
        StubBaselineRunner(),
    )

    assert workflow.entry_names() == ["left", "right"]
    assert state["final_output"] == "left:x\n\nright:x"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"nodes": [], "edges": []}, "non-empty"),
        (
            {
                "nodes": [
                    {"name": "a", "type": "input"},
                    {"name": "b", "type": "output"},
                ],
                "edges": [
                    {"source": "a", "target": "b"},
                    {"source": "b", "target": "a"},
                ],
            },
            "acyclic",
        ),
        (
            {
                "nodes": [
                    {"name": "a", "type": "input"},
                    {"name": "b", "type": "output"},
                ],
                "edges": [],
            },
            "exactly one terminal",
        ),
        (
            {
                "nodes": [{"name": "a", "type": "input"}],
                "edges": [{"source": "a", "target": "a"}],
            },
            "self-edge",
        ),
        (
            {
                "nodes": [
                    {"name": "a", "type": "input"},
                    {"name": "b", "type": "output"},
                ],
                "edges": [
                    {"source": "a", "target": "b"},
                    {"source": "a", "target": "b"},
                ],
            },
            "edges must be unique",
        ),
        (
            {
                "nodes": [{"name": "a", "type": "input"}],
                "edges": [{"source": "a", "target": "missing"}],
            },
            "unknown node",
        ),
    ],
)
def test_baseline_workflow_rejects_invalid_graphs(
    payload: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        BaselineWorkflow.model_validate(payload)
