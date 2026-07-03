from __future__ import annotations

import pytest
import ray
from langchain.agents import AgentState
from langchain_core.messages import AIMessage, HumanMessage
from ray.util.queue import Queue

from workflow.types import (
    ExecutionConfig,
    RetryConfig,
    WorkerQueueItem,
    WorkerState,
    WorkflowDataItem,
)
from workflow.worker import NodeWorker, message_text, resolve_session_content


def make_item(session_id: str, source_node: str, text: str) -> WorkflowDataItem:
    return WorkflowDataItem(
        session_id=session_id,
        item_id=f"{session_id}-{source_node}",
        source_node=source_node,
        target_node="joiner",
        message=AgentState(messages=[AIMessage(content=text)]),
    )


def make_execution_config() -> ExecutionConfig:
    return ExecutionConfig(
        model_name="stub",
        model_path="/models/stub",
        devices=["cuda:0"],
    )


class FlakyAgent:
    def __init__(self, fail_times: int) -> None:
        self.remaining_failures = fail_times

    def invoke(self, state: AgentState) -> AgentState:
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise RuntimeError("flaky failure")
        text = state["messages"][-1].content
        return AgentState(messages=[AIMessage(content=f"ok:{text}")])


def make_flaky_factory(fail_times: int):
    def factory(config, system_prompt, tools, device):
        return FlakyAgent(fail_times)

    return factory


def test_resolve_session_content_passes_through_for_entry_node():
    store: dict[str, dict[str, AgentState]] = {}
    item = make_item("s1", "client", "hello")
    ready, content = resolve_session_content(store, [], item)
    assert ready
    assert [message_text(m) for m in content] == ["hello"]
    assert store == {}


def test_resolve_session_content_keys_entries_by_source_node():
    store: dict[str, dict[str, AgentState]] = {}
    ready, _ = resolve_session_content(store, ["a", "b"], make_item("s1", "a", "from-a"))
    assert not ready
    assert set(store["s1"]) == {"a"}


def test_resolve_session_content_ready_only_when_all_dependencies_present():
    store: dict[str, dict[str, AgentState]] = {}
    ready, _ = resolve_session_content(store, ["a", "b"], make_item("s1", "b", "from-b"))
    assert not ready
    ready, content = resolve_session_content(store, ["a", "b"], make_item("s1", "a", "from-a"))
    assert ready
    assert [message_text(m) for m in content] == ["from-a", "from-b"]


def test_resolve_session_content_clears_store_after_join_fires():
    store: dict[str, dict[str, AgentState]] = {}
    resolve_session_content(store, ["a", "b"], make_item("s1", "a", "from-a"))
    ready, _ = resolve_session_content(store, ["a", "b"], make_item("s1", "b", "from-b"))
    assert ready
    assert store == {}


def test_resolve_session_content_rejects_duplicate_item_from_same_source():
    store: dict[str, dict[str, AgentState]] = {}
    resolve_session_content(store, ["a", "b"], make_item("s1", "a", "first"))
    with pytest.raises(AssertionError):
        resolve_session_content(store, ["a", "b"], make_item("s1", "a", "second"))


def test_resolve_session_content_keeps_sessions_independent():
    store: dict[str, dict[str, AgentState]] = {}
    ready, _ = resolve_session_content(store, ["a", "b"], make_item("s1", "a", "s1-a"))
    assert not ready
    ready, content = resolve_session_content(store, ["a", "b"], make_item("s2", "b", "s2-b"))
    assert not ready
    ready, content = resolve_session_content(store, ["a", "b"], make_item("s2", "a", "s2-a"))
    assert ready
    assert [message_text(m) for m in content] == ["s2-a", "s2-b"]
    assert set(store) == {"s1"}


def submit_data_item(input_queue: Queue, text: str) -> None:
    input_queue.put(
        WorkerQueueItem(
            data=WorkflowDataItem(
                session_id="s1",
                item_id="i1",
                source_node="client",
                target_node="node",
                message=AgentState(messages=[HumanMessage(content=text)]),
            )
        )
    )


def test_process_item_stores_result_for_terminal_node_with_no_output_queues(
    ray_session, wait_until
):
    input_queue = Queue()
    worker = NodeWorker.remote(
        node_name="sink",
        execution_config=None,
        input_queue=input_queue,
        output_queues={},
    )
    worker.loop.remote()
    submit_data_item(input_queue, "terminal-payload")
    assert wait_until(lambda: ray.get(worker.has_result.remote("s1")))
    result = ray.get(worker.get_result.remote("s1"))
    assert result["messages"][-1].content == "terminal-payload"
    input_queue.put(WorkerQueueItem(worker_state=WorkerState.STOPPED))
    assert wait_until(lambda: ray.get(worker.get_state.remote()) == WorkerState.STOPPED)


def test_invoke_agent_with_retry_succeeds_after_transient_failure(
    ray_session, wait_until
):
    input_queue = Queue()
    output_queue = Queue()
    failure_queue = Queue()
    worker = NodeWorker.remote(
        node_name="agent_node",
        execution_config=make_execution_config(),
        input_queue=input_queue,
        output_queues={"next": output_queue},
        retry_config=RetryConfig(max_attempts=2, retry_delay_sec=0),
        failure_queue=failure_queue,
        agent_factory=make_flaky_factory(fail_times=1),
    )
    ray.get(worker.load.remote())
    worker.loop.remote()
    submit_data_item(input_queue, "hello")
    forwarded: WorkerQueueItem = output_queue.get(timeout=30)
    assert forwarded.data is not None
    assert forwarded.data.source_node == "agent_node"
    assert forwarded.data.message["messages"][-1].content == "ok:hello"
    assert failure_queue.qsize() == 0
    input_queue.put(WorkerQueueItem(worker_state=WorkerState.STOPPED))
    assert wait_until(lambda: ray.get(worker.get_state.remote()) == WorkerState.STOPPED)


def test_invoke_agent_with_retry_records_failure_and_skips_item(
    ray_session, wait_until
):
    input_queue = Queue()
    output_queue = Queue()
    failure_queue = Queue()
    worker = NodeWorker.remote(
        node_name="agent_node",
        execution_config=make_execution_config(),
        input_queue=input_queue,
        output_queues={"next": output_queue},
        retry_config=RetryConfig(max_attempts=2, retry_delay_sec=0, on_exhausted="skip_item"),
        failure_queue=failure_queue,
        agent_factory=make_flaky_factory(fail_times=10),
    )
    ray.get(worker.load.remote())
    worker.loop.remote()
    submit_data_item(input_queue, "hello")
    record = failure_queue.get(timeout=30)
    assert record.node_name == "agent_node"
    assert record.attempt == 2
    assert record.error_type == "RuntimeError"
    assert output_queue.qsize() == 0
    assert ray.get(worker.get_state.remote()) == WorkerState.RUNNING
    input_queue.put(WorkerQueueItem(worker_state=WorkerState.STOPPED))
    assert wait_until(lambda: ray.get(worker.get_state.remote()) == WorkerState.STOPPED)


def test_invoke_agent_with_retry_fail_workflow_stops_worker(ray_session, wait_until):
    input_queue = Queue()
    failure_queue = Queue()
    worker = NodeWorker.remote(
        node_name="agent_node",
        execution_config=make_execution_config(),
        input_queue=input_queue,
        output_queues={"next": Queue()},
        retry_config=RetryConfig(max_attempts=1, retry_delay_sec=0),
        failure_queue=failure_queue,
        agent_factory=make_flaky_factory(fail_times=10),
    )
    ray.get(worker.load.remote())
    loop_ref = worker.loop.remote()
    submit_data_item(input_queue, "hello")
    with pytest.raises(ray.exceptions.RayTaskError):
        ray.get(loop_ref)
    assert failure_queue.get(timeout=30).error_type == "RuntimeError"
    assert wait_until(lambda: ray.get(worker.get_state.remote()) == WorkerState.STOPPED)
