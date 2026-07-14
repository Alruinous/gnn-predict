from __future__ import annotations

import asyncio
import inspect
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import pytest
from langchain.agents import AgentState
from langchain_core.messages import AIMessage
from ray.exceptions import RayActorError

from workflow.replica import PromptEncoding, ReplicaInferenceResult
from workflow.scheduler import (
    AcquireAlreadyGrantedError,
    AgentTaskRuntimeReport,
    CompleteDecision,
    FunctionTaskRuntimeReport,
    InputTraceReport,
    OutputReport,
    SessionInactiveError,
    TaskCancelledError,
)
from workflow.storage import ResultPersistenceError
from workflow.schema import AgentNodeConfig, FunctionNodeConfig
from workflow.types import NodeWorkerState, WorkflowDataItem
from workflow.worker import (
    NodeWorker,
    build_prompt_context,
    execute_agent,
    execute_function,
    resolve_fanin,
)


def state(text: str) -> AgentState:
    return AgentState(messages=[AIMessage(content=text)])


def item(
    session_id: str,
    source_node: str | None,
    *,
    target_node: str = "merge",
    text: str | None = None,
    session_input_ref: object | None = None,
) -> WorkflowDataItem:
    return WorkflowDataItem(
        session_id=session_id,
        source_node=source_node,
        target_node=target_node,
        message=state(text or source_node or "entry"),
        session_input_ref=(
            session_input_ref
            if session_input_ref is not None
            else {"content": f"input-{session_id}", "topic": "tests"}
        ),
    )


def function_node(
    *,
    name: str = "merge",
    function: str = "merge_states",
    routing: Literal["broadcast", "targeted"] = "broadcast",
    max_attempts: int = 1,
) -> FunctionNodeConfig:
    return FunctionNodeConfig.model_validate(
        {
            "name": name,
            "type": "function",
            "function": function,
            "routing": routing,
            "retry": {"max_attempts": max_attempts},
        }
    )


def agent_node(
    *,
    prompt_template: str = "Summarize {content}",
    max_attempts: int = 1,
    retry_delay_sec: int = 0,
) -> AgentNodeConfig:
    return AgentNodeConfig.model_validate(
        {
            "name": "agent",
            "type": "agent",
            "model": {"name": "test-model"},
            "execution": {
                "model_path": "/models/test-model",
                "use_chat_template": False,
                "serving": {
                    "max_model_len": 1024,
                    "max_num_seqs": 3,
                    "max_num_batched_tokens": 1536,
                },
            },
            "token_budget": {
                "min_max_new_tokens": 8,
                "default_max_new_tokens": 16,
                "max_max_new_tokens": 32,
            },
            "prompt_template": prompt_template,
            "system_prompt": "system",
            "retry": {
                "max_attempts": max_attempts,
                "retry_delay_sec": retry_delay_sec,
            },
        }
    )


class FakeQueue:
    def __init__(self, before_put: Callable[[], object] | None = None) -> None:
        self.items: list[WorkflowDataItem] = []
        self._incoming: asyncio.Queue[WorkflowDataItem] = asyncio.Queue()
        self.before_put = before_put

    async def put_async(self, value: WorkflowDataItem) -> None:
        if self.before_put is not None:
            result = self.before_put()
            if inspect.isawaitable(result):
                await result
        self.items.append(value)

    async def get_async(self) -> WorkflowDataItem:
        return await self._incoming.get()

    def feed(self, value: WorkflowDataItem) -> None:
        self._incoming.put_nowait(value)


class FakeResultStore:
    def __init__(
        self,
        events: list[str] | None = None,
        before_put: Callable[[], None] | None = None,
    ) -> None:
        self.results: dict[str, AgentState] = {}
        self.events = events
        self.before_put = before_put

    async def put(self, session_id: str, node_id: str, result: AgentState) -> None:
        if self.before_put is not None:
            self.before_put()
        if self.events is not None:
            self.events.append("persist")
        self.results[session_id] = result


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def build_prompt(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> PromptEncoding:
        self.calls.append((user_prompt, system_prompt))
        return PromptEncoding(
            text=f"rendered:{user_prompt}",
            token_ids=(0, 1, 2, 3, 4, 5, 6),
        )

    def decode(self, token_ids: tuple[int, ...]) -> str:
        assert token_ids == (20, 21, 22, 23)
        return "generated"


class FakeReplica:
    def __init__(self, results: list[ReplicaInferenceResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, tuple[int, ...], int]] = []
        self.aborted: list[str] = []

    async def invoke(
        self,
        request_id: str,
        prompt_token_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        generation: object,
    ) -> ReplicaInferenceResult:
        self.calls.append((request_id, prompt_token_ids, max_new_tokens))
        return self.results.pop(0).model_copy(update={"request_id": request_id})

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


class BlockingReplica(FakeReplica):
    def __init__(self, result_count: int) -> None:
        super().__init__([inference_result() for _ in range(result_count)])
        self.active = 0
        self.peak = 0
        self.release = asyncio.Event()

    async def invoke(
        self,
        request_id: str,
        prompt_token_ids: tuple[int, ...],
        *,
        max_new_tokens: int,
        generation: object,
    ) -> ReplicaInferenceResult:
        self.calls.append((request_id, prompt_token_ids, max_new_tokens))
        self.active += 1
        self.peak = max(self.peak, self.active)
        await self.release.wait()
        self.active -= 1
        return self.results.pop(0).model_copy(update={"request_id": request_id})


@dataclass(frozen=True)
class FakeGrant:
    acquire_id: str
    task_id: str
    replica_id: str
    backend_handle: object
    accelerator_ids: tuple[str, ...]
    model_key: str
    gpu_kind: str
    input_tokens: int
    granted_max_new_tokens: int


class FakeScheduler:
    def __init__(
        self,
        *,
        decisions: list[CompleteDecision] | None = None,
        grants: list[FakeGrant] | None = None,
        events: list[str] | None = None,
        inactive_sessions: set[str] | None = None,
        grant_on_cancel: FakeGrant | None = None,
        on_complete: Callable[[], object] | None = None,
        finish_cancelled: bool = False,
        fail_cancelled: bool = False,
    ) -> None:
        self.decisions = decisions or [CompleteDecision(emit_output=True)]
        self.available_grants = grants or []
        self.events = events
        self.inactive_sessions = inactive_sessions or set()
        self.grant_on_cancel = grant_on_cancel
        self.on_complete = on_complete
        self.finish_cancelled = finish_cancelled
        self.fail_cancelled = fail_cancelled
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.reports: list[FunctionTaskRuntimeReport | AgentTaskRuntimeReport] = []
        self.output_reports: list[OutputReport] = []
        self.input_traces: list[InputTraceReport] = []
        self.cancelled_acquires: list[str] = []
        self._acquire_count = 0

    async def observe_input(self, report: InputTraceReport) -> None:
        self.input_traces.append(report)

    async def begin_node(
        self, session_id: str, node_id: str, input_item_ids: list[str]
    ) -> str:
        self.calls.append(
            ("begin_node", (session_id, node_id, tuple(input_item_ids)), {})
        )
        if session_id in self.inactive_sessions:
            raise SessionInactiveError(f"session is not active: {session_id}")
        return f"task-{session_id}-{node_id}"

    async def request_acquire(
        self, task_id: str, input_tokens: int, created_at: float
    ) -> str:
        self._acquire_count += 1
        acquire_id = f"acquire-{self._acquire_count}"
        self.calls.append(
            (
                "request_acquire",
                (task_id, input_tokens),
                {"created_at": created_at},
            )
        )
        return acquire_id

    async def poll_grant(self, acquire_id: str) -> FakeGrant | None:
        self.calls.append(("poll_grant", (acquire_id,), {}))
        if not self.available_grants:
            return None
        grant = self.available_grants[0]
        if grant.acquire_id != acquire_id:
            return None
        return self.available_grants.pop(0)

    async def cancel_acquire(self, acquire_id: str) -> None:
        self.cancelled_acquires.append(acquire_id)
        if (
            self.grant_on_cancel is not None
            and self.grant_on_cancel.acquire_id == acquire_id
        ):
            self.available_grants.append(self.grant_on_cancel)
            self.grant_on_cancel = None
            raise AcquireAlreadyGrantedError(acquire_id)

    async def complete(
        self,
        task_id: str,
        runtime_report: FunctionTaskRuntimeReport | AgentTaskRuntimeReport,
        acquire_id: str | None = None,
    ) -> CompleteDecision:
        self.calls.append(
            ("complete", (task_id, runtime_report), {"acquire_id": acquire_id})
        )
        self.reports.append(runtime_report)
        if self.on_complete is not None:
            result = self.on_complete()
            if inspect.isawaitable(result):
                await result
        return self.decisions.pop(0)

    async def finish_node(self, task_id: str, output_report: OutputReport) -> None:
        if self.finish_cancelled:
            raise TaskCancelledError(task_id)
        if self.events is not None:
            self.events.append("finish")
        self.calls.append(("finish_node", (task_id, output_report), {}))
        self.output_reports.append(output_report)

    async def fail_node(
        self, task_id: str, error_type: str, error_message: str
    ) -> None:
        if self.fail_cancelled:
            raise TaskCancelledError(task_id)
        self.calls.append(("fail_node", (task_id, error_type, error_message), {}))


class AwaitableRef:
    def __init__(self, value: object) -> None:
        self.value = value

    def __await__(self):
        async def resolve() -> object:
            if inspect.isawaitable(self.value):
                return await self.value
            return self.value

        return resolve().__await__()


class RemoteMethod:
    def __init__(self, function: Callable[..., object]) -> None:
        self.function = function

    def remote(self, *args: object, **kwargs: object) -> AwaitableRef:
        return AwaitableRef(self.function(*args, **kwargs))


class RemoteHandle:
    def __init__(self, target: object) -> None:
        self.target = target

    def __getattr__(self, name: str) -> RemoteMethod:
        return RemoteMethod(cast(Callable[..., object], getattr(self.target, name)))


def inference_result(
    status: Literal["success", "failed", "oom"] = "success",
) -> ReplicaInferenceResult:
    return ReplicaInferenceResult(
        request_id="request",
        output_token_ids=(20, 21, 22, 23) if status == "success" else (),
        input_tokens=7,
        output_tokens=4 if status == "success" else 0,
        hit_token_limit=False,
        finish_reason="stop" if status == "success" else None,
        queue_time_sec=None,
        time_to_first_token_sec=None,
        replica_inflight_at_start=1,
        started_at=10.0,
        finished_at=11.0,
        duration_sec=1.0,
        status=status,
        error_type=None if status == "success" else "BackendError",
        error_message=None if status == "success" else "failed",
    )


def worker(
    node: AgentNodeConfig | FunctionNodeConfig,
    scheduler: object,
    *,
    dependencies: tuple[str, ...] = (),
    successors: tuple[str, ...] = (),
    output_queues: Mapping[str, FakeQueue] | None = None,
    result_store: FakeResultStore | None = None,
    functions: Mapping[str, Callable[..., object]] | None = None,
    tokenizer: FakeTokenizer | None = None,
    input_queue: FakeQueue | None = None,
    input_resolver: Callable[[object], object] | None = None,
    acquire_timeout_sec: float = 0.01,
    grant_poll_interval_sec: float = 0.001,
) -> NodeWorker:
    return NodeWorker(
        node=node,
        dependencies=dependencies,
        successors=successors,
        input_queue=input_queue or FakeQueue(),
        output_queues=output_queues or {},
        scheduler=scheduler,
        result_store=result_store or FakeResultStore(),
        functions=functions or {},
        input_resolver=(
            input_resolver or (lambda value: cast(dict[str, object], value))
        ),
        prompt_tokenizer_factory=(
            (lambda _: tokenizer) if tokenizer is not None else None
        ),
        grant_poll_interval_sec=grant_poll_interval_sec,
        acquire_timeout_sec=acquire_timeout_sec,
        control_poll_interval_sec=0.001,
    )


def test_fanin_is_scoped_by_session_and_source_order() -> None:
    store: dict[str, dict[str, WorkflowDataItem]] = {}

    assert resolve_fanin(store, ("left", "right"), item("s1", "right")) is None
    assert resolve_fanin(store, ("left", "right"), item("s2", "left")) is None
    ready = resolve_fanin(store, ("left", "right"), item("s1", "left"))

    assert ready is not None
    assert tuple(ready.states) == ("left", "right")
    assert [message["messages"][-1].content for message in ready.states.values()] == [
        "left",
        "right",
    ]
    assert "s1" not in store
    assert tuple(store["s2"]) == ("left",)


def test_fanin_rejects_duplicate_source_for_one_session() -> None:
    store: dict[str, dict[str, WorkflowDataItem]] = {}
    resolve_fanin(store, ("left", "right"), item("s1", "left"))

    with pytest.raises(ValueError, match="duplicate fan-in source"):
        resolve_fanin(store, ("left", "right"), item("s1", "left"))


def test_fanin_rejects_late_items_after_session_is_ready() -> None:
    store: dict[str, dict[str, WorkflowDataItem]] = {}
    completed_sessions: set[str] = set()
    resolve_fanin(
        store,
        ("left", "right"),
        item("s1", "left"),
        completed_sessions=completed_sessions,
    )
    ready = resolve_fanin(
        store,
        ("left", "right"),
        item("s1", "right"),
        completed_sessions=completed_sessions,
    )
    assert ready is not None

    with pytest.raises(ValueError, match="already resolved"):
        resolve_fanin(
            store,
            ("left", "right"),
            item("s1", "left"),
            completed_sessions=completed_sessions,
        )

    assert store == {}


def test_fanin_requires_one_session_input_reference() -> None:
    store: dict[str, dict[str, WorkflowDataItem]] = {}
    resolve_fanin(
        store,
        ("left", "right"),
        item("s1", "left", session_input_ref={"content": "left"}),
    )

    with pytest.raises(ValueError, match="session input reference"):
        resolve_fanin(
            store,
            ("left", "right"),
            item("s1", "right", session_input_ref={"content": "right"}),
        )


def test_prompt_context_preserves_sources_and_limits_previous_output() -> None:
    session_inputs = {"topic": "runtime", "content": "original"}
    one = build_prompt_context(session_inputs, {"left": state("left text")})
    many = build_prompt_context(
        session_inputs,
        {"left": state("left text"), "right": state("right text")},
    )

    assert one["topic"] == "runtime"
    assert one["content"] == "left text"
    assert one["previous_output"] == "left text"
    assert json.loads(cast(str, one["node_outputs_json"])) == {"left": "left text"}
    assert many["content"] == "left text\n\nright text"
    assert "previous_output" not in many
    assert json.loads(cast(str, many["node_outputs_json"])) == {
        "left": "left text",
        "right": "right text",
    }


def test_execute_function_runs_sync_callable_off_the_event_loop() -> None:
    caller_thread = threading.get_ident()
    called_thread: int | None = None

    def merge(
        inputs: Mapping[str, object],
        states: Mapping[str, AgentState],
        parameters: Mapping[str, object],
    ) -> AgentState:
        nonlocal called_thread
        called_thread = threading.get_ident()
        return state(f"{inputs['topic']}:{len(states)}{parameters['suffix']}")

    result = asyncio.run(
        execute_function(
            merge,
            {"topic": "runtime"},
            {"left": state("left")},
            {"suffix": "!"},
        )
    )

    assert result["messages"][-1].content == "runtime:1!"
    assert called_thread is not None
    assert called_thread != caller_thread


def test_execute_function_awaits_async_callable() -> None:
    async def merge(
        inputs: Mapping[str, object],
        states: Mapping[str, AgentState],
        parameters: Mapping[str, object],
    ) -> AgentState:
        await asyncio.sleep(0)
        return state(f"{inputs['topic']}:{len(states)}")

    result = asyncio.run(execute_function(merge, {"topic": "async"}, {}, {}))

    assert result["messages"][-1].content == "async:0"


def test_execute_function_passes_read_only_mappings() -> None:
    def inspect_inputs(
        inputs: Mapping[str, object],
        states: Mapping[str, AgentState],
        parameters: Mapping[str, object],
    ) -> AgentState:
        for value in (inputs, states, parameters):
            with pytest.raises(TypeError):
                cast(dict[str, object], value)["mutated"] = True
        return state("immutable")

    result = asyncio.run(
        execute_function(
            inspect_inputs,
            {"content": "input"},
            {"source": state("output")},
            {"limit": 1},
        )
    )

    assert result["messages"][-1].content == "immutable"


def test_agent_formats_and_validates_prompt_before_acquire() -> None:
    node = agent_node(prompt_template="Use {missing}")
    scheduler = FakeScheduler()
    tokenizer = FakeTokenizer()

    with pytest.raises(KeyError, match="missing"):
        asyncio.run(
            execute_agent(
                node=node,
                task_id="task-s1-agent",
                session_id="s1",
                input_item_ids=["item-1"],
                prompt_context={"content": "text"},
                scheduler=scheduler,
                tokenizer=tokenizer,
                acquire_timeout_sec=0.01,
                grant_poll_interval_sec=0.001,
            )
        )

    assert not any(call[0] == "request_acquire" for call in scheduler.calls)


def test_agent_uses_granted_budget_and_builds_runtime_report() -> None:
    replica = FakeReplica([inference_result()])
    grant = FakeGrant(
        acquire_id="acquire-1",
        task_id="task-s1-agent",
        replica_id="replica-1",
        backend_handle=replica,
        accelerator_ids=("host/v100:0",),
        model_key="model-key",
        gpu_kind="v100",
        input_tokens=7,
        granted_max_new_tokens=32,
    )
    scheduler = FakeScheduler(grants=[grant])
    tokenizer = FakeTokenizer()

    execution = asyncio.run(
        execute_agent(
            node=agent_node(),
            task_id="task-s1-agent",
            session_id="s1",
            input_item_ids=["item-1"],
            prompt_context={"content": "text"},
            scheduler=scheduler,
            tokenizer=tokenizer,
            acquire_timeout_sec=0.01,
            grant_poll_interval_sec=0.001,
        )
    )

    assert tokenizer.calls == [("Summarize text", "system")]
    assert replica.calls == [
        ("acquire-1", (0, 1, 2, 3, 4, 5, 6), 32)
    ]
    assert execution.output is not None
    assert execution.output["messages"][-1].content == "generated"
    assert execution.report.acquire_id == "acquire-1"
    assert execution.report.accelerator_id == "host/v100:0"
    assert execution.report.granted_max_new_tokens == 32


def test_agent_cancels_pending_acquire_on_timeout() -> None:
    scheduler = FakeScheduler()

    with pytest.raises(TimeoutError, match="acquire"):
        asyncio.run(
            execute_agent(
                node=agent_node(),
                task_id="task-s1-agent",
                session_id="s1",
                input_item_ids=["item-1"],
                prompt_context={"content": "text"},
                scheduler=scheduler,
                tokenizer=FakeTokenizer(),
                acquire_timeout_sec=0.002,
                grant_poll_interval_sec=0.001,
            )
        )

    assert scheduler.cancelled_acquires == ["acquire-1"]


def test_acquire_cancel_race_consumes_the_grant() -> None:
    replica = FakeReplica([inference_result()])
    grant = FakeGrant(
        acquire_id="acquire-1",
        task_id="task-s1-agent",
        replica_id="replica-1",
        backend_handle=replica,
        accelerator_ids=("host/v100:0",),
        model_key="model-key",
        gpu_kind="v100",
        input_tokens=7,
        granted_max_new_tokens=16,
    )
    scheduler = FakeScheduler(grant_on_cancel=grant)
    node_worker = worker(
        agent_node(),
        scheduler,
        tokenizer=FakeTokenizer(),
        acquire_timeout_sec=0.002,
    )

    asyncio.run(node_worker.process_item(item("s1", None, target_node="agent")))

    assert scheduler.cancelled_acquires == ["acquire-1"]
    assert len(scheduler.reports) == 1
    assert scheduler.reports[0].status == "success"
    assert scheduler.output_reports[0].persisted_terminal_result is True


def test_inactive_acquire_does_not_stop_worker_before_next_session() -> None:
    class InactiveAcquireScheduler(FakeScheduler):
        def __init__(self, grant: FakeGrant) -> None:
            super().__init__(grants=[grant])
            self.inactive_polled = asyncio.Event()

        async def poll_grant(self, acquire_id: str) -> FakeGrant | None:
            if acquire_id == "acquire-1":
                self.inactive_polled.set()
                raise SessionInactiveError("session is not active: s1")
            return await super().poll_grant(acquire_id)

    replica = FakeReplica([inference_result()])
    grant = FakeGrant(
        acquire_id="acquire-2",
        task_id="task-s2-agent",
        replica_id="replica-1",
        backend_handle=replica,
        accelerator_ids=("host/v100:0",),
        model_key="model-key",
        gpu_kind="v100",
        input_tokens=7,
        granted_max_new_tokens=16,
    )
    scheduler = InactiveAcquireScheduler(grant)
    input_queue = FakeQueue()
    result_ready = asyncio.Event()
    result_store = FakeResultStore(before_put=result_ready.set)
    node_worker = worker(
        agent_node(),
        scheduler,
        input_queue=input_queue,
        result_store=result_store,
        tokenizer=FakeTokenizer(),
    )

    async def run_sessions() -> None:
        run_task = asyncio.create_task(node_worker.run())
        try:
            input_queue.feed(item("s1", None, target_node="agent"))
            await asyncio.wait_for(scheduler.inactive_polled.wait(), timeout=0.5)
            done, _ = await asyncio.wait({run_task}, timeout=0.01)
            assert not done

            input_queue.feed(item("s2", None, target_node="agent"))
            await asyncio.wait_for(result_ready.wait(), timeout=0.5)
        finally:
            node_worker.stop()
            await asyncio.gather(run_task, return_exceptions=True)

    asyncio.run(run_sessions())

    assert result_store.results["s2"]["messages"][-1].content == "generated"
    assert node_worker.get_state() == NodeWorkerState.STOPPED


def test_remote_awaitable_handles_use_remote_methods() -> None:
    replica = FakeReplica([inference_result()])
    remote_replica = RemoteHandle(replica)
    grant = FakeGrant(
        acquire_id="acquire-1",
        task_id="task-s1-agent",
        replica_id="replica-1",
        backend_handle=remote_replica,
        accelerator_ids=("host/v100:0",),
        model_key="model-key",
        gpu_kind="v100",
        input_tokens=7,
        granted_max_new_tokens=16,
    )
    scheduler = FakeScheduler(grants=[grant])

    execution = asyncio.run(
        execute_agent(
            node=agent_node(),
            task_id="task-s1-agent",
            session_id="s1",
            input_item_ids=["item-1"],
            prompt_context={"content": "text"},
            scheduler=RemoteHandle(scheduler),
            tokenizer=FakeTokenizer(),
            acquire_timeout_sec=0.01,
            grant_poll_interval_sec=0.001,
        )
    )

    assert execution.report.status == "success"
    assert replica.calls == [
        ("acquire-1", (0, 1, 2, 3, 4, 5, 6), 16)
    ]


def test_broadcast_function_uses_common_complete_emit_finish_flow() -> None:
    scheduler = FakeScheduler()
    left_queue = FakeQueue()
    right_queue = FakeQueue()

    def broadcast(
        inputs: Mapping[str, object],
        states: Mapping[str, AgentState],
        parameters: Mapping[str, object],
    ) -> AgentState:
        return state(f"{inputs['content']}:{len(states)}")

    node_worker = worker(
        function_node(function="broadcast"),
        scheduler,
        successors=("left", "right"),
        output_queues={"left": left_queue, "right": right_queue},
        functions={"broadcast": broadcast},
    )

    asyncio.run(node_worker.process_item(item("s1", None, target_node="merge")))

    assert [call[0] for call in scheduler.calls] == [
        "begin_node",
        "complete",
        "finish_node",
    ]
    assert len(left_queue.items) == len(right_queue.items) == 1
    left_item = left_queue.items[0]
    right_item = right_queue.items[0]
    assert left_item.item_id != right_item.item_id
    assert left_item.source_node == right_item.source_node == "merge"
    assert left_item.session_input_ref is right_item.session_input_ref
    assert scheduler.output_reports[0].persisted_terminal_result is False
    assert set(scheduler.output_reports[0].output_item_ids) == {
        left_item.item_id,
        right_item.item_id,
    }


def test_targeted_function_requires_exact_static_successors() -> None:
    scheduler = FakeScheduler()
    left_queue = FakeQueue()
    right_queue = FakeQueue()

    def targeted(
        inputs: Mapping[str, object],
        states: Mapping[str, AgentState],
        parameters: Mapping[str, object],
    ) -> dict[str, AgentState]:
        return {"left": state("left only")}

    node_worker = worker(
        function_node(function="targeted", routing="targeted"),
        scheduler,
        successors=("left", "right"),
        output_queues={"left": left_queue, "right": right_queue},
        functions={"targeted": targeted},
    )

    asyncio.run(node_worker.process_item(item("s1", None)))

    assert [call[0] for call in scheduler.calls] == [
        "begin_node",
        "complete",
        "fail_node",
    ]
    assert left_queue.items == right_queue.items == []


def test_cancellation_after_complete_prevents_output_side_effects() -> None:
    scheduler = FakeScheduler()
    left_queue = FakeQueue()
    right_queue = FakeQueue()

    def broadcast(
        inputs: Mapping[str, object],
        states: Mapping[str, AgentState],
        parameters: Mapping[str, object],
    ) -> AgentState:
        return state("done")

    node_worker = worker(
        function_node(function="broadcast"),
        scheduler,
        successors=("left", "right"),
        output_queues={"left": left_queue, "right": right_queue},
        functions={"broadcast": broadcast},
    )
    scheduler.on_complete = lambda: node_worker.cancel_session("s1")

    asyncio.run(node_worker.process_item(item("s1", None)))

    assert left_queue.items == right_queue.items == []
    assert not any(call[0] == "finish_node" for call in scheduler.calls)


def test_cancellation_between_outputs_stops_remaining_side_effects() -> None:
    scheduler = FakeScheduler()
    right_queue = FakeQueue()

    def broadcast(
        inputs: Mapping[str, object],
        states: Mapping[str, AgentState],
        parameters: Mapping[str, object],
    ) -> AgentState:
        return state("done")

    node_worker = worker(
        function_node(function="broadcast"),
        scheduler,
        successors=("left", "right"),
        output_queues={
            "left": FakeQueue(lambda: node_worker.cancel_session("s1")),
            "right": right_queue,
        },
        functions={"broadcast": broadcast},
    )

    asyncio.run(node_worker.process_item(item("s1", None)))

    assert right_queue.items == []
    assert not any(call[0] == "finish_node" for call in scheduler.calls)


def test_cancelled_finish_and_fail_are_benign() -> None:
    finish_scheduler = FakeScheduler(finish_cancelled=True)
    fail_scheduler = FakeScheduler(fail_cancelled=True)

    def broadcast(
        inputs: Mapping[str, object],
        states: Mapping[str, AgentState],
        parameters: Mapping[str, object],
    ) -> AgentState:
        return state("done")

    def targeted(
        inputs: Mapping[str, object],
        states: Mapping[str, AgentState],
        parameters: Mapping[str, object],
    ) -> dict[str, AgentState]:
        return {"left": state("missing right")}

    finish_worker = worker(
        function_node(function="broadcast"),
        finish_scheduler,
        functions={"broadcast": broadcast},
    )
    fail_worker = worker(
        function_node(function="targeted", routing="targeted"),
        fail_scheduler,
        successors=("left", "right"),
        output_queues={"left": FakeQueue(), "right": FakeQueue()},
        functions={"targeted": targeted},
    )

    asyncio.run(finish_worker.process_item(item("s1", None)))
    asyncio.run(fail_worker.process_item(item("s2", None)))


def test_function_retries_then_reports_one_success() -> None:
    scheduler = FakeScheduler()
    attempts = 0

    def flaky(
        inputs: Mapping[str, object],
        states: Mapping[str, AgentState],
        parameters: Mapping[str, object],
    ) -> AgentState:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("retry")
        return state("ok")

    node_worker = worker(
        function_node(function="flaky", max_attempts=2),
        scheduler,
        result_store=FakeResultStore(),
        functions={"flaky": flaky},
    )

    asyncio.run(node_worker.process_item(item("s1", None)))

    assert attempts == 2
    assert len(scheduler.reports) == 1
    report = scheduler.reports[0]
    assert isinstance(report, FunctionTaskRuntimeReport)
    assert report.status == "success"


def test_function_retry_exhaustion_reports_failure_without_emission() -> None:
    scheduler = FakeScheduler(decisions=[CompleteDecision(emit_output=False)])
    attempts = 0

    def broken(
        inputs: Mapping[str, object],
        states: Mapping[str, AgentState],
        parameters: Mapping[str, object],
    ) -> AgentState:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("broken")

    node_worker = worker(
        function_node(function="broken", max_attempts=2),
        scheduler,
        functions={"broken": broken},
    )

    asyncio.run(node_worker.process_item(item("s1", None)))

    assert attempts == 2
    report = scheduler.reports[0]
    assert isinstance(report, FunctionTaskRuntimeReport)
    assert report.status == "failed"
    assert report.error_type == "RuntimeError"
    assert [call[0] for call in scheduler.calls] == ["begin_node", "complete"]


def test_terminal_result_is_persisted_before_finish() -> None:
    events: list[str] = []
    scheduler = FakeScheduler(events=events)
    result_store = FakeResultStore(events)

    def terminal(
        inputs: Mapping[str, object],
        states: Mapping[str, AgentState],
        parameters: Mapping[str, object],
    ) -> AgentState:
        return state("done")

    node_worker = worker(
        function_node(name="terminal", function="terminal"),
        scheduler,
        result_store=result_store,
        functions={"terminal": terminal},
    )

    asyncio.run(node_worker.process_item(item("s1", None, target_node="terminal")))

    assert events == ["persist", "finish"]
    assert result_store.results["s1"]["messages"][-1].content == "done"
    assert scheduler.output_reports[0].persisted_terminal_result is True


def test_agent_retry_decision_reacquires_once_then_emits() -> None:
    first_replica = FakeReplica([inference_result("oom")])
    second_replica = FakeReplica([inference_result()])
    grants = [
        FakeGrant(
            acquire_id="acquire-1",
            task_id="task-s1-agent",
            replica_id="replica-1",
            backend_handle=first_replica,
            accelerator_ids=("host/v100:0",),
            model_key="model-key",
            gpu_kind="v100",
            input_tokens=7,
            granted_max_new_tokens=16,
        ),
        FakeGrant(
            acquire_id="acquire-2",
            task_id="task-s1-agent",
            replica_id="replica-2",
            backend_handle=second_replica,
            accelerator_ids=("host/a100:0",),
            model_key="model-key",
            gpu_kind="a100",
            input_tokens=7,
            granted_max_new_tokens=32,
        ),
    ]
    scheduler = FakeScheduler(
        decisions=[
            CompleteDecision(emit_output=False, retry_acquire=True),
            CompleteDecision(emit_output=True),
        ],
        grants=grants,
    )
    result_store = FakeResultStore()
    node_worker = worker(
        agent_node(),
        scheduler,
        result_store=result_store,
        tokenizer=FakeTokenizer(),
    )

    asyncio.run(node_worker.process_item(item("s1", None, target_node="agent")))

    assert len(scheduler.reports) == 2
    assert [report.status for report in scheduler.reports] == ["oom", "success"]
    assert result_store.results["s1"]["messages"][-1].content == "generated"
    assert scheduler.output_reports[0].persisted_terminal_result is True


def test_non_oom_agent_retry_honors_retry_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("workflow.worker.asyncio.sleep", record_sleep)
    first_replica = FakeReplica([inference_result("failed")])
    second_replica = FakeReplica([inference_result()])
    scheduler = FakeScheduler(
        decisions=[
            CompleteDecision(emit_output=False, retry_acquire=True),
            CompleteDecision(emit_output=True),
        ],
        grants=[
            FakeGrant(
                acquire_id="acquire-1",
                task_id="task-s1-agent",
                replica_id="replica-1",
                backend_handle=first_replica,
                accelerator_ids=("host/v100:0",),
                model_key="model-key",
                gpu_kind="v100",
                input_tokens=7,
                granted_max_new_tokens=16,
            ),
            FakeGrant(
                acquire_id="acquire-2",
                task_id="task-s1-agent",
                replica_id="replica-2",
                backend_handle=second_replica,
                accelerator_ids=("host/a100:0",),
                model_key="model-key",
                gpu_kind="a100",
                input_tokens=7,
                granted_max_new_tokens=16,
            ),
        ],
    )
    node_worker = worker(
        agent_node(max_attempts=2, retry_delay_sec=3),
        scheduler,
        tokenizer=FakeTokenizer(),
    )

    asyncio.run(node_worker.process_item(item("s1", None, target_node="agent")))

    assert [report.status for report in scheduler.reports] == ["failed", "success"]
    assert delays == [3]


def test_acquire_timeout_retries_until_node_attempts_are_exhausted() -> None:
    scheduler = FakeScheduler()
    node_worker = worker(
        agent_node(max_attempts=2),
        scheduler,
        tokenizer=FakeTokenizer(),
        acquire_timeout_sec=0.001,
    )

    asyncio.run(node_worker.process_item(item("s1", None, target_node="agent")))

    assert len(scheduler.cancelled_acquires) == 2
    assert [call[0] for call in scheduler.calls].count("request_acquire") == 2
    assert [call[0] for call in scheduler.calls][-1] == "fail_node"


def test_cancelled_session_purges_fanin_and_discards_later_items() -> None:
    scheduler = FakeScheduler()

    def merge(
        inputs: Mapping[str, object],
        states: Mapping[str, AgentState],
        parameters: Mapping[str, object],
    ) -> AgentState:
        return state("never")

    node_worker = worker(
        function_node(),
        scheduler,
        dependencies=("left", "right"),
        functions={"merge_states": merge},
    )
    asyncio.run(node_worker.process_item(item("s1", "left")))
    assert "s1" in node_worker.fanin_store

    asyncio.run(node_worker.cancel_session("s1"))
    asyncio.run(node_worker.process_item(item("s1", "right")))

    assert "s1" not in node_worker.fanin_store
    assert scheduler.calls == []


def test_inactive_session_item_is_discarded_after_dequeue() -> None:
    scheduler = FakeScheduler(inactive_sessions={"s1"})
    resolved = False

    def resolve(value: object) -> object:
        nonlocal resolved
        resolved = True
        return value

    node_worker = worker(
        function_node(),
        scheduler,
        functions={"merge_states": lambda inputs, states, parameters: state("never")},
        input_resolver=resolve,
    )

    asyncio.run(node_worker.process_item(item("s1", None)))

    assert [call[0] for call in scheduler.calls] == ["begin_node"]
    assert scheduler.reports == []
    assert resolved is False
    assert node_worker.get_state() == NodeWorkerState.IDLE


def test_worker_is_running_only_while_processing() -> None:
    scheduler = FakeScheduler()
    input_queue = FakeQueue()
    observed_states: list[NodeWorkerState] = []

    def execute(
        inputs: Mapping[str, object],
        states: Mapping[str, AgentState],
        parameters: Mapping[str, object],
    ) -> AgentState:
        observed_states.append(node_worker.get_state())
        return state("done")

    node_worker = worker(
        function_node(),
        scheduler,
        functions={"merge_states": execute},
        input_queue=input_queue,
    )

    async def run_one() -> None:
        run_task = asyncio.create_task(node_worker.run())
        input_queue.feed(item("s1", None))
        while not scheduler.output_reports:
            await asyncio.sleep(0)
        while node_worker.get_state() == NodeWorkerState.RUNNING:
            await asyncio.sleep(0)
        assert node_worker.get_state() == NodeWorkerState.IDLE
        node_worker.stop()
        await run_task

    asyncio.run(run_one())

    assert observed_states == [NodeWorkerState.RUNNING]
    assert node_worker.get_state() == NodeWorkerState.STOPPED


def test_agent_worker_runs_three_sessions_concurrently() -> None:
    replica = BlockingReplica(result_count=3)
    grants = [
        FakeGrant(
            acquire_id=f"acquire-{index}",
            task_id=f"task-s{index}-agent",
            replica_id="replica-1",
            backend_handle=replica,
            accelerator_ids=("host/v100:0",),
            model_key="model-key",
            gpu_kind="v100",
            input_tokens=7,
            granted_max_new_tokens=16,
        )
        for index in range(1, 4)
    ]
    scheduler = FakeScheduler(
        decisions=[CompleteDecision(emit_output=True) for _ in range(3)],
        grants=grants,
    )
    input_queue = FakeQueue()
    result_store = FakeResultStore()
    tokenizer = FakeTokenizer()
    node_worker = worker(
        agent_node(),
        scheduler,
        tokenizer=tokenizer,
        input_queue=input_queue,
        result_store=result_store,
    )

    async def scenario() -> None:
        run_task = asyncio.create_task(node_worker.run())
        for index in range(1, 4):
            input_queue.feed(item(f"s{index}", None, target_node="agent"))
        while replica.peak < 3:
            await asyncio.sleep(0)
        replica.release.set()
        while len(result_store.results) < 3:
            await asyncio.sleep(0)
        node_worker.stop()
        await run_task

    asyncio.run(scenario())

    assert replica.peak == 3
    assert set(result_store.results) == {"s1", "s2", "s3"}
    assert len(tokenizer.calls) == 3
    assert node_worker.get_state() == NodeWorkerState.STOPPED


def test_run_stops_without_queue_control_messages() -> None:
    scheduler = FakeScheduler()
    input_queue = FakeQueue()
    node_worker = worker(
        function_node(),
        scheduler,
        functions={"merge_states": lambda inputs, states, parameters: state("unused")},
        input_queue=input_queue,
    )

    async def run_and_stop() -> None:
        run_task = asyncio.create_task(node_worker.run())
        await asyncio.sleep(0.003)
        node_worker.stop()
        await run_task

    asyncio.run(run_and_stop())

    assert node_worker.get_state() == NodeWorkerState.STOPPED


def test_result_persistence_failure_escapes_as_run_level_failure() -> None:
    class FailingResultStore(FakeResultStore):
        async def put(
            self,
            session_id: str,
            node_id: str,
            result: AgentState,
        ) -> None:
            raise ResultPersistenceError("result flush failed")

    scheduler = FakeScheduler()
    node_worker = worker(
        function_node(),
        scheduler,
        functions={"merge_states": lambda inputs, states, parameters: state("answer")},
        result_store=FailingResultStore(),
    )

    with pytest.raises(ResultPersistenceError, match="flush"):
        asyncio.run(node_worker.process_item(item("s1", None)))

    assert [call[0] for call in scheduler.calls] == ["begin_node", "complete"]


def test_result_store_actor_failure_escapes_worker_run() -> None:
    class UnavailableResultStore(FakeResultStore):
        async def put(
            self,
            session_id: str,
            node_id: str,
            result: AgentState,
        ) -> None:
            raise RayActorError(error_msg="result store actor unavailable")

    scheduler = FakeScheduler()
    input_queue = FakeQueue()
    input_queue.feed(item("s1", None))
    node_worker = worker(
        function_node(),
        scheduler,
        functions={"merge_states": lambda inputs, states, parameters: state("answer")},
        input_queue=input_queue,
        result_store=UnavailableResultStore(),
    )

    async def run_worker() -> None:
        await asyncio.wait_for(node_worker.run(), timeout=0.1)

    with pytest.raises(RayActorError):
        asyncio.run(run_worker())

    assert node_worker.get_state() == NodeWorkerState.STOPPED
    assert [call[0] for call in scheduler.calls] == ["begin_node", "complete"]
