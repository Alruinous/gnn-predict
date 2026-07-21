from __future__ import annotations

import asyncio
import inspect
import json
import string
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from queue import Empty
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast, overload

import ray
from langchain.agents import AgentState
from langchain_core.messages import AIMessage
from ray.exceptions import RayActorError

from workflow.actor_support import invoke
from workflow.replica import PromptEncoding, ReplicaInferenceResult
from workflow.scheduler import (
    AcquireAlreadyGrantedError,
    AgentTaskRuntimeReport,
    CompleteDecision,
    FunctionTaskRuntimeReport,
    InputTraceReport,
    ItemTraceReport,
    OutputReport,
    SessionInactiveError,
    TaskCancelledError,
)
from workflow.schema import (
    AgentNodeConfig,
    ExecutionConfig,
    FunctionNodeConfig,
    NodeConfig,
)
from workflow.storage import ResultPersistenceError
from workflow.types import NodeWorkerState, WorkflowDataItem

FaninStore = dict[str, dict[str, WorkflowDataItem]]
NodeFunction = Callable[..., object]
InputResolver = Callable[[object], object]

_FORMATTER = string.Formatter()


class PromptTokenizerProtocol(Protocol):
    def build_prompt(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> PromptEncoding: ...

    def decode(self, token_ids: tuple[int, ...]) -> str: ...


PromptTokenizerFactory = Callable[[ExecutionConfig], PromptTokenizerProtocol]


class GrantInfoProtocol(Protocol):
    acquire_id: str
    task_id: str
    replica_id: str
    backend_handle: object
    accelerator_ids: tuple[str, ...]
    model_key: str
    gpu_kind: str
    input_tokens: int
    max_new_tokens: int


@dataclass(frozen=True, slots=True)
class ResolvedFanin:
    items: tuple[WorkflowDataItem, ...]
    states: dict[str, AgentState]

    @property
    def input_item_ids(self) -> list[str]:
        return [item.item_id for item in self.items]


@dataclass(frozen=True, slots=True)
class AgentExecutionResult:
    report: AgentTaskRuntimeReport
    output: AgentState | None


def resolve_fanin(
    store: FaninStore,
    dependencies: tuple[str, ...],
    item: WorkflowDataItem,
    *,
    completed_sessions: set[str] | None = None,
) -> ResolvedFanin | None:
    if completed_sessions is not None and item.session_id in completed_sessions:
        raise ValueError(f"fan-in already resolved for session {item.session_id!r}")
    if not dependencies:
        if item.source_node is not None:
            raise ValueError("entry item must not have a source node")
        if completed_sessions is not None:
            completed_sessions.add(item.session_id)
        return ResolvedFanin(items=(item,), states={})

    source_node = item.source_node
    if source_node not in dependencies:
        raise ValueError(f"unexpected fan-in source: {source_node!r}")
    session_items = store.setdefault(item.session_id, {})
    if source_node in session_items:
        raise ValueError(
            f"duplicate fan-in source {source_node!r} for session {item.session_id!r}"
        )
    session_items[source_node] = item
    if len(session_items) != len(dependencies):
        return None

    items = tuple(session_items[source] for source in dependencies)
    session_input_ref = items[0].session_input_ref
    if any(entry.session_input_ref != session_input_ref for entry in items[1:]):
        raise ValueError("fan-in items must share one session input reference")
    states = {
        source: cast(AgentState, session_items[source].message)
        for source in dependencies
    }
    del store[item.session_id]
    if completed_sessions is not None:
        completed_sessions.add(item.session_id)
    return ResolvedFanin(items=items, states=states)


def message_text(state: AgentState) -> str:
    messages = state.get("messages")
    if not messages:
        raise ValueError("AgentState must contain at least one message")
    content = messages[-1].content
    if not isinstance(content, str):
        raise TypeError("the final AgentState message must contain text")
    return content


def build_prompt_context(
    session_inputs: Mapping[str, object],
    states: Mapping[str, AgentState],
) -> dict[str, object]:
    context = dict(session_inputs)
    outputs = {source: message_text(state) for source, state in states.items()}
    if outputs:
        context["content"] = "\n\n".join(outputs.values())
    else:
        context.setdefault("content", "")
    context["node_outputs_json"] = json.dumps(outputs, ensure_ascii=False)
    if len(outputs) == 1:
        context["previous_output"] = next(iter(outputs.values()))
    else:
        context.pop("previous_output", None)
    return context


@overload
async def execute_function[ResultT](
    function: Callable[..., Awaitable[ResultT]],
    session_inputs: Mapping[str, object],
    states: Mapping[str, AgentState],
    parameters: Mapping[str, object],
) -> ResultT: ...


@overload
async def execute_function[ResultT](
    function: Callable[..., ResultT],
    session_inputs: Mapping[str, object],
    states: Mapping[str, AgentState],
    parameters: Mapping[str, object],
) -> ResultT: ...


async def execute_function(
    function: Callable[..., object],
    session_inputs: Mapping[str, object],
    states: Mapping[str, AgentState],
    parameters: Mapping[str, object],
) -> object:
    readonly_inputs = MappingProxyType(dict(session_inputs))
    readonly_states = MappingProxyType(dict(states))
    readonly_parameters = MappingProxyType(dict(parameters))
    if inspect.iscoroutinefunction(function):
        result = function(readonly_inputs, readonly_states, readonly_parameters)
    else:
        result = await asyncio.to_thread(
            function,
            readonly_inputs,
            readonly_states,
            readonly_parameters,
        )
    if inspect.isawaitable(result):
        return await result
    return result


async def execute_agent(
    *,
    node: AgentNodeConfig,
    task_id: str,
    session_id: str,
    input_item_ids: list[str],
    prompt_context: Mapping[str, object],
    scheduler: object,
    tokenizer: PromptTokenizerProtocol,
    acquire_timeout_sec: float,
    grant_poll_interval_sec: float,
    grant_observer: Callable[[GrantInfoProtocol], None] | None = None,
    grant_finished: Callable[[], None] | None = None,
) -> AgentExecutionResult:
    user_prompt = _render_prompt(node.prompt_template, prompt_context)
    prompt = tokenizer.build_prompt(
        user_prompt,
        system_prompt=node.system_prompt,
    )
    input_tokens = prompt.input_tokens
    acquire_value = await invoke(
        scheduler,
        "request_acquire",
        task_id,
        input_tokens,
        time.time(),
    )
    if not isinstance(acquire_value, str):
        raise TypeError("request_acquire must return an acquire id")
    grant = await _wait_for_grant(
        scheduler,
        acquire_value,
        acquire_timeout_sec=acquire_timeout_sec,
        grant_poll_interval_sec=grant_poll_interval_sec,
    )
    _validate_grant(
        grant,
        acquire_value,
        task_id,
        input_tokens,
        node.execution.max_new_tokens,
    )
    if grant_observer is not None:
        grant_observer(grant)
    try:
        result_value = await invoke(
            grant.backend_handle,
            "invoke",
            acquire_value,
            prompt.token_ids,
            max_new_tokens=node.execution.max_new_tokens,
            generation=node.execution,
        )
    finally:
        if grant_finished is not None:
            grant_finished()
    if not isinstance(result_value, ReplicaInferenceResult):
        raise TypeError("replica invoke must return ReplicaInferenceResult")
    if result_value.request_id != acquire_value:
        raise ValueError("replica result does not match the acquire request")
    report = AgentTaskRuntimeReport(
        acquire_id=acquire_value,
        task_id=task_id,
        session_id=session_id,
        node_id=node.name,
        input_item_ids=input_item_ids,
        model_key=grant.model_key,
        accelerator_id=grant.accelerator_ids[0],
        gpu_kind=grant.gpu_kind,
        input_tokens=result_value.input_tokens,
        max_new_tokens=node.execution.max_new_tokens,
        output_tokens=result_value.output_tokens,
        hit_token_limit=result_value.hit_token_limit,
        finish_reason=result_value.finish_reason,
        queue_time_sec=result_value.queue_time_sec,
        time_to_first_token_sec=result_value.time_to_first_token_sec,
        replica_inflight_at_start=result_value.replica_inflight_at_start,
        started_at=result_value.started_at,
        finished_at=result_value.finished_at,
        duration_sec=result_value.duration_sec,
        status=result_value.status,
        engine_failed=result_value.engine_failed,
        error_type=result_value.error_type,
    )
    output = None
    if result_value.status == "success":
        output_text = tokenizer.decode(result_value.output_token_ids)
        output = AgentState(messages=[AIMessage(content=output_text)])
    return AgentExecutionResult(report=report, output=output)


class NodeWorker:
    def __init__(
        self,
        *,
        node: NodeConfig,
        dependencies: tuple[str, ...],
        successors: tuple[str, ...],
        input_queue: object,
        output_queues: Mapping[str, object],
        scheduler: object,
        result_store: object,
        functions: Mapping[str, NodeFunction],
        input_resolver: InputResolver | None = None,
        prompt_tokenizer_factory: PromptTokenizerFactory | None = None,
        acquire_timeout_sec: float = 60.0,
        grant_poll_interval_sec: float = 0.05,
        control_poll_interval_sec: float = 0.05,
    ) -> None:
        if set(output_queues) != set(successors):
            raise ValueError("output queues must exactly match static successors")
        if isinstance(node, FunctionNodeConfig) and node.function not in functions:
            raise KeyError(f"function is not registered: {node.function}")
        if acquire_timeout_sec <= 0 or grant_poll_interval_sec <= 0:
            raise ValueError("acquire timing must be positive")
        if control_poll_interval_sec <= 0:
            raise ValueError("control poll interval must be positive")

        self.node = node
        self.dependencies = dependencies
        self.successors = successors
        self.input_queue = input_queue
        self.output_queues = dict(output_queues)
        self.scheduler = scheduler
        self.result_store = result_store
        self.functions = dict(functions)
        self.input_resolver = input_resolver or _resolve_session_inputs
        self.prompt_tokenizer_factory = (
            prompt_tokenizer_factory or _create_prompt_tokenizer
        )
        self.acquire_timeout_sec = acquire_timeout_sec
        self.grant_poll_interval_sec = grant_poll_interval_sec
        self.control_poll_interval_sec = control_poll_interval_sec
        self.fanin_store: FaninStore = {}
        self.processed_sessions: set[str] = set()
        self.cancelled_sessions: set[str] = set()
        self.status = NodeWorkerState.IDLE
        self._stopping = False
        self._run_started = False
        self._active_tasks: dict[str, asyncio.Task[None]] = {}
        self._active_replica_requests: dict[str, tuple[object, str]] = {}
        self._tokenizer: PromptTokenizerProtocol | None = None

    async def run(self) -> None:
        if self._run_started:
            raise RuntimeError("node worker has already started")
        self._run_started = True
        try:
            while True:
                self._reap_finished_tasks()
                if self._stopping:
                    if self._active_tasks:
                        await self._wait_for_finished_task()
                        continue
                    break
                if len(self._active_tasks) >= self._concurrency_limit:
                    await self._wait_for_finished_task()
                    continue
                item_value = await _poll_input_queue(
                    self.input_queue,
                    self.control_poll_interval_sec,
                )
                if item_value is None:
                    continue
                if not isinstance(item_value, WorkflowDataItem):
                    raise TypeError("node input queues carry WorkflowDataItem only")
                if item_value.session_id in self._active_tasks:
                    await self._active_tasks[item_value.session_id]
                    self._reap_finished_tasks()
                task = asyncio.create_task(self.process_item(item_value))
                self._active_tasks[item_value.session_id] = task
                self.status = NodeWorkerState.RUNNING
        finally:
            tasks = tuple(self._active_tasks.values())
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self._active_tasks.clear()
            self.status = NodeWorkerState.STOPPED

    async def process_item(self, item: WorkflowDataItem) -> None:
        managed = asyncio.current_task() in self._active_tasks.values()
        if not managed:
            self.status = NodeWorkerState.RUNNING
        try:
            await self._process_item(item)
        finally:
            if not managed and not self._stopping:
                self.status = NodeWorkerState.IDLE

    async def _process_item(self, item: WorkflowDataItem) -> None:
        if item.target_node != self.node.name:
            raise ValueError(
                f"item target {item.target_node!r} does not match {self.node.name!r}"
            )
        if item.session_id in self.cancelled_sessions:
            return
        dequeued_at = time.time()
        ready = resolve_fanin(
            self.fanin_store,
            self.dependencies,
            item,
            completed_sessions=self.processed_sessions,
        )
        if ready is None:
            received = self.fanin_store[item.session_id]
            await self._observe_input(
                item,
                dequeued_at,
                fanin_state="wait",
                waiting_for_sources=tuple(
                    source for source in self.dependencies if source not in received
                ),
            )
            return
        await self._observe_input(
            item,
            dequeued_at,
            fanin_state="ready",
            source_nodes=tuple(
                entry.source_node
                for entry in ready.items
                if entry.source_node is not None
            ),
            input_item_ids=tuple(ready.input_item_ids),
        )
        try:
            task_value = await invoke(
                self.scheduler,
                "begin_node",
                item.session_id,
                self.node.name,
                ready.input_item_ids,
            )
        except SessionInactiveError:
            return
        if not isinstance(task_value, str):
            raise TypeError("begin_node must return a task id")
        session_inputs = await self._resolve_inputs(ready.items[0].session_input_ref)

        if isinstance(self.node, FunctionNodeConfig):
            await self._process_function(
                task_value,
                item.session_id,
                ready,
                session_inputs,
            )
        else:
            await self._process_agent(
                task_value,
                item.session_id,
                ready,
                session_inputs,
            )

    async def cancel_session(self, session_id: str) -> None:
        self.cancelled_sessions.add(session_id)
        self.fanin_store.pop(session_id, None)
        active = self._active_replica_requests.get(session_id)
        if active is not None:
            backend_handle, request_id = active
            await invoke(backend_handle, "abort", request_id)

    def stop(self) -> None:
        self._stopping = True

    def get_state(self) -> NodeWorkerState:
        return self.status

    @property
    def _concurrency_limit(self) -> int:
        if isinstance(self.node, AgentNodeConfig):
            return self.node.execution.serving.max_num_seqs
        return self.node.max_concurrency

    def _reap_finished_tasks(self) -> None:
        for session_id, task in tuple(self._active_tasks.items()):
            if not task.done():
                continue
            del self._active_tasks[session_id]
            task.result()
        if not self._active_tasks and not self._stopping:
            self.status = NodeWorkerState.IDLE

    async def _wait_for_finished_task(self) -> None:
        if not self._active_tasks:
            return
        await asyncio.wait(
            tuple(self._active_tasks.values()),
            return_when=asyncio.FIRST_COMPLETED,
        )
        self._reap_finished_tasks()

    async def _observe_input(
        self,
        item: WorkflowDataItem,
        dequeued_at: float,
        *,
        fanin_state: Literal["wait", "ready"],
        waiting_for_sources: tuple[str, ...] = (),
        source_nodes: tuple[str, ...] = (),
        input_item_ids: tuple[str, ...] = (),
    ) -> None:
        await invoke(
            self.scheduler,
            "observe_input",
            InputTraceReport(
                session_id=item.session_id,
                item_id=item.item_id,
                source_node=item.source_node,
                target_node=item.target_node,
                dequeued_at=dequeued_at,
                fanin_state=fanin_state,
                waiting_for_sources=waiting_for_sources,
                source_nodes=source_nodes,
                input_item_ids=input_item_ids,
            ),
        )

    async def _resolve_inputs(self, input_ref: object) -> Mapping[str, object]:
        resolved = await asyncio.to_thread(self.input_resolver, input_ref)
        if inspect.isawaitable(resolved):
            resolved = await resolved
        if not isinstance(resolved, Mapping):
            raise TypeError("session input reference must resolve to a mapping")
        if any(not isinstance(key, str) for key in resolved):
            raise TypeError("session input keys must be strings")
        return cast(Mapping[str, object], resolved)

    async def _process_function(
        self,
        task_id: str,
        session_id: str,
        ready: ResolvedFanin,
        session_inputs: Mapping[str, object],
    ) -> None:
        node = cast(FunctionNodeConfig, self.node)
        function = self.functions[node.function]
        started_at = time.time()
        duration_sec = 0.0
        output: object | None = None
        last_error: Exception | None = None
        for attempt in range(node.retry.max_attempts):
            attempt_started = time.perf_counter()
            try:
                output = await execute_function(
                    function,
                    session_inputs,
                    ready.states,
                    node.parameters,
                )
                last_error = None
                break
            except Exception as error:
                last_error = error
            finally:
                duration_sec += time.perf_counter() - attempt_started
            if last_error is not None and attempt + 1 < node.retry.max_attempts:
                await asyncio.sleep(node.retry.retry_delay_sec)

        finished_at = time.time()
        report = FunctionTaskRuntimeReport(
            task_id=task_id,
            session_id=session_id,
            node_id=node.name,
            input_item_ids=ready.input_item_ids,
            started_at=started_at,
            finished_at=finished_at,
            duration_sec=duration_sec,
            status="failed" if last_error is not None else "success",
            error_type=type(last_error).__name__ if last_error is not None else None,
            error_message=str(last_error) if last_error is not None else None,
        )
        decision = await self._complete(task_id, report)
        if last_error is not None or not decision.emit_output:
            return
        if self._session_cancelled(session_id):
            return

        try:
            routed = _route_function_output(node, output, self.successors)
            output_report = await self._emit(
                task_id,
                session_id,
                ready.items[0].session_input_ref,
                routed,
            )
        except (ResultPersistenceError, RayActorError):
            raise
        except Exception as error:
            await self._fail_node(task_id, session_id, error)
            return
        if output_report is None:
            return
        await self._finish_node(task_id, session_id, output_report)

    async def _process_agent(
        self,
        task_id: str,
        session_id: str,
        ready: ResolvedFanin,
        session_inputs: Mapping[str, object],
    ) -> None:
        node = cast(AgentNodeConfig, self.node)
        tokenizer = self._agent_tokenizer(node.execution)
        prompt_context = build_prompt_context(session_inputs, ready.states)
        timeout_attempts = 0
        while True:
            try:
                execution = await execute_agent(
                    node=node,
                    task_id=task_id,
                    session_id=session_id,
                    input_item_ids=ready.input_item_ids,
                    prompt_context=prompt_context,
                    scheduler=self.scheduler,
                    tokenizer=tokenizer,
                    acquire_timeout_sec=self.acquire_timeout_sec,
                    grant_poll_interval_sec=self.grant_poll_interval_sec,
                    grant_observer=lambda grant: self._track_replica_request(
                        session_id, grant
                    ),
                    grant_finished=lambda: self._forget_replica_request(session_id),
                )
            except TimeoutError as error:
                timeout_attempts += 1
                if timeout_attempts < node.retry.max_attempts:
                    await asyncio.sleep(node.retry.retry_delay_sec)
                    continue
                await self._fail_node(task_id, session_id, error)
                return
            except SessionInactiveError:
                return
            except KeyError as error:
                await self._fail_node(task_id, session_id, error)
                return

            decision = await self._complete(
                task_id,
                execution.report,
                acquire_id=execution.report.acquire_id,
            )
            if decision.retry_acquire:
                if decision.emit_output:
                    raise RuntimeError("retry decision cannot emit output")
                await asyncio.sleep(node.retry.retry_delay_sec)
                continue
            if not decision.emit_output:
                return
            if self._session_cancelled(session_id):
                return
            if execution.output is None:
                raise RuntimeError("emitting agent completion has no output")

            routed = {successor: execution.output for successor in self.successors}
            try:
                output_report = await self._emit(
                    task_id,
                    session_id,
                    ready.items[0].session_input_ref,
                    routed,
                    terminal_output=execution.output,
                )
            except (ResultPersistenceError, RayActorError):
                raise
            except Exception as error:
                await self._fail_node(task_id, session_id, error)
                return
            if output_report is None:
                return
            await self._finish_node(task_id, session_id, output_report)
            return

    def _agent_tokenizer(
        self,
        execution: ExecutionConfig,
    ) -> PromptTokenizerProtocol:
        if self._tokenizer is None:
            self._tokenizer = self.prompt_tokenizer_factory(execution)
        return self._tokenizer

    def _track_replica_request(
        self,
        session_id: str,
        grant: GrantInfoProtocol,
    ) -> None:
        if session_id in self._active_replica_requests:
            raise RuntimeError("session already has an active replica request")
        self._active_replica_requests[session_id] = (
            grant.backend_handle,
            grant.acquire_id,
        )

    def _forget_replica_request(self, session_id: str) -> None:
        self._active_replica_requests.pop(session_id, None)

    async def _complete(
        self,
        task_id: str,
        report: FunctionTaskRuntimeReport | AgentTaskRuntimeReport,
        *,
        acquire_id: str | None = None,
    ) -> CompleteDecision:
        value = await invoke(
            self.scheduler,
            "complete",
            task_id,
            report,
            acquire_id=acquire_id,
        )
        if not isinstance(value, CompleteDecision):
            raise TypeError("complete must return CompleteDecision")
        return value

    async def _emit(
        self,
        task_id: str,
        session_id: str,
        session_input_ref: object,
        routed: Mapping[str, AgentState],
        *,
        terminal_output: AgentState | None = None,
    ) -> OutputReport | None:
        if self.successors:
            output_item_ids: list[str] = []
            output_items: list[ItemTraceReport] = []
            for successor in self.successors:
                if self._session_cancelled(session_id):
                    return None
                state = routed[successor]
                emitted_at = time.time()
                emitted = WorkflowDataItem(
                    session_id=session_id,
                    source_node=self.node.name,
                    target_node=successor,
                    message=state,
                    session_input_ref=session_input_ref,
                )
                await invoke(self.output_queues[successor], "put_async", emitted)
                enqueued_at = time.time()
                output_item_ids.append(emitted.item_id)
                output_items.append(
                    ItemTraceReport(
                        session_id=session_id,
                        item_id=emitted.item_id,
                        source_node=self.node.name,
                        target_node=successor,
                        emitted_at=emitted_at,
                        enqueued_at=enqueued_at,
                    )
                )
            return OutputReport(
                task_id=task_id,
                output_item_ids=output_item_ids,
                persisted_terminal_result=False,
                output_items=tuple(output_items),
            )

        result = terminal_output
        if result is None:
            result = routed[""]
        if self._session_cancelled(session_id):
            return None
        await invoke(
            self.result_store,
            "put",
            session_id,
            self.node.name,
            result,
        )
        return OutputReport(
            task_id=task_id,
            output_item_ids=[],
            persisted_terminal_result=True,
        )

    def _session_cancelled(self, session_id: str) -> bool:
        return session_id in self.cancelled_sessions

    async def _finish_node(
        self,
        task_id: str,
        session_id: str,
        output_report: OutputReport,
    ) -> None:
        if self._session_cancelled(session_id):
            return
        try:
            await invoke(self.scheduler, "finish_node", task_id, output_report)
        except TaskCancelledError:
            return

    async def _fail_node(
        self,
        task_id: str,
        session_id: str,
        error: Exception,
    ) -> None:
        if self._session_cancelled(session_id):
            return
        try:
            await invoke(
                self.scheduler,
                "fail_node",
                task_id,
                type(error).__name__,
                str(error),
            )
        except TaskCancelledError:
            return


def _render_prompt(template: str, context: Mapping[str, object]) -> str:
    fields = {
        field_name for _, field_name, _, _ in _FORMATTER.parse(template) if field_name
    }
    missing = sorted(field for field in fields if field not in context)
    if missing:
        raise KeyError(f"prompt template fields are unavailable: {', '.join(missing)}")
    return template.format_map(dict(context))


async def _wait_for_grant(
    scheduler: object,
    acquire_id: str,
    *,
    acquire_timeout_sec: float,
    grant_poll_interval_sec: float,
) -> GrantInfoProtocol:
    deadline = time.monotonic() + acquire_timeout_sec
    while True:
        grant_value = await invoke(scheduler, "poll_grant", acquire_id)
        if grant_value is not None:
            return cast(GrantInfoProtocol, grant_value)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            try:
                await invoke(scheduler, "cancel_acquire", acquire_id)
            except AcquireAlreadyGrantedError:
                grant_value = await invoke(scheduler, "poll_grant", acquire_id)
                if grant_value is None:
                    raise RuntimeError(
                        "granted acquire must be visible after cancellation race"
                    ) from None
                return cast(GrantInfoProtocol, grant_value)
            raise TimeoutError(f"acquire {acquire_id!r} timed out")
        await asyncio.sleep(min(grant_poll_interval_sec, remaining))


def _validate_grant(
    grant: GrantInfoProtocol,
    acquire_id: str,
    task_id: str,
    input_tokens: int,
    max_new_tokens: int,
) -> None:
    if grant.acquire_id != acquire_id or grant.task_id != task_id:
        raise ValueError("grant identity does not match the acquire request")
    if grant.input_tokens != input_tokens:
        raise ValueError("grant input token count does not match the prompt")
    if grant.max_new_tokens != max_new_tokens:
        raise ValueError("grant output limit does not match the workflow")
    if len(grant.accelerator_ids) != 1:
        raise ValueError("current model replicas require one accelerator")


def _route_function_output(
    node: FunctionNodeConfig,
    output: object,
    successors: tuple[str, ...],
) -> dict[str, AgentState]:
    if node.routing == "targeted":
        if not successors:
            raise ValueError("terminal functions must use broadcast routing")
        if not isinstance(output, dict) or "messages" in output:
            raise TypeError("targeted function must return target-to-state mapping")
        if any(not isinstance(target, str) for target in output):
            raise TypeError("targeted function keys must be strings")
        if set(output) != set(successors):
            raise ValueError("targeted function output must match static successors")
        typed_output = cast(dict[str, object], output)
        return {
            target: _validate_agent_state(value)
            for target, value in typed_output.items()
        }

    state = _validate_agent_state(output)
    if successors:
        return {successor: state for successor in successors}
    return {"": state}


def _validate_agent_state(value: object) -> AgentState:
    if not isinstance(value, dict) or "messages" not in value:
        raise TypeError("node output must be AgentState")
    return cast(AgentState, value)


def _resolve_session_inputs(input_ref: object) -> object:
    if isinstance(input_ref, Mapping):
        return input_ref
    return ray.get(cast(ray.ObjectRef[Any], input_ref))


def _create_prompt_tokenizer(execution: ExecutionConfig) -> PromptTokenizerProtocol:
    # Function-only workers should not import the Transformers runtime.
    from workflow.tokenizer import PromptTokenizer

    return PromptTokenizer(execution)


async def _poll_input_queue(queue: object, timeout_sec: float) -> object | None:
    get_async = cast(Any, queue).get_async
    if "timeout" in inspect.signature(get_async).parameters:
        try:
            return await invoke(queue, "get_async", timeout=timeout_sec)
        except Empty:
            return None
    try:
        return await asyncio.wait_for(
            invoke(queue, "get_async"),
            timeout=timeout_sec,
        )
    except TimeoutError:
        return None


_remote_with_options = cast(Any, ray.remote)
NodeWorkerActor = _remote_with_options(max_concurrency=8)(NodeWorker)
