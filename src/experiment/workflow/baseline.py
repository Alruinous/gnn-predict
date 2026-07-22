from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from threading import BoundedSemaphore, Lock
from types import MappingProxyType
from typing import Annotated, Any, NotRequired, Protocol, TypedDict, cast
from uuid import uuid4

import ray
from langchain.agents import AgentState
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from experiment.workflow.artifacts import write_jsonl_exclusive
from workflow.baseline.vllm_engine import StaticVLLMEngineActor
from workflow.replica import (
    ModelDeploymentConfig,
    ReplicaInferenceResult,
    ReplicaLoadResult,
)
from workflow.schema import AgentNodeConfig, FunctionNodeConfig, NodeConfig, Workflow
from workflow.tokenizer import PromptTokenizer
from workflow.types import TraceEvent
from workflow.worker import build_prompt_context, message_text

NodeFunction = Callable[..., object]


def merge_edge_outputs(
    left: dict[str, AgentState],
    right: dict[str, AgentState],
) -> dict[str, AgentState]:
    overlap = set(left) & set(right)
    if overlap:
        raise ValueError(f"baseline edge output collision: {sorted(overlap)}")
    merged = dict(left)
    merged.update(right)
    return merged


class BaselineState(TypedDict):
    session_id: str
    inputs: dict[str, object]
    edge_outputs: Annotated[dict[str, AgentState], merge_edge_outputs]
    final_output: AgentState | None


class BaselineStateUpdate(TypedDict):
    edge_outputs: dict[str, AgentState]
    final_output: NotRequired[AgentState]


@dataclass(frozen=True, slots=True)
class BaselineSessionRequest:
    session_id: str
    inputs: Mapping[str, object]
    arrival_offset_sec: float


@dataclass(frozen=True, slots=True)
class BaselineSessionResult:
    session_id: str
    final_output: AgentState
    submitted_at: float
    completed_at: float


class BaselineInvoker(Protocol):
    def invoke(self, request: BaselineSessionRequest) -> BaselineSessionResult: ...


class TraceRecorder:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._events: list[TraceEvent] = []
        self._lock = Lock()

    def record(
        self,
        event_type: str,
        *,
        ts: float | None = None,
        session_id: str | None = None,
        item_id: str | None = None,
        task_id: str | None = None,
        node_id: str | None = None,
        source_node: str | None = None,
        target_node: str | None = None,
        model_key: str | None = None,
        replica_id: str | None = None,
        accelerator_id: str | None = None,
        gpu_kind: str | None = None,
        acquire_id: str | None = None,
        payload: dict[str, object] | None = None,
    ) -> None:
        with self._lock:
            self._events.append(
                TraceEvent(
                    run_id=self.run_id,
                    event_seq=len(self._events),
                    event_type=event_type,
                    ts=time.time() if ts is None else ts,
                    session_id=session_id,
                    item_id=item_id,
                    task_id=task_id,
                    node_id=node_id,
                    source_node=source_node,
                    target_node=target_node,
                    model_key=model_key,
                    replica_id=replica_id,
                    accelerator_id=accelerator_id,
                    gpu_kind=gpu_kind,
                    acquire_id=acquire_id,
                    payload={} if payload is None else payload,
                )
            )

    def write(self, path: Path) -> None:
        with self._lock:
            events = tuple(self._events)
        write_jsonl_exclusive(path, events)


class StaticEnginePool:
    def __init__(
        self,
        workflow: Workflow,
        *,
        vllm_python: Path,
        recorder: TraceRecorder,
    ) -> None:
        self.workflow = workflow
        self.vllm_python = vllm_python.absolute()
        self.recorder = recorder
        self._actors: dict[str, object] = {}
        self._load_refs: dict[str, ray.ObjectRef[Any]] = {}
        self._load_futures: dict[str, Future[ReplicaLoadResult]] = {}
        self._load_results: dict[str, ReplicaLoadResult] = {}
        self._tokenizers: dict[str, PromptTokenizer] = {}
        self._load_executor: ThreadPoolExecutor | None = None
        self._state_lock = Lock()
        self._loading_started = False
        self._used_model_keys: set[str] = set()

    def load(self) -> None:
        self.start_loading()
        for model_key in self._actors:
            self._wait_until_loaded(model_key)

    def start_loading(self) -> None:
        if self._loading_started:
            raise RuntimeError("static baseline engines are already loaded")
        self._loading_started = True
        deployments: dict[str, ModelDeploymentConfig] = {}
        for node in self.workflow.nodes:
            if not isinstance(node, AgentNodeConfig):
                continue
            deployment = ModelDeploymentConfig.from_node(node)
            deployments.setdefault(deployment.model_key, deployment)
            self._tokenizers[node.name] = PromptTokenizer(node.execution)
        for model_key, deployment in deployments.items():
            replica_id = self._replica_id(model_key)
            self.recorder.record(
                "model_load_started",
                model_key=model_key,
                replica_id=replica_id,
                gpu_kind="v100",
                payload={
                    "reason": "static_baseline",
                    "max_model_len": deployment.serving.max_model_len,
                    "max_num_seqs": deployment.serving.max_num_seqs,
                    "max_num_batched_tokens": (
                        deployment.serving.max_num_batched_tokens
                    ),
                },
            )
            self._actors[model_key] = self._create_actor(deployment)
        self._load_refs = {
            model_key: _remote_call(actor, "load")
            for model_key, actor in self._actors.items()
        }
        if not self._load_refs:
            return
        self._load_executor = ThreadPoolExecutor(
            max_workers=len(self._load_refs),
            thread_name_prefix="baseline-model-load",
        )
        self._load_futures = {
            model_key: self._load_executor.submit(
                self._resolve_load,
                model_key,
                load_ref,
            )
            for model_key, load_ref in self._load_refs.items()
        }

    def generate(
        self,
        node: AgentNodeConfig,
        user_prompt: str,
        *,
        request_id: str,
        session_id: str,
        task_id: str,
    ) -> AgentState:
        deployment = ModelDeploymentConfig.from_node(node)
        load_result = self._wait_until_loaded(deployment.model_key)
        actor = self._actors[deployment.model_key]
        encoding = self._tokenizers[node.name].build_prompt(
            user_prompt,
            system_prompt=node.system_prompt,
        )
        self.recorder.record(
            "acquire_requested",
            session_id=session_id,
            task_id=task_id,
            node_id=node.name,
            acquire_id=request_id,
            model_key=deployment.model_key,
            payload={"input_tokens": encoding.input_tokens},
        )
        accelerator_id = f"baseline/v100:{load_result.physical_gpu_id}"
        replica_id = self._replica_id(deployment.model_key)
        self.recorder.record(
            "acquire_granted",
            session_id=session_id,
            task_id=task_id,
            node_id=node.name,
            acquire_id=request_id,
            model_key=deployment.model_key,
            replica_id=replica_id,
            accelerator_id=accelerator_id,
            gpu_kind="v100",
            payload={
                "input_tokens": encoding.input_tokens,
                "max_new_tokens": node.execution.max_new_tokens,
            },
        )
        with self._state_lock:
            reused = deployment.model_key in self._used_model_keys
            self._used_model_keys.add(deployment.model_key)
        if reused:
            self.recorder.record(
                "model_reused",
                session_id=session_id,
                task_id=task_id,
                node_id=node.name,
                acquire_id=request_id,
                model_key=deployment.model_key,
                replica_id=replica_id,
                accelerator_id=accelerator_id,
                gpu_kind="v100",
            )
        result = ray.get(
            _remote_call(
                actor,
                "invoke",
                request_id,
                encoding.token_ids,
                max_new_tokens=node.execution.max_new_tokens,
                generation=node.execution,
            )
        )
        if not isinstance(result, ReplicaInferenceResult):
            raise TypeError("static baseline engine returned an invalid result")
        payload: dict[str, object] = {
            "status": result.status,
            "duration_sec": result.duration_sec,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "input_tokens": result.input_tokens,
            "max_new_tokens": node.execution.max_new_tokens,
            "output_tokens": result.output_tokens,
            "hit_token_limit": result.hit_token_limit,
            "finish_reason": result.finish_reason,
            "queue_time_sec": result.queue_time_sec,
            "time_to_first_token_sec": result.time_to_first_token_sec,
            "replica_inflight_at_start": result.replica_inflight_at_start,
            "engine_failed": result.engine_failed,
        }
        self.recorder.record(
            "task_execution_finished",
            ts=result.finished_at,
            session_id=session_id,
            task_id=task_id,
            node_id=node.name,
            acquire_id=request_id,
            model_key=deployment.model_key,
            replica_id=replica_id,
            accelerator_id=accelerator_id,
            gpu_kind="v100",
            payload=payload,
        )
        if result.status != "success":
            raise RuntimeError(
                f"baseline inference failed: {node.name}: {result.error_type}"
            )
        text = self._tokenizers[node.name].decode(result.output_token_ids)
        return AgentState(messages=[AIMessage(content=text)])

    def shutdown(self) -> None:
        try:
            actors = tuple(self._actors.items())
            for model_key, _ in actors:
                self._wait_until_loaded(model_key)
            for model_key, _ in actors:
                result = self._load_results[model_key]
                self.recorder.record(
                    "model_eviction_started",
                    model_key=model_key,
                    replica_id=self._replica_id(model_key),
                    accelerator_id=f"baseline/v100:{result.physical_gpu_id}",
                    gpu_kind="v100",
                    payload={"reason": "trial_finished"},
                )
            shutdown_refs = [_remote_call(actor, "shutdown") for _, actor in actors]
            ray.get(shutdown_refs)
            for model_key, actor in actors:
                result = self._load_results[model_key]
                self.recorder.record(
                    "model_evicted",
                    model_key=model_key,
                    replica_id=self._replica_id(model_key),
                    accelerator_id=f"baseline/v100:{result.physical_gpu_id}",
                    gpu_kind="v100",
                    payload={"reason": "trial_finished"},
                )
                ray.kill(cast(Any, actor), no_restart=True)
        finally:
            if self._load_executor is not None:
                self._load_executor.shutdown(wait=True, cancel_futures=True)
            self._load_executor = None
            self._load_futures.clear()
            self._actors.clear()
            self._load_refs.clear()
            self._load_results.clear()
            self._used_model_keys.clear()

    def _wait_until_loaded(self, model_key: str) -> ReplicaLoadResult:
        try:
            future = self._load_futures[model_key]
        except KeyError:
            raise KeyError(
                f"static baseline model is not loading: {model_key}"
            ) from None
        return future.result()

    def _resolve_load(
        self,
        model_key: str,
        load_ref: ray.ObjectRef[Any],
    ) -> ReplicaLoadResult:
        result = ray.get(load_ref)
        if not isinstance(result, ReplicaLoadResult):
            raise TypeError("static baseline engine returned an invalid load result")
        with self._state_lock:
            self._load_results[model_key] = result
        self.recorder.record(
            "model_load_finished",
            model_key=model_key,
            replica_id=self._replica_id(model_key),
            accelerator_id=f"baseline/v100:{result.physical_gpu_id}",
            gpu_kind="v100",
            payload=result.model_dump(mode="json"),
        )
        return result

    def _create_actor(self, deployment: ModelDeploymentConfig) -> object:
        source_root = str(Path(__file__).resolve().parents[2])
        current_pythonpath = os.environ.get("PYTHONPATH")
        pythonpath = (
            source_root
            if not current_pythonpath
            else os.pathsep.join((source_root, current_pythonpath))
        )
        return StaticVLLMEngineActor.options(
            max_concurrency=1024,
            runtime_env={
                "py_executable": str(self.vllm_python),
                "env_vars": {
                    "PYTHONPATH": pythonpath,
                    "VLLM_ATTENTION_BACKEND": "XFORMERS",
                    "VLLM_NO_USAGE_STATS": "1",
                    "VLLM_USE_V1": "0",
                },
            },
        ).remote(deployment)

    @staticmethod
    def _replica_id(model_key: str) -> str:
        return f"baseline-{model_key}"


class LangGraphBaseline:
    def __init__(
        self,
        workflow: Workflow,
        *,
        functions: Mapping[str, NodeFunction],
        engines: StaticEnginePool,
        recorder: TraceRecorder,
    ) -> None:
        self.workflow = workflow
        self.functions = dict(functions)
        self.engines = engines
        self.recorder = recorder
        self._semaphores = {
            node.name: BoundedSemaphore(node.max_concurrency)
            for node in workflow.nodes
            if isinstance(node, FunctionNodeConfig)
        }
        self._fanin_lock = Lock()
        self._fanin_arrivals: dict[tuple[str, str], set[str]] = {}
        self.graph = self._build_graph()

    def invoke(self, request: BaselineSessionRequest) -> BaselineSessionResult:
        submitted_at = time.time()
        self.recorder.record(
            "session_submitted",
            ts=submitted_at,
            session_id=request.session_id,
        )
        try:
            result = self.graph.invoke(
                BaselineState(
                    session_id=request.session_id,
                    inputs=dict(request.inputs),
                    edge_outputs={},
                    final_output=None,
                )
            )
            if not isinstance(result, dict):
                raise TypeError("LangGraph baseline returned an invalid state")
            final_output = result.get("final_output")
            if not isinstance(final_output, dict):
                raise TypeError("LangGraph baseline produced no terminal output")
        except Exception as error:
            failed_at = time.time()
            self.recorder.record(
                "session_failed",
                ts=failed_at,
                session_id=request.session_id,
                payload={
                    "latency_sec": failed_at - submitted_at,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )
            raise
        completed_at = time.time()
        self.recorder.record(
            "session_completed",
            ts=completed_at,
            session_id=request.session_id,
            payload={"latency_sec": completed_at - submitted_at},
        )
        return BaselineSessionResult(
            session_id=request.session_id,
            final_output=cast(AgentState, final_output),
            submitted_at=submitted_at,
            completed_at=completed_at,
        )

    def _build_graph(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        graph = StateGraph(cast(Any, BaselineState))
        for node in self.workflow.nodes:
            graph.add_node(node.name, cast(Any, self._node_action(node)))
        for target, sources in self.workflow.graph.dependencies.items():
            if len(sources) == 1:
                graph.add_edge(sources[0], target)
            elif len(sources) > 1:
                graph.add_edge(list(sources), target)
        graph.add_edge(START, self.workflow.graph.entry_node)
        graph.add_edge(self.workflow.graph.terminal_node, END)
        return cast(CompiledStateGraph[Any, Any, Any, Any], graph.compile())

    def _node_action(self, node: NodeConfig) -> Callable[[BaselineState], object]:
        def action(state: BaselineState) -> BaselineStateUpdate:
            session_id = state["session_id"]
            task_id = str(uuid4())
            self.recorder.record(
                "task_started",
                session_id=session_id,
                task_id=task_id,
                node_id=node.name,
            )
            try:
                return self._execute_node(node, state, task_id, session_id)
            except Exception as error:
                self.recorder.record(
                    "task_failed",
                    session_id=session_id,
                    task_id=task_id,
                    node_id=node.name,
                    payload={
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                )
                raise

        return action

    def _execute_node(
        self,
        node: NodeConfig,
        state: BaselineState,
        task_id: str,
        session_id: str,
    ) -> BaselineStateUpdate:
        dependencies = self.workflow.graph.dependencies[node.name]
        states = {
            source: state["edge_outputs"][_edge_key(source, node.name)]
            for source in dependencies
        }
        if isinstance(node, AgentNodeConfig):
            context = build_prompt_context(state["inputs"], states)
            user_prompt = node.prompt_template.format_map(context)
            output = self.engines.generate(
                node,
                user_prompt,
                request_id=str(uuid4()),
                session_id=session_id,
                task_id=task_id,
            )
            successors = self.workflow.graph.adjacency[node.name]
            routed = (
                {target: output for target in successors}
                if successors
                else {"": output}
            )
        else:
            routed = self._run_function(
                node,
                state["inputs"],
                states,
                task_id,
                session_id,
            )
        update = BaselineStateUpdate(
            edge_outputs={
                _edge_key(node.name, target): output
                for target, output in routed.items()
            }
        )
        if node.name == self.workflow.graph.terminal_node:
            update["final_output"] = routed[""]
        self._record_edge_delivery(session_id, node.name, tuple(routed))
        self.recorder.record(
            "task_completed",
            session_id=session_id,
            task_id=task_id,
            node_id=node.name,
            payload={"status": "success"},
        )
        return update

    def _record_edge_delivery(
        self,
        session_id: str,
        source_node: str,
        targets: Sequence[str],
    ) -> None:
        for target_node in targets:
            if not target_node:
                continue
            timestamp = time.time()
            item_id = str(uuid4())
            for event_type in ("item_emitted", "item_enqueued", "item_dequeued"):
                node_id = target_node if event_type == "item_dequeued" else source_node
                self.recorder.record(
                    event_type,
                    ts=timestamp,
                    session_id=session_id,
                    item_id=item_id,
                    node_id=node_id,
                    source_node=source_node,
                    target_node=target_node,
                )
            dependencies = self.workflow.graph.dependencies[target_node]
            if len(dependencies) < 2:
                continue
            key = (session_id, target_node)
            with self._fanin_lock:
                arrived = self._fanin_arrivals.setdefault(key, set())
                if source_node in arrived:
                    raise RuntimeError("baseline fan-in source arrived twice")
                if not arrived:
                    self.recorder.record(
                        "fanin_wait",
                        ts=time.time(),
                        session_id=session_id,
                        item_id=item_id,
                        node_id=target_node,
                        source_node=source_node,
                        target_node=target_node,
                    )
                arrived.add(source_node)
                if len(arrived) == len(dependencies):
                    self.recorder.record(
                        "fanin_ready",
                        ts=time.time(),
                        session_id=session_id,
                        item_id=item_id,
                        node_id=target_node,
                        source_node=source_node,
                        target_node=target_node,
                    )
                    del self._fanin_arrivals[key]

    def _run_function(
        self,
        node: FunctionNodeConfig,
        inputs: Mapping[str, object],
        states: Mapping[str, AgentState],
        task_id: str,
        session_id: str,
    ) -> dict[str, AgentState]:
        function = self.functions[node.function]
        started_at = time.time()
        with self._semaphores[node.name]:
            output = function(
                MappingProxyType(dict(inputs)),
                MappingProxyType(dict(states)),
                MappingProxyType(dict(node.parameters)),
            )
        finished_at = time.time()
        self.recorder.record(
            "task_execution_finished",
            ts=finished_at,
            session_id=session_id,
            task_id=task_id,
            node_id=node.name,
            payload={
                "status": "success",
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_sec": finished_at - started_at,
            },
        )
        successors = self.workflow.graph.adjacency[node.name]
        if not successors:
            return {"": _agent_state(output)}
        if node.routing == "broadcast":
            state = _agent_state(output)
            return {successor: state for successor in successors}
        if not isinstance(output, Mapping):
            raise TypeError("targeted baseline function must return a mapping")
        if set(output) != set(successors):
            raise ValueError("targeted baseline output must match static successors")
        routed = cast(Mapping[str, object], output)
        return {target: _agent_state(routed[target]) for target in successors}


def run_langgraph_sessions(
    baseline: BaselineInvoker,
    requests: Sequence[BaselineSessionRequest],
    *,
    timeout_sec: float | None = None,
) -> tuple[BaselineSessionResult, ...]:
    if not requests:
        return ()
    if timeout_sec is not None and timeout_sec <= 0:
        raise ValueError("baseline timeout must be positive")
    start = time.monotonic()

    def invoke_at_arrival(request: BaselineSessionRequest) -> BaselineSessionResult:
        remaining = start + request.arrival_offset_sec - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        return baseline.invoke(request)

    executor = ThreadPoolExecutor(max_workers=len(requests))
    futures = [executor.submit(invoke_at_arrival, request) for request in requests]
    timed_out = False
    try:
        _, pending = wait(futures, timeout=timeout_sec)
        if pending:
            timed_out = True
            for future in pending:
                future.cancel()
            raise TimeoutError("LangGraph baseline trial timed out")
        return tuple(future.result() for future in futures)
    finally:
        executor.shutdown(wait=not timed_out, cancel_futures=timed_out)


def result_rows(
    run_id: str,
    results: Sequence[BaselineSessionResult],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "run_id": run_id,
            "session_id": result.session_id,
            "final_output": message_text(result.final_output),
            "submitted_at": result.submitted_at,
            "completed_at": result.completed_at,
        }
        for result in results
    )


def _edge_key(source: str, target: str) -> str:
    return f"{source}->{target}"


def _agent_state(value: object) -> AgentState:
    if not isinstance(value, dict) or "messages" not in value:
        raise TypeError("workflow functions must return AgentState")
    return cast(AgentState, value)


def _remote_call(
    actor: object,
    method_name: str,
    *args: object,
    **kwargs: object,
) -> ray.ObjectRef[Any]:
    method = getattr(actor, method_name)
    remote = getattr(method, "remote", None)
    if not callable(remote):
        raise TypeError(f"actor method is not remote: {method_name}")
    return cast(ray.ObjectRef[Any], remote(*args, **kwargs))
