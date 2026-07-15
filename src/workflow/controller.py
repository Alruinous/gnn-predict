from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from contextlib import suppress
from pathlib import Path
from threading import Lock
from typing import Any, Self, cast, overload

import ray
import yaml
from langchain.agents import AgentState
from pydantic import ConfigDict, JsonValue, TypeAdapter
from ray.util.queue import Queue

from workflow.artifacts import (
    AcceleratorConfig,
    PredictionCache,
    SchedulerConfig,
    load_prediction_cache,
)
from workflow.scheduler import ItemTraceReport
from workflow.schema import AgentNodeConfig, FunctionNodeConfig, Workflow
from workflow.storage import (
    RESULTS_FILENAME,
    SUMMARY_FILENAME,
    TRACE_FILENAME,
    write_run_summary,
)
from workflow.types import SessionState, WorkflowDataItem

NodeFunction = Callable[..., object]
QueueFactory = Callable[..., object]
ActorFactory = object
_SESSION_INPUTS_ADAPTER = TypeAdapter(
    dict[str, JsonValue],
    config=ConfigDict(strict=True),
)


@overload
def prepare_queues(
    workflow: Workflow,
) -> tuple[dict[str, Queue], dict[str, dict[str, Queue]]]: ...


@overload
def prepare_queues[QueueT](
    workflow: Workflow,
    *,
    queue_factory: Callable[..., QueueT],
) -> tuple[dict[str, QueueT], dict[str, dict[str, QueueT]]]: ...


def prepare_queues(
    workflow: Workflow,
    *,
    queue_factory: QueueFactory = Queue,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    inputs: dict[str, object] = {}
    try:
        _populate_input_queues(workflow, queue_factory, inputs)
    except Exception:
        for queue in inputs.values():
            _shutdown_queue(queue, force=True)
        raise
    return inputs, _prepare_output_queues(workflow, inputs)


class WorkflowController:
    def __init__(
        self,
        workflow: Workflow,
        *,
        functions: Mapping[str, NodeFunction],
        scheduler_config: SchedulerConfig,
        output_dir: str | Path,
        run_id: str,
        prediction_path: str | Path | None = None,
        replica_actor_factory: object | None = None,
        prompt_tokenizer_factory: Callable[..., object] | None = None,
        queue_factory: QueueFactory = Queue,
        scheduler_actor_factory: ActorFactory | None = None,
        worker_actor_factory: ActorFactory | None = None,
        result_store_actor_factory: ActorFactory | None = None,
        trace_writer_actor_factory: ActorFactory | None = None,
    ) -> None:
        self.workflow = workflow
        self.functions = dict(functions)
        self.scheduler_config = scheduler_config
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self._replica_actor_factory = replica_actor_factory
        self.predictions = (
            load_prediction_cache(prediction_path)
            if prediction_path is not None
            else None
        )
        self._validate_configuration()

        self._queue_factory = queue_factory
        self._scheduler_actor_factory = scheduler_actor_factory
        self._worker_actor_factory = worker_actor_factory
        self._result_store_actor_factory = result_store_actor_factory
        self._trace_writer_actor_factory = trace_writer_actor_factory
        self._prompt_tokenizer_factory = prompt_tokenizer_factory
        self.input_queues: dict[str, object] = {}
        self.output_queues: dict[str, dict[str, object]] = {}
        self.workers: dict[str, object] = {}
        self.scheduler: object | None = None
        self.result_store: object | None = None
        self.trace_writer: object | None = None
        self.scheduler_run_ref: object | None = None
        self.worker_run_refs: dict[str, object] = {}
        self._session_input_refs: dict[str, object] = {}
        self._accepted_session_ids: set[str] = set()
        self._terminal_futures: dict[str, Future[Any]] = {}
        self._latency_record_refs: dict[str, object] = {}
        self._latency_errors: list[Exception] = []
        self._latency_lock = Lock()
        self._force_stopping = False
        self._completed_results: dict[str, AgentState] = {}
        self._terminal_states: dict[str, SessionState] = {}
        self._started = False
        self._stopped = False
        self._admission_open = False
        self._owns_ray = False

    @classmethod
    def from_yaml(
        cls,
        workflow_path: str | Path,
        *,
        functions: Mapping[str, NodeFunction],
        scheduler_config: SchedulerConfig,
        prediction_path: str | Path | None,
        output_dir: str | Path,
        run_id: str,
        replica_actor_factory: object | None = None,
        prompt_tokenizer_factory: Callable[..., object] | None = None,
    ) -> Self:
        with Path(workflow_path).open(encoding="utf-8") as stream:
            workflow = Workflow.model_validate(yaml.safe_load(stream))
        return cls(
            workflow,
            functions=functions,
            scheduler_config=scheduler_config,
            prediction_path=prediction_path,
            output_dir=output_dir,
            run_id=run_id,
            replica_actor_factory=replica_actor_factory,
            prompt_tokenizer_factory=prompt_tokenizer_factory,
        )

    def start(self) -> None:
        if self._started or self._stopped:
            raise RuntimeError("workflow controller has already started or stopped")

        self._validate_output_collision()
        created_directories = _missing_directories(self.output_dir)
        try:
            if not ray.is_initialized():
                ray.init()
                self._owns_ray = True
            ray_node_ids = self._validate_live_accelerators()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            _populate_input_queues(
                self.workflow,
                self._queue_factory,
                self.input_queues,
            )
            self.output_queues = _prepare_output_queues(
                self.workflow,
                self.input_queues,
            )

            result_factory, trace_factory, scheduler_factory, worker_factory = (
                self._resolve_actor_factories()
            )
            self.result_store = _spawn_actor(
                result_factory,
                self.output_dir,
                self.run_id,
            )
            self.trace_writer = _spawn_actor(
                trace_factory,
                self.output_dir,
                self.run_id,
            )
            self.scheduler = _spawn_actor(
                scheduler_factory,
                workflow=self.workflow,
                scheduler_config=self.scheduler_config,
                predictions=self.predictions,
                trace_writer=self.trace_writer,
                ray_node_ids=ray_node_ids,
                run_id=self.run_id,
                replica_factory=self._replica_actor_factory,
            )
            node_map = self.workflow.node_map()
            for node_id in self.workflow.graph.topological_order:
                self.workers[node_id] = _spawn_actor(
                    worker_factory,
                    node=node_map[node_id],
                    dependencies=self.workflow.graph.dependencies[node_id],
                    successors=self.workflow.graph.adjacency[node_id],
                    input_queue=self.input_queues[node_id],
                    output_queues=self.output_queues[node_id],
                    scheduler=self.scheduler,
                    result_store=self.result_store,
                    functions=self.functions,
                    prompt_tokenizer_factory=self._prompt_tokenizer_factory,
                    acquire_timeout_sec=self.scheduler_config.acquire_timeout_sec,
                    grant_poll_interval_sec=(
                        self.scheduler_config.grant_poll_interval_sec
                    ),
                )
            self.scheduler_run_ref = _actor_call(self.scheduler, "run")
            _resolve(_actor_call(self.scheduler, "register_workers", self.workers))
            self.worker_run_refs = {
                node_id: _actor_call(worker, "run")
                for node_id, worker in self.workers.items()
            }
        except Exception:
            self._rollback_start(created_directories)
            raise
        self._started = True
        self._admission_open = True

    def submit(self, session_id: str, inputs: dict[str, JsonValue]) -> None:
        started_at = time.perf_counter()
        submitted_at = time.time()
        self._require_running()
        if not self._admission_open:
            raise RuntimeError("workflow admission is closed or stopped")
        try:
            self._raise_if_loop_ended()
        except Exception:
            self._admission_open = False
            raise
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must not be empty")
        if session_id in self._accepted_session_ids:
            raise ValueError(f"duplicate session id: {session_id}")
        validated_inputs = _SESSION_INPUTS_ADAPTER.validate_python(inputs)

        scheduler = self._scheduler()
        _resolve(
            _actor_call(
                scheduler,
                "register_session",
                session_id,
                submitted_at,
                True,
            )
        )
        self._accepted_session_ids.add(session_id)
        terminal_ref = _actor_call(scheduler, "wait_session_terminal", session_id)
        self._watch_session_terminal(
            session_id,
            terminal_ref,
            started_at,
            scheduler,
        )
        input_ref: object | None = None
        try:
            input_ref = ray.put(validated_inputs)
            with self._latency_lock:
                self._session_input_refs[session_id] = input_ref
            item = WorkflowDataItem(
                session_id=session_id,
                source_node=None,
                target_node=self.workflow.graph.entry_node,
                message=AgentState(messages=[]),
                session_input_ref=input_ref,
            )
            entry_queue = self.input_queues[self.workflow.graph.entry_node]
            cast(Queue, entry_queue).put(item)
            _resolve(
                _actor_call(
                    scheduler,
                    "record_item_enqueued",
                    ItemTraceReport(
                        session_id=session_id,
                        item_id=item.item_id,
                        source_node=None,
                        target_node=self.workflow.graph.entry_node,
                        emitted_at=submitted_at,
                        enqueued_at=time.time(),
                    ),
                )
            )
        except Exception as error:
            _resolve(
                _actor_call(
                    scheduler,
                    "cancel_session",
                    session_id,
                    type(error).__name__,
                    str(error),
                )
            )
            if input_ref is not None:
                with self._latency_lock:
                    self._session_input_refs.pop(session_id, None)
            raise

    def get_session_state(self, session_id: str) -> SessionState:
        if self._stopped:
            return self._terminal_states[session_id]
        value = _resolve(
            _actor_call(self._scheduler(), "get_session_state", session_id)
        )
        if not isinstance(value, SessionState):
            raise TypeError("scheduler returned an invalid session state")
        return value

    def has_result(self, session_id: str) -> bool:
        if self._stopped:
            return session_id in self._completed_results
        value = _resolve(_actor_call(self._result_store(), "has_result", session_id))
        if not isinstance(value, bool):
            raise TypeError("result store returned an invalid presence flag")
        return value

    def get_result(self, session_id: str) -> AgentState:
        if self._stopped:
            return self._completed_results[session_id]
        value = _resolve(_actor_call(self._result_store(), "get_result", session_id))
        if not isinstance(value, dict):
            raise TypeError("result store returned an invalid AgentState")
        return cast(AgentState, value)

    def drain_and_stop(self, timeout_sec: float | None = None) -> None:
        self._require_running()
        self._admission_open = False
        deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
        self._wait_for_drain(deadline)
        self._cache_terminal_state(deadline)

        worker_stop_refs = [
            _actor_call(worker, "stop") for worker in self.workers.values()
        ]
        self._wait_refs(worker_stop_refs, deadline)
        self._wait_refs(list(self.worker_run_refs.values()), deadline)
        scheduler = self._scheduler()
        self._wait_refs([_actor_call(scheduler, "evict_all")], deadline)
        self._wait_scheduler_flag("replicas_empty", deadline)
        self._wait_scheduler_flag("trace_idle", deadline)
        self._wait_refs([_actor_call(scheduler, "stop_loop")], deadline)
        if self.scheduler_run_ref is not None:
            self._wait_refs([self.scheduler_run_ref], deadline)
        self._wait_refs([_actor_call(self._result_store(), "close")], deadline)
        self._wait_refs([_actor_call(self._trace_writer(), "close")], deadline)
        for queue in self.input_queues.values():
            cast(Queue, queue).shutdown(force=False)
        with self._latency_lock:
            self._session_input_refs.clear()
        try:
            write_run_summary(self.output_dir, self.run_id)
        finally:
            self._stopped = True
            self._kill_runtime_actors()
            if self._owns_ray:
                ray.shutdown()

    def stop_now(self, timeout_sec: float = 5.0) -> None:
        if timeout_sec < 0:
            raise ValueError("stop timeout must be non-negative")
        self._admission_open = False
        if not self._started or self._stopped:
            self._stopped = True
            return

        deadline = time.monotonic() + timeout_sec
        refs: list[object] = []
        scheduler: object | None = None
        with self._latency_lock:
            self._force_stopping = True
        try:
            scheduler = self._scheduler()
            with suppress(Exception):
                refs.append(_actor_call(scheduler, "cancel_pending"))
            for worker in self.workers.values():
                with suppress(Exception):
                    refs.append(_actor_call(worker, "stop"))
            with suppress(Exception):
                refs.append(_actor_call(scheduler, "stop_loop"))
            refs.extend(self.worker_run_refs.values())
            if self.scheduler_run_ref is not None:
                refs.append(self.scheduler_run_ref)
            with suppress(Exception):
                self._wait_refs(refs, deadline)
            for ref in refs:
                if isinstance(ref, ray.ObjectRef):
                    with suppress(Exception):
                        ray.cancel(ref, force=False)
        finally:
            if scheduler is not None:
                with suppress(Exception):
                    _resolve(
                        _actor_call(scheduler, "force_kill_replicas"),
                        timeout=_remaining(deadline),
                    )
            self._kill_runtime_actors(wait_timeout_sec=1.0)
            for queue in self.input_queues.values():
                _shutdown_queue(queue, force=True)
            with self._latency_lock:
                self._session_input_refs.clear()
            self._stopped = True
            if self._owns_ray:
                ray.shutdown()

    def _watch_session_terminal(
        self,
        session_id: str,
        terminal_ref: object,
        started_at: float,
        scheduler: object,
    ) -> None:
        if not isinstance(terminal_ref, ray.ObjectRef):
            raise TypeError("terminal waiter must return a Ray ObjectRef")
        future = terminal_ref.future()
        with self._latency_lock:
            self._terminal_futures[session_id] = future

        def record_latency(completed: Future[Any]) -> None:
            try:
                state = completed.result()
                if state not in (SessionState.COMPLETED, SessionState.FAILED):
                    raise TypeError("terminal waiter returned a non-terminal state")
                finished_at = time.time()
                latency_sec = time.perf_counter() - started_at
                with self._latency_lock:
                    self._terminal_futures.pop(session_id, None)
                    self._session_input_refs.pop(session_id, None)
                    if self._force_stopping:
                        return
                record_ref = _actor_call(
                    scheduler,
                    "record_session_latency",
                    session_id,
                    latency_sec,
                    finished_at,
                )
                with self._latency_lock:
                    self._latency_record_refs[session_id] = record_ref
            except Exception as error:
                with self._latency_lock:
                    self._latency_errors.append(error)

        future.add_done_callback(record_latency)

    def _raise_latency_errors(self) -> None:
        with self._latency_lock:
            if self._latency_errors:
                error = self._latency_errors.pop(0)
                raise RuntimeError("session latency recording failed") from error
            refs = {
                session_id: ref
                for session_id, ref in self._latency_record_refs.items()
                if isinstance(ref, ray.ObjectRef)
            }
        if not refs:
            return
        ready, _ = ray.wait(list(refs.values()), num_returns=len(refs), timeout=0)
        if not ready:
            return
        ray.get(ready)
        ready_set = set(ready)
        with self._latency_lock:
            for session_id, ref in refs.items():
                if ref in ready_set:
                    self._latency_record_refs.pop(session_id, None)

    def _validate_configuration(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        missing = sorted(
            {
                node.function
                for node in self.workflow.nodes
                if isinstance(node, FunctionNodeConfig)
                and node.function not in self.functions
            }
        )
        if missing:
            raise KeyError(f"function registrations are missing: {', '.join(missing)}")
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise NotADirectoryError(self.output_dir)
        self._validate_output_collision()
        self._validate_agent_artifacts()

    def _validate_output_collision(self) -> None:
        for filename in (RESULTS_FILENAME, TRACE_FILENAME, SUMMARY_FILENAME):
            path = self.output_dir / filename
            if path.exists():
                raise FileExistsError(path)

    def _validate_agent_artifacts(self) -> None:
        agents = [
            node for node in self.workflow.nodes if isinstance(node, AgentNodeConfig)
        ]
        if not agents:
            return
        if self.predictions is None:
            raise ValueError("agent workflows require a prediction cache")
        if not self.scheduler_config.accelerators:
            raise ValueError("agent workflows require configured accelerators")
        gpu_kinds = {item.gpu_kind for item in self.scheduler_config.accelerators}
        for node in agents:
            model_path = Path(node.execution.model_path)
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            for gpu_kind in gpu_kinds:
                sequence_lengths = self.predictions.decode_sequence_lengths(
                    node.model.name,
                    gpu_kind,
                )
                if not sequence_lengths:
                    raise KeyError(
                        f"prediction coverage is missing: {node.model.name}/{gpu_kind}"
                    )
                if not any(
                    output_length >= node.execution.max_new_tokens
                    for sequence_length in sequence_lengths
                    for output_length in self.predictions.decode_output_lengths(
                        node.model.name,
                        gpu_kind,
                        sequence_length,
                    )
                ):
                    raise KeyError(
                        "prediction coverage has no output bucket: "
                        f"{node.model.name}/{gpu_kind}/"
                        f"max_new_tokens={node.execution.max_new_tokens}"
                    )
        executable = self.scheduler_config.vllm_python_executable
        if executable is None:
            raise ValueError("vllm_python_executable is required")
        executable_path = Path(executable)
        if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
            raise FileNotFoundError(executable_path)

    def _validate_live_accelerators(self) -> dict[str, str]:
        if not self.scheduler_config.accelerators:
            return {}
        live_nodes = [node for node in ray.nodes() if node.get("Alive")]
        by_hostname: dict[str, list[AcceleratorConfig]] = {}
        for accelerator in self.scheduler_config.accelerators:
            by_hostname.setdefault(accelerator.hostname, []).append(accelerator)
        ray_node_ids: dict[str, str] = {}
        for hostname, accelerators in by_hostname.items():
            matches = [
                node
                for node in live_nodes
                if node.get("NodeManagerHostname") == hostname
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"configured hostname must resolve to one live Ray node: {hostname}"
                )
            resources = matches[0].get("Resources", {})
            gpu_count = int(resources.get("GPU", 0))
            if len(accelerators) != gpu_count:
                raise ValueError(
                    f"configured accelerators do not match Ray GPUs on {hostname}"
                )
            reported_kinds = {
                key.removeprefix("accelerator_type:").casefold()
                for key, value in resources.items()
                if key.startswith("accelerator_type:") and value
            }
            if reported_kinds and accelerators[0].gpu_kind not in reported_kinds:
                raise ValueError(
                    f"configured GPU kind does not match Ray node: {hostname}"
                )
            node_id = matches[0].get("NodeID")
            if not isinstance(node_id, str) or not node_id:
                raise ValueError(f"Ray node has no stable node id: {hostname}")
            ray_node_ids[hostname] = node_id
        return ray_node_ids

    def _resolve_actor_factories(
        self,
    ) -> tuple[ActorFactory, ActorFactory, ActorFactory, ActorFactory]:
        result_factory = self._result_store_actor_factory
        if result_factory is None:
            from workflow.storage import ResultStoreActor

            result_factory = ResultStoreActor
        trace_factory = self._trace_writer_actor_factory
        if trace_factory is None:
            from workflow.storage import TraceWriterActor

            trace_factory = TraceWriterActor
        scheduler_factory = self._scheduler_actor_factory
        if scheduler_factory is None:
            from workflow.scheduler import SchedulerActor

            scheduler_factory = SchedulerActor
        worker_factory = self._worker_actor_factory
        if worker_factory is None:
            from workflow.worker import NodeWorkerActor

            worker_factory = NodeWorkerActor
        return (
            result_factory,
            trace_factory,
            scheduler_factory,
            worker_factory,
        )

    def _require_running(self) -> None:
        if not self._started or self._stopped:
            raise RuntimeError("workflow controller must be started and not stopped")

    def _scheduler(self) -> object:
        if self.scheduler is None:
            raise RuntimeError("scheduler actor is unavailable")
        return self.scheduler

    def _result_store(self) -> object:
        if self.result_store is None:
            raise RuntimeError("result store actor is unavailable")
        return self.result_store

    def _trace_writer(self) -> object:
        if self.trace_writer is None:
            raise RuntimeError("trace writer actor is unavailable")
        return self.trace_writer

    def _cache_terminal_state(self, deadline: float | None) -> None:
        scheduler = self._scheduler()
        result_store = self._result_store()
        for session_id in sorted(self._accepted_session_ids):
            state = _resolve(
                _actor_call(scheduler, "get_session_state", session_id),
                timeout=_remaining(deadline),
            )
            if state not in (SessionState.COMPLETED, SessionState.FAILED):
                raise RuntimeError(f"session is not terminal after drain: {session_id}")
            self._terminal_states[session_id] = cast(SessionState, state)
            has_result = _resolve(
                _actor_call(result_store, "has_result", session_id),
                timeout=_remaining(deadline),
            )
            if not isinstance(has_result, bool):
                raise TypeError("result store returned an invalid presence flag")
            if not has_result:
                continue
            result = _resolve(
                _actor_call(result_store, "get_result", session_id),
                timeout=_remaining(deadline),
            )
            if not isinstance(result, dict):
                raise TypeError("result store returned an invalid AgentState")
            self._completed_results[session_id] = cast(AgentState, result)

    def _wait_scheduler_flag(self, method_name: str, deadline: float | None) -> None:
        scheduler = self._scheduler()
        while True:
            self._raise_if_loop_ended(include_workers=False)
            value = _resolve(
                _actor_call(scheduler, method_name),
                timeout=_remaining(deadline),
            )
            if not isinstance(value, bool):
                raise TypeError(f"scheduler {method_name} returned a non-boolean value")
            if value:
                return
            time.sleep(0.02)

    def _kill_runtime_actors(self, *, wait_timeout_sec: float = 0.0) -> None:
        actors = tuple(
            actor
            for actor in (
                *self.workers.values(),
                self.scheduler,
                self.result_store,
                self.trace_writer,
            )
            if isinstance(actor, ray.actor.ActorHandle)
        )
        termination_refs: list[ray.ObjectRef[Any]] = []
        for actor in actors:
            with suppress(Exception):
                termination_refs.append(actor.__ray_terminate__.remote())
            with suppress(Exception):
                ray.kill(actor, no_restart=True)
        if wait_timeout_sec <= 0 or not termination_refs:
            return
        ray.wait(
            termination_refs,
            num_returns=len(termination_refs),
            timeout=wait_timeout_sec,
        )

    def _rollback_start(self, created_directories: tuple[Path, ...]) -> None:
        self._kill_runtime_actors()
        for queue in self.input_queues.values():
            with suppress(Exception):
                cast(Queue, queue).shutdown(force=True)
        self.input_queues.clear()
        self.output_queues.clear()
        self.workers.clear()
        self.worker_run_refs.clear()
        self.scheduler = None
        self.result_store = None
        self.trace_writer = None
        self.scheduler_run_ref = None
        for filename in (RESULTS_FILENAME, TRACE_FILENAME, SUMMARY_FILENAME):
            with suppress(OSError):
                (self.output_dir / filename).unlink()
        for directory in created_directories:
            with suppress(OSError):
                directory.rmdir()
        if self._owns_ray:
            ray.shutdown()
            self._owns_ray = False

    def _wait_for_drain(self, deadline: float | None) -> None:
        scheduler = self._scheduler()
        while True:
            self._raise_if_loop_ended()
            self._raise_latency_errors()
            if _resolve(
                _actor_call(scheduler, "drain_complete"),
                timeout=_remaining(deadline),
            ):
                return
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("workflow drain timed out")
            time.sleep(0.02)

    def _raise_if_loop_ended(self, *, include_workers: bool = True) -> None:
        candidates = [self.scheduler_run_ref]
        if include_workers:
            candidates.extend(self.worker_run_refs.values())
        refs = [ref for ref in candidates if isinstance(ref, ray.ObjectRef)]
        if not refs:
            return
        ready, _ = ray.wait(refs, num_returns=len(refs), timeout=0)
        if ready:
            ray.get(ready)
            raise RuntimeError("workflow runtime loop stopped before drain completed")

    @staticmethod
    def _wait_refs(refs: list[object], deadline: float | None) -> None:
        object_refs = [ref for ref in refs if isinstance(ref, ray.ObjectRef)]
        if not object_refs:
            return
        timeout = _remaining(deadline)
        ready, remaining = ray.wait(
            object_refs,
            num_returns=len(object_refs),
            timeout=timeout,
        )
        ray.get(ready)
        if remaining:
            raise TimeoutError("workflow shutdown timed out")


def _spawn_actor(factory: ActorFactory, *args: object, **kwargs: object) -> object:
    remote = getattr(factory, "remote", None)
    if callable(remote):
        return remote(*args, **kwargs)
    if callable(factory):
        return cast(Callable[..., object], factory)(*args, **kwargs)
    raise TypeError("actor factory must be callable or expose remote()")


def _populate_input_queues[QueueT](
    workflow: Workflow,
    queue_factory: Callable[..., QueueT],
    inputs: dict[str, QueueT],
) -> None:
    for node in workflow.nodes:
        inputs[node.name] = queue_factory(maxsize=node.queue_capacity)


def _prepare_output_queues[QueueT](
    workflow: Workflow,
    inputs: Mapping[str, QueueT],
) -> dict[str, dict[str, QueueT]]:
    return {
        source: {target: inputs[target] for target in targets}
        for source, targets in workflow.graph.adjacency.items()
    }


def _shutdown_queue(queue: object, *, force: bool) -> None:
    shutdown = getattr(queue, "shutdown", None)
    if callable(shutdown):
        with suppress(Exception):
            shutdown(force=force)


def _actor_call(
    actor: object,
    method_name: str,
    *args: object,
    **kwargs: object,
) -> object:
    method = getattr(actor, method_name)
    remote = getattr(method, "remote", None)
    if callable(remote):
        return remote(*args, **kwargs)
    if callable(method):
        return method(*args, **kwargs)
    raise TypeError(f"actor method is not callable: {method_name}")


def _resolve(value: object, *, timeout: float | None = None) -> object:
    if isinstance(value, ray.ObjectRef):
        return ray.get(value, timeout=timeout)
    return value


def _remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("workflow operation timed out")
    return remaining


def _missing_directories(path: Path) -> tuple[Path, ...]:
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    return tuple(missing)
