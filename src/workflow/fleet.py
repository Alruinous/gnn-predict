from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, cast

import ray
from langchain.agents import AgentState
from pydantic import ConfigDict, JsonValue, TypeAdapter
from ray.util.queue import Queue

from workflow.actor_support import dispatch, resolve
from workflow.artifacts import (
    AcceleratorConfig,
    ResourceContractCache,
    SchedulerConfig,
    load_resource_contract_cache,
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


@dataclass
class _WorkflowBinding:
    workflow: Workflow
    functions: dict[str, NodeFunction]
    output_dir: Path
    run_id: str
    prompt_tokenizer_factory: Callable[..., object] | None
    input_queues: dict[str, object] = field(default_factory=dict)
    output_queues: dict[str, dict[str, object]] = field(default_factory=dict)
    workers: dict[str, object] = field(default_factory=dict)
    result_store: object | None = None
    worker_run_refs: dict[str, object] = field(default_factory=dict)
    session_input_refs: dict[str, object] = field(default_factory=dict)
    accepted_session_ids: set[str] = field(default_factory=set)
    terminal_futures: dict[str, Future[Any]] = field(default_factory=dict)
    latency_record_refs: dict[str, object] = field(default_factory=dict)
    latency_errors: list[Exception] = field(default_factory=list)
    latency_lock: Lock = field(default_factory=Lock)
    force_stopping: bool = False
    completed_results: dict[str, AgentState] = field(default_factory=dict)
    terminal_states: dict[str, SessionState] = field(default_factory=dict)
    admission_open: bool = False
    stopped: bool = False


class WorkflowFleet:
    def __init__(
        self,
        *,
        scheduler_config: SchedulerConfig,
        output_dir: str | Path,
        run_id: str,
        prediction_path: str | Path | None = None,
        replica_actor_factory: object | None = None,
        queue_factory: QueueFactory = Queue,
        scheduler_actor_factory: ActorFactory | None = None,
        worker_actor_factory: ActorFactory | None = None,
        result_store_actor_factory: ActorFactory | None = None,
        trace_writer_actor_factory: ActorFactory | None = None,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.predictions = (
            load_resource_contract_cache(prediction_path)
            if prediction_path is not None
            else None
        )
        self._replica_actor_factory = replica_actor_factory
        self._queue_factory = queue_factory
        self._scheduler_actor_factory = scheduler_actor_factory
        self._worker_actor_factory = worker_actor_factory
        self._result_store_actor_factory = result_store_actor_factory
        self._trace_writer_actor_factory = trace_writer_actor_factory
        self.scheduler: object | None = None
        self.scheduler_run_ref: object | None = None
        self.trace_writer: object | None = None
        self._ray_node_ids: dict[str, str] = {}
        self._bindings: dict[str, _WorkflowBinding] = {}
        # Drained bindings move here (not deleted) so get_session_state /
        # has_result / get_result can still serve cached terminal state after
        # a drain, while register_workflow only rejects re-registration
        # against a still-live binding.
        self._drained_bindings: dict[str, _WorkflowBinding] = {}
        self._result_output_dirs: list[Path] = []
        self._started = False
        self._stopped = False
        self._owns_ray = False
        self._force_stopping = False

    def start(self) -> None:
        if self._started or self._stopped:
            raise RuntimeError("workflow fleet has already started or stopped")
        _validate_fleet_output_collision(self.output_dir)
        created_directories = _missing_directories(self.output_dir)
        try:
            if not ray.is_initialized():
                ray.init()
                self._owns_ray = True
            self._ray_node_ids = _validate_live_accelerators(self.scheduler_config)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            trace_factory = self._trace_writer_actor_factory
            if trace_factory is None:
                from workflow.storage import TraceWriterActor

                trace_factory = TraceWriterActor
            self.trace_writer = _spawn_actor(
                trace_factory, self.output_dir, self.run_id
            )
        except Exception:
            self._kill_fleet_actors()
            for filename in (TRACE_FILENAME, SUMMARY_FILENAME):
                with suppress(OSError):
                    (self.output_dir / filename).unlink()
            for directory in created_directories:
                with suppress(OSError):
                    directory.rmdir()
            if self._owns_ray:
                ray.shutdown()
                self._owns_ray = False
            raise
        self._started = True

    def register_workflow(
        self,
        workflow: Workflow,
        *,
        functions: Mapping[str, NodeFunction],
        output_dir: str | Path,
        priority_weight: float = 1.0,
        prompt_tokenizer_factory: Callable[..., object] | None = None,
        queue_factory: QueueFactory | None = None,
    ) -> None:
        # Every registered workflow shares the fleet's own run_id: trace
        # events already share one run_id via the single scheduler actor, and
        # write_run_summary() validates result records against that same
        # run_id when it sums result_count across every binding's directory.
        if not self._started or self._stopped:
            raise RuntimeError("workflow fleet must be started and not stopped")
        if workflow.workflow_name in self._bindings:
            raise ValueError(f"workflow already registered: {workflow.workflow_name}")

        binding_output_dir = Path(output_dir)
        _validate_workflow_registration(
            workflow,
            functions,
            self.scheduler_config,
            self.predictions,
            binding_output_dir,
        )

        binding = _WorkflowBinding(
            workflow=workflow,
            functions=dict(functions),
            output_dir=binding_output_dir,
            run_id=self.run_id,
            prompt_tokenizer_factory=prompt_tokenizer_factory,
        )
        created_directories = _missing_directories(binding_output_dir)
        effective_queue_factory = queue_factory or self._queue_factory
        try:
            binding_output_dir.mkdir(parents=True, exist_ok=True)
            _populate_input_queues(
                workflow, effective_queue_factory, binding.input_queues
            )
            binding.output_queues = _prepare_output_queues(
                workflow, binding.input_queues
            )

            result_factory, worker_factory = self._resolve_binding_actor_factories()
            binding.result_store = _spawn_actor(
                result_factory, binding_output_dir, self.run_id
            )

            self._ensure_scheduler_spawned(workflow, priority_weight)

            node_map = workflow.node_map()
            for node_id in workflow.graph.topological_order:
                binding.workers[node_id] = _spawn_actor(
                    worker_factory,
                    node=node_map[node_id],
                    dependencies=workflow.graph.dependencies[node_id],
                    successors=workflow.graph.adjacency[node_id],
                    input_queue=binding.input_queues[node_id],
                    output_queues=binding.output_queues[node_id],
                    scheduler=self.scheduler,
                    result_store=binding.result_store,
                    functions=binding.functions,
                    prompt_tokenizer_factory=binding.prompt_tokenizer_factory,
                    acquire_timeout_sec=self.scheduler_config.acquire_timeout_sec,
                    grant_poll_interval_sec=self.scheduler_config.grant_poll_interval_sec,
                )
            # Only start the shared scheduler's run loop once, and only after
            # this workflow's own workers exist, matching the single-workflow
            # controller's original ordering (spawn workers, then run(), then
            # register_workers()).
            if self.scheduler_run_ref is None:
                self.scheduler_run_ref = dispatch(self.scheduler, "run")
            resolve(
                dispatch(
                    self.scheduler,
                    "register_workers",
                    workflow.workflow_name,
                    binding.workers,
                )
            )
            binding.worker_run_refs = {
                node_id: dispatch(worker, "run")
                for node_id, worker in binding.workers.items()
            }
        except Exception:
            self._rollback_registration(binding, created_directories)
            raise
        binding.admission_open = True
        self._bindings[workflow.workflow_name] = binding
        self._result_output_dirs.append(binding_output_dir)

    def submit(
        self,
        workflow_name: str,
        session_id: str,
        inputs: dict[str, JsonValue],
    ) -> None:
        started_at = time.perf_counter()
        submitted_at = time.time()
        binding = self._binding(workflow_name)
        self._require_binding_running(binding)
        if not binding.admission_open:
            raise RuntimeError("workflow admission is closed or stopped")
        try:
            self._raise_if_loop_ended(binding)
        except Exception:
            binding.admission_open = False
            raise
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must not be empty")
        if session_id in binding.accepted_session_ids:
            raise ValueError(f"duplicate session id: {session_id}")
        validated_inputs = _SESSION_INPUTS_ADAPTER.validate_python(inputs)

        scheduler = self._scheduler()
        resolve(
            dispatch(
                scheduler,
                "register_session",
                session_id,
                workflow_name,
                submitted_at,
                True,
            )
        )
        binding.accepted_session_ids.add(session_id)
        terminal_ref = dispatch(scheduler, "wait_session_terminal", session_id)
        self._watch_session_terminal(binding, session_id, terminal_ref, started_at)
        input_ref: object | None = None
        try:
            input_ref = ray.put(validated_inputs)
            with binding.latency_lock:
                binding.session_input_refs[session_id] = input_ref
            item = WorkflowDataItem(
                session_id=session_id,
                source_node=None,
                target_node=binding.workflow.graph.entry_node,
                message=AgentState(messages=[]),
                session_input_ref=input_ref,
            )
            entry_queue = binding.input_queues[binding.workflow.graph.entry_node]
            cast(Queue, entry_queue).put(item)
            resolve(
                dispatch(
                    scheduler,
                    "record_item_enqueued",
                    ItemTraceReport(
                        session_id=session_id,
                        item_id=item.item_id,
                        source_node=None,
                        target_node=binding.workflow.graph.entry_node,
                        emitted_at=submitted_at,
                        enqueued_at=time.time(),
                    ),
                )
            )
        except Exception as error:
            resolve(
                dispatch(
                    scheduler,
                    "cancel_session",
                    session_id,
                    type(error).__name__,
                    str(error),
                )
            )
            if input_ref is not None:
                with binding.latency_lock:
                    binding.session_input_refs.pop(session_id, None)
            raise

    def get_session_state(self, workflow_name: str, session_id: str) -> SessionState:
        binding = self._binding(workflow_name)
        if binding.stopped:
            return binding.terminal_states[session_id]
        value = resolve(dispatch(self._scheduler(), "get_session_state", session_id))
        if not isinstance(value, SessionState):
            raise TypeError("scheduler returned an invalid session state")
        return value

    def has_result(self, workflow_name: str, session_id: str) -> bool:
        binding = self._binding(workflow_name)
        if binding.stopped:
            return session_id in binding.completed_results
        value = resolve(dispatch(self._result_store(binding), "has_result", session_id))
        if not isinstance(value, bool):
            raise TypeError("result store returned an invalid presence flag")
        return value

    def get_result(self, workflow_name: str, session_id: str) -> AgentState:
        binding = self._binding(workflow_name)
        if binding.stopped:
            return binding.completed_results[session_id]
        value = resolve(dispatch(self._result_store(binding), "get_result", session_id))
        if not isinstance(value, dict):
            raise TypeError("result store returned an invalid AgentState")
        return cast(AgentState, value)

    def drain_workflow(
        self,
        workflow_name: str,
        timeout_sec: float | None = None,
    ) -> None:
        binding = self._binding(workflow_name)
        self._require_binding_running(binding)
        binding.admission_open = False
        deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
        self._wait_for_drain(binding, deadline)
        self._cache_terminal_state(binding, deadline)

        scheduler = self._scheduler()
        worker_stop_refs = [
            dispatch(worker, "stop") for worker in binding.workers.values()
        ]
        _wait_refs(worker_stop_refs, deadline)
        _wait_refs(list(binding.worker_run_refs.values()), deadline)
        _wait_refs(
            [dispatch(self._result_store(binding), "close")],
            deadline,
        )
        for queue in binding.input_queues.values():
            cast(Queue, queue).shutdown(force=False)
        with binding.latency_lock:
            binding.session_input_refs.clear()
        resolve(
            dispatch(scheduler, "deregister_workflow", workflow_name),
            timeout=_remaining(deadline),
        )
        binding.stopped = True
        self._kill_binding_actors(binding)
        del self._bindings[workflow_name]
        self._drained_bindings[workflow_name] = binding

    def shutdown(self, timeout_sec: float | None = None) -> None:
        if not self._started or self._stopped:
            self._stopped = True
            return
        deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
        for workflow_name in tuple(self._bindings):
            self.drain_workflow(workflow_name, _remaining(deadline))

        # The scheduler only exists once a workflow has been successfully
        # registered at least once; a fleet that started but never completed
        # a registration has nothing scheduler-side to drain.
        if self.scheduler is not None:
            scheduler = self.scheduler
            _wait_refs([dispatch(scheduler, "evict_all")], deadline)
            self._wait_scheduler_flag("replicas_empty", deadline)
            self._wait_scheduler_flag("trace_idle", deadline)
            _wait_refs([dispatch(scheduler, "stop_loop")], deadline)
            if self.scheduler_run_ref is not None:
                _wait_refs([self.scheduler_run_ref], deadline)
        try:
            if self.trace_writer is not None:
                _wait_refs([dispatch(self.trace_writer, "close")], deadline)
            if self.scheduler is not None:
                write_run_summary(
                    self.output_dir,
                    self.run_id,
                    result_directories=self._result_output_dirs or None,
                )
        finally:
            self._stopped = True
            self._kill_fleet_actors()
            if self._owns_ray:
                ray.shutdown()

    def stop_now(self, timeout_sec: float = 5.0) -> None:
        if timeout_sec < 0:
            raise ValueError("stop timeout must be non-negative")
        for binding in self._bindings.values():
            binding.admission_open = False
        if not self._started or self._stopped:
            self._stopped = True
            return

        deadline = time.monotonic() + timeout_sec
        refs: list[object] = []
        scheduler: object | None = None
        self._force_stopping = True
        for binding in self._bindings.values():
            with binding.latency_lock:
                binding.force_stopping = True
        try:
            scheduler = self.scheduler
            if scheduler is not None:
                with suppress(Exception):
                    refs.append(dispatch(scheduler, "cancel_pending"))
            for binding in self._bindings.values():
                for worker in binding.workers.values():
                    with suppress(Exception):
                        refs.append(dispatch(worker, "stop"))
            if scheduler is not None:
                with suppress(Exception):
                    refs.append(dispatch(scheduler, "stop_loop"))
            for binding in self._bindings.values():
                refs.extend(binding.worker_run_refs.values())
            if self.scheduler_run_ref is not None:
                refs.append(self.scheduler_run_ref)
            with suppress(Exception):
                _wait_refs(refs, deadline)
            for ref in refs:
                if isinstance(ref, ray.ObjectRef):
                    with suppress(Exception):
                        ray.cancel(ref, force=False)
        finally:
            if scheduler is not None:
                with suppress(Exception):
                    resolve(
                        dispatch(scheduler, "force_kill_replicas"),
                        timeout=_remaining(deadline),
                    )
            self._kill_fleet_actors(wait_timeout_sec=1.0)
            for workflow_name, binding in tuple(self._bindings.items()):
                self._kill_binding_actors(binding, wait_timeout_sec=1.0)
                for queue in binding.input_queues.values():
                    _shutdown_queue(queue, force=True)
                with binding.latency_lock:
                    binding.session_input_refs.clear()
                binding.stopped = True
                del self._bindings[workflow_name]
                self._drained_bindings[workflow_name] = binding
            self._stopped = True
            if self._owns_ray:
                ray.shutdown()

    # -- internal helpers -------------------------------------------------

    def _ensure_scheduler_spawned(
        self,
        workflow: Workflow,
        priority_weight: float,
    ) -> None:
        if self.scheduler is None:
            scheduler_factory = self._scheduler_actor_factory
            if scheduler_factory is None:
                from workflow.scheduler import SchedulerActor

                scheduler_factory = SchedulerActor
            self.scheduler = _spawn_actor(
                scheduler_factory,
                workflow=workflow,
                scheduler_config=self.scheduler_config,
                predictions=self.predictions,
                trace_writer=self.trace_writer,
                ray_node_ids=self._ray_node_ids,
                run_id=self.run_id,
                replica_factory=self._replica_actor_factory,
                priority_weight=priority_weight,
            )
        else:
            resolve(
                dispatch(
                    self.scheduler,
                    "register_workflow",
                    workflow,
                    priority_weight=priority_weight,
                )
            )

    def _resolve_binding_actor_factories(self) -> tuple[ActorFactory, ActorFactory]:
        result_factory = self._result_store_actor_factory
        if result_factory is None:
            from workflow.storage import ResultStoreActor

            result_factory = ResultStoreActor
        worker_factory = self._worker_actor_factory
        if worker_factory is None:
            from workflow.worker import NodeWorkerActor

            worker_factory = NodeWorkerActor
        return result_factory, worker_factory

    def _binding(self, workflow_name: str) -> _WorkflowBinding:
        binding = self._bindings.get(workflow_name) or self._drained_bindings.get(
            workflow_name
        )
        if binding is None:
            raise KeyError(f"unknown workflow: {workflow_name}")
        return binding

    def _scheduler(self) -> object:
        if self.scheduler is None:
            raise RuntimeError("scheduler actor is unavailable")
        return self.scheduler

    def _result_store(self, binding: _WorkflowBinding) -> object:
        if binding.result_store is None:
            raise RuntimeError("result store actor is unavailable")
        return binding.result_store

    def _require_binding_running(self, binding: _WorkflowBinding) -> None:
        if binding.stopped:
            raise RuntimeError("workflow must be started and not stopped")

    def _watch_session_terminal(
        self,
        binding: _WorkflowBinding,
        session_id: str,
        terminal_ref: object,
        started_at: float,
    ) -> None:
        if not isinstance(terminal_ref, ray.ObjectRef):
            raise TypeError("terminal waiter must return a Ray ObjectRef")
        future = terminal_ref.future()
        with binding.latency_lock:
            binding.terminal_futures[session_id] = future

        def record_latency(completed: Future[Any]) -> None:
            try:
                state = completed.result()
                if state not in (SessionState.COMPLETED, SessionState.FAILED):
                    raise TypeError("terminal waiter returned a non-terminal state")
                finished_at = time.time()
                latency_sec = time.perf_counter() - started_at
                with binding.latency_lock:
                    binding.terminal_futures.pop(session_id, None)
                    binding.session_input_refs.pop(session_id, None)
                    if binding.force_stopping:
                        return
                record_ref = dispatch(
                    self._scheduler(),
                    "record_session_latency",
                    session_id,
                    latency_sec,
                    finished_at,
                )
                with binding.latency_lock:
                    binding.latency_record_refs[session_id] = record_ref
            except Exception as error:
                with binding.latency_lock:
                    binding.latency_errors.append(error)

        future.add_done_callback(record_latency)

    def _raise_latency_errors(self, binding: _WorkflowBinding) -> None:
        with binding.latency_lock:
            if binding.latency_errors:
                error = binding.latency_errors.pop(0)
                raise RuntimeError("session latency recording failed") from error
            refs = {
                session_id: ref
                for session_id, ref in binding.latency_record_refs.items()
                if isinstance(ref, ray.ObjectRef)
            }
        if not refs:
            return
        ready, _ = ray.wait(list(refs.values()), num_returns=len(refs), timeout=0)
        if not ready:
            return
        ray.get(ready)
        ready_set = set(ready)
        with binding.latency_lock:
            for session_id, ref in refs.items():
                if ref in ready_set:
                    binding.latency_record_refs.pop(session_id, None)

    def _cache_terminal_state(
        self,
        binding: _WorkflowBinding,
        deadline: float | None,
    ) -> None:
        scheduler = self._scheduler()
        result_store = self._result_store(binding)
        for session_id in sorted(binding.accepted_session_ids):
            state = resolve(
                dispatch(scheduler, "get_session_state", session_id),
                timeout=_remaining(deadline),
            )
            if state not in (SessionState.COMPLETED, SessionState.FAILED):
                raise RuntimeError(f"session is not terminal after drain: {session_id}")
            binding.terminal_states[session_id] = cast(SessionState, state)
            has_result = resolve(
                dispatch(result_store, "has_result", session_id),
                timeout=_remaining(deadline),
            )
            if not isinstance(has_result, bool):
                raise TypeError("result store returned an invalid presence flag")
            if not has_result:
                continue
            result = resolve(
                dispatch(result_store, "get_result", session_id),
                timeout=_remaining(deadline),
            )
            if not isinstance(result, dict):
                raise TypeError("result store returned an invalid AgentState")
            binding.completed_results[session_id] = cast(AgentState, result)

    def _wait_scheduler_flag(self, method_name: str, deadline: float | None) -> None:
        scheduler = self._scheduler()
        while True:
            self._raise_if_fleet_loop_ended()
            value = resolve(
                dispatch(scheduler, method_name),
                timeout=_remaining(deadline),
            )
            if not isinstance(value, bool):
                raise TypeError(f"scheduler {method_name} returned a non-boolean value")
            if value:
                return
            time.sleep(0.02)

    def _kill_fleet_actors(self, *, wait_timeout_sec: float = 0.0) -> None:
        actors = tuple(
            actor
            for actor in (self.scheduler, self.trace_writer)
            if isinstance(actor, ray.actor.ActorHandle)
        )
        _kill_actors(actors, wait_timeout_sec=wait_timeout_sec)

    def _kill_binding_actors(
        self,
        binding: _WorkflowBinding,
        *,
        wait_timeout_sec: float = 0.0,
    ) -> None:
        actors = tuple(
            actor
            for actor in (*binding.workers.values(), binding.result_store)
            if isinstance(actor, ray.actor.ActorHandle)
        )
        _kill_actors(actors, wait_timeout_sec=wait_timeout_sec)

    def rollback_start(self, created_directories: tuple[Path, ...]) -> None:
        """Tear down a fleet whose start() succeeded but whose only caller's
        registration failed before any workflow was fully registered."""
        self._kill_fleet_actors()
        for filename in (TRACE_FILENAME, SUMMARY_FILENAME):
            with suppress(OSError):
                (self.output_dir / filename).unlink()
        for directory in created_directories:
            with suppress(OSError):
                directory.rmdir()
        self.scheduler = None
        self.scheduler_run_ref = None
        self.trace_writer = None
        if self._owns_ray:
            ray.shutdown()
            self._owns_ray = False
        self._started = False

    def _rollback_registration(
        self,
        binding: _WorkflowBinding,
        created_directories: tuple[Path, ...],
    ) -> None:
        self._kill_binding_actors(binding)
        for queue in binding.input_queues.values():
            with suppress(Exception):
                cast(Queue, queue).shutdown(force=True)
        with suppress(OSError):
            (binding.output_dir / RESULTS_FILENAME).unlink()
        for directory in created_directories:
            with suppress(OSError):
                directory.rmdir()

    def _wait_for_drain(
        self,
        binding: _WorkflowBinding,
        deadline: float | None,
    ) -> None:
        scheduler = self._scheduler()
        while True:
            self._raise_if_loop_ended(binding)
            self._raise_latency_errors(binding)
            if resolve(
                dispatch(scheduler, "drain_complete", binding.workflow.workflow_name),
                timeout=_remaining(deadline),
            ):
                return
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("workflow drain timed out")
            time.sleep(0.02)

    def _raise_if_loop_ended(
        self,
        binding: _WorkflowBinding,
        *,
        include_workers: bool = True,
    ) -> None:
        candidates = [self.scheduler_run_ref]
        if include_workers:
            candidates.extend(binding.worker_run_refs.values())
        refs = [ref for ref in candidates if isinstance(ref, ray.ObjectRef)]
        if not refs:
            return
        ready, _ = ray.wait(refs, num_returns=len(refs), timeout=0)
        if ready:
            ray.get(ready)
            raise RuntimeError("workflow runtime loop stopped before drain completed")

    def _raise_if_fleet_loop_ended(self) -> None:
        if not isinstance(self.scheduler_run_ref, ray.ObjectRef):
            return
        ready, _ = ray.wait([self.scheduler_run_ref], num_returns=1, timeout=0)
        if ready:
            ray.get(ready)
            raise RuntimeError("workflow runtime loop stopped before drain completed")


def _validate_fleet_output_collision(output_dir: Path) -> None:
    for filename in (TRACE_FILENAME, SUMMARY_FILENAME):
        path = output_dir / filename
        if path.exists():
            raise FileExistsError(path)


def _validate_workflow_registration(
    workflow: Workflow,
    functions: Mapping[str, NodeFunction],
    scheduler_config: SchedulerConfig,
    predictions: ResourceContractCache | None,
    output_dir: Path,
) -> None:
    missing = sorted(
        {
            node.function
            for node in workflow.nodes
            if isinstance(node, FunctionNodeConfig) and node.function not in functions
        }
    )
    if missing:
        raise KeyError(f"function registrations are missing: {', '.join(missing)}")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    results_path = output_dir / RESULTS_FILENAME
    if results_path.exists():
        raise FileExistsError(results_path)
    _validate_agent_artifacts(workflow, scheduler_config, predictions)


def _validate_agent_artifacts(
    workflow: Workflow,
    scheduler_config: SchedulerConfig,
    predictions: ResourceContractCache | None,
) -> None:
    agents = [node for node in workflow.nodes if isinstance(node, AgentNodeConfig)]
    if not agents:
        return
    if predictions is None:
        raise ValueError("agent workflows require a prediction cache")
    if not scheduler_config.accelerators:
        raise ValueError("agent workflows require configured accelerators")
    gpu_kinds = {item.gpu_kind for item in scheduler_config.accelerators}
    for node in agents:
        model_path = Path(node.execution.model_path)
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        for gpu_kind in gpu_kinds:
            sequence_lengths = predictions.decode_sequence_lengths(
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
                for output_length in predictions.decode_output_lengths(
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
    executable = scheduler_config.vllm_python_executable
    if executable is None:
        raise ValueError("vllm_python_executable is required")
    executable_path = Path(executable)
    if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
        raise FileNotFoundError(executable_path)


def _validate_live_accelerators(scheduler_config: SchedulerConfig) -> dict[str, str]:
    if not scheduler_config.accelerators:
        return {}
    live_nodes = [node for node in ray.nodes() if node.get("Alive")]
    by_hostname: dict[str, list[AcceleratorConfig]] = {}
    for accelerator in scheduler_config.accelerators:
        by_hostname.setdefault(accelerator.hostname, []).append(accelerator)
    ray_node_ids: dict[str, str] = {}
    for hostname, accelerators in by_hostname.items():
        matches = [
            node for node in live_nodes if node.get("NodeManagerHostname") == hostname
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
            raise ValueError(f"configured GPU kind does not match Ray node: {hostname}")
        node_id = matches[0].get("NodeID")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError(f"Ray node has no stable node id: {hostname}")
        ray_node_ids[hostname] = node_id
    return ray_node_ids


def _kill_actors(
    actors: tuple[ray.actor.ActorHandle, ...],
    *,
    wait_timeout_sec: float = 0.0,
) -> None:
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
