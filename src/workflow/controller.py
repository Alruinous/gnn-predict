from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Self

import yaml
from langchain.agents import AgentState
from pydantic import JsonValue
from ray.util.queue import Queue

from workflow.artifacts import SchedulerConfig, load_resource_contract_cache
from workflow.fleet import (
    ActorFactory,
    NodeFunction,
    QueueFactory,
    WorkflowFleet,
    _missing_directories,
    _validate_fleet_output_collision,
    _validate_workflow_registration,
    prepare_queues,
)
from workflow.schema import Workflow
from workflow.types import SessionState

__all__ = ["WorkflowController", "prepare_queues"]


class WorkflowController:
    """A single-workflow facade over a private, single-member WorkflowFleet."""

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
        self._prompt_tokenizer_factory = prompt_tokenizer_factory
        self.predictions = (
            load_resource_contract_cache(prediction_path)
            if prediction_path is not None
            else None
        )
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        _validate_fleet_output_collision(self.output_dir)
        _validate_workflow_registration(
            workflow,
            self.functions,
            scheduler_config,
            self.predictions,
            self.output_dir,
        )

        self._fleet = WorkflowFleet(
            scheduler_config=scheduler_config,
            output_dir=self.output_dir,
            run_id=run_id,
            prediction_path=prediction_path,
            replica_actor_factory=replica_actor_factory,
            queue_factory=queue_factory,
            scheduler_actor_factory=scheduler_actor_factory,
            worker_actor_factory=worker_actor_factory,
            result_store_actor_factory=result_store_actor_factory,
            trace_writer_actor_factory=trace_writer_actor_factory,
        )
        self._started = False
        self._stopped = False

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
        created_directories = _missing_directories(self.output_dir)
        self._fleet.start()
        try:
            self._fleet.register_workflow(
                self.workflow,
                functions=self.functions,
                output_dir=self.output_dir,
                prompt_tokenizer_factory=self._prompt_tokenizer_factory,
            )
        except Exception:
            self._fleet.rollback_start(created_directories)
            raise
        self._started = True

    @property
    def input_queues(self) -> dict[str, object]:
        return self._fleet._binding(self.workflow.workflow_name).input_queues

    def submit(self, session_id: str, inputs: dict[str, JsonValue]) -> None:
        self._require_running()
        self._fleet.submit(self.workflow.workflow_name, session_id, inputs)

    def get_session_state(self, session_id: str) -> SessionState:
        return self._fleet.get_session_state(self.workflow.workflow_name, session_id)

    def has_result(self, session_id: str) -> bool:
        return self._fleet.has_result(self.workflow.workflow_name, session_id)

    def get_result(self, session_id: str) -> AgentState:
        return self._fleet.get_result(self.workflow.workflow_name, session_id)

    def drain_and_stop(self, timeout_sec: float | None = None) -> None:
        self._require_running()
        self._fleet.drain_workflow(self.workflow.workflow_name, timeout_sec)
        self._fleet.shutdown(timeout_sec)
        self._stopped = True

    def stop_now(self, timeout_sec: float = 5.0) -> None:
        if timeout_sec < 0:
            raise ValueError("stop timeout must be non-negative")
        if not self._started or self._stopped:
            self._stopped = True
            return
        self._fleet.stop_now(timeout_sec)
        self._stopped = True

    def _require_running(self) -> None:
        if not self._started or self._stopped:
            raise RuntimeError("workflow controller must be started and not stopped")
