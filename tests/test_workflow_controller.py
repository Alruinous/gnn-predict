from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import pytest
import ray
from ray.exceptions import RayActorError

from workflow.artifacts import AcceleratorConfig, SchedulerConfig
from workflow.controller import WorkflowController, prepare_queues
from workflow.schema import Workflow


def function_workflow() -> Workflow:
    return Workflow.model_validate(
        {
            "nodes": [
                {
                    "name": "split",
                    "type": "function",
                    "function": "split_input",
                    "routing": "targeted",
                    "queue_capacity": 4,
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


class FakeQueue:
    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize


class TrackingQueue(FakeQueue):
    def __init__(self, maxsize: int) -> None:
        super().__init__(maxsize)
        self.shutdown_forces: list[bool] = []

    def shutdown(self, force: bool = False) -> None:
        self.shutdown_forces.append(force)


class RollbackWorker:
    def ping(self) -> str:
        return "alive"


RollbackWorkerActor = cast(Any, ray.remote)(RollbackWorker)


class ShutdownActor:
    def cancel_pending(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def stop_loop(self) -> None:
        return None

    def ping(self) -> str:
        return "alive"


def fail_runtime_loop() -> None:
    raise RuntimeError("runtime loop failed")


ShutdownActorRemote = cast(Any, ray.remote)(ShutdownActor)
FailedRuntimeTask = cast(Any, ray.remote)(fail_runtime_loop)


def test_prepare_queues_reuses_each_target_input_queue() -> None:
    workflow = function_workflow()

    inputs, outputs = prepare_queues(workflow, queue_factory=FakeQueue)

    assert inputs["split"].maxsize == 4
    assert outputs["split"]["left"] is inputs["left"]
    assert outputs["split"]["right"] is inputs["right"]
    assert outputs["left"]["merge"] is inputs["merge"]
    assert outputs["right"]["merge"] is inputs["merge"]


def test_validation_precedes_queue_and_actor_factories(tmp_path: Path) -> None:
    calls: list[str] = []

    def unexpected_factory(*args: object, **kwargs: object) -> object:
        calls.append("called")
        return object()

    with pytest.raises(KeyError, match="function"):
        WorkflowController(
            function_workflow(),
            functions={},
            scheduler_config=SchedulerConfig(),
            output_dir=tmp_path,
            run_id="invalid",
            queue_factory=unexpected_factory,
            scheduler_actor_factory=unexpected_factory,
            worker_actor_factory=unexpected_factory,
            result_store_actor_factory=unexpected_factory,
            trace_writer_actor_factory=unexpected_factory,
        )

    assert calls == []
    assert tuple(tmp_path.iterdir()) == ()


def test_submit_requires_started_controller(tmp_path: Path) -> None:
    functions: dict[str, Callable[..., object]] = {
        "split_input": lambda inputs, states, parameters: {},
        "left": lambda inputs, states, parameters: {},
        "right": lambda inputs, states, parameters: {},
        "merge": lambda inputs, states, parameters: {},
    }
    controller = WorkflowController(
        function_workflow(),
        functions=functions,
        scheduler_config=SchedulerConfig(),
        output_dir=tmp_path,
        run_id="not-started",
    )

    with pytest.raises(RuntimeError, match="start"):
        controller.submit("session-1", {"value": "one"})


def test_agent_prediction_budgets_are_validated_before_start(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    prediction_path = tmp_path / "predictions.yaml"
    prediction_path.write_text(
        """
version: 1
entries:
  - key:
      model_name: test-model
      phase: decode
      gpu_name: v100
      batch_size: 1
      sequence_length: 128
      decode_output_length: 32
    predicted_load_sec: 5.0
    predicted_run_sec: 1.0
    predicted_peak_vram_mb: 1024.0
""".lstrip(),
        encoding="utf-8",
    )
    workflow = Workflow.model_validate(
        {
            "nodes": [
                {
                    "name": "agent",
                    "type": "agent",
                    "model": {"name": "test-model"},
                    "execution": {
                        "model_path": str(model_path),
                        "serving": {
                            "max_model_len": 128,
                            "max_num_seqs": 1,
                            "max_num_batched_tokens": 128,
                        },
                    },
                    "token_budget": {
                        "min_max_new_tokens": 8,
                        "default_max_new_tokens": 16,
                        "max_max_new_tokens": 16,
                    },
                    "prompt_template": "{content}",
                }
            ],
            "edges": [],
        }
    )

    with pytest.raises(KeyError, match="prediction coverage"):
        WorkflowController(
            workflow,
            functions={},
            scheduler_config=SchedulerConfig(
                accelerators=(
                    AcceleratorConfig(
                        hostname="gpu-node",
                        gpu_kind="v100",
                        local_index=0,
                        total_mem_mb=16_000,
                    ),
                )
            ),
            prediction_path=prediction_path,
            output_dir=tmp_path / "output",
            run_id="invalid-predictions",
        )


def test_agent_requires_contiguous_prediction_batch_coverage(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    prediction_path = tmp_path / "predictions.yaml"
    prediction_path.write_text(
        """
version: 1
entries:
  - key:
      model_name: test-model
      phase: decode
      gpu_name: v100
      batch_size: 1
      sequence_length: 128
      decode_output_length: 32
    predicted_load_sec: 5.0
    predicted_run_sec: 1.0
    predicted_peak_vram_mb: 1024.0
""".lstrip(),
        encoding="utf-8",
    )
    workflow = Workflow.model_validate(
        {
            "nodes": [
                {
                    "name": "agent",
                    "type": "agent",
                    "model": {"name": "test-model"},
                    "execution": {
                        "model_path": str(model_path),
                        "serving": {
                            "max_model_len": 256,
                            "max_num_seqs": 2,
                            "max_num_batched_tokens": 256,
                        },
                    },
                    "token_budget": {
                        "min_max_new_tokens": 32,
                        "default_max_new_tokens": 32,
                        "max_max_new_tokens": 32,
                    },
                    "prompt_template": "{content}",
                }
            ],
            "edges": [],
        }
    )

    with pytest.raises(KeyError, match="batch coverage.*batch=2"):
        WorkflowController(
            workflow,
            functions={},
            scheduler_config=SchedulerConfig(
                accelerators=(
                    AcceleratorConfig(
                        hostname="gpu-node",
                        gpu_kind="v100",
                        local_index=0,
                        total_mem_mb=16_000,
                    ),
                )
            ),
            prediction_path=prediction_path,
            output_dir=tmp_path / "output",
            run_id="missing-batch",
        )


def test_replica_factory_does_not_bypass_vllm_executable_validation(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    prediction_path = tmp_path / "predictions.yaml"
    prediction_path.write_text(
        """
version: 1
entries:
  - key:
      model_name: test-model
      phase: decode
      gpu_name: v100
      batch_size: 1
      sequence_length: 64
      decode_output_length: 16
    predicted_load_sec: 5.0
    predicted_run_sec: 1.0
    predicted_peak_vram_mb: 1024.0
""".lstrip(),
        encoding="utf-8",
    )
    workflow = Workflow.model_validate(
        {
            "nodes": [
                {
                    "name": "agent",
                    "type": "agent",
                    "model": {"name": "test-model"},
                    "execution": {
                        "model_path": str(model_path),
                        "serving": {
                            "max_model_len": 128,
                            "max_num_seqs": 1,
                            "max_num_batched_tokens": 128,
                        },
                    },
                    "token_budget": {
                        "min_max_new_tokens": 16,
                        "default_max_new_tokens": 16,
                        "max_max_new_tokens": 16,
                    },
                    "prompt_template": "{content}",
                }
            ],
            "edges": [],
        }
    )

    with pytest.raises(ValueError, match="vllm_python_executable is required"):
        WorkflowController(
            workflow,
            functions={},
            scheduler_config=SchedulerConfig(
                accelerators=(
                    AcceleratorConfig(
                        hostname="gpu-node",
                        gpu_kind="v100",
                        local_index=0,
                        total_mem_mb=16_000,
                    ),
                )
            ),
            prediction_path=prediction_path,
            output_dir=tmp_path / "output",
            run_id="missing-vllm-executable",
            replica_actor_factory=lambda action: object(),
        )


def test_start_rolls_back_queues_created_before_factory_failure(
    ray_session: None,
    tmp_path: Path,
) -> None:
    created: list[TrackingQueue] = []

    def queue_factory(maxsize: int) -> TrackingQueue:
        if created:
            raise RuntimeError("queue factory failed")
        queue = TrackingQueue(maxsize)
        created.append(queue)
        return queue

    controller = WorkflowController(
        function_workflow(),
        functions={
            "split_input": lambda inputs, states, parameters: {},
            "left": lambda inputs, states, parameters: {},
            "right": lambda inputs, states, parameters: {},
            "merge": lambda inputs, states, parameters: {},
        },
        scheduler_config=SchedulerConfig(),
        output_dir=tmp_path,
        run_id="queue-rollback",
        queue_factory=queue_factory,
    )

    with pytest.raises(RuntimeError, match="queue factory failed"):
        controller.start()

    assert created[0].shutdown_forces == [True]


def test_start_rolls_back_workers_created_before_factory_failure(
    ray_session: None,
    tmp_path: Path,
) -> None:
    created: list[ray.actor.ActorHandle] = []

    def worker_factory(**kwargs: object) -> object:
        if created:
            raise RuntimeError("worker factory failed")
        worker = RollbackWorkerActor.remote()
        created.append(worker)
        return worker

    controller = WorkflowController(
        function_workflow(),
        functions={
            "split_input": lambda inputs, states, parameters: {},
            "left": lambda inputs, states, parameters: {},
            "right": lambda inputs, states, parameters: {},
            "merge": lambda inputs, states, parameters: {},
        },
        scheduler_config=SchedulerConfig(),
        output_dir=tmp_path,
        run_id="worker-rollback",
        queue_factory=TrackingQueue,
        scheduler_actor_factory=lambda **kwargs: object(),
        worker_actor_factory=worker_factory,
        result_store_actor_factory=lambda *args: object(),
        trace_writer_actor_factory=lambda *args: object(),
    )

    with pytest.raises(RuntimeError, match="worker factory failed"):
        controller.start()

    with pytest.raises(RayActorError):
        ray.get(created[0].ping.remote(), timeout=5)


def test_stop_now_cleans_up_after_runtime_ref_has_failed(
    ray_session: None,
    tmp_path: Path,
) -> None:
    controller = WorkflowController(
        function_workflow(),
        functions={
            "split_input": lambda inputs, states, parameters: {},
            "left": lambda inputs, states, parameters: {},
            "right": lambda inputs, states, parameters: {},
            "merge": lambda inputs, states, parameters: {},
        },
        scheduler_config=SchedulerConfig(),
        output_dir=tmp_path,
        run_id="failed-runtime-stop",
    )
    queue = TrackingQueue(maxsize=4)
    scheduler = ShutdownActorRemote.remote()
    worker = ShutdownActorRemote.remote()
    failed_ref = FailedRuntimeTask.remote()
    ready, _ = ray.wait([failed_ref], timeout=30)
    assert ready == [failed_ref]

    controller.input_queues = {"split": queue}
    controller.workers = {"split": worker}
    controller.scheduler = scheduler
    controller.scheduler_run_ref = failed_ref
    controller._started = True
    controller._admission_open = True
    try:
        started = time.monotonic()
        controller.stop_now(timeout_sec=0.2)

        assert time.monotonic() - started < 3.0
        assert queue.shutdown_forces == [True]
        assert controller._stopped
        with pytest.raises(RayActorError):
            ray.get(scheduler.ping.remote(), timeout=5)
        with pytest.raises(RayActorError):
            ray.get(worker.ping.remote(), timeout=5)
    finally:
        with suppress(Exception):
            ray.kill(scheduler, no_restart=True)
        with suppress(Exception):
            ray.kill(worker, no_restart=True)
