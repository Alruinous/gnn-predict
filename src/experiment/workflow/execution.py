from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import ray
from pydantic import JsonValue

from dataset.schema import TaskSample
from experiment.workflow.analysis import summarize_trace
from experiment.workflow.artifacts import (
    write_json_exclusive,
    write_jsonl_exclusive,
)
from experiment.workflow.baseline import (
    BaselineSessionRequest,
    LangGraphBaseline,
    StaticEnginePool,
    TraceRecorder,
    result_rows,
    run_langgraph_sessions,
)
from experiment.workflow.config import ExperimentConfig, Scenario, TrialSpec
from experiment.workflow.mbpp import (
    build_mbpp_session_inputs,
    build_mbpp_workflow,
    load_mbpp_experiment_samples,
    mbpp_functions,
)
from experiment.workflow.qmsum import (
    build_qmsum_session_inputs,
    build_qmsum_workflow,
    load_qmsum_experiment_samples,
    qmsum_functions,
)
from experiment.workflow.telemetry import PeriodicRecorder, QueueSampler
from workflow.artifacts import AcceleratorConfig, SchedulerConfig
from workflow.controller import WorkflowController
from workflow.schema import Workflow
from workflow.types import SessionState
from workflow.worker import message_text

NodeFunction = Callable[..., object]


@dataclass(frozen=True, slots=True)
class ScenarioBundle:
    scenario: Scenario
    workflow: Workflow
    functions: Mapping[str, NodeFunction]
    samples: tuple[TaskSample, ...]
    inputs_by_sample_id: Mapping[str, dict[str, JsonValue]]


@dataclass(frozen=True, slots=True)
class SessionInvocation:
    position: int
    sample_id: str
    session_id: str
    arrival_offset_sec: float
    inputs: dict[str, JsonValue]

    def arrival_row(self) -> dict[str, object]:
        return {
            "position": self.position,
            "sample_id": self.sample_id,
            "session_id": self.session_id,
            "arrival_offset_sec": self.arrival_offset_sec,
        }


def load_scenario_bundle(
    config: ExperimentConfig,
    trial: TrialSpec,
    sample_manifest_path: Path,
    *,
    expected_session_count: int,
) -> ScenarioBundle:
    if trial.scenario == "qmsum":
        samples = load_qmsum_experiment_samples(
            config.qmsum_path,
            sample_manifest_path,
        )
        workflow = build_qmsum_workflow(
            max_num_seqs=trial.max_num_seqs,
            queue_capacity=trial.queue_capacity,
            qwen3_4b_path=str(config.qwen3_4b.path),
            qwen3_8b_path=str(config.qwen3_8b.path),
        )
        functions = qmsum_functions()
        build_inputs = build_qmsum_session_inputs
    else:
        samples = load_mbpp_experiment_samples(
            config.mbpp_path,
            sample_manifest_path,
        )
        workflow = build_mbpp_workflow(
            max_num_seqs=trial.max_num_seqs,
            queue_capacity=trial.queue_capacity,
            qwen3_4b_path=str(config.qwen3_4b.path),
            qwen3_8b_path=str(config.qwen3_8b.path),
            qwen3_14b_path=str(config.qwen3_14b.path),
        )
        functions = mbpp_functions()
        build_inputs = build_mbpp_session_inputs
    if len(samples) != expected_session_count:
        raise ValueError(
            f"scenario manifest requires {expected_session_count} samples, got "
            f"{len(samples)}"
        )
    return ScenarioBundle(
        scenario=trial.scenario,
        workflow=workflow,
        functions=functions,
        samples=samples,
        inputs_by_sample_id={
            sample.sample_id: build_inputs(sample) for sample in samples
        },
    )


def build_session_invocations(
    bundle: ScenarioBundle,
    ordered_sample_ids: Sequence[str],
    arrival_offsets_sec: Sequence[float],
) -> tuple[SessionInvocation, ...]:
    if len(ordered_sample_ids) != len(arrival_offsets_sec):
        raise ValueError("sample order and arrival offsets must have equal lengths")
    if set(ordered_sample_ids) != set(bundle.inputs_by_sample_id):
        raise ValueError("sample order must contain every fixed sample exactly once")
    if len(ordered_sample_ids) != len(set(ordered_sample_ids)):
        raise ValueError("sample order contains duplicates")
    if any(offset < 0 for offset in arrival_offsets_sec):
        raise ValueError("arrival offsets must be non-negative")
    if any(right < left for left, right in pairwise(arrival_offsets_sec)):
        raise ValueError("arrival offsets must be ordered")
    return tuple(
        SessionInvocation(
            position=position,
            sample_id=sample_id,
            session_id=f"session-{position:03d}-{sample_id}",
            arrival_offset_sec=float(arrival_offsets_sec[position]),
            inputs=dict(bundle.inputs_by_sample_id[sample_id]),
        )
        for position, sample_id in enumerate(ordered_sample_ids)
    )


def initialize_local_ray(gpu_count: int) -> str:
    if ray.is_initialized():
        raise RuntimeError("each experiment worker requires a fresh Ray runtime")
    ray.init(num_gpus=gpu_count, include_dashboard=False)
    live_nodes = [node for node in ray.nodes() if node.get("Alive")]
    if len(live_nodes) != 1:
        ray.shutdown()
        raise RuntimeError("local experiment runner requires exactly one Ray node")
    hostname = live_nodes[0].get("NodeManagerHostname")
    if not isinstance(hostname, str) or not hostname:
        ray.shutdown()
        raise RuntimeError("Ray did not report a node hostname")
    return hostname


def build_scheduler_config(
    config: ExperimentConfig,
    trial: TrialSpec,
    hostname: str,
) -> SchedulerConfig:
    policy = trial.scheduler_policy
    if policy is None:
        raise ValueError("LG-Batch has no workflow scheduler configuration")
    return SchedulerConfig(
        policy=policy,
        accelerators=tuple(
            AcceleratorConfig(
                hostname=hostname,
                gpu_kind="v100",
                local_index=index,
                total_mem_mb=config.gpu_total_mem_mb,
            )
            for index in range(trial.gpu_count)
        ),
        vllm_python_executable=str(config.vllm_python),
        acquire_timeout_sec=config.trial_timeout_sec,
    )


def run_workflow_runtime(
    bundle: ScenarioBundle,
    sessions: Sequence[SessionInvocation],
    *,
    scheduler_config: SchedulerConfig,
    prediction_path: Path,
    trial_dir: Path,
    run_id: str,
    timeout_sec: float,
    telemetry_interval_sec: float,
) -> dict[str, str]:
    controller = WorkflowController(
        bundle.workflow,
        functions=bundle.functions,
        scheduler_config=scheduler_config,
        prediction_path=prediction_path,
        output_dir=trial_dir,
        run_id=run_id,
    )
    try:
        controller.start()
        queue_recorder = PeriodicRecorder(
            trial_dir / "queue_telemetry.jsonl",
            QueueSampler(controller.input_queues),
            telemetry_interval_sec,
        )
        queue_recorder.start()
        try:
            _submit_sessions(controller, sessions)
            _wait_sessions_completed(controller, sessions, timeout_sec)
        finally:
            queue_recorder.stop()
        controller.drain_and_stop(timeout_sec=timeout_sec)
        return _completed_outputs(controller, sessions)
    except Exception:
        controller.stop_now()
        raise


def run_workflow_calibration(
    bundle: ScenarioBundle,
    sessions: Sequence[SessionInvocation],
    *,
    scheduler_config: SchedulerConfig,
    prediction_path: Path,
    calibration_dir: Path,
    run_id: str,
    concurrency: int,
    timeout_sec: float,
    telemetry_interval_sec: float,
) -> None:
    if concurrency <= 0 or concurrency > len(sessions):
        raise ValueError("calibration concurrency is outside the session count")
    controller = WorkflowController(
        bundle.workflow,
        functions=bundle.functions,
        scheduler_config=scheduler_config,
        prediction_path=prediction_path,
        output_dir=calibration_dir,
        run_id=run_id,
    )
    try:
        controller.start()
        queue_recorder = PeriodicRecorder(
            calibration_dir / "queue_telemetry.jsonl",
            QueueSampler(controller.input_queues),
            telemetry_interval_sec,
        )
        queue_recorder.start()
        try:
            _run_closed_loop(controller, sessions, concurrency, timeout_sec)
            _assert_sessions_completed(controller, sessions)
        finally:
            queue_recorder.stop()
        controller.drain_and_stop(timeout_sec=timeout_sec)
    except Exception:
        controller.stop_now()
        raise


def run_langgraph_baseline(
    bundle: ScenarioBundle,
    sessions: Sequence[SessionInvocation],
    *,
    vllm_python: Path,
    trial_dir: Path,
    run_id: str,
    timeout_sec: float,
) -> dict[str, str]:
    recorder = TraceRecorder(run_id)
    engines = StaticEnginePool(
        bundle.workflow,
        vllm_python=vllm_python,
        recorder=recorder,
    )
    baseline = LangGraphBaseline(
        bundle.workflow,
        functions=bundle.functions,
        engines=engines,
        recorder=recorder,
    )
    recorder.record("run_started")
    try:
        engines.start_loading()
        try:
            results = run_langgraph_sessions(
                baseline,
                tuple(
                    BaselineSessionRequest(
                        session_id=session.session_id,
                        inputs=session.inputs,
                        arrival_offset_sec=session.arrival_offset_sec,
                    )
                    for session in sessions
                ),
                timeout_sec=timeout_sec,
            )
        finally:
            engines.shutdown()
    except Exception as error:
        recorder.record(
            "run_finished",
            payload={
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        recorder.write(trial_dir / "workflow_trace.jsonl")
        raise
    recorder.record("run_finished", payload={"status": "completed"})
    recorder.write(trial_dir / "workflow_trace.jsonl")
    write_jsonl_exclusive(
        trial_dir / "session_results.jsonl",
        result_rows(run_id, results),
    )
    write_json_exclusive(
        trial_dir / "run_summary.json",
        summarize_trace(trial_dir / "workflow_trace.jsonl"),
    )
    return {result.session_id: message_text(result.final_output) for result in results}


def normalized_output_rows(
    sessions: Sequence[SessionInvocation],
    outputs: Mapping[str, str],
) -> tuple[dict[str, object], ...]:
    if set(outputs) != {session.session_id for session in sessions}:
        raise ValueError("runtime outputs do not match the submitted sessions")
    return tuple(
        {
            "position": session.position,
            "sample_id": session.sample_id,
            "session_id": session.session_id,
            "output": outputs[session.session_id],
        }
        for session in sessions
    )


def _submit_sessions(
    controller: WorkflowController,
    sessions: Sequence[SessionInvocation],
) -> None:
    started = time.monotonic()

    def submit(session: SessionInvocation) -> None:
        remaining = started + session.arrival_offset_sec - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        controller.submit(session.session_id, session.inputs)

    with ThreadPoolExecutor(max_workers=len(sessions)) as executor:
        futures = [executor.submit(submit, session) for session in sessions]
        for future in futures:
            future.result()


def _wait_sessions_completed(
    controller: WorkflowController,
    sessions: Sequence[SessionInvocation],
    timeout_sec: float,
) -> None:
    pending = {session.session_id for session in sessions}
    deadline = time.monotonic() + timeout_sec
    while pending:
        if time.monotonic() >= deadline:
            raise TimeoutError("workflow sessions timed out")
        completed: list[str] = []
        for session_id in pending:
            state = controller.get_session_state(session_id)
            if state == SessionState.FAILED:
                raise RuntimeError(f"workflow session failed: {session_id}")
            if state == SessionState.COMPLETED:
                completed.append(session_id)
        pending.difference_update(completed)
        if pending:
            time.sleep(0.05)


def _completed_outputs(
    controller: WorkflowController,
    sessions: Sequence[SessionInvocation],
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    _assert_sessions_completed(controller, sessions)
    for session in sessions:
        outputs[session.session_id] = message_text(
            controller.get_result(session.session_id)
        )
    return outputs


def _assert_sessions_completed(
    controller: WorkflowController,
    sessions: Sequence[SessionInvocation],
) -> None:
    for session in sessions:
        state = controller.get_session_state(session.session_id)
        if state != SessionState.COMPLETED:
            raise RuntimeError(
                f"session did not complete: {session.session_id}: {state}"
            )


def _run_closed_loop(
    controller: WorkflowController,
    sessions: Sequence[SessionInvocation],
    concurrency: int,
    timeout_sec: float,
) -> None:
    initial = sessions[:concurrency]
    _submit_sessions(controller, initial)
    active = {session.session_id: session for session in initial}
    next_index = concurrency
    deadline = time.monotonic() + timeout_sec
    while active:
        if time.monotonic() >= deadline:
            raise TimeoutError("capacity calibration timed out")
        terminal_ids: list[str] = []
        for session_id in tuple(active):
            state = controller.get_session_state(session_id)
            if state == SessionState.FAILED:
                raise RuntimeError(f"calibration session failed: {session_id}")
            if state == SessionState.COMPLETED:
                terminal_ids.append(session_id)
        if not terminal_ids:
            time.sleep(0.05)
            continue
        for session_id in terminal_ids:
            del active[session_id]
            if next_index == len(sessions):
                continue
            session = sessions[next_index]
            controller.submit(session.session_id, session.inputs)
            active[session.session_id] = session
            next_index += 1
