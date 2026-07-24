from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiment.workflow.cache_replay import (
    AgentDurationTable,
    LoadSample,
    ReplaySimulator,
    ReplayWorkload,
    TaskSample,
    TaskSpec,
    TraceRun,
    calibrate_prediction_cache,
    read_trace_run,
)
from workflow.artifacts import (
    ResourceContract,
    ResourceContractCache,
    ResourceContractSource,
    ResourceEvidence,
)
from workflow.schema import Workflow
from workflow.types import WorkflowModelFeatureKey


def prediction(
    model_name: str,
    gpu_kind: str,
    *,
    run_sec: float = 10.0,
    load_sec: float = 10.0,
    peak_vram_mb: float = 8_000.0,
    sequence_length: int = 128,
) -> ResourceContract:
    return ResourceContract(
        key=WorkflowModelFeatureKey(
            model_name=model_name,
            phase="decode",
            gpu_name=gpu_kind,
            batch_size=1,
            sequence_length=sequence_length,
            decode_output_length=16,
        ),
        source=ResourceContractSource.SYNTHETIC_FIXTURE,
        predicted_load_sec=load_sec,
        predicted_run_sec=run_sec,
        predicted_peak_vram_mb=peak_vram_mb,
        peak_vram_mb_upper_bound=peak_vram_mb,
        peak_vram_mb_evidence=ResourceEvidence(
            method="point_estimate_only",
            sample_count=1,
        ),
    )


def trace_run(
    name: str,
    task: TaskSample,
    *,
    load_duration_sec: float,
) -> TraceRun:
    prediction_key = task.prediction_key
    model_key = task.model_key
    gpu_kind = task.gpu_kind
    assert prediction_key is not None
    assert model_key is not None
    assert gpu_kind is not None
    return TraceRun(
        path=Path(name),
        run_id=name,
        run_started=0.0,
        run_finished=10.0,
        session_arrivals={"s": 0.0},
        session_workflows={"s": task.workflow_name},
        tasks={(task.session_id, task.node_id): task},
        loads=(
            LoadSample(
                model_name=prediction_key.model_name,
                model_key=model_key,
                gpu_kind=gpu_kind,
                duration_sec=load_duration_sec,
                cold_accelerator=False,
            ),
        ),
        eviction_durations={"v100": (0.5,)},
        model_names={model_key: prediction_key.model_name},
        completed_session_count=1,
    )


def test_trace_calibration_changes_only_soft_timing() -> None:
    observed = prediction("model", "v100")
    unseen = prediction("unseen", "v100", sequence_length=256)
    cache = ResourceContractCache(version=1, entries=(observed, unseen))
    task = TaskSample(
        session_id="s",
        workflow_name="wf",
        node_id="agent",
        duration_sec=2.0,
        input_tokens=64,
        output_tokens=8,
        admitted_batch_size=1,
        model_key="model-key",
        gpu_kind="v100",
        prediction_key=observed.key,
        predicted_run_sec=10.0,
        predicted_load_sec=10.0,
    )
    runs = tuple(
        trace_run(f"run-{index}", task, load_duration_sec=3.0)
        for index in range(3)
    )

    result = calibrate_prediction_cache(cache, runs)
    calibrated = result.cache.lookup(observed.key)
    fallback = result.cache.lookup(unseen.key)

    assert calibrated.predicted_run_sec == pytest.approx(2.0)
    assert calibrated.predicted_load_sec == pytest.approx(3.0)
    assert calibrated.predicted_peak_vram_mb == observed.predicted_peak_vram_mb
    assert calibrated.peak_vram_mb_upper_bound == observed.peak_vram_mb_upper_bound
    assert fallback.predicted_run_sec == unseen.predicted_run_sec
    assert fallback.predicted_load_sec == unseen.predicted_load_sec


def test_read_trace_run_rejects_incomplete_trace(tmp_path: Path) -> None:
    path = tmp_path / "workflow_trace.jsonl"
    rows = [
        {
            "run_id": "run",
            "event_type": "run_started",
            "ts": 0.0,
            "payload": {},
        },
        {
            "run_id": "run",
            "event_type": "session_submitted",
            "ts": 0.0,
            "session_id": "s",
            "workflow_name": "wf",
            "payload": {},
        },
        {
            "run_id": "run",
            "event_type": "task_execution_finished",
            "ts": 1.0,
            "session_id": "s",
            "workflow_name": "wf",
            "node_id": "function",
            "payload": {"duration_sec": 1.0, "status": "success"},
        },
        {
            "run_id": "run",
            "event_type": "session_completed",
            "ts": 1.0,
            "session_id": "s",
            "workflow_name": "wf",
            "payload": {"latency_sec": 1.0},
        },
    ]
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="one run start and finish"):
        read_trace_run(path)


def test_submission_closed_reclaim_reduces_bubble_without_tail_regression() -> None:
    workflow = branched_workflow()
    cache = ResourceContractCache(
        version=1,
        entries=(
            prediction(
                "model-a",
                "a100",
                run_sec=1.0,
                load_sec=1.0,
                peak_vram_mb=40_000.0,
            ),
            prediction(
                "model-b",
                "v100",
                run_sec=10.0,
                load_sec=1.0,
            ),
        ),
    )
    workload = branched_workload_spec(workflow)

    baseline = ReplaySimulator(
        workload,
        policy="profile_cache",
        profile_cache=cache,
        calibrated_cache=cache,
    ).run()
    reclaimed = ReplaySimulator(
        workload,
        policy="trace_calibrated_cache_reclaim",
        profile_cache=cache,
        calibrated_cache=cache,
    ).run()

    assert reclaimed.replay_completion_span_sec == baseline.replay_completion_span_sec
    assert reclaimed.idle_resident_gpu_seconds < baseline.idle_resident_gpu_seconds
    assert reclaimed.pipeline_bubble_ratio < baseline.pipeline_bubble_ratio
    assert reclaimed.model_load_count == baseline.model_load_count
    assert reclaimed.reclaimed_replica_count == 2


def branched_workflow() -> Workflow:
    def agent(name: str, model_name: str, model_path: str) -> dict[str, object]:
        return {
            "name": name,
            "type": "agent",
            "model": {"name": model_name},
            "execution": {
                "model_path": model_path,
                "max_new_tokens": 16,
                "dtype": "float16",
                "serving": {
                    "max_model_len": 128,
                    "max_num_seqs": 1,
                    "max_num_batched_tokens": 128,
                },
            },
            "prompt_template": "{content}",
        }

    return Workflow.model_validate(
        {
            "workflow_name": "wf",
            "nodes": [
                {
                    "name": "root",
                    "type": "function",
                    "function": "root",
                    "routing": "broadcast",
                },
                agent("fast", "model-a", "/models/a"),
                agent("slow", "model-b", "/models/b"),
                {
                    "name": "merge",
                    "type": "function",
                    "function": "merge",
                    "routing": "broadcast",
                },
            ],
            "edges": [
                {"source": "root", "target": "fast"},
                {"source": "root", "target": "slow"},
                {"source": "fast", "target": "merge"},
                {"source": "slow", "target": "merge"},
            ],
        }
    )


def branched_workload_spec(workflow: Workflow) -> ReplayWorkload:
    task_specs = {
        ("s", "root"): TaskSpec("s", "wf", "root", None, None, 1.0),
        ("s", "fast"): TaskSpec("s", "wf", "fast", 64, 8, None),
        ("s", "slow"): TaskSpec("s", "wf", "slow", 64, 8, None),
        ("s", "merge"): TaskSpec("s", "wf", "merge", None, None, 1.0),
    }
    return ReplayWorkload(
        name="fixture",
        workflows={"wf": workflow},
        session_arrivals={"s": 0.0},
        session_workflows={"s": "wf"},
        task_specs=task_specs,
        agent_durations=AgentDurationTable(
            exact={("s", "fast", 1): 1.0, ("s", "slow", 1): 10.0},
            node_batch={},
            model_batch={},
            task_any_batch={},
        ),
        cold_load_sec={
            ("model-a", "a100"): 1.0,
            ("model-b", "v100"): 1.0,
        },
        warm_load_sec={
            ("model-a", "a100"): 1.0,
            ("model-b", "v100"): 1.0,
        },
        any_load_sec={},
        eviction_sec={"a100": 0.1, "v100": 0.1},
        gpu_slots={"a100": 1, "v100": 1},
    )
