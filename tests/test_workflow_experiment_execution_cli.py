from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain.agents import AgentState
from langchain_core.messages import AIMessage
from pydantic import JsonValue

import experiment.workflow.baseline as baseline_module
import experiment.workflow.cli as cli_module
import experiment.workflow.execution as execution_module
from dataset.schema import TaskSample
from experiment.workflow.artifacts import (
    EnvironmentManifest,
    TrialManifest,
    complete_trial,
    prepare_trial_directory,
)
from experiment.workflow.baseline import (
    BaselineSessionRequest,
    BaselineSessionResult,
    LangGraphBaseline,
    StaticEnginePool,
    TraceRecorder,
    run_langgraph_sessions,
)
from experiment.workflow.calibration_runner import (
    _calibration_sample_manifest_path,
    _calibration_sessions,
)
from experiment.workflow.cli import main
from experiment.workflow.config import ExperimentConfig, TrialSpec
from experiment.workflow.environment import (
    GPU_ID_ENV,
    selected_gpu_ids,
    validate_visible_devices,
)
from experiment.workflow.mbpp import MBPP_MANIFEST_PATH
from experiment.workflow.prepare import prepare_experiment
from experiment.workflow.qmsum import QMSUM_MANIFEST_PATH
from experiment.workflow.runner import (
    TrialArrival,
    _existing_trial,
    _trial_arrival,
    _trial_manifest,
)
from experiment.workflow.telemetry import TelemetryConfig
from workflow.replica import ReplicaLoadResult
from workflow.schema import Workflow


class ConcurrentBaseline:
    def __init__(self, count: int) -> None:
        self.barrier = Barrier(count)

    def invoke(self, request: BaselineSessionRequest) -> BaselineSessionResult:
        self.barrier.wait(timeout=2)
        now = time.time()
        return BaselineSessionResult(
            session_id=request.session_id,
            final_output=AgentState(messages=[AIMessage(content=request.session_id)]),
            submitted_at=now,
            completed_at=now,
        )


class SlowBaseline:
    def invoke(self, request: BaselineSessionRequest) -> BaselineSessionResult:
        time.sleep(0.1)
        now = time.time()
        return BaselineSessionResult(
            session_id=request.session_id,
            final_output=AgentState(messages=[AIMessage(content="done")]),
            submitted_at=now,
            completed_at=now,
        )


def _trace_split(
    inputs: Mapping[str, object],
    states: Mapping[str, AgentState],
    parameters: Mapping[str, object],
) -> dict[str, AgentState]:
    if inputs or states or parameters:
        raise ValueError("trace split expects empty inputs")
    return {
        "left": AgentState(messages=[AIMessage(content="left")]),
        "right": AgentState(messages=[AIMessage(content="right")]),
    }


def _trace_forward(
    inputs: Mapping[str, object],
    states: Mapping[str, AgentState],
    parameters: Mapping[str, object],
) -> AgentState:
    if inputs or len(states) != 1 or parameters:
        raise ValueError("trace branch expects one state")
    return next(iter(states.values()))


def _trace_merge(
    inputs: Mapping[str, object],
    states: Mapping[str, AgentState],
    parameters: Mapping[str, object],
) -> AgentState:
    if inputs or set(states) != {"left", "right"} or parameters:
        raise ValueError("trace merge expects both branches")
    return AgentState(messages=[AIMessage(content="merged")])


def test_langgraph_session_driver_is_concurrent_and_bounded() -> None:
    requests = tuple(
        BaselineSessionRequest(
            session_id=f"session-{index}",
            inputs={},
            arrival_offset_sec=0,
        )
        for index in range(3)
    )
    results = run_langgraph_sessions(
        ConcurrentBaseline(3),
        requests,
        timeout_sec=2,
    )
    assert {result.session_id for result in results} == {
        "session-0",
        "session-1",
        "session-2",
    }

    with pytest.raises(TimeoutError, match="timed out"):
        run_langgraph_sessions(
            SlowBaseline(),
            requests[:1],
            timeout_sec=0.01,
        )


def test_langgraph_trace_normalizes_edges_and_fanin(tmp_path: Path) -> None:
    workflow = Workflow.model_validate(
        {
            "workflow_name": "langgraph-trace-workflow",
            "nodes": [
                {
                    "name": "split",
                    "type": "function",
                    "function": "split",
                    "routing": "targeted",
                },
                {"name": "left", "type": "function", "function": "forward"},
                {"name": "right", "type": "function", "function": "forward"},
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
    recorder = TraceRecorder("baseline-trace")
    baseline = LangGraphBaseline(
        workflow,
        functions={
            "split": _trace_split,
            "forward": _trace_forward,
            "merge": _trace_merge,
        },
        engines=cast(Any, object()),
        recorder=recorder,
    )
    recorder.record("run_started")
    baseline.invoke(
        BaselineSessionRequest(
            session_id="session",
            inputs={},
            arrival_offset_sec=0,
        )
    )
    recorder.record("run_finished")
    trace_path = tmp_path / "workflow_trace.jsonl"
    recorder.write(trace_path)
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    emitted = [event for event in events if event["event_type"] == "item_emitted"]
    enqueued = [event for event in events if event["event_type"] == "item_enqueued"]
    waits = [event for event in events if event["event_type"] == "fanin_wait"]
    ready = [event for event in events if event["event_type"] == "fanin_ready"]

    assert len(emitted) == len(enqueued) == 4
    assert {event["item_id"] for event in emitted} == {
        event["item_id"] for event in enqueued
    }
    assert len(waits) == len(ready) == 1
    assert waits[0]["node_id"] == ready[0]["node_id"] == "merge"
    assert waits[0]["ts"] <= ready[0]["ts"]


def test_static_pool_records_load_completion_without_waiting_for_first_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = Workflow.model_validate(
        {
            "workflow_name": "static-pool-workflow",
            "nodes": [
                {
                    "name": "agent",
                    "type": "agent",
                    "task": "text_generation",
                    "model": {"name": "test-model"},
                    "execution": {
                        "model_path": "/models/test-model",
                        "max_new_tokens": 8,
                        "do_sample": False,
                        "serving": {
                            "max_model_len": 1024,
                            "max_num_seqs": 1,
                            "max_num_batched_tokens": 1024,
                        },
                    },
                    "prompt_template": "{task}",
                }
            ],
            "edges": [],
        }
    )
    load_result = ReplicaLoadResult(
        physical_gpu_id=0,
        duration_sec=1.0,
        idle_vram_mb=1024.0,
        max_num_seqs=1,
        block_size=16,
        num_gpu_blocks=100,
        gpu_kv_tokens=1600,
    )
    monkeypatch.setattr(baseline_module, "PromptTokenizer", lambda execution: object())
    monkeypatch.setattr(
        baseline_module,
        "_remote_call",
        lambda actor, method_name, *args, **kwargs: method_name,
    )
    monkeypatch.setattr(
        baseline_module.ray,
        "get",
        lambda value: [None for _ in value] if isinstance(value, list) else load_result,
    )
    monkeypatch.setattr(baseline_module.ray, "kill", lambda *args, **kwargs: None)
    recorder = TraceRecorder("load-watcher")
    pool = StaticEnginePool(
        workflow,
        vllm_python=Path("/python"),
        recorder=recorder,
    )
    monkeypatch.setattr(pool, "_create_actor", lambda deployment: object())

    try:
        pool.start_loading()
        next(iter(pool._load_futures.values())).result(timeout=1)
        trace_path = tmp_path / "load_trace.jsonl"
        recorder.write(trace_path)
        event_types = [
            json.loads(line)["event_type"]
            for line in trace_path.read_text().splitlines()
        ]
        assert event_types == ["model_load_started", "model_load_finished"]
    finally:
        pool.shutdown()


def test_burst_arrival_is_shared_across_strategies(tmp_path: Path) -> None:
    config = ExperimentConfig(output_root=tmp_path)
    prepared = prepare_experiment(config)
    fifo = TrialSpec(
        scenario="qmsum",
        strategy="wf-fifo",
        gpu_count=2,
        workload="burst",
        repetition=1,
    )
    cache = fifo.model_copy(update={"strategy": "wf-cache"})

    fifo_arrival = _trial_arrival(config, prepared, fifo)
    cache_arrival = _trial_arrival(config, prepared, cache)

    assert fifo_arrival.sha256 == cache_arrival.sha256
    assert fifo_arrival.sample_ids == cache_arrival.sample_ids
    assert set(fifo_arrival.offsets_sec) == {0.0}


def test_scenario_bundle_loads_the_explicit_formal_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "prepared" / "samples" / "qmsum.jsonl"
    sample = TaskSample(
        sample_id="formal-sample-00",
        source_dataset="qmsum",
        split="test",
        task_type="query_focused_summarization",
        input_text="Speaker: text",
        quality_metric="summary_quality",
        gold_answer="answer",
        metadata={"query": "query", "query_type": "specific", "turn_count": 1},
    )
    samples = tuple(
        sample.model_copy(update={"sample_id": f"formal-sample-{index:02d}"})
        for index in range(24)
    )
    loaded_paths: list[Path] = []

    def load_samples(dataset_path: Path, selected_path: Path) -> tuple[TaskSample, ...]:
        assert dataset_path == ExperimentConfig().qmsum_path
        loaded_paths.append(selected_path)
        return samples

    monkeypatch.setattr(execution_module, "load_qmsum_experiment_samples", load_samples)
    trial = TrialSpec(
        scenario="qmsum",
        strategy="wf-cache",
        gpu_count=2,
        workload="burst",
        repetition=1,
    )

    bundle = execution_module.load_scenario_bundle(
        ExperimentConfig(),
        trial,
        manifest_path,
        expected_session_count=24,
    )

    assert loaded_paths == [manifest_path]
    assert tuple(bundle.inputs_by_sample_id) == tuple(
        f"formal-sample-{index:02d}" for index in range(24)
    )


def test_trial_manifest_uses_the_frozen_preparation_environment() -> None:
    environment = EnvironmentManifest(
        hostname="prepared-host",
        python_version="3.12",
        platform="linux",
        git_commit="commit",
        git_dirty=True,
        git_diff_sha256="prepared-diff",
        package_versions={},
    )
    prepared = cast(
        Any,
        SimpleNamespace(
            sample_manifest_sha256={"qmsum": "formal-samples"},
            synthetic_cache_sha256="cache",
            environment=environment,
        ),
    )
    trial = TrialSpec(
        scenario="qmsum",
        strategy="lg-batch",
        gpu_count=2,
        workload="burst",
        repetition=1,
    )
    arrival = TrialArrival(
        payload={},
        sha256="arrival",
        absolute_sessions_per_sec=None,
        sample_ids=(),
        offsets_sec=(),
    )
    telemetry = TelemetryConfig(
        interval_sec=0.2,
        physical_gpu_ids=(0, 1),
        energy_baseline_method="baseline",
        memory_baseline_method="baseline",
    )

    manifest = _trial_manifest(
        ExperimentConfig(),
        prepared,
        trial,
        arrival,
        {},
        None,
        {},
        telemetry,
    )

    assert manifest.environment == environment


def test_completed_trial_is_skipped_without_runtime_initialization(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig(output_root=tmp_path)
    prepared = prepare_experiment(config)
    trial = TrialSpec(
        scenario="qmsum",
        strategy="wf-cache",
        gpu_count=2,
        workload="burst",
        repetition=1,
    )
    manifest = TrialManifest(
        experiment_id=config.experiment_id,
        trial=trial,
        created_at=1,
        sample_manifest_sha256="sample",
        prediction_cache_sha256="cache",
        arrival_trace_sha256="arrival",
        workflow_config={},
        scheduler_config={},
        serving_environment={},
        environment=EnvironmentManifest(
            hostname="host",
            python_version="3.12",
            platform="linux",
            git_commit="commit",
            git_dirty=False,
            git_diff_sha256="diff",
            package_versions={},
        ),
    )
    _, trial_dir = prepare_trial_directory(prepared.root, manifest)
    complete_trial(
        trial_dir,
        trial.trial_id,
        ("trial_manifest.json",),
        completed_at=2,
    )

    outcome = _existing_trial(prepared, trial)

    assert outcome is not None
    assert outcome.action == "skipped"


def test_calibration_input_repeats_all_fixed_samples_three_times() -> None:
    inputs: dict[str, dict[str, JsonValue]] = {
        f"sample-{index}": {"value": index} for index in range(60)
    }

    sessions = _calibration_sessions(inputs, seed=42, repetition=1)

    assert len(sessions) == 180
    assert len({session.session_id for session in sessions}) == 180
    assert {
        sample_id: sum(session.sample_id == sample_id for session in sessions)
        for sample_id in inputs
    } == {sample_id: 3 for sample_id in inputs}


def test_historical_calibration_keeps_the_parent_sixty_sample_manifests() -> None:
    assert _calibration_sample_manifest_path("qmsum") == QMSUM_MANIFEST_PATH
    assert _calibration_sample_manifest_path("mbpp") == MBPP_MANIFEST_PATH


def test_worker_disables_ray_uv_runtime_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_env: dict[str, str] = {}

    def fake_run(
        _command: tuple[str, ...],
        *,
        check: bool,
        env: dict[str, str],
    ) -> None:
        assert check is True
        captured_env.update(env)

    monkeypatch.setattr(cli_module.subprocess, "run", fake_run)
    cli_module._spawn_worker(
        ExperimentConfig(output_root=tmp_path),
        ("_run-calibration", "qmsum", "2", "1"),
        (2, 3),
    )

    assert captured_env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert captured_env["WORKFLOW_EXPERIMENT_GPU_IDS"] == "2,3"
    assert captured_env["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"


def test_trial_hard_timeout_is_not_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ExperimentConfig(output_root=tmp_path, trial_timeout_sec=10.0)
    trial = TrialSpec(
        scenario="qmsum",
        strategy="wf-cache",
        gpu_count=2,
        workload="burst",
        repetition=1,
    )
    calls: list[float | None] = []

    def spawn(
        _config: ExperimentConfig,
        _trial: TrialSpec,
        _gpu_ids: tuple[int, ...],
        *,
        timeout_sec: float | None = None,
    ) -> cli_module.WorkerAttempt:
        calls.append(timeout_sec)
        log_path = tmp_path / "campaign" / "worker_logs" / "timeout.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("worker exceeded the trial deadline\n", encoding="utf-8")
        return cli_module.WorkerAttempt(-9, True, log_path)

    monkeypatch.setattr(cli_module, "_spawn_trial_worker", spawn)

    outcome = cli_module._run_trial_with_policy(
        config,
        trial,
        (0, 1),
        deadline=time.monotonic() + 100.0,
    )

    assert outcome == "failed"
    assert calls == [10.0]
    failures = sorted((tmp_path / "failed" / trial.trial_id).glob("attempt-*"))
    assert len(failures) == 1
    payload = json.loads((failures[0] / "failure.json").read_text())
    assert payload["category"] == "performance-timeout"
    assert payload["retry_scheduled"] is False


def test_infrastructure_failure_is_retried_once_and_both_attempts_are_archived(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ExperimentConfig(output_root=tmp_path, trial_timeout_sec=10.0)
    trial = TrialSpec(
        scenario="qmsum",
        strategy="wf-cache",
        gpu_count=2,
        workload="burst",
        repetition=1,
    )
    calls = 0

    def spawn(
        _config: ExperimentConfig,
        _trial: TrialSpec,
        _gpu_ids: tuple[int, ...],
        *,
        timeout_sec: float | None = None,
    ) -> cli_module.WorkerAttempt:
        nonlocal calls
        assert timeout_sec == 10.0
        calls += 1
        partial_trial = tmp_path / "trials" / trial.trial_id
        partial_trial.mkdir(parents=True)
        (partial_trial / "partial.txt").write_text(str(calls), encoding="utf-8")
        log_path = tmp_path / "campaign" / "worker_logs" / f"attempt-{calls}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("Failed to connect to GCS\n", encoding="utf-8")
        return cli_module.WorkerAttempt(1, False, log_path)

    monkeypatch.setattr(cli_module, "_spawn_trial_worker", spawn)

    outcome = cli_module._run_trial_with_policy(
        config,
        trial,
        (0, 1),
        deadline=time.monotonic() + 100.0,
    )

    assert outcome == "failed"
    assert calls == 2
    failures = sorted((tmp_path / "failed" / trial.trial_id).glob("attempt-*"))
    assert len(failures) == 2
    payloads = [
        json.loads((failure / "failure.json").read_text()) for failure in failures
    ]
    assert [payload["category"] for payload in payloads] == [
        "infrastructure",
        "infrastructure",
    ]
    assert [payload["retry_scheduled"] for payload in payloads] == [True, False]
    assert all(
        (failure / "partial_trial" / "partial.txt").is_file()
        for failure in failures
    )


def test_campaign_deadline_stops_before_dispatching_another_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ExperimentConfig(output_root=tmp_path, campaign_timeout_sec=1.0)
    trials = tuple(
        TrialSpec(
            scenario="qmsum",
            strategy="wf-cache",
            gpu_count=2,
            workload="burst",
            repetition=repetition,
        )
        for repetition in (1, 2)
    )
    monotonic_values = iter((10.0, 11.0))
    monkeypatch.setattr(cli_module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(cli_module, "_publish_results", lambda config: None)
    monkeypatch.setattr(
        cli_module,
        "_run_trial_with_policy",
        lambda *args, **kwargs: pytest.fail("expired campaign dispatched a trial"),
    )

    exit_code = cli_module._run_campaign(config, trials, (0, 1))

    assert exit_code == 1
    status = json.loads((tmp_path / "campaign" / "status.json").read_text())
    assert status["state"] == "partial"
    assert status["selected_trial_count"] == 2
    assert status["processed_trial_count"] == 0


def test_gpu_visibility_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(GPU_ID_ENV, "2,3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert selected_gpu_ids(2) == (2, 3)
    validate_visible_devices((2, 3))

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES=2,3"):
        validate_visible_devices((2, 3))


def test_cli_prepares_lists_and_reports_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_document = tmp_path / "results.md"
    monkeypatch.setattr(cli_module, "RESULTS_DOCUMENT", results_document)
    common = ("--output-root", str(tmp_path))
    assert main((*common, "prepare")) == 0
    prepared_output = capsys.readouterr().out
    assert "prepared 75 trials" in prepared_output

    assert main((*common, "list", "--scenario", "qmsum")) == 0
    listed = capsys.readouterr().out.splitlines()
    assert len(listed) == 45
    assert all(line.startswith("qmsum__") for line in listed)

    assert main((*common, "status")) == 0
    assert "pending=75" in capsys.readouterr().out

    assert main((*common, "analyze")) == 0
    analyzed_output = capsys.readouterr().out
    assert "analyzed 0 completed trials" in analyzed_output
    assert f"report={results_document}" in analyzed_output
    assert "figures=0" in analyzed_output
    assert (tmp_path / "analysis" / "trial_metrics.jsonl").is_file()
    assert (tmp_path / "analysis" / "group_metrics.jsonl").is_file()
    assert (tmp_path / "analysis" / "report_summary.json").is_file()
    assert results_document.is_file()


def test_cli_status_counts_failed_trials_as_processed_not_pending(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trial = cli_module.build_trial_matrix()[0]
    failure_dir = tmp_path / "failed" / trial.trial_id / "attempt-001"
    failure_dir.mkdir(parents=True)

    cli_module._print_status(ExperimentConfig(output_root=tmp_path))

    status = dict(
        field.split("=", maxsplit=1)
        for field in capsys.readouterr().out.strip().split()
    )
    assert status == {
        "complete": "0",
        "incomplete": "0",
        "failed": "1",
        "pending": "74",
        "total": "75",
    }

def test_queue_telemetry_stops_before_workflow_queues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Controller:
        input_queues: dict[str, object] = {}

        def start(self) -> None:
            events.append("controller:start")

        def drain_and_stop(self, *, timeout_sec: float) -> None:
            assert timeout_sec == 10.0
            events.append("controller:drain")

        def stop_now(self) -> None:
            events.append("controller:stop-now")

    class Recorder:
        def __init__(self, *args: object) -> None:
            pass

        def start(self) -> None:
            events.append("recorder:start")

        def stop(self) -> None:
            events.append("recorder:stop")

    controller = Controller()
    bundle = execution_module.ScenarioBundle(
        scenario="qmsum",
        workflow=cast(Any, object()),
        functions={},
        samples=(),
        inputs_by_sample_id={},
    )
    session = execution_module.SessionInvocation(
        position=0,
        sample_id="sample",
        session_id="session",
        arrival_offset_sec=0.0,
        inputs={},
    )
    monkeypatch.setattr(
        execution_module,
        "WorkflowController",
        lambda *args, **kwargs: controller,
    )
    monkeypatch.setattr(execution_module, "PeriodicRecorder", Recorder)
    monkeypatch.setattr(
        execution_module,
        "_submit_sessions",
        lambda *args: events.append("sessions:submit"),
    )
    monkeypatch.setattr(
        execution_module,
        "_wait_sessions_completed",
        lambda *args: events.append("sessions:complete"),
    )
    monkeypatch.setattr(
        execution_module,
        "_completed_outputs",
        lambda *args: {"session": "output"},
    )

    outputs = execution_module.run_workflow_runtime(
        bundle,
        (session,),
        scheduler_config=cast(Any, object()),
        prediction_path=tmp_path / "cache.jsonl",
        trial_dir=tmp_path,
        run_id="run",
        timeout_sec=10.0,
        telemetry_interval_sec=0.1,
    )

    assert outputs == {"session": "output"}
    assert events == [
        "controller:start",
        "recorder:start",
        "sessions:submit",
        "sessions:complete",
        "recorder:stop",
        "controller:drain",
    ]


def test_calibration_telemetry_stops_before_workflow_queues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Controller:
        input_queues: dict[str, object] = {}

        def start(self) -> None:
            events.append("controller:start")

        def drain_and_stop(self, *, timeout_sec: float) -> None:
            assert timeout_sec == 10.0
            events.append("controller:drain")

        def stop_now(self) -> None:
            events.append("controller:stop-now")

    class Recorder:
        def __init__(self, *args: object) -> None:
            pass

        def start(self) -> None:
            events.append("recorder:start")

        def stop(self) -> None:
            events.append("recorder:stop")

    controller = Controller()
    bundle = execution_module.ScenarioBundle(
        scenario="qmsum",
        workflow=cast(Any, object()),
        functions={},
        samples=(),
        inputs_by_sample_id={},
    )
    session = execution_module.SessionInvocation(
        position=0,
        sample_id="sample",
        session_id="session",
        arrival_offset_sec=0.0,
        inputs={},
    )
    monkeypatch.setattr(
        execution_module,
        "WorkflowController",
        lambda *args, **kwargs: controller,
    )
    monkeypatch.setattr(execution_module, "PeriodicRecorder", Recorder)
    monkeypatch.setattr(
        execution_module,
        "_run_closed_loop",
        lambda *args: events.append("sessions:complete"),
    )
    monkeypatch.setattr(
        execution_module,
        "_assert_sessions_completed",
        lambda *args: events.append("sessions:assert"),
    )

    execution_module.run_workflow_calibration(
        bundle,
        (session,),
        scheduler_config=cast(Any, object()),
        prediction_path=tmp_path / "cache.jsonl",
        calibration_dir=tmp_path,
        run_id="run",
        concurrency=1,
        timeout_sec=10.0,
        telemetry_interval_sec=0.1,
    )

    assert events == [
        "controller:start",
        "recorder:start",
        "sessions:complete",
        "sessions:assert",
        "recorder:stop",
        "controller:drain",
    ]
