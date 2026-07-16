from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from experiment.workflow.aggregate import regenerate_analysis
from experiment.workflow.artifacts import (
    canonical_json,
    validate_completion,
    write_json_exclusive,
)
from experiment.workflow.calibration_runner import run_calibration
from experiment.workflow.config import ExperimentConfig, Scenario, TrialSpec
from experiment.workflow.plan import build_trial_matrix
from experiment.workflow.prepare import prepare_experiment
from experiment.workflow.runner import run_trial

RESULTS_DOCUMENT = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "workflow"
    / "system_experiment_results_20260713.md"
)
EVIDENCE_GROUP_ENDS = frozenset((15, 25, 45, 65, 75))
INFRASTRUCTURE_FAILURE_MARKERS = (
    "Failed to connect to GCS",
    "Failed to register worker",
    "NVMLError_Unknown",
    "RaySystemError",
    "Unable to connect to GCS",
    "driver/library version mismatch",
    "worker launch failed",
)


@dataclass(frozen=True, slots=True)
class WorkerAttempt:
    return_code: int
    timed_out: bool
    log_path: Path


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    config = ExperimentConfig(output_root=args.output_root)
    if args.command == "prepare":
        prepared = prepare_experiment(config)
        print(f"prepared {len(prepared.trial_matrix)} trials at {prepared.root}")
        return 0
    if args.command == "list":
        trials = _select_trials(args, build_trial_matrix())
        for trial in trials:
            print(trial.trial_id)
        return 0
    if args.command == "status":
        _print_status(config)
        return 0
    if args.command == "analyze":
        result = regenerate_analysis(config.output_root.resolve())
        from experiment.workflow.report import publish_report

        published = publish_report(config.output_root, RESULTS_DOCUMENT)
        print(
            f"analyzed {result.completed_trial_count} completed trials into "
            f"{result.aligned_group_count} aligned groups at "
            f"{result.trial_metrics_path.parent}; report={published.results_document} "
            f"figures={len(published.figure_paths)}"
        )
        return 0
    if args.command == "run":
        prepare_experiment(config)
        trials = _select_trials(args, build_trial_matrix())
        if not trials:
            parser.error("run selection is empty")
        if not args.all and not args.trial_id and not _has_run_filter(args):
            parser.error("run requires --trial-id, a filter, or --all")
        if args.limit is not None:
            trials = trials[: args.limit]
        gpu_ids = _parse_gpu_ids(args.gpu_ids)
        return _run_campaign(config, trials, gpu_ids)
    if args.command == "_run-trial":
        trial = _trial_by_id(args.trial_id)
        outcome = run_trial(config, trial)
        print(f"{outcome.action}: {outcome.trial_dir}")
        return 0
    if args.command == "_run-calibration":
        outcome = run_calibration(
            config,
            cast(Scenario, args.scenario),
            cast(Literal[1, 2, 3], args.gpu_count),
            cast(Literal[1, 2, 3], args.repetition),
        )
        print(f"{outcome}: {args.scenario} g{args.gpu_count} rep{args.repetition}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the workflow scheduling system experiment",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ExperimentConfig().output_root,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")

    list_parser = subparsers.add_parser("list")
    _add_trial_filters(list_parser)

    subparsers.add_parser("status")
    subparsers.add_parser("analyze")

    run = subparsers.add_parser("run")
    _add_trial_filters(run)
    run.add_argument("--all", action="store_true")
    run.add_argument("--limit", type=int)
    run.add_argument("--gpu-ids", default="0,1,2,3")

    worker = subparsers.add_parser("_run-trial")
    worker.add_argument("trial_id")

    calibration_worker = subparsers.add_parser("_run-calibration")
    calibration_worker.add_argument("scenario", choices=("qmsum", "mbpp"))
    calibration_worker.add_argument("gpu_count", type=int, choices=(1, 2, 3))
    calibration_worker.add_argument("repetition", type=int, choices=(1, 2, 3))
    return parser


def _add_trial_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trial-id", action="append")
    parser.add_argument("--scenario", choices=("qmsum", "mbpp"))
    parser.add_argument(
        "--strategy",
        choices=("lg-batch", "wf-fifo", "wf-history", "wf-cache"),
    )
    parser.add_argument("--workload", choices=("burst", "open-loop"))
    parser.add_argument("--gpu-count", type=int, choices=(1, 2, 3))
    parser.add_argument("--repetition", type=int, choices=(1, 2, 3, 4, 5))
    parser.add_argument("--load-percent", type=int, choices=(75, 125))


def _select_trials(
    args: argparse.Namespace,
    trials: tuple[TrialSpec, ...],
) -> tuple[TrialSpec, ...]:
    trial_ids = set(args.trial_id) if args.trial_id else None
    selected = tuple(
        trial
        for trial in trials
        if (trial_ids is None or trial.trial_id in trial_ids)
        and (args.scenario is None or trial.scenario == args.scenario)
        and (args.strategy is None or trial.strategy == args.strategy)
        and (args.workload is None or trial.workload == args.workload)
        and (args.gpu_count is None or trial.gpu_count == args.gpu_count)
        and (args.repetition is None or trial.repetition == args.repetition)
        and (args.load_percent is None or trial.load_percent == args.load_percent)
    )
    if trial_ids is not None:
        missing = trial_ids - {trial.trial_id for trial in selected}
        if missing:
            raise KeyError(f"unknown or filtered trial ids: {sorted(missing)}")
    return selected


def _has_run_filter(args: argparse.Namespace) -> bool:
    return any(
        getattr(args, name) is not None
        for name in (
            "scenario",
            "strategy",
            "workload",
            "gpu_count",
            "repetition",
            "load_percent",
        )
    )


def _spawn_trial_worker(
    config: ExperimentConfig,
    trial: TrialSpec,
    available_gpu_ids: tuple[int, ...],
    *,
    timeout_sec: float | None = None,
) -> WorkerAttempt:
    return _run_worker_attempt(
        config,
        ("_run-trial", trial.trial_id),
        available_gpu_ids[: trial.gpu_count],
        timeout_sec=config.trial_timeout_sec if timeout_sec is None else timeout_sec,
        log_path=_next_worker_log_path(config, trial),
    )


def _run_campaign(
    config: ExperimentConfig,
    trials: tuple[TrialSpec, ...],
    available_gpu_ids: tuple[int, ...],
) -> int:
    if len(available_gpu_ids) < max(trial.gpu_count for trial in trials):
        raise ValueError("--gpu-ids does not provide enough GPUs for the selection")
    started_at = time.time()
    started_monotonic = time.monotonic()
    deadline = started_monotonic + config.campaign_timeout_sec
    formal_positions = {
        trial.trial_id: index
        for index, trial in enumerate(build_trial_matrix(), start=1)
    }
    failed = 0
    expired = False
    state = "running"
    _write_campaign_status(
        config,
        state=state,
        started_at=started_at,
        selected=len(trials),
        processed=0,
        failed=0,
        current_trial=None,
    )
    try:
        for selected_index, trial in enumerate(trials, start=1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                expired = True
                break
            print(f"[{selected_index}/{len(trials)}] {trial.trial_id}", flush=True)
            outcome = _run_trial_with_policy(
                config,
                trial,
                available_gpu_ids,
                deadline=deadline,
            )
            if outcome == "failed":
                failed += 1
            elif outcome == "campaign-timeout":
                failed += 1
                expired = True
            _write_campaign_status(
                config,
                state=state,
                started_at=started_at,
                selected=len(trials),
                processed=selected_index,
                failed=failed,
                current_trial=trial.trial_id,
            )
            if formal_positions[trial.trial_id] in EVIDENCE_GROUP_ENDS:
                _publish_results(config)
            if expired:
                break
    except Exception:
        state = "aborted"
        raise
    finally:
        _publish_results(config)
        failed = _failed_trial_count(config, trials)
        if state != "aborted":
            state = "partial" if failed or expired else "completed"
        _write_campaign_status(
            config,
            state=state,
            started_at=started_at,
            selected=len(trials),
            processed=_processed_trial_count(config, trials),
            failed=failed,
            current_trial=None,
        )
    return 1 if failed or expired else 0


def _run_trial_with_policy(
    config: ExperimentConfig,
    trial: TrialSpec,
    available_gpu_ids: tuple[int, ...],
    *,
    deadline: float,
) -> Literal["completed", "failed", "campaign-timeout"]:
    for attempt_number in (1, 2):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "campaign-timeout"
        timeout_sec = min(config.trial_timeout_sec, remaining)
        attempt = _spawn_trial_worker(
            config,
            trial,
            available_gpu_ids,
            timeout_sec=timeout_sec,
        )
        if attempt.return_code == 0:
            return "completed"
        category = _failure_category(
            attempt,
            campaign_limited=timeout_sec < config.trial_timeout_sec,
        )
        _archive_failed_attempt(
            config,
            trial,
            attempt,
            category=category,
            retry_scheduled=category == "infrastructure" and attempt_number == 1,
        )
        print(
            f"failed: {trial.trial_id} category={category} log={attempt.log_path}",
            flush=True,
        )
        if category == "campaign-timeout":
            return "campaign-timeout"
        if category == "performance-timeout":
            return "failed"
        if category != "infrastructure":
            raise subprocess.CalledProcessError(
                attempt.return_code,
                ("_run-trial", trial.trial_id),
            )
        if attempt_number == 2:
            return "failed"
    raise AssertionError("trial retry loop exhausted")


def _run_worker_attempt(
    config: ExperimentConfig,
    worker_args: tuple[str, ...],
    gpu_ids: tuple[int, ...],
    *,
    timeout_sec: float,
    log_path: Path,
) -> WorkerAttempt:
    expected_gpu_count = _worker_gpu_count(worker_args)
    if len(gpu_ids) != expected_gpu_count:
        raise ValueError(f"worker requires {expected_gpu_count} GPUs")
    visible = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = visible
    env["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    env["WORKFLOW_EXPERIMENT_GPU_IDS"] = visible
    command = (
        sys.executable,
        "-m",
        "experiment.workflow",
        "--output-root",
        str(config.output_root),
        *worker_args,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("xb") as log_file:
        try:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as error:
            log_file.write(f"worker launch failed: {error}\n".encode())
            return WorkerAttempt(127, False, log_path)
        try:
            return_code = process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            _stop_worker_process_group(process)
            return WorkerAttempt(process.returncode or -signal.SIGKILL, True, log_path)
        _stop_worker_process_group(process)
        return WorkerAttempt(return_code, False, log_path)


def _stop_worker_process_group(process: subprocess.Popen[bytes]) -> None:
    for requested_signal in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, requested_signal)
        except ProcessLookupError:
            break
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                continue
            break


def _failure_category(
    attempt: WorkerAttempt,
    *,
    campaign_limited: bool,
) -> Literal[
    "campaign-timeout",
    "performance-timeout",
    "infrastructure",
    "worker-error",
]:
    if attempt.timed_out:
        return "campaign-timeout" if campaign_limited else "performance-timeout"
    log = attempt.log_path.read_text(encoding="utf-8", errors="replace")
    if "TimeoutError" in log or "timed out" in log:
        return "performance-timeout"
    if any(marker in log for marker in INFRASTRUCTURE_FAILURE_MARKERS):
        return "infrastructure"
    return "worker-error"


def _archive_failed_attempt(
    config: ExperimentConfig,
    trial: TrialSpec,
    attempt: WorkerAttempt,
    *,
    category: str,
    retry_scheduled: bool,
) -> Path:
    root = config.output_root.resolve()
    trial_failure_root = root / "failed" / trial.trial_id
    attempt_number = len(tuple(trial_failure_root.glob("attempt-*"))) + 1
    attempt_dir = trial_failure_root / f"attempt-{attempt_number:03d}"
    attempt_dir.mkdir(parents=True)
    partial_trial = root / "trials" / trial.trial_id
    archived_trial: Path | None = None
    if partial_trial.exists():
        archived_trial = attempt_dir / "partial_trial"
        shutil.move(partial_trial, archived_trial)
    payload = {
        "version": 1,
        "experiment_id": config.experiment_id,
        "trial_id": trial.trial_id,
        **trial.model_dump(mode="json"),
        "failed_at": time.time(),
        "category": category,
        "archive_reason": category,
        "return_code": attempt.return_code,
        "timed_out": attempt.timed_out,
        "retry_scheduled": retry_scheduled,
        "worker_log": attempt.log_path.resolve().relative_to(root).as_posix(),
        "partial_trial": (
            None
            if archived_trial is None
            else archived_trial.resolve().relative_to(root).as_posix()
        ),
    }
    write_json_exclusive(attempt_dir / "failure.json", payload)
    return attempt_dir


def _next_worker_log_path(config: ExperimentConfig, trial: TrialSpec) -> Path:
    directory = config.output_root.resolve() / "campaign/worker_logs" / trial.trial_id
    number = len(tuple(directory.glob("attempt-*.log"))) + 1
    return directory / f"attempt-{number:03d}.log"


def _publish_results(config: ExperimentConfig) -> None:
    result = regenerate_analysis(config.output_root.resolve())
    from experiment.workflow.report import publish_report

    published = publish_report(config.output_root, RESULTS_DOCUMENT)
    print(
        f"analysis: completed={result.completed_trial_count} "
        f"groups={result.aligned_group_count} report={published.results_document}",
        flush=True,
    )


def _write_campaign_status(
    config: ExperimentConfig,
    *,
    state: str,
    started_at: float,
    selected: int,
    processed: int,
    failed: int,
    current_trial: str | None,
) -> None:
    path = config.output_root.resolve() / "campaign/status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    payload = {
        "version": 1,
        "experiment_id": config.experiment_id,
        "state": state,
        "started_at": started_at,
        "updated_at": time.time(),
        "selected_trial_count": selected,
        "processed_trial_count": processed,
        "failed_trial_count": failed,
        "current_trial_id": current_trial,
    }
    temporary.write_text(f"{canonical_json(payload)}\n", encoding="utf-8")
    os.replace(temporary, path)


def _processed_trial_count(
    config: ExperimentConfig,
    trials: tuple[TrialSpec, ...],
) -> int:
    root = config.output_root.resolve()
    return sum(
        (root / "trials" / trial.trial_id / "completed.json").is_file()
        or (root / "failed" / trial.trial_id).is_dir()
        for trial in trials
    )


def _failed_trial_count(
    config: ExperimentConfig,
    trials: tuple[TrialSpec, ...],
) -> int:
    root = config.output_root.resolve()
    return sum(
        not (root / "trials" / trial.trial_id / "completed.json").is_file()
        and (root / "failed" / trial.trial_id).is_dir()
        for trial in trials
    )


def _spawn_calibration_worker(
    config: ExperimentConfig,
    scenario: Scenario,
    gpu_count: int,
    repetition: int,
    available_gpu_ids: tuple[int, ...],
) -> None:
    _spawn_worker(
        config,
        ("_run-calibration", scenario, str(gpu_count), str(repetition)),
        available_gpu_ids[:gpu_count],
    )


def _spawn_worker(
    config: ExperimentConfig,
    worker_args: tuple[str, ...],
    gpu_ids: tuple[int, ...],
) -> None:
    expected_gpu_count = _worker_gpu_count(worker_args)
    if len(gpu_ids) != expected_gpu_count:
        raise ValueError(f"worker requires {expected_gpu_count} GPUs")
    visible = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = visible
    env["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    env["WORKFLOW_EXPERIMENT_GPU_IDS"] = visible
    subprocess.run(
        (
            sys.executable,
            "-m",
            "experiment.workflow",
            "--output-root",
            str(config.output_root),
            *worker_args,
        ),
        check=True,
        env=env,
    )


def _worker_gpu_count(worker_args: tuple[str, ...]) -> int:
    if worker_args[0] == "_run-trial":
        return _trial_by_id(worker_args[1]).gpu_count
    if worker_args[0] == "_run-calibration":
        return int(worker_args[2])
    raise ValueError(f"unknown worker command: {worker_args[0]}")


def _parse_gpu_ids(value: str) -> tuple[int, ...]:
    try:
        gpu_ids = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise ValueError("--gpu-ids must contain comma-separated integers") from error
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)) or any(i < 0 for i in gpu_ids):
        raise ValueError("--gpu-ids must contain unique non-negative integers")
    return gpu_ids


def _trial_by_id(trial_id: str) -> TrialSpec:
    matches = [trial for trial in build_trial_matrix() if trial.trial_id == trial_id]
    if len(matches) != 1:
        raise KeyError(f"unknown trial id: {trial_id}")
    return matches[0]


def _print_status(config: ExperimentConfig) -> None:
    output_root = config.output_root.resolve()
    trial_root = output_root / "trials"
    failure_root = output_root / "failed"
    complete = 0
    incomplete = 0
    failed = 0
    for trial in build_trial_matrix():
        directory = trial_root / trial.trial_id
        if (directory / "completed.json").is_file():
            validate_completion(directory)
            complete += 1
        elif (failure_root / trial.trial_id).is_dir():
            failed += 1
        elif directory.exists():
            incomplete += 1
    total = len(build_trial_matrix())
    pending = total - complete - failed - incomplete
    print(
        f"complete={complete} incomplete={incomplete} failed={failed} "
        f"pending={pending} total={total}"
    )
