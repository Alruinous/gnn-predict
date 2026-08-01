"""Master: attach Ray, register workflow files, run one experiment, then stop."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ray
import yaml

from workflow.artifacts import AcceleratorConfig, SchedulerConfig
from workflow.fleet import NodeFunction, WorkflowFleet
from workflow.fusion import fuse_workflow
from workflow.schema import Workflow


def resolve_dotted(spec: str) -> Any:
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name.strip() or not attr.strip():
        raise ValueError(f"dotted path must be 'module:attr': {spec!r}")
    module = importlib.import_module(module_name.strip())
    try:
        return getattr(module, attr.strip())
    except AttributeError as error:
        raise AttributeError(f"{module_name} has no attribute {attr!r}") from error


def resolve_functions(spec: str) -> dict[str, NodeFunction]:
    target = resolve_dotted(spec)
    registry = target() if callable(target) else target
    return dict(registry)


def parse_mapping(text: str | None) -> dict[str, str]:
    if not text:
        return {}
    result: dict[str, str] = {}
    for pair in text.split(","):
        key, sep, value = pair.partition("=")
        if not sep or not key.strip() or not value.strip():
            raise ValueError(f"invalid key=value entry: {pair!r}")
        result[key.strip()] = value.strip()
    return result


def parse_int_mapping(text: str | None) -> dict[str, int]:
    return {key: int(value) for key, value in parse_mapping(text).items()}


def parse_float_mapping(text: str | None) -> dict[str, float]:
    return {key: float(value) for key, value in parse_mapping(text).items()}


def load_workflows(paths: Sequence[Path]) -> list[Workflow]:
    workflows: list[Workflow] = []
    seen: set[str] = set()
    for path in paths:
        with path.open() as stream:
            workflow = Workflow.model_validate(yaml.safe_load(stream))
        if workflow.workflow_name in seen:
            raise ValueError(
                f"duplicate workflow_name across files: {workflow.workflow_name}"
            )
        seen.add(workflow.workflow_name)
        workflows.append(workflow)
    return workflows


def load_scheduler_mapping(path: Path) -> dict[str, object]:
    with path.open() as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise TypeError(f"expected a mapping in {path}")
    return data


def _resolve_gpu_kind(
    resources: Mapping[str, object],
    hostname: str,
    host_gpu_kind: Mapping[str, str],
) -> str:
    reported = {
        key.removeprefix("accelerator_type:").casefold()
        for key, value in resources.items()
        if key.startswith("accelerator_type:") and value
    }
    override = host_gpu_kind.get(hostname)
    if override is not None:
        override = override.casefold()
        if reported and override not in reported:
            raise ValueError(
                f"--host-gpu-kind {hostname}={override} conflicts with "
                f"Ray-reported {sorted(reported)}"
            )
        return override
    if len(reported) == 1:
        return next(iter(reported))
    if not reported:
        raise ValueError(
            f"Ray node {hostname} has no accelerator_type; "
            f"pass --host-gpu-kind {hostname}=<kind>"
        )
    raise ValueError(
        f"Ray node {hostname} reports multiple GPU kinds: {sorted(reported)}"
    )


def discover_accelerators(
    nodes: Sequence[Mapping[str, Any]],
    *,
    gpu_mem_mb: Mapping[str, int],
    host_gpu_kind: Mapping[str, str] | None = None,
) -> tuple[AcceleratorConfig, ...]:
    host_gpu_kind = host_gpu_kind or {}
    accelerators: list[AcceleratorConfig] = []
    for node in nodes:
        if not node.get("Alive"):
            continue
        resources = node.get("Resources") or {}
        gpu_count = int(resources.get("GPU", 0))
        if gpu_count <= 0:
            continue
        hostname = node["NodeManagerHostname"]
        gpu_kind = _resolve_gpu_kind(resources, hostname, host_gpu_kind)
        if gpu_kind not in gpu_mem_mb:
            raise ValueError(f"no --gpu-mem entry for gpu_kind: {gpu_kind}")
        accelerators.extend(
            AcceleratorConfig(
                hostname=hostname,
                gpu_kind=gpu_kind,
                local_index=local_index,
                total_mem_mb=gpu_mem_mb[gpu_kind],
            )
            for local_index in range(gpu_count)
        )
    if not accelerators:
        raise RuntimeError("no live Ray GPU nodes were discovered")
    return tuple(accelerators)


def wait_for_gpu_nodes(
    min_gpus: int,
    timeout_sec: float,
    *,
    poll_interval_sec: float = 1.0,
) -> list[Mapping[str, object]]:
    deadline = time.monotonic() + timeout_sec
    while True:
        nodes = [node for node in ray.nodes() if node.get("Alive")]
        total = sum(int((node.get("Resources") or {}).get("GPU", 0)) for node in nodes)
        if total >= min_gpus:
            return nodes
        if time.monotonic() >= deadline:
            raise TimeoutError(f"only {total}/{min_gpus} GPUs joined before timeout")
        time.sleep(poll_interval_sec)


def _install_signal_handlers() -> None:
    def handler(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="workflow.master")
    parser.add_argument(
        "--workflow-files", required=True, help="comma-separated workflow YAMLs"
    )
    parser.add_argument(
        "--functions", required=True, help="module:attr -> name->callable registry"
    )
    parser.add_argument(
        "--experiment", required=True, help="module:attr -> experiment run callable"
    )
    parser.add_argument(
        "--experiment-config", default=None, help="experiment config YAML"
    )
    parser.add_argument(
        "--scheduler-config", required=True, help="SchedulerConfig YAML"
    )
    parser.add_argument(
        "--vllm-python", default=None, help="override vllm_python_executable"
    )
    parser.add_argument(
        "--predictions", default=None, help="resource contract cache (agent workflows)"
    )
    parser.add_argument(
        "--gpu-mem", default=None, help="gpu_kind=total_mem_mb,... for discovery"
    )
    parser.add_argument(
        "--host-gpu-kind", default=None, help="hostname=gpu_kind,... override"
    )
    parser.add_argument(
        "--min-gpus", type=int, default=0, help="wait for this many GPUs then discover"
    )
    parser.add_argument("--discover-timeout-sec", type=float, default=180.0)
    parser.add_argument(
        "--priority-weight", default=None, help="workflow_name=weight,..."
    )
    parser.add_argument(
        "--fuse-nodes",
        action="store_true",
        help="collapse adjacent same-model agent chains into one node",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="run root; fleet dir is <output-dir>/<run-id>",
    )
    parser.add_argument("--run-id", default=None, help="defaults to a timestamp")
    parser.add_argument("--ray-address", default=None, help="defaults to 'auto'")
    parser.add_argument("--shutdown-timeout-sec", type=float, default=600.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_id
    workflow_paths = [
        Path(item.strip()) for item in args.workflow_files.split(",") if item.strip()
    ]
    if not workflow_paths:
        raise ValueError("--workflow-files must list at least one YAML")

    # Attach before the fleet so it sees an initialized Ray and never tears the
    # head down on shutdown (WorkflowFleet._owns_ray stays False).
    ray.init(address=args.ray_address or "auto")
    try:
        config_data = load_scheduler_mapping(Path(args.scheduler_config))
        if args.vllm_python:
            config_data["vllm_python_executable"] = args.vllm_python
        if args.min_gpus > 0:
            wait_for_gpu_nodes(args.min_gpus, args.discover_timeout_sec)
            accelerators = discover_accelerators(
                list(ray.nodes()),
                gpu_mem_mb=parse_int_mapping(args.gpu_mem),
                host_gpu_kind=parse_mapping(args.host_gpu_kind),
            )
            config_data["accelerators"] = [acc.model_dump() for acc in accelerators]
        scheduler_config = SchedulerConfig.model_validate(config_data)

        functions = resolve_functions(args.functions)
        workflows = load_workflows(workflow_paths)
        if args.fuse_nodes:
            workflows = [fuse_workflow(workflow) for workflow in workflows]
        weights = parse_float_mapping(args.priority_weight)
        experiment = resolve_dotted(args.experiment)
        if not callable(experiment):
            raise TypeError(
                f"--experiment must resolve to a callable: {args.experiment!r}"
            )
        experiment_config = _load_experiment_config(args.experiment_config)

        fleet = WorkflowFleet(
            scheduler_config=scheduler_config,
            output_dir=run_dir,
            run_id=run_id,
            prediction_path=args.predictions,
        )
        fleet.start()
        write_run_manifest(
            run_dir,
            run_id=run_id,
            scheduler_config=scheduler_config,
            workflow_paths=workflow_paths,
            predictions=args.predictions,
            experiment_config_path=args.experiment_config,
            experiment_config=experiment_config,
            fuse_nodes=args.fuse_nodes,
            priority_weight=weights,
        )
        _install_signal_handlers()
        try:
            for workflow in workflows:
                fleet.register_workflow(
                    workflow,
                    functions=functions,
                    output_dir=run_dir / workflow.workflow_name,
                    priority_weight=weights.get(workflow.workflow_name, 1.0),
                )
            experiment(
                fleet, output_dir=run_dir, run_id=run_id, config=experiment_config
            )
        finally:
            fleet.shutdown(timeout_sec=args.shutdown_timeout_sec)
    finally:
        if ray.is_initialized():
            ray.shutdown()
    return 0


MANIFEST_FILENAME = "run_manifest.json"


def _file_digest(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _git_revision() -> dict[str, object] | None:
    # Provenance is nice to have; a checkout without git must not fail an experiment.
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": commit, "dirty": bool(status.strip())}


def write_run_manifest(
    run_dir: Path,
    *,
    run_id: str,
    scheduler_config: SchedulerConfig,
    workflow_paths: Sequence[Path],
    predictions: str | None,
    experiment_config_path: str | None,
    experiment_config: Mapping[str, object],
    fuse_nodes: bool,
    priority_weight: Mapping[str, float],
) -> Path:
    """Record every factor that distinguishes this run from its neighbours.

    The trace records what happened, not what was asked for: the accelerator
    inventory, the prediction cache and the fusion flag appear nowhere in it. A
    factorial sweep has to recover those from somewhere other than the directory
    name, which is what this file is for.
    """
    manifest: dict[str, object] = {
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "git": _git_revision(),
        "scheduler_config": scheduler_config.model_dump(mode="json"),
        "predictions": None if predictions is None else _file_digest(predictions),
        "workflow_files": [_file_digest(path) for path in workflow_paths],
        "fuse_nodes": fuse_nodes,
        "experiment_config": (
            None
            if experiment_config_path is None
            else _file_digest(experiment_config_path)
        ),
        "experiment": dict(experiment_config),
        "priority_weight": dict(priority_weight),
    }
    path = run_dir / MANIFEST_FILENAME
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def _load_experiment_config(path: str | None) -> dict[str, object]:
    if not path:
        return {}
    with Path(path).open() as stream:
        data = yaml.safe_load(stream)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError("experiment config must be a mapping")
    return data


if __name__ == "__main__":
    raise SystemExit(main())
