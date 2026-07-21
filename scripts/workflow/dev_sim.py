"""Simulate the master + workers topology locally with multiple processes (no GPU).

Starts a local Ray head + workers on loopback, then runs `python -m workflow.master`
against function-only echo workflows to exercise register / submit / trace / shutdown.
Assumes exclusive use of this box's Ray (ray stop clears it).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
RAY_HOST = "127.0.0.1"
RAY_PORT = "6699"
RAY_ADDRESS = f"{RAY_HOST}:{RAY_PORT}"
WORKFLOW_NAMES = ("dev-echo-a", "dev-echo-b")


def _ray_bin() -> str:
    candidate = Path(sys.executable).with_name("ray")
    return str(candidate) if candidate.exists() else "ray"


def _echo_workflow(name: str) -> dict[str, object]:
    return {
        "workflow_name": name,
        "nodes": [
            {
                "name": "ingest",
                "type": "function",
                "function": "echo",
                "routing": "broadcast",
            },
            {
                "name": "reply",
                "type": "function",
                "function": "echo",
                "routing": "broadcast",
            },
        ],
        "edges": [{"source": "ingest", "target": "reply"}],
    }


def _static_config(names: Sequence[str], sessions_each: int) -> dict[str, object]:
    return {
        "timeout_sec": 120,
        "sessions": [
            {
                "workflow": name,
                "session_id": f"{name}-s{index}",
                "inputs": {"text": f"hello {name} {index}"},
            }
            for name in names
            for index in range(sessions_each)
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dev_sim")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sessions-each", type=int, default=3)
    args = parser.parse_args(argv)

    ray_bin = _ray_bin()
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    workdir = Path(tempfile.mkdtemp(prefix="workflow_dev_sim_"))
    subprocess.run([ray_bin, "stop", "--force"], check=False, env=env)
    try:
        subprocess.run(
            [
                ray_bin,
                "start",
                "--head",
                "--node-ip-address",
                RAY_HOST,
                "--port",
                RAY_PORT,
                "--num-cpus",
                "4",
                "--disable-usage-stats",
            ],
            check=True,
            env=env,
        )
        for _ in range(args.workers):
            subprocess.run(
                [
                    ray_bin,
                    "start",
                    "--address",
                    RAY_ADDRESS,
                    "--node-ip-address",
                    RAY_HOST,
                    "--num-cpus",
                    "2",
                    "--disable-usage-stats",
                ],
                check=True,
                env=env,
            )

        workflow_paths = []
        for name in WORKFLOW_NAMES:
            path = workdir / f"{name}.yaml"
            path.write_text(yaml.safe_dump(_echo_workflow(name), sort_keys=False))
            workflow_paths.append(path)
        scheduler_path = workdir / "scheduler.yaml"
        scheduler_path.write_text(yaml.safe_dump({"policy": "fifo"}))
        experiment_path = workdir / "static.yaml"
        experiment_path.write_text(
            yaml.safe_dump(
                _static_config(WORKFLOW_NAMES, args.sessions_each), sort_keys=False
            )
        )
        output_dir = workdir / "out"
        run_id = "devrun"

        subprocess.run(
            [
                sys.executable,
                "-m",
                "workflow.master",
                "--workflow-files",
                ",".join(str(path) for path in workflow_paths),
                "--functions",
                "experiment.workflow.dev_functions:build_registry",
                "--experiment",
                "experiment.workflow.experiments.static_inputs:run",
                "--experiment-config",
                str(experiment_path),
                "--scheduler-config",
                str(scheduler_path),
                "--output-dir",
                str(output_dir),
                "--run-id",
                run_id,
                "--ray-address",
                RAY_ADDRESS,
            ],
            check=True,
            env=env,
        )

        trace = output_dir / run_id / "workflow_trace.jsonl"
        if not trace.is_file() or trace.stat().st_size == 0:
            raise RuntimeError(f"expected trace not written: {trace}")
        print(f"OK dev-sim trace: {trace} ({trace.stat().st_size} bytes)")
        return 0
    finally:
        subprocess.run([ray_bin, "stop", "--force"], check=False, env=env)
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
