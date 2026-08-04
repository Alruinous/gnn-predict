"""Render the shared-pool 12-GPU replay with the paper evaluation layout."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = ROOT / "paper" / "hpca2027-sagepilot"
PAPER_SCRIPTS = PAPER_ROOT / "scripts" / "evaluation"
if str(PAPER_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PAPER_SCRIPTS))

plot: Any = importlib.import_module("plot_evaluation")

CALCULATION = "homothetic_12gpu_shared_pool_counterfactual"
ORACLE_CALCULATION = "homothetic_trace_offline_reference"
ARRIVALS = (
    ("burst", "Burst"),
    ("poisson_r050", "Poisson 0.150"),
    ("poisson_r025", "Poisson 0.075"),
)
GPU_STATES = (
    ("generation_gpu_sec", "Generation", ""),
    ("idle_resident_gpu_sec", "Idle residency", "////"),
    ("loading_gpu_sec", "Model loading", "\\\\\\\\"),
    ("evicting_gpu_sec", "Eviction", "xx"),
)
DEFAULT_DATA_DIR = PAPER_ROOT / "data" / "evaluation_12gpu_counterfactual"
DEFAULT_OUTPUT_DIR = ROOT / "tmp" / "evaluation_12gpu_counterfactual_drafts"
FIGURE_STEMS = (
    "fig_system_performance",
    "fig_gpu_time_efficiency",
    "fig_component_ablation",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    return float(value)


def _selected_rows(
    rows: Sequence[object],
    manifest: Mapping[str, object],
) -> list[Mapping[str, object]]:
    selection = _mapping(manifest.get("selection"), "selection")
    selected = []
    for arrival, _ in ARRIVALS:
        arrival_selection = _mapping(selection.get(arrival), f"selection.{arrival}")
        for arm in plot.ALL_SYSTEM_DATA_ARMS:
            repeats = arrival_selection.get(arm)
            if not isinstance(repeats, list) or not all(
                isinstance(repeat, str) for repeat in repeats
            ):
                raise TypeError(f"selection.{arrival}.{arm} must be a string array")
            for repeat in repeats:
                matches = []
                for value in rows:
                    row = _mapping(value, "replay row")
                    if (
                        row.get("arrival") == arrival
                        and row.get("arm") == arm
                        and row.get("repeat") == repeat
                    ):
                        matches.append(row)
                if len(matches) != 1:
                    raise ValueError(
                        f"expected one replay row for {arrival}/{arm}/{repeat}"
                    )
                selected.append(matches[0])
    if len(selected) != 26:
        raise ValueError(f"expected 26 selected replay rows, got {len(selected)}")
    return selected


def _oracle_plot_row(value: object) -> dict[str, object]:
    row = _mapping(value, "Oracle row")
    overall = _mapping(row.get("overall"), "Oracle overall")
    evidence = _mapping(row.get("evidence"), "Oracle evidence")
    if evidence.get("calculation") != ORACLE_CALCULATION:
        raise ValueError("Oracle row uses a stale calculation")
    models = {}
    for model_name in ("Qwen3-4B", "Qwen3-8B", "Qwen3-14B"):
        metrics = _mapping(row.get(model_name), f"Oracle {model_name}")
        models[model_name] = {
            "session_model_makespan_sec": _number(
                metrics.get("makespan_sec"), f"Oracle {model_name} makespan"
            ),
            "session_model_p95_sec": _number(
                metrics.get("p95_sec"), f"Oracle {model_name} p95"
            ),
        }
    return {
        "hardware": row.get("hardware"),
        "arrival": row.get("arrival"),
        "repeat": row.get("repeat"),
        "session_count": 180,
        "makespan_sec": _number(overall.get("makespan_sec"), "Oracle overall makespan"),
        "session_p95_sec": _number(overall.get("p95_sec"), "Oracle overall p95"),
        "model": models,
        "evidence": dict(evidence),
    }


def _configure_plot_contract() -> None:
    plot.ARRIVALS = ARRIVALS
    plot.GPU_EFFICIENCY_WORKFLOW_SESSION_COUNT = 60
    plot.GPU_EFFICIENCY_STATES = GPU_STATES


def _validate_plot_payload(payload: Mapping[str, object]) -> None:
    for arrival, _ in ARRIVALS:
        for kind, key, _ in plot.SYSTEM_PANELS:
            for metric in ("makespan", "p95"):
                path = plot.metric_path(kind, key, metric)
                oracle_value = plot.aggregate_oracle(payload, path, arrival=arrival)
                sagepilot_value = plot.aggregate(
                    payload,
                    "sagepilot",
                    path,
                    arrival=arrival,
                )
                if not 0.0 < oracle_value <= sagepilot_value:
                    raise ValueError(
                        f"Oracle exceeds SagePilot for {arrival}/{'.'.join(path)}"
                    )
    plot.gpu_time_per_workflow_session(payload)


def build_plot_payload(data_dir: Path) -> dict[str, object]:
    _configure_plot_contract()
    replay = read_json_object(data_dir / "replay_rows.json")
    oracle = read_json_object(data_dir / "oracle_rows.json")
    summary = read_json_object(data_dir / "summary.json")
    manifest = read_json_object(data_dir / "manifest.json")
    for name, payload in (
        ("replay rows", replay),
        ("summary", summary),
        ("manifest", manifest),
    ):
        if payload.get("calculation") != CALCULATION:
            raise ValueError(f"{name} uses a stale calculation")
    if oracle.get("calculation") != ORACLE_CALCULATION:
        raise ValueError("Oracle payload uses a stale calculation")

    workload = _mapping(manifest.get("workload"), "workload")
    if workload.get("sessions_per_workflow") != 60:
        raise ValueError("12-GPU replay must contain 60 sessions per workflow")
    shared_pool = _mapping(manifest.get("shared_pool"), "shared_pool")
    if shared_pool.get("topology") != "one_shared_12_gpu_pool":
        raise ValueError("12-GPU replay must use one shared GPU pool")
    if shared_pool.get("cross_workflow_model_sharing") is not True:
        raise ValueError("12-GPU replay must share models across workflows")
    workflow_sessions = 60
    gpu_summary = _mapping(summary.get("gpu_time_efficiency"), "gpu_time_efficiency")
    workflows: dict[str, object] = {}
    for workflow_name, _ in plot.GPU_EFFICIENCY_WORKFLOWS:
        source_methods = _mapping(gpu_summary.get(workflow_name), workflow_name)
        methods: dict[str, object] = {}
        for arm in plot.SYSTEM_ARMS:
            source_metrics = _mapping(source_methods.get(arm), arm)
            methods[arm] = {
                field: _number(source_metrics.get(field), field) * workflow_sessions
                for field, _, _ in GPU_STATES
            }
        workflows[workflow_name] = methods

    rows = replay.get("rows")
    oracle_rows = oracle.get("rows")
    if not isinstance(rows, list) or len(rows) != 72:
        raise ValueError("shared-pool replay must contain 72 rows")
    if not isinstance(oracle_rows, list) or len(oracle_rows) != 3:
        raise ValueError("12-GPU Oracle must contain three rows")
    payload = {
        "schema": 2,
        "rows": _selected_rows(rows, manifest),
        "oracle_lower_bound": {"rows": [_oracle_plot_row(row) for row in oracle_rows]},
        "gpu_time_efficiency": {
            "arrival": "burst",
            "workflow_session_count": workflow_sessions,
            "shared_time_allocation": "per_model_generation_share",
            "workflows": workflows,
        },
    }
    _validate_plot_payload(payload)
    return cast(dict[str, object], payload)


def render_png(pdf_path: Path) -> None:
    output_prefix = pdf_path.with_suffix("")
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-singlefile",
            "-r",
            "200",
            str(pdf_path),
            str(output_prefix),
        ],
        check=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    formal_figure_dir = (PAPER_ROOT / "figures").resolve()
    if output_dir == formal_figure_dir or formal_figure_dir in output_dir.parents:
        raise ValueError("draft output must stay outside the paper figures directory")
    payload = build_plot_payload(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot.configure_matplotlib()
    plot.plot_system_performance(payload, output_dir)
    plot.plot_gpu_time_efficiency(payload, output_dir)
    plot.plot_component_ablation(payload, output_dir)
    for stem in FIGURE_STEMS:
        render_png(output_dir / f"{stem}.pdf")
    print(f"wrote 12-GPU evaluation drafts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
