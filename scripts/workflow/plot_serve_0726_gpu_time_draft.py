"""Plot GPU-time composition for the archived serve_0726 Poisson cohort."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from experiment.workflow.analysis import summarize_trial

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = ROOT / "output" / "serve_0726" / "6w_poisson_a1v2"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "serve_0726_gpu_time_draft"

GPU_COUNT = 3
EXPECTED_SESSIONS = 120
METHODS = ("fifo", "kairos", "gbdt", "cache")
METHOD_LABELS = {
    "fifo": "Parrot",
    "kairos": "Kairos",
    "gbdt": "GBDT",
    "cache": "SagePilot",
}
STATE_KEYS = (
    "generation_gpu_sec",
    "idle_resident_gpu_sec",
    "loading_gpu_sec",
    "other_gpu_sec",
)
STATE_COLORS = {
    "generation_gpu_sec": "#3D7F83",
    "idle_resident_gpu_sec": "#DDD4BE",
    "loading_gpu_sec": "#D98A52",
    "other_gpu_sec": "#626A73",
}
STATE_LABELS = {
    "generation_gpu_sec": "Generation",
    "idle_resident_gpu_sec": "Idle resident",
    "loading_gpu_sec": "Loading",
    "other_gpu_sec": "Other",
}
STATE_TEXT = {
    "generation_gpu_sec": "white",
    "idle_resident_gpu_sec": "#202124",
    "loading_gpu_sec": "#202124",
    "other_gpu_sec": "white",
}

TEXT = "#202124"
MUTED = "#626A73"
GRID = "#D9DEE3"
FIGURE_WIDTH_IN = 3.33
FONT_SIZE = 8.5
ANNOTATION_FONT_SIZE = 8.0
FIXED_TIMESTAMP = datetime(2026, 8, 1, tzinfo=UTC)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def number_at(value: Mapping[str, object], key: str, context: str) -> float:
    raw = value.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError(f"{context}.{key} must be numeric")
    result = float(raw)
    if not math.isfinite(result):
        raise ValueError(f"{context}.{key} must be finite")
    return result


def integer_at(value: Mapping[str, object], key: str, context: str) -> int:
    raw = value.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError(f"{context}.{key} must be an integer")
    return raw


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def analyze_run(trial_dir: Path) -> dict[str, object]:
    archived = read_json_object(trial_dir / "run_summary.json")
    for key in ("submitted_session_count", "completed_session_count", "result_count"):
        actual = integer_at(archived, key, trial_dir.name)
        if actual != EXPECTED_SESSIONS:
            raise ValueError(
                f"{trial_dir.name}.{key} is {actual}, expected {EXPECTED_SESSIONS}"
            )
    failed = integer_at(archived, "failed_session_count", trial_dir.name)
    if failed != 0:
        raise ValueError(f"{trial_dir.name} contains {failed} failed sessions")

    summary = summarize_trial(trial_dir)
    completed = integer_at(summary, "completed_session_count", trial_dir.name)
    if completed != EXPECTED_SESSIONS:
        raise ValueError(
            f"{trial_dir.name} trace has {completed} completions, "
            f"expected {EXPECTED_SESSIONS}"
        )

    run_duration = number_at(summary, "run_duration_sec", trial_dir.name)
    generation = number_at(summary, "active_gpu_seconds", trial_dir.name)
    idle_resident = number_at(summary, "idle_resident_gpu_seconds", trial_dir.name)
    loading = number_at(summary, "loading_gpu_seconds", trial_dir.name)
    gpu_window = GPU_COUNT * run_duration
    other = gpu_window - generation - idle_resident - loading
    if run_duration <= 0.0 or other < 0.0:
        raise ValueError(f"{trial_dir.name} has incomplete GPU-time accounting")

    state_seconds = {
        "generation_gpu_sec": generation,
        "idle_resident_gpu_sec": idle_resident,
        "loading_gpu_sec": loading,
        "other_gpu_sec": other,
    }
    shares = {key: 100.0 * value / gpu_window for key, value in state_seconds.items()}
    if not math.isclose(sum(shares.values()), 100.0, abs_tol=1e-9):
        raise ValueError(f"{trial_dir.name} GPU-time shares do not sum to 100%")

    latency = summary.get("session_latency_sec")
    if not isinstance(latency, Mapping):
        raise TypeError(f"{trial_dir.name}.session_latency_sec must be an object")
    latency_values = cast(Mapping[str, object], latency)
    return {
        "run_id": trial_dir.name,
        "completed_session_count": completed,
        "makespan_sec": number_at(summary, "makespan_sec", trial_dir.name),
        "run_duration_sec": run_duration,
        "session_p50_sec": number_at(
            latency_values, "p50", f"{trial_dir.name}.latency"
        ),
        "session_p95_sec": number_at(
            latency_values, "p95", f"{trial_dir.name}.latency"
        ),
        "model_load_count": integer_at(summary, "model_load_count", trial_dir.name),
        "gpu_window_sec": gpu_window,
        **state_seconds,
        "gpu_time_distribution_pct": shares,
    }


def build_payload(input_root: Path) -> dict[str, object]:
    methods = {
        method: {
            "label": METHOD_LABELS[method],
            **analyze_run(input_root / method),
        }
        for method in METHODS
    }
    return {
        "schema": 1,
        "description": "GPU-time composition for serve_0726/6w_poisson_a1v2",
        "source": str(input_root),
        "experiment": {
            "arrival": "poisson",
            "nominal_aggregate_rate_per_sec": 0.075,
            "workflow_count": 6,
            "session_count": EXPECTED_SESSIONS,
            "gpu_count": GPU_COUNT,
            "hardware": "1xA100 + 2xV100",
            "runs_per_method": 1,
        },
        "methods": methods,
    }


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": 9.0,
            "legend.fontsize": FONT_SIZE,
            "xtick.labelsize": ANNOTATION_FONT_SIZE,
            "ytick.labelsize": ANNOTATION_FONT_SIZE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": TEXT,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.45,
            "pdf.fonttype": 42,
            "savefig.bbox": None,
            "savefig.facecolor": "white",
        }
    )


def plot_gpu_time(payload: Mapping[str, object], output_dir: Path) -> None:
    raw_methods = payload.get("methods")
    if not isinstance(raw_methods, Mapping):
        raise TypeError("payload.methods must be an object")
    methods = cast(Mapping[str, object], raw_methods)
    distributions: dict[str, list[float]] = {}
    for method in METHODS:
        raw_method = methods.get(method)
        if not isinstance(raw_method, Mapping):
            raise TypeError(f"payload.methods.{method} must be an object")
        method_values = cast(Mapping[str, object], raw_method)
        raw_distribution = method_values.get("gpu_time_distribution_pct")
        if not isinstance(raw_distribution, Mapping):
            raise TypeError(f"payload.methods.{method}.distribution must be an object")
        distribution_values = cast(Mapping[str, object], raw_distribution)
        distributions[method] = [
            number_at(
                distribution_values, state, f"payload.methods.{method}.distribution"
            )
            for state in STATE_KEYS
        ]

    figure, axes = plt.subplots(figsize=(FIGURE_WIDTH_IN, 3.05))
    positions = list(range(len(METHODS)))
    bottoms = [0.0] * len(METHODS)
    for state_index, state in enumerate(STATE_KEYS):
        values = [distributions[method][state_index] for method in METHODS]
        bars = axes.bar(
            positions,
            values,
            bottom=bottoms,
            width=0.62,
            color=STATE_COLORS[state],
            edgecolor="white",
            linewidth=0.45,
            label=STATE_LABELS[state],
            zorder=2,
        )
        for bar, bottom, value in zip(bars, bottoms, values, strict=True):
            if value >= 8.0:
                axes.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bottom + value / 2.0,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color=STATE_TEXT[state],
                    fontsize=ANNOTATION_FONT_SIZE,
                    fontweight="bold",
                )
        bottoms = [
            bottom + value for bottom, value in zip(bottoms, values, strict=True)
        ]

    axes.set_xticks(positions, [METHOD_LABELS[method] for method in METHODS])
    axes.set_ylim(0.0, 100.0)
    axes.set_yticks(
        (0.0, 25.0, 50.0, 75.0, 100.0),
        ("0%", "25%", "50%", "75%", "100%"),
    )
    axes.set_ylabel("Fraction of GPU time")
    axes.grid(axis="y", zorder=0)
    axes.set_axisbelow(True)
    axes.tick_params(axis="x", length=0, pad=3.0)
    axes.tick_params(axis="y", length=2.0, pad=1.5)
    axes.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.3,
    )
    figure.suptitle(
        "Poisson 0.075/s · 6 workflows\n1xA100 + 2xV100 · one run per system",
        x=0.58,
        y=0.985,
        fontsize=9.0,
        fontweight="bold",
        linespacing=0.9,
    )
    figure.subplots_adjust(left=0.19, right=0.98, bottom=0.18, top=0.69)
    save_figure(figure, output_dir, "fig_gpu_time_distribution_serve_0726_draft")


def save_figure(figure: Figure, output_dir: Path, stem: str) -> None:
    metadata = {
        "Title": stem,
        "Author": "SagePilot",
        "Subject": "serve_0726 GPU-time distribution draft",
        "Keywords": "GPU time distribution, Poisson serving",
        "Creator": Path(__file__).name,
        "Producer": "Matplotlib",
        "CreationDate": FIXED_TIMESTAMP,
        "ModDate": FIXED_TIMESTAMP,
    }
    figure.savefig(output_dir / f"{stem}.pdf", metadata=metadata)
    figure.savefig(output_dir / f"{stem}.png", dpi=300)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_root: Path = args.input_root.resolve()
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(input_root)
    (output_dir / "analysis_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    configure_matplotlib()
    plot_gpu_time(payload, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
