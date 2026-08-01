"""Plot the standalone GPU-time distribution from eval_20260727 Burst traces."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plot_eval_20260728_drafts as evaluation
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "output" / "eval_20260727_gpu_time_draft"

TEXT = "#202124"
MUTED = "#626A73"
GRID = "#D9DEE3"
STATE_SPECS = (
    ("generation_gpu_sec", "Generation", "#3D7F83", "white"),
    ("idle_resident_gpu_sec", "Idle resident", "#DDD4BE", TEXT),
    ("loading_gpu_sec", "Loading", "#D98A52", TEXT),
    ("other_gpu_sec", "Other", "#626A73", "white"),
)
FIGURE_WIDTH_IN = 3.33
FONT_SIZE = 8.5
ANNOTATION_FONT_SIZE = 8.0
FIXED_TIMESTAMP = datetime(2026, 8, 1, tzinfo=UTC)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def build_payload(
    burst: Mapping[str, Sequence[evaluation.BurstRun]],
) -> dict[str, object]:
    policies: dict[str, object] = {}
    for label in evaluation.EVALUATION_ORDER:
        runs = burst[label]
        policies[label] = {
            "run_ids": [run.run_id for run in runs],
            "runs": [
                {
                    "run_id": run.run_id,
                    "run_duration_sec": run.run_duration_sec,
                    "generation_gpu_sec": run.generation_gpu_sec,
                    "idle_resident_gpu_sec": run.idle_resident_gpu_sec,
                    "loading_gpu_sec": run.loading_gpu_sec,
                    "other_gpu_sec": run.other_gpu_sec,
                }
                for run in runs
            ],
            "gpu_time_distribution_pct": evaluation.gpu_time_distribution(runs),
        }
    return {
        "schema": 1,
        "description": "GPU-time distribution from eval_20260727 Burst traces",
        "source": str(evaluation.EVAL_ROOT.relative_to(ROOT)),
        "experiment": {
            "arrival": "burst",
            "workflow_count": 3,
            "session_count": evaluation.EXPECTED_BURST_SESSIONS,
            "gpu_count": evaluation.EXPECTED_GPU_COUNT,
            "hardware": "1xA100 + 2xV100",
            "runs_per_policy": 2,
        },
        "policies": policies,
    }


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.titleweight": "bold",
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


def plot_gpu_time(
    distributions: Mapping[str, Mapping[str, float]], output_dir: Path
) -> None:
    figure, axes = plt.subplots(figsize=(FIGURE_WIDTH_IN, 2.70))
    order = evaluation.EVALUATION_ORDER
    positions = list(range(len(order)))
    bottoms = [0.0] * len(order)
    for field, component, color, text_color in STATE_SPECS:
        values = [distributions[label][field] for label in order]
        bars = axes.bar(
            positions,
            values,
            bottom=bottoms,
            width=0.62,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            label=component,
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
                    color=text_color,
                    fontsize=ANNOTATION_FONT_SIZE,
                    fontweight="bold",
                )
        bottoms = [
            bottom + value for bottom, value in zip(bottoms, values, strict=True)
        ]

    axes.set_xticks(positions, order, rotation=15, ha="right")
    for label in axes.get_xticklabels():
        label.set_color(TEXT)
        label.set_fontweight("normal")
    axes.set_ylim(0.0, 100.0)
    axes.set_yticks(
        (0.0, 25.0, 50.0, 75.0, 100.0),
        ("0%", "25%", "50%", "75%", "100%"),
    )
    axes.set_ylabel("Fraction of GPU time")
    axes.grid(axis="y", zorder=0)
    axes.set_axisbelow(True)
    axes.tick_params(axis="x", length=0, pad=2.0)
    axes.tick_params(axis="y", length=2.0, pad=1.5)
    axes.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.3,
    )
    figure.subplots_adjust(left=0.18, right=0.98, bottom=0.22, top=0.78)
    save_figure(figure, output_dir, "fig_gpu_time_distribution_eval_20260727_draft")


def save_figure(figure: Figure, output_dir: Path, stem: str) -> None:
    metadata = {
        "Title": stem,
        "Author": "SagePilot",
        "Subject": "eval_20260727 GPU-time distribution draft",
        "Keywords": "GPU time distribution, Burst serving",
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
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    burst = evaluation.load_burst_results()
    payload = build_payload(burst)
    (output_dir / "analysis_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    distributions = {
        label: evaluation.gpu_time_distribution(burst[label])
        for label in evaluation.EVALUATION_ORDER
    }
    configure_matplotlib()
    plot_gpu_time(distributions, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
