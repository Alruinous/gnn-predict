from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from scripts.motivation.common import stats

BAR_COLORS = {
    "chunk": "#4C78A8",
    "chunk_0": "#4C78A8",
    "chunk_1": "#72B7B2",
    "chunk_2": "#F58518",
    "merge": "#E45756",
    "idle_sync": "#D3D3D3",
    "idle_barrier": "#A0A0A0",
    "coder": "#4C78A8",
    "repair": "#F58518",
    "tester": "#54A24B",
    "resident": "#D3D3D3",
}
QMSUM_BASELINE = "#4C78A8"
QMSUM_ORACLE = "#54A24B"
MBPP_RESIDENT = "#E45756"
MBPP_ACTIVE = "#72B7B2"


def write_plots(
    *,
    qmsum_summary: dict[str, Any],
    mbpp_summary: dict[str, Any],
    image_dir: Path,
) -> dict[str, Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    paths = {
        "qmsum_gantt": image_dir / "motivation_20260704_qmsum_gantt.png",
        "qmsum_bubble_cdf": image_dir / "motivation_20260704_qmsum_bubble_cdf.png",
        "qmsum_speedup": image_dir / "motivation_20260704_qmsum_speedup.png",
        "mbpp_gap": image_dir / "motivation_20260704_mbpp_gap.png",
    }
    plot_qmsum_gantt(qmsum_summary, paths["qmsum_gantt"])
    plot_qmsum_bubble_cdf(qmsum_summary, paths["qmsum_bubble_cdf"])
    plot_qmsum_speedup(qmsum_summary, paths["qmsum_speedup"])
    plot_mbpp_gap(mbpp_summary, paths["mbpp_gap"])
    return paths


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def plot_qmsum_gantt(summary: dict[str, Any], path: Path) -> None:
    samples = summary["samples"]
    mean_chunk_durs = [
        sum(s["chunk_durations"][i] for s in samples) / len(samples) for i in range(3)
    ]
    mean_merge_dur = summary["merge_duration_sec"]["mean"]
    chunk_ends = mean_chunk_durs
    merge_start = max(chunk_ends)

    fig, ax = plt.subplots(figsize=(5.8, 2.0))
    y_pos = [3, 2, 1, 0]
    colors = [BAR_COLORS["chunk_0"], BAR_COLORS["chunk_1"], BAR_COLORS["chunk_2"]]

    total = merge_start + mean_merge_dur
    for i in range(3):
        ax.barh(y_pos[i], chunk_ends[i], left=0, height=0.5, color=colors[i], zorder=3)
        sync_i = max(0.0, merge_start - chunk_ends[i])
        if sync_i > 0.5:
            ax.barh(
                y_pos[i],
                sync_i,
                left=chunk_ends[i],
                height=0.5,
                color=BAR_COLORS["idle_sync"],
                alpha=0.35,
                zorder=2,
                label="Peer sync idle" if i == 0 else None,
            )
        ax.barh(
            y_pos[i],
            mean_merge_dur,
            left=merge_start,
            height=0.5,
            color=BAR_COLORS["idle_barrier"],
            alpha=0.35,
            zorder=2,
            label="Session barrier idle" if i == 0 else None,
        )

    ax.barh(
        y_pos[3],
        mean_merge_dur,
        left=merge_start,
        height=0.5,
        color=BAR_COLORS["merge"],
        zorder=3,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(["Chunk 0", "Chunk 1", "Chunk 2", "Merge"], fontsize=7)
    ax.set_xlim(0, total * 1.02)
    ax.set_xlabel("Average time (s)", fontsize=8, labelpad=2)
    ax.set_title("QMSum 3-way fan-in workflow node timing (n=60)", fontsize=9, pad=4)

    for edge in ["left", "top", "right", "bottom"]:
        ax.spines[edge].set_visible(False)
    ax.tick_params(axis="x", length=2, pad=1)
    ax.tick_params(axis="y", length=0, pad=1)

    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncols=4,
        frameon=False,
        fontsize=7.5,
    )

    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_qmsum_bubble_cdf(summary: dict[str, Any], path: Path) -> None:
    samples = summary["samples"]
    ratios = np.sort([s["pipeline_bubble_ratio"] for s in samples])
    n = len(ratios)
    y = np.arange(1, n + 1) / n

    r_mean = summary["pipeline_bubble_ratio"]["mean"]
    r_p95 = summary["pipeline_bubble_ratio"]["p95"]

    fig, ax1 = plt.subplots(figsize=(3.6, 2.4))
    ax1.plot(ratios, y, color="#4C78A8", linewidth=1.2, zorder=3)
    ax1.fill_between(ratios, y, alpha=0.08, color="#4C78A8")

    ax1.axvline(r_mean, color="#E45756", linewidth=0.8, linestyle="--", zorder=2)
    ax1.axvline(r_p95, color="#E45756", linewidth=0.8, linestyle="--", zorder=2)

    ax1.annotate(
        f"mean={r_mean:.3f}",
        xy=(r_mean, 0.5),
        xytext=(r_mean + 0.04, 0.45),
        fontsize=7,
        color="#E45756",
        ha="left",
        arrowprops=dict(arrowstyle="->", color="#E45756", lw=0.6),
    )

    ax1.annotate(
        f"p95={r_p95:.3f}",
        xy=(r_p95, 0.95),
        xytext=(r_p95 - 0.04, 0.82),
        fontsize=7,
        color="#E45756",
        ha="right",
        arrowprops=dict(arrowstyle="->", color="#E45756", lw=0.6),
    )

    ax1.set_xlabel("Pipeline bubble ratio")
    ax1.set_ylabel("CDF")
    ax1.set_title("Bubble ratio distribution (n=60)")

    speedup = summary["speedup_upper_bound"]
    ax1.text(
        0.98,
        0.08,
        f"Speedup bound: {speedup:.3f}x",
        transform=ax1.transAxes,
        fontsize=7.5,
        ha="right",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#F5F5F5",
            edgecolor="#CCCCCC",
            linewidth=0.5,
        ),
    )

    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_qmsum_speedup(summary: dict[str, Any], path: Path) -> None:
    baseline = summary["baseline_total_sec"]
    oracle = summary["oracle_total_sec"]
    speedup = summary["speedup_upper_bound"]

    fig, ax = plt.subplots(figsize=(2.8, 2.4))
    x_pos = [0, 1]
    bars = ax.bar(
        x_pos,
        [baseline, oracle],
        width=0.5,
        color=[QMSUM_BASELINE, QMSUM_ORACLE],
        zorder=3,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(["LangGraph\nnative", "Pipeline\noracle"])
    ax.set_ylabel("Total generation time (s)")

    def fmt(val):
        return f"{val:.0f}s" if val > 100 else f"{val:.1f}s"

    for bar, val in zip(bars, [baseline, oracle], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 80,
            fmt(val),
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    ax.plot(
        [0, 1],
        [baseline, oracle],
        color="#555555",
        linewidth=0.8,
        linestyle="--",
        marker="o",
        markersize=4,
        zorder=1,
    )

    ax.set_title("Trace-replay speedup bound")

    mid_x = (x_pos[0] + x_pos[1]) / 2
    mid_y = (baseline + oracle) / 2
    ax.annotate(
        f"{speedup:.3f}x",
        xy=(mid_x, mid_y),
        xytext=(mid_x, mid_y + 200),
        ha="center",
        fontsize=9,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
        color="#333333",
    )

    ax.set_ylim(0, baseline * 1.18)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_mbpp_gap(summary: dict[str, Any], path: Path) -> None:
    resident = summary["resident_model_seconds"]["total"]
    active = summary["active_model_seconds"]["total"]
    gap = summary["resource_gap"]
    samples = summary["samples"]

    fig, ax = plt.subplots(figsize=(2.8, 2.4))
    x_pos = [0, 1]
    bars = ax.bar(
        x_pos,
        [resident, active],
        width=0.5,
        color=[MBPP_RESIDENT, MBPP_ACTIVE],
        zorder=3,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Static\nresidency", "Active\nfrontier"])
    ax.set_ylabel("Total GPU-seconds")

    def fmt(val):
        if val > 1000:
            return f"{val / 1000:.1f}K"
        return f"{val:.0f}"

    for bar, val in zip(bars, [resident, active], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 100,
            fmt(val),
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    ax.plot(
        [0, 1],
        [resident, active],
        color="#555555",
        linewidth=0.8,
        linestyle="--",
        marker="o",
        markersize=4,
        zorder=1,
    )

    ax.set_title("Model resource waste")

    mid_x = (x_pos[0] + x_pos[1]) / 2
    gap_y = (resident + active) / 2
    ax.annotate(
        f"{gap:.3f}x",
        xy=(mid_x, gap_y),
        xytext=(mid_x, gap_y + 600),
        ha="center",
        fontsize=9,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
        color="#333333",
    )

    sample_gaps = [s["resource_gap"] for s in samples]
    gap_stats_data = stats(sample_gaps)
    ax.text(
        0.98,
        0.10,
        f"gap mean={gap_stats_data['mean']:.2f}x\n"
        f"frontier limit={summary['frontier_gap']:.1f}x",
        transform=ax.transAxes,
        fontsize=6.5,
        ha="right",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#F5F5F5",
            edgecolor="#CCCCCC",
            linewidth=0.5,
        ),
    )

    ax.set_ylim(0, resident * 1.18)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
