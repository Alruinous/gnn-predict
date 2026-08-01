"""Generate paper-style layouts from the selected a1v3 replay results."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plot_serve_0728_bar_drafts as source
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT / "output" / "serve_0728_bar_drafts" / "analysis_metrics.json"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "serve_0728_layout_drafts"

SYSTEM_PERFORMANCE_PANELS = (
    ("overall", "", "Overall"),
    ("model", "Qwen3-4B", "4B"),
    ("model", "Qwen3-8B", "8B"),
    ("model", "Qwen3-14B", "14B"),
)
MODEL_PANELS = (
    ("model", "Qwen3-0.6B", "0.6B"),
    ("model", "Qwen3-1.7B", "1.7B"),
    ("model", "Qwen3-4B", "4B"),
    ("model", "Qwen3-8B", "8B"),
    ("model", "Qwen3-14B", "14B"),
)
ABLATION_PANELS = (
    ("overall", "", "Overall"),
    ("workflow", "chain_qmsum", "QMSum"),
    ("model", "Qwen3-4B", "4B"),
    ("model", "Qwen3-8B", "8B"),
)

TEXT = source.TEXT
DOUBLE_COLUMN_WIDTH_IN = source.DOUBLE_COLUMN_WIDTH_IN
SINGLE_COLUMN_WIDTH_IN = source.SINGLE_COLUMN_WIDTH_IN
BASE_FONT_SIZE = 8.5
PANEL_FONT_SIZE = 9.0
ANNOTATION_FONT_SIZE = 8.0
FONT_FAMILY = "STIXGeneral"
SYSTEM_HATCHES = {
    "parrot": "",
    "kairos": "//",
    "analytical": "..",
    "gbdt": "\\\\",
    "sagepilot": "xx",
}
P50_HATCH = "////"
P95_HATCH = "////"
BURST_ARRIVAL = (("burst", "Burst"),)
SYSTEM_PERFORMANCE_ARRIVALS = (
    ("burst", "Burst"),
    ("poisson_r050", "Poisson 0.050"),
    ("poisson_r025", "Poisson 0.025"),
    ("poisson_r0125", "Poisson 0.0125"),
)
ORACLE_ARM = "oracle_lb"
SYSTEM_PERFORMANCE_ARMS = (*source.SYSTEM_ARMS, ORACLE_ARM)
SYSTEM_PERFORMANCE_COLORS = {
    **source.SYSTEM_COLORS,
    ORACLE_ARM: "#30343B",
}
FIXED_TIMESTAMP = datetime(2026, 7, 31, tzinfo=UTC)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def save_figure(figure: Figure, output_dir: Path, stem: str) -> None:
    metadata = {
        "Title": stem,
        "Author": "SagePilot",
        "Subject": "PPoPP evaluation layout draft",
        "Keywords": "PPoPP evaluation",
        "Creator": Path(__file__).name,
        "Producer": "Matplotlib",
        "CreationDate": FIXED_TIMESTAMP,
        "ModDate": FIXED_TIMESTAMP,
    }
    figure.savefig(output_dir / f"{stem}.pdf", metadata=metadata)
    figure.savefig(output_dir / f"{stem}.png", dpi=300)
    plt.close(figure)


def configure_matplotlib() -> None:
    source.configure_matplotlib()
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": BASE_FONT_SIZE,
            "axes.titlesize": PANEL_FONT_SIZE,
            "axes.titleweight": "bold",
            "legend.fontsize": BASE_FONT_SIZE,
            "xtick.labelsize": ANNOTATION_FONT_SIZE,
            "ytick.labelsize": ANNOTATION_FONT_SIZE,
            "hatch.linewidth": 0.35,
        }
    )


def plot_bars(
    axes: Axes,
    values: Sequence[float],
    arms: Sequence[str],
    colors: Mapping[str, str],
    hatches: Mapping[str, str],
    *,
    annotation_rotation: float = 0.0,
) -> BarContainer:
    positions = list(range(len(arms)))
    bars = axes.bar(
        positions,
        values,
        width=0.68,
        color=[colors[arm] for arm in arms],
        edgecolor=TEXT,
        linewidth=0.35,
        zorder=2,
    )
    for bar, arm in zip(bars, arms, strict=True):
        bar.set_hatch(hatches.get(arm, ""))
    axes.set_xticks([])
    axes.set_xlim(-0.6, len(arms) - 0.4)
    axes.set_ylim(0.0, max(values) * 1.17)
    source.style_panel(axes)
    annotate_bars(axes, bars, values, rotation=annotation_rotation)
    return bars


def annotate_bars(
    axes: Axes,
    bars: BarContainer,
    values: Sequence[float],
    *,
    rotation: float = 0.0,
    x_offset_points: float = 0.0,
) -> None:
    for bar, value in zip(bars, values, strict=True):
        axes.annotate(
            f"{value:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(x_offset_points, 2.0),
            textcoords="offset points",
            ha="left" if rotation else "center",
            va="bottom",
            rotation=rotation,
            rotation_mode="anchor",
            fontsize=ANNOTATION_FONT_SIZE,
            clip_on=False,
        )


def plot_metric_facets(
    payload: Mapping[str, object],
    output_dir: Path,
    *,
    panels: Sequence[tuple[str, str, str]],
    arms: Sequence[str],
    colors: Mapping[str, str],
    hatches: Mapping[str, str],
    metric: str,
    ylabel: str,
    stem: str,
) -> None:
    figure, axes_array = plt.subplots(
        1,
        len(panels),
        figsize=(DOUBLE_COLUMN_WIDTH_IN, 2.05),
        squeeze=False,
    )
    for (kind, key, title), axes in zip(panels, axes_array[0], strict=True):
        values = [
            source.aggregate(payload, arm, source.metric_path(kind, key, metric))
            for arm in arms
        ]
        plot_bars(
            axes,
            values,
            arms,
            colors,
            hatches,
        )
        axes.text(
            0.5,
            -0.16,
            title,
            transform=axes.transAxes,
            ha="center",
            va="top",
            fontsize=PANEL_FONT_SIZE,
            fontweight="bold",
        )
    figure.legend(
        handles=source.method_handles(arms, colors, hatches),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(arms),
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.35,
    )
    figure.text(
        0.012,
        0.48,
        ylabel,
        rotation=90,
        ha="left",
        va="center",
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.22,
        top=0.83,
        wspace=0.30,
    )
    save_figure(figure, output_dir, stem)


def plot_performance_grid(
    payload: Mapping[str, object],
    output_dir: Path,
    *,
    panels: Sequence[tuple[str, str, str]],
    arms: Sequence[str],
    colors: Mapping[str, str],
    arrivals: Sequence[tuple[str, str]],
    latency_ylabel: str,
    stem: str,
) -> None:
    row_specs = [
        (arrival, arrival_label, "makespan") for arrival, arrival_label in arrivals
    ]
    row_specs.extend(
        (arrival, arrival_label, "latency") for arrival, arrival_label in arrivals
    )
    multi_arrival = len(arrivals) > 1
    figure, axes_array = plt.subplots(
        len(row_specs),
        len(panels),
        figsize=(DOUBLE_COLUMN_WIDTH_IN, 7.0 if multi_arrival else 3.60),
        squeeze=False,
    )
    positions = list(range(len(arms)))
    bar_width = 0.28
    offset = 0.19
    for row_index, (arrival, arrival_label, metric) in enumerate(row_specs):
        for (kind, key, title), axes in zip(panels, axes_array[row_index], strict=True):
            if metric == "makespan":
                values = [
                    source.aggregate(
                        payload,
                        arm,
                        source.metric_path(kind, key, "makespan"),
                        arrival=arrival,
                    )
                    for arm in arms
                ]
                plot_bars(axes, values, arms, colors, {})
            else:
                p50_values = [
                    source.aggregate(
                        payload,
                        arm,
                        source.metric_path(kind, key, "p50"),
                        arrival=arrival,
                    )
                    for arm in arms
                ]
                p95_values = [
                    source.aggregate(
                        payload,
                        arm,
                        source.metric_path(kind, key, "p95"),
                        arrival=arrival,
                    )
                    for arm in arms
                ]
                p95_bars = axes.bar(
                    [position - offset for position in positions],
                    p95_values,
                    width=bar_width,
                    color=[colors[arm] for arm in arms],
                    edgecolor=TEXT,
                    linewidth=0.35,
                    zorder=2,
                )
                p50_bars = axes.bar(
                    [position + offset for position in positions],
                    p50_values,
                    width=bar_width,
                    color="white",
                    linewidth=0.75,
                    zorder=2,
                )
                for bar, arm in zip(p50_bars, arms, strict=True):
                    bar.set_edgecolor(colors[arm])
                    bar.set_hatch(P50_HATCH)
                axes.set_xticks([])
                axes.set_xlim(-0.55, len(arms) - 0.45)
                axes.set_ylim(0.0, max((*p50_values, *p95_values)) * 1.20)
                source.style_panel(axes)
                annotate_bars(
                    axes,
                    p95_bars,
                    p95_values,
                    rotation=60.0,
                    x_offset_points=1.0,
                )
                annotate_bars(
                    axes,
                    p50_bars,
                    p50_values,
                    rotation=60.0,
                    x_offset_points=1.0,
                )
            if multi_arrival:
                axes.set_title(f"{arrival_label} + {title}", loc="center", pad=3.0)
            elif row_index == 0:
                axes.set_title(title, pad=3.0)
    method_handles = source.method_handles(arms, colors, {})
    latency_handles = [
        Patch(
            facecolor=source.MUTED,
            edgecolor=TEXT,
            linewidth=0.35,
            label="p95",
        ),
        Patch(
            facecolor="white",
            edgecolor=TEXT,
            linewidth=0.75,
            hatch=P50_HATCH,
            label="p50",
        ),
    ]
    figure.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99 if multi_arrival else 0.985),
        ncol=len(method_handles),
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.35,
    )
    figure.legend(
        handles=latency_handles,
        loc="upper center" if multi_arrival else "center",
        bbox_to_anchor=(0.5, 0.955 if multi_arrival else 0.47),
        ncol=len(latency_handles),
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.3,
        borderaxespad=0.0,
    )
    if multi_arrival:
        figure.subplots_adjust(
            left=0.075,
            right=0.995,
            bottom=0.045,
            top=0.91,
            hspace=0.48,
            wspace=0.22,
        )
        for row_index, (_, _, metric) in enumerate(row_specs):
            bounds = axes_array[row_index, 0].get_position()
            figure.text(
                0.012,
                (bounds.y0 + bounds.y1) / 2.0,
                "Makespan (s)" if metric == "makespan" else latency_ylabel,
                rotation=90,
                ha="left",
                va="center",
            )
    else:
        figure.text(
            0.012,
            0.63,
            "Makespan (s)",
            rotation=90,
            ha="left",
            va="center",
        )
        figure.text(
            0.012,
            0.25,
            latency_ylabel,
            rotation=90,
            ha="left",
            va="center",
        )
        figure.subplots_adjust(
            left=0.065,
            right=0.995,
            bottom=0.08,
            top=0.86,
            hspace=0.32,
            wspace=0.22,
        )
    save_figure(figure, output_dir, stem)


def plot_system_performance_grid(
    payload: Mapping[str, object], output_dir: Path
) -> None:
    arms = SYSTEM_PERFORMANCE_ARMS
    colors = SYSTEM_PERFORMANCE_COLORS
    figure, axes_array = plt.subplots(
        len(SYSTEM_PERFORMANCE_ARRIVALS),
        len(SYSTEM_PERFORMANCE_PANELS),
        figsize=(DOUBLE_COLUMN_WIDTH_IN, 7.0),
        squeeze=False,
    )
    positions = list(range(len(arms)))
    bar_width = 0.25
    offset = 0.16
    for row_index, (arrival, arrival_label) in enumerate(SYSTEM_PERFORMANCE_ARRIVALS):
        for (kind, key, title), axes in zip(
            SYSTEM_PERFORMANCE_PANELS,
            axes_array[row_index],
            strict=True,
        ):
            makespan_values = [
                performance_aggregate(
                    payload,
                    arm,
                    source.metric_path(kind, key, "makespan"),
                    arrival=arrival,
                )
                for arm in arms
            ]
            p95_values = [
                performance_aggregate(
                    payload,
                    arm,
                    source.metric_path(kind, key, "p95"),
                    arrival=arrival,
                )
                for arm in arms
            ]
            makespan_bars = axes.bar(
                [position - offset for position in positions],
                makespan_values,
                width=bar_width,
                color=[colors[arm] for arm in arms],
                edgecolor=TEXT,
                linewidth=0.35,
                zorder=2,
            )
            p95_bars = axes.bar(
                [position + offset for position in positions],
                p95_values,
                width=bar_width,
                color="white",
                linewidth=0.75,
                zorder=2,
            )
            for bar, arm in zip(p95_bars, arms, strict=True):
                bar.set_edgecolor(colors[arm])
                bar.set_hatch(P95_HATCH)
            axes.set_xticks([])
            axes.set_xlim(-0.55, len(arms) - 0.45)
            axes.set_ylim(0.0, max((*makespan_values, *p95_values)) * 1.25)
            source.style_panel(axes)
            annotate_bars(
                axes,
                makespan_bars,
                makespan_values,
                rotation=60.0,
                x_offset_points=1.0,
            )
            annotate_bars(
                axes,
                p95_bars,
                p95_values,
                rotation=60.0,
                x_offset_points=1.0,
            )
            axes.set_title(f"{arrival_label} + {title}", loc="center", pad=3.0)

    figure.legend(
        handles=performance_method_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(arms),
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.35,
    )
    figure.legend(
        handles=[
            Patch(
                facecolor=source.MUTED,
                edgecolor=TEXT,
                linewidth=0.35,
                label="Makespan",
            ),
            Patch(
                facecolor="white",
                edgecolor=TEXT,
                linewidth=0.75,
                hatch=P95_HATCH,
                label="p95",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.3,
        borderaxespad=0.0,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.045,
        top=0.91,
        hspace=0.48,
        wspace=0.22,
    )
    for row_index in range(len(SYSTEM_PERFORMANCE_ARRIVALS)):
        bounds = axes_array[row_index, 0].get_position()
        figure.text(
            0.012,
            (bounds.y0 + bounds.y1) / 2.0,
            "Completion time (s)",
            rotation=90,
            ha="left",
            va="center",
        )
    save_figure(figure, output_dir, "fig_system_performance_layout_draft")


def performance_aggregate(
    payload: Mapping[str, object],
    arm: str,
    path: Sequence[str],
    *,
    arrival: str,
) -> float:
    if arm == ORACLE_ARM:
        return source.oracle_aggregate(payload, path, arrival=arrival)
    return source.aggregate(payload, arm, path, arrival=arrival)


def performance_method_handles() -> list[Patch]:
    labels = {
        **source.replay.ARM_LABELS,
        ORACLE_ARM: "Oracle",
    }
    return [
        Patch(
            facecolor=SYSTEM_PERFORMANCE_COLORS[arm],
            edgecolor=TEXT,
            linewidth=0.35,
            label=labels[arm],
        )
        for arm in SYSTEM_PERFORMANCE_ARMS
    ]


def plot_gpu_time(payload: Mapping[str, object], output_dir: Path) -> None:
    figure, axes = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH_IN, 2.70))
    states = tuple(source.GPU_STATE_COLORS)
    positions = list(range(len(source.SYSTEM_ARMS)))
    bottoms = [0.0] * len(source.SYSTEM_ARMS)
    distributions: dict[str, list[float]] = {}
    for arm in source.SYSTEM_ARMS:
        totals = [source.aggregate(payload, arm, (state,)) for state in states]
        total = sum(totals)
        distributions[arm] = [100.0 * value / total for value in totals]
    for state_index, state in enumerate(states):
        values = [distributions[arm][state_index] for arm in source.SYSTEM_ARMS]
        bars = axes.bar(
            positions,
            values,
            bottom=bottoms,
            width=0.62,
            color=source.GPU_STATE_COLORS[state],
            edgecolor="white",
            linewidth=0.45,
            label=source.GPU_STATE_LABELS[state],
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
                    color=source.GPU_STATE_TEXT[state],
                    fontsize=ANNOTATION_FONT_SIZE,
                    fontweight="bold",
                )
        bottoms = [
            bottom + value for bottom, value in zip(bottoms, values, strict=True)
        ]
    labels = [source.replay.ARM_LABELS[arm] for arm in source.SYSTEM_ARMS]
    axes.set_xticks(positions, labels, rotation=15, ha="right")
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
    save_figure(figure, output_dir, "fig_gpu_time_distribution_layout_draft")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = source.load_metrics(args.data.resolve())
    configure_matplotlib()
    plot_system_performance_grid(payload, output_dir)
    plot_metric_facets(
        payload,
        output_dir,
        panels=MODEL_PANELS,
        arms=source.SYSTEM_ARMS,
        colors=source.SYSTEM_COLORS,
        hatches=SYSTEM_HATCHES,
        metric="makespan",
        ylabel="Completion time (s)",
        stem="fig_completion_models_layout_draft",
    )
    plot_performance_grid(
        payload,
        output_dir,
        panels=ABLATION_PANELS,
        arms=source.ABLATION_ARMS,
        colors=source.ABLATION_COLORS,
        arrivals=BURST_ARRIVAL,
        latency_ylabel="Completion latency (s)",
        stem="fig_ablation_performance_layout_draft",
    )
    plot_gpu_time(payload, output_dir)
    print(f"wrote layout drafts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
