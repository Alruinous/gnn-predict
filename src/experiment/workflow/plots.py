from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

Interval = tuple[float, float, float]

BLUE = "#0072B2"
ORANGE = "#D55E00"
DARK_NEUTRAL = "#3D3D3D"
MID_NEUTRAL = "#9A9A9A"
LIGHT_NEUTRAL = "#D9D9D9"

QMSUM_GROUPS = (
    ("LG-Batch", "lg-batch", 2),
    ("WF-FIFO", "wf-fifo", 2),
    ("WF-Cache", "wf-cache", 2),
)
QMSUM_COLORS = (DARK_NEUTRAL, MID_NEUTRAL, ORANGE)
QMSUM_MARKERS = ("o", "s", "D")
QMSUM_HATCHES = ("//", "xx", "\\\\")

MBPP_GROUPS = (
    ("LG-Batch\n3 GPU", "lg-batch", 3),
    ("WF-Cache\n1 GPU", "wf-cache", 1),
    ("WF-Cache\n2 GPU", "wf-cache", 2),
    ("WF-Cache\n3 GPU", "wf-cache", 3),
)
MBPP_COLORS = (DARK_NEUTRAL, BLUE, ORANGE, MID_NEUTRAL)
MBPP_MARKERS = ("o", "s", "^", "D")


def generate_workflow_plots(
    group_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> tuple[Path, ...]:
    artifacts: list[Path] = []
    builders = (
        ("qmsum_strategy_performance", _qmsum_strategy_performance),
        ("qmsum_pipeline_bubble", _qmsum_pipeline_bubble),
        ("mbpp_resource_tradeoff", _mbpp_resource_tradeoff),
        ("mbpp_gpu_time_composition", _mbpp_gpu_time_composition),
    )
    for stem, builder in builders:
        figure = builder(group_rows)
        if figure is None:
            continue
        artifacts.extend(_save_figure(figure, output_dir, stem))
    return tuple(artifacts)


def _qmsum_strategy_performance(
    rows: Sequence[Mapping[str, object]],
) -> Figure | None:
    groups = _select_groups(rows, "qmsum", QMSUM_GROUPS)
    if groups is None:
        return None
    makespan = _intervals(groups, ("makespan_sec",))
    latency = _intervals(groups, ("session_latency_sec", "p95"))
    if makespan is None or latency is None:
        return None
    figure = _figure((8.0, 3.5))
    left = figure.add_subplot(1, 2, 1)
    right = figure.add_subplot(1, 2, 2)
    labels = tuple(group[0] for group in QMSUM_GROUPS)
    _point_panel(left, labels, makespan, QMSUM_COLORS, QMSUM_MARKERS)
    _point_panel(right, labels, latency, QMSUM_COLORS, QMSUM_MARKERS)
    left.set_ylabel("Makespan (s)")
    right.set_ylabel("P95 session latency (s)")
    _heading(
        figure,
        "QMSum burst performance",
        "n=5 trials per group; points and error bars show mean and 95% CI",
    )
    _layout(figure, bottom=0.25)
    return figure


def _qmsum_pipeline_bubble(
    rows: Sequence[Mapping[str, object]],
) -> Figure | None:
    groups = _select_groups(rows, "qmsum", QMSUM_GROUPS)
    if groups is None:
        return None
    intervals = _intervals(groups, ("pipeline_bubble_ratio",))
    if intervals is None:
        return None
    figure = _figure((5.2, 3.5))
    axis = figure.add_subplot(1, 1, 1)
    labels = tuple(group[0] for group in QMSUM_GROUPS)
    _bar_panel(axis, labels, intervals, QMSUM_COLORS, QMSUM_HATCHES)
    axis.set_ylabel("Pipeline bubble ratio")
    _heading(
        figure,
        "QMSum burst pipeline bubble",
        "n=5 trials per group; bars and error bars show mean and 95% CI",
    )
    _layout(figure, bottom=0.25)
    return figure


def _mbpp_resource_tradeoff(
    rows: Sequence[Mapping[str, object]],
) -> Figure | None:
    groups = _select_groups(rows, "mbpp", MBPP_GROUPS)
    if groups is None:
        return None
    throughput = _intervals(groups, ("sessions_per_min",))
    resident = _intervals(groups, ("resident_gpu_seconds",))
    if throughput is None or resident is None:
        return None
    figure = _figure((8.0, 3.5))
    left = figure.add_subplot(1, 2, 1)
    right = figure.add_subplot(1, 2, 2)
    labels = tuple(group[0] for group in MBPP_GROUPS)
    _point_panel(left, labels, throughput, MBPP_COLORS, MBPP_MARKERS)
    _point_panel(right, labels, resident, MBPP_COLORS, MBPP_MARKERS)
    left.set_ylabel("Throughput (sessions/min)")
    right.set_ylabel("Resident GPU time (GPU·s)")
    _heading(
        figure,
        "MBPP burst throughput and residency",
        "n=5 trials per group; points and error bars show mean and 95% CI",
    )
    _layout(figure, bottom=0.29)
    return figure


def _mbpp_gpu_time_composition(
    rows: Sequence[Mapping[str, object]],
) -> Figure | None:
    groups = _select_groups(rows, "mbpp", MBPP_GROUPS)
    if groups is None:
        return None
    components = (
        ("Loading", ("loading_gpu_seconds",), BLUE, "//"),
        ("Active", ("active_gpu_seconds",), ORANGE, "xx"),
        ("Idle resident", ("idle_resident_gpu_seconds",), LIGHT_NEUTRAL, ".."),
    )
    component_intervals = tuple(
        _intervals(groups, path) for _, path, _, _ in components
    )
    if any(intervals is None for intervals in component_intervals):
        return None
    figure = _figure((6.7, 3.8))
    axis = figure.add_subplot(1, 1, 1)
    labels = tuple(group[0] for group in MBPP_GROUPS)
    bottoms = [0.0] * len(labels)
    positions = tuple(range(len(labels)))
    for component, intervals in zip(components, component_intervals, strict=True):
        name, _, color, hatch = component
        assert intervals is not None
        means = [interval[0] for interval in intervals]
        bars = axis.bar(
            positions,
            means,
            bottom=bottoms,
            color=color,
            edgecolor=DARK_NEUTRAL,
            linewidth=0.8,
            hatch=hatch,
            label=name,
            zorder=2,
        )
        assert len(bars) == len(labels)
        for index, interval in enumerate(intervals):
            mean, low, high = interval
            axis.errorbar(
                index,
                bottoms[index] + mean,
                yerr=((mean - low,), (high - mean,)),
                fmt="none",
                ecolor=DARK_NEUTRAL,
                elinewidth=0.9,
                capsize=3,
                zorder=4,
            )
            bottoms[index] += mean
    axis.set_xticks(positions, labels)
    axis.set_ylabel("GPU time (GPU·s)")
    axis.legend(frameon=False, ncols=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    _style_axis(axis)
    _heading(
        figure,
        "MBPP burst GPU time composition",
        "n=5 trials per group; stacked means with component 95% CI",
    )
    _layout(figure, bottom=0.24)
    return figure


def _select_groups(
    rows: Sequence[Mapping[str, object]],
    scenario: str,
    specs: Sequence[tuple[str, str, int]],
) -> tuple[Mapping[str, object], ...] | None:
    groups = []
    for _, strategy, gpu_count in specs:
        dimensions = {
            "scenario": scenario,
            "strategy": strategy,
            "gpu_count": gpu_count,
            "workload": "burst",
            "max_num_seqs": 3,
            "queue_capacity": 16,
            "load_percent": None,
        }
        matches = [
            row
            for row in rows
            if all(
                name in row and row[name] == value for name, value in dimensions.items()
            )
        ]
        if len(matches) != 1 or matches[0].get("trial_count") != 5:
            return None
        groups.append(matches[0])
    return tuple(groups)


def _intervals(
    groups: Sequence[Mapping[str, object]],
    path: Sequence[str],
) -> tuple[Interval, ...] | None:
    intervals = tuple(_metric_interval(group, path) for group in groups)
    if any(interval is None for interval in intervals):
        return None
    return tuple(interval for interval in intervals if interval is not None)


def _metric_interval(
    group: Mapping[str, object],
    path: Sequence[str],
) -> Interval | None:
    value: object = group.get("metrics")
    for name in path:
        if not isinstance(value, Mapping) or name not in value:
            return None
        value = cast(Mapping[str, object], value)[name]
    if not isinstance(value, Mapping):
        return None
    interval = cast(Mapping[str, object], value)
    mean = _finite_number(interval.get("mean"))
    low = _finite_number(interval.get("ci95_low"))
    high = _finite_number(interval.get("ci95_high"))
    if mean is None or low is None or high is None:
        return None
    if mean < 0 or low > mean or high < mean:
        return None
    return mean, low, high


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _figure(size: tuple[float, float]) -> Figure:
    figure = Figure(figsize=size, facecolor="white")
    FigureCanvasAgg(figure)
    return figure


def _point_panel(
    axis: Axes,
    labels: Sequence[str],
    intervals: Sequence[Interval],
    colors: Sequence[str],
    markers: Sequence[str],
) -> None:
    positions = tuple(range(len(labels)))
    for index, (interval, color, marker) in enumerate(
        zip(intervals, colors, markers, strict=True)
    ):
        mean, low, high = interval
        axis.errorbar(
            index,
            mean,
            yerr=((mean - low,), (high - mean,)),
            fmt=marker,
            color=color,
            markeredgecolor=DARK_NEUTRAL,
            markeredgewidth=0.7,
            markersize=7,
            linestyle="none",
            elinewidth=1.2,
            capsize=4,
            zorder=3,
        )
    axis.set_xticks(positions, labels, rotation=18, ha="right")
    _style_axis(axis)


def _bar_panel(
    axis: Axes,
    labels: Sequence[str],
    intervals: Sequence[Interval],
    colors: Sequence[str],
    hatches: Sequence[str],
) -> None:
    positions = tuple(range(len(labels)))
    means = [interval[0] for interval in intervals]
    bars = axis.bar(
        positions,
        means,
        color=colors,
        edgecolor=DARK_NEUTRAL,
        linewidth=0.8,
        zorder=2,
    )
    for bar, hatch in zip(bars, hatches, strict=True):
        bar.set_hatch(hatch)
    axis.errorbar(
        positions,
        means,
        yerr=(
            [
                mean - interval[1]
                for mean, interval in zip(means, intervals, strict=True)
            ],
            [
                interval[2] - mean
                for mean, interval in zip(means, intervals, strict=True)
            ],
        ),
        fmt="none",
        ecolor=DARK_NEUTRAL,
        elinewidth=1.0,
        capsize=4,
        zorder=3,
    )
    axis.set_xticks(positions, labels, rotation=18, ha="right")
    _style_axis(axis)


def _style_axis(axis: Axes) -> None:
    axis.set_facecolor("white")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color=LIGHT_NEUTRAL, linewidth=0.7, linestyle=(0, (2, 2)))
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=8.5)
    axis.yaxis.label.set_fontsize(9.5)


def _heading(figure: Figure, title: str, subtitle: str) -> None:
    figure.suptitle(title, fontsize=12, fontweight="semibold", y=0.98)
    figure.text(0.5, 0.91, subtitle, ha="center", va="top", fontsize=8.5)


def _layout(figure: Figure, *, bottom: float) -> None:
    figure.subplots_adjust(left=0.11, right=0.98, bottom=bottom, top=0.78, wspace=0.36)


def _save_figure(
    figure: Figure,
    output_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    figure.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="white")
    figure.savefig(
        png_path,
        format="png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.clear()
    return pdf_path, png_path
