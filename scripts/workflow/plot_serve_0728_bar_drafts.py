"""Generate selected a1v3 bar-chart metrics from the unified-load replay."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import matplotlib
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator

matplotlib.use("Agg")
import compute_serve_0728_oracle as offline_oracle
import matplotlib.pyplot as plt
import plot_serve_0728_counterfactual_drafts as replay
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "output" / "serve_0728_bar_drafts"
METRICS_PATH = OUTPUT_ROOT / "analysis_metrics.json"

SYSTEM_ARMS = replay.SYSTEM_ARMS
ABLATION_ARMS = ("sagepilot", "nofuse", "noprefetch", "noxwf")
SELECTED_ARRIVALS = ("burst", "poisson_r050", "poisson_r025")
SYSTEM_COLORS = {
    "parrot": "#3B549D",
    "kairos": "#3F7F24",
    "analytical": "#7B5AA6",
    "gbdt": "#D88716",
    "sagepilot": "#D73027",
}
ABLATION_COLORS = {
    "sagepilot": "#D73027",
    "nofuse": "#4D4D4D",
    "noprefetch": "#969696",
    "noxwf": "#C9C9C9",
}
ABLATION_HATCHES = {
    "sagepilot": "",
    "nofuse": "////",
    "noprefetch": "\\\\",
    "noxwf": "xx",
}
GPU_STATE_COLORS = {
    "generation_gpu_sec": "#3D7F83",
    "idle_resident_gpu_sec": "#DDD4BE",
    "loading_gpu_sec": "#D98A52",
    "other_gpu_sec": "#626A73",
}
GPU_STATE_LABELS = {
    "generation_gpu_sec": "Generation",
    "idle_resident_gpu_sec": "Idle resident",
    "loading_gpu_sec": "Loading",
    "other_gpu_sec": "Other",
}
GPU_STATE_TEXT = {
    "generation_gpu_sec": "white",
    "idle_resident_gpu_sec": "#202124",
    "loading_gpu_sec": "#202124",
    "other_gpu_sec": "white",
}

PANELS = (
    ("overall", "", "Overall"),
    ("workflow", "moa_gsm8k", "GSM8K"),
    ("workflow", "repair_mbpp", "MBPP"),
    ("workflow", "chain_qmsum", "QMSum"),
    ("model", "Qwen3-0.6B", "0.6B"),
    ("model", "Qwen3-1.7B", "1.7B"),
    ("model", "Qwen3-4B", "4B"),
    ("model", "Qwen3-8B", "8B"),
    ("model", "Qwen3-14B", "14B"),
)

TEXT = "#202124"
MUTED = "#626A73"
GRID = "#D9DEE3"
DOUBLE_COLUMN_WIDTH_IN = 7.0
SINGLE_COLUMN_WIDTH_IN = 3.33
FONT_SIZE = 8.0
FIXED_TIMESTAMP = datetime(2026, 7, 31, tzinfo=UTC)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plot-only", action="store_true")
    mode.add_argument(
        "--refresh-oracle-from-traces",
        action="store_true",
        help="refresh the offline Oracle reference from existing SagePilot traces",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args(argv)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.titleweight": "bold",
            "legend.fontsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
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


def save_figure(figure: Figure, output_dir: Path, stem: str) -> None:
    metadata = {
        "Title": stem,
        "Author": "SagePilot",
        "Subject": "PPoPP evaluation draft",
        "Keywords": "PPoPP evaluation",
        "Creator": Path(__file__).name,
        "Producer": "Matplotlib",
        "CreationDate": FIXED_TIMESTAMP,
        "ModDate": FIXED_TIMESTAMP,
    }
    figure.savefig(output_dir / f"{stem}.pdf", metadata=metadata)
    figure.savefig(output_dir / f"{stem}.png", dpi=300)
    plt.close(figure)


def build_trace_oracle() -> dict[str, object]:
    return offline_oracle.build_reference_oracle()


def attach_trace_oracle(payload: dict[str, object]) -> dict[str, object]:
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise TypeError("rows must be a list")
    for row in raw_rows:
        if not isinstance(row, dict):
            raise TypeError("row must be an object")
        cast(dict[str, object], row).pop("oracle_lower_bound", None)
    payload["schema"] = 2
    payload["description"] = (
        "Selected a1v3 Burst, Poisson 0.050, and Poisson 0.025 detail with "
        "a trace-only time-indexed SagePilot Oracle reference using Section 11 "
        "selections"
    )
    payload["oracle_lower_bound"] = build_trace_oracle()
    return payload


def generate_metrics() -> dict[str, object]:
    trace_cache: dict[Path, replay.TraceRun] = {}
    workload_cache: dict[tuple[str, str, bool], replay.ReplayWorkload] = {}
    cache_store: dict[tuple[Path, str], replay.ResourceContractCache] = {}
    rows: list[dict[str, object]] = []
    for arrival in SELECTED_ARRIVALS:
        for arm in replay.ALL_ARMS:
            for repeat in replay.A1V3_SELECTIONS[arrival][arm]:
                fused = arm in replay.FUSED_ARMS
                key = (arrival, repeat, fused)
                workload = workload_cache.get(key)
                if workload is None:
                    workload = replay.build_fixed_workload(
                        "a1v3",
                        arrival,
                        repeat,
                        fused=fused,
                        trace_cache=trace_cache,
                    )
                    workload_cache[key] = workload
                rows.append(
                    replay.replay_one(
                        "a1v3",
                        arrival,
                        arm,
                        repeat,
                        workload,
                        cache_store,
                    )
                )
    payload: dict[str, object] = {
        "schema": 1,
        "description": "Selected a1v3 detail using Section 11 manual selections",
        "selection": "serve_0728_reference.md Section 11",
        "rows": rows,
    }
    attach_trace_oracle(payload)
    validate_metrics(payload)
    return payload


def read_metrics(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"metrics must be an object: {path}")
    return cast(dict[str, object], value)


def load_metrics(path: Path) -> dict[str, object]:
    payload = read_metrics(path)
    validate_metrics(payload)
    return payload


def rows_for(
    payload: Mapping[str, object], arm: str, *, arrival: str = "burst"
) -> list[dict[str, Any]]:
    return replay.rows_for(payload, "a1v3", arrival, arm)


def aggregate(
    payload: Mapping[str, object],
    arm: str,
    path: Sequence[str],
    *,
    arrival: str = "burst",
) -> float:
    return replay.aggregate_value(rows_for(payload, arm, arrival=arrival), path)


def oracle_aggregate(
    payload: Mapping[str, object],
    path: Sequence[str],
    *,
    arrival: str,
) -> float:
    oracle = payload.get("oracle_lower_bound")
    if not isinstance(oracle, dict):
        raise TypeError("oracle_lower_bound must be an object")
    raw_rows = cast(dict[str, object], oracle).get("rows")
    if not isinstance(raw_rows, list):
        raise TypeError("oracle_lower_bound.rows must be a list")
    rows = []
    for value in raw_rows:
        if not isinstance(value, dict):
            raise TypeError("oracle_lower_bound row must be an object")
        row = cast(dict[str, Any], value)
        if row.get("arrival") == arrival:
            rows.append(row)
    return replay.aggregate_value(rows, path)


def validate_metrics(payload: Mapping[str, object]) -> None:
    if payload.get("schema") != 2:
        raise ValueError("metrics schema must be 2")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise TypeError("rows must be a list")
    expected_count = sum(
        len(repeats)
        for arrival in SELECTED_ARRIVALS
        for repeats in replay.A1V3_SELECTIONS[arrival].values()
    )
    if len(raw_rows) != expected_count:
        raise ValueError(
            f"expected {expected_count} selected rows, got {len(raw_rows)}"
        )
    for row in raw_rows:
        if not isinstance(row, dict):
            raise TypeError("row must be an object")
        if "oracle_lower_bound" in row:
            raise ValueError("Oracle must not be embedded in replay rows")
    oracle = payload.get("oracle_lower_bound")
    if not isinstance(oracle, dict):
        raise TypeError("oracle_lower_bound must be an object")
    oracle_rows = cast(dict[str, object], oracle).get("rows")
    if not isinstance(oracle_rows, list):
        raise TypeError("oracle_lower_bound.rows must be a list")
    expected_oracle_count = sum(
        len(replay.A1V3_SELECTIONS[arrival]["sagepilot"])
        for arrival in SELECTED_ARRIVALS
    )
    if len(oracle_rows) != expected_oracle_count:
        raise ValueError(
            f"expected {expected_oracle_count} Oracle rows, got {len(oracle_rows)}"
        )
    for arrival in SELECTED_ARRIVALS:
        for arm in replay.ALL_ARMS:
            rows = rows_for(payload, arm, arrival=arrival)
            actual_repeats = tuple(sorted(cast(str, row["repeat"]) for row in rows))
            expected_repeats = tuple(sorted(replay.A1V3_SELECTIONS[arrival][arm]))
            if actual_repeats != expected_repeats:
                raise ValueError(f"repeat selection mismatch for {arrival}/{arm}")
            expected = replay.EXPECTED_A1V3[(arrival, arm)]
            actual = (
                aggregate(payload, arm, ("makespan_sec",), arrival=arrival),
                aggregate(payload, arm, ("session_p50_sec",), arrival=arrival),
                aggregate(payload, arm, ("session_p95_sec",), arrival=arrival),
            )
            if tuple(round(value, 1) for value in actual) != expected:
                raise ValueError(
                    f"Section 11 mismatch for {arrival}/{arm}: {actual} != {expected}"
                )
        selected_repeats = tuple(sorted(replay.A1V3_SELECTIONS[arrival]["sagepilot"]))
        trace_rows = []
        for value in oracle_rows:
            if not isinstance(value, dict):
                raise TypeError("oracle_lower_bound row must be an object")
            oracle_row = cast(dict[str, object], value)
            if oracle_row.get("arrival") == arrival:
                trace_rows.append(oracle_row)
        repeats = []
        for row in trace_rows:
            repeat = row.get("repeat")
            if not isinstance(repeat, str):
                raise TypeError("Oracle repeat must be a string")
            repeats.append(repeat)
        actual_repeats = tuple(sorted(repeats))
        if actual_repeats != selected_repeats:
            raise ValueError(f"Oracle repeat mismatch for {arrival}")
        for row in trace_rows:
            evidence = row.get("evidence")
            if not isinstance(evidence, Mapping):
                raise TypeError("Oracle evidence must be an object")
            if evidence.get("calculation") != offline_oracle.CALCULATION:
                raise ValueError("Oracle must use the time-indexed offline reference")
        for path in (
            ("makespan_sec",),
            ("session_p95_sec",),
            *(
                ("model", model_name, field)
                for model_name in offline_oracle.TARGET_MODELS
                for field in (
                    "session_model_makespan_sec",
                    "session_model_p95_sec",
                )
            ),
        ):
            oracle = oracle_aggregate(payload, path, arrival=arrival)
            sagepilot = aggregate(
                payload,
                "sagepilot",
                path,
                arrival=arrival,
            )
            if not 0.0 < oracle <= sagepilot:
                raise ValueError(
                    f"invalid Oracle for {arrival}/{'.'.join(path)}: "
                    f"{oracle} > {sagepilot}"
                )


def metric_path(kind: str, key: str, metric: str) -> tuple[str, ...]:
    if kind == "overall":
        fields = {
            "makespan": "makespan_sec",
            "p50": "session_p50_sec",
            "p95": "session_p95_sec",
        }
        return (fields[metric],)
    if kind == "workflow":
        fields = {"makespan": "makespan_sec", "p50": "p50_sec", "p95": "p95_sec"}
        return ("workflow", key, fields[metric])
    if kind == "model":
        fields = {
            "makespan": "session_model_makespan_sec",
            "p50": "session_model_p50_sec",
            "p95": "session_model_p95_sec",
        }
        return ("model", key, fields[metric])
    raise ValueError(f"unknown panel kind: {kind}")


def style_panel(axes: Axes) -> None:
    axes.grid(axis="y", zorder=0)
    axes.set_axisbelow(True)
    axes.tick_params(axis="x", length=0)
    axes.tick_params(axis="y", length=2.0, pad=1.5)
    axes.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))


def method_handles(
    arms: Sequence[str], colors: Mapping[str, str], hatches: Mapping[str, str]
) -> list[Patch]:
    return [
        Patch(
            facecolor=colors[arm],
            edgecolor=TEXT,
            linewidth=0.35,
            hatch=hatches.get(arm, ""),
            label=replay.ARM_LABELS[arm],
        )
        for arm in arms
    ]


def plot_makespan(
    payload: Mapping[str, object],
    output_dir: Path,
    *,
    arms: Sequence[str],
    colors: Mapping[str, str],
    hatches: Mapping[str, str],
    stem: str,
) -> None:
    figure, axes_array = plt.subplots(
        3,
        3,
        figsize=(DOUBLE_COLUMN_WIDTH_IN, 5.05),
    )
    for index, ((kind, key, title), axes) in enumerate(
        zip(PANELS, axes_array.flat, strict=True)
    ):
        values = [
            aggregate(payload, arm, metric_path(kind, key, "makespan")) for arm in arms
        ]
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
        for bar, arm, value in zip(bars, arms, values, strict=True):
            bar.set_hatch(hatches.get(arm, ""))
            axes.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + max(values) * 0.025,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=7.0,
                color=TEXT,
            )
        axes.set_title(f"({chr(97 + index)}) {title}", loc="left", pad=3.0)
        axes.set_xticks([])
        axes.set_xlim(-0.6, len(arms) - 0.4)
        axes.set_ylim(0.0, max(values) * 1.18)
        style_panel(axes)
    figure.legend(
        handles=method_handles(arms, colors, hatches),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(arms),
        frameon=False,
        columnspacing=1.25,
        handletextpad=0.35,
    )
    figure.text(
        0.012,
        0.50,
        "Makespan (s)",
        rotation=90,
        ha="left",
        va="center",
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.06,
        top=0.91,
        hspace=0.42,
        wspace=0.24,
    )
    save_figure(figure, output_dir, stem)


def plot_latency(
    payload: Mapping[str, object],
    output_dir: Path,
    *,
    arms: Sequence[str],
    colors: Mapping[str, str],
    hatches: Mapping[str, str],
    stem: str,
) -> None:
    figure, axes_array = plt.subplots(
        3,
        3,
        figsize=(DOUBLE_COLUMN_WIDTH_IN, 5.05),
    )
    width = 0.34
    for index, ((kind, key, title), axes) in enumerate(
        zip(PANELS, axes_array.flat, strict=True)
    ):
        p50_values = [
            aggregate(payload, arm, metric_path(kind, key, "p50")) for arm in arms
        ]
        p95_values = [
            aggregate(payload, arm, metric_path(kind, key, "p95")) for arm in arms
        ]
        positions = list(range(len(arms)))
        axes.bar(
            [position - width / 2.0 for position in positions],
            p50_values,
            width=width,
            color="white",
            edgecolor=[colors[arm] for arm in arms],
            linewidth=0.9,
            hatch="////",
            zorder=2,
        )
        p95_bars = axes.bar(
            [position + width / 2.0 for position in positions],
            p95_values,
            width=width,
            color=[colors[arm] for arm in arms],
            edgecolor=TEXT,
            linewidth=0.35,
            zorder=2,
        )
        for bar, arm in zip(p95_bars, arms, strict=True):
            bar.set_hatch(hatches.get(arm, ""))
        axes.set_title(f"({chr(97 + index)}) {title}", loc="left", pad=3.0)
        axes.set_xticks([])
        axes.set_xlim(-0.6, len(arms) - 0.4)
        axes.set_ylim(0.0, max(p95_values) * 1.10)
        style_panel(axes)
    handles = method_handles(arms, colors, hatches)
    handles.extend(
        (
            Patch(
                facecolor="white",
                edgecolor=MUTED,
                linewidth=0.8,
                hatch="////",
                label="p50",
            ),
            Patch(facecolor=MUTED, edgecolor=TEXT, linewidth=0.35, label="p95"),
        )
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(handles),
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.3,
    )
    figure.text(
        0.012,
        0.50,
        "Completion latency (s)",
        rotation=90,
        ha="left",
        va="center",
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.06,
        top=0.91,
        hspace=0.42,
        wspace=0.24,
    )
    save_figure(figure, output_dir, stem)


def plot_gpu_residency(payload: Mapping[str, object], output_dir: Path) -> None:
    figure, axes = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH_IN, 2.75))
    states = tuple(GPU_STATE_COLORS)
    positions = list(range(len(SYSTEM_ARMS)))
    bottoms = [0.0] * len(SYSTEM_ARMS)
    distributions: dict[str, list[float]] = {}
    for arm in SYSTEM_ARMS:
        totals = [aggregate(payload, arm, (state,)) for state in states]
        total = sum(totals)
        distributions[arm] = [100.0 * value / total for value in totals]
    for state_index, state in enumerate(states):
        values = [distributions[arm][state_index] for arm in SYSTEM_ARMS]
        bars = axes.bar(
            positions,
            values,
            bottom=bottoms,
            width=0.62,
            color=GPU_STATE_COLORS[state],
            edgecolor="white",
            linewidth=0.45,
            label=GPU_STATE_LABELS[state],
            zorder=2,
        )
        for bar, bottom, value in zip(bars, bottoms, values, strict=True):
            if value >= 7.5:
                axes.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bottom + value / 2.0,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color=GPU_STATE_TEXT[state],
                    fontsize=7.0,
                    fontweight="bold",
                )
        bottoms = [
            bottom + value for bottom, value in zip(bottoms, values, strict=True)
        ]
    for position, arm in zip(positions, SYSTEM_ARMS, strict=True):
        axes.add_patch(
            Rectangle(
                (position - 0.31, 0.0),
                0.62,
                100.0,
                fill=False,
                edgecolor=SYSTEM_COLORS[arm],
                linewidth=1.1,
                zorder=3,
            )
        )
    labels = [replay.ARM_LABELS[arm] for arm in SYSTEM_ARMS]
    axes.set_xticks(positions, labels, rotation=24, ha="right")
    for label, arm in zip(axes.get_xticklabels(), SYSTEM_ARMS, strict=True):
        label.set_color(SYSTEM_COLORS[arm])
        label.set_fontweight("bold")
    axes.set_ylim(0.0, 100.0)
    axes.set_yticks((0.0, 25.0, 50.0, 75.0, 100.0), ("0%", "25%", "50%", "75%", "100%"))
    axes.set_ylabel("GPU time")
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
    figure.subplots_adjust(left=0.16, right=0.98, bottom=0.25, top=0.80)
    save_figure(figure, output_dir, "fig_gpu_residency_systems_draft")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / METRICS_PATH.name
    if args.plot_only:
        payload = load_metrics(metrics_path)
    elif args.refresh_oracle_from_traces:
        payload = attach_trace_oracle(read_metrics(metrics_path))
        validate_metrics(payload)
        metrics_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        payload = generate_metrics()
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    configure_matplotlib()
    no_hatches: dict[str, str] = {}
    plot_makespan(
        payload,
        output_dir,
        arms=SYSTEM_ARMS,
        colors=SYSTEM_COLORS,
        hatches=no_hatches,
        stem="fig_makespan_systems_draft",
    )
    plot_makespan(
        payload,
        output_dir,
        arms=ABLATION_ARMS,
        colors=ABLATION_COLORS,
        hatches=ABLATION_HATCHES,
        stem="fig_makespan_ablations_draft",
    )
    plot_latency(
        payload,
        output_dir,
        arms=SYSTEM_ARMS,
        colors=SYSTEM_COLORS,
        hatches=no_hatches,
        stem="fig_latency_systems_draft",
    )
    plot_latency(
        payload,
        output_dir,
        arms=ABLATION_ARMS,
        colors=ABLATION_COLORS,
        hatches=ABLATION_HATCHES,
        stem="fig_latency_ablations_draft",
    )
    plot_gpu_residency(payload, output_dir)
    print(f"wrote selected a1v3 bar drafts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
