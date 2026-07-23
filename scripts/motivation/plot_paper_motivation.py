from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, cast

from reportlab.lib.colors import Color, HexColor
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

STAGES = ("chunk_0", "chunk_1", "chunk_2", "merge")
FONT_REGULAR = "FigureSans"
FONT_BOLD = "FigureSans-Bold"
FIGURE_WIDTH = 7.16 * inch
FIGURE_HEIGHT = 3.10 * inch
MIN_FONT_SIZE = 9.0
PANEL_FONT_SIZE = 10.0
METRIC_FONT_SIZE = 9.5
COLORS = {
    "text": HexColor("#202124"),
    "muted": HexColor("#50545A"),
    "grid": HexColor("#C9CDD2"),
    "chunk_0": HexColor("#26456E"),
    "chunk_1": HexColor("#5C8D89"),
    "chunk_2": HexColor("#D89B39"),
    "merge": HexColor("#8F3B4F"),
    "idle": HexColor("#E6E8EB"),
    "recovered": HexColor("#E2EDDF"),
    "recovered_text": HexColor("#3F6B3C"),
    "resident": HexColor("#F2F3F4"),
    "resident_line": HexColor("#73777D"),
}

pdfmetrics.registerFont(TTFont(FONT_REGULAR, "Vera.ttf"))
pdfmetrics.registerFont(TTFont(FONT_BOLD, "VeraBd.ttf"))

Interval = tuple[float, float]
Schedule = dict[str, list[Interval]]


@dataclass(frozen=True)
class QMSumSample:
    chunk_durations: tuple[float, float, float]
    chunk_offsets: tuple[float, float, float]
    merge_duration: float
    merge_offset: float
    generation_duration: float


@dataclass(frozen=True)
class MBPPSample:
    duration: float
    coder: Interval
    reviewer: Interval
    repair: Interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/motivation"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("paper/hpca2027-sagepilot/figures"),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def group_by_sample(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["sample_id"])].append(record)
    return dict(grouped)


def single_event(events: list[dict[str, Any]], node_name: str) -> dict[str, Any]:
    matched = [event for event in events if event["node_name"] == node_name]
    assert len(matched) == 1, (node_name, len(matched))
    return matched[0]


def load_qmsum_samples(path: Path) -> list[QMSumSample]:
    samples = []
    for sample_id, events in group_by_sample(read_jsonl(path)).items():
        chunks = sorted(
            [event for event in events if str(event["node_name"]).startswith("chunk_")],
            key=lambda event: str(event["node_name"]),
        )
        assert len(chunks) == 3, (sample_id, len(chunks))
        merge = single_event(events, "merge")
        first_start = min(float(event["started_at"]) for event in chunks)
        samples.append(
            QMSumSample(
                chunk_durations=cast(
                    tuple[float, float, float],
                    tuple(float(event["duration_sec"]) for event in chunks),
                ),
                chunk_offsets=cast(
                    tuple[float, float, float],
                    tuple(float(event["started_at"]) - first_start for event in chunks),
                ),
                merge_duration=float(merge["duration_sec"]),
                merge_offset=float(merge["started_at"]) - first_start,
                generation_duration=float(merge["ended_at"]) - first_start,
            )
        )
    return samples


def load_mbpp_samples(path: Path) -> list[MBPPSample]:
    samples = []
    for events in group_by_sample(read_jsonl(path)).values():
        coder = single_event(events, "coder")
        reviewer = single_event(events, "reviewer")
        repair = single_event(events, "repair")
        final_tester = single_event(events, "final_tester")
        start = float(coder["started_at"])
        end = float(final_tester["ended_at"])
        samples.append(
            MBPPSample(
                duration=end - start,
                coder=relative_interval(coder, start),
                reviewer=relative_interval(reviewer, start),
                repair=relative_interval(repair, start),
            )
        )
    return samples


def relative_interval(event: dict[str, Any], origin: float) -> Interval:
    start = float(event["started_at"])
    end = float(event["ended_at"])
    return start - origin, end - start


def observed_schedule(
    samples: list[QMSumSample],
) -> tuple[Schedule, list[Interval], float]:
    schedule = {stage: [] for stage in STAGES}
    session_spans = []
    cursor = 0.0
    for sample in samples:
        for index, (offset, duration) in enumerate(
            zip(sample.chunk_offsets, sample.chunk_durations, strict=True)
        ):
            schedule[f"chunk_{index}"].append((cursor + offset, duration))
        schedule["merge"].append((cursor + sample.merge_offset, sample.merge_duration))
        session_spans.append((cursor, sample.generation_duration))
        cursor += sample.generation_duration
    return schedule, session_spans, cursor


def replay_schedule(samples: list[QMSumSample]) -> tuple[Schedule, float]:
    schedule = {stage: [] for stage in STAGES}
    chunk_available = [0.0, 0.0, 0.0]
    merge_available = 0.0
    for sample in samples:
        done_times = []
        for index, duration in enumerate(sample.chunk_durations):
            start = chunk_available[index]
            schedule[f"chunk_{index}"].append((start, duration))
            chunk_available[index] = start + duration
            done_times.append(chunk_available[index])
        merge_start = max(max(done_times), merge_available)
        schedule["merge"].append((merge_start, sample.merge_duration))
        merge_available = merge_start + sample.merge_duration
    return schedule, merge_available


def draw_text(
    pdf: Canvas,
    x: float,
    y: float,
    value: str,
    *,
    size: float,
    color: Color | None = None,
    bold: bool = False,
    align: str = "left",
) -> None:
    assert size >= MIN_FONT_SIZE, size
    pdf.setFont(FONT_BOLD if bold else FONT_REGULAR, size)
    pdf.setFillColor(color or COLORS["text"])
    if align == "right":
        pdf.drawRightString(x, y, value)
    elif align == "center":
        pdf.drawCentredString(x, y, value)
    else:
        pdf.drawString(x, y, value)


def hatch_rect(
    pdf: Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    pdf.saveState()
    pdf.setFillColor(COLORS["resident"])
    pdf.rect(x, y, width, height, stroke=0, fill=1)
    path = pdf.beginPath()
    path.rect(x, y, width, height)
    pdf.clipPath(path, stroke=0, fill=0)
    pdf.setStrokeColor(COLORS["resident_line"])
    pdf.setLineWidth(0.6)
    offset = -height
    while offset < width:
        pdf.line(x + offset, y, x + offset + height, y + height)
        offset += 4.0
    pdf.restoreState()
    pdf.setStrokeColor(COLORS["resident_line"])
    pdf.setLineWidth(0.6)
    pdf.rect(x, y, width, height, stroke=1, fill=0)


def interval_rect(
    interval: Interval,
    *,
    x0: float,
    x1: float,
    horizon: float,
) -> tuple[float, float]:
    start, duration = interval
    left = x0 + (x1 - x0) * start / horizon
    width = max(0.25, (x1 - x0) * duration / horizon)
    return left, width


def draw_schedule(
    pdf: Canvas,
    *,
    schedule: Schedule,
    session_spans: list[Interval] | None,
    x0: float,
    x1: float,
    row_y: tuple[float, float, float, float],
    bar_height: float,
    baseline_span: float,
) -> None:
    if session_spans is not None:
        pdf.setFillColor(COLORS["idle"])
        for y in row_y[:3]:
            for interval in session_spans:
                left, width = interval_rect(
                    interval,
                    x0=x0,
                    x1=x1,
                    horizon=baseline_span,
                )
                pdf.rect(left, y, width, bar_height, stroke=0, fill=1)
    for stage, y in zip(STAGES, row_y, strict=True):
        pdf.setFillColor(COLORS[stage])
        for interval in schedule[stage]:
            left, width = interval_rect(
                interval,
                x0=x0,
                x1=x1,
                horizon=baseline_span,
            )
            pdf.rect(left, y, width, bar_height, stroke=0, fill=1)


def draw_lifecycle_figure(
    output_path: Path,
    qmsum_samples: list[QMSumSample],
    mbpp_samples: list[MBPPSample],
    qmsum_summary: dict[str, Any],
    mbpp_summary: dict[str, Any],
) -> None:
    pdf = Canvas(
        str(output_path),
        pagesize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        pageCompression=1,
        invariant=1,
        initialFontName=FONT_REGULAR,
    )
    pdf.setTitle("SagePilot lifecycle motivation evidence")
    pdf.setCreator("scripts/motivation/plot_paper_motivation.py")
    observed, session_spans, baseline_span = observed_schedule(qmsum_samples)
    replay, replay_span = replay_schedule(qmsum_samples)
    assert math.isclose(
        baseline_span, float(qmsum_summary["baseline_total_sec"]), abs_tol=1e-6
    )
    assert math.isclose(
        replay_span, float(qmsum_summary["oracle_total_sec"]), abs_tol=1e-6
    )
    draw_qmsum_schedule_panel(
        pdf,
        observed=observed,
        replay=replay,
        session_spans=session_spans,
        baseline_span=baseline_span,
        replay_span=replay_span,
    )
    draw_qmsum_cdf_panel(pdf, qmsum_summary)
    draw_mbpp_panel(pdf, mbpp_samples, mbpp_summary)
    pdf.showPage()
    pdf.save()


def draw_qmsum_schedule_panel(
    pdf: Canvas,
    *,
    observed: Schedule,
    replay: Schedule,
    session_spans: list[Interval],
    baseline_span: float,
    replay_span: float,
) -> None:
    x0 = 34.0
    x1 = 244.0
    speedup = baseline_span / replay_span
    draw_text(
        pdf,
        6.0,
        211.0,
        "(a) QMSum cross-session overlap",
        size=PANEL_FONT_SIZE,
        bold=True,
    )
    draw_text(
        pdf,
        x1,
        198.0,
        f"{baseline_span:,.0f} -> {replay_span:,.0f} s | {speedup:.2f}x",
        size=METRIC_FONT_SIZE,
        color=COLORS["chunk_0"],
        bold=True,
        align="right",
    )
    draw_text(
        pdf, x0, 184.0, "Session-at-a-time", size=MIN_FONT_SIZE, bold=True
    )
    draw_text(
        pdf,
        x0 + 91.0,
        184.0,
        "gray = pipeline bubble",
        size=MIN_FONT_SIZE,
        color=COLORS["muted"],
    )
    observed_y = (163.0, 150.0, 137.0, 124.0)
    replay_y = (84.0, 71.0, 58.0, 45.0)
    draw_schedule(
        pdf,
        schedule=observed,
        session_spans=session_spans,
        x0=x0,
        x1=x1,
        row_y=observed_y,
        bar_height=8.0,
        baseline_span=baseline_span,
    )
    draw_text(pdf, x0, 105.0, "Offline replay", size=MIN_FONT_SIZE, bold=True)
    replay_end_x = x0 + (x1 - x0) * replay_span / baseline_span
    pdf.setFillColor(COLORS["recovered"])
    pdf.rect(replay_end_x, 42.0, x1 - replay_end_x, 51.0, stroke=0, fill=1)
    draw_schedule(
        pdf,
        schedule=replay,
        session_spans=None,
        x0=x0,
        x1=x1,
        row_y=replay_y,
        bar_height=8.0,
        baseline_span=baseline_span,
    )
    draw_text(
        pdf,
        (replay_end_x + x1) / 2,
        65.0,
        "recovered span",
        size=MIN_FONT_SIZE,
        color=COLORS["recovered_text"],
        bold=True,
        align="center",
    )
    pdf.setStrokeColor(COLORS["text"])
    pdf.setLineWidth(0.9)
    pdf.line(replay_end_x, 42.0, replay_end_x, 93.0)
    draw_text(
        pdf,
        replay_end_x - 2.0,
        95.0,
        f"{replay_span:,.0f}s",
        size=MIN_FONT_SIZE,
        bold=True,
        align="right",
    )
    for y, label in zip(observed_y, ("C0", "C1", "C2", "M"), strict=True):
        draw_text(
            pdf,
            x0 - 7.0,
            y + 0.8,
            label,
            size=MIN_FONT_SIZE,
            color=COLORS["muted"],
            align="right",
        )
    for y, label in zip(replay_y, ("C0", "C1", "C2", "M"), strict=True):
        draw_text(
            pdf,
            x0 - 7.0,
            y + 0.8,
            label,
            size=MIN_FONT_SIZE,
            color=COLORS["muted"],
            align="right",
        )
    axis_y = 30.0
    pdf.setStrokeColor(COLORS["text"])
    pdf.setLineWidth(0.8)
    pdf.line(x0, axis_y, x1, axis_y)
    for tick in (0, 2000, 4000, 6000):
        tick_x = x0 + (x1 - x0) * tick / baseline_span
        pdf.line(tick_x, axis_y, tick_x, axis_y - 3.0)
        draw_text(
            pdf,
            tick_x,
            axis_y - 12.0,
            f"{tick:,}",
            size=MIN_FONT_SIZE,
            color=COLORS["muted"],
            align="center",
        )
    draw_text(
        pdf,
        (x0 + x1) / 2,
        3.5,
        "Generation-only span (s)",
        size=MIN_FONT_SIZE,
        align="center",
    )


def draw_qmsum_cdf_panel(pdf: Canvas, summary: dict[str, Any]) -> None:
    draw_text(
        pdf, 252.0, 211.0, "(b) Bubble ratio", size=PANEL_FONT_SIZE, bold=True
    )
    draw_text(
        pdf,
        252.0,
        198.0,
        "stage utilization",
        size=MIN_FONT_SIZE,
        color=COLORS["recovered_text"],
    )
    draw_text(
        pdf,
        352.0,
        198.0,
        f"{float(summary['stage_utilization']):.0%}",
        size=MIN_FONT_SIZE,
        color=COLORS["recovered_text"],
        bold=True,
        align="right",
    )
    x0 = 273.0
    x1 = 352.0
    y0 = 30.0
    y1 = 187.0
    x_min = 0.26
    x_max = 0.54

    def map_x(value: float) -> float:
        return x0 + (x1 - x0) * (value - x_min) / (x_max - x_min)

    def map_y(value: float) -> float:
        return y0 + (y1 - y0) * value

    pdf.setStrokeColor(COLORS["grid"])
    pdf.setLineWidth(0.6)
    for tick in (0.0, 0.5, 1.0):
        tick_y = map_y(tick)
        pdf.line(x0, tick_y, x1, tick_y)
        draw_text(
            pdf,
            x0 - 6.0,
            tick_y - 2.5,
            f"{tick:.1f}",
            size=MIN_FONT_SIZE,
            color=COLORS["muted"],
            align="right",
        )
    pdf.setStrokeColor(COLORS["text"])
    pdf.setLineWidth(0.8)
    pdf.line(x0, y0, x0, y1)
    pdf.line(x0, y0, x1, y0)
    ratios = sorted(
        float(sample["pipeline_bubble_ratio"]) for sample in summary["samples"]
    )
    path = pdf.beginPath()
    for index, ratio in enumerate(ratios):
        point = map_x(ratio), map_y((index + 1) / len(ratios))
        if index == 0:
            path.moveTo(*point)
        else:
            path.lineTo(*point)
    pdf.setStrokeColor(COLORS["chunk_0"])
    pdf.setLineWidth(1.6)
    pdf.drawPath(path, stroke=1, fill=0)
    mean = float(summary["pipeline_bubble_ratio"]["mean"])
    p95 = float(summary["pipeline_bubble_ratio"]["p95"])
    for value, dash in ((mean, (3, 2)), (p95, (1, 2))):
        line_x = map_x(value)
        pdf.setDash(*dash)
        pdf.setStrokeColor(COLORS["muted"])
        pdf.setLineWidth(0.8)
        pdf.line(line_x, y0, line_x, y1)
    pdf.setDash()
    draw_text(
        pdf,
        map_x(mean) - 4.0,
        map_y(0.70),
        f"mean {mean:.3f}",
        size=MIN_FONT_SIZE,
        color=COLORS["muted"],
        bold=True,
        align="right",
    )
    draw_text(
        pdf,
        x1 - 2.0,
        map_y(0.88),
        f"P95 {p95:.3f}",
        size=MIN_FONT_SIZE,
        color=COLORS["muted"],
        bold=True,
        align="right",
    )
    for tick in (0.3, 0.4, 0.5):
        tick_x = map_x(tick)
        pdf.setStrokeColor(COLORS["text"])
        pdf.setLineWidth(0.8)
        pdf.line(tick_x, y0, tick_x, y0 - 3.0)
        draw_text(
            pdf,
            tick_x,
            y0 - 12.0,
            f"{tick:.1f}",
            size=MIN_FONT_SIZE,
            color=COLORS["muted"],
            align="center",
        )
    draw_text(
        pdf,
        (x0 + x1) / 2,
        3.5,
        "Bubble ratio",
        size=MIN_FONT_SIZE,
        align="center",
    )


def draw_mbpp_panel(
    pdf: Canvas,
    samples: list[MBPPSample],
    summary: dict[str, Any],
) -> None:
    resident_total = float(summary["resident_model_seconds"]["total"])
    active_total = float(summary["active_model_seconds"]["total"])
    trace_resident_total = sum(3.0 * sample.duration for sample in samples)
    trace_active_total = sum(
        interval[1]
        for sample in samples
        for interval in (sample.coder, sample.reviewer, sample.repair)
    )
    assert math.isclose(trace_resident_total, resident_total, abs_tol=1e-6)
    assert math.isclose(trace_active_total, active_total, abs_tol=1e-6)
    draw_text(
        pdf, 360.0, 211.0, "(c) MBPP residency gap", size=PANEL_FONT_SIZE, bold=True
    )
    draw_text(
        pdf,
        360.0,
        198.0,
        "3 resident models; <= 1 active",
        size=MIN_FONT_SIZE,
        color=COLORS["muted"],
    )
    median_duration = median(sample.duration for sample in samples)
    sample = min(samples, key=lambda item: abs(item.duration - median_duration))
    x0 = 401.0
    x1 = 507.0
    row_y = (161.0, 126.0, 91.0)
    names = ("Coder", "Reviewer", "Repair")
    intervals = (sample.coder, sample.reviewer, sample.repair)
    stage_colors = (COLORS["chunk_0"], COLORS["chunk_1"], COLORS["chunk_2"])
    stage_text_colors = (HexColor("#FFFFFF"), HexColor("#FFFFFF"), COLORS["text"])
    for name, interval, color, text_color, y in zip(
        names, intervals, stage_colors, stage_text_colors, row_y, strict=True
    ):
        draw_text(
            pdf,
            x0 - 7.0,
            y + 3.5,
            name,
            size=MIN_FONT_SIZE,
            color=COLORS["muted"],
            bold=True,
            align="right",
        )
        hatch_rect(pdf, x0, y, x1 - x0, 16.0)
        left, width = interval_rect(
            interval,
            x0=x0,
            x1=x1,
            horizon=sample.duration,
        )
        pdf.setFillColor(color)
        pdf.rect(left, y, width, 16.0, stroke=0, fill=1)
        if width >= 20.0:
            draw_text(
                pdf,
                left + width / 2,
                y + 4.0,
                f"{interval[1]:.0f}s",
                size=MIN_FONT_SIZE,
                color=text_color,
                bold=True,
                align="center",
            )
    axis_y = 78.0
    pdf.setStrokeColor(COLORS["text"])
    pdf.setLineWidth(0.8)
    pdf.line(x0, axis_y, x1, axis_y)
    for fraction in (0.0, 0.5, 1.0):
        tick_x = x0 + (x1 - x0) * fraction
        pdf.line(tick_x, axis_y, tick_x, axis_y - 3.0)
        draw_text(
            pdf,
            tick_x,
            axis_y - 12.0,
            f"{sample.duration * fraction:.0f}",
            size=MIN_FONT_SIZE,
            color=COLORS["muted"],
            align="center",
        )
    draw_text(
        pdf,
        (x0 + x1) / 2,
        52.0,
        "Relative session time (s)",
        size=MIN_FONT_SIZE,
        align="center",
    )
    pdf.setFillColor(COLORS["chunk_0"])
    pdf.rect(362.0, 39.0, 10.0, 8.0, stroke=0, fill=1)
    draw_text(pdf, 377.0, 38.5, "active", size=MIN_FONT_SIZE)
    hatch_rect(pdf, 435.0, 39.0, 10.0, 8.0)
    draw_text(pdf, 450.0, 38.5, "resident", size=MIN_FONT_SIZE)
    active_fraction = active_total / resident_total
    hatch_rect(pdf, x0, 22.0, x1 - x0, 12.0)
    active_width = (x1 - x0) * active_fraction
    pdf.setFillColor(COLORS["chunk_0"])
    pdf.rect(x0, 22.0, active_width, 12.0, stroke=0, fill=1)
    draw_text(
        pdf,
        x0 + active_width / 2,
        24.5,
        f"{active_fraction:.0%}",
        size=MIN_FONT_SIZE,
        color=HexColor("#FFFFFF"),
        bold=True,
        align="center",
    )
    draw_text(
        pdf,
        x0 + active_width + (x1 - x0 - active_width) / 2,
        24.5,
        f"{1 - active_fraction:.0%}",
        size=MIN_FONT_SIZE,
        color=COLORS["muted"],
        bold=True,
        align="center",
    )
    draw_text(
        pdf,
        (360.0 + 507.0) / 2,
        5.0,
        f"{active_total:,.0f} -> {resident_total:,.0f} GPU-s "
        f"({float(summary['resource_gap']):.2f}x)",
        size=MIN_FONT_SIZE,
        bold=True,
        align="center",
    )


def main() -> None:
    args = parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    qmsum_summary = json.loads(
        (args.input_dir / "qmsum_3way_summary.json").read_text("utf-8")
    )
    mbpp_summary = json.loads(
        (args.input_dir / "mbpp_chain_summary.json").read_text("utf-8")
    )
    qmsum_samples = load_qmsum_samples(args.input_dir / "qmsum_3way_trace.jsonl")
    mbpp_samples = load_mbpp_samples(args.input_dir / "mbpp_chain_trace.jsonl")
    draw_lifecycle_figure(
        args.figure_dir / "lifecycle_motivation.pdf",
        qmsum_samples,
        mbpp_samples,
        qmsum_summary,
        mbpp_summary,
    )


if __name__ == "__main__":
    main()
