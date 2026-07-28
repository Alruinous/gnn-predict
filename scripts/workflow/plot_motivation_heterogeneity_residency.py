from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from reportlab.lib.colors import Color, HexColor
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import EmbeddedType1Face, Font
from reportlab.pdfgen.canvas import Canvas

from experiment.workflow.analysis import summarize_trace

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = REPO_ROOT / "output" / "motivation_20260727_heterogeneity_residency"
SUMMARY_PATH = EXPERIMENT_DIR / "motivation_summary.json"
OUTPUT_FIGURE = EXPERIMENT_DIR / "figures" / "heterogeneity_residency_draft.pdf"

FIGURE_WIDTH = 3.33 * inch
FIGURE_HEIGHT = 1.35 * inch
MIN_FONT_SIZE = 8.0
FONT_REGULAR = "FigureSans"
FONT_BOLD = "FigureSans-Bold"
EXPECTED_ARRIVAL_DIGEST = "e12d09229e"
EXPECTED_SESSION_COUNT = 60
EXPECTED_ACCESS_COUNT = 360
EXPECTED_GPU_COUNT = 3

COLORS = {
    "text": HexColor("#202124"),
    "muted": HexColor("#50545A"),
    "grid": HexColor("#C9CDD2"),
    "generation": HexColor("#31577D"),
    "idle_resident": HexColor("#D2D6DA"),
    "loading": HexColor("#D89B39"),
    "other": HexColor("#777C83"),
    "white": HexColor("#FFFFFF"),
    "hideable_line": HexColor("#416F6B"),
}


@dataclass(frozen=True, slots=True)
class RunSpec:
    run_id: str
    label: str
    model_count: int


@dataclass(frozen=True, slots=True)
class RunMetrics:
    label: str
    generation_pct: float
    idle_resident_pct: float
    loading_pct: float
    other_pct: float
    hideable_pct: float


RUN_SPECS = (
    RunSpec("h5_parrot", "Parrot", 5),
    RunSpec("h5_kairos", "Kairos", 5),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw the heterogeneous-model residency motivation figure."
    )
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_FIGURE)
    return parser.parse_args()


def tex_font_path(filename: str) -> Path:
    result = subprocess.run(
        ["kpsewhich", filename],
        check=True,
        capture_output=True,
        text=True,
    )
    path = Path(result.stdout.strip())
    if not path.is_file():
        raise FileNotFoundError(filename)
    return path


def register_fonts() -> None:
    for font_name, stem in (
        (FONT_REGULAR, "lmss10"),
        (FONT_BOLD, "lmssbx10"),
    ):
        face = EmbeddedType1Face(
            str(tex_font_path(f"{stem}.afm")),
            str(tex_font_path(f"{stem}.pfb")),
        )
        pdfmetrics.registerTypeFace(face)
        pdfmetrics.registerFont(Font(font_name, face.name, "WinAnsiEncoding"))


def require_object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{path} must be a JSON object")
    return cast(dict[str, object], value)


def object_at(parent: dict[str, object], key: str, path: str) -> dict[str, object]:
    if key not in parent:
        raise KeyError(f"{path}.{key}")
    return require_object(parent[key], f"{path}.{key}")


def number_at(parent: dict[str, object], key: str, path: str) -> float:
    if key not in parent:
        raise KeyError(f"{path}.{key}")
    value = parent[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path}.{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path}.{key} must be finite")
    return result


def integer_at(parent: dict[str, object], key: str, path: str) -> int:
    if key not in parent:
        raise KeyError(f"{path}.{key}")
    value = parent[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path}.{key} must be an integer")
    return value


def string_at(parent: dict[str, object], key: str, path: str) -> str:
    if key not in parent:
        raise KeyError(f"{path}.{key}")
    value = parent[key]
    if not isinstance(value, str):
        raise TypeError(f"{path}.{key} must be a string")
    return value


def validate_ratio(value: float, path: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{path} must be in [0, 1]")


def load_run_metrics(
    root: dict[str, object],
    spec: RunSpec,
    experiment_dir: Path,
) -> RunMetrics:
    run = object_at(root, spec.run_id, "summary")
    sessions = object_at(run, "sessions", spec.run_id)
    if integer_at(sessions, "completed", f"{spec.run_id}.sessions") != (
        EXPECTED_SESSION_COUNT
    ):
        raise ValueError(f"{spec.run_id} must contain 60 completed sessions")
    if integer_at(sessions, "failed", f"{spec.run_id}.sessions") != 0:
        raise ValueError(f"{spec.run_id} contains failed sessions")
    digest = string_at(run, "arrival_digest", spec.run_id)
    if digest != EXPECTED_ARRIVAL_DIGEST:
        raise ValueError(f"{spec.run_id} has arrival digest {digest}")

    demand = object_at(run, "demand", spec.run_id)
    model_count = integer_at(demand, "distinct_model_keys", f"{spec.run_id}.demand")
    if model_count != spec.model_count:
        raise ValueError(
            f"{spec.run_id} has {model_count} models, expected {spec.model_count}"
        )
    belady = object_at(run, "belady", spec.run_id)
    access_count = integer_at(belady, "access_count", f"{spec.run_id}.belady")
    if access_count != EXPECTED_ACCESS_COUNT:
        raise ValueError(f"{spec.run_id} must contain 360 model accesses")

    loading = object_at(run, "loading", spec.run_id)
    loading_seconds = number_at(
        loading,
        "loading_gpu_seconds",
        f"{spec.run_id}.loading",
    )
    generation_seconds = number_at(
        loading,
        "generation_gpu_seconds",
        f"{spec.run_id}.loading",
    )
    loading_ratio = number_at(
        loading,
        "loading_over_generation",
        f"{spec.run_id}.loading",
    )
    if loading_seconds < 0.0 or generation_seconds <= 0.0:
        raise ValueError(f"{spec.run_id} has invalid GPU-time totals")
    computed_loading_ratio = loading_seconds / generation_seconds
    if not math.isclose(loading_ratio, computed_loading_ratio, rel_tol=1e-9):
        raise ValueError(f"{spec.run_id} has inconsistent loading ratio")

    preload = object_at(run, "preload", spec.run_id)
    hideable_seconds = number_at(
        preload,
        "hideable_load_sec",
        f"{spec.run_id}.preload",
    )
    hideable_loading_ratio = number_at(
        preload,
        "hideable_load_fraction",
        f"{spec.run_id}.preload",
    )
    if not 0.0 <= hideable_seconds <= loading_seconds:
        raise ValueError(f"{spec.run_id} has invalid hideable loading time")
    validate_ratio(
        hideable_loading_ratio,
        f"{spec.run_id}.preload.hideable_load_fraction",
    )
    if loading_seconds > 0.0 and not math.isclose(
        hideable_loading_ratio,
        hideable_seconds / loading_seconds,
        rel_tol=1e-9,
    ):
        raise ValueError(f"{spec.run_id} has inconsistent hideable fraction")

    trace = require_object(
        summarize_trace(experiment_dir / spec.run_id / "workflow_trace.jsonl"),
        f"{spec.run_id}.trace",
    )
    run_duration = number_at(trace, "run_duration_sec", f"{spec.run_id}.trace")
    generation = number_at(trace, "active_gpu_seconds", f"{spec.run_id}.trace")
    resident = number_at(trace, "resident_gpu_seconds", f"{spec.run_id}.trace")
    idle_resident = number_at(
        trace,
        "idle_resident_gpu_seconds",
        f"{spec.run_id}.trace",
    )
    trace_loading = number_at(
        trace,
        "loading_gpu_seconds",
        f"{spec.run_id}.trace",
    )
    evicting = number_at(
        trace,
        "evicting_gpu_seconds",
        f"{spec.run_id}.trace",
    )
    gpu_times = (generation, idle_resident, trace_loading, evicting)
    if run_duration <= 0.0 or min(gpu_times) < 0:
        raise ValueError(f"{spec.run_id} has invalid trace GPU-time totals")
    if not math.isclose(resident, generation + idle_resident, rel_tol=1e-9):
        raise ValueError(f"{spec.run_id} has inconsistent resident GPU time")
    if not math.isclose(trace_loading, loading_seconds, rel_tol=1e-9):
        raise ValueError(f"{spec.run_id} has inconsistent loading GPU time")

    total_gpu_seconds = EXPECTED_GPU_COUNT * run_duration
    other = total_gpu_seconds - generation - idle_resident - trace_loading
    if other < evicting:
        raise ValueError(f"{spec.run_id} has incomplete GPU-time accounting")
    percentages = (
        100.0 * generation / total_gpu_seconds,
        100.0 * idle_resident / total_gpu_seconds,
        100.0 * trace_loading / total_gpu_seconds,
        100.0 * other / total_gpu_seconds,
    )
    if not math.isclose(sum(percentages), 100.0, abs_tol=1e-9):
        raise ValueError(f"{spec.run_id} GPU-time shares do not sum to 100%")

    return RunMetrics(
        label=spec.label,
        generation_pct=percentages[0],
        idle_resident_pct=percentages[1],
        loading_pct=percentages[2],
        other_pct=percentages[3],
        hideable_pct=100.0 * hideable_seconds / total_gpu_seconds,
    )


def load_figure_data(path: Path) -> tuple[RunMetrics, RunMetrics]:
    raw: object = json.loads(path.read_text("utf-8"))
    root = require_object(raw, "summary")
    return (
        load_run_metrics(root, RUN_SPECS[0], path.parent),
        load_run_metrics(root, RUN_SPECS[1], path.parent),
    )


def draw_text(
    pdf: Canvas,
    x: float,
    y: float,
    value: str,
    *,
    size: float = MIN_FONT_SIZE,
    color: Color | None = None,
    bold: bool = False,
    align: str = "left",
) -> None:
    if size < MIN_FONT_SIZE:
        raise ValueError(f"Figure label below minimum size: {size}")
    pdf.setFont(FONT_BOLD if bold else FONT_REGULAR, size)
    pdf.setFillColor(color or COLORS["text"])
    if align == "right":
        pdf.drawRightString(x, y, value)
    elif align == "center":
        pdf.drawCentredString(x, y, value)
    elif align == "left":
        pdf.drawString(x, y, value)
    else:
        raise ValueError(f"Unsupported alignment: {align}")


def draw_hatch_overlay(
    pdf: Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    if width <= 0.0:
        return
    pdf.saveState()
    path = pdf.beginPath()
    path.rect(x, y, width, height)
    pdf.clipPath(path, stroke=0, fill=0)
    pdf.setStrokeColor(COLORS["hideable_line"])
    pdf.setLineWidth(0.55)
    offset = -height
    while offset < width:
        pdf.line(x + offset, y, x + offset + height, y + height)
        offset += 3.0
    pdf.restoreState()


def draw_gpu_time_bar(
    pdf: Canvas,
    metrics: RunMetrics,
    *,
    x0: float,
    scale_width: float,
    y: float,
) -> None:
    height = 11.0

    draw_text(
        pdf,
        x0 - 4.0,
        y + 1.4,
        metrics.label,
        color=COLORS["muted"],
        bold=True,
        align="right",
    )

    segments = (
        (metrics.generation_pct, "generation", COLORS["white"]),
        (metrics.idle_resident_pct, "idle_resident", COLORS["text"]),
        (metrics.loading_pct, "loading", COLORS["text"]),
        (metrics.other_pct, "other", COLORS["white"]),
    )
    positions: list[tuple[float, float, float, Color]] = []
    left = x0
    loading_x = 0.0
    loading_width = 0.0
    for percentage, color_name, text_color in segments:
        width = scale_width * percentage / 100.0
        pdf.setFillColor(COLORS[color_name])
        pdf.setStrokeColor(COLORS["grid"])
        pdf.setLineWidth(0.35)
        pdf.rect(left, y, width, height, stroke=1, fill=1)
        positions.append((left, width, percentage, text_color))
        if color_name == "loading":
            loading_x = left
            loading_width = width
        left += width

    hideable_width = scale_width * metrics.hideable_pct / 100.0
    if hideable_width > loading_width:
        raise ValueError(f"{metrics.label} hideable time exceeds loading time")
    draw_hatch_overlay(
        pdf,
        loading_x + loading_width - hideable_width,
        y,
        hideable_width,
        height,
    )
    for left, width, percentage, text_color in positions:
        if width < 24.0:
            continue
        draw_text(
            pdf,
            left + width / 2,
            y + 1.4,
            f"{percentage:.0f}%",
            color=text_color,
            bold=True,
            align="center",
        )

    pdf.setStrokeColor(COLORS["grid"])
    pdf.setLineWidth(0.5)
    pdf.rect(x0, y, scale_width, height, stroke=1, fill=0)


def draw_legend_swatch(
    pdf: Canvas,
    x: float,
    y: float,
    color_name: str,
    *,
    hatched: bool = False,
) -> None:
    width = 10.0
    height = 7.0
    pdf.setFillColor(COLORS[color_name])
    pdf.setStrokeColor(COLORS["grid"])
    pdf.setLineWidth(0.35)
    pdf.rect(x, y, width, height, stroke=1, fill=1)
    if hatched:
        draw_hatch_overlay(pdf, x, y, width, height)


def draw_gpu_time_chart(
    pdf: Canvas,
    data: tuple[RunMetrics, RunMetrics],
) -> None:
    x0 = 52.0
    scale_width = 178.0
    for x, label, color_name in (
        (5.0, "Generation", "generation"),
        (65.0, "Idle resident", "idle_resident"),
        (142.0, "Loading", "loading"),
        (193.0, "Other", "other"),
    ):
        draw_legend_swatch(pdf, x, 82.0, color_name)
        draw_text(pdf, x + 13.0, 81.5, label, color=COLORS["muted"])

    draw_legend_swatch(pdf, 43.0, 65.0, "loading", hatched=True)
    draw_text(
        pdf,
        60.0,
        64.5,
        "Hatch = loading that could ideally start earlier",
        color=COLORS["muted"],
    )

    for tick in (0.0, 25.0, 50.0, 75.0, 100.0):
        tick_x = x0 + scale_width * tick / 100.0
        pdf.setStrokeColor(COLORS["grid"])
        pdf.setLineWidth(0.35)
        pdf.line(tick_x, 18.0, tick_x, 57.0)
        draw_text(
            pdf,
            tick_x,
            2.0,
            f"{tick:.0f}%",
            color=COLORS["muted"],
            align="center",
        )

    for metrics, y in zip(data, (43.0, 25.0), strict=True):
        draw_gpu_time_bar(
            pdf,
            metrics,
            x0=x0,
            scale_width=scale_width,
            y=y,
        )


def draw_figure(
    output: Path,
    data: tuple[RunMetrics, RunMetrics],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = Canvas(
        str(output),
        pagesize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        pageCompression=1,
        invariant=1,
        initialFontName=FONT_REGULAR,
    )
    pdf.setTitle("GPU time distribution under role-specific models")
    pdf.setCreator("scripts/workflow/plot_motivation_heterogeneity_residency.py")
    draw_gpu_time_chart(pdf, data)
    pdf.showPage()
    pdf.save()


def main() -> None:
    args = parse_args()
    register_fonts()
    draw_figure(args.output, load_figure_data(args.summary))


if __name__ == "__main__":
    main()
