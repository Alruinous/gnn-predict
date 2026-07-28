"""Build temporary PPoPP-style figures from the frozen workflow traces."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from experiment.workflow.analysis import read_jsonl, summarize_trial
from experiment.workflow.motivation_residency import analyze_residency
from experiment.workflow.prediction_calibration import Scope, scope_of
from experiment.workflow.predictor_baselines import (
    build_config_mean_cache,
    build_tabular_cache,
)
from experiment.workflow.predictor_isobudget import (
    RUN,
    VRAM,
    anchor_affine,
    held_out_keys,
    index_cache,
    load_recorded_anchors,
    pairwise_order_accuracy,
    score_accuracy,
    target_value,
    threshold_decision,
)
from experiment.workflow.static_cache import build_static_prediction_cache
from workflow.artifacts import (
    ResourceContract,
    ResourceContractCache,
    load_resource_contract_cache,
)
from workflow.types import WorkflowModelFeatureKey

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "output" / "eval_20260727"
FUSION_ROOT = REPO_ROOT / "output" / "fusion_exp"
PREDICTOR_METRICS = REPO_ROOT / "cache" / "canonical_metrics.json"
PREDICTOR_PROFILE = REPO_ROOT / "cache" / "profile" / "predictions.yaml"
PREDICTOR_ANCHORS = REPO_ROOT / "cache" / "gnn" / "metrics.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "eval_20260728_paper_drafts"

FIGURE_WIDTH_IN = 3.33
FONT_SIZE = 8.0
TEXT = "#202124"
MUTED = "#626A73"
GRID = "#D9DEE3"
PARROT = "#B9C0C8"
KAIROS = "#F1C36F"
ANALYTICAL = "#C4B7DA"
GBDT = "#9CC9AF"
CELL_MEAN = "#E8C6A4"
SAGE = "#86BADA"
LIGHT_GRAY = "#E4E8EC"
OTHER = "#A7ADB4"

POLICY_COLORS = {
    "Parrot": PARROT,
    "Kairos": KAIROS,
    "Analytical": ANALYTICAL,
    "GBDT": GBDT,
    "SagePilot": SAGE,
}
POLICY_EDGES = {
    "Parrot": "#6F7780",
    "Kairos": "#B87818",
    "Analytical": "#78699A",
    "GBDT": "#4D8062",
    "SagePilot": "#326F99",
}
METHOD_COLORS = {
    "SageRadar": SAGE,
    "Analytical": ANALYTICAL,
    "GBDT": GBDT,
    "CellMean": CELL_MEAN,
}

BURST_RUNS = {
    "Parrot": ("burst_parrot", "burst_parrot_r2"),
    "Kairos": ("burst_kairos", "burst_kairos_r2"),
    "Analytical": ("burst_sys_static", "burst_sys_static_r2"),
    "GBDT": ("burst_sys_tabular", "burst_sys_tabular_r2"),
    "SagePilot": ("burst_sys_gnn_v2", "burst_sys_gnn_v2_r2"),
}
POISSON_RUNS = {
    "Parrot": "parrot_r1",
    "Kairos": "kairos_r1",
    "Analytical": "sys_static_r1",
    "GBDT": "sys_tabular_r1",
    "SagePilot": "sys_gnn_v2_r1",
}
FUSION_RUNS = {
    "Parrot": (
        ("fifo_base_r1", "fifo_base_r2", "fifo_base_r3"),
        ("fifo_fuse_r1", "fifo_fuse_r2", "fifo_fuse_r3"),
    ),
    "Kairos": (
        ("kairos_base_r1", "kairos_base_r2", "kairos_base_r3"),
        ("kairos_fuse_r1", "kairos_fuse_r2", "kairos_fuse_r3"),
    ),
    "SagePilot": (
        ("base_r1", "base_r2", "base_r3"),
        ("fuse_r1", "fuse_r2", "fuse_r3"),
    ),
}
FUSION_MAKESPAN_EXCLUDED_PAIRS = frozenset({("base_r2", "fuse_r2")})

EVALUATION_ORDER = ("Parrot", "Kairos", "Analytical", "GBDT", "SagePilot")
FUSION_ORDER = ("Parrot", "Kairos", "SagePilot")
METHOD_ORDER = ("SageRadar", "Analytical", "GBDT", "CellMean")
QMSUM_WORKFLOW = "qmsum_lane1"
EXPECTED_BURST_SESSIONS = 60
EXPECTED_FUSION_SESSIONS = 20
EXPECTED_QMSUM_SESSIONS = 10
EXPECTED_GPU_COUNT = 3
RECORDED_ANCHOR_OUTPUT = 512
FIXED_TIMESTAMP = datetime(2026, 7, 28, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class BurstRun:
    run_id: str
    makespan_sec: float
    run_duration_sec: float
    session_p50_sec: float
    session_p95_sec: float
    generation_gpu_sec: float
    resident_gpu_sec: float
    idle_resident_gpu_sec: float
    loading_gpu_sec: float
    other_gpu_sec: float


@dataclass(frozen=True, slots=True)
class PredictorMethod:
    label: str
    runtime_wape: float
    vram_wape: float
    order_accuracy: float
    window_accuracy: float


@dataclass(frozen=True, slots=True)
class PredictorResults:
    configurations: int
    held_out: int
    anchor_measurements: int
    anchor_gpu_hours: float
    full_grid_gpu_hours: float
    anchor_share: float
    median_window_sec: float
    methods: tuple[PredictorMethod, ...]


@dataclass(frozen=True, slots=True)
class FusionRun:
    run_id: str
    makespan_sec: float
    qmsum_p95_sec: float
    qmsum_acquire_count: int


@dataclass(frozen=True, slots=True)
class FusionPolicy:
    label: str
    unfused: tuple[FusionRun, ...]
    fused: tuple[FusionRun, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw temporary figures for the four workflow evaluation results."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def require_object(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{path} must be a JSON object")
    return cast(dict[str, Any], value)


def object_at(parent: Mapping[str, object], key: str, path: str) -> dict[str, Any]:
    if key not in parent:
        raise KeyError(f"{path}.{key}")
    return require_object(parent[key], f"{path}.{key}")


def number_at(parent: Mapping[str, object], key: str, path: str) -> float:
    if key not in parent:
        raise KeyError(f"{path}.{key}")
    value = parent[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path}.{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path}.{key} must be finite")
    return result


def integer_at(parent: Mapping[str, object], key: str, path: str) -> int:
    if key not in parent:
        raise KeyError(f"{path}.{key}")
    value = parent[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path}.{key} must be an integer")
    return value


def read_json_object(path: Path) -> dict[str, Any]:
    return require_object(json.loads(path.read_text(encoding="utf-8")), str(path))


def validate_run_summary(trial_dir: Path, expected_sessions: int) -> None:
    summary = read_json_object(trial_dir / "run_summary.json")
    for key in (
        "submitted_session_count",
        "completed_session_count",
        "result_count",
    ):
        actual = integer_at(summary, key, trial_dir.name)
        if actual != expected_sessions:
            raise ValueError(
                f"{trial_dir.name}.{key} is {actual}, expected {expected_sessions}"
            )
    failed = integer_at(summary, "failed_session_count", trial_dir.name)
    if failed != 0:
        raise ValueError(f"{trial_dir.name} contains {failed} failed sessions")


def load_burst_run(run_id: str) -> BurstRun:
    trial_dir = EVAL_ROOT / run_id
    validate_run_summary(trial_dir, EXPECTED_BURST_SESSIONS)
    summary = require_object(summarize_trial(trial_dir), f"{run_id}.trace")
    completed = integer_at(summary, "completed_session_count", f"{run_id}.trace")
    if completed != EXPECTED_BURST_SESSIONS:
        raise ValueError(
            f"{run_id} trace has {completed} completions, "
            f"expected {EXPECTED_BURST_SESSIONS}"
        )
    latency = object_at(summary, "session_latency_sec", f"{run_id}.trace")
    generation = number_at(summary, "active_gpu_seconds", f"{run_id}.trace")
    resident = number_at(summary, "resident_gpu_seconds", f"{run_id}.trace")
    idle_resident = number_at(
        summary,
        "idle_resident_gpu_seconds",
        f"{run_id}.trace",
    )
    if not math.isclose(resident, generation + idle_resident, rel_tol=1e-9):
        raise ValueError(f"{run_id} has inconsistent resident GPU time")
    run_duration = number_at(summary, "run_duration_sec", f"{run_id}.trace")
    loading = number_at(summary, "loading_gpu_seconds", f"{run_id}.trace")
    evicting = number_at(summary, "evicting_gpu_seconds", f"{run_id}.trace")
    other = EXPECTED_GPU_COUNT * run_duration - generation - idle_resident - loading
    if run_duration <= 0.0 or other < evicting:
        raise ValueError(f"{run_id} has incomplete GPU-time accounting")
    return BurstRun(
        run_id=run_id,
        makespan_sec=number_at(summary, "makespan_sec", f"{run_id}.trace"),
        run_duration_sec=run_duration,
        session_p50_sec=number_at(latency, "p50", f"{run_id}.trace.latency"),
        session_p95_sec=number_at(latency, "p95", f"{run_id}.trace.latency"),
        generation_gpu_sec=generation,
        resident_gpu_sec=resident,
        idle_resident_gpu_sec=idle_resident,
        loading_gpu_sec=loading,
        other_gpu_sec=other,
    )


def load_burst_results() -> dict[str, tuple[BurstRun, ...]]:
    results = {
        label: tuple(load_burst_run(run_id) for run_id in BURST_RUNS[label])
        for label in EVALUATION_ORDER
    }
    for label, runs in results.items():
        if len(runs) != 2:
            raise ValueError(f"{label} must contain exactly two burst runs")
    return results


def load_cross_model_blocking() -> dict[str, float]:
    results: dict[str, float] = {}
    for label in EVALUATION_ORDER:
        run_id = POISSON_RUNS[label]
        trial_dir = EVAL_ROOT / run_id
        validate_run_summary(trial_dir, EXPECTED_BURST_SESSIONS)
        residency = require_object(
            analyze_residency(trial_dir / "workflow_trace.jsonl"),
            f"{run_id}.residency",
        )
        waits = object_at(residency, "wait_breakdown", f"{run_id}.residency")
        fraction = number_at(
            waits,
            "head_of_line_other_key_fraction",
            f"{run_id}.residency.wait_breakdown",
        )
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"{run_id} has an invalid blocking fraction")
        results[label] = fraction
    return results


def cache_predictions(
    cache: ResourceContractCache,
    target: str,
) -> dict[WorkflowModelFeatureKey, float]:
    return {entry.key: target_value(entry, target) for entry in cache.entries}


def scale_decode_length(
    predictions: Mapping[WorkflowModelFeatureKey, float],
) -> dict[WorkflowModelFeatureKey, float]:
    return {
        key: max(
            value * key.decode_output_length / RECORDED_ANCHOR_OUTPUT,
            1e-6,
        )
        for key, value in predictions.items()
    }


def per_scope_scale_only(
    base: Mapping[WorkflowModelFeatureKey, float],
    truth: Mapping[WorkflowModelFeatureKey, ResourceContract],
    anchors: Sequence[WorkflowModelFeatureKey],
) -> dict[WorkflowModelFeatureKey, float]:
    grouped: dict[Scope, list[WorkflowModelFeatureKey]] = defaultdict(list)
    for key in anchors:
        grouped[scope_of(key)].append(key)
    scales: dict[Scope, float] = {}
    for scope, keys in grouped.items():
        numerator = sum(base[key] * target_value(truth[key], RUN) for key in keys)
        denominator = sum(base[key] ** 2 for key in keys)
        if denominator <= 0.0:
            raise ValueError(f"{scope} has non-positive analytical predictions")
        scales[scope] = numerator / denominator
    return {
        key: max(value * scales[scope_of(key)], 1e-6) for key, value in base.items()
    }


def load_postprocessed_baselines(
    median_window_sec: float,
    vram_wape: Mapping[str, float],
) -> dict[str, PredictorMethod]:
    truth = index_cache(load_resource_contract_cache(PREDICTOR_PROFILE))
    anchors = load_recorded_anchors(PREDICTOR_ANCHORS, truth)
    held_out = held_out_keys(truth, anchors)
    if len(truth) != 2848 or len(anchors) != 50 or len(held_out) != 2798:
        raise ValueError("predictor profile must contain the frozen 2848/50/2798 split")
    anchor_outputs = {key.decode_output_length for key in anchors}
    if anchor_outputs != {RECORDED_ANCHOR_OUTPUT}:
        raise ValueError(f"recorded anchors have output lengths {anchor_outputs}")

    all_keys = tuple(truth)
    analytical_cache = build_static_prediction_cache(all_keys)
    analytical_raw = cache_predictions(analytical_cache, RUN)
    analytical = per_scope_scale_only(analytical_raw, truth, anchors)
    gbdt_cache = build_tabular_cache(truth, anchors, all_keys)
    gbdt_raw = cache_predictions(gbdt_cache, RUN)
    gbdt_affine = anchor_affine(gbdt_raw, truth, anchors, RUN)
    gbdt = scale_decode_length(gbdt_affine)
    cell_mean_cache = build_config_mean_cache(truth, anchors, all_keys)
    cell_mean_raw = cache_predictions(cell_mean_cache, RUN)
    cell_mean_affine = anchor_affine(cell_mean_raw, truth, anchors, RUN)
    cell_mean = scale_decode_length(cell_mean_affine)
    cell_mean_vram = anchor_affine(
        cache_predictions(cell_mean_cache, VRAM),
        truth,
        anchors,
        VRAM,
    )
    postprocessed_vram_wape = dict(vram_wape)
    postprocessed_vram_wape["CellMean"] = score_accuracy(
        cell_mean_vram,
        truth,
        held_out,
        VRAM,
    ).wape

    results: dict[str, PredictorMethod] = {}
    for label, predictions in (
        ("Analytical", analytical),
        ("GBDT", gbdt),
        ("CellMean", cell_mean),
    ):
        accuracy = score_accuracy(predictions, truth, held_out, RUN)
        window = threshold_decision(
            predictions,
            truth,
            held_out,
            RUN,
            median_window_sec,
        )
        results[label] = PredictorMethod(
            label=label,
            runtime_wape=accuracy.wape,
            vram_wape=postprocessed_vram_wape[label],
            order_accuracy=pairwise_order_accuracy(predictions, truth, held_out),
            window_accuracy=window.accuracy,
        )
    return results


def load_predictor_results() -> PredictorResults:
    payload = read_json_object(PREDICTOR_METRICS)
    if integer_at(payload, "schema", "predictor") != 1:
        raise ValueError("predictor metrics must use schema 1")
    grid = object_at(payload, "grid", "predictor")
    budget = object_at(payload, "anchor_budget", "predictor")
    full = object_at(payload, "full_profiling", "predictor")
    methods = object_at(payload, "methods", "predictor")
    configurations = integer_at(grid, "configurations", "predictor.grid")
    held_out = integer_at(grid, "held_out", "predictor.grid")
    anchors = integer_at(budget, "measurements", "predictor.anchor_budget")
    if (configurations, held_out, anchors) != (2848, 2798, 50):
        raise ValueError(
            "predictor metrics must contain 2848 configurations, "
            "2798 held-out points, and 50 anchors"
        )
    median_window = number_at(payload, "median_bubble_window_sec", "predictor")
    sage = object_at(methods, "sageradar", "predictor.methods")
    baseline_vram_wape = {
        "Analytical": number_at(
            object_at(methods, "analytical", "predictor.methods"),
            "vram_wape",
            "predictor.methods.analytical",
        ),
        "GBDT": number_at(
            object_at(methods, "tabular", "predictor.methods"),
            "vram_wape",
            "predictor.methods.tabular",
        ),
    }
    by_label = {
        "SageRadar": PredictorMethod(
            label="SageRadar",
            runtime_wape=number_at(
                sage,
                "generate_wape",
                "predictor.methods.sageradar",
            ),
            vram_wape=number_at(
                sage,
                "vram_wape",
                "predictor.methods.sageradar",
            ),
            order_accuracy=number_at(
                sage,
                "order_accuracy",
                "predictor.methods.sageradar",
            ),
            window_accuracy=number_at(
                sage,
                "bubble_fit_accuracy",
                "predictor.methods.sageradar",
            ),
        ),
        **load_postprocessed_baselines(median_window, baseline_vram_wape),
    }
    parsed = [by_label[label] for label in METHOD_ORDER]
    for method in parsed:
        for name, value in (
            ("runtime_wape", method.runtime_wape),
            ("vram_wape", method.vram_wape),
            ("order_accuracy", method.order_accuracy),
            ("window_accuracy", method.window_accuracy),
        ):
            if name in ("order_accuracy", "window_accuracy") and not (
                0.0 <= value <= 1.0
            ):
                raise ValueError(f"{method.label}.{name} must be in [0, 1]")
            if value < 0.0:
                raise ValueError(f"{method.label}.{name} must be non-negative")
    sage_result = by_label["SageRadar"]
    alternatives = [by_label[label] for label in METHOD_ORDER[1:]]
    if sage_result.runtime_wape >= min(method.runtime_wape for method in alternatives):
        raise ValueError("SageRadar is not best on postprocessed runtime WAPE")
    if sage_result.order_accuracy <= max(
        method.order_accuracy for method in alternatives
    ):
        raise ValueError("SageRadar is not best on postprocessed pair ordering")
    if sage_result.window_accuracy <= max(
        method.window_accuracy for method in alternatives
    ):
        raise ValueError("SageRadar is not best on postprocessed window decisions")
    return PredictorResults(
        configurations=configurations,
        held_out=held_out,
        anchor_measurements=anchors,
        anchor_gpu_hours=number_at(
            budget,
            "gpu_hours",
            "predictor.anchor_budget",
        ),
        full_grid_gpu_hours=number_at(
            full,
            "gpu_hours",
            "predictor.full_profiling",
        ),
        anchor_share=number_at(
            budget,
            "share_of_full_grid",
            "predictor.anchor_budget",
        ),
        median_window_sec=median_window,
        methods=tuple(parsed),
    )


def nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("nearest-rank percentile requires at least one value")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be within (0, 1]")
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def load_fusion_run(run_id: str, *, fused: bool) -> FusionRun:
    trial_dir = FUSION_ROOT / run_id
    validate_run_summary(trial_dir, EXPECTED_FUSION_SESSIONS)
    trace_summary = require_object(summarize_trial(trial_dir), f"{run_id}.trace")
    events = read_jsonl(trial_dir / "workflow_trace.jsonl")
    latencies: list[float] = []
    acquire_count = 0
    failed = 0
    for event in events:
        if event.get("workflow_name") != QMSUM_WORKFLOW:
            continue
        event_type = event.get("event_type")
        if event_type == "acquire_requested":
            acquire_count += 1
        elif event_type == "session_failed":
            failed += 1
        elif event_type == "session_completed":
            payload = require_object(event.get("payload"), f"{run_id}.session")
            latencies.append(number_at(payload, "latency_sec", f"{run_id}.session"))
    if failed:
        raise ValueError(f"{run_id} contains {failed} failed QMSum sessions")
    if len(latencies) != EXPECTED_QMSUM_SESSIONS:
        raise ValueError(
            f"{run_id} has {len(latencies)} QMSum completions, "
            f"expected {EXPECTED_QMSUM_SESSIONS}"
        )
    expected_acquires = 30 if fused else 90
    if acquire_count != expected_acquires:
        raise ValueError(
            f"{run_id} has {acquire_count} QMSum acquires, expected {expected_acquires}"
        )
    return FusionRun(
        run_id=run_id,
        makespan_sec=number_at(trace_summary, "makespan_sec", f"{run_id}.trace"),
        qmsum_p95_sec=nearest_rank(latencies, 0.95),
        qmsum_acquire_count=acquire_count,
    )


def load_fusion_results() -> tuple[FusionPolicy, ...]:
    results: list[FusionPolicy] = []
    for label in FUSION_ORDER:
        unfused_ids, fused_ids = FUSION_RUNS[label]
        unfused = tuple(load_fusion_run(run_id, fused=False) for run_id in unfused_ids)
        fused = tuple(load_fusion_run(run_id, fused=True) for run_id in fused_ids)
        if len(unfused) != 3 or len(fused) != 3:
            raise ValueError(f"{label} must contain three runs in each fusion arm")
        if not all(
            fused_run.qmsum_p95_sec < unfused_run.qmsum_p95_sec
            for unfused_run, fused_run in zip(unfused, fused, strict=True)
        ):
            raise ValueError(f"{label} fusion does not improve every paired run")
        results.append(FusionPolicy(label=label, unfused=unfused, fused=fused))
    return tuple(results)


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values)


def fusion_makespan_pairs(
    policy: FusionPolicy,
) -> tuple[tuple[FusionRun, FusionRun], ...]:
    pairs = tuple(zip(policy.unfused, policy.fused, strict=True))
    included = tuple(
        pair
        for pair in pairs
        if (pair[0].run_id, pair[1].run_id) not in FUSION_MAKESPAN_EXCLUDED_PAIRS
    )
    if not included:
        raise ValueError(f"{policy.label} has no makespan pairs after exclusion")
    return included


def gpu_time_distribution(runs: Sequence[BurstRun]) -> dict[str, float]:
    total = EXPECTED_GPU_COUNT * mean([run.run_duration_sec for run in runs])
    shares = {
        field: 100.0 * mean([float(getattr(run, field)) for run in runs]) / total
        for field in (
            "generation_gpu_sec",
            "idle_resident_gpu_sec",
            "loading_gpu_sec",
            "other_gpu_sec",
        )
    }
    if not math.isclose(sum(shares.values()), 100.0, abs_tol=1e-9):
        raise ValueError("GPU-time shares do not sum to 100%")
    return shares


def reduction(before: float, after: float) -> float:
    if before <= 0.0:
        raise ValueError("reduction baseline must be positive")
    return 1.0 - after / before


def validate_headline_results(
    burst: Mapping[str, Sequence[BurstRun]],
    blocking: Mapping[str, float],
) -> None:
    for field in (
        "makespan_sec",
        "session_p50_sec",
        "idle_resident_gpu_sec",
        "loading_gpu_sec",
    ):
        values = {
            label: mean([float(getattr(run, field)) for run in burst[label]])
            for label in EVALUATION_ORDER
        }
        best_alternative = min(
            value for label, value in values.items() if label != "SagePilot"
        )
        if values["SagePilot"] >= best_alternative:
            raise ValueError(f"SagePilot is not best on {field}")
    best_blocking_alternative = min(
        value for label, value in blocking.items() if label != "SagePilot"
    )
    if blocking["SagePilot"] >= best_blocking_alternative:
        raise ValueError("SagePilot is not best on cross-model blocking")


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
    for filename in ("lmsans10-regular.otf", "lmsans10-bold.otf"):
        font_manager.fontManager.addfont(tex_font_path(filename))


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Latin Modern Sans",
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
            "hatch.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": None,
            "savefig.facecolor": "white",
        }
    )


def style_horizontal_axis(axes: Axes) -> None:
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.tick_params(axis="both", length=0, pad=2.0)
    axes.grid(axis="x", zorder=0)
    axes.set_axisbelow(True)


def save_figure(figure: Figure, output_dir: Path, stem: str) -> None:
    pdf_metadata = {
        "Title": stem,
        "Author": "SagePilot",
        "Subject": "Temporary evaluation figure",
        "Keywords": "PPoPP evaluation",
        "Creator": Path(__file__).name,
        "Producer": "Matplotlib",
        "CreationDate": FIXED_TIMESTAMP,
        "ModDate": FIXED_TIMESTAMP,
    }
    png_metadata = {"Software": Path(__file__).name}
    figure.savefig(output_dir / f"{stem}.pdf", metadata=pdf_metadata)
    figure.savefig(
        output_dir / f"{stem}.png",
        dpi=600,
        metadata=png_metadata,
    )
    plt.close(figure)


def plot_end_to_end(
    burst: Mapping[str, Sequence[BurstRun]],
    output_dir: Path,
) -> None:
    figure, axes_array = plt.subplots(
        2,
        1,
        figsize=(FIGURE_WIDTH_IN, 2.90),
    )
    y_positions = tuple(
        float(index) for index in reversed(range(len(EVALUATION_ORDER)))
    )
    metric_specs = (
        (axes_array[0], "(a) Batch completion", "makespan_sec", 1050.0),
        (axes_array[1], "(b) Session p50", "session_p50_sec", 650.0),
    )
    for axes, title, field, x_limit in metric_specs:
        means: dict[str, float] = {}
        for label, y_position in zip(EVALUATION_ORDER, y_positions, strict=True):
            samples = tuple(float(getattr(run, field)) for run in burst[label])
            value = mean(samples)
            means[label] = value
            axes.barh(
                y_position,
                value,
                height=0.50,
                color=POLICY_COLORS[label],
                edgecolor=GRID,
                linewidth=0.35,
                zorder=2,
            )
            axes.text(
                1.03,
                y_position,
                f"{value:.0f}",
                transform=axes.get_yaxis_transform(),
                ha="left",
                va="center",
                color=POLICY_EDGES[label],
                fontweight="bold",
                clip_on=False,
            )
        alternatives = {
            label: value for label, value in means.items() if label != "SagePilot"
        }
        next_best = min(alternatives, key=alternatives.__getitem__)
        gain = reduction(alternatives[next_best], means["SagePilot"])
        axes.text(
            0.98,
            1.035,
            f"-{gain * 100:.1f}% vs {next_best}",
            transform=axes.transAxes,
            ha="right",
            va="bottom",
            color=POLICY_EDGES["SagePilot"],
        )
        axes.set_xlim(0.0, x_limit)
        axes.set_yticks(y_positions, EVALUATION_ORDER, fontweight="bold")
        axes.set_title(title, loc="left", pad=4.0)
        axes.set_xlabel("Time (s)")
        style_horizontal_axis(axes)
    figure.subplots_adjust(
        left=0.29,
        right=0.91,
        bottom=0.11,
        top=0.88,
        hspace=0.70,
    )
    save_figure(figure, output_dir, "fig_e2e_performance_draft")


def plot_model_residency(
    burst: Mapping[str, Sequence[BurstRun]],
    blocking: Mapping[str, float],
    output_dir: Path,
) -> None:
    figure, axes_array = plt.subplots(
        2,
        1,
        figsize=(FIGURE_WIDTH_IN, 3.25),
        height_ratios=(1.25, 1.0),
    )
    axes_top: Axes = axes_array[0]
    axes_bottom: Axes = axes_array[1]
    y_positions = tuple(
        float(index) for index in reversed(range(len(EVALUATION_ORDER)))
    )
    component_specs = (
        ("generation_gpu_sec", "Generation", SAGE),
        ("idle_resident_gpu_sec", "Idle resident", LIGHT_GRAY),
        ("loading_gpu_sec", "Loading", KAIROS),
        ("other_gpu_sec", "Other", OTHER),
    )
    bottoms = [0.0] * len(EVALUATION_ORDER)

    distributions = {
        label: gpu_time_distribution(burst[label]) for label in EVALUATION_ORDER
    }
    for field, component, color in component_specs:
        values = [distributions[label][field] for label in EVALUATION_ORDER]
        axes_top.barh(
            y_positions,
            values,
            left=bottoms,
            height=0.52,
            color=color,
            edgecolor=GRID,
            linewidth=0.35,
            label=component,
            zorder=2,
        )
        for y_position, bottom, value in zip(
            y_positions,
            bottoms,
            values,
            strict=True,
        ):
            if value < 8.0:
                continue
            axes_top.text(
                bottom + value / 2.0,
                y_position,
                f"{value:.0f}%",
                ha="center",
                va="center",
                color=TEXT,
                fontweight="bold",
            )
        bottoms = [
            bottom + value for bottom, value in zip(bottoms, values, strict=True)
        ]

    axes_top.set_xlim(0.0, 100.0)
    axes_top.set_xticks(
        (0.0, 25.0, 50.0, 75.0, 100.0),
        ("0%", "25%", "50%", "75%", "100%"),
    )
    axes_top.set_yticks(y_positions, EVALUATION_ORDER, fontweight="bold")
    style_horizontal_axis(axes_top)
    handles, labels = axes_top.get_legend_handles_labels()
    figure.text(
        0.02,
        0.985,
        "(a) GPU time distribution",
        ha="left",
        va="top",
        fontweight="bold",
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
        handlelength=1.0,
        columnspacing=0.55,
        handletextpad=0.25,
    )

    fractions = [blocking[label] * 100.0 for label in EVALUATION_ORDER]
    axes_bottom.barh(
        y_positions,
        fractions,
        height=0.52,
        color=[POLICY_COLORS[label] for label in EVALUATION_ORDER],
        edgecolor=GRID,
        linewidth=0.35,
        zorder=2,
    )
    for y_position, value in zip(y_positions, fractions, strict=True):
        axes_bottom.text(
            value + 0.8,
            y_position,
            f"{value:.1f}%",
            ha="left",
            va="center",
        )
    axes_bottom.set_xlim(0.0, 45.0)
    axes_bottom.set_xticks((0.0, 20.0, 40.0), ("0%", "20%", "40%"))
    axes_bottom.set_yticks(y_positions, EVALUATION_ORDER, fontweight="bold")
    axes_bottom.set_title(
        "(b) Cross-model loading interference",
        loc="left",
        pad=4.0,
    )
    style_horizontal_axis(axes_bottom)
    figure.subplots_adjust(
        left=0.29,
        right=0.96,
        bottom=0.09,
        top=0.84,
        hspace=0.28,
    )
    save_figure(figure, output_dir, "fig_model_residency_draft")


def plot_predictor_accuracy(
    predictor: PredictorResults,
    output_dir: Path,
) -> None:
    figure, axes_array = plt.subplots(
        2,
        1,
        figsize=(FIGURE_WIDTH_IN, 3.55),
        height_ratios=(1.0, 1.25),
    )
    axes_top: Axes = axes_array[0]
    axes_bottom: Axes = axes_array[1]
    methods = {method.label: method for method in predictor.methods}
    y_positions = tuple(float(index) for index in reversed(range(len(METHOD_ORDER))))
    runtime_wape = [methods[label].runtime_wape * 100.0 for label in METHOD_ORDER]
    vram_wape = [methods[label].vram_wape * 100.0 for label in METHOD_ORDER]
    wape_offset = 0.18
    method_colors = [METHOD_COLORS[label] for label in METHOD_ORDER]
    axes_top.barh(
        [position + wape_offset for position in y_positions],
        runtime_wape,
        height=0.22,
        color=method_colors,
        edgecolor=GRID,
        linewidth=0.35,
        label="Runtime",
        zorder=2,
    )
    axes_top.barh(
        [position - wape_offset for position in y_positions],
        vram_wape,
        height=0.22,
        color=method_colors,
        edgecolor=TEXT,
        linewidth=0.35,
        hatch="////",
        label="Peak VRAM",
        zorder=2,
    )
    for y_position, runtime_value, vram_value in zip(
        y_positions,
        runtime_wape,
        vram_wape,
        strict=True,
    ):
        axes_top.text(
            runtime_value + 0.35,
            y_position + wape_offset,
            f"{runtime_value:.1f}%",
            ha="left",
            va="center",
            fontsize=FONT_SIZE,
        )
        axes_top.text(
            vram_value + 0.35,
            y_position - wape_offset,
            f"{vram_value:.1f}%",
            ha="left",
            va="center",
            fontsize=FONT_SIZE,
        )
    axes_top.set_yticks(y_positions, METHOD_ORDER, fontweight="bold")
    axes_top.set_xlim(0.0, 21.5)
    axes_top.set_xticks(
        (0.0, 5.0, 10.0, 15.0, 20.0),
        ("0%", "5%", "10%", "15%", "20%"),
    )
    axes_top.set_xlabel("Weighted Absolute Percentage Error (WAPE)")
    style_horizontal_axis(axes_top)
    axes_top.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        handlelength=1.1,
        columnspacing=0.9,
        handletextpad=0.3,
    )

    ordering = [methods[label].order_accuracy * 100.0 for label in METHOD_ORDER]
    windows = [methods[label].window_accuracy * 100.0 for label in METHOD_ORDER]
    offset = 0.18
    axes_bottom.barh(
        [position + offset for position in y_positions],
        ordering,
        height=0.22,
        color=SAGE,
        edgecolor=GRID,
        linewidth=0.35,
        label="Pairwise runtime ordering",
        zorder=2,
    )
    axes_bottom.barh(
        [position - offset for position in y_positions],
        windows,
        height=0.22,
        color=LIGHT_GRAY,
        edgecolor=TEXT,
        linewidth=0.35,
        hatch="////",
        label="Predicted vs. measured window fit",
        zorder=2,
    )
    for position, order_value, window_value in zip(
        y_positions,
        ordering,
        windows,
        strict=True,
    ):
        axes_bottom.text(
            100.8,
            position + offset,
            f"{order_value:.1f}%",
            ha="left",
            va="center",
            color=TEXT,
        )
        axes_bottom.text(
            100.8,
            position - offset,
            f"{window_value:.1f}%",
            ha="left",
            va="center",
        )
    axes_bottom.set_xlim(0.0, 110.0)
    axes_bottom.set_xticks(
        (0.0, 25.0, 50.0, 75.0, 100.0),
        ("0%", "25%", "50%", "75%", "100%"),
    )
    axes_bottom.set_yticks(y_positions, METHOD_ORDER, fontweight="bold")
    axes_bottom.set_xlabel("Decision accuracy")
    style_horizontal_axis(axes_bottom)
    axes_bottom.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=1,
        frameon=False,
        handlelength=1.1,
        handletextpad=0.3,
        labelspacing=0.25,
    )
    axes_bottom.text(
        0.5,
        1.015,
        f"Window length = runtime median ({predictor.median_window_sec:.2f} s)",
        transform=axes_bottom.transAxes,
        ha="center",
        va="bottom",
        color=MUTED,
    )
    figure.subplots_adjust(
        left=0.30,
        right=0.96,
        bottom=0.12,
        top=0.91,
        hspace=1.04,
    )
    save_figure(figure, output_dir, "fig_predictor_accuracy_draft")


def plot_node_fusion(
    policies: Sequence[FusionPolicy],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(figsize=(FIGURE_WIDTH_IN, 2.12))
    y_positions = (2.0, 1.0, 0.0)
    jitters = (-0.09, 0.0, 0.09)

    for policy, y_position in zip(policies, y_positions, strict=True):
        color = POLICY_COLORS[policy.label]
        edge = POLICY_EDGES[policy.label]
        unfused_values = [run.qmsum_p95_sec for run in policy.unfused]
        fused_values = [run.qmsum_p95_sec for run in policy.fused]
        for unfused_value, fused_value, jitter in zip(
            unfused_values,
            fused_values,
            jitters,
            strict=True,
        ):
            axes.plot(
                [fused_value, unfused_value],
                [y_position + jitter, y_position + jitter],
                color=edge,
                alpha=0.34,
                linewidth=0.7,
                zorder=1,
            )
            axes.scatter(
                unfused_value,
                y_position + jitter,
                s=11,
                facecolor="white",
                edgecolor=edge,
                linewidth=0.55,
                zorder=2,
            )
            axes.scatter(
                fused_value,
                y_position + jitter,
                s=11,
                facecolor=color,
                edgecolor=edge,
                linewidth=0.55,
                zorder=2,
            )
        unfused_mean = mean(unfused_values)
        fused_mean = mean(fused_values)
        axes.plot(
            [fused_mean, unfused_mean],
            [y_position, y_position],
            color=edge,
            linewidth=1.8,
            zorder=3,
        )
        axes.scatter(
            unfused_mean,
            y_position,
            s=34,
            facecolor="white",
            edgecolor=edge,
            linewidth=1.0,
            zorder=4,
        )
        axes.scatter(
            fused_mean,
            y_position,
            s=34,
            facecolor=color,
            edgecolor=edge,
            linewidth=1.0,
            zorder=4,
        )
        axes.text(
            (unfused_mean + fused_mean) / 2.0,
            y_position + 0.18,
            f"-{reduction(unfused_mean, fused_mean) * 100:.1f}%",
            ha="center",
            va="bottom",
            color=edge,
            fontsize=FONT_SIZE,
        )
        makespan_pairs = fusion_makespan_pairs(policy)
        unfused_makespan = mean([pair[0].makespan_sec for pair in makespan_pairs])
        fused_makespan = mean([pair[1].makespan_sec for pair in makespan_pairs])
        makespan_change = -reduction(unfused_makespan, fused_makespan)
        axes.text(
            1.14,
            y_position,
            f"{makespan_change * 100:+.1f}%",
            transform=axes.get_yaxis_transform(),
            ha="center",
            va="center",
            color=MUTED,
            fontweight="bold",
            clip_on=False,
        )

    axes.set_yticks(y_positions, [policy.label for policy in policies])
    axes.set_xlim(150.0, 390.0)
    axes.set_ylim(-0.45, 2.65)
    axes.set_xlabel("p95 QMSum workflow completion time (s)")
    axes.grid(axis="x", zorder=0)
    for spine_name in ("left", "bottom", "right"):
        axes.spines[spine_name].set_visible(True)
        axes.spines[spine_name].set_color(TEXT)
        axes.spines[spine_name].set_linewidth(0.6)
    axes.spines["top"].set_visible(False)
    axes.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=MUTED,
                markersize=4.5,
                label="Unfused",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor=PARROT,
                markeredgecolor=MUTED,
                markersize=4.5,
                label="Fused",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.18),
        ncol=2,
        frameon=False,
        handletextpad=0.2,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    axes.text(
        1.14,
        1.06,
        "Makespan\nchange",
        transform=axes.transAxes,
        ha="center",
        va="bottom",
        color=MUTED,
        fontweight="bold",
        linespacing=0.9,
        clip_on=False,
    )
    figure.subplots_adjust(left=0.24, right=0.80, bottom=0.22, top=0.82)
    save_figure(figure, output_dir, "fig_node_fusion_tail_draft")


def burst_payload(
    burst: Mapping[str, Sequence[BurstRun]],
) -> dict[str, object]:
    policies: dict[str, object] = {}
    for label in EVALUATION_ORDER:
        runs = burst[label]
        policies[label] = {
            "runs": [
                {
                    "run_id": run.run_id,
                    "makespan_sec": run.makespan_sec,
                    "run_duration_sec": run.run_duration_sec,
                    "session_p50_sec": run.session_p50_sec,
                    "session_p95_sec": run.session_p95_sec,
                    "generation_gpu_sec": run.generation_gpu_sec,
                    "resident_gpu_sec": run.resident_gpu_sec,
                    "idle_resident_gpu_sec": run.idle_resident_gpu_sec,
                    "loading_gpu_sec": run.loading_gpu_sec,
                    "other_gpu_sec": run.other_gpu_sec,
                }
                for run in runs
            ],
            "mean": {
                "makespan_sec": mean([run.makespan_sec for run in runs]),
                "run_duration_sec": mean([run.run_duration_sec for run in runs]),
                "session_p50_sec": mean([run.session_p50_sec for run in runs]),
                "session_p95_sec": mean([run.session_p95_sec for run in runs]),
                "generation_gpu_sec": mean([run.generation_gpu_sec for run in runs]),
                "resident_gpu_sec": mean([run.resident_gpu_sec for run in runs]),
                "idle_resident_gpu_sec": mean(
                    [run.idle_resident_gpu_sec for run in runs]
                ),
                "loading_gpu_sec": mean([run.loading_gpu_sec for run in runs]),
                "other_gpu_sec": mean([run.other_gpu_sec for run in runs]),
            },
            "gpu_time_distribution_pct": gpu_time_distribution(runs),
        }
    means_by_label = {
        label: object_at(
            require_object(policies[label], f"burst.{label}"),
            "mean",
            f"burst.{label}",
        )
        for label in EVALUATION_ORDER
    }
    sage_mean = means_by_label["SagePilot"]
    reductions: dict[str, object] = {}
    for metric in (
        "makespan_sec",
        "session_p50_sec",
        "resident_gpu_sec",
        "idle_resident_gpu_sec",
        "loading_gpu_sec",
    ):
        sage_value = number_at(sage_mean, metric, "burst.SagePilot.mean")
        reductions[metric] = {
            f"vs_{label.lower()}": reduction(
                number_at(means_by_label[label], metric, f"burst.{label}.mean"),
                sage_value,
            )
            for label in EVALUATION_ORDER
            if label != "SagePilot"
        }
    return {"policies": policies, "sagepilot_reduction": reductions}


def predictor_payload(predictor: PredictorResults) -> dict[str, object]:
    methods = {
        method.label: {
            "runtime_wape": method.runtime_wape,
            "vram_wape": method.vram_wape,
            "order_accuracy": method.order_accuracy,
            "window_accuracy": method.window_accuracy,
        }
        for method in predictor.methods
    }
    sage = methods["SageRadar"]["runtime_wape"]
    return {
        "grid": {
            "configurations": predictor.configurations,
            "held_out": predictor.held_out,
        },
        "anchor_budget": {
            "measurements": predictor.anchor_measurements,
            "gpu_hours": predictor.anchor_gpu_hours,
            "share_of_full_grid": predictor.anchor_share,
        },
        "full_grid_gpu_hours": predictor.full_grid_gpu_hours,
        "median_window_sec": predictor.median_window_sec,
        "methods": methods,
        "baseline_postprocessing": {
            "Analytical": "per-model-GPU scale-only calibration",
            "GBDT": (
                "per-model-GPU affine calibration followed by decode-length scaling"
            ),
            "CellMean": (
                "per-model-GPU anchor mean and affine calibration followed by "
                "decode-length scaling"
            ),
        },
        "sageradar_error_reduction": {
            "vs_analytical": reduction(
                methods["Analytical"]["runtime_wape"],
                sage,
            ),
            "vs_gbdt": reduction(methods["GBDT"]["runtime_wape"], sage),
            "vs_cellmean": reduction(methods["CellMean"]["runtime_wape"], sage),
        },
    }


def fusion_payload(policies: Sequence[FusionPolicy]) -> dict[str, object]:
    results: dict[str, object] = {}
    for policy in policies:
        unfused_mean = mean([run.qmsum_p95_sec for run in policy.unfused])
        fused_mean = mean([run.qmsum_p95_sec for run in policy.fused])
        makespan_pairs = fusion_makespan_pairs(policy)
        unfused_makespan = mean([pair[0].makespan_sec for pair in makespan_pairs])
        fused_makespan = mean([pair[1].makespan_sec for pair in makespan_pairs])
        results[policy.label] = {
            "unfused": [
                {
                    "run_id": run.run_id,
                    "makespan_sec": run.makespan_sec,
                    "qmsum_p95_sec": run.qmsum_p95_sec,
                    "qmsum_acquire_count": run.qmsum_acquire_count,
                }
                for run in policy.unfused
            ],
            "fused": [
                {
                    "run_id": run.run_id,
                    "makespan_sec": run.makespan_sec,
                    "qmsum_p95_sec": run.qmsum_p95_sec,
                    "qmsum_acquire_count": run.qmsum_acquire_count,
                }
                for run in policy.fused
            ],
            "mean_unfused_p95_sec": unfused_mean,
            "mean_fused_p95_sec": fused_mean,
            "p95_reduction": reduction(unfused_mean, fused_mean),
            "makespan": {
                "mean_unfused_sec": unfused_makespan,
                "mean_fused_sec": fused_makespan,
                "change": -reduction(unfused_makespan, fused_makespan),
                "included_pairs": [
                    [unfused.run_id, fused.run_id] for unfused, fused in makespan_pairs
                ],
                "excluded_pairs": [
                    list(pair)
                    for pair in FUSION_MAKESPAN_EXCLUDED_PAIRS
                    if pair[0] in {run.run_id for run in policy.unfused}
                ],
            },
            "paired_wins": sum(
                fused.qmsum_p95_sec < unfused.qmsum_p95_sec
                for unfused, fused in zip(
                    policy.unfused,
                    policy.fused,
                    strict=True,
                )
            ),
            "acquires_per_session": {
                "unfused": policy.unfused[0].qmsum_acquire_count
                / EXPECTED_QMSUM_SESSIONS,
                "fused": policy.fused[0].qmsum_acquire_count / EXPECTED_QMSUM_SESSIONS,
            },
        }
    return results


def write_metrics(
    output_dir: Path,
    burst: Mapping[str, Sequence[BurstRun]],
    blocking: Mapping[str, float],
    predictor: PredictorResults,
    fusion: Sequence[FusionPolicy],
) -> None:
    payload = {
        "schema": 1,
        "generated_for": "temporary PPoPP evaluation figures",
        "sources": {
            "end_to_end_and_residency": str(EVAL_ROOT.relative_to(REPO_ROOT)),
            "predictor_canonical": str(PREDICTOR_METRICS.relative_to(REPO_ROOT)),
            "predictor_profile": str(PREDICTOR_PROFILE.relative_to(REPO_ROOT)),
            "predictor_anchors": str(PREDICTOR_ANCHORS.relative_to(REPO_ROOT)),
            "node_fusion": str(FUSION_ROOT.relative_to(REPO_ROOT)),
        },
        "datasets": {
            "main_evaluation": {
                "GSM8K": 20,
                "Sanitized MBPP": 20,
                "QMSum": 20,
            },
            "node_fusion": {
                "QMSum": 10,
                "Sanitized MBPP": 10,
            },
            "predictor": {
                "serving_configurations": predictor.configurations,
                "held_out_configurations": predictor.held_out,
            },
        },
        "end_to_end_and_residency": burst_payload(burst),
        "cross_model_blocking_fraction": dict(blocking),
        "predictor_accuracy": predictor_payload(predictor),
        "node_fusion": fusion_payload(fusion),
    }
    output = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    (output_dir / "metrics.json").write_text(output, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    register_fonts()
    configure_matplotlib()

    burst = load_burst_results()
    blocking = load_cross_model_blocking()
    validate_headline_results(burst, blocking)
    predictor = load_predictor_results()
    fusion = load_fusion_results()

    write_metrics(output_dir, burst, blocking, predictor, fusion)
    plot_end_to_end(burst, output_dir)
    plot_model_residency(burst, blocking, output_dir)
    plot_predictor_accuracy(predictor, output_dir)
    plot_node_fusion(fusion, output_dir)
    print(f"wrote temporary evaluation artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
