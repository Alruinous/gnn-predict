"""Score SageRadar against the paper's predictor baselines at one profiling budget.

Every lookup/learned method sees the same real measurements; the analytical
formula sees none; the profile cache is the upper bound. Emits
output/predictor_isobudget/{tables.md, accuracy.json, runtime_wape.csv,
decisions.csv, bound_quality.csv, pareto.csv}.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment.workflow.predictor_isobudget import (  # noqa: E402
    HELD_OUT,
    LONG_DECODE,
    RUN,
    VRAM,
    ProfilingCost,
    SchemeResult,
    baseline_marginal_seconds,
    evaluate_scheme,
    graph_forward_seconds,
    index_cache,
    load_recorded_anchors,
    ordered_methods,
    profiling_cost,
    spread_anchors,
)
from workflow.artifacts import load_resource_contract_cache  # noqa: E402

Results = Mapping[str, SchemeResult]
LABEL = {
    "sageradar": "SageRadar (ours)",
    "analytical_cal": "Analytical+cal",
    "analytical": "Analytical",
    "nearest_profile": "Nearest-profile",
    "config_mean": "Config-mean",
    "tabular": "Tabular-GBDT",
    "anchor_scaled": "Anchor-scaled (excluded)",
}


def _primary(results: Results) -> SchemeResult:
    return results.get("recorded") or next(iter(results.values()))


def _accuracy_rows(result: SchemeResult, scheme: str) -> list[str]:
    lines = [
        "| method | held-out run WAPE | long-decode run WAPE | p95 rel err "
        "| pair-order | long-decode pair-order |",
        "|---|---|---|---|---|---|",
        "| Profile (upper bound) | 0.0000 | 0.0000 | 0.000 | 1.0000 | 1.0000 |",
    ]
    for name, per_stratum in result.accuracy.items():
        held = per_stratum[HELD_OUT][RUN]
        long_decode = per_stratum[LONG_DECODE][RUN]
        lines.append(
            f"| {LABEL[name]} | {held.wape:.4f} | {long_decode.wape:.4f} "
            f"| {long_decode.p95_relative_error:.3f} "
            f"| {result.ordering[name][HELD_OUT]:.4f} "
            f"| {result.ordering[name][LONG_DECODE]:.4f} |"
        )
    return [f"### {scheme} anchors — decode run time", "", *lines, ""]


def _vram_rows(result: SchemeResult, scheme: str) -> list[str]:
    lines = [
        "| method | vram WAPE | under rate | max under (GB) | bound violation "
        "| mean over-reservation | p95 over-reservation |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, per_stratum in result.accuracy.items():
        held = per_stratum[HELD_OUT][VRAM]
        bound = result.bounds[name]
        lines.append(
            f"| {LABEL[name]} | {held.wape:.4f} | {held.underestimate_rate:.3f} "
            f"| {held.max_underestimate / 1024:.2f} "
            f"| {bound.violation_rate:.3f} "
            f"| {100 * bound.mean_over_reservation:.1f}% "
            f"| {100 * bound.p95_over_reservation:.1f}% |"
        )
    return [f"### {scheme} anchors — peak VRAM and admission bound", "", *lines, ""]


def _decision_rows(result: SchemeResult, scheme: str) -> list[str]:
    windows = result.run_windows_sec
    capacities = result.vram_capacities_mb
    header = (
        "| method | "
        + " | ".join(f"bubble fit W={w:.0f}s" for w in windows)
        + " | "
        + " | ".join(f"admit C={c / 1024:.1f}GB" for c in capacities)
        + " |"
    )
    lines = [header, "|---" * (1 + len(windows) + len(capacities)) + "|"]
    for name, per_target in result.decisions.items():
        cells = [f"{d.accuracy:.3f}" for d in per_target[RUN]]
        cells += [
            f"{d.accuracy:.3f} (cost={d.asymmetric_cost:.3f})" for d in per_target[VRAM]
        ]
        lines.append(f"| {LABEL[name]} | " + " | ".join(cells) + " |")
    return [
        f"### {scheme} anchors — scheduler decision accuracy "
        "(admit cost weights a false admit 5x)",
        "",
        *lines,
        "",
    ]


def _cost_rows(
    profile_cost: ProfilingCost,
    anchor_cost: ProfilingCost,
    marginal: Mapping[str, float],
) -> list[str]:
    share = 100 * anchor_cost.total_gpu_hours / profile_cost.total_gpu_hours
    lines = [
        "## Cost",
        "",
        f"Full-grid profiling: {profile_cost.config_count} configurations, "
        f"{profile_cost.total_gpu_hours:.2f} GPU-hours, "
        f"{profile_cost.marginal_sec_per_config:.1f} s per configuration.",
        "",
        f"Anchor campaign: {anchor_cost.config_count} measurements, "
        f"{anchor_cost.total_gpu_hours:.2f} GPU-hours ({share:.2f}% of the full grid).",
        "",
        "| method | marginal cost per new configuration | needs a GPU |",
        "|---|---|---|",
        f"| Profile (upper bound) | "
        f"{profile_cost.marginal_sec_per_config:.1f} s | yes |",
    ]
    for name, seconds in marginal.items():
        rendered = (
            f"{seconds * 1e3:.1f} ms" if seconds >= 1e-3 else f"{seconds * 1e6:.0f} us"
        )
        lines.append(f"| {LABEL[name]} | {rendered} | no |")
    return [*lines, ""]


def _jsonable(result: SchemeResult) -> dict[str, object]:
    return {
        "anchor_count": result.anchor_count,
        "anchor_cost": asdict(result.anchor_cost),
        "stratum_counts": dict(result.stratum_counts),
        "run_windows_sec": list(result.run_windows_sec),
        "vram_capacities_mb": list(result.vram_capacities_mb),
        "accuracy": {
            name: {
                stratum: {target: asdict(score) for target, score in per_target.items()}
                for stratum, per_target in per_stratum.items()
            }
            for name, per_stratum in result.accuracy.items()
        },
        "ordering": {
            name: dict(per_stratum) for name, per_stratum in result.ordering.items()
        },
        "decisions": {
            name: {
                target: [asdict(decision) for decision in items]
                for target, items in per_target.items()
            }
            for name, per_target in result.decisions.items()
        },
        "bounds": {name: asdict(bound) for name, bound in result.bounds.items()},
    }


def _write_tables(
    out_dir: Path,
    results: Results,
    profile_cost: ProfilingCost,
    marginal: Mapping[str, float],
) -> None:
    lines = [
        "# Iso-budget predictor comparison",
        "",
        "Ground truth is the exhaustive profile cache. Every lookup/learned method "
        "sees the same anchor measurements; the analytical formula sees none.",
        "",
    ]
    for scheme, result in results.items():
        lines += [
            f"## {scheme} anchor scheme ({result.anchor_count} measurements, "
            f"{result.anchor_cost.total_gpu_hours:.2f} GPU-hours)",
            "",
            f"held-out = {result.stratum_counts[HELD_OUT]} configurations, "
            f"long-decode stratum = {result.stratum_counts[LONG_DECODE]}",
            "",
        ]
        lines += _accuracy_rows(result, scheme)
        lines += _vram_rows(result, scheme)
        lines += _decision_rows(result, scheme)
    lines += _cost_rows(profile_cost, _primary(results).anchor_cost, marginal)
    (out_dir / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csvs(
    out_dir: Path,
    results: Results,
    profile_cost: ProfilingCost,
    marginal: Mapping[str, float],
) -> None:
    runtime = ["scheme,method,stratum,count,wape,p95_relative_error,pair_order"]
    for scheme, result in results.items():
        for name, per_stratum in result.accuracy.items():
            for stratum, per_target in per_stratum.items():
                score = per_target[RUN]
                runtime.append(
                    f"{scheme},{name},{stratum},{score.count},{score.wape:.5f},"
                    f"{score.p95_relative_error:.5f},"
                    f"{result.ordering[name][stratum]:.5f}"
                )
    (out_dir / "runtime_wape.csv").write_text(
        "\n".join(runtime) + "\n", encoding="utf-8"
    )

    decisions = [
        "scheme,method,target,threshold,false_positive,false_negative,"
        "accuracy,asymmetric_cost"
    ]
    for scheme, result in results.items():
        for name, per_target in result.decisions.items():
            for target, items in per_target.items():
                decisions.extend(
                    f"{scheme},{name},{target},{d.threshold:.4f},{d.false_positive},"
                    f"{d.false_negative},{d.accuracy:.5f},{d.asymmetric_cost:.5f}"
                    for d in items
                )
    (out_dir / "decisions.csv").write_text(
        "\n".join(decisions) + "\n", encoding="utf-8"
    )

    bounds = [
        "scheme,method,quantile_level,violation_rate,mean_over_reservation,"
        "p95_over_reservation"
    ]
    for scheme, result in results.items():
        bounds.extend(
            f"{scheme},{name},{b.quantile_level:.2f},{b.violation_rate:.5f},"
            f"{b.mean_over_reservation:.5f},{b.p95_over_reservation:.5f}"
            for name, b in result.bounds.items()
        )
    (out_dir / "bound_quality.csv").write_text(
        "\n".join(bounds) + "\n", encoding="utf-8"
    )

    primary = _primary(results)
    pareto = [
        "method,run_wape,vram_wape,marginal_cost_sec,build_gpu_hours",
        f"profile,0.00000,0.00000,{profile_cost.marginal_sec_per_config:.5f},"
        f"{profile_cost.total_gpu_hours:.3f}",
    ]
    for name, per_stratum in primary.accuracy.items():
        if name not in marginal:
            continue
        hours = 0.0 if name == "analytical" else primary.anchor_cost.total_gpu_hours
        pareto.append(
            f"{name},{per_stratum[HELD_OUT][RUN].wape:.5f},"
            f"{per_stratum[HELD_OUT][VRAM].wape:.5f},{marginal[name]:.8f},"
            f"{hours:.3f}"
        )
    (out_dir / "pareto.csv").write_text("\n".join(pareto) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval_predictor_isobudget")
    parser.add_argument(
        "--profile-cache", type=Path, default=ROOT / "cache/profile/predictions.yaml"
    )
    parser.add_argument(
        "--gnn-cache", type=Path, default=ROOT / "cache/gnn/predictions.yaml"
    )
    parser.add_argument(
        "--gnn-metrics", type=Path, default=ROOT / "cache/gnn/metrics.json"
    )
    parser.add_argument(
        "--anchor-scheme", choices=("recorded", "spread", "both"), default="both"
    )
    parser.add_argument(
        "--include-heuristic",
        action="store_true",
        help="also score the excluded sparse-profiling heuristic (internal record)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "output/predictor_isobudget"
    )
    args = parser.parse_args(argv)

    truth = index_cache(load_resource_contract_cache(args.profile_cache))
    gnn = index_cache(load_resource_contract_cache(args.gnn_cache))
    uncovered = [key for key in truth if key not in gnn]
    assert not uncovered, f"gnn cache misses {len(uncovered)} profiled configurations"

    recorded = load_recorded_anchors(args.gnn_metrics, truth)
    schemes = {} if args.anchor_scheme == "spread" else {"recorded": recorded}
    if args.anchor_scheme in ("spread", "both"):
        schemes["spread"] = spread_anchors(truth)

    results = {
        scheme: evaluate_scheme(
            truth, gnn, anchors, include_heuristic=args.include_heuristic
        )
        for scheme, anchors in schemes.items()
    }
    profile_cost = profiling_cost(truth, tuple(truth))
    measured = baseline_marginal_seconds(truth, recorded)
    measured["sageradar"] = graph_forward_seconds()
    marginal = {name: measured[name] for name in ordered_methods(measured)}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_tables(args.out_dir, results, profile_cost, marginal)
    _write_csvs(args.out_dir, results, profile_cost, marginal)
    (args.out_dir / "accuracy.json").write_text(
        json.dumps(
            {
                "profile_cost": asdict(profile_cost),
                "marginal_cost_per_config_sec": marginal,
                "schemes": {
                    scheme: _jsonable(result) for scheme, result in results.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    primary = _primary(results)
    print(f"full-grid profiling = {profile_cost.total_gpu_hours:.2f} GPU-hours")
    print(f"anchor campaign     = {primary.anchor_cost.total_gpu_hours:.2f} GPU-hours")
    for name, per_stratum in primary.accuracy.items():
        print(
            f"  {name:16s} run WAPE held-out={per_stratum[HELD_OUT][RUN].wape:.4f} "
            f"long-decode={per_stratum[LONG_DECODE][RUN].wape:.4f} "
            f"vram WAPE={per_stratum[HELD_OUT][VRAM].wape:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
