"""Score predictor caches against the profile ground truth (offline accuracy+cost).

Profile is the empirical upper bound; every other method (gnn, static,
config_mean, nn, tabular) is scored on the same held-out split. Emits
output/cache_eval/{accuracy_cost.json, tables.md, pareto.csv}.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment.workflow.predictor_baselines import (  # noqa: E402
    build_config_mean_cache,
    build_nearest_profile_cache,
    build_tabular_cache,
    deterministic_split,
    stratified_anchor_sample,
)
from experiment.workflow.static_cache import (  # noqa: E402
    build_static_prediction_cache,
)

from workflow.artifacts import (  # noqa: E402
    ResourceContract,
    load_resource_contract_cache,
)
from workflow.types import WorkflowModelFeatureKey  # noqa: E402

TARGETS = ("load_sec", "run_sec", "peak_vram_mb", "power_watts")
FIELD = {
    "load_sec": "predicted_load_sec",
    "run_sec": "predicted_run_sec",
    "peak_vram_mb": "predicted_peak_vram_mb",
    "power_watts": "predicted_power_watts",
}
METHOD_CACHES = ("gnn", "static", "config_mean", "nn", "tabular")


def _value(contract: ResourceContract, target: str) -> float:
    return float(getattr(contract, FIELD[target]) or 0.0)


@dataclass
class Score:
    n: int
    wape: float
    mae: float
    rmse: float
    mean_bias: float
    underestimate_rate: float
    max_underestimate: float


def score(pairs: Sequence[tuple[float, float]]) -> Score:
    assert pairs, "cannot score an empty target"
    abs_err = [abs(p - t) for p, t in pairs]
    truth_abs = sum(abs(t) for _, t in pairs)
    under = [t - p for p, t in pairs if p < t]
    return Score(
        n=len(pairs),
        wape=sum(abs_err) / truth_abs if truth_abs else 0.0,
        mae=statistics.fmean(abs_err),
        rmse=(statistics.fmean([(p - t) ** 2 for p, t in pairs])) ** 0.5,
        mean_bias=statistics.fmean([p - t for p, t in pairs]),
        underestimate_rate=len(under) / len(pairs),
        max_underestimate=max(under, default=0.0),
    )


@dataclass
class MethodResult:
    coverage: float
    targets: dict[str, Score] = field(default_factory=dict)
    vram_by_cell: dict[str, Score] = field(default_factory=dict)


def _score_method(
    truth: dict[WorkflowModelFeatureKey, ResourceContract],
    method: dict[WorkflowModelFeatureKey, ResourceContract],
    test_keys: Sequence[WorkflowModelFeatureKey],
) -> MethodResult:
    covered = [key for key in test_keys if key in method]
    result = MethodResult(coverage=len(covered) / len(test_keys))
    for target in TARGETS:
        pairs = [(_value(method[k], target), _value(truth[k], target)) for k in covered]
        result.targets[target] = score(pairs)
    cells: dict[str, list[tuple[float, float]]] = {}
    for key in covered:
        cell = f"{key.model_name}/{key.gpu_name}"
        cells.setdefault(cell, []).append(
            (_value(method[key], "peak_vram_mb"), _value(truth[key], "peak_vram_mb"))
        )
    result.vram_by_cell = {cell: score(pairs) for cell, pairs in cells.items()}
    return result


def _profile_cost(entries: Sequence[ResourceContract]) -> dict[str, float]:
    run_gpu_sec = sum(e.predicted_run_sec for e in entries)
    cells: dict[tuple[str, str], tuple[float, int]] = {}
    for e in entries:
        raw = e.predictor_metadata.get("load_sample_count", 1)
        samples = int(raw) if isinstance(raw, (int, float, str)) else 1
        cells[(e.key.model_name, e.key.gpu_name)] = (e.predicted_load_sec, samples)
    load_gpu_sec = sum(load * samples for load, samples in cells.values())
    total = run_gpu_sec + load_gpu_sec
    return {
        "run_gpu_sec": run_gpu_sec,
        "load_gpu_sec": load_gpu_sec,
        "total_gpu_sec": total,
        "total_gpu_hours": total / 3600.0,
        "marginal_gpu_sec_per_config": total / len(entries),
    }


def _gnn_forward_seconds(iters: int = 20) -> float | None:
    try:
        import torch
        from torch_geometric.data import Data

        from gnn_model.data.constants import (
            EDGE_FEATURE_DIM,
            GRAPH_FEATURE_DIM,
            NODE_FEATURE_DIM,
            OP_TYPE_COUNT,
            OP_TYPE_EMBEDDING_DIM,
        )
        from gnn_model.models.predictor import IntelliGraphLargeModelPredictor

        n_nodes, n_edges = 800, 1600
        data = Data(
            x=torch.randn(n_nodes, NODE_FEATURE_DIM),
            edge_index=torch.randint(0, n_nodes, (2, n_edges)),
            edge_attr=torch.randn(n_edges, EDGE_FEATURE_DIM),
        )
        data.graph_features = torch.randn(1, GRAPH_FEATURE_DIM)
        data.op_type_ids = torch.randint(0, OP_TYPE_COUNT, (n_nodes,))
        model = IntelliGraphLargeModelPredictor(
            node_dim=NODE_FEATURE_DIM,
            edge_dim=EDGE_FEATURE_DIM,
            graph_dim=GRAPH_FEATURE_DIM,
            op_type_count=OP_TYPE_COUNT,
            op_type_embedding_dim=OP_TYPE_EMBEDDING_DIM,
            hidden_dim=128,
            targets=[f"t{i}" for i in range(9)],
            num_heads=8,
            num_layers=2,
            readout_mode="mean_sum_max",
            structural_context_mode="basic",
        ).eval()
        samples: list[float] = []
        with torch.no_grad():
            for _ in range(3):
                model(data)
            for _ in range(iters):
                start = time.perf_counter()
                model(data)
                samples.append(time.perf_counter() - start)
        return statistics.median(samples)
    except (ImportError, RuntimeError, ValueError, AssertionError) as exc:
        print(f"  (gnn forward microbench skipped: {exc})")
        return None


def _baseline_costs(
    truth: dict[WorkflowModelFeatureKey, ResourceContract],
    train_keys: Sequence[WorkflowModelFeatureKey],
    all_keys: Sequence[WorkflowModelFeatureKey],
) -> dict[str, float]:
    n = len(all_keys)
    t0 = time.perf_counter()
    build_static_prediction_cache(all_keys)
    static_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    build_config_mean_cache(truth, train_keys, all_keys)
    cm_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    build_nearest_profile_cache(truth, train_keys, all_keys)
    nn_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    build_tabular_cache(truth, train_keys, all_keys)
    tab_sec = time.perf_counter() - t0
    return {
        "static_per_config_sec": static_sec / n,
        "config_mean_per_config_sec": cm_sec / n,
        "nn_per_config_sec": nn_sec / n,
        "tabular_total_sec": tab_sec,
        "tabular_per_config_sec": tab_sec / n,
    }


BUDGETS_PER_CELL = (1, 2, 3, 5, 10, 20, 50)


def _target_wape(
    truth: dict[WorkflowModelFeatureKey, ResourceContract],
    method: dict[WorkflowModelFeatureKey, ResourceContract],
    test_keys: Sequence[WorkflowModelFeatureKey],
    target: str,
) -> float:
    return score(
        [(_value(method[k], target), _value(truth[k], target)) for k in test_keys]
    ).wape


def _budget_sweep(
    truth: dict[WorkflowModelFeatureKey, ResourceContract],
    train_keys: Sequence[WorkflowModelFeatureKey],
    test_keys: Sequence[WorkflowModelFeatureKey],
) -> list[dict[str, object]]:
    """Anchor-budget-matched comparison: memorization baselines see only N
    profiles per cell — the regime where the GNN's generalization shows."""
    builders = {
        "config_mean": build_config_mean_cache,
        "nn": build_nearest_profile_cache,
        "tabular": build_tabular_cache,
    }
    rows: list[dict[str, object]] = []
    for per_cell in BUDGETS_PER_CELL:
        anchors = stratified_anchor_sample(train_keys, per_cell)
        for name, build in builders.items():
            cache = build(truth, anchors, test_keys)
            method = {entry.key: entry for entry in cache.entries}
            rows.append(
                {
                    "per_cell": per_cell,
                    "anchors": len(anchors),
                    "method": name,
                    "run_wape": _target_wape(truth, method, test_keys, "run_sec"),
                    "vram_wape": _target_wape(truth, method, test_keys, "peak_vram_mb"),
                }
            )
    return rows


def _fmt_scores(name: str, result: MethodResult) -> list[str]:
    header = "| target | WAPE | MAE | under_rate | max_under | mean_bias |"
    rows = [
        f"### {name}  (coverage={result.coverage:.3f})",
        "",
        header,
        "|---|---|---|---|---|---|",
    ]
    rows.extend(
        f"| {target} | {s.wape:.4f} | {s.mae:.3f} | {s.underestimate_rate:.3f} "
        f"| {s.max_underestimate:.2f} | {s.mean_bias:+.3f} |"
        for target, s in result.targets.items()
    )
    rows.append("")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval_prediction_caches")
    parser.add_argument(
        "--profile-cache", type=Path, default=ROOT / "cache/profile/predictions.yaml"
    )
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output/cache_eval")
    args = parser.parse_args(argv)

    profile = load_resource_contract_cache(args.profile_cache)
    truth = {entry.key: entry for entry in profile.entries}
    all_keys = tuple(truth)
    train_keys, test_keys = deterministic_split(all_keys)

    methods: dict[str, dict[WorkflowModelFeatureKey, ResourceContract]] = {}
    for name in METHOD_CACHES:
        path = args.cache_dir / name / "predictions.yaml"
        if not path.exists():
            print(f"  (missing {name} cache at {path}; run build script first)")
            continue
        cache = load_resource_contract_cache(path)
        methods[name] = {entry.key: entry for entry in cache.entries}

    results = {
        name: _score_method(truth, method, test_keys)
        for name, method in methods.items()
    }
    sweep = _budget_sweep(truth, train_keys, test_keys)

    profile_cost = _profile_cost(profile.entries)
    baseline_cost = _baseline_costs(truth, train_keys, all_keys)
    gnn_forward_sec = _gnn_forward_seconds()

    marginal_cost = {
        "profile": profile_cost["marginal_gpu_sec_per_config"],
        "gnn": gnn_forward_sec,
        "static": baseline_cost["static_per_config_sec"],
        "config_mean": baseline_cost["config_mean_per_config_sec"],
        "nn": baseline_cost["nn_per_config_sec"],
        "tabular": baseline_cost["tabular_per_config_sec"],
    }

    payload = {
        "test_key_count": len(test_keys),
        "train_key_count": len(train_keys),
        "profile_cost": profile_cost,
        "baseline_cost": baseline_cost,
        "gnn_forward_sec": gnn_forward_sec,
        "marginal_cost_per_config_sec": marginal_cost,
        "accuracy": {
            name: {
                "coverage": r.coverage,
                "targets": {t: vars(s) for t, s in r.targets.items()},
                "vram_by_cell": {c: vars(s) for c, s in r.vram_by_cell.items()},
            }
            for name, r in results.items()
        },
        "budget_sweep": sweep,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "accuracy_cost.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# 预测缓存精度-成本评估",
        "",
        f"held-out test keys = {len(test_keys)} (seed=7, 20%)",
        "",
    ]
    for name in ("gnn", "static", "config_mean", "nn", "tabular"):
        if name in results:
            lines += _fmt_scores(name, results[name])
    lines += [
        "## 边际成本 (每个新配置)",
        "",
        "| method | cost/config | note |",
        "|---|---|---|",
        f"| profile | {marginal_cost['profile']:.3f} s | GPU 真机; 换卡须重跑 |",
    ]
    gnn_txt = (
        f"{gnn_forward_sec * 1000:.3f} ms" if gnn_forward_sec is not None else "n/a"
    )
    lines.append(f"| gnn | {gnn_txt} | 图上一次 forward, 无 GPU generate; 训练摊销 |")
    cpu_cost_key = {
        "static": "static_per_config_sec",
        "config_mean": "config_mean_per_config_sec",
        "nn": "nn_per_config_sec",
        "tabular": "tabular_per_config_sec",
    }
    lines.extend(
        f"| {name} | {baseline_cost[cpu_cost_key[name]] * 1e6:.2f} us "
        "| 公式/查表/GBDT, 纯 CPU |"
        for name in ("static", "config_mean", "nn", "tabular")
    )
    (args.out_dir / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    pareto = ["method,vram_wape,run_wape,vram_underestimate_rate,marginal_cost_sec"]
    pareto.extend(
        f"{name},{r.targets['peak_vram_mb'].wape:.5f},{r.targets['run_sec'].wape:.5f},"
        f"{r.targets['peak_vram_mb'].underestimate_rate:.4f},"
        f"{marginal_cost[name] if marginal_cost[name] else ''}"
        for name, r in results.items()
    )
    (args.out_dir / "pareto.csv").write_text("\n".join(pareto) + "\n", encoding="utf-8")

    sweep_lines = ["per_cell,anchors,method,run_wape,vram_wape"]
    sweep_lines.extend(
        f"{row['per_cell']},{row['anchors']},{row['method']},"
        f"{row['run_wape']:.5f},{row['vram_wape']:.5f}"
        for row in sweep
    )
    if "gnn" in results:
        g = results["gnn"].targets
        sweep_lines.append(
            f"gnn_ref,~518,gnn,{g['run_sec'].wape:.5f},{g['peak_vram_mb'].wape:.5f}"
        )
    if "static" in results:
        s = results["static"].targets
        sweep_lines.append(
            f"static_ref,0,static,{s['run_sec'].wape:.5f},{s['peak_vram_mb'].wape:.5f}"
        )
    (args.out_dir / "budget_sweep.csv").write_text(
        "\n".join(sweep_lines) + "\n", encoding="utf-8"
    )

    print(f"profile GPU-hours to build cache = {profile_cost['total_gpu_hours']:.2f}")
    for name in ("gnn", "static", "tabular", "config_mean", "nn"):
        if name in results:
            v = results[name].targets["peak_vram_mb"]
            print(
                f"  {name:12s} vram WAPE={v.wape:.4f} "
                f"under_rate={v.underestimate_rate:.3f} "
                f"max_under={v.max_underestimate / 1024:.2f}GB"
            )
    matched = [row for row in sweep if row["per_cell"] == 5]
    if matched and "gnn" in results:
        print("run_sec WAPE @ 5 anchors/cell (GNN's runtime-calibration budget):")
        print(f"  gnn(fixed)   {results['gnn'].targets['run_sec'].wape:.4f}")
        for row in matched:
            print(f"  {row['method']:12s} {row['run_wape']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
