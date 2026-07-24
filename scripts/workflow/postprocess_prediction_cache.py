"""Post-process a base GNN cache into a deployable v2 (VRAM + load calibration).

Non-destructive: the base cache is only read (its SHA256 is recorded in the
derived cache). Writes cache/gnn_v2/{predictions.yaml, calibration_report.json}.
Reusable by any later cache generation — point --base-cache at a new cache and
--trace-roots at the serving traces that calibrate its load estimates.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment.workflow.artifacts import (  # noqa: E402
    file_sha256,
    write_json_exclusive,
)
from experiment.workflow.cache_postprocess import build_v2_cache  # noqa: E402
from workflow.artifacts import load_resource_contract_cache  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="postprocess_prediction_cache")
    parser.add_argument(
        "--base-cache", type=Path, default=ROOT / "cache/gnn/predictions.yaml"
    )
    parser.add_argument(
        "--reference-cache", type=Path, default=ROOT / "cache/profile/predictions.yaml"
    )
    parser.add_argument(
        "--trace-roots",
        type=Path,
        nargs="+",
        default=[ROOT / "output/serve", ROOT / "output/serve_0723"],
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "cache/gnn_v2")
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="calibrate load only, keep base VRAM/run (use for the profile cache)",
    )
    args = parser.parse_args(argv)

    base_sha_before = file_sha256(args.base_cache)
    v2, load_calibration = build_v2_cache(
        args.base_cache,
        args.trace_roots,
        None if args.load_only else args.reference_cache,
        calibrate_vram=not args.load_only,
    )
    if file_sha256(args.base_cache) != base_sha_before:
        raise RuntimeError("base cache was mutated during post-processing")

    predictions_path = args.out_dir / "predictions.yaml"
    if predictions_path.exists():
        predictions_path.unlink()
    write_json_exclusive(predictions_path, v2)
    load_resource_contract_cache(predictions_path)  # revalidate

    base = load_resource_contract_cache(args.base_cache)
    original_load = {
        (e.key.model_name, e.key.gpu_name): e.predicted_load_sec for e in base.entries
    }
    load_rows = []
    for (model, gpu), tier in sorted(load_calibration.model_gpu.items()):
        base_load = original_load.get((model, gpu))
        load_rows.append(
            {
                "model": model,
                "gpu": gpu,
                "original_load_sec": base_load,
                "calibrated_load_sec": tier.value,
                "overestimate_factor": (base_load / tier.value if base_load else None),
                "run_count": tier.run_count,
                "warm_sample_count": tier.sample_count,
            }
        )
    report = {
        "base_cache": str(args.base_cache),
        "base_cache_sha256": base_sha_before,
        "trace_roots": [str(root) for root in args.trace_roots],
        "warm_sample_total": load_calibration.warm_sample_total,
        "cold_sample_total": load_calibration.cold_sample_total,
        "gpu_scale": dict(load_calibration.gpu_scale),
        "model_gpu_load_calibration": load_rows,
        "deployment_load_calibration": [
            {
                "model_key": model_key,
                "gpu": gpu,
                "warm_reload_sec": tier.value,
                "run_count": tier.run_count,
                "warm_sample_count": tier.sample_count,
            }
            for (model_key, gpu), tier in sorted(load_calibration.deployment.items())
        ],
    }
    report_path = args.out_dir / "calibration_report.json"
    if report_path.exists():
        report_path.unlink()
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"v2 cache -> {predictions_path} ({len(v2.entries)} entries)")
    print(
        f"warm={load_calibration.warm_sample_total} "
        f"cold={load_calibration.cold_sample_total}"
    )
    print("model x gpu load calibration (original -> calibrated, factor):")
    for row in load_rows:
        print(
            f"  {row['model']:12s} {row['gpu']:5s} "
            f"{row['original_load_sec']:.2f} -> {row['calibrated_load_sec']:.2f} "
            f"({row['overestimate_factor']:.2f}x, "
            f"runs={row['run_count']}, n={row['warm_sample_count']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
