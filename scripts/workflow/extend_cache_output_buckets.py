"""Add short decode_output_length buckets to a prediction cache, in place.

The profiled caches start at a 128-token output bucket, but most serve requests stop
far below it (median observed output is ~34% of max_new_tokens, and gsm8k's refine node
emits ~5 tokens). Pricing those at the 128 floor overstates decode several-fold, so the
scheduler's timing path cannot resolve them.

Profiled run time is close to affine in output length over the measured range, so the
short buckets come from the fit through the two smallest measured buckets — an
interpolation toward zero, not an extrapolation past the measured span. Only
``predicted_run_sec`` is recomputed; VRAM and power are copied from the 128 bucket
rather than scaled down, so an added row can never make an admission decision more
optimistic than a measured one. Each entry records how it was derived in its metadata.

Idempotent: buckets already present are left untouched.
"""

from __future__ import annotations

import argparse
import collections
import copy
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from workflow.artifacts import ResourceContractCache  # noqa: E402

DEFAULT_BUCKETS = (2, 4, 8, 16, 24, 32, 48, 64, 96)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_path", type=Path)
    parser.add_argument(
        "--buckets",
        default=",".join(str(bucket) for bucket in DEFAULT_BUCKETS),
        help="comma-separated decode_output_length values to add",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    buckets = tuple(int(value) for value in args.buckets.split(","))
    document = yaml.safe_load(args.cache_path.read_text(encoding="utf-8"))
    entries = document["entries"]

    groups: dict[tuple[str, str, int, int], dict[int, dict]] = collections.defaultdict(
        dict
    )
    for entry in entries:
        key = entry["key"]
        groups[
            (
                key["model_name"],
                key["gpu_name"],
                key["batch_size"],
                key["sequence_length"],
            )
        ][key["decode_output_length"]] = entry

    added: list[dict] = []
    residuals: list[float] = []
    clamped_count = 0
    for group, by_output in sorted(groups.items()):
        outputs = sorted(by_output)
        if len(outputs) < 2:
            raise SystemExit(f"group {group} has too few buckets to fit")
        low, high = outputs[0], outputs[1]
        slope = (
            by_output[high]["predicted_run_sec"] - by_output[low]["predicted_run_sec"]
        ) / (high - low)
        if slope <= 0:
            raise SystemExit(f"group {group} has a non-physical slope: {slope}")
        intercept = by_output[low]["predicted_run_sec"] - slope * low
        # A slightly negative intercept is fit noise on models whose prefill is ~0;
        # clamping keeps run_sec positive and biases short outputs upward.
        clamped = intercept < 0
        if clamped:
            clamped_count += 1
            intercept = 0.0
        residuals.append(
            max(
                abs(intercept + slope * output - by_output[output]["predicted_run_sec"])
                / by_output[output]["predicted_run_sec"]
                for output in outputs
            )
        )
        for output in buckets:
            if output in by_output:
                continue
            entry = copy.deepcopy(by_output[low])
            entry["key"]["decode_output_length"] = output
            entry["predicted_run_sec"] = intercept + slope * output
            metadata = entry["predictor_metadata"]
            metadata["source"] = "derived_linear_output_scaling"
            metadata["measurement_semantics"] = "expected_generate_at_output_length"
            metadata["run_sample_count"] = 0
            metadata["derivation"] = {
                "method": "linear_fit_on_two_smallest_measured_output_buckets",
                "fit_output_lengths": [low, high],
                "prefill_intercept_sec": intercept,
                "per_output_token_sec": slope,
                "vram_and_power": f"copied_from_output_{low}_bucket_conservative",
                "intercept_clamped_to_zero": clamped,
            }
            added.append(entry)

    if not added:
        print(f"{args.cache_path}: already extended, nothing to do")
        return 0

    document["entries"] = entries + added
    document.setdefault("environment", {})["output_bucket_extension"] = {
        "added_buckets": list(buckets),
        "method": "linear_fit_on_two_smallest_measured_output_buckets",
        "reason": "timing lookups need sub-128-token resolution; VRAM rows unchanged",
    }
    args.cache_path.write_text(
        yaml.safe_dump(document, sort_keys=True, allow_unicode=True), encoding="utf-8"
    )
    cache = ResourceContractCache.model_validate(
        yaml.safe_load(args.cache_path.read_text(encoding="utf-8"))
    )
    print(
        f"{args.cache_path}: groups={len(groups)} added={len(added)} "
        f"total={len(cache.entries)} intercept_clamped={clamped_count}"
    )
    print(
        f"  linear-fit residual over measured range: "
        f"median={statistics.median(residuals) * 100:.1f}% "
        f"max={max(residuals) * 100:.1f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
