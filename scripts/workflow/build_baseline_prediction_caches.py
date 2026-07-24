"""Build predictor-baseline resource caches over the profile cache key space.

Emits cache/{static,config_mean,nn,tabular}/predictions.yaml — every method
predicts the identical keys as cache/profile so they can be scored head-to-head
(and, in principle, fed to the scheduler exactly like the profile/gnn caches).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
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
)
from experiment.workflow.static_cache import (  # noqa: E402
    build_static_prediction_cache,
    load_arch_table,
)

from experiment.workflow.artifacts import write_json_exclusive  # noqa: E402
from workflow.artifacts import (  # noqa: E402
    ResourceContractCache,
    load_resource_contract_cache,
)


def _write(cache: ResourceContractCache, path: Path) -> None:
    if path.exists():
        path.unlink()
    write_json_exclusive(path, cache)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build_baseline_prediction_caches")
    parser.add_argument(
        "--profile-cache", type=Path, default=ROOT / "cache/profile/predictions.yaml"
    )
    parser.add_argument(
        "--cache-config", type=Path, default=ROOT / "config/workflow/cache.yaml"
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "cache")
    args = parser.parse_args(argv)

    profile = load_resource_contract_cache(args.profile_cache)
    truth = {entry.key: entry for entry in profile.entries}
    all_keys = tuple(truth)
    train_keys, test_keys = deterministic_split(all_keys)
    arch_table = load_arch_table(args.cache_config)

    caches = {
        "static": build_static_prediction_cache(all_keys, arch_table),
        "config_mean": build_config_mean_cache(truth, train_keys, all_keys),
        "nn": build_nearest_profile_cache(truth, train_keys, all_keys),
        "tabular": build_tabular_cache(truth, train_keys, all_keys),
    }

    profile_key_set = set(all_keys)
    print(f"profile keys={len(all_keys)} train={len(train_keys)} test={len(test_keys)}")
    for name, cache in caches.items():
        path = args.out_dir / name / "predictions.yaml"
        _write(cache, path)
        reloaded = load_resource_contract_cache(path)
        covered = {entry.key for entry in reloaded.entries}
        coverage = len(covered & profile_key_set) / len(profile_key_set)
        print(
            f"  {name:12s} entries={len(reloaded.entries)} "
            f"coverage={coverage:.4f} -> {path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
