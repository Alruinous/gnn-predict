"""Build the frozen GSM8K sample manifest asset (60 samples, stratified, seed 42)."""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset.gsm8k import load_gsm8k_split  # noqa: E402
from dataset.schema import TaskSample  # noqa: E402
from experiment.workflow.artifacts import canonical_json  # noqa: E402
from experiment.workflow.gsm8k import GSM8K_MANIFEST_PATH, _gsm8k_metadata  # noqa: E402
from experiment.workflow.sample import (  # noqa: E402
    SampleManifestEntry,
    SampleStratum,
    stable_sample_digest,
)

DEFAULT_DATASET_PATH = ROOT / "dataset" / "gsm8k" / "data" / "test.jsonl"
STRATA: tuple[SampleStratum, ...] = ("short", "medium", "long")
TOTAL_COUNT = 60
SEED = 42


def _split_evenly(
    items: Sequence[TaskSample], parts: int
) -> list[Sequence[TaskSample]]:
    bounds = [len(items) * index // parts for index in range(parts + 1)]
    return [items[bounds[index] : bounds[index + 1]] for index in range(parts)]


def select_stratified(
    samples: Sequence[TaskSample],
) -> list[tuple[SampleStratum, TaskSample]]:
    per_stratum = TOTAL_COUNT // len(STRATA)
    ordered = sorted(samples, key=lambda sample: len(sample.input_text))
    rng = random.Random(SEED)
    selected: list[tuple[SampleStratum, TaskSample]] = []
    for stratum, part in zip(STRATA, _split_evenly(ordered, len(STRATA)), strict=True):
        if len(part) < per_stratum:
            raise ValueError(f"stratum {stratum} has too few samples: {len(part)}")
        chosen = rng.sample(list(part), per_stratum)
        selected.extend((stratum, sample) for sample in chosen)
    return selected


def build_entries(
    selected: Sequence[tuple[SampleStratum, TaskSample]],
) -> list[SampleManifestEntry]:
    by_id = sorted(selected, key=lambda item: item[1].sample_id)
    return [
        SampleManifestEntry(
            position=position,
            sample_id=sample.sample_id,
            stratum=stratum,
            metadata=_gsm8k_metadata(sample),
            digest=stable_sample_digest(sample),
        )
        for position, (stratum, sample) in enumerate(by_id)
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build_gsm8k_manifest")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=GSM8K_MANIFEST_PATH)
    args = parser.parse_args(argv)
    samples = load_gsm8k_split(args.dataset_path, "test")
    entries = build_entries(select_stratified(samples))
    payload = "".join(f"{canonical_json(entry)}\n" for entry in entries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(f"wrote {len(entries)} entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
