"""Merge per-GPU-kind ResourceContractCache YAML files into one heterogeneous cache."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml  # noqa: E402

from workflow.artifacts import (  # noqa: E402
    ResourceContract,
    ResourceContractCache,
    load_resource_contract_cache,
)


@dataclass(frozen=True, slots=True)
class ProfileSource:
    gpu_kind: str
    path: Path


def parse_profile_source(raw: str) -> ProfileSource:
    gpu_kind, separator, path_text = raw.partition("=")
    gpu_kind = gpu_kind.strip().casefold()
    path_text = path_text.strip()
    if not separator or not gpu_kind or not path_text:
        raise argparse.ArgumentTypeError(f"expected GPU_KIND=PATH, got: {raw!r}")
    return ProfileSource(gpu_kind, Path(path_text))


def _entry_sort_key(entry: ResourceContract) -> tuple[str, str, str, int, int, int]:
    key = entry.key
    return (
        key.gpu_name,
        key.model_name,
        key.phase,
        key.batch_size,
        key.sequence_length,
        key.decode_output_length,
    )


def compose_prediction_cache(sources: Sequence[ProfileSource]) -> ResourceContractCache:
    if not sources:
        raise ValueError("compose_prediction_cache requires at least one source")
    seen_gpu_kinds: set[str] = set()
    version: int | None = None
    entries: list[ResourceContract] = []
    for source in sources:
        if source.gpu_kind in seen_gpu_kinds:
            raise ValueError(f"duplicate GPU kind in sources: {source.gpu_kind}")
        seen_gpu_kinds.add(source.gpu_kind)
        cache = load_resource_contract_cache(source.path)
        if version is None:
            version = cache.version
        elif cache.version != version:
            raise ValueError(
                f"cache version mismatch: {source.path} has {cache.version}, "
                f"expected {version}"
            )
        entries.extend(
            entry for entry in cache.entries if entry.key.gpu_name == source.gpu_kind
        )
    entries.sort(key=_entry_sort_key)
    assert version is not None
    return ResourceContractCache(
        version=version, environment={}, entries=tuple(entries)
    )


def write_prediction_cache(path: Path, cache: ResourceContractCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(cache.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compose per-GPU-kind ResourceContractCache YAML files into one cache."
        )
    )
    parser.add_argument(
        "--source",
        type=parse_profile_source,
        action="append",
        required=True,
        metavar="GPU_KIND=PATH",
        help="Repeatable. Each source contributes only its own GPU kind's entries.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache = compose_prediction_cache(args.source)
    write_prediction_cache(args.output, cache)
    print(f"wrote {len(cache.entries)} entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
