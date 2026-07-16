from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, NonNegativeInt

from common.validate import NonEmptyStr
from dataset.schema import TaskSample

SampleStratum = Literal["short", "medium", "long"]
Sha256Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class SampleManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    position: NonNegativeInt
    sample_id: NonEmptyStr
    stratum: SampleStratum
    metadata: dict[str, JsonValue]
    digest: Sha256Digest


def stable_sample_digest(sample: TaskSample) -> str:
    payload = json.dumps(
        sample.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def load_sample_manifest(path: str | Path) -> tuple[SampleManifestEntry, ...]:
    manifest_path = Path(path)
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"sample manifest is empty: {manifest_path}")
    if any(not line.strip() for line in lines):
        raise ValueError(f"sample manifest contains an empty row: {manifest_path}")
    entries = tuple(SampleManifestEntry.model_validate_json(line) for line in lines)
    positions = [entry.position for entry in entries]
    if positions != list(range(len(entries))):
        raise ValueError("sample manifest positions must be contiguous and ordered")
    sample_ids = [entry.sample_id for entry in entries]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample manifest contains duplicate sample ids")
    return entries


def resolve_manifest_samples(
    entries: Sequence[SampleManifestEntry],
    samples: Sequence[TaskSample],
) -> tuple[TaskSample, ...]:
    sample_map: dict[str, TaskSample] = {}
    for sample in samples:
        if sample.sample_id in sample_map:
            raise ValueError(
                f"dataset contains duplicate sample id: {sample.sample_id}"
            )
        sample_map[sample.sample_id] = sample

    resolved: list[TaskSample] = []
    for entry in entries:
        sample = sample_map.get(entry.sample_id)
        if sample is None:
            raise KeyError(
                f"sample manifest id is missing from dataset: {entry.sample_id}"
            )
        digest = stable_sample_digest(sample)
        if digest != entry.digest:
            raise ValueError(f"sample digest mismatch: {entry.sample_id}")
        resolved.append(sample)
    return tuple(resolved)
