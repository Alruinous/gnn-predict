from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

Status = Literal["ok", "oom", "timeout", "empty_output", "eval_fail"]


@dataclass(frozen=True)
class TraceEvent:
    workflow_name: str
    sample_id: str
    node_name: str
    node_type: str
    started_at: float
    ended_at: float
    duration_sec: float
    ready_at: float
    release_at: float
    pipeline_idle_time: float
    model_name: str | None
    model_instance_id: str | None
    resident_started_at: float | None
    resident_ended_at: float | None
    input_token_count: int
    output_token_count: int
    status: Status
    quality_result: dict[str, Any] | None = None
    output_text: str | None = None


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def event_dict(event: TraceEvent) -> dict[str, Any]:
    return asdict(event)


def stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "total": 0.0}
    ordered = sorted(values)
    return {
        "count": float(len(values)),
        "mean": sum(values) / len(values),
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "total": sum(values),
    }


def percentile(ordered_values: Sequence[float], q: float) -> float:
    assert ordered_values, ordered_values
    index = (len(ordered_values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered_values) - 1)
    weight = index - lower
    return ordered_values[lower] * (1 - weight) + ordered_values[upper] * weight


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
