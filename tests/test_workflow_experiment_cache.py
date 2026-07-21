from __future__ import annotations

from pathlib import Path

from experiment.workflow.cache import (
    build_synthetic_prediction_cache,
    write_synthetic_prediction_cache,
)
from workflow.artifacts import load_resource_contract_cache


def test_synthetic_cache_is_deterministic_and_covers_formal_models(
    tmp_path: Path,
) -> None:
    first = build_synthetic_prediction_cache()
    second = build_synthetic_prediction_cache()

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert {entry.key.model_name for entry in first.entries} == {
        "Qwen3-4B",
        "Qwen3-8B",
        "Qwen3-14B",
    }
    assert {entry.key.batch_size for entry in first.entries} == {1, 2, 3}

    path = tmp_path / "synthetic_prediction_cache.json"
    digest = write_synthetic_prediction_cache(path)
    loaded = load_resource_contract_cache(path)
    assert loaded == first
    assert len(digest) == 64


def test_synthetic_cache_uses_smallest_formal_context_buckets() -> None:
    cache = build_synthetic_prediction_cache()

    assert cache.decode_sequence_lengths("Qwen3-4B", "v100")[-1] == 7168
    assert cache.decode_sequence_lengths("Qwen3-8B", "v100")[-1] == 3584
    assert cache.decode_output_lengths("Qwen3-14B", "v100", 512) == (512,)
