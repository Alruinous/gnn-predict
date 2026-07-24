from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiment.workflow.static_cache import build_static_prediction_cache
from workflow.types import WorkflowModelFeatureKey


def _key(
    model: str, gpu: str, batch: int, seq: int, out: int
) -> WorkflowModelFeatureKey:
    return WorkflowModelFeatureKey(
        model_name=model,
        phase="decode",
        gpu_name=gpu,
        batch_size=batch,
        sequence_length=seq,
        decode_output_length=out,
    )


def _vram(cache, key: WorkflowModelFeatureKey) -> float:
    return cache.lookup(key).predicted_peak_vram_mb


def test_static_cache_produces_valid_positive_contracts() -> None:
    keys = [_key("Qwen3-4B", "a100", 1, 1024, 128)]
    cache = build_static_prediction_cache(keys)
    contract = cache.lookup(keys[0])
    assert contract.predicted_run_sec > 0
    assert contract.predicted_load_sec > 0
    assert contract.predicted_peak_vram_mb > 0
    assert contract.peak_vram_mb_upper_bound >= contract.predicted_peak_vram_mb


def test_static_vram_grows_with_batch_and_sequence() -> None:
    base = _key("Qwen3-4B", "a100", 1, 1024, 128)
    more_batch = _key("Qwen3-4B", "a100", 3, 1024, 128)
    more_seq = _key("Qwen3-4B", "a100", 1, 8192, 128)
    cache = build_static_prediction_cache([base, more_batch, more_seq])
    assert _vram(cache, more_batch) > _vram(cache, base)
    assert _vram(cache, more_seq) > _vram(cache, base)


def test_static_vram_grows_with_model_size() -> None:
    small = _key("Qwen3-0.6B", "a100", 1, 1024, 128)
    large = _key("Qwen3-14B", "a100", 1, 1024, 128)
    cache = build_static_prediction_cache([small, large])
    assert _vram(cache, large) > 5 * _vram(cache, small)


def test_static_decode_faster_on_higher_bandwidth_gpu() -> None:
    a100 = _key("Qwen3-8B", "a100", 1, 1024, 512)
    v100 = _key("Qwen3-8B", "v100", 1, 1024, 512)
    cache = build_static_prediction_cache([a100, v100])
    assert cache.lookup(a100).predicted_run_sec < cache.lookup(v100).predicted_run_sec
