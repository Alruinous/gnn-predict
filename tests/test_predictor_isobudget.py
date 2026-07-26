from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiment.workflow.predictor_isobudget import (
    RUN,
    VRAM,
    bound_quality,
    load_recorded_anchors,
    pairwise_order_accuracy,
    profiling_cost,
    score_accuracy,
    threshold_decision,
)
from workflow.artifacts import (
    ResourceContract,
    ResourceContractSource,
    ResourceEvidence,
)
from workflow.types import WorkflowModelFeatureKey


def make_key(
    model_name: str = "Qwen3-0.6B",
    gpu_name: str = "a100",
    batch_size: int = 1,
    sequence_length: int = 1024,
    decode_output_length: int = 512,
) -> WorkflowModelFeatureKey:
    return WorkflowModelFeatureKey(
        model_name=model_name,
        phase="decode",
        gpu_name=gpu_name,
        batch_size=batch_size,
        sequence_length=sequence_length,
        decode_output_length=decode_output_length,
    )


def make_contract(
    key: WorkflowModelFeatureKey,
    *,
    run_sec: float,
    peak_vram_mb: float,
    load_sec: float = 24.0,
) -> ResourceContract:
    return ResourceContract(
        key=key,
        source=ResourceContractSource.EMPIRICAL_PROFILE,
        predicted_load_sec=load_sec,
        predicted_run_sec=run_sec,
        predicted_peak_vram_mb=peak_vram_mb,
        peak_vram_mb_upper_bound=peak_vram_mb,
        peak_vram_mb_evidence=ResourceEvidence(
            method="point_estimate_only", sample_count=1
        ),
    )


@pytest.fixture
def truth() -> dict[WorkflowModelFeatureKey, ResourceContract]:
    table = {}
    for index, output_length in enumerate((128, 256, 512, 1024, 2048, 4096)):
        key = make_key(decode_output_length=output_length)
        table[key] = make_contract(
            key, run_sec=10.0 * (index + 1), peak_vram_mb=1000.0 * (index + 1)
        )
    return table


def test_load_recorded_anchors_rejects_a_key_absent_from_the_profile(
    tmp_path: Path, truth
) -> None:
    absent = make_key(model_name="Qwen3-14B", decode_output_length=768)
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {"runtime_calibration": {"selected_sample_keys": [absent.model_dump()]}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="absent from profile truth"):
        load_recorded_anchors(metrics, truth)


def test_load_recorded_anchors_returns_the_recorded_keys(tmp_path: Path, truth) -> None:
    present = next(iter(truth))
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {"runtime_calibration": {"selected_sample_keys": [present.model_dump()]}}
        ),
        encoding="utf-8",
    )
    assert load_recorded_anchors(metrics, truth) == (present,)


def test_pairwise_order_accuracy_is_one_for_a_monotone_prediction(truth) -> None:
    keys = tuple(truth)
    offset = {key: contract.predicted_run_sec + 5.0 for key, contract in truth.items()}
    assert pairwise_order_accuracy(offset, truth, keys) == 1.0


def test_pairwise_order_accuracy_is_zero_for_a_reversed_prediction(truth) -> None:
    keys = tuple(truth)
    reversed_values = {
        key: 1.0 / contract.predicted_run_sec for key, contract in truth.items()
    }
    assert pairwise_order_accuracy(reversed_values, truth, keys) == 0.0


def test_threshold_decision_separates_false_admits_from_false_rejects(truth) -> None:
    keys = tuple(truth)
    # Half the true run time: everything looks like it fits a mid-range window.
    optimistic = {key: contract.predicted_run_sec / 2 for key, contract in truth.items()}
    decision = threshold_decision(optimistic, truth, keys, RUN, threshold=25.0)
    assert decision.false_negative == 0
    # True 30/40/50 s look like 15/20/25 s, so all three pass a 25 s window.
    assert decision.false_positive == 3
    assert decision.asymmetric_cost == pytest.approx(15.0 / len(keys))


def test_threshold_decision_is_exact_for_a_perfect_predictor(truth) -> None:
    keys = tuple(truth)
    exact = {key: contract.predicted_run_sec for key, contract in truth.items()}
    decision = threshold_decision(exact, truth, keys, RUN, threshold=25.0)
    assert (decision.false_positive, decision.false_negative) == (0, 0)
    assert decision.accuracy == 1.0


def test_score_accuracy_reports_the_worst_shortfall(truth) -> None:
    keys = tuple(truth)
    short = {
        key: contract.predicted_peak_vram_mb - 100.0 for key, contract in truth.items()
    }
    score = score_accuracy(short, truth, keys, VRAM)
    assert score.underestimate_rate == 1.0
    assert score.max_underestimate == pytest.approx(100.0)


def test_bound_quality_never_violates_a_conservative_bound(truth) -> None:
    keys = tuple(truth)
    conservative = {
        key: contract.predicted_peak_vram_mb * 2 for key, contract in truth.items()
    }
    quality = bound_quality(conservative, truth, keys, keys)
    assert quality.violation_rate == 0.0
    assert quality.mean_over_reservation > 0.0


def test_bound_quality_flags_an_underestimating_predictor(truth) -> None:
    keys = tuple(truth)
    # Anchor residuals are all large and positive, so the fitted margin is large
    # for the anchors yet the flat prediction still misses the widest configs.
    flat = dict.fromkeys(truth, 1000.0)
    quality = bound_quality(flat, truth, keys[:2], keys)
    assert quality.violation_rate > 0.0


def test_profiling_cost_charges_one_load_per_model_gpu_cell(truth) -> None:
    keys = tuple(truth)
    cost = profiling_cost(truth, keys)
    run_sec = sum(contract.predicted_run_sec for contract in truth.values())
    assert cost.config_count == len(keys)
    assert cost.run_gpu_hours == pytest.approx(run_sec / 3600.0)
    assert cost.load_gpu_hours == pytest.approx(24.0 / 3600.0)
