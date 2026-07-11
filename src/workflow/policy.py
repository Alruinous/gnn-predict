from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
)

from common.validate import NonEmptyStr
from workflow.artifacts import (
    AcceleratorConfig,
    GpuKind,
    PredictionCache,
    PredictionEntry,
)
from workflow.schema import AgentNodeConfig
from workflow.types import (
    ModelReplicaState,
    TokenBudgetAction,
    WorkflowModelFeatureKey,
)


class PolicyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TokenBudgetDecision(PolicyModel):
    accelerator_id: NonEmptyStr | None = None
    gpu_kind: GpuKind | None = None
    sequence_length: PositiveInt | None = None
    granted_max_new_tokens: PositiveInt | None = None
    prediction_key: WorkflowModelFeatureKey | None = None
    predicted_run_sec: PositiveFloat | None = None
    predicted_peak_vram_mb: PositiveFloat | None = None
    effective_vram_mb: PositiveFloat | None = None
    action: TokenBudgetAction
    reason: (
        Literal[
            "input_bucket_missing",
            "prediction_bucket_missing",
            "token_budget_infeasible",
        ]
        | None
    ) = None


class EvictionCandidate(PolicyModel):
    replica_id: NonEmptyStr
    state: ModelReplicaState
    idle_since: NonNegativeFloat
    reuse_distance_sec: NonNegativeFloat | None
    reload_cost_sec: NonNegativeFloat | None


@dataclass(frozen=True, slots=True)
class _FeasibleCandidate:
    accelerator: AcceleratorConfig
    prediction: PredictionEntry
    effective_vram_mb: float


def select_token_budget(
    *,
    node: AgentNodeConfig,
    input_tokens: int,
    accelerators: Sequence[AcceleratorConfig],
    predictions: PredictionCache,
    history: Mapping[str, float],
    oom_penalties: Mapping[WorkflowModelFeatureKey, float],
    eps_mem_mb: float,
    hit_limit_rate_threshold: float = 0.1,
) -> TokenBudgetDecision:
    if input_tokens <= 0:
        raise ValueError("input_tokens must be positive")
    if not 0 <= hit_limit_rate_threshold <= 1:
        raise ValueError("hit_limit_rate_threshold must be between 0 and 1")

    feasible: list[_FeasibleCandidate] = []
    has_input_bucket = False
    has_output_bucket = False
    for accelerator in accelerators:
        sequence_length = _covering_sequence_length(
            predictions,
            node.model.name,
            accelerator.gpu_kind,
            input_tokens,
        )
        if sequence_length is None:
            continue
        has_input_bucket = True
        output_lengths = predictions.decode_output_lengths(
            node.model.name,
            accelerator.gpu_kind,
            sequence_length,
        )
        candidates = [
            output_length
            for output_length in output_lengths
            if node.token_budget.min_max_new_tokens
            <= output_length
            <= node.token_budget.max_max_new_tokens
        ]
        if candidates:
            has_output_bucket = True
        candidate = _largest_feasible_budget(
            node,
            accelerator,
            sequence_length,
            reversed(candidates),
            predictions,
            oom_penalties,
            eps_mem_mb,
        )
        if candidate is not None:
            feasible.append(candidate)

    if not feasible:
        if not has_input_bucket:
            reason = "input_bucket_missing"
        elif not has_output_bucket:
            reason = "prediction_bucket_missing"
        else:
            reason = "token_budget_infeasible"
        return TokenBudgetDecision(action=TokenBudgetAction.INFEASIBLE, reason=reason)

    chosen = _choose_placement(
        feasible,
        node.token_budget.default_max_new_tokens,
        history,
        hit_limit_rate_threshold,
    )
    granted = chosen.prediction.key.decode_output_length
    if granted == node.token_budget.default_max_new_tokens:
        action = TokenBudgetAction.FIXED
    elif granted > node.token_budget.default_max_new_tokens:
        action = TokenBudgetAction.UPSCALED
    else:
        action = TokenBudgetAction.DOWNSCALED
    return TokenBudgetDecision(
        accelerator_id=chosen.accelerator.accelerator_id,
        gpu_kind=chosen.accelerator.gpu_kind,
        sequence_length=chosen.prediction.key.sequence_length,
        granted_max_new_tokens=granted,
        prediction_key=chosen.prediction.key,
        predicted_run_sec=chosen.prediction.predicted_run_sec,
        predicted_peak_vram_mb=chosen.prediction.predicted_peak_vram_mb,
        effective_vram_mb=chosen.effective_vram_mb,
        action=action,
    )


def select_eviction_victim(
    candidates: Sequence[EvictionCandidate],
) -> EvictionCandidate | None:
    legal = [
        candidate
        for candidate in candidates
        if candidate.state in (ModelReplicaState.IDLE, ModelReplicaState.SUSPECT)
    ]
    if not legal:
        return None
    computable = [
        candidate
        for candidate in legal
        if candidate.reuse_distance_sec is not None
        and candidate.reload_cost_sec is not None
    ]
    if computable:
        return min(
            computable,
            key=lambda candidate: (
                -(candidate.reuse_distance_sec - candidate.reload_cost_sec),
                candidate.replica_id,
            ),
        )
    return min(
        legal,
        key=lambda candidate: (candidate.idle_since, candidate.replica_id),
    )


def _covering_sequence_length(
    predictions: PredictionCache,
    model_name: str,
    gpu_kind: GpuKind,
    input_tokens: int,
) -> int | None:
    return next(
        (
            sequence_length
            for sequence_length in predictions.decode_sequence_lengths(
                model_name, gpu_kind
            )
            if sequence_length >= input_tokens
        ),
        None,
    )


def _largest_feasible_budget(
    node: AgentNodeConfig,
    accelerator: AcceleratorConfig,
    sequence_length: int,
    output_lengths: Iterable[int],
    predictions: PredictionCache,
    oom_penalties: Mapping[WorkflowModelFeatureKey, float],
    eps_mem_mb: float,
) -> _FeasibleCandidate | None:
    for output_length in output_lengths:
        prediction = predictions.lookup_decode(
            model_name=node.model.name,
            gpu_kind=accelerator.gpu_kind,
            sequence_length=sequence_length,
            decode_output_length=output_length,
        )
        effective_vram_mb = (
            prediction.predicted_peak_vram_mb
            + eps_mem_mb
            + oom_penalties.get(prediction.key, 0.0)
        )
        if effective_vram_mb <= accelerator.total_mem_mb:
            return _FeasibleCandidate(
                accelerator=accelerator,
                prediction=prediction,
                effective_vram_mb=effective_vram_mb,
            )
    return None


def _choose_placement(
    candidates: Sequence[_FeasibleCandidate],
    default_max_new_tokens: int,
    history: Mapping[str, float],
    hit_limit_rate_threshold: float,
) -> _FeasibleCandidate:
    smallest = min(
        candidates,
        key=lambda candidate: (
            candidate.accelerator.total_mem_mb,
            -candidate.prediction.key.decode_output_length,
            candidate.prediction.predicted_run_sec,
            candidate.accelerator.accelerator_id,
        ),
    )
    if (
        smallest.prediction.key.decode_output_length >= default_max_new_tokens
        and history.get(smallest.accelerator.gpu_kind, 0.0) <= hit_limit_rate_threshold
    ):
        return smallest
    return min(
        candidates,
        key=lambda candidate: (
            -candidate.prediction.key.decode_output_length,
            candidate.prediction.predicted_run_sec,
            candidate.accelerator.accelerator_id,
        ),
    )
