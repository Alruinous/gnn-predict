from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    ResourceContract,
    ResourceContractCache,
)
from workflow.schema import AgentNodeConfig
from workflow.types import (
    ModelReplicaState,
    WorkflowModelFeatureKey,
)


class PolicyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlacementDecision(PolicyModel):
    accelerator_id: NonEmptyStr | None = None
    gpu_kind: GpuKind | None = None
    sequence_length: PositiveInt | None = None
    max_new_tokens: PositiveInt | None = None
    prediction_key: WorkflowModelFeatureKey | None = None
    predicted_load_sec: PositiveFloat | None = None
    predicted_run_sec: PositiveFloat | None = None
    predicted_peak_vram_mb: PositiveFloat | None = None
    effective_vram_mb: PositiveFloat | None = None
    feasible: bool
    reason: (
        Literal[
            "input_bucket_missing",
            "prediction_bucket_missing",
            "request_infeasible",
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
    prediction: ResourceContract
    effective_vram_mb: float


def select_placement(
    *,
    node: AgentNodeConfig,
    input_tokens: int,
    accelerators: Sequence[AcceleratorConfig],
    predictions: ResourceContractCache,
    oom_penalties: Mapping[WorkflowModelFeatureKey, float],
    eps_mem_mb: float,
    batch_size: int = 1,
) -> PlacementDecision:
    if input_tokens <= 0:
        raise ValueError("input_tokens must be positive")
    if (
        input_tokens + node.execution.max_new_tokens
        > node.execution.serving.max_model_len
    ):
        return PlacementDecision(feasible=False, reason="request_infeasible")

    feasible: list[_FeasibleCandidate] = []
    has_input_bucket = False
    has_output_bucket = False
    for accelerator in accelerators:
        sequence_length = _covering_sequence_length(
            predictions,
            node.model.name,
            accelerator.gpu_kind,
            input_tokens,
            batch_size,
        )
        if sequence_length is None:
            continue
        has_input_bucket = True
        output_lengths = predictions.decode_output_lengths(
            node.model.name,
            accelerator.gpu_kind,
            sequence_length,
            batch_size,
        )
        output_length = next(
            (
                candidate
                for candidate in output_lengths
                if candidate >= node.execution.max_new_tokens
            ),
            None,
        )
        if output_length is not None:
            has_output_bucket = True
        candidate = _feasible_placement(
            accelerator=accelerator,
            model_name=node.model.name,
            sequence_length=sequence_length,
            output_length=output_length,
            predictions=predictions,
            oom_penalties=oom_penalties,
            eps_mem_mb=eps_mem_mb,
            batch_size=batch_size,
        )
        if candidate is not None:
            feasible.append(candidate)

    if not feasible:
        if not has_input_bucket:
            reason = "input_bucket_missing"
        elif not has_output_bucket:
            reason = "prediction_bucket_missing"
        else:
            reason = "request_infeasible"
        return PlacementDecision(feasible=False, reason=reason)

    chosen = min(
        feasible,
        key=lambda candidate: (
            candidate.accelerator.total_mem_mb,
            candidate.prediction.predicted_run_sec,
            candidate.accelerator.accelerator_id,
        ),
    )
    return PlacementDecision(
        accelerator_id=chosen.accelerator.accelerator_id,
        gpu_kind=chosen.accelerator.gpu_kind,
        sequence_length=chosen.prediction.key.sequence_length,
        max_new_tokens=node.execution.max_new_tokens,
        prediction_key=chosen.prediction.key,
        predicted_load_sec=chosen.prediction.predicted_load_sec,
        predicted_run_sec=chosen.prediction.predicted_run_sec,
        predicted_peak_vram_mb=chosen.prediction.predicted_peak_vram_mb,
        effective_vram_mb=chosen.effective_vram_mb,
        feasible=True,
    )


def _feasible_placement(
    *,
    accelerator: AcceleratorConfig,
    model_name: str,
    sequence_length: int,
    output_length: int | None,
    predictions: ResourceContractCache,
    oom_penalties: Mapping[WorkflowModelFeatureKey, float],
    eps_mem_mb: float,
    batch_size: int,
) -> _FeasibleCandidate | None:
    if output_length is None:
        return None
    prediction = predictions.lookup_decode(
        model_name=model_name,
        gpu_kind=accelerator.gpu_kind,
        batch_size=batch_size,
        sequence_length=sequence_length,
        decode_output_length=output_length,
    )
    effective_vram_mb = (
        prediction.peak_vram_mb_upper_bound
        + eps_mem_mb
        + oom_penalties.get(prediction.key, 0.0)
    )
    if effective_vram_mb > accelerator.total_mem_mb:
        return None
    return _FeasibleCandidate(
        accelerator=accelerator,
        prediction=prediction,
        effective_vram_mb=effective_vram_mb,
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
    predictions: ResourceContractCache,
    model_name: str,
    gpu_kind: GpuKind,
    input_tokens: int,
    batch_size: int,
) -> int | None:
    return next(
        (
            sequence_length
            for sequence_length in predictions.decode_sequence_lengths(
                model_name, gpu_kind, batch_size
            )
            if sequence_length >= input_tokens
        ),
        None,
    )
