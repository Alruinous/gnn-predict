from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from workflow.gnn_predictor import ToolPrediction
from workflow.schema import WorkflowNodeConfig


class WorkflowSchedulingError(RuntimeError):
    pass


class WorkflowDeviceState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    gpu_name: str
    total_memory_gb: float
    available_memory_gb: float

    @field_validator("name", "gpu_name")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("device text fields must not be empty")
        return normalized_value

    @field_validator("total_memory_gb", "available_memory_gb")
    @classmethod
    def validate_memory(cls, value: float) -> float:
        if value < 0:
            raise ValueError("device memory must be non-negative")
        return value


class WorkflowCachedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_key: str
    device_name: str
    memory_gb: float

    @field_validator("model_key", "device_name")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("cached model text fields must not be empty")
        return normalized_value

    @field_validator("memory_gb")
    @classmethod
    def validate_memory(cls, value: float) -> float:
        if value < 0:
            raise ValueError("cached model memory must be non-negative")
        return value


class WorkflowSchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    devices: list[WorkflowDeviceState]
    cached_models: list[WorkflowCachedModel] = Field(default_factory=list)
    memory_safety_margin_gb: float = 0.5

    @field_validator("devices")
    @classmethod
    def validate_devices(
        cls, value: list[WorkflowDeviceState]
    ) -> list[WorkflowDeviceState]:
        if not value:
            raise ValueError("devices must not be empty")
        names = [device.name for device in value]
        if len(names) != len(set(names)):
            raise ValueError("device names must be unique")
        return value

    @field_validator("memory_safety_margin_gb")
    @classmethod
    def validate_safety_margin(cls, value: float) -> float:
        if value < 0:
            raise ValueError("memory_safety_margin_gb must be non-negative")
        return value


class ToolScheduleDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_name: str
    device_name: str
    gpu_name: str
    prediction: ToolPrediction
    required_memory_gb: float
    score: float


class WorkflowScheduler:
    def __init__(self, config: WorkflowSchedulerConfig) -> None:
        self.memory_safety_margin_gb = config.memory_safety_margin_gb
        self.devices = {device.name: device.model_copy() for device in config.devices}
        self.cached_models = {
            (cached.device_name, cached.model_key): cached.model_copy()
            for cached in config.cached_models
        }

    def select_device(
        self,
        node: WorkflowNodeConfig,
        predictions: Sequence[ToolPrediction],
    ) -> ToolScheduleDecision:
        decisions = self.build_candidate_decisions(node, predictions, self.devices)
        fitting_decisions = [
            decision
            for decision in decisions
            if decision.required_memory_gb
            <= self.devices[decision.device_name].available_memory_gb
        ]
        if not fitting_decisions:
            raise WorkflowSchedulingError(
                f"no device has enough memory for workflow node: {node.name}"
            )
        return min(fitting_decisions, key=lambda decision: decision.score)

    def plan_ready_tools(
        self,
        predictions_by_node: Mapping[str, Sequence[ToolPrediction]],
    ) -> list[list[ToolScheduleDecision]]:
        remaining = {
            name: device.available_memory_gb for name, device in self.devices.items()
        }
        pending = sorted(predictions_by_node)
        batches: list[list[ToolScheduleDecision]] = []
        while pending:
            batch: list[ToolScheduleDecision] = []
            next_pending: list[str] = []
            for node_name in pending:
                decision = self.select_prediction_for_remaining_memory(
                    predictions_by_node[node_name],
                    remaining,
                )
                if decision is None:
                    next_pending.append(node_name)
                    continue
                batch.append(decision)
                remaining[decision.device_name] -= decision.required_memory_gb
            if not batch:
                raise WorkflowSchedulingError("ready tools cannot fit on any device")
            batches.append(batch)
            pending = next_pending
            remaining = {
                name: device.available_memory_gb
                for name, device in self.devices.items()
            }
        return batches

    def mark_deployed(
        self,
        node: WorkflowNodeConfig,
        decision: ToolScheduleDecision,
    ) -> None:
        model_key = build_model_key(node)
        device = self.devices[decision.device_name]
        device.available_memory_gb = max(
            0.0,
            device.available_memory_gb - decision.required_memory_gb,
        )
        self.cached_models[(decision.device_name, model_key)] = WorkflowCachedModel(
            model_key=model_key,
            device_name=decision.device_name,
            memory_gb=prediction_memory_gb(decision.prediction),
        )

    def build_candidate_decisions(
        self,
        node: WorkflowNodeConfig,
        predictions: Sequence[ToolPrediction],
        devices: Mapping[str, WorkflowDeviceState],
    ) -> list[ToolScheduleDecision]:
        model_key = build_model_key(node)
        decisions: list[ToolScheduleDecision] = []
        for prediction in predictions:
            device = devices[prediction.device_name]
            cached_memory = self.cached_models.get((prediction.device_name, model_key))
            required_memory_gb = prediction_memory_gb(prediction)
            if cached_memory is not None:
                required_memory_gb = max(
                    0.0, required_memory_gb - cached_memory.memory_gb
                )
            required_memory_gb += self.memory_safety_margin_gb
            decisions.append(
                ToolScheduleDecision(
                    node_name=node.name,
                    device_name=device.name,
                    gpu_name=device.gpu_name,
                    prediction=prediction,
                    required_memory_gb=required_memory_gb,
                    score=prediction_score(prediction, cached_memory is not None),
                )
            )
        return decisions

    def select_prediction_for_remaining_memory(
        self,
        predictions: Sequence[ToolPrediction],
        remaining: Mapping[str, float],
    ) -> ToolScheduleDecision | None:
        decisions = [
            build_decision_from_prediction(prediction, self.memory_safety_margin_gb)
            for prediction in predictions
            if prediction.device_name in remaining
        ]
        fitting_decisions = [
            decision
            for decision in decisions
            if decision.required_memory_gb <= remaining[decision.device_name]
        ]
        if not fitting_decisions:
            return None
        return min(fitting_decisions, key=lambda decision: decision.score)


def load_scheduler_config(path: str | Path) -> WorkflowSchedulerConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return WorkflowSchedulerConfig.model_validate(payload)


def build_model_key(node: WorkflowNodeConfig) -> str:
    assert node.model is not None
    payload = {
        "name": node.model.name,
        "parameters": node.model.parameters,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def build_decision_from_prediction(
    prediction: ToolPrediction,
    memory_safety_margin_gb: float,
) -> ToolScheduleDecision:
    return ToolScheduleDecision(
        node_name=prediction.node_name,
        device_name=prediction.device_name,
        gpu_name=prediction.gpu_name,
        prediction=prediction,
        required_memory_gb=prediction_memory_gb(prediction) + memory_safety_margin_gb,
        score=prediction_score(prediction, False),
    )


def prediction_memory_gb(prediction: ToolPrediction) -> float:
    gpu_mem_used_gb = prediction.metric("gpu_mem_used_mb_max") / 1024.0
    memory_delta_gb = prediction.metric("memory_delta_gb_max")
    return max(gpu_mem_used_gb, memory_delta_gb, 0.0)


def prediction_score(prediction: ToolPrediction, cached: bool) -> float:
    deployment_cost = (
        0.0 if cached else prediction.metric("deployment_duration_sec_avg")
    )
    run_cost = prediction.metric("run_duration_sec_avg")
    power_cost = prediction.metric("gpu_power_watts_avg") / 1000.0
    return deployment_cost + run_cost + power_cost
