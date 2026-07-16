from __future__ import annotations

from pathlib import Path
from typing import Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

Scenario = Literal["qmsum", "mbpp"]
Strategy = Literal["lg-batch", "wf-fifo", "wf-history", "wf-cache"]
Workload = Literal["burst", "open-loop"]
LoadPercent = Literal[75, 125]

EXPERIMENT_ID = "system_20260713_v2"
DEFAULT_OUTPUT_ROOT = Path("output/workflow/experiments") / EXPERIMENT_ID
DEFAULT_VLLM_PYTHON = Path("envs/vllm-v100/.venv/bin/python")


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSpec(FrozenModel):
    name: str
    path: Path


class ExperimentConfig(FrozenModel):
    experiment_id: str = EXPERIMENT_ID
    output_root: Path = DEFAULT_OUTPUT_ROOT
    qmsum_path: Path = Path("dataset/QMSum/data/ALL")
    mbpp_path: Path = Path("dataset/mbpp/sanitized-mbpp.json")
    vllm_python: Path = DEFAULT_VLLM_PYTHON
    seed: int = 42
    repetitions: PositiveInt = 5
    session_count: PositiveInt = 24
    telemetry_interval_sec: PositiveFloat = 0.2
    telemetry_baseline_method: Literal["per_gpu_first_sample_clamped_subtraction"] = (
        "per_gpu_first_sample_clamped_subtraction"
    )
    steady_state_start_fraction: float = Field(default=0.2, ge=0, lt=1)
    steady_state_end_fraction: float = Field(default=0.8, gt=0, le=1)
    trial_timeout_sec: PositiveFloat = 3_600.0
    campaign_timeout_sec: PositiveFloat = 86_400.0
    gpu_total_mem_mb: PositiveInt = 32_768
    gpu_idle_memory_limit_mb: float = Field(default=512.0, ge=0)
    qwen3_4b: ModelSpec = ModelSpec(
        name="Qwen3-4B",
        path=Path("/data/Models/Qwen/Qwen3-4B"),
    )
    qwen3_8b: ModelSpec = ModelSpec(
        name="Qwen3-8B",
        path=Path("/data/Models/Qwen/Qwen3-8B"),
    )
    qwen3_14b: ModelSpec = ModelSpec(
        name="Qwen3-14B",
        path=Path("/data/Models/Qwen/Qwen3-14B"),
    )

    @model_validator(mode="after")
    def validate_experiment_constants(self) -> Self:
        if self.repetitions != 5:
            raise ValueError("the formal experiment requires five repetitions")
        if self.session_count != 24:
            raise ValueError("the formal experiment requires twenty-four sessions")
        if self.steady_state_start_fraction >= self.steady_state_end_fraction:
            raise ValueError("steady-state window must be ordered")
        return self


class TrialSpec(FrozenModel):
    scenario: Scenario
    strategy: Strategy
    gpu_count: Literal[1, 2, 3]
    workload: Workload
    repetition: Literal[1, 2, 3, 4, 5]
    max_num_seqs: Literal[1, 2, 3] = 3
    queue_capacity: Literal[1, 4, 16] = 16
    load_percent: LoadPercent | None = None

    @model_validator(mode="after")
    def validate_workload(self) -> Self:
        if (self.workload == "open-loop") != (self.load_percent is not None):
            raise ValueError("open-loop trials require one load percentage")
        if self.strategy == "lg-batch":
            expected = 2 if self.scenario == "qmsum" else 3
            if self.gpu_count != expected:
                raise ValueError("LG-Batch uses the scenario's static GPU budget")
        return self

    @property
    def scheduler_policy(self) -> Literal["fifo", "history", "cache"] | None:
        if self.strategy == "lg-batch":
            return None
        return cast(
            Literal["fifo", "history", "cache"],
            self.strategy.removeprefix("wf-"),
        )

    @property
    def trial_id(self) -> str:
        parts = [
            self.scenario,
            self.strategy,
            f"g{self.gpu_count}",
            self.workload,
        ]
        if self.load_percent is not None:
            parts.append(f"load{self.load_percent:03d}")
        parts.extend(
            (
                f"seq{self.max_num_seqs}",
                f"queue{self.queue_capacity}",
                f"rep{self.repetition}",
            )
        )
        return "__".join(parts)
