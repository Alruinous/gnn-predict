from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    PrivateAttr,
    field_validator,
    model_validator,
)

from common.validate import NonEmptyStr
from workflow.types import WorkflowModelFeatureKey

GpuKind = NonEmptyStr
SchedulerPolicy = Literal["fifo", "history", "cache"]
OpenUnitInterval = Annotated[float, Field(gt=0, le=1)]


class ArtifactModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AcceleratorConfig(ArtifactModel):
    hostname: NonEmptyStr
    gpu_kind: GpuKind
    local_index: NonNegativeInt
    total_mem_mb: PositiveInt

    @property
    def accelerator_id(self) -> str:
        return f"{self.hostname}/{self.gpu_kind}:{self.local_index}"


class SchedulerConfig(ArtifactModel):
    policy: SchedulerPolicy = "cache"
    accelerators: tuple[AcceleratorConfig, ...] = ()
    vllm_python_executable: NonEmptyStr | None = None
    eps_mem_mb: NonNegativeFloat = 512.0
    eps_time_sec: NonNegativeFloat = 0.25
    max_tick_interval_sec: PositiveFloat = 0.5
    acquire_timeout_sec: PositiveFloat = 60.0
    grant_poll_interval_sec: PositiveFloat = 0.05
    eviction_timeout_sec: PositiveFloat = 30.0
    history_ema_alpha: OpenUnitInterval = 0.2

    @field_validator("vllm_python_executable")
    @classmethod
    def normalize_vllm_python_executable(cls, value: str | None) -> str | None:
        if value is None:
            return None
        # Resolving a venv Python symlink discards the environment's sys.prefix.
        return str(Path(value).expanduser().absolute())

    @model_validator(mode="after")
    def validate_accelerators(self) -> Self:
        positions = [
            (accelerator.hostname, accelerator.local_index)
            for accelerator in self.accelerators
        ]
        if len(positions) != len(set(positions)):
            raise ValueError("host/local accelerator indexes must be unique")
        host_kinds: dict[str, set[GpuKind]] = {}
        host_memory: dict[str, set[int]] = {}
        for accelerator in self.accelerators:
            host_kinds.setdefault(accelerator.hostname, set()).add(accelerator.gpu_kind)
            host_memory.setdefault(accelerator.hostname, set()).add(
                accelerator.total_mem_mb
            )
        if any(len(kinds) != 1 for kinds in host_kinds.values()):
            raise ValueError("one Ray hostname must expose one GPU kind")
        if any(len(sizes) != 1 for sizes in host_memory.values()):
            raise ValueError("one Ray hostname must expose one GPU memory size")
        return self


class ResourceContractSource(StrEnum):
    """Where a contract's numbers came from — never affects scheduling, trace-only."""

    EMPIRICAL_PROFILE = "empirical_profile"
    SYNTHETIC_FIXTURE = "synthetic_fixture"
    GNN_PREDICTED = "gnn_predicted"


class ResourceEvidence(ArtifactModel):
    method: Literal[
        "repeated_sample_margin", "fixed_margin_fallback", "point_estimate_only"
    ]
    sample_count: PositiveInt
    margin_fraction: NonNegativeFloat | None = None


class ResourceContract(ArtifactModel):
    key: WorkflowModelFeatureKey
    source: ResourceContractSource
    predicted_load_sec: PositiveFloat
    predicted_run_sec: PositiveFloat
    predicted_peak_vram_mb: PositiveFloat
    peak_vram_mb_upper_bound: PositiveFloat
    peak_vram_mb_evidence: ResourceEvidence
    run_sec_upper_bound: PositiveFloat | None = None
    run_sec_evidence: ResourceEvidence | None = None
    predicted_power_watts: NonNegativeFloat | None = None
    predictor_metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("key", mode="before")
    @classmethod
    def reject_unknown_key_fields(cls, value: object) -> object:
        if isinstance(value, Mapping):
            unknown = set(value) - set(WorkflowModelFeatureKey.model_fields)
            if unknown:
                names = ", ".join(sorted(str(name) for name in unknown))
                raise ValueError(f"unknown prediction key fields: {names}")
        return value

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.peak_vram_mb_upper_bound < self.predicted_peak_vram_mb:
            raise ValueError(
                "peak_vram_mb_upper_bound must be >= predicted_peak_vram_mb"
            )
        if (
            self.run_sec_upper_bound is not None
            and self.run_sec_upper_bound < self.predicted_run_sec
        ):
            raise ValueError("run_sec_upper_bound must be >= predicted_run_sec")
        return self


class ResourceContractCache(ArtifactModel):
    version: PositiveInt
    environment: dict[str, JsonValue] = Field(default_factory=dict)
    entries: tuple[ResourceContract, ...]
    _index: Mapping[WorkflowModelFeatureKey, ResourceContract] = PrivateAttr(
        default_factory=dict
    )

    @model_validator(mode="after")
    def build_index(self) -> Self:
        index: dict[WorkflowModelFeatureKey, ResourceContract] = {}
        for entry in self.entries:
            if entry.key in index:
                raise ValueError(f"duplicate prediction key: {entry.key.model_dump()}")
            index[entry.key] = entry
        self._index = MappingProxyType(index)
        return self

    def lookup(self, key: WorkflowModelFeatureKey) -> ResourceContract:
        return self._index[key]

    def lookup_decode(
        self,
        *,
        model_name: str,
        gpu_kind: GpuKind,
        batch_size: int = 1,
        sequence_length: int,
        decode_output_length: int,
    ) -> ResourceContract:
        key = WorkflowModelFeatureKey(
            model_name=model_name,
            phase="decode",
            gpu_name=gpu_kind,
            batch_size=batch_size,
            sequence_length=sequence_length,
            decode_output_length=decode_output_length,
        )
        return self.lookup(key)

    def decode_sequence_lengths(
        self,
        model_name: str,
        gpu_kind: GpuKind,
        batch_size: int = 1,
    ) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    entry.key.sequence_length
                    for entry in self.entries
                    if entry.key.model_name == model_name
                    and entry.key.phase == "decode"
                    and entry.key.gpu_name == gpu_kind
                    and entry.key.batch_size == batch_size
                }
            )
        )

    def decode_output_lengths(
        self,
        model_name: str,
        gpu_kind: GpuKind,
        sequence_length: int,
        batch_size: int = 1,
    ) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    entry.key.decode_output_length
                    for entry in self.entries
                    if entry.key.model_name == model_name
                    and entry.key.phase == "decode"
                    and entry.key.gpu_name == gpu_kind
                    and entry.key.batch_size == batch_size
                    and entry.key.sequence_length == sequence_length
                }
            )
        )

    def missing_decode_batch_keys(
        self,
        model_name: str,
        gpu_kind: GpuKind,
        max_batch_size: int,
    ) -> tuple[WorkflowModelFeatureKey, ...]:
        base_keys = sorted(
            (
                entry.key
                for entry in self.entries
                if entry.key.model_name == model_name
                and entry.key.phase == "decode"
                and entry.key.gpu_name == gpu_kind
                and entry.key.batch_size == 1
            ),
            key=lambda key: (key.sequence_length, key.decode_output_length),
        )
        missing: list[WorkflowModelFeatureKey] = []
        for batch_size in range(1, max_batch_size + 1):
            for base_key in base_keys:
                key = base_key.model_copy(update={"batch_size": batch_size})
                if key not in self._index:
                    missing.append(key)
        return tuple(missing)


def load_resource_contract_cache(path: str | Path) -> ResourceContractCache:
    with Path(path).open(encoding="utf-8") as stream:
        return ResourceContractCache.model_validate(yaml.safe_load(stream))
