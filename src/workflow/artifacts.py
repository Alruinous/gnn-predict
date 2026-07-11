from __future__ import annotations

from collections.abc import Mapping
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

GpuKind = Literal["v100", "a100"]
DeploymentProfileIndexKey = tuple[str, str, str, str]
OpenUnitInterval = Annotated[float, Field(gt=0, le=1)]
UnitInterval = Annotated[float, Field(ge=0, le=1)]


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
    accelerators: tuple[AcceleratorConfig, ...] = ()
    eps_mem_mb: NonNegativeFloat = 512.0
    eps_time_sec: NonNegativeFloat = 0.25
    max_tick_interval_sec: PositiveFloat = 0.5
    acquire_timeout_sec: PositiveFloat = 60.0
    grant_poll_interval_sec: PositiveFloat = 0.05
    eviction_timeout_sec: PositiveFloat = 30.0
    default_load_sec: PositiveFloat = 30.0
    history_ema_alpha: OpenUnitInterval = 0.2
    hit_limit_rate_threshold: UnitInterval = 0.1

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


class PredictionEntry(ArtifactModel):
    key: WorkflowModelFeatureKey
    predicted_run_sec: PositiveFloat
    predicted_peak_vram_mb: PositiveFloat
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


class PredictionCache(ArtifactModel):
    version: PositiveInt
    entries: tuple[PredictionEntry, ...]
    _index: Mapping[WorkflowModelFeatureKey, PredictionEntry] = PrivateAttr(
        default_factory=dict
    )

    @model_validator(mode="after")
    def build_index(self) -> Self:
        index: dict[WorkflowModelFeatureKey, PredictionEntry] = {}
        for entry in self.entries:
            if entry.key in index:
                raise ValueError(f"duplicate prediction key: {entry.key.model_dump()}")
            index[entry.key] = entry
        self._index = MappingProxyType(index)
        return self

    def lookup(self, key: WorkflowModelFeatureKey) -> PredictionEntry:
        return self._index[key]

    def lookup_decode(
        self,
        *,
        model_name: str,
        gpu_kind: GpuKind,
        sequence_length: int,
        decode_output_length: int,
    ) -> PredictionEntry:
        key = WorkflowModelFeatureKey(
            model_name=model_name,
            phase="decode",
            gpu_name=gpu_kind,
            batch_size=1,
            sequence_length=sequence_length,
            decode_output_length=decode_output_length,
        )
        return self.lookup(key)

    def decode_sequence_lengths(
        self, model_name: str, gpu_kind: GpuKind
    ) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    entry.key.sequence_length
                    for entry in self.entries
                    if entry.key.model_name == model_name
                    and entry.key.phase == "decode"
                    and entry.key.gpu_name == gpu_kind
                    and entry.key.batch_size == 1
                }
            )
        )

    def decode_output_lengths(
        self,
        model_name: str,
        gpu_kind: GpuKind,
        sequence_length: int,
    ) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    entry.key.decode_output_length
                    for entry in self.entries
                    if entry.key.model_name == model_name
                    and entry.key.phase == "decode"
                    and entry.key.gpu_name == gpu_kind
                    and entry.key.batch_size == 1
                    and entry.key.sequence_length == sequence_length
                }
            )
        )


class DeploymentProfileEntry(ArtifactModel):
    model_name: NonEmptyStr
    model_path: NonEmptyStr
    dtype: NonEmptyStr
    gpu_kind: GpuKind
    samples: PositiveInt | None = None
    load_sec_median: PositiveFloat | None = None
    load_sec_p95: PositiveFloat | None = None
    idle_vram_mb_median: NonNegativeFloat | None = None


class DeploymentProfile(ArtifactModel):
    version: PositiveInt
    environment: dict[str, JsonValue] = Field(default_factory=dict)
    entries: tuple[DeploymentProfileEntry, ...]
    _index: Mapping[DeploymentProfileIndexKey, DeploymentProfileEntry] = PrivateAttr(
        default_factory=dict
    )

    @model_validator(mode="after")
    def build_index(self) -> Self:
        index: dict[DeploymentProfileIndexKey, DeploymentProfileEntry] = {}
        for entry in self.entries:
            index_key = _deployment_profile_index_key(
                entry.model_name,
                entry.model_path,
                entry.dtype,
                entry.gpu_kind,
            )
            if index_key in index:
                raise ValueError(f"duplicate deployment profile key: {index_key}")
            index[index_key] = entry
        self._index = MappingProxyType(index)
        return self

    def lookup(
        self,
        model_name: str,
        model_path: str,
        dtype: str,
        gpu_kind: GpuKind,
    ) -> DeploymentProfileEntry | None:
        return self._index.get(
            _deployment_profile_index_key(model_name, model_path, dtype, gpu_kind)
        )

    def load_cost_sec(
        self,
        model_name: str,
        model_path: str,
        dtype: str,
        gpu_kind: GpuKind,
        default_load_sec: float,
    ) -> float:
        entry = self.lookup(model_name, model_path, dtype, gpu_kind)
        if entry is None:
            return default_load_sec
        if entry.load_sec_p95 is not None:
            return entry.load_sec_p95
        if entry.load_sec_median is not None:
            return entry.load_sec_median
        return default_load_sec


def load_prediction_cache(path: str | Path) -> PredictionCache:
    with Path(path).open(encoding="utf-8") as stream:
        return PredictionCache.model_validate(yaml.safe_load(stream))


def load_deployment_profile(path: str | Path) -> DeploymentProfile:
    with Path(path).open(encoding="utf-8") as stream:
        return DeploymentProfile.model_validate(yaml.safe_load(stream))


def _deployment_profile_index_key(
    model_name: str,
    model_path: str,
    dtype: str,
    gpu_kind: GpuKind,
) -> DeploymentProfileIndexKey:
    return model_name, model_path, dtype, gpu_kind
