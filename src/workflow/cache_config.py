from __future__ import annotations

import pickle
from itertools import product
from pathlib import Path
from typing import Literal, cast

import yaml
from pydantic import BaseModel, Field, PositiveInt, field_validator
from torch_geometric.data import Data

from common.validate import NonEmptyStr
from workflow.types import WorkflowModelFeatureKey

WorkflowPhase = Literal["prefill", "decode"]
TorchDtypeName = Literal["float16", "bfloat16", "float32"]


class GraphCacheKey(BaseModel):
    model_name: NonEmptyStr
    phase: WorkflowPhase
    batch_size: PositiveInt
    sequence_length: PositiveInt
    decode_output_length: int = Field(ge=0)

    def digest_payload(self) -> str:
        return "|".join(
            (
                self.model_name,
                self.phase,
                str(self.batch_size),
                str(self.sequence_length),
                str(self.decode_output_length),
            )
        )


class GraphCacheGroup(BaseModel):
    name: NonEmptyStr
    model_name: NonEmptyStr
    model_path: Path
    phases: list[WorkflowPhase] = cast(
        list[WorkflowPhase],
        Field(default_factory=lambda: ["prefill", "decode"]),
    )
    batch_size_list: list[PositiveInt]
    sequence_length_list: list[PositiveInt]
    decode_max_output_length_list: list[PositiveInt] = Field(default_factory=list)
    dtype: TorchDtypeName = "float16"

    @field_validator(
        "phases",
        "batch_size_list",
        "sequence_length_list",
        "decode_max_output_length_list",
    )
    @classmethod
    def validate_unique_values(cls, value: list[object]) -> list[object]:
        if len(value) != len(set(value)):
            raise ValueError("list values must be unique")
        return value

    @field_validator(
        "phases",
        "batch_size_list",
        "sequence_length_list",
    )
    @classmethod
    def validate_non_empty_values(cls, value: list[object]) -> list[object]:
        if not value:
            raise ValueError("list values must not be empty")
        return value


class GraphCacheConfig(BaseModel):
    cache_groups: list[GraphCacheGroup]

    @field_validator("cache_groups")
    @classmethod
    def validate_unique_group_names(
        cls,
        value: list[GraphCacheGroup],
    ) -> list[GraphCacheGroup]:
        names = [group.name for group in value]
        if len(names) != len(set(names)):
            raise ValueError("cache group names must be unique")
        return value


class GraphCacheExportSpec(BaseModel):
    group_name: NonEmptyStr
    model_path: Path
    dtype: TorchDtypeName
    key: GraphCacheKey


def load_graph_cache_config(path: Path) -> GraphCacheConfig:
    with path.open() as f:
        raw = yaml.safe_load(f)
    return GraphCacheConfig.model_validate(raw)


def expand_graph_cache_specs(
    config: GraphCacheConfig,
    *,
    selected_group_names: set[str] | None = None,
    selected_phases: set[WorkflowPhase] | None = None,
) -> list[GraphCacheExportSpec]:
    specs: list[GraphCacheExportSpec] = []
    seen: set[tuple[str, str, int, int, int]] = set()
    for group in config.cache_groups:
        if selected_group_names is not None and group.name not in selected_group_names:
            continue
        phases = [
            phase
            for phase in group.phases
            if selected_phases is None or phase in selected_phases
        ]
        for phase, batch_size, sequence_length in product(
            phases,
            group.batch_size_list,
            group.sequence_length_list,
        ):
            decode_lengths = [0]
            if phase == "decode":
                if not group.decode_max_output_length_list:
                    raise ValueError(
                        f"cache group {group.name} requires decode lengths"
                    )
                decode_lengths = group.decode_max_output_length_list
            for decode_output_length in decode_lengths:
                key = GraphCacheKey(
                    model_name=group.model_name,
                    phase=phase,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    decode_output_length=decode_output_length,
                )
                ident = (
                    key.model_name,
                    key.phase,
                    key.batch_size,
                    key.sequence_length,
                    key.decode_output_length,
                )
                if ident in seen:
                    continue
                seen.add(ident)
                specs.append(
                    GraphCacheExportSpec(
                        group_name=group.name,
                        model_path=group.model_path,
                        dtype=group.dtype,
                        key=key,
                    )
                )
    return specs


class GraphFeatureCacheError(RuntimeError):
    pass


def load_graph_feature_cache(cache_dir: Path) -> dict[WorkflowModelFeatureKey, Data]:
    with open(cache_dir / "manifest.yaml") as f:
        manifest: dict = yaml.safe_load(f)

    features: dict[WorkflowModelFeatureKey, Data] = {}

    for entry in manifest["entries"]:
        entry_key = WorkflowModelFeatureKey.model_validate(entry["key"])
        expected_name = f"{entry_key.stable_digest}.pkl"
        cached_path = cache_dir / expected_name
        if not cached_path.exists() or not cached_path.is_file():
            raise GraphFeatureCacheError(
                f"missing cached graph feature for {entry_key} at {cached_path}"
            )
        with cached_path.open("rb") as f:
            key, data = pickle.load(f)
        if not isinstance(key, WorkflowModelFeatureKey):
            raise GraphFeatureCacheError(
                f"cached graph feature for {key} at {cached_path} has invalid key type"
            )
        if key.stable_digest != entry_key.stable_digest:
            raise GraphFeatureCacheError(
                f"cached graph feature for {key} at {cached_path} "
                "has mismatched stable_digest"
            )
        if not isinstance(data, Data):
            raise GraphFeatureCacheError(
                f"cached graph feature for {key} at {cached_path} is not a Data object"
            )
        features[key] = data

    return features
