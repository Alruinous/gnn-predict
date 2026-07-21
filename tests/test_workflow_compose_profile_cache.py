from __future__ import annotations

from pathlib import Path

import pytest

from scripts.workflow.compose_profile_cache import (
    ProfileSource,
    compose_prediction_cache,
    parse_profile_source,
    write_prediction_cache,
)
from workflow.artifacts import (
    AcceleratorConfig,
    ResourceContract,
    ResourceContractCache,
    ResourceContractSource,
    ResourceEvidence,
    SchedulerConfig,
)
from workflow.policy import select_placement
from workflow.schema import AgentNodeConfig
from workflow.types import WorkflowModelFeatureKey


def prediction(gpu_kind: str, run_sec: float) -> ResourceContract:
    return ResourceContract(
        key=WorkflowModelFeatureKey(
            model_name="test-model",
            phase="decode",
            gpu_name=gpu_kind,
            batch_size=1,
            sequence_length=128,
            decode_output_length=32,
        ),
        source=ResourceContractSource.SYNTHETIC_FIXTURE,
        predicted_load_sec=2.0,
        predicted_run_sec=run_sec,
        predicted_peak_vram_mb=1024.0,
        peak_vram_mb_upper_bound=1024.0,
        peak_vram_mb_evidence=ResourceEvidence(
            method="point_estimate_only", sample_count=1
        ),
    )


def write_source(
    path: Path, entries: tuple[ResourceContract, ...], *, version: int = 2
) -> None:
    write_prediction_cache(
        path,
        ResourceContractCache(version=version, environment={}, entries=entries),
    )


def test_profile_source_accepts_generic_gpu_kind() -> None:
    source = parse_profile_source(" P100 = cache/p100.yaml ")

    assert source == ProfileSource("p100", Path("cache/p100.yaml"))


def test_compose_selects_only_the_declared_kind_from_each_source(
    tmp_path: Path,
) -> None:
    legacy_v100 = tmp_path / "legacy-v100.yaml"
    measured_a100 = tmp_path / "measured-a100.yaml"
    write_source(
        legacy_v100,
        (prediction("v100", 4.0), prediction("a100", 4.0)),
    )
    write_source(measured_a100, (prediction("a100", 2.0),))

    cache = compose_prediction_cache(
        [
            ProfileSource("v100", legacy_v100),
            ProfileSource("a100", measured_a100),
        ]
    )

    assert [(entry.key.gpu_name, entry.predicted_run_sec) for entry in cache.entries] == [
        ("a100", 2.0),
        ("v100", 4.0),
    ]


def test_compose_rejects_duplicate_gpu_kind_sources(tmp_path: Path) -> None:
    v100_path = tmp_path / "v100.yaml"
    write_source(v100_path, (prediction("v100", 4.0),))

    with pytest.raises(ValueError, match="duplicate GPU kind"):
        compose_prediction_cache(
            [ProfileSource("v100", v100_path), ProfileSource("v100", v100_path)]
        )


def test_compose_rejects_mismatched_cache_versions(tmp_path: Path) -> None:
    v100_path = tmp_path / "v100.yaml"
    a100_path = tmp_path / "a100.yaml"
    write_source(v100_path, (prediction("v100", 4.0),), version=1)
    write_source(a100_path, (prediction("a100", 2.0),), version=2)

    with pytest.raises(ValueError, match="version mismatch"):
        compose_prediction_cache(
            [ProfileSource("v100", v100_path), ProfileSource("a100", a100_path)]
        )


def agent_node() -> AgentNodeConfig:
    return AgentNodeConfig.model_validate(
        {
            "name": "agent",
            "type": "agent",
            "model": {"name": "test-model"},
            "execution": {
                "model_path": "/models/test-model",
                "max_new_tokens": 32,
                "dtype": "float16",
                "serving": {
                    "max_model_len": 256,
                    "max_num_seqs": 1,
                    "max_num_batched_tokens": 256,
                },
            },
            "prompt_template": "{content}",
        }
    )


def test_composed_heterogeneous_cache_dispatches_on_either_gpu_kind(
    tmp_path: Path,
) -> None:
    v100_path = tmp_path / "v100.yaml"
    a100_path = tmp_path / "a100.yaml"
    write_source(v100_path, (prediction("v100", 4.0),))
    write_source(a100_path, (prediction("a100", 2.0),))
    cache = compose_prediction_cache(
        [ProfileSource("v100", v100_path), ProfileSource("a100", a100_path)]
    )

    # A SchedulerConfig spanning two GPU kinds on two hosts must validate.
    config = SchedulerConfig(
        accelerators=(
            AcceleratorConfig(
                hostname="v100-node",
                gpu_kind="v100",
                local_index=0,
                total_mem_mb=16_000,
            ),
            AcceleratorConfig(
                hostname="a100-node",
                gpu_kind="a100",
                local_index=0,
                total_mem_mb=40_000,
            ),
        )
    )
    node = agent_node()

    v100_only = select_placement(
        node=node,
        input_tokens=100,
        accelerators=[a for a in config.accelerators if a.gpu_kind == "v100"],
        predictions=cache,
        oom_penalties={},
        eps_mem_mb=config.eps_mem_mb,
    )
    a100_only = select_placement(
        node=node,
        input_tokens=100,
        accelerators=[a for a in config.accelerators if a.gpu_kind == "a100"],
        predictions=cache,
        oom_penalties={},
        eps_mem_mb=config.eps_mem_mb,
    )

    assert v100_only.feasible is True
    assert v100_only.gpu_kind == "v100"
    assert a100_only.feasible is True
    assert a100_only.gpu_kind == "a100"
