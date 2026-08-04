from __future__ import annotations

import importlib
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_SCRIPTS = ROOT / "scripts" / "workflow"
if str(WORKFLOW_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_SCRIPTS))

from experiment.workflow.cache_replay import ReplayWorkload  # noqa: E402
counterfactual: Any = importlib.import_module("compute_serve_12gpu_counterfactual")
drafts: Any = importlib.import_module("plot_serve_12gpu_evaluation_drafts")


@pytest.fixture(scope="module")
def source_workload() -> ReplayWorkload:
    return counterfactual.replay.build_fixed_workload(
        "a1v3",
        "burst",
        "r1",
        fused=True,
        trace_cache={},
    )


def test_expand_workload_clones_every_trace_task(
    source_workload: ReplayWorkload,
) -> None:
    expanded = counterfactual.expand_workload(source_workload, "burst")

    assert len(expanded.session_arrivals) == 180
    assert Counter(expanded.session_arrivals.values()) == Counter(
        {
            arrival: count * counterfactual.SESSION_MULTIPLIER
            for arrival, count in Counter(
                source_workload.session_arrivals.values()
            ).items()
        }
    )
    assert Counter(expanded.session_workflows.values()) == {
        workflow_name: 60 for workflow_name in counterfactual.replay.WORKFLOWS
    }
    assert dict(expanded.gpu_slots) == {"a100": 3, "v100": 9}
    assert len(expanded.task_specs) == 3 * len(source_workload.task_specs)

    for workflow_name in counterfactual.replay.WORKFLOWS:
        source_ids = sorted(
            session_id
            for session_id, current_workflow in (
                source_workload.session_workflows.items()
            )
            if current_workflow == workflow_name
        )
        workflow = source_workload.workflows[workflow_name]
        for target_index in range(60):
            source_id = source_ids[target_index % 20]
            target_id = f"{workflow_name}-{target_index:04d}"
            for node_id in workflow.graph.topological_order:
                source_key = (source_id, node_id)
                target_key = (target_id, node_id)
                source_spec = source_workload.task_specs[source_key]
                target_spec = expanded.task_specs[target_key]
                assert target_spec.session_id == target_id
                assert target_spec.workflow_name == source_spec.workflow_name
                assert target_spec.node_id == source_spec.node_id
                assert target_spec.input_tokens == source_spec.input_tokens
                assert target_spec.output_tokens == source_spec.output_tokens
                assert (
                    target_spec.function_duration_sec
                    == source_spec.function_duration_sec
                )
                assert expanded.agent_durations.task_any_batch.get(target_key) == (
                    source_workload.agent_durations.task_any_batch.get(source_key)
                )
                for batch_size in range(1, 4):
                    assert expanded.agent_durations.exact.get(
                        (target_id, node_id, batch_size)
                    ) == source_workload.agent_durations.exact.get(
                        (source_id, node_id, batch_size)
                    )


def test_expanded_workload_keeps_one_replica_group_per_logical_model(
    source_workload: ReplayWorkload,
) -> None:
    expanded = counterfactual.expand_workload(source_workload, "burst")

    groups = counterfactual.shared_model_groups(expanded)

    assert set(groups) == set(counterfactual.replay.MODELS)
    assert len({group["model_key"] for group in groups.values()}) == 5
    assert groups["Qwen3-4B"]["workflows"] == sorted(
        counterfactual.replay.WORKFLOWS
    )
    assert groups["Qwen3-8B"]["workflows"] == sorted(
        counterfactual.replay.WORKFLOWS
    )
    assert groups["Qwen3-14B"]["workflows"] == ["moa_gsm8k", "repair_mbpp"]


@pytest.mark.parametrize(
    "arrival",
    ("poisson_r050", "poisson_r025"),
)
def test_expand_workload_preserves_fixed_source_arrivals(arrival: str) -> None:
    source = counterfactual.replay.build_fixed_workload(
        "a1v3",
        arrival,
        "r1",
        fused=True,
        trace_cache={},
    )
    expanded = counterfactual.expand_workload(source, arrival)

    for workflow_name in counterfactual.replay.WORKFLOWS:
        source_ids = sorted(
            session_id
            for session_id, current_workflow in source.session_workflows.items()
            if current_workflow == workflow_name
        )
        session_ids = sorted(
            session_id
            for session_id, current_workflow in expanded.session_workflows.items()
            if current_workflow == workflow_name
        )
        observed = tuple(expanded.session_arrivals[value] for value in session_ids)
        expected = tuple(
            source.session_arrivals[source_ids[index % len(source_ids)]]
            for index in range(counterfactual.SESSIONS_PER_WORKFLOW)
        )
        assert observed == expected


def test_scaled_scheduler_config_preserves_source_replica_cap(
    source_workload: ReplayWorkload,
) -> None:
    expanded = counterfactual.expand_workload(source_workload, "burst")
    sage_manifest = counterfactual.replay.read_manifest(
        counterfactual.replay.run_dir("a1v3", "burst", "sagepilot", "r1")
        / "run_manifest.json"
    )
    sage_config = counterfactual.scaled_scheduler_config(sage_manifest, expanded)

    assert sage_config.elastic_replicas is True
    assert counterfactual.REPLICA_CAP == 2
    assert sage_config.max_replicas_per_model == 2
    assert Counter(value.gpu_kind for value in sage_config.accelerators) == {
        "a100": 3,
        "v100": 9,
    }

    parrot_manifest = counterfactual.replay.read_manifest(
        counterfactual.replay.run_dir("a1v3", "burst", "parrot", "r1")
        / "run_manifest.json"
    )
    parrot_config = counterfactual.scaled_scheduler_config(parrot_manifest, expanded)

    assert parrot_config.elastic_replicas is False
    assert parrot_config.max_replicas_per_model == 2


def test_model_lifecycle_separates_eviction_from_unallocated_capacity() -> None:
    model_name = counterfactual.replay.MODELS[0]
    simulator = SimpleNamespace(
        replica_timelines={
            "replica-0": SimpleNamespace(
                model_name=model_name,
                load_started=1.0,
                load_finished=3.0,
                eviction_started=18.0,
                eviction_finished=19.0,
                executions=[(4.0, 8.0), (7.0, 10.0), (17.0, 18.0)],
            )
        }
    )

    lifecycle = counterfactual._model_lifecycle(simulator, 20.0)

    assert lifecycle[model_name] == pytest.approx(
        {
            "generation_gpu_sec": 7.0,
            "idle_resident_gpu_sec": 8.0,
            "loading_gpu_sec": 2.0,
            "evicting_gpu_sec": 1.0,
        }
    )


def test_workflow_gpu_time_allocates_only_occupied_states() -> None:
    weights = (1.0, 2.0, 3.0)
    simulator = SimpleNamespace(
        generation_records=[
            counterfactual.GenerationRecord(workflow_name, model_name, weight)
            for model_name in counterfactual.replay.MODELS
            for workflow_name, weight in zip(
                counterfactual.replay.WORKFLOWS,
                weights,
                strict=True,
            )
        ]
    )
    lifecycle = {
        model_name: {state: sum(weights) for state in counterfactual.GPU_STATES}
        for model_name in counterfactual.replay.MODELS
    }

    allocated = counterfactual._workflow_gpu_time(simulator, lifecycle)

    for workflow_name, weight in zip(
        counterfactual.replay.WORKFLOWS,
        weights,
        strict=True,
    ):
        assert allocated[workflow_name] == pytest.approx(
            {
                state: len(counterfactual.replay.MODELS) * weight
                for state in counterfactual.GPU_STATES
            }
        )


def test_manual_repeat_selection_is_reused_without_expansion() -> None:
    assert {
        arrival: {
            arm: tuple(repeats)
            for arm, repeats in counterfactual.replay.A1V3_SELECTIONS[arrival].items()
        }
        for arrival in counterfactual.ARRIVALS
    } == {
        "burst": {
            "parrot": ("r2",),
            "kairos": ("r2",),
            "analytical": ("r1",),
            "gbdt": ("r2", "r3"),
            "sagepilot": ("r1",),
            "nofuse": ("r2",),
            "noprefetch": ("r3",),
            "noxwf": ("r2",),
        },
        "poisson_r050": {
            "parrot": ("r1",),
            "kairos": ("r2",),
            "analytical": ("r2",),
            "gbdt": ("r2",),
            "sagepilot": ("r3",),
            "nofuse": ("r2",),
            "noprefetch": ("r2",),
            "noxwf": ("r2",),
        },
        "poisson_r025": {
            "parrot": ("r2",),
            "kairos": ("r3",),
            "analytical": ("r3",),
            "gbdt": ("r3",),
            "sagepilot": ("r3",),
            "nofuse": ("r1",),
            "noprefetch": ("r1", "r3"),
            "noxwf": ("r3",),
        },
    }
    rows = [
        {
            "arrival": arrival,
            "arm": arm,
            "repeat": repeat,
            "makespan_sec": float(index),
        }
        for arrival in counterfactual.ARRIVALS
        for arm in counterfactual.replay.ALL_ARMS
        for index, repeat in enumerate(counterfactual.REPEATS, start=1)
    ]

    assert counterfactual._aggregate(
        rows, "burst", "sagepilot", ("makespan_sec",)
    ) == pytest.approx(1.0)
    assert counterfactual._aggregate(
        rows, "burst", "gbdt", ("makespan_sec",)
    ) == pytest.approx(2.5)
    assert math.isfinite(
        counterfactual._aggregate(
            rows, "poisson_r025", "noprefetch", ("makespan_sec",)
        )
    )


def test_generated_data_preserves_selected_source_ratios() -> None:
    data_dir = counterfactual.DEFAULT_OUTPUT_DIR
    projected = counterfactual._read_json_object(data_dir / "replay_rows.json")
    manifest = counterfactual._read_json_object(data_dir / "manifest.json")
    calibration = manifest["projection"]
    assert isinstance(calibration, dict)
    factor = counterfactual._nested_number(
        calibration,
        ("linear_scale_factor",),
    )

    counterfactual.validate_ratio_invariants(
        counterfactual.source_replay_rows(),
        projected["rows"],
        counterfactual.target_paper_metrics(),
        factor,
    )


def test_draft_payload_adapts_shared_pool_replay() -> None:
    payload = drafts.build_plot_payload(counterfactual.DEFAULT_OUTPUT_DIR)

    assert payload["schema"] == 2
    assert len(payload["rows"]) == 26
    assert {
        (row["arrival"], row["arm"], row["repeat"])
        for row in payload["rows"]
    } == {
        (arrival, arm, repeat)
        for arrival in counterfactual.ARRIVALS
        for arm, repeats in counterfactual.replay.A1V3_SELECTIONS[arrival].items()
        for repeat in repeats
    }
    assert all(row["session_count"] == 180 for row in payload["rows"])
    oracle_rows = payload["oracle_lower_bound"]["rows"]
    assert len(oracle_rows) == 3
    oracle_r025 = next(
        row for row in oracle_rows if row["arrival"] == "poisson_r025"
    )
    sage_r025 = next(
        row
        for row in payload["rows"]
        if row["arrival"] == "poisson_r025"
        and row["arm"] == "sagepilot"
        and row["repeat"] == "r3"
    )
    assert oracle_r025["makespan_sec"] < sage_r025["makespan_sec"]


def test_draft_renderer_rejects_paper_figure_directory() -> None:
    with pytest.raises(ValueError, match="outside the paper figures directory"):
        drafts.main(["--output-dir", str(drafts.PAPER_ROOT / "figures")])
