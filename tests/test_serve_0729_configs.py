"""The serve_0729 baseline matrix: the baseline arms must carry no SagePilot mechanism.

serve_0728 defined "gbdt" and "analytical" as SagePilot with its predictor swapped.
serve_0729 redefines the same two predictors as standalone baselines, so the thing worth
pinning here is that the redefinition actually took: a baseline arm that quietly kept
prefetch or elastic replicas would be indistinguishable from the old definition in the
figure, and only a trace audit would catch it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from workflow.artifacts import SchedulerConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config" / "workflow" / "serve" / "serve_0729"
SERVE_0728_DIR = ROOT / "config" / "workflow" / "serve" / "serve_0728"
RUNNER = ROOT / "scripts" / "workflow" / "run_serve_0729.sh"

BASELINE_ARMS = ("parrot", "kairos", "formula_baseline", "gbdt_baseline")
SAGEPILOT_ARMS = ("sagepilot", "formula_with_sagepilot", "gbdt_with_sagepilot")
ARMS = BASELINE_ARMS + SAGEPILOT_ARMS
ARRIVALS = ("burst", "poisson_r050")


def scheduler_config(name: str) -> SchedulerConfig:
    with (CONFIG_DIR / f"scheduler_{name}.yaml").open() as stream:
        return SchedulerConfig.model_validate(yaml.safe_load(stream))


def runner_line(arm: str) -> str:
    script = RUNNER.read_text(encoding="utf-8")
    return next(
        stripped
        for stripped in (item.strip() for item in script.splitlines())
        if stripped.startswith(f"{arm})")
    )


def runner_field(arm: str, key: str) -> str:
    return runner_line(arm).split(f"{key}=")[1].split(";")[0].strip()


@pytest.mark.parametrize("arm", ARMS)
def test_runner_maps_every_arm_to_files_that_exist(arm: str) -> None:
    # The runner is the only place the arm -> (scheduler, cache, fusion) mapping lives;
    # a stale path there fails a run minutes in, after the cluster is already busy.
    assert (CONFIG_DIR / f"{runner_field(arm, 'SCHED')}.yaml").is_file()
    assert (ROOT / runner_field(arm, "PREDICTIONS")).is_file()


@pytest.mark.parametrize("arm", BASELINE_ARMS)
def test_every_baseline_arm_runs_without_a_sagepilot_mechanism(arm: str) -> None:
    config = scheduler_config(runner_field(arm, "SCHED").removeprefix("scheduler_"))

    # Outside "cache" the mechanisms are off structurally: _collect_near_ready_tasks
    # returns [], _reload_cost returns None, _order_by_load_yield is a no-op and the
    # locality-first branch of _order_pending is not entered.
    assert config.policy in ("fifo", "kairos")
    assert not config.elastic_replicas
    assert config.min_residency_load_multiple == 0.0
    # Node fusion is a graph rewrite, not a scheduler field, so it is pinned separately.
    assert runner_field(arm, "FUSE") == ""


def test_the_three_fifo_baselines_differ_only_in_their_prediction_cache() -> None:
    schedulers = {
        arm: runner_field(arm, "SCHED") for arm in BASELINE_ARMS if arm != "kairos"
    }
    caches = {arm: runner_field(arm, "PREDICTIONS") for arm in schedulers}

    assert set(schedulers.values()) == {"scheduler_baseline_fifo"}, schedulers
    assert len(set(caches.values())) == len(caches), caches


@pytest.mark.parametrize("arm", SAGEPILOT_ARMS)
def test_every_sagepilot_arm_keeps_the_full_mechanism_set(arm: str) -> None:
    config = scheduler_config(runner_field(arm, "SCHED").removeprefix("scheduler_"))

    assert config.policy == "cache"
    assert config.elastic_replicas
    assert config.enable_prefetch
    assert config.cross_workflow_lifecycle
    assert config.min_residency_load_multiple == 1.0
    assert runner_field(arm, "FUSE") == "1"


def test_the_two_predictor_pairs_share_a_cache_across_the_two_definitions() -> None:
    # formula_baseline vs formula_with_sagepilot must isolate the scheduler, so the pair
    # has to read the same cache; the same for the gbdt pair.
    for baseline, with_sagepilot in (
        ("formula_baseline", "formula_with_sagepilot"),
        ("gbdt_baseline", "gbdt_with_sagepilot"),
    ):
        assert runner_field(baseline, "PREDICTIONS") == runner_field(
            with_sagepilot, "PREDICTIONS"
        )


@pytest.mark.parametrize("name", ("kairos", "sagepilot"))
def test_the_shared_arms_are_identical_to_their_serve_0728_counterparts(
    name: str,
) -> None:
    # The two datasets are meant to be read together, so a drift here would silently
    # make serve_0729's full-system cell a different system from serve_0728's.
    with (SERVE_0728_DIR / f"scheduler_{name}.yaml").open() as stream:
        reference = SchedulerConfig.model_validate(yaml.safe_load(stream))

    assert scheduler_config(name).model_dump() == reference.model_dump()


def test_every_arm_keeps_the_same_starvation_threshold() -> None:
    # _order_pending and _order_by_load_yield both derive aging from
    # 0.5 * acquire_timeout_sec, so an arm with a different value is not comparable.
    timeouts = {
        SchedulerConfig.model_validate(yaml.safe_load(path.open())).acquire_timeout_sec
        for path in CONFIG_DIR.glob("scheduler_*.yaml")
    }
    assert timeouts == {1800.0}


@pytest.mark.parametrize("arrival", ARRIVALS)
def test_the_runner_reuses_the_serve_0728_arrival_configs(arrival: str) -> None:
    # Reusing the originals rather than copying them makes the manifest's sha256 prove
    # both datasets were offered the identical session sequence.
    script = RUNNER.read_text(encoding="utf-8")

    assert 'ARRIVAL_CFG="config/workflow/serve/serve_0728"' in script
    assert (SERVE_0728_DIR / f"replay_{arrival}.yaml").is_file()
