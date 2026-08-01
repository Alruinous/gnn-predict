"""The serve_0728 matrix configs: every cell must parse, and the factors must differ.

A typo in one of these YAMLs produces a run that looks valid and only reveals itself
as the wrong cell after ~15 GPU-minutes, so the arms are pinned here instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from experiment.workflow.experiments.dataset_replay import _arrival_offsets
from workflow.artifacts import SchedulerConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config" / "workflow" / "serve" / "serve_0728"
RUNNER = ROOT / "scripts" / "workflow" / "run_serve_0728.sh"

ARRIVALS = ("burst", "poisson_r0125", "poisson_r025", "poisson_r050")
ARMS = (
    "sagepilot",
    "analytical",
    "gbdt",
    "nofuse",
    "noprefetch",
    "noxwf",
    "parrot",
    "kairos",
)
# The reference cell and the three ablations must share one cache, or each ablation
# differs from the reference in two factors and its effect cannot be attributed.
REFERENCE_CACHE_ARMS = ("sagepilot", "nofuse", "noprefetch", "noxwf")
EXPECTED_SESSION_COUNT = 60


def scheduler_config(name: str) -> SchedulerConfig:
    with (CONFIG_DIR / f"scheduler_{name}.yaml").open() as stream:
        return SchedulerConfig.model_validate(yaml.safe_load(stream))


def replay_config(name: str) -> dict[str, object]:
    with (CONFIG_DIR / f"replay_{name}.yaml").open() as stream:
        return yaml.safe_load(stream)


def runner_line(arm: str) -> str:
    script = RUNNER.read_text(encoding="utf-8")
    return next(
        stripped
        for stripped in (item.strip() for item in script.splitlines())
        if stripped.startswith(f"{arm})")
    )


def test_reference_cell_has_every_mechanism_on() -> None:
    config = scheduler_config("sagepilot")

    assert config.policy == "cache"
    assert config.elastic_replicas
    assert config.enable_prefetch
    assert config.cross_workflow_lifecycle


@pytest.mark.parametrize(
    ("name", "field"),
    [
        ("sagepilot_noprefetch", "enable_prefetch"),
        ("sagepilot_noxwf", "cross_workflow_lifecycle"),
    ],
)
def test_each_ablation_flips_exactly_one_field(name: str, field: str) -> None:
    reference = scheduler_config("sagepilot").model_dump()
    ablation = scheduler_config(name).model_dump()

    assert ablation[field] is False
    assert {key for key in reference if reference[key] != ablation[key]} == {field}


def test_every_cache_policy_arm_carries_the_flap_guard() -> None:
    # An arm without it would thrash on a pool whose device count equals its model
    # count, which is exactly the a1v4 configuration in this matrix.
    for name in ("sagepilot", "sagepilot_noprefetch", "sagepilot_noxwf"):
        assert scheduler_config(name).min_residency_load_multiple == 1.0


def test_baselines_use_published_orderings_without_lifecycle_extras() -> None:
    assert scheduler_config("parrot").policy == "fifo"
    assert scheduler_config("kairos").policy == "kairos"
    for name in ("parrot", "kairos"):
        # elastic_replicas requires the cache policy, so the baselines must not carry it.
        assert not scheduler_config(name).elastic_replicas


def test_every_arm_keeps_the_same_starvation_threshold() -> None:
    # _order_pending and _order_by_load_yield both derive aging from
    # 0.5 * acquire_timeout_sec, so an arm with a different value is not comparable.
    timeouts = {
        SchedulerConfig.model_validate(yaml.safe_load(path.open())).acquire_timeout_sec
        for path in CONFIG_DIR.glob("scheduler_*.yaml")
    }
    assert timeouts == {1800.0}


@pytest.mark.parametrize("arrival", ARRIVALS)
def test_every_arrival_config_offers_the_same_sessions_on_the_same_seed(
    arrival: str,
) -> None:
    config = replay_config(arrival)

    assert config["arrival_seed"] == 42
    assert sum(scenario["sample_count"] for scenario in config["scenarios"]) == (
        EXPECTED_SESSION_COUNT
    )
    assert {scenario["workflow_name"] for scenario in config["scenarios"]} == {
        "moa_gsm8k",
        "repair_mbpp",
        "chain_qmsum",
    }


def test_the_three_poisson_rates_are_one_arrival_sequence_at_three_speeds() -> None:
    # expovariate(lambda) = -ln(U)/lambda over an RNG seeded only from
    # (arrival_seed, "arrival", workflow_name), so halving the rate doubles every
    # offset. That makes the rate dimension a paired comparison with no new jitter.
    offsets = {
        arrival: _arrival_offsets(replay_config(arrival)["scenarios"][0], 20, seed=42)
        for arrival in ("poisson_r0125", "poisson_r025", "poisson_r050")
    }

    for index, base in enumerate(offsets["poisson_r025"]):
        assert offsets["poisson_r0125"][index] == pytest.approx(base * 2.0)
        assert offsets["poisson_r050"][index] == pytest.approx(base / 2.0)


def test_burst_offers_every_session_at_once() -> None:
    scenario = replay_config("burst")["scenarios"][0]

    assert set(_arrival_offsets(scenario, 20, seed=42)) == {0.0}


@pytest.mark.parametrize("arm", ARMS)
def test_runner_maps_every_arm_to_files_that_exist(arm: str) -> None:
    # The runner is the only place the arm -> (scheduler, cache, fusion) mapping lives;
    # a stale path there fails a run minutes in, after the cluster is already busy.
    line = runner_line(arm)
    scheduler = line.split("SCHED=")[1].split(";")[0].strip()
    predictions = line.split("PREDICTIONS=")[1].split(";")[0].strip()

    assert (CONFIG_DIR / f"{scheduler}.yaml").is_file()
    assert (ROOT / predictions).is_file()


def test_reference_and_ablation_arms_share_one_prediction_cache() -> None:
    caches = {
        arm: runner_line(arm).split("PREDICTIONS=")[1].split(";")[0].strip()
        for arm in REFERENCE_CACHE_ARMS
    }

    assert len(set(caches.values())) == 1, caches


def test_predictor_arms_each_use_their_own_cache() -> None:
    reference = runner_line("sagepilot").split("PREDICTIONS=")[1].split(";")[0].strip()
    for arm in ("analytical", "gbdt"):
        cache = runner_line(arm).split("PREDICTIONS=")[1].split(";")[0].strip()
        assert cache != reference


@pytest.mark.parametrize("arrival", ARRIVALS)
def test_runner_arrival_names_match_the_committed_configs(arrival: str) -> None:
    assert f"replay_{arrival}.yaml" in {path.name for path in CONFIG_DIR.iterdir()}
    assert arrival in RUNNER.read_text(encoding="utf-8")
