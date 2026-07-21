from __future__ import annotations

from pathlib import Path

import yaml

from workflow.artifacts import SchedulerConfig, load_resource_contract_cache
from workflow.policy import select_placement
from workflow.schema import AgentNodeConfig, Workflow

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "config" / "workflow" / "multi_workflow_demo"


def load_workflow(name: str) -> Workflow:
    payload = yaml.safe_load((DEMO_DIR / name).read_text(encoding="utf-8"))
    return Workflow.model_validate(payload)


def load_scheduler_config() -> SchedulerConfig:
    payload = yaml.safe_load(
        (DEMO_DIR / "scheduler_config.yaml").read_text(encoding="utf-8")
    )
    return SchedulerConfig.model_validate(payload)


def test_demo_workflows_have_distinct_names_and_validate() -> None:
    quick_qa = load_workflow("quick_qa.yaml")
    long_report = load_workflow("long_report.yaml")

    assert quick_qa.workflow_name != long_report.workflow_name
    assert quick_qa.graph.entry_node == "prepare"
    assert long_report.graph.entry_node == "prepare"


def test_demo_scheduler_config_spans_two_gpu_kinds_on_two_hosts() -> None:
    config = load_scheduler_config()

    kinds = {accelerator.gpu_kind for accelerator in config.accelerators}
    hostnames = {accelerator.hostname for accelerator in config.accelerators}
    assert kinds == {"v100", "a100"}
    assert len(hostnames) == 2


def test_demo_predictions_place_each_workflow_on_its_intended_gpu_kind() -> None:
    quick_qa = load_workflow("quick_qa.yaml")
    long_report = load_workflow("long_report.yaml")
    config = load_scheduler_config()
    predictions = load_resource_contract_cache(DEMO_DIR / "predictions.yaml")

    qa_agent = quick_qa.node_map()["answer"]
    report_agent = long_report.node_map()["write"]
    assert isinstance(qa_agent, AgentNodeConfig)
    assert isinstance(report_agent, AgentNodeConfig)

    # Both accelerators are offered to each workflow; the prediction cache
    # alone (one GPU kind per model) must steer each onto the right host.
    qa_decision = select_placement(
        node=qa_agent,
        input_tokens=200,
        accelerators=config.accelerators,
        predictions=predictions,
        oom_penalties={},
        eps_mem_mb=config.eps_mem_mb,
    )
    report_decision = select_placement(
        node=report_agent,
        input_tokens=1000,
        accelerators=config.accelerators,
        predictions=predictions,
        oom_penalties={},
        eps_mem_mb=config.eps_mem_mb,
    )

    assert qa_decision.feasible is True
    assert qa_decision.gpu_kind == "v100"
    assert report_decision.feasible is True
    assert report_decision.gpu_kind == "a100"
