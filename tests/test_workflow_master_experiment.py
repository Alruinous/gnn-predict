from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
import yaml

from langchain.agents import AgentState
from langchain_core.messages import AIMessage

from experiment.workflow.experiments import dataset_replay, static_inputs
from workflow import master
from workflow.artifacts import SchedulerConfig
from workflow.fleet import WorkflowFleet
from workflow.schema import Workflow
from workflow.worker import message_text


def echo(
    inputs: Mapping[str, object],
    states: Mapping[str, AgentState],
    parameters: Mapping[str, object],
) -> AgentState:
    return AgentState(messages=[AIMessage(content=str(inputs["text"]))])


def build_registry() -> dict[str, object]:
    """Resolved by --functions; must live in a module the Ray workers can import."""
    return {"echo": echo}


def echo_workflow(workflow_name: str) -> Workflow:
    return Workflow.model_validate(
        {
            "workflow_name": workflow_name,
            "nodes": [{"name": "echo", "type": "function", "function": "echo"}],
            "edges": [],
        }
    )


def test_static_inputs_experiment_drives_two_workflows(
    ray_session: None,
    tmp_path: Path,
) -> None:
    fleet = WorkflowFleet(
        scheduler_config=SchedulerConfig(),
        output_dir=tmp_path / "fleet",
        run_id="master-exp",
    )
    fleet.start()
    try:
        fleet.register_workflow(
            echo_workflow("wf-a"), functions={"echo": echo}, output_dir=tmp_path / "wf-a"
        )
        fleet.register_workflow(
            echo_workflow("wf-b"), functions={"echo": echo}, output_dir=tmp_path / "wf-b"
        )
        config = {
            "timeout_sec": 30,
            "sessions": [
                {"workflow": "wf-a", "session_id": "a-1", "inputs": {"text": "alpha"}},
                {"workflow": "wf-b", "session_id": "b-1", "inputs": {"text": "beta"}},
            ],
        }
        static_inputs.run(fleet, output_dir=tmp_path / "fleet", run_id="master-exp", config=config)
        assert message_text(fleet.get_result("wf-a", "a-1")) == "alpha"
        assert message_text(fleet.get_result("wf-b", "b-1")) == "beta"
    finally:
        fleet.shutdown()

    assert (tmp_path / "fleet" / "workflow_trace.jsonl").is_file()
    assert (tmp_path / "fleet" / "run_summary.json").is_file()


def test_master_main_writes_a_run_manifest_beside_the_trace(
    ray_session: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Covers the call site, not just write_run_manifest: a sweep recovers its factors
    # from this file, so a main() that forgets to emit it loses them for good.
    # ray.init/shutdown are stubbed because main() would otherwise tear down the
    # session-scoped cluster the rest of the suite shares.
    workflow_path = tmp_path / "wf.yaml"
    workflow_path.write_text(yaml.safe_dump(echo_workflow("wf-a").model_dump(mode="json")))
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        yaml.safe_dump(
            {
                "timeout_sec": 30,
                "sessions": [
                    {"workflow": "wf-a", "session_id": "a-1", "inputs": {"text": "alpha"}}
                ],
            }
        )
    )
    scheduler_path = tmp_path / "scheduler.yaml"
    scheduler_path.write_text(
        yaml.safe_dump({"policy": "cache", "enable_prefetch": False})
    )
    monkeypatch.setattr(master.ray, "init", lambda **_: None)
    monkeypatch.setattr(master.ray, "shutdown", lambda: None)

    exit_code = master.main(
        [
            "--workflow-files",
            str(workflow_path),
            "--functions",
            "test_workflow_master_experiment:build_registry",
            "--experiment",
            "experiment.workflow.experiments.static_inputs:run",
            "--experiment-config",
            str(experiment_path),
            "--scheduler-config",
            str(scheduler_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--run-id",
            "r1",
            "--fuse-nodes",
        ]
    )

    assert exit_code == 0
    run_dir = tmp_path / "out" / "r1"
    assert (run_dir / "workflow_trace.jsonl").is_file()
    manifest = json.loads((run_dir / master.MANIFEST_FILENAME).read_text())
    assert manifest["run_id"] == "r1"
    assert manifest["fuse_nodes"] is True
    assert manifest["scheduler_config"]["enable_prefetch"] is False
    assert manifest["experiment"]["timeout_sec"] == 30
    assert [entry["path"] for entry in manifest["workflow_files"]] == [str(workflow_path)]
    assert manifest["predictions"] is None


def test_static_inputs_requires_sessions() -> None:
    with pytest.raises(KeyError):
        static_inputs.run(cast(WorkflowFleet, object()), output_dir=Path("."), run_id="x", config={})


def test_dataset_replay_namespaces_and_caps_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = ("s0", "s1", "s2")
    monkeypatch.setitem(
        dataset_replay._SCENARIOS,
        "qmsum",
        (lambda path: samples, lambda sample: {"text": sample}),
    )
    sessions = dataset_replay._scenario_sessions(
        {"dataset": "qmsum", "dataset_path": "/x", "sample_count": 2}
    )
    assert [session.session_id for session in sessions] == ["qmsum-0000", "qmsum-0001"]
    assert sessions[0].inputs == {"text": "s0"}
    assert sessions[0].workflow_name == "qmsum"


def test_dataset_replay_rejects_unknown_dataset() -> None:
    with pytest.raises(ValueError):
        dataset_replay._scenario_sessions({"dataset": "nope", "dataset_path": "/x"})
