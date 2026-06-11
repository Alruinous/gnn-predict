from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from workflow.loader import load_workflow, load_workflows


def build_valid_workflow_payload(node_name: str = "planner") -> dict[str, object]:
    return {
        "nodes": [
            {"name": "input", "type": "input"},
            {
                "name": node_name,
                "type": "tool",
                "model": {
                    "task": "object_detection",
                    "name": "yolov5n",
                    "input_channels": 3,
                    "output_classes": 80,
                    "image_size": [640, 640],
                },
                "runtime": {
                    "batch_size": 4,
                    "input_shape": [4, 3, 640, 640],
                    "phase": "inference",
                },
            },
            {"name": "output", "type": "output"},
        ],
        "edges": [
            {"source": "input", "target": node_name, "attributes": {}},
            {"source": node_name, "target": "output", "attributes": {}},
        ],
    }


def write_workflow(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_load_workflow_reads_valid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "workflow.yaml"
    write_workflow(config_path, build_valid_workflow_payload())

    workflow = load_workflow(config_path)
    planner = workflow.node_map()["planner"]

    assert workflow.node_names() == ["input", "planner", "output"]
    assert planner.runtime is not None
    assert planner.runtime.input_shape == [4, 3, 640, 640]


def test_load_workflows_reads_directory_in_name_order(tmp_path: Path) -> None:
    write_workflow(tmp_path / "b.yaml", build_valid_workflow_payload("b_node"))
    write_workflow(tmp_path / "a.yaml", build_valid_workflow_payload("a_node"))
    (tmp_path / "ignore.txt").write_text("ignored", encoding="utf-8")

    workflows = load_workflows(tmp_path)

    assert [workflow.nodes[1].name for workflow in workflows] == ["a_node", "b_node"]


def test_load_workflows_accepts_single_file(tmp_path: Path) -> None:
    config_path = tmp_path / "workflow.yaml"
    write_workflow(config_path, build_valid_workflow_payload())

    workflows = load_workflows(config_path)

    assert len(workflows) == 1
    assert "planner" in workflows[0].node_map()


def test_load_workflow_rejects_non_yaml_file(tmp_path: Path) -> None:
    config_path = tmp_path / "workflow.txt"
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML"):
        load_workflow(config_path)


def test_load_workflow_rejects_yml_file(tmp_path: Path) -> None:
    config_path = tmp_path / "workflow.yml"
    write_workflow(config_path, build_valid_workflow_payload())

    with pytest.raises(ValueError, match="YAML"):
        load_workflow(config_path)


def test_load_workflows_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no YAML"):
        load_workflows(tmp_path)
