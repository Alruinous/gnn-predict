from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from experiment.workflow.qmsum import build_qmsum_workflow
from workflow import master
from workflow.artifacts import SchedulerConfig

ROOT = Path(__file__).resolve().parents[1]


def _node(hostname: str, gpu: int, kind: str | None = None) -> dict[str, object]:
    resources: dict[str, object] = {"GPU": float(gpu)}
    if kind is not None:
        resources[f"accelerator_type:{kind}"] = 1.0
    return {"Alive": True, "NodeManagerHostname": hostname, "Resources": resources}


def test_parse_mapping_variants() -> None:
    assert master.parse_mapping(None) == {}
    assert master.parse_mapping("v100=32768,a100=81920") == {"v100": "32768", "a100": "81920"}
    assert master.parse_int_mapping("v100=32768") == {"v100": 32768}
    assert master.parse_float_mapping("qmsum=1.0,mbpp=2") == {"qmsum": 1.0, "mbpp": 2.0}
    with pytest.raises(ValueError):
        master.parse_mapping("missing_equals")


def test_resolve_dotted_and_functions() -> None:
    build = master.resolve_dotted("experiment.workflow.scenario_functions:build_registry")
    assert callable(build)
    registry = master.resolve_functions("experiment.workflow.scenario_functions:build_registry")
    assert "split_qmsum" in registry and "evaluate_mbpp_initial" in registry
    with pytest.raises(ValueError):
        master.resolve_dotted("no_colon_here")


def test_discover_accelerators_reads_accelerator_type() -> None:
    nodes = [_node("head", 0), _node("w0", 1, "A100"), _node("w1", 1, "V100")]
    accelerators = master.discover_accelerators(nodes, gpu_mem_mb={"a100": 81920, "v100": 32768})
    by_host = {acc.hostname: acc for acc in accelerators}
    assert set(by_host) == {"w0", "w1"}
    assert by_host["w0"].gpu_kind == "a100" and by_host["w0"].total_mem_mb == 81920
    assert by_host["w1"].gpu_kind == "v100" and by_host["w1"].local_index == 0


def test_discover_accelerators_multi_gpu_host_with_override() -> None:
    accelerators = master.discover_accelerators(
        [_node("w0", 2)], gpu_mem_mb={"v100": 32768}, host_gpu_kind={"w0": "v100"}
    )
    assert [acc.local_index for acc in accelerators] == [0, 1]
    assert {acc.gpu_kind for acc in accelerators} == {"v100"}


def test_discover_accelerators_requires_gpu_mem_and_known_kind() -> None:
    with pytest.raises(ValueError):
        master.discover_accelerators([_node("w0", 1, "A100")], gpu_mem_mb={})
    with pytest.raises(ValueError):
        master.discover_accelerators([_node("w0", 1)], gpu_mem_mb={"v100": 32768})
    with pytest.raises(RuntimeError):
        master.discover_accelerators([_node("head", 0)], gpu_mem_mb={"v100": 32768})


def test_scheduler_config_merge_with_discovered_accelerators() -> None:
    accelerators = master.discover_accelerators([_node("w0", 1, "V100")], gpu_mem_mb={"v100": 32768})
    config_data: dict[str, object] = {"policy": "cache"}
    config_data["accelerators"] = [acc.model_dump() for acc in accelerators]
    scheduler_config = SchedulerConfig.model_validate(config_data)
    assert scheduler_config.policy == "cache"
    assert scheduler_config.accelerators[0].hostname == "w0"
    assert scheduler_config.accelerators[0].gpu_kind == "v100"


def test_load_committed_serve_workflows() -> None:
    paths = [
        ROOT / "config" / "workflow" / "serve" / "qmsum.yaml",
        ROOT / "config" / "workflow" / "serve" / "mbpp.yaml",
    ]
    workflows = master.load_workflows(paths)
    assert {workflow.workflow_name for workflow in workflows} == {"qmsum", "mbpp"}


def test_load_workflows_rejects_duplicate_names(tmp_path: Path) -> None:
    data = build_qmsum_workflow().model_dump(mode="json")
    first = tmp_path / "a.yaml"
    second = tmp_path / "b.yaml"
    first.write_text(yaml.safe_dump(data))
    second.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        master.load_workflows([first, second])
