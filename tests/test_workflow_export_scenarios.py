from __future__ import annotations

import yaml

from experiment.workflow.mbpp import build_mbpp_workflow
from experiment.workflow.qmsum import build_qmsum_workflow
from experiment.workflow.scenario_functions import build_registry
from workflow.schema import FunctionNodeConfig, Workflow


def test_scenario_builders_roundtrip_through_yaml_and_cover_functions() -> None:
    registry = build_registry()
    for build, expected_name in ((build_qmsum_workflow, "qmsum"), (build_mbpp_workflow, "mbpp")):
        dumped = yaml.safe_dump(build().model_dump(mode="json"), sort_keys=False)
        reloaded = Workflow.model_validate(yaml.safe_load(dumped))
        assert reloaded.workflow_name == expected_name
        for node in reloaded.nodes:
            if isinstance(node, FunctionNodeConfig):
                assert node.function in registry
