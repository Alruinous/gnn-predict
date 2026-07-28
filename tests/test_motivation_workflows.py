from __future__ import annotations

import pytest

from experiment.workflow.motivation_workflows import (
    HETEROGENEITY_LEVELS,
    MOTIVATION_CHUNK_COUNT,
    MOTIVATION_SERVING,
    Heterogeneity,
    build_all,
    build_registry,
    split_qmsum_chunks,
)
from workflow.replica import ModelDeploymentConfig
from workflow.schema import AgentNodeConfig

EXPECTED_KEY_COUNT = {1: 1, 5: 5}


def model_keys(heterogeneity: Heterogeneity) -> set[str]:
    return {
        ModelDeploymentConfig.from_node(node).model_key
        for workflow in build_all(heterogeneity)
        for node in workflow.nodes
        if isinstance(node, AgentNodeConfig)
    }


@pytest.mark.parametrize("heterogeneity", HETEROGENEITY_LEVELS)
def test_heterogeneity_level_yields_the_intended_distinct_model_count(heterogeneity):
    assert len(model_keys(heterogeneity)) == EXPECTED_KEY_COUNT[heterogeneity]


def test_graph_shape_is_identical_across_heterogeneity_levels():
    # The experiment attributes every difference to the role-to-model mapping, so the
    # graphs, node names and workflow names must not vary with H.
    shapes = {
        heterogeneity: [
            (
                workflow.workflow_name,
                tuple(node.name for node in workflow.nodes),
                tuple((edge.source, edge.target) for edge in workflow.edges),
            )
            for workflow in build_all(heterogeneity)
        ]
        for heterogeneity in HETEROGENEITY_LEVELS
    }
    assert shapes[1] == shapes[5]


def test_every_model_shares_one_serving_profile():
    # model_key hashes the serving block, so a divergent profile would silently split
    # the key and make a model loaded for one workflow unusable by another.
    for heterogeneity in HETEROGENEITY_LEVELS:
        for workflow in build_all(heterogeneity):
            for node in workflow.nodes:
                if isinstance(node, AgentNodeConfig):
                    serving = node.execution.serving.model_dump(mode="json")
                    assert serving == dict(MOTIVATION_SERVING)


def test_workflow_names_do_not_encode_heterogeneity():
    # poisson_arrival_offsets seeds on the workflow name; differing names would give
    # the H arms different arrival traces and break the paired comparison.
    names = {
        heterogeneity: [workflow.workflow_name for workflow in build_all(heterogeneity)]
        for heterogeneity in HETEROGENEITY_LEVELS
    }
    assert (
        names[1]
        == names[5]
        == ["moa_gsm8k", "repair_mbpp", "mapreduce_qmsum", "chain_qmsum"]
    )


def test_split_function_emits_one_state_per_chunk_node():
    workflow = next(
        candidate
        for candidate in build_all(5)
        if candidate.workflow_name == "mapreduce_qmsum"
    )
    chunk_nodes = {node.name for node in workflow.nodes if node.name.startswith("chunk_")}
    states = split_qmsum_chunks(
        {"transcript": "\n\n".join(f"turn {index}" for index in range(40))},
        {},
        {"chunk_count": MOTIVATION_CHUNK_COUNT},
    )
    assert set(states) == chunk_nodes


def test_registry_exposes_every_function_the_workflows_reference():
    registry = build_registry()
    referenced = {
        node.function
        for heterogeneity in HETEROGENEITY_LEVELS
        for workflow in build_all(heterogeneity)
        for node in workflow.nodes
        if not isinstance(node, AgentNodeConfig)
    }
    assert referenced <= set(registry)


def test_chain_qmsum_shares_the_serving_profile_of_the_other_workflows():
    # model_key hashes the whole serving block, so a chain workflow with its own
    # profile would split Qwen3-8B into two deployments and silently change how many
    # models compete for cards.
    from workflow.replica import ModelDeploymentConfig
    from workflow.schema import AgentNodeConfig

    keys: dict[str, set[str]] = {}
    for workflow in build_all(5):
        for node in workflow.nodes:
            if isinstance(node, AgentNodeConfig):
                key = ModelDeploymentConfig.from_node(node).model_key
                keys.setdefault(node.model.name, set()).add(key)
    assert all(len(value) == 1 for value in keys.values()), keys
    assert len(keys) == 5


def test_chain_qmsum_exposes_three_same_model_chains():
    from workflow.fusion import find_fusable_chains, fuse_workflow

    workflow = next(
        candidate
        for candidate in build_all(5)
        if candidate.workflow_name == "chain_qmsum"
    )
    chains = find_fusable_chains(workflow)
    assert [[node.name for node in chain] for chain in chains] == [
        ["lane_a_0", "lane_a_1", "lane_a_2"],
        ["lane_b_0", "lane_b_1", "lane_b_2"],
        ["merge", "expand", "finalize"],
    ]
    assert len(fuse_workflow(workflow).nodes) == 4
