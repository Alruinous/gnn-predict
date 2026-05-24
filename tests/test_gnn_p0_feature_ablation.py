from __future__ import annotations

import torch

from gnn_model.data.constants import (
    EDGE_SHAPE_FEATURE_NAMES,
    GRAPH_SHAPE_FEATURE_NAMES,
    NODE_SHAPE_FEATURE_NAMES,
    OP_TYPE_TO_INDEX,
)
from gnn_model_test_utils import build_synthetic_graph
from scripts.gnn_p0_feature_ablation import (
    EXPERIMENT_MASK_GROUPS,
    build_ablation_runs,
    clone_graph_with_feature_mask,
    resolve_feature_groups,
)


def test_resolve_feature_groups_maps_p0_feature_blocks() -> None:
    groups = resolve_feature_groups()

    assert groups["op"].mask_op_type is True
    assert groups["op"].node_indices == ()
    assert groups["op"].graph_indices == ()
    assert len(groups["shape"].node_indices) == len(NODE_SHAPE_FEATURE_NAMES)
    assert len(groups["shape"].edge_indices) == len(EDGE_SHAPE_FEATURE_NAMES)
    assert len(groups["shape"].graph_indices) == len(GRAPH_SHAPE_FEATURE_NAMES)


def test_clone_graph_with_feature_mask_zeroes_only_selected_columns() -> None:
    graph = build_synthetic_graph(3)
    original_x = graph.x.clone()
    original_edge_attr = graph.edge_attr.clone()
    original_graph_features = graph.graph_features.clone()
    mask_spec = resolve_feature_groups()["shape"]

    masked_graph = clone_graph_with_feature_mask(graph, mask_spec)

    assert torch.count_nonzero(masked_graph.x[:, list(mask_spec.node_indices)]) == 0
    assert (
        torch.count_nonzero(masked_graph.edge_attr[:, list(mask_spec.edge_indices)])
        == 0
    )
    assert (
        torch.count_nonzero(
            masked_graph.graph_features[:, list(mask_spec.graph_indices)]
        )
        == 0
    )
    assert_unmasked_columns_equal(
        masked_graph.x,
        original_x,
        mask_spec.node_indices,
    )
    assert_unmasked_columns_equal(
        masked_graph.edge_attr,
        original_edge_attr,
        mask_spec.edge_indices,
    )
    assert_unmasked_columns_equal(
        masked_graph.graph_features,
        original_graph_features,
        mask_spec.graph_indices,
    )
    torch.testing.assert_close(graph.x, original_x)
    torch.testing.assert_close(graph.edge_attr, original_edge_attr)
    torch.testing.assert_close(graph.graph_features, original_graph_features)


def test_clone_graph_with_feature_mask_can_mask_op_type_ids() -> None:
    graph = build_synthetic_graph(3)
    original_op_type_ids = graph.op_type_ids.clone()
    mask_spec = resolve_feature_groups()["op"]

    masked_graph = clone_graph_with_feature_mask(graph, mask_spec)

    assert torch.all(masked_graph.op_type_ids == OP_TYPE_TO_INDEX["op_other"])
    torch.testing.assert_close(graph.op_type_ids, original_op_type_ids)


def test_build_ablation_runs_uses_expected_mask_combinations() -> None:
    groups = resolve_feature_groups()
    runs = build_ablation_runs(tuple(EXPERIMENT_MASK_GROUPS), (7,), groups)
    by_name = {run.experiment_name: run for run in runs}

    assert by_name["p0_all"].masked_groups == ()
    assert by_name["p0_off"].masked_groups == ("op", "shape")
    assert by_name["op_only"].masked_groups == ("shape",)
    assert by_name["shape_only"].masked_groups == ("op",)
    assert by_name["no_op"].masked_groups == ("op",)
    assert by_name["no_shape"].masked_groups == ("shape",)
    assert by_name["p0_all"].mask_spec.node_indices == ()
    assert by_name["p0_off"].mask_spec.mask_op_type is True
    assert len(by_name["p0_off"].mask_spec.graph_indices) == len(
        GRAPH_SHAPE_FEATURE_NAMES
    )


def assert_unmasked_columns_equal(
    actual: torch.Tensor,
    expected: torch.Tensor,
    masked_indices: tuple[int, ...],
) -> None:
    unmasked_indices = sorted(set(range(actual.size(1))) - set(masked_indices))
    torch.testing.assert_close(actual[:, unmasked_indices], expected[:, unmasked_indices])
