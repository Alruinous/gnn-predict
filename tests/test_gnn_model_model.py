from __future__ import annotations

import torch
from torch_geometric.loader import DataLoader

from gnn_model.data.constants import (
    EDGE_FEATURE_DIM,
    GRAPH_FEATURE_DIM,
    NODE_FEATURE_DIM,
    OP_TYPE_COUNT,
    OP_TYPE_EMBEDDING_DIM,
)
from gnn_model.models import IntelliGraphLargeModelPredictor
from gnn_model.models.predictor import ReadoutMode
from gnn_model_test_utils import TARGET_NAMES, build_synthetic_graph


def test_intelligraph_large_model_predictor_forward_is_finite() -> None:
    graphs = [build_synthetic_graph(index) for index in range(2)]
    batch = next(iter(DataLoader(graphs, batch_size=2, shuffle=False)))
    model = IntelliGraphLargeModelPredictor(
        node_dim=NODE_FEATURE_DIM,
        edge_dim=EDGE_FEATURE_DIM,
        graph_dim=GRAPH_FEATURE_DIM,
        op_type_count=OP_TYPE_COUNT,
        op_type_embedding_dim=OP_TYPE_EMBEDDING_DIM,
        hidden_dim=64,
        targets=list(TARGET_NAMES),
        num_heads=4,
        num_layers=2,
        dropout_rate=0.1,
    )

    predictions = model(batch)

    assert predictions.shape == (2, len(TARGET_NAMES))
    assert torch.isfinite(predictions).all()


def test_intelligraph_large_model_predictor_default_state_dict_is_checkpoint_compatible() -> None:
    model = IntelliGraphLargeModelPredictor(
        node_dim=NODE_FEATURE_DIM,
        edge_dim=EDGE_FEATURE_DIM,
        graph_dim=GRAPH_FEATURE_DIM,
        op_type_count=OP_TYPE_COUNT,
        op_type_embedding_dim=OP_TYPE_EMBEDDING_DIM,
        hidden_dim=64,
        targets=list(TARGET_NAMES),
        num_heads=4,
        num_layers=2,
        dropout_rate=0.1,
    )

    keys = set(model.state_dict())

    assert "structural_norm.weight" not in keys
    assert "structural_norm.bias" not in keys


def test_intelligraph_large_model_predictor_supports_multi_pool_readout() -> None:
    graphs = [build_synthetic_graph(index) for index in range(2)]
    batch = next(iter(DataLoader(graphs, batch_size=2, shuffle=False)))
    model = IntelliGraphLargeModelPredictor(
        node_dim=NODE_FEATURE_DIM,
        edge_dim=EDGE_FEATURE_DIM,
        graph_dim=GRAPH_FEATURE_DIM,
        op_type_count=OP_TYPE_COUNT,
        op_type_embedding_dim=OP_TYPE_EMBEDDING_DIM,
        hidden_dim=64,
        targets=list(TARGET_NAMES),
        num_heads=4,
        num_layers=2,
        dropout_rate=0.1,
        readout_mode="mean_sum_max",
    )

    predictions = model(batch)

    assert predictions.shape == (2, len(TARGET_NAMES))
    assert torch.isfinite(predictions).all()


def test_intelligraph_large_model_predictor_supports_target_readout_modes() -> None:
    graphs = [build_synthetic_graph(index) for index in range(2)]
    batch = next(iter(DataLoader(graphs, batch_size=2, shuffle=False)))
    target_readout_modes: dict[str, ReadoutMode] = {
        "memory_delta_gb_p95": "mean_sum_max"
    }
    model = IntelliGraphLargeModelPredictor(
        node_dim=NODE_FEATURE_DIM,
        edge_dim=EDGE_FEATURE_DIM,
        graph_dim=GRAPH_FEATURE_DIM,
        op_type_count=OP_TYPE_COUNT,
        op_type_embedding_dim=OP_TYPE_EMBEDDING_DIM,
        hidden_dim=64,
        targets=list(TARGET_NAMES),
        num_heads=4,
        num_layers=2,
        dropout_rate=0.1,
        target_readout_modes=target_readout_modes,
    )

    predictions = model(batch)

    assert predictions.shape == (2, len(TARGET_NAMES))
    assert torch.isfinite(predictions).all()


def test_intelligraph_large_model_predictor_supports_structural_context() -> None:
    graphs = [build_synthetic_graph(index) for index in range(2)]
    batch = next(iter(DataLoader(graphs, batch_size=2, shuffle=False)))
    model = IntelliGraphLargeModelPredictor(
        node_dim=NODE_FEATURE_DIM,
        edge_dim=EDGE_FEATURE_DIM,
        graph_dim=GRAPH_FEATURE_DIM,
        op_type_count=OP_TYPE_COUNT,
        op_type_embedding_dim=OP_TYPE_EMBEDDING_DIM,
        hidden_dim=64,
        targets=list(TARGET_NAMES),
        num_heads=4,
        num_layers=2,
        dropout_rate=0.1,
        structural_context_mode="basic",
    )

    predictions = model(batch)

    assert predictions.shape == (2, len(TARGET_NAMES))
    assert torch.isfinite(predictions).all()
