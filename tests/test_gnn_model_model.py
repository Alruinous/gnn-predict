from __future__ import annotations

import torch
from torch_geometric.loader import DataLoader

from gnn_model.data.constants import (
    EDGE_FEATURE_DIM,
    GRAPH_FEATURE_DIM,
    NODE_FEATURE_DIM,
)
from gnn_model.models import IntelliGraphLargeModelPredictor
from gnn_model_test_utils import TARGET_NAMES, build_synthetic_graph


def test_intelligraph_large_model_predictor_forward_is_finite() -> None:
    graphs = [build_synthetic_graph(index) for index in range(2)]
    batch = next(iter(DataLoader(graphs, batch_size=2, shuffle=False)))
    model = IntelliGraphLargeModelPredictor(
        node_dim=NODE_FEATURE_DIM,
        edge_dim=EDGE_FEATURE_DIM,
        graph_dim=GRAPH_FEATURE_DIM,
        hidden_dim=64,
        targets=list(TARGET_NAMES),
        num_heads=4,
        num_layers=2,
        dropout_rate=0.1,
    )

    predictions = model(batch)

    assert predictions.shape == (2, len(TARGET_NAMES))
    assert torch.isfinite(predictions).all()
