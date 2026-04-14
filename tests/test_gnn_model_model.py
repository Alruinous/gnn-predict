from __future__ import annotations

import torch
from torch_geometric.loader import DataLoader

from gnn_model.data import build_fake_graph_datasets
from gnn_model.data.constants import (
    DEFAULT_TARGET_NAMES,
    EDGE_FEATURE_DIM,
    GRAPH_FEATURE_DIM,
    GRAPH_METRIC_DIM,
    NODE_FEATURE_DIM,
)
from gnn_model.models import IntelliGraphLargeModelPredictor


def test_intelligraph_large_model_predictor_forward_is_finite() -> None:
    datasets = build_fake_graph_datasets(dataset_size=6, seed=5)
    batch = next(iter(DataLoader(datasets.train_data[:2], batch_size=2, shuffle=False)))
    model = IntelliGraphLargeModelPredictor(
        node_dim=NODE_FEATURE_DIM,
        edge_dim=EDGE_FEATURE_DIM,
        graph_dim=GRAPH_FEATURE_DIM,
        graph_metric_dim=GRAPH_METRIC_DIM,
        hidden_dim=64,
        targets=list(DEFAULT_TARGET_NAMES),
        num_heads=4,
        num_layers=2,
        dropout_rate=0.1,
    )

    predictions = model(batch)

    assert predictions.shape == (2, len(DEFAULT_TARGET_NAMES))
    assert torch.isfinite(predictions).all()

