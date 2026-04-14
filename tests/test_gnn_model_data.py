from __future__ import annotations

from pathlib import Path

from gnn_model.data.constants import (
    EDGE_FEATURE_DIM,
    GRAPH_FEATURE_DIM,
    GRAPH_METRIC_DIM,
    NODE_FEATURE_DIM,
)
from gnn_model.data.fake_dataset import (
    build_fake_graph_datasets,
    build_toy_model,
    export_architecture_only_onnx,
)
from gnn_model.data.onnx_graph import build_graph_data_from_onnx
from common.onnx_initializer import write_randomized_onnx_model


def test_build_graph_data_from_onnx_returns_expected_shapes(tmp_path: Path) -> None:
    architecture_only_path = tmp_path / "toy_architecture.onnx"
    initialized_path = tmp_path / "toy_initialized.onnx"

    export_architecture_only_onnx(
        build_toy_model(0),
        architecture_only_path,
        (1, 3, 32, 32),
    )
    write_randomized_onnx_model(
        architecture_only_path,
        initialized_path,
        runtime_input_names=["inputs"],
        seed=11,
    )

    data = build_graph_data_from_onnx(initialized_path)

    assert data.x.shape[1] == NODE_FEATURE_DIM
    assert data.edge_attr.shape[1] == EDGE_FEATURE_DIM
    assert data.graph_features.shape == (1, GRAPH_FEATURE_DIM)
    assert data.graph_metrics.shape == (1, GRAPH_METRIC_DIM)
    assert data.edge_index.shape[0] == 2
    assert data.node_op_token_id.shape[0] == data.x.shape[0]


def test_build_fake_graph_datasets_produces_non_empty_splits() -> None:
    datasets = build_fake_graph_datasets(dataset_size=10, seed=3)

    assert len(datasets.train_data) > 0
    assert len(datasets.val_data) > 0
    assert len(datasets.test_data) > 0
    assert datasets.train_data[0].y.shape == (1, 5)

