from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.data import Data

from gnn_model.config import SplitDataConfig
from gnn_model.data.constants import (
    EDGE_FEATURE_DIM,
    GRAPH_FEATURE_DIM,
    NODE_FEATURE_DIM,
)

SPLIT_FILE_NAMES = {
    "train": "train.pt",
    "val": "val.pt",
    "test": "test.pt",
}


@dataclass(frozen=True)
class GraphDatasetBundle:
    train_data: list[Data]
    val_data: list[Data]
    test_data: list[Data]
    target_names: tuple[str, ...]


def load_split_graph_datasets(config: SplitDataConfig) -> GraphDatasetBundle:
    target_names = tuple(config.target_names)
    split_files = config.split_files
    data_dir = Path(config.data_dir)
    train_data = load_graph_split(
        data_dir / split_files["train"],
        target_dim=len(target_names),
    )
    val_data = load_graph_split(
        data_dir / split_files["val"],
        target_dim=len(target_names),
    )
    test_data = load_graph_split(
        data_dir / split_files["test"],
        target_dim=len(target_names),
    )
    return GraphDatasetBundle(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        target_names=target_names,
    )


def load_graph_split(path: Path, *, target_dim: int) -> list[Data]:
    split_path = Path(path)
    if not split_path.exists():
        raise FileNotFoundError(f"graph split does not exist: {split_path}")
    graphs = torch.load(split_path, weights_only=False)
    if not isinstance(graphs, list) or not graphs:
        raise ValueError(f"graph split must be a non-empty list: {split_path}")
    for graph in graphs:
        validate_graph_data(graph, target_dim=target_dim, split_path=split_path)
    return graphs


def validate_graph_data(graph: object, *, target_dim: int, split_path: Path) -> None:
    if not isinstance(graph, Data):
        raise TypeError(f"split contains non-Data object: {split_path}")
    x = graph.x
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"node feature dim mismatch in {split_path}")
    if x.dim() != 2 or x.size(1) != NODE_FEATURE_DIM:
        raise ValueError(f"node feature dim mismatch in {split_path}")
    edge_attr = graph.edge_attr
    if not isinstance(edge_attr, torch.Tensor) or edge_attr.size(1) != EDGE_FEATURE_DIM:
        raise ValueError(f"edge feature dim mismatch in {split_path}")
    graph_features = getattr(graph, "graph_features", None)
    if (
        not isinstance(graph_features, torch.Tensor)
        or graph_features.shape != (1, GRAPH_FEATURE_DIM)
    ):
        raise ValueError(f"graph feature dim mismatch in {split_path}")
    y = graph.y
    if not isinstance(y, torch.Tensor) or y.shape != (1, target_dim):
        raise ValueError(f"target dim mismatch in {split_path}")


def resolve_split_counts(
    dataset_size: int,
    *,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    val_count = max(1, round(dataset_size * val_ratio))
    test_count = max(1, round(dataset_size * test_ratio))
    train_count = dataset_size - val_count - test_count
    if train_count <= 0:
        raise ValueError("dataset_size is too small for the requested split ratios")
    return train_count, val_count, test_count
