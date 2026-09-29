from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch_geometric
import torch_geometric.data
from sklearn.preprocessing import RobustScaler

from common.log import get_logger
from gnn_model.evaluation.metrics import TargetScaler

logger = get_logger(Path(__file__).name)


def generate_scalers(
    graph_files: list[Path],
    target_columns: list[str],
    saved_path: Path | None = None,
    transform: Callable[[torch_geometric.data.Data], torch_geometric.data.Data]
    | None = None,
) -> tuple[dict[str, RobustScaler], dict[str, RobustScaler]]:
    node_feature_scaler = RobustScaler()
    edge_feature_scaler = RobustScaler()
    graph_feature_scaler = RobustScaler()
    target_scalers = {target_column: RobustScaler() for target_column in target_columns}

    node_feature_list = []
    edge_feature_list = []
    graph_feature_list = []
    target_list = []
    for graph_file in graph_files:
        graphs: list[torch_geometric.data.Data] = torch.load(
            graph_file, weights_only=False
        )
        for graph in graphs:
            if transform is not None:
                graph = transform(graph)
            node_feature_list.append(graph.x.numpy())  # (num_nodes, num_features)
            if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
                edge_feature_list.append(
                    graph.edge_attr.numpy()
                )  # (num_edges, num_features)
            graph_feature_list.append(graph.graph_features.numpy())  # (1, num_features)
            target_list.append(graph.y[...].numpy())  # (1, num_target)

    if not node_feature_list:
        raise ValueError("training split has no graphs to fit scalers")
    node_features = np.concatenate(node_feature_list, axis=0)
    node_feature_scaler.fit(node_features)

    if edge_feature_list:
        edge_features = np.concatenate(edge_feature_list, axis=0)
        edge_feature_scaler.fit(edge_features)

    graph_features = np.concatenate(graph_feature_list, axis=0)
    graph_feature_scaler.fit(graph_features)
    target_features = np.concatenate(target_list, axis=0)
    if target_features.shape[1] != len(target_columns):
        raise ValueError(
            "target columns do not match training labels: "
            f"{len(target_columns)} != {target_features.shape[1]}"
        )
    for i, target_name in enumerate(target_columns):
        target_scalers[target_name].fit(target_features[:, i : i + 1])

    feature_scalers = {
        "node_feature_scaler": node_feature_scaler,
        "edge_feature_scaler": edge_feature_scaler,
        "graph_feature_scaler": graph_feature_scaler,
    }

    if saved_path is not None:
        saved_path.mkdir(parents=True, exist_ok=True)
        for scaler_name, scaler in feature_scalers.items():
            with open(saved_path / f"{scaler_name}.pkl", "wb") as f:
                pickle.dump(scaler, f)
        with open(saved_path / "target_scalers.pkl", "wb") as f:
            pickle.dump(target_scalers, f)
    return feature_scalers, target_scalers


def scale_data(
    data: torch_geometric.data.Data,
    target_columns: list[str],
    feature_scalers: dict[str, RobustScaler],
    target_scalers: dict[str, RobustScaler],
) -> torch_geometric.data.Data:
    node_feature_scaler = feature_scalers["node_feature_scaler"]
    normalized_node_feature = node_feature_scaler.transform(data.x.numpy())
    assert not np.isnan(normalized_node_feature).any(), (
        "Node features contain NaN after normalization"
    )
    data.x = torch.from_numpy(normalized_node_feature)

    if hasattr(data, "graph_features") and data.graph_features is not None:
        graph_feature_scaler = feature_scalers["graph_feature_scaler"]
        normalized_graph_feature = graph_feature_scaler.transform(
            data.graph_features.numpy()
        )
        assert not np.isnan(normalized_graph_feature).any(), (
            "Graph features contain NaN after normalization"
        )
        data.graph_features = torch.from_numpy(normalized_graph_feature)

    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        edge_feature_scaler = feature_scalers["edge_feature_scaler"]
        normalized_edge_feature = edge_feature_scaler.transform(data.edge_attr.numpy())
        assert not np.isnan(normalized_edge_feature).any(), (
            "Edge features contain NaN after normalization"
        )
        data.edge_attr = torch.from_numpy(normalized_edge_feature)

    normalized_target = np.concatenate(
        [
            target_scalers[t].transform(data.y[..., i : i + 1].numpy())
            for i, t in enumerate(target_columns)
        ],
        axis=1,
    )
    assert len(normalized_target.shape) == len(data.y.shape)
    assert not np.isnan(normalized_target).any(), (
        "Target contains NaN after normalization"
    )
    data.y = torch.from_numpy(normalized_target)

    return data


def load_target_scalers(
    scaler_dir: Path,
    target_names: list[str],
) -> dict[str, TargetScaler]:
    scaler_path = Path(scaler_dir) / "target_scalers.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"target scalers do not exist: {scaler_path}")
    with scaler_path.open("rb") as file:
        raw_scalers = pickle.load(file)
    if not isinstance(raw_scalers, dict):
        raise TypeError(f"target scalers must be a mapping: {scaler_path}")

    target_scalers: dict[str, TargetScaler] = {}
    for target_name in target_names:
        scaler = raw_scalers.get(target_name)
        if scaler is None:
            raise ValueError(f"target scaler missing: {target_name}")
        if not hasattr(scaler, "inverse_transform"):
            raise TypeError(f"target scaler cannot inverse transform: {target_name}")
        target_scalers[target_name] = cast(TargetScaler, scaler)
    return target_scalers


def main():
    parser = argparse.ArgumentParser(
        description="Generate scaler parameters from the training data.",
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Path to the prepared dataset directory containing train/val/test splits.",
    )
    parser.add_argument(
        "--scaler_output_path",
        default="data/scalers",
        help="Path to save the generated scaler parameters JSON.",
    )
    parser.add_argument(
        "--scaled_data_output_path",
        default="scaled",
        help="Path to save the scaled data.",
    )
    parser.add_argument(
        "--target_fields",
        default=None,
        help="Comma-separated target names; defaults to manifest.json target_names.",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )
    if args.target_fields:
        target_columns = [field.strip() for field in args.target_fields.split(",")]
        if manifest is not None and target_columns != manifest["target_names"]:
            raise ValueError("target fields must match manifest.json target_names")
    else:
        if manifest is None:
            raise FileNotFoundError(f"target names require {manifest_path}")
        target_columns = manifest["target_names"]
    if not target_columns or any(not field for field in target_columns):
        raise ValueError("target fields must not be empty")
    graph_files = [data_dir / "train.pt", data_dir / "val.pt", data_dir / "test.pt"]
    feature_scalers, target_scalers = generate_scalers(
        graph_files=[graph_files[0]],
        target_columns=target_columns,
        saved_path=Path(args.scaler_output_path) if args.scaler_output_path else None,
    )
    logger.info(f"Generated scalers and saved to {args.scaler_output_path}")

    if args.scaled_data_output_path is not None:
        scale_data_output_path = Path(args.scaled_data_output_path)
        scale_data_output_path.mkdir(parents=True, exist_ok=True)
        for graph_file in graph_files:
            graphs: list[torch_geometric.data.Data] = torch.load(
                graph_file, weights_only=False
            )
            scaled_graphs = [
                scale_data(
                    graph,
                    target_columns=target_columns,
                    feature_scalers=feature_scalers,
                    target_scalers=target_scalers,
                )
                for graph in graphs
            ]
            torch.save(scaled_graphs, scale_data_output_path / graph_file.name)
        if manifest is not None:
            scaled_manifest = {
                **manifest,
                "normalization": {
                    "fit_split": "train",
                    "scaler_dir": str(Path(args.scaler_output_path).resolve()),
                },
            }
            (scale_data_output_path / "manifest.json").write_text(
                json.dumps(scaled_manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    logger.info(f"Scaled data saved to {args.scaled_data_output_path}")


if __name__ == "__main__":
    raise SystemExit(main())
