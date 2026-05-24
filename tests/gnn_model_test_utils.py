from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import RobustScaler
from torch_geometric.data import Data

from common.onnx_initializer import (
    ONNX_EXPORT_MODE_METADATA_KEY,
    RUNTIME_INPUT_NAMES_METADATA_KEY,
    set_model_metadata_value,
)
from gnn_model.data.constants import (
    EDGE_FEATURE_DIM,
    GRAPH_FEATURE_DIM,
    NODE_FEATURE_DIM,
    OP_TYPE_COUNT,
)
from gnn_model.data.dataset import SPLIT_FILE_NAMES

TARGET_NAMES = (
    "duration_sec_avg",
    "cpu_cores_p95",
    "memory_delta_gb_p95",
    "gpu_util_percent_p95",
    "gpu_sm_active_percent_p95",
    "gpu_sm_occupancy_percent_p95",
    "gpu_mem_used_mb_p95",
)


class TinyConvNet(nn.Module):
    def __init__(self, width: int, output_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(width, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def build_toy_model(sample_index: int) -> nn.Module:
    width = 4 + sample_index
    return TinyConvNet(width=width, output_dim=3)


def export_architecture_only_onnx(
    model: nn.Module,
    output_path: Path,
    input_shape: tuple[int, ...],
) -> Path:
    model.eval()
    example_inputs = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        (example_inputs,),
        output_path,
        input_names=["inputs"],
        output_names=["outputs"],
        opset_version=14,
        dynamo=False,
        export_params=False,
    )
    onnx_model = onnx.load(output_path)
    set_model_metadata_value(
        onnx_model,
        RUNTIME_INPUT_NAMES_METADATA_KEY,
        json.dumps(["inputs"]),
    )
    set_model_metadata_value(
        onnx_model,
        ONNX_EXPORT_MODE_METADATA_KEY,
        "architecture_only",
    )
    onnx.save(onnx_model, output_path)
    onnx.checker.check_model(onnx_model)
    return output_path


def build_synthetic_graph(
    sample_index: int,
    *,
    target_dim: int = len(TARGET_NAMES),
    node_dim: int = NODE_FEATURE_DIM,
) -> Data:
    num_nodes = 4
    num_edges = 4
    x = torch.arange(num_nodes * node_dim, dtype=torch.float32).reshape(
        num_nodes,
        node_dim,
    )
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 0]],
        dtype=torch.long,
    )
    edge_attr = torch.arange(num_edges * EDGE_FEATURE_DIM, dtype=torch.float32).reshape(
        num_edges,
        EDGE_FEATURE_DIM,
    )
    graph_features = torch.full((1, GRAPH_FEATURE_DIM), float(sample_index + 1))
    op_type_ids = torch.arange(num_nodes, dtype=torch.long) % OP_TYPE_COUNT
    y = torch.arange(target_dim, dtype=torch.float32).unsqueeze(0) + sample_index
    data = Data(
        x=x / 100.0,
        edge_index=edge_index,
        edge_attr=edge_attr / 100.0,
        graph_features=graph_features / 10.0,
        op_type_ids=op_type_ids,
        y=y,
    )
    data.variant_name = f"synthetic_{sample_index}"
    data.phase = "training"
    data.batch_size = 2
    data.gpu_name = "v100"
    data.model_name = "synthetic"
    return data


def write_split_dataset(
    root: Path,
    *,
    target_dim: int = len(TARGET_NAMES),
    counts: dict[str, int] | None = None,
) -> Path:
    split_counts = counts or {"train": 3, "val": 1, "test": 1}
    root.mkdir(parents=True, exist_ok=True)
    offset = 0
    for split_name, file_name in SPLIT_FILE_NAMES.items():
        graphs = [
            build_synthetic_graph(offset + index, target_dim=target_dim)
            for index in range(split_counts[split_name])
        ]
        torch.save(graphs, root / file_name)
        offset += split_counts[split_name]
    return root


def write_result_json(
    res_root: Path,
    target_name: str,
    variant_names: tuple[str, ...],
    *,
    base_model_name: str = "toy",
    model_kind: str = "image",
    variant_config_by_name: dict[str, dict[str, object]] | None = None,
) -> Path:
    result_path = res_root / target_name / "results" / f"{target_name}_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "2.0.0",
        "config_path": str(res_root / target_name / "config" / f"{target_name}.yaml"),
        "gpu_node": "v100",
        "timings": {},
        "variants": [
            {
                "name": variant_name,
                "base_model_name": base_model_name,
                "base_model_pretrained": False,
                "source": "test",
                "group_total_variants_defined": len(variant_names),
                "variant_config": (
                    variant_config_by_name or {}
                ).get(variant_name, {"example_input_shape": [1, 3, 32, 32]}),
                "mutations": [],
                "timings": {},
                "training": None,
                "inference": None,
                "onnx_export": None,
                "metadata": {"model_kind": model_kind},
            }
            for variant_name in variant_names
        ],
        "summary": {"variant_count": len(variant_names)},
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    return result_path


def write_target_scalers(scaler_dir: Path, target_names: tuple[str, ...]) -> Path:
    scaler_dir.mkdir(parents=True, exist_ok=True)
    target_scalers = {}
    source = np.arange(len(target_names) * 4, dtype=np.float64).reshape(4, -1)
    for index, target_name in enumerate(target_names):
        target_scalers[target_name] = RobustScaler().fit(source[:, index : index + 1])
    with (scaler_dir / "target_scalers.pkl").open("wb") as file:
        pickle.dump(target_scalers, file)
    return scaler_dir


def write_split_config(
    path: Path,
    *,
    experiment_name: str,
    data_dir: Path,
    scaler_dir: Path,
    target_names: tuple[str, ...] = TARGET_NAMES,
) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": experiment_name,
                "data": {
                    "kind": "split",
                    "data_dir": str(data_dir),
                    "target_names": list(target_names),
                    "scaler_dir": str(scaler_dir),
                },
                "model": {
                    "hidden_dim": 16,
                    "num_layers": 1,
                    "num_heads": 4,
                    "dropout_rate": 0.1,
                },
                "training": {
                    "batch_size": 1,
                    "num_epochs": 1,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                },
            }
        ),
        encoding="utf-8",
    )
    return path
