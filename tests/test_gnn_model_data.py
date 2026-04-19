from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest
import torch

from common.onnx_initializer import write_randomized_onnx_model
from gnn_model.config import PreparedDataConfig, SplitDataConfig
from gnn_model.data.constants import (
    EDGE_FEATURE_DIM,
    GRAPH_FEATURE_DIM,
    GRAPH_METRIC_DIM,
    NODE_FEATURE_DIM,
)
from gnn_model.data.dataset import load_split_graph_datasets
from gnn_model.data.extract import (
    TARGET_FIELDS,
    ModelRecordInfo,
    build_model_record_info,
    build_dataset,
    process_csv,
    resolve_onnx_path,
)
from gnn_model.data.onnx_graph import build_graph_data_from_onnx
from gnn_model.data.prepared_dataset import load_prepared_graph_datasets
from gnn_model_test_utils import (
    TARGET_NAMES,
    build_synthetic_graph,
    build_toy_model,
    export_architecture_only_onnx,
    write_split_dataset,
)


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

    data = build_graph_data_from_onnx(initialized_path, batch_size=8, gpu_name="v100")

    assert data.x.shape[1] == NODE_FEATURE_DIM
    assert data.edge_attr.shape[1] == EDGE_FEATURE_DIM
    assert data.graph_features.shape == (1, GRAPH_FEATURE_DIM)
    assert data.graph_features[0, -1].item() == 8.0
    assert data.graph_metrics.shape == (1, GRAPH_METRIC_DIM)
    assert data.edge_index.shape[0] == 2
    assert data.node_op_token_id.shape[0] == data.x.shape[0]


def test_load_split_graph_datasets_reads_split_directory(tmp_path: Path) -> None:
    data_dir = write_split_dataset(tmp_path / "scaled")

    datasets = load_split_graph_datasets(
        SplitDataConfig(
            data_dir=str(data_dir),
            target_names=list(TARGET_NAMES),
            scaler_dir=str(tmp_path / "scalers"),
        )
    )

    assert len(datasets.train_data) > 0
    assert len(datasets.val_data) > 0
    assert len(datasets.test_data) > 0
    assert datasets.target_names == TARGET_NAMES
    assert datasets.train_data[0].y.shape == (1, len(TARGET_NAMES))


def test_load_split_graph_datasets_rejects_missing_split(tmp_path: Path) -> None:
    data_dir = tmp_path / "scaled"
    data_dir.mkdir()
    torch.save([build_synthetic_graph(0)], data_dir / "train.pt")
    torch.save([build_synthetic_graph(1)], data_dir / "val.pt")

    with pytest.raises(FileNotFoundError, match="test.pt"):
        load_split_graph_datasets(
            SplitDataConfig(
                data_dir=str(data_dir),
                target_names=list(TARGET_NAMES),
                scaler_dir=str(tmp_path / "scalers"),
            )
        )


def test_load_split_graph_datasets_rejects_bad_target_dim(tmp_path: Path) -> None:
    data_dir = write_split_dataset(tmp_path / "scaled", target_dim=3)

    with pytest.raises(ValueError, match="target dim mismatch"):
        load_split_graph_datasets(
            SplitDataConfig(
                data_dir=str(data_dir),
                target_names=list(TARGET_NAMES),
                scaler_dir=str(tmp_path / "scalers"),
            )
        )


def test_load_split_graph_datasets_rejects_bad_feature_dim(tmp_path: Path) -> None:
    data_dir = tmp_path / "scaled"
    data_dir.mkdir()
    bad_graph = build_synthetic_graph(0, node_dim=NODE_FEATURE_DIM - 1)
    for split_name in ("train", "val", "test"):
        torch.save([bad_graph], data_dir / f"{split_name}.pt")

    with pytest.raises(ValueError, match="node feature dim mismatch"):
        load_split_graph_datasets(
            SplitDataConfig(
                data_dir=str(data_dir),
                target_names=list(TARGET_NAMES),
                scaler_dir=str(tmp_path / "scalers"),
            )
        )


def test_build_model_record_info_resolves_path_and_targets(tmp_path: Path) -> None:
    model_path = tmp_path / "res/demo/onnx_models/variant_a.onnx"
    row = {
        **build_monitor_row(tmp_path / "res", "variant_a", "training", 12.5),
        "row_number": 2,
        "variant_path": str(model_path),
        "duration_sec_avg": 2.5,
    }

    info = build_model_record_info(
        row,
        row_number=2,
        csv_path=tmp_path / "demo_monitor.csv",
    )

    assert isinstance(info, ModelRecordInfo)
    assert resolve_onnx_path(row["result_json"], row["variant_name"]) == model_path
    assert info.model_path == model_path
    assert info.sample_id == "variant_a::training"
    assert info.batch_size == 2
    assert info.gpu_name == "v100"
    assert info.target == pytest.approx((2.5, 75.0, 12.5, 2048.0))


def test_process_csv_builds_graphs_with_hardcoded_targets(tmp_path: Path) -> None:
    res_root = tmp_path / "res"
    write_variant_onnx_files(res_root, "demo", ("variant_a",))
    csv_path = tmp_path / "demo_monitor.csv"
    write_monitor_csv(
        csv_path,
        [build_monitor_row(res_root, "variant_a", "training", 12.0)],
    )

    graphs = process_csv(csv_path)

    assert len(graphs) == 1
    graph = graphs[0]
    assert graph.y.shape == (1, len(TARGET_FIELDS))
    assert graph.y[0].tolist() == pytest.approx([4.0, 75.0, 12.5, 2048.0])
    assert graph.batch_size == 2
    assert graph.gpu_node == "v100"
    assert graph.gpu_name == "v100"
    assert graph.graph_features.shape == (1, GRAPH_FEATURE_DIM)


def test_build_prepared_dataset_writes_manifest_and_loads(tmp_path: Path) -> None:
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    res_root = tmp_path / "res"
    variant_names = ("variant_a", "variant_b", "variant_c")
    write_variant_onnx_files(res_root, "demo", variant_names)
    write_monitor_csv(
        csv_dir / "demo_monitor.csv",
        [
            build_monitor_row(res_root, variant_name, phase, duration_sec)
            for variant_name, phase, duration_sec in (
                ("variant_a", "training", 12.0),
                ("variant_b", "inference", 15.0),
                ("variant_c", "training", 18.0),
            )
        ],
    )
    manifest_path = build_dataset(
        csv_dir=csv_dir,
        output_dir=tmp_path / "prepared",
        val_ratio=0.2,
        test_ratio=0.2,
        seed=1,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["target_names"] == list(TARGET_FIELDS)
    assert "extract_config_path" not in manifest
    assert manifest["sample_count"] == 3

    bundle = load_prepared_graph_datasets(
        PreparedDataConfig(manifest_path=str(manifest_path))
    )

    assert bundle.target_names == TARGET_FIELDS
    assert len(bundle.train_data) == 1
    assert len(bundle.val_data) == 1
    assert len(bundle.test_data) == 1
    assert bundle.train_data[0].y.shape == (1, len(TARGET_FIELDS))
    assert bundle.train_data[0].x.shape[1] == NODE_FEATURE_DIM
    assert bundle.train_data[0].phase in {"training", "inference"}


def write_monitor_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_monitor_row(
    res_root: Path,
    variant_name: str,
    phase: str,
    duration_sec: float,
) -> dict[str, object]:
    return {
        "target_name": "demo",
        "result_json": str(res_root / "demo/results/demo_results.json"),
        "config_path": str(res_root / "demo/config/demo.yaml"),
        "variant_name": variant_name,
        "base_model_name": "toy",
        "phase": phase,
        "gpu_node": "v100",
        "gpu_id": 0,
        "batch_size": 2,
        "sample_count": 3,
        "resolved_gpu_label": "0",
        "resolved_device_label": "nvidia0",
        "duration_sec": duration_sec,
        "phase_rounds": 3,
        "gpu_util_percent_p95": 75.0,
        "gpu_sm_occupancy_percent_p95": 12.5,
        "gpu_mem_used_mb_p95": 2048.0,
    }


def write_variant_onnx_files(
    res_root: Path,
    target_name: str,
    variant_names: tuple[str, ...],
) -> None:
    onnx_dir = res_root / target_name / "onnx_models"
    onnx_dir.mkdir(parents=True)
    first_path = onnx_dir / f"{variant_names[0]}.onnx"
    architecture_only_path = onnx_dir / "architecture.onnx"
    export_architecture_only_onnx(
        build_toy_model(0),
        architecture_only_path,
        (1, 3, 32, 32),
    )
    write_randomized_onnx_model(
        architecture_only_path,
        first_path,
        runtime_input_names=["inputs"],
        seed=17,
    )
    architecture_only_path.unlink()
    for variant_name in variant_names[1:]:
        shutil.copyfile(first_path, onnx_dir / f"{variant_name}.onnx")
