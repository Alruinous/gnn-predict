from __future__ import annotations

import csv
import json
import math
import shutil
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from gnn_model.config import PreparedDataConfig, SplitDataConfig
from gnn_model.data.constants import (
    EDGE_FEATURE_DIM,
    EDGE_FEATURE_NAMES,
    GRAPH_FEATURE_DIM,
    GRAPH_FEATURE_NAMES,
    NODE_FEATURE_DIM,
    NODE_FEATURE_NAMES,
    OP_TYPE_NAMES,
    OP_TYPE_TO_INDEX,
)
from gnn_model.data.dataset import load_graph_split, load_split_graph_datasets
from gnn_model.data.extract import (
    TARGET_FIELDS,
    ModelRecordInfo,
    build_model_record_info,
    build_dataset,
    process_csv,
    resolve_onnx_path,
    resolve_profile_summary,
)
from gnn_model.data.onnx_graph import (
    build_graph_data_from_onnx,
    count_elements,
    estimate_peak_live_activation_bytes,
    resolve_profile_tensor_shape,
    sum_tensor_bytes,
    validate_or_build_execution_order,
)
from gnn_model.data.prepared_dataset import load_prepared_graph_datasets
from gnn_model_test_utils import (
    TARGET_NAMES,
    build_synthetic_graph,
    build_toy_model,
    export_architecture_only_onnx,
    write_result_json,
    write_split_dataset,
)


def test_build_graph_data_from_onnx_returns_expected_shapes(tmp_path: Path) -> None:
    architecture_only_path = tmp_path / "toy_architecture.onnx"

    export_architecture_only_onnx(
        build_toy_model(0),
        architecture_only_path,
        (1, 3, 32, 32),
    )

    data = build_graph_data_from_onnx(
        architecture_only_path,
        batch_size=8,
        gpu_name="v100",
        phase="inference",
        sample_count=3,
    )

    assert data.x.shape[1] == NODE_FEATURE_DIM
    assert data.edge_attr.shape[1] == EDGE_FEATURE_DIM
    assert data.graph_features.shape == (1, GRAPH_FEATURE_DIM)
    assert data.graph_features[0, 0].item() == 1.0
    assert data.graph_features[0, 1].item() == 8.0
    assert data.graph_features[0, 2].item() == 3.0
    profile_available_index = GRAPH_FEATURE_NAMES.index("profile_available")
    assert data.graph_features[0, profile_available_index].item() == 0.0
    profile_macs_index = GRAPH_FEATURE_NAMES.index("profile_total_macs")
    assert data.graph_features[0, profile_macs_index] > 0
    assert not hasattr(data, "graph_metrics")
    assert data.edge_index.shape[0] == 2


def test_build_graph_data_from_onnx_adds_op_ids_and_shape_features(
    tmp_path: Path,
) -> None:
    architecture_only_path = tmp_path / "toy_architecture.onnx"
    export_architecture_only_onnx(
        build_toy_model(0),
        architecture_only_path,
        (1, 3, 32, 32),
    )

    data = build_graph_data_from_onnx(
        architecture_only_path,
        batch_size=4,
        gpu_name="v100",
        phase="inference",
        sample_count=1,
    )

    assert isinstance(data.op_type_ids, torch.Tensor)
    assert data.op_type_ids.shape == (data.x.size(0),)
    assert data.op_type_ids.dtype == torch.long
    assert OP_TYPE_TO_INDEX["op_conv"] in data.op_type_ids.tolist()
    assert OP_TYPE_TO_INDEX["op_activation"] in data.op_type_ids.tolist()
    assert OP_TYPE_TO_INDEX["op_dense"] in data.op_type_ids.tolist()
    assert all(name not in NODE_FEATURE_NAMES for name in OP_TYPE_NAMES)

    edge_channel_index = EDGE_FEATURE_NAMES.index("tensor_dim1_log")
    assert data.edge_attr[:, edge_channel_index].max().item() >= math.log1p(4)
    activation_index = GRAPH_FEATURE_NAMES.index("activation_bytes_sum_log")
    peak_activation_index = GRAPH_FEATURE_NAMES.index("peak_live_activation_bytes_log")
    assert data.graph_features[0, activation_index].item() > 0
    activation_bytes = math.expm1(data.graph_features[0, activation_index].item())
    peak_live_bytes = math.expm1(
        data.graph_features[0, peak_activation_index].item()
    )
    assert 0 <= peak_live_bytes <= activation_bytes


def test_build_graph_data_from_onnx_adds_variant_context_features(
    tmp_path: Path,
) -> None:
    architecture_only_path = tmp_path / "yolo11_tiny_ic3_oc80_activation_relu.onnx"
    export_architecture_only_onnx(
        build_toy_model(0),
        architecture_only_path,
        (1, 3, 32, 32),
    )

    data = build_graph_data_from_onnx(
        architecture_only_path,
        batch_size=4,
        gpu_name="v100",
        phase="inference",
        sample_count=1,
        model_name="yolo11_tiny",
        variant_name="yolo11_tiny_ic3_oc80_activation_relu",
    )

    family_index = GRAPH_FEATURE_NAMES.index("variant_family_yolov11")
    size_index = GRAPH_FEATURE_NAMES.index("variant_size_tiny")
    activation_index = GRAPH_FEATURE_NAMES.index("variant_activation_relu")
    input_channels_index = GRAPH_FEATURE_NAMES.index(
        "variant_numeric_input_channels_log"
    )
    output_classes_index = GRAPH_FEATURE_NAMES.index(
        "variant_numeric_output_classes_log"
    )
    assert data.graph_features[0, family_index].item() == 1.0
    assert data.graph_features[0, size_index].item() == 1.0
    assert data.graph_features[0, activation_index].item() == 1.0
    assert data.graph_features[0, input_channels_index].item() == pytest.approx(
        math.log1p(3.0)
    )
    assert data.graph_features[0, output_classes_index].item() == pytest.approx(
        math.log1p(80.0)
    )


def test_build_graph_data_from_onnx_adds_runtime_profile_features(
    tmp_path: Path,
) -> None:
    architecture_only_path = tmp_path / "toy_architecture.onnx"
    export_architecture_only_onnx(
        build_toy_model(0),
        architecture_only_path,
        (1, 3, 32, 32),
    )

    data = build_graph_data_from_onnx(
        architecture_only_path,
        batch_size=4,
        gpu_name="v100",
        phase="inference",
        sample_count=1,
        profile_summary={
            "profiled_steps": 2,
            "wall_time_sec": 1.5,
            "event_count": 10,
            "op_count": 7,
            "launch_event_count": 12,
            "kernel_event_count": 15,
            "total_device_time_us": 1000.0,
            "total_self_device_time_us": 700.0,
            "total_device_memory_pos": 2048.0,
            "max_event_device_memory": 1024.0,
            "peak_device_memory": 4096.0,
            "total_flops": 123456.0,
            "conv_device_time_share": 0.4,
            "matmul_device_time_share": 0.2,
            "top_events": [
                {"key": "kernel_a", "device_time_us": 500.0},
                {"key": "kernel_b", "device_time_us": 300.0},
            ],
        },
    )

    profile_available_index = GRAPH_FEATURE_NAMES.index("profile_available")
    steps_index = GRAPH_FEATURE_NAMES.index("profiled_steps")
    launch_index = GRAPH_FEATURE_NAMES.index("profile_launch_event_count_log")
    top1_index = GRAPH_FEATURE_NAMES.index("profile_top1_device_time_us_log")
    top_share_index = GRAPH_FEATURE_NAMES.index("profile_top_event_time_share")
    top3_share_index = GRAPH_FEATURE_NAMES.index("profile_top3_event_time_share")
    top_category_count_index = GRAPH_FEATURE_NAMES.index(
        "profile_top_event_category_count"
    )
    top_other_index = GRAPH_FEATURE_NAMES.index("profile_top_other_device_time_share")
    assert data.graph_features.shape == (1, GRAPH_FEATURE_DIM)
    assert data.graph_features[0, profile_available_index].item() == 1.0
    assert data.graph_features[0, steps_index].item() == 2.0
    assert data.graph_features[0, launch_index].item() == pytest.approx(
        math.log1p(12.0)
    )
    assert data.graph_features[0, top1_index].item() == pytest.approx(
        math.log1p(500.0)
    )
    assert data.graph_features[0, top_share_index].item() == pytest.approx(0.5)
    assert data.graph_features[0, top3_share_index].item() == pytest.approx(0.8)
    assert data.graph_features[0, top_category_count_index].item() == 1.0
    assert data.graph_features[0, top_other_index].item() == pytest.approx(0.8)


def test_unknown_op_type_uses_other_category() -> None:
    from gnn_model.data.onnx_graph import resolve_op_type_index

    assert resolve_op_type_index("CustomExperimentalOp") == OP_TYPE_TO_INDEX["op_other"]


def test_resolve_profile_summary_reads_phase_specific_result(tmp_path: Path) -> None:
    result_json = tmp_path / "result.json"
    result_json.write_text(
        json.dumps(
            {
                "variants": [
                    {
                        "name": "variant_a",
                        "training": {"profile_summary": {"event_count": 3}},
                        "inference": {"profile_summary": {"event_count": 5}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = resolve_profile_summary(result_json, "variant_a", "inference")

    assert summary == {"event_count": 5}


def test_resolve_profile_tensor_shape_keeps_scalar_and_zero_length_shapes() -> None:
    scalar_shape = resolve_profile_tensor_shape(SimpleNamespace(shape=()))
    zero_length_shape = resolve_profile_tensor_shape(SimpleNamespace(shape=(0,)))

    assert scalar_shape == ()
    assert count_elements(scalar_shape) == 1
    assert zero_length_shape == (0,)
    assert count_elements(zero_length_shape) == 0


def test_estimate_peak_live_activation_bytes_keeps_branch_tensor_until_last_use() -> None:
    graph = build_lifetime_graph(
        {
            "a": ((), ("a_out",)),
            "b": (("a_out",), ("b_out",)),
            "c": (("a_out",), ("c_out",)),
            "d": (("b_out", "c_out"), ("d_out",)),
        },
        {
            "a_out": (100,),
            "b_out": (40,),
            "c_out": (60,),
            "d_out": (20,),
        },
    )

    peak_bytes = estimate_peak_live_activation_bytes(
        graph=graph,
        node_names=["a", "b", "c", "d"],
        raw_edges=[
            (0, 1, "a_out"),
            (0, 2, "a_out"),
            (1, 3, "b_out"),
            (2, 3, "c_out"),
        ],
        graph_output_names={"d_out"},
    )

    assert peak_bytes == 200
    assert peak_bytes <= sum_tensor_bytes(graph, graph.tensormap)


def test_estimate_peak_live_activation_bytes_keeps_graph_output_live() -> None:
    graph = build_lifetime_graph(
        {
            "a": ((), ("graph_out",)),
            "b": ((), ("orphan",)),
        },
        {
            "graph_out": (100,),
            "orphan": (20,),
        },
    )

    peak_bytes = estimate_peak_live_activation_bytes(
        graph=graph,
        node_names=["a", "b"],
        raw_edges=[],
        graph_output_names={"graph_out"},
    )

    assert peak_bytes == 120


def test_estimate_peak_live_activation_bytes_handles_zero_sized_tensor() -> None:
    graph = build_lifetime_graph(
        {"a": ((), ("zero_out",))},
        {"zero_out": (0,)},
    )

    peak_bytes = estimate_peak_live_activation_bytes(
        graph=graph,
        node_names=["a"],
        raw_edges=[],
        graph_output_names={"zero_out"},
    )

    assert peak_bytes == 0


def test_estimate_peak_live_activation_bytes_falls_back_to_topological_order() -> None:
    graph = build_lifetime_graph(
        {
            "consumer": (("producer_out",), ("consumer_out",)),
            "producer": ((), ("producer_out",)),
        },
        {
            "producer_out": (8,),
            "consumer_out": (8,),
        },
    )
    raw_edges = [(1, 0, "producer_out")]

    peak_bytes = estimate_peak_live_activation_bytes(
        graph=graph,
        node_names=["consumer", "producer"],
        raw_edges=raw_edges,
        graph_output_names={"consumer_out"},
    )

    assert validate_or_build_execution_order(2, raw_edges) == [1, 0]
    assert peak_bytes == 16


def test_build_graph_data_from_onnx_supports_squeeze_without_axes(
    tmp_path: Path,
) -> None:
    class SqueezeLinearModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(3, 1)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.proj(inputs.squeeze())

    architecture_only_path = tmp_path / "squeeze_architecture.onnx"
    export_architecture_only_onnx(
        SqueezeLinearModel(),
        architecture_only_path,
        (2, 3, 1),
    )

    data = build_graph_data_from_onnx(
        architecture_only_path,
        batch_size=2,
        gpu_name="v100",
        phase="training",
        sample_count=1,
    )

    profile_macs_index = GRAPH_FEATURE_NAMES.index("profile_total_macs")
    assert data.x.shape[1] == NODE_FEATURE_DIM
    assert data.edge_attr.shape[1] == EDGE_FEATURE_DIM
    assert data.graph_features[0, profile_macs_index] > 0


def test_build_graph_data_from_onnx_supports_softplus(tmp_path: Path) -> None:
    class SoftplusModel(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.softplus(inputs)

    architecture_only_path = tmp_path / "softplus_architecture.onnx"
    export_architecture_only_onnx(
        SoftplusModel(),
        architecture_only_path,
        (1, 4),
    )

    data = build_graph_data_from_onnx(
        architecture_only_path,
        batch_size=2,
        gpu_name="v100",
        phase="training",
        sample_count=1,
    )

    profile_macs_index = GRAPH_FEATURE_NAMES.index("profile_total_macs")
    assert data.x.shape == (1, NODE_FEATURE_DIM)
    assert data.edge_attr.shape[1] == EDGE_FEATURE_DIM
    assert data.graph_features.shape == (1, GRAPH_FEATURE_DIM)
    assert data.graph_features[0, profile_macs_index] > 0
    assert data.x[0, 0] > 0


def test_build_graph_data_from_onnx_supports_elu(tmp_path: Path) -> None:
    class EluModel(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.elu(inputs)

    architecture_only_path = tmp_path / "elu_architecture.onnx"
    export_architecture_only_onnx(
        EluModel(),
        architecture_only_path,
        (1, 4),
    )

    data = build_graph_data_from_onnx(
        architecture_only_path,
        batch_size=2,
        gpu_name="v100",
        phase="training",
        sample_count=1,
    )

    profile_macs_index = GRAPH_FEATURE_NAMES.index("profile_total_macs")
    assert data.x.shape == (1, NODE_FEATURE_DIM)
    assert data.edge_attr.shape[1] == EDGE_FEATURE_DIM
    assert data.graph_features.shape == (1, GRAPH_FEATURE_DIM)
    assert data.graph_features[0, profile_macs_index] > 0
    assert data.x[0, 0] > 0


def test_build_graph_data_from_onnx_supports_selu(tmp_path: Path) -> None:
    class SeluModel(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.selu(inputs)

    architecture_only_path = tmp_path / "selu_architecture.onnx"
    export_architecture_only_onnx(
        SeluModel(),
        architecture_only_path,
        (1, 4),
    )

    data = build_graph_data_from_onnx(
        architecture_only_path,
        batch_size=2,
        gpu_name="v100",
        phase="training",
        sample_count=1,
    )

    profile_macs_index = GRAPH_FEATURE_NAMES.index("profile_total_macs")
    assert data.x.shape == (1, NODE_FEATURE_DIM)
    assert data.edge_attr.shape[1] == EDGE_FEATURE_DIM
    assert data.graph_features.shape == (1, GRAPH_FEATURE_DIM)
    assert data.graph_features[0, profile_macs_index] > 0
    assert data.x[0, 0] > 0


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


def test_load_prepared_graph_datasets_rejects_feature_name_mismatch(
    tmp_path: Path,
) -> None:
    data_dir = write_split_dataset(tmp_path / "prepared")
    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "3.0.0",
                "target_names": list(TARGET_NAMES),
                "node_feature_names": ["bad_node_feature"],
                "op_type_names": list(OP_TYPE_NAMES),
                "edge_feature_names": list(EDGE_FEATURE_NAMES),
                "graph_feature_names": list(GRAPH_FEATURE_NAMES),
                "split_files": {
                    "train": "train.pt",
                    "val": "val.pt",
                    "test": "test.pt",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="node_feature_names mismatch"):
        load_prepared_graph_datasets(PreparedDataConfig(manifest_path=str(manifest_path)))


def test_build_model_record_info_resolves_path_and_targets(tmp_path: Path) -> None:
    model_path = tmp_path / "res/demo/onnx_models/variant_a.onnx"
    write_result_json(tmp_path / "res", "demo", ("variant_a",))
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
    assert info.target == pytest.approx((2.5, 4.25, 1.25, 75.0, 65.0, 12.5, 2048.0))


def test_process_csv_builds_graphs_with_hardcoded_targets(tmp_path: Path) -> None:
    res_root = tmp_path / "res"
    write_variant_onnx_files(res_root, "demo", ("variant_a",))
    csv_path = tmp_path / "demo_monitor.csv"
    write_monitor_csv(
        csv_path,
        [build_monitor_row(res_root, "variant_a", "training", 12.0)],
    )

    graphs = process_csv(csv_path).records

    assert len(graphs) == 1
    graph = graphs[0]
    assert graph.y.shape == (1, len(TARGET_FIELDS))
    assert graph.y[0].tolist() == pytest.approx(
        [4.0, 4.25, 1.25, 75.0, 65.0, 12.5, 2048.0]
    )
    assert graph.batch_size == 2
    assert graph.gpu_node == "v100"
    assert graph.gpu_name == "v100"
    assert graph.graph_features.shape == (1, GRAPH_FEATURE_DIM)


def test_load_graph_split_pads_legacy_graph_features(tmp_path: Path) -> None:
    graph = build_synthetic_graph(0)
    legacy_dim = GRAPH_FEATURE_DIM - 3
    graph.graph_features = graph.graph_features[:, :legacy_dim]
    split_path = tmp_path / "train.pt"
    torch.save([graph], split_path)

    loaded_graphs = load_graph_split(split_path, target_dim=len(TARGET_NAMES))

    loaded_graph = loaded_graphs[0]
    assert loaded_graph.graph_features.shape == (1, GRAPH_FEATURE_DIM)
    assert loaded_graph.graph_features[0, :legacy_dim].tolist() == pytest.approx(
        graph.graph_features[0].tolist()
    )
    assert loaded_graph.graph_features[0, legacy_dim:].tolist() == [0.0, 0.0, 0.0]


def test_process_csv_filters_untrusted_gpu_metrics(tmp_path: Path) -> None:
    res_root = tmp_path / "res"
    variant_names = (
        "keep_normal",
        "bad_gpu_mem_zero",
        "bad_gpu_util_with_mem",
        "bad_gpu_util_with_sm",
        "bad_gpu_sm_with_util",
        "keep_high_gpu_mem_tail",
    )
    write_variant_onnx_files(res_root, "demo", variant_names)
    rows = [
        build_monitor_row(res_root, "keep_normal", "training", 12.0),
        {
            **build_monitor_row(res_root, "bad_gpu_mem_zero", "training", 12.0),
            "gpu_mem_used_mb_p95": 0.0,
        },
        {
            **build_monitor_row(res_root, "bad_gpu_util_with_mem", "training", 12.0),
            "gpu_util_percent_p95": 0.0,
        },
        {
            **build_monitor_row(res_root, "bad_gpu_util_with_sm", "training", 12.0),
            "gpu_util_percent_p95": 0.0,
            "gpu_sm_occupancy_percent_p95": 0.08,
        },
        {
            **build_monitor_row(res_root, "bad_gpu_sm_with_util", "training", 12.0),
            "gpu_util_percent_p95": 5.0,
            "gpu_sm_occupancy_percent_p95": 0.0,
        },
        {
            **build_monitor_row(res_root, "keep_high_gpu_mem_tail", "training", 12.0),
            "memory_delta_gb_p95": 12.0,
            "gpu_mem_used_mb_p95": 28347.0,
        },
    ]
    csv_path = tmp_path / "demo_monitor.csv"
    write_monitor_csv(csv_path, rows)

    graphs = process_csv(csv_path).records

    assert [graph.variant_name for graph in graphs] == [
        "keep_normal",
        "keep_high_gpu_mem_tail",
    ]
    high_tail = graphs[1]
    memory_delta_index = TARGET_FIELDS.index("memory_delta_gb_p95")
    gpu_mem_index = TARGET_FIELDS.index("gpu_mem_used_mb_p95")
    assert high_tail.y[0, memory_delta_index].item() == pytest.approx(12.0)
    assert high_tail.y[0, gpu_mem_index].item() == pytest.approx(28347.0)


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
    assert manifest["schema_version"] == "3.0.0"
    assert manifest["feature_source"] == "onnx_tool_profile_p0_features"
    assert manifest["target_names"] == list(TARGET_FIELDS)
    assert manifest["node_feature_names"] == list(NODE_FEATURE_NAMES)
    assert manifest["op_type_names"] == list(OP_TYPE_NAMES)
    assert manifest["edge_feature_names"] == list(EDGE_FEATURE_NAMES)
    assert manifest["graph_feature_names"] == list(GRAPH_FEATURE_NAMES)
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


def build_lifetime_graph(
    nodes: dict[str, tuple[tuple[str, ...], tuple[str, ...]]],
    tensor_shapes: dict[str, tuple[int, ...]],
) -> SimpleNamespace:
    return SimpleNamespace(
        nodemap={
            name: SimpleNamespace(input=inputs, output=outputs)
            for name, (inputs, outputs) in nodes.items()
        },
        tensormap={
            name: SimpleNamespace(shape=shape, dtype="uint8")
            for name, shape in tensor_shapes.items()
        },
    )


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
        "cpu_cores_p95": 4.25,
        "memory_gb_p95": 1.5,
        "memory_delta_gb_p95": 1.25,
        "gpu_util_percent_p95": 75.0,
        "gpu_sm_active_percent_p95": 65.0,
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
    export_architecture_only_onnx(
        build_toy_model(0),
        first_path,
        (1, 3, 32, 32),
    )
    for variant_name in variant_names[1:]:
        shutil.copyfile(first_path, onnx_dir / f"{variant_name}.onnx")
    write_result_json(res_root, target_name, variant_names)
