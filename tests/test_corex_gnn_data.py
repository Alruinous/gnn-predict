from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pytest
import torch

from gnn_model.config import SplitDataConfig
from gnn_model.data.constants import GPU_SPEC_FIELDS, GPU_SPECS, normalize_gpu_name
from gnn_model.data.dataset import load_split_graph_datasets
from gnn_model.data.extract import (
    COREX_BI_V150_TARGET_FIELDS,
    build_dataset,
    resolve_graph_path,
)
from gnn_model.data.scaler import generate_scalers, main as scaler_main
from gnn_model.runner import run_experiment
from gnn_model_test_utils import (
    build_synthetic_graph,
    write_split_config,
    write_split_dataset,
    write_target_scalers,
)
from test_gnn_model_cli import (
    build_monitor_row,
    write_monitor_csv,
    write_variant_graph_files,
)


def test_bi_v150_extracts_valid_rows_without_nvidia_sm_metrics(
    tmp_path: Path,
) -> None:
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    res_root = tmp_path / "res"
    variants = ("variant_a", "variant_b", "variant_c", "variant_d", "variant_elu_e")
    write_variant_graph_files(res_root, "demo", variants)
    rows = []
    for variant_name in variants:
        for phase in ("training", "inference"):
            row = build_monitor_row(res_root, variant_name, phase, 40.0)
            row["gpu_node"] = "bi-v150"
            row["gpu_sm_active_percent_max"] = ""
            row["gpu_sm_occupancy_percent_max"] = ""
            if phase == "training":
                row["batch_size"] = 16
            rows.append(row)
    write_monitor_csv(csv_dir / "monitor.csv", rows)

    output_dir = tmp_path / "prepared"
    manifest_path = build_dataset(csv_dirs=[csv_dir], output_dir=output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["target_names"] == list(COREX_BI_V150_TARGET_FIELDS)
    assert manifest["split_unit"] == "gpu_model_variant"
    assert manifest["sample_count"] == len(rows)
    assert manifest["graph_capture_batch_sizes"] == [2]
    assert manifest["quality_report"]["retained_row_count"] == len(rows)
    split_membership: dict[str, set[str]] = {}
    for split_name in ("train", "val", "test"):
        graphs = torch.load(output_dir / f"{split_name}.pt", weights_only=False)
        assert graphs
        for graph in graphs:
            assert graph.y.shape == (1, len(COREX_BI_V150_TARGET_FIELDS))
            assert graph.graph_capture_batch_size == 2
            assert graph.batch_size == (16 if graph.phase == "training" else 2)
            split_membership.setdefault(graph.variant_name, set()).add(split_name)
    assert all(len(splits) == 1 for splits in split_membership.values())
    assert "variant_elu_e" in split_membership


def test_bi_v150_and_nvidia_csvs_cannot_share_one_dataset(tmp_path: Path) -> None:
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    res_root = tmp_path / "res"
    write_variant_graph_files(res_root, "demo", ("variant_a",))
    corex_row = build_monitor_row(res_root, "variant_a", "training", 40.0)
    corex_row["gpu_node"] = "bi-v150"
    corex_row["gpu_sm_active_percent_max"] = ""
    corex_row["gpu_sm_occupancy_percent_max"] = ""
    write_monitor_csv(csv_dir / "corex.csv", [corex_row])
    write_monitor_csv(
        csv_dir / "nvidia.csv",
        [build_monitor_row(res_root, "variant_a", "training", 40.0)],
    )

    with pytest.raises(ValueError, match="incompatible target schemas"):
        build_dataset(csv_dirs=[csv_dir], output_dir=tmp_path / "prepared")


def test_bi_v150_aliases_and_neutral_hardware_features() -> None:
    assert normalize_gpu_name("Iluvatar BI-V150") == "bi-v150"
    assert normalize_gpu_name("bi_v150") == "bi-v150"
    assert GPU_SPECS["bi-v150"] == (0.0,) * len(GPU_SPEC_FIELDS)


def test_graph_path_is_resolved_relative_to_monitor_csv(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "moved" / "resnet50"
    graph_dir = experiment_dir / "fx_graphs"
    graph_dir.mkdir(parents=True)
    graph_path = graph_dir / "variant_a.pt2"
    graph_path.touch()
    csv_path = experiment_dir / "monitor.csv"

    assert resolve_graph_path(
        "results/result.json", "variant_a", csv_path=csv_path
    ) == graph_path
    assert resolve_graph_path(
        "/home/old-container/output/resnet50/results/result.json",
        "variant_a",
        csv_path=csv_path,
    ) == graph_path
    assert resolve_graph_path(
        "output/resnet50/results/result.json", "variant_a", csv_path=csv_path
    ) == graph_path


def test_scalers_can_fit_only_the_training_split(tmp_path: Path) -> None:
    train_graph = build_synthetic_graph(0, target_dim=1)
    val_graph = build_synthetic_graph(100, target_dim=1)
    train_file = tmp_path / "train.pt"
    torch.save([train_graph], train_file)
    torch.save([val_graph], tmp_path / "val.pt")

    _, target_scalers = generate_scalers([train_file], ["runtime"])

    assert float(target_scalers["runtime"].center_[0]) == 0.0


def test_scaler_cli_reads_manifest_and_does_not_fit_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "manifest.json").write_text(
        json.dumps({"target_names": ["runtime"]}), encoding="utf-8"
    )
    for name, index in (("train", 0), ("val", 100), ("test", 200)):
        torch.save(
            [build_synthetic_graph(index, target_dim=1)],
            raw_dir / f"{name}.pt",
        )
    scaler_dir = tmp_path / "scalers"
    scaled_dir = tmp_path / "scaled"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scaler",
            "--data_dir",
            str(raw_dir),
            "--scaler_output_path",
            str(scaler_dir),
            "--scaled_data_output_path",
            str(scaled_dir),
        ],
    )

    scaler_main()

    with (scaler_dir / "target_scalers.pkl").open("rb") as file:
        scalers = pickle.load(file)
    assert float(scalers["runtime"].center_[0]) == 0.0
    assert all((scaled_dir / f"{name}.pt").exists() for name in ("train", "val", "test"))
    scaled_manifest = json.loads(
        (scaled_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert scaled_manifest["normalization"]["fit_split"] == "train"


def test_training_rejects_target_order_mismatch(tmp_path: Path) -> None:
    data_dir = write_split_dataset(tmp_path / "scaled", target_dim=2)
    (data_dir / "manifest.json").write_text(
        json.dumps({"target_names": ["runtime", "memory"]}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="target_names do not match"):
        load_split_graph_datasets(
            SplitDataConfig(
                data_dir=str(data_dir),
                target_names=["memory", "runtime"],
            )
        )


def test_bi_v150_target_schema_trains_gnn_on_cpu(tmp_path: Path) -> None:
    target_names = COREX_BI_V150_TARGET_FIELDS
    data_dir = write_split_dataset(tmp_path / "scaled", target_dim=len(target_names))
    scaler_dir = write_target_scalers(tmp_path / "scalers", target_names)
    config_path = write_split_config(
        tmp_path / "corex_training.yaml",
        experiment_name="corex_training_test",
        data_dir=data_dir,
        scaler_dir=scaler_dir,
        target_names=target_names,
    )

    result, result_path = run_experiment(
        config_path=config_path,
        output_dir=tmp_path / "output",
        requested_device="cpu",
    )

    assert result.target_names == list(target_names)
    assert result.training.best_epoch == 1
    assert Path(result.training.checkpoint_path).exists()
    assert result_path.exists()
