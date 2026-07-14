from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from gnn_model_test_utils import (
    TARGET_NAMES,
    build_toy_model,
    export_architecture_graph,
    write_result_json,
    write_split_config,
    write_split_dataset,
    write_target_scalers,
)

from gnn_model.data.extract import TARGET_FIELDS


def test_gnn_model_cli_runs_end_to_end(tmp_path: Path) -> None:
    data_dir = write_split_dataset(tmp_path / "scaled")
    scaler_dir = write_target_scalers(tmp_path / "scalers", TARGET_NAMES)
    config_path = tmp_path / "cli_config.yaml"
    write_split_config(
        config_path,
        experiment_name="cli_smoke",
        data_dir=data_dir,
        scaler_dir=scaler_dir,
    )
    output_dir = tmp_path / "cli_output"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gnn_model",
            "--config",
            str(config_path),
            "--output_dir",
            str(output_dir),
            "--device",
            "cpu",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result_files = list((output_dir / "cli_smoke" / "results").glob("*.json"))
    assert len(result_files) == 1
    payload = json.loads(result_files[0].read_text(encoding="utf-8"))
    assert payload["experiment_name"] == "cli_smoke"


def test_gnn_model_cli_extract_writes_manifest(tmp_path: Path) -> None:
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    res_root = tmp_path / "res"
    write_variant_graph_files(res_root, "demo", ("variant_a", "variant_b", "variant_c"))
    write_monitor_csv(
        csv_dir / "demo_monitor.csv",
        [
            build_monitor_row(res_root, "variant_a", "training", 12.0),
            build_monitor_row(res_root, "variant_b", "inference", 15.0),
            build_monitor_row(res_root, "variant_c", "training", 18.0),
        ],
    )
    output_dir = tmp_path / "prepared"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gnn_model.data.extract",
            "--csv_dir",
            str(csv_dir),
            "--output_dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    manifest_path = output_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "5.0.0"
    assert payload["feature_source"] == (
        "pytorch_export_inference_ir_static_metrics_shape_topology_v1"
    )
    assert payload["total_record_count"] == 3
    assert payload["target_names"] == list(TARGET_FIELDS)
    assert "extract_config_path" not in payload
    graphs = []
    for split_name in ("train", "val", "test"):
        graphs.extend(torch.load(output_dir / f"{split_name}.pt", weights_only=False))
    active_index = TARGET_FIELDS.index("gpu_sm_active_percent_max")
    occupancy_index = TARGET_FIELDS.index("gpu_sm_occupancy_percent_max")
    deployment_index = TARGET_FIELDS.index("deployment_duration_sec_avg")
    assert all(abs(float(graph.y[0, active_index]) - 65.0) < 1e-6 for graph in graphs)
    assert all(
        abs(float(graph.y[0, occupancy_index]) - 12.5) < 1e-6 for graph in graphs
    )
    assert all(
        abs(float(graph.y[0, deployment_index]) - 1.25) < 1e-6 for graph in graphs
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
    *,
    decode_output_length: int = 0,
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
        "decode_output_length": decode_output_length,
        "resolved_gpu_label": "0",
        "resolved_device_label": "nvidia0",
        "duration_sec": duration_sec,
        "deployment_duration_sec_avg": 1.25,
        "phase_rounds": 3,
        "cpu_cores_max": 4.25,
        "memory_gb_max": 1.5,
        "memory_delta_gb_max": 1.25,
        "gpu_util_percent_max": 75.0,
        "gpu_sm_active_percent_max": 65.0,
        "gpu_sm_occupancy_percent_max": 12.5,
        "gpu_mem_used_mb_max": 2048.0,
        "gpu_power_watts_avg": 100.0,
    }


def write_variant_graph_files(
    res_root: Path,
    target_name: str,
    variant_names: tuple[str, ...],
) -> None:
    graph_dir = res_root / target_name / "fx_graphs"
    graph_dir.mkdir(parents=True)
    first_path = graph_dir / f"{variant_names[0]}.pt2"
    export_architecture_graph(
        build_toy_model(0),
        first_path,
        (2, 3, 32, 32),
    )
    for variant_name in variant_names[1:]:
        shutil.copyfile(first_path, graph_dir / f"{variant_name}.pt2")
    write_result_json(res_root, target_name, variant_names)
