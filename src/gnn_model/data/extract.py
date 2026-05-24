from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import torch
from torch_geometric.data import Data

from common.log import get_logger
from gnn_model.data.constants import (
    EDGE_FEATURE_NAMES,
    GRAPH_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    OP_TYPE_NAMES,
    normalize_gpu_name,
)
from gnn_model.data.dataset import SPLIT_FILE_NAMES, resolve_split_counts

ID_FIELDS = ("variant_name", "phase")
TARGET_FIELDS = (
    "duration_sec_avg",
    "cpu_cores_p95",
    "memory_delta_gb_p95",
    "gpu_util_percent_p95",
    "gpu_sm_active_percent_p95",
    "gpu_sm_occupancy_percent_p95",
    "gpu_mem_used_mb_p95",
)
METADATA_FIELDS = (
    "target_name",
    "variant_name",
    "base_model_name",
    "phase",
    "gpu_node",
    "gpu_id",
    "batch_size",
    "sample_count",
    "resolved_gpu_label",
    "resolved_device_label",
    "result_json",
    "config_path",
)
SOURCE_TARGET_FIELDS = tuple(
    field for field in TARGET_FIELDS if field != "duration_sec_avg"
)
REQUIRED_COLUMNS = (
    *METADATA_FIELDS,
    "duration_sec",
    "phase_rounds",
    *SOURCE_TARGET_FIELDS,
)

logger = get_logger(Path(__file__).name)
GPU_SM_OCCUPANCY_CONFLICT_THRESHOLD = 0.05
GPU_UTIL_CONFLICT_THRESHOLD = 5.0
MEMORY_DELTA_LOG_THRESHOLD_GB = 10.0


@dataclass(frozen=True)
class ModelRecordInfo:
    csv_path: Path
    row_number: int
    model_path: Path
    model_name: str
    variant_name: str
    phase: str
    batch_size: int
    gpu_name: str
    target: tuple[float, ...]
    metadata: dict[str, object]

    @property
    def sample_id(self) -> str:
        return f"{self.variant_name}::{self.phase}"


@dataclass(frozen=True)
class CsvProcessResult:
    records: list[Data]
    gpu_quality_filtered_count: int
    high_memory_delta_count: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract monitoring CSV rows into a prepared gnn_model dataset.",
    )
    parser.add_argument(
        "--csv_dir",
        default="csv",
        help="Directory containing *_monitor.csv files.",
    )
    parser.add_argument(
        "--output_dir",
        default="data",
        help="Directory for manifest.json and split .pt files.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Split random seed.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_dataset(
        csv_dir=Path(args.csv_dir),
        output_dir=Path(args.output_dir),
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    return 0


def build_dataset(
    *,
    csv_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Path:
    csv_paths = sorted(Path(csv_dir).glob("*.csv"))
    assert csv_paths, f"No csv files found in {csv_dir}"
    csv_results = [process_csv(csv_path) for csv_path in csv_paths]
    graphs = [graph for result in csv_results for graph in result.records]
    assert graphs
    total_gpu_quality_filtered_count = sum(
        result.gpu_quality_filtered_count for result in csv_results
    )
    total_high_memory_delta_count = sum(
        result.high_memory_delta_count for result in csv_results
    )
    logger.info(
        f"GPU metric quality filter total: CSVs: {len(csv_results)}, "
        f"Filtered: {total_gpu_quality_filtered_count}, "
        f"HighMemoryDeltaGt10: {total_high_memory_delta_count}"
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_data, val_data, test_data = split_graphs(
        graphs,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    split_data = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }
    for split_name, file_name in SPLIT_FILE_NAMES.items():
        torch.save(split_data[split_name], output_path / file_name)

    manifest = build_manifest(
        csv_dir=csv_dir,
        split_data=split_data,
        sample_count=len(graphs),
        total_record_count=len(graphs),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_path


def process_csv(csv_file: str | Path) -> CsvProcessResult:
    logger.info(f"Processing CSV: {csv_file}")
    start_time = time.time()
    csv_path = Path(csv_file)
    df = pl.read_csv(csv_path)
    assert set(REQUIRED_COLUMNS) <= set(df.columns), csv_path
    df = df.with_columns(
        (pl.col("duration_sec") / pl.col("phase_rounds")).alias("duration_sec_avg"),
        pl.struct(["result_json", "variant_name"])
        .map_elements(
            lambda row: str(resolve_onnx_path(row["result_json"], row["variant_name"])),
            return_dtype=pl.String,
        )
        .alias("variant_path"),
    )
    df = df.drop_nans(subset=["batch_size", *TARGET_FIELDS]).filter(
        pl.all_horizontal(
            [
                pl.col("variant_name").str.len_chars() > 0,
                ~pl.col("variant_name").str.to_lowercase().str.contains("_elu_"),
                pl.col("variant_path").map_elements(
                    lambda path: Path(path).is_file(),
                    return_dtype=pl.Boolean,
                ),
            ]
        )
    )
    assert df.height > 0, csv_path
    pre_gpu_quality_count = df.height
    high_memory_delta_count = df.filter(
        pl.col("memory_delta_gb_p95") > MEMORY_DELTA_LOG_THRESHOLD_GB
    ).height
    df = df.filter(~invalid_gpu_metric_expr())
    gpu_quality_filtered_count = pre_gpu_quality_count - df.height
    assert df.unique(ID_FIELDS).height == df.height, csv_path
    records = [
        extract_feature_target(
            build_model_record_info(
                record,
                row_number=i,
                csv_path=csv_path,
            )
        )
        for i, record in enumerate(df.to_dicts())
    ]

    end_time = time.time()
    duration = end_time - start_time
    logger.info(
        f"Processed CSV: {csv_file}, Records: {len(records)}, "
        f"GPUQualityFiltered: {gpu_quality_filtered_count}, "
        f"HighMemoryDeltaGt10: {high_memory_delta_count}, "
        f"Duration: {duration:.2f} seconds"
    )
    return CsvProcessResult(
        records=records,
        gpu_quality_filtered_count=gpu_quality_filtered_count,
        high_memory_delta_count=high_memory_delta_count,
    )


def invalid_gpu_metric_expr() -> pl.Expr:
    gpu_mem = pl.col("gpu_mem_used_mb_p95")
    gpu_util = pl.col("gpu_util_percent_p95")
    gpu_sm_occupancy = pl.col("gpu_sm_occupancy_percent_p95")
    return (
        (gpu_mem <= 0)
        | ((gpu_util <= 0) & (gpu_mem > 0))
        | (
            (gpu_util <= 0)
            & (gpu_sm_occupancy > GPU_SM_OCCUPANCY_CONFLICT_THRESHOLD)
        )
        | ((gpu_sm_occupancy <= 0) & (gpu_util >= GPU_UTIL_CONFLICT_THRESHOLD))
    )


def build_model_record_info(
    record: Mapping[str, object],
    row_number: int,
    *,
    csv_path: Path,
) -> ModelRecordInfo:
    phase = str(record["phase"]).strip()
    assert phase in {"training", "inference"}
    batch_size = parse_int_value(record["batch_size"])
    assert batch_size > 0
    gpu_name = normalize_gpu_name(record["gpu_node"])
    target = tuple(parse_float_value(record[field]) for field in TARGET_FIELDS)
    assert all(math.isfinite(value) for value in target)
    metadata = {field: record[field] for field in METADATA_FIELDS}
    return ModelRecordInfo(
        csv_path=csv_path,
        row_number=row_number,
        model_path=Path(str(record["variant_path"])),
        model_name=str(record["base_model_name"]).strip(),
        variant_name=str(record["variant_name"]).strip(),
        phase=phase,
        batch_size=batch_size,
        gpu_name=gpu_name,
        target=target,
        metadata=metadata,
    )


def resolve_onnx_path(result_json: object, variant_name: object) -> Path:
    result_path = Path(str(result_json).strip())
    variant = str(variant_name).strip()
    return result_path.parent.parent / "onnx_models" / f"{variant}.onnx"


def parse_int_value(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("boolean values are not valid integers")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"integer value must be whole: {value}")
        return int(value)
    if isinstance(value, str):
        parsed = float(value)
        if not parsed.is_integer():
            raise ValueError(f"integer value must be whole: {value}")
        return int(parsed)
    raise TypeError(f"unsupported integer value type: {type(value)}")


def parse_float_value(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError("boolean values are not valid floats")
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"unsupported float value type: {type(value)}")


def extract_feature_target(info: ModelRecordInfo) -> Data:
    from gnn_model.data.onnx_graph import build_graph_data_from_onnx

    assert info.model_path.exists()
    data = build_graph_data_from_onnx(
        info.model_path,
        batch_size=info.batch_size,
        gpu_name=info.gpu_name,
        phase=info.phase,
        sample_count=parse_int_value(info.metadata["sample_count"]),
    )
    data.y = torch.tensor(info.target, dtype=torch.float32).unsqueeze(0)
    data.source_csv = str(info.csv_path)
    data.csv_row_number = info.row_number
    data.model_name = info.model_name
    data.variant_name = info.variant_name
    data.phase = info.phase
    data.batch_size = info.batch_size
    data.gpu_name = info.gpu_name
    for field, value in info.metadata.items():
        setattr(data, field, value)
    return data


def split_graphs(
    graphs: list[Data],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[Data], list[Data], list[Data]]:
    train_count, val_count, test_count = resolve_split_counts(
        len(graphs),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    indices = list(range(len(graphs)))
    random.Random(seed).shuffle(indices)
    train_indices = indices[:train_count]
    val_indices = indices[train_count : train_count + val_count]
    test_indices = indices[
        train_count + val_count : train_count + val_count + test_count
    ]
    return (
        [graphs[index] for index in train_indices],
        [graphs[index] for index in val_indices],
        [graphs[index] for index in test_indices],
    )


def build_manifest(
    *,
    csv_dir: Path,
    split_data: dict[str, list[Data]],
    sample_count: int,
    total_record_count: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": "3.0.0",
        "feature_source": "onnx_tool_profile_p0_features",
        "csv_dir": str(Path(csv_dir).resolve()),
        "id_fields": list(ID_FIELDS),
        "target_names": list(TARGET_FIELDS),
        "node_feature_names": list(NODE_FEATURE_NAMES),
        "op_type_names": list(OP_TYPE_NAMES),
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "graph_feature_names": list(GRAPH_FEATURE_NAMES),
        "split_files": SPLIT_FILE_NAMES,
        "split_counts": {
            split_name: len(graphs) for split_name, graphs in split_data.items()
        },
        "sample_count": sample_count,
        "total_record_count": total_record_count,
        "split_seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }


if __name__ == "__main__":
    raise SystemExit(main())
