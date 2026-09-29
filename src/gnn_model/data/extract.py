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
    "deployment_duration_sec_avg",
    "run_duration_sec_avg",
    "cpu_cores_max",
    "memory_delta_gb_max",
    "gpu_util_percent_max",
    "gpu_sm_active_percent_max",
    "gpu_sm_occupancy_percent_max",
    "gpu_mem_used_mb_max",
    "gpu_power_watts_avg",
)
COREX_BI_V150_TARGET_FIELDS = (
    "deployment_duration_sec_avg",
    "run_duration_sec_avg",
    "cpu_cores_max",
    "memory_delta_gb_max",
    "gpu_util_percent_max",
    "gpu_mem_used_mb_max",
    "gpu_power_watts_avg",
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
    "decode_output_length",
    "resolved_gpu_label",
    "resolved_device_label",
    "result_json",
    "config_path",
)
RUN_DURATION_SOURCE_FIELD = "run_duration_sec_avg"
SOURCE_TARGET_FIELDS = tuple(
    field for field in TARGET_FIELDS if field != RUN_DURATION_SOURCE_FIELD
)
REQUIRED_COLUMNS = (
    *METADATA_FIELDS,
    "duration_sec",
    "phase_rounds",
    *SOURCE_TARGET_FIELDS,
)

logger = get_logger(Path(__file__).name)
GPU_SM_OCCUPANCY_CONFLICT_THRESHOLD = 5.0
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
    target_names: tuple[str, ...] = TARGET_FIELDS
    source_row_count: int = 0
    excluded_before_gpu_quality_count: int = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract monitoring CSV rows into a prepared gnn_model dataset.",
    )
    parser.add_argument(
        "--csv_dirs",
        "--csv_dir",
        default="csv/v100,csv/a100",
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
    csv_dirs = args.csv_dirs.strip().split(",")
    build_dataset(
        csv_dirs=[Path(csv_dir) for csv_dir in csv_dirs],
        output_dir=Path(args.output_dir),
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    return 0


def build_dataset(
    *,
    csv_dirs: list[Path],
    output_dir: Path,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Path:
    csv_paths = []
    for csv_dir in csv_dirs:
        csv_paths.extend(sorted(csv_dir.glob("*.csv")))
    assert csv_paths, f"No csv files found in {csv_dirs}"
    csv_results = [process_csv(csv_path) for csv_path in csv_paths]
    target_names = csv_results[0].target_names
    if any(result.target_names != target_names for result in csv_results[1:]):
        raise ValueError(
            "CSV files use incompatible target schemas; prepare NVIDIA and "
            "BI-V150 datasets separately"
        )
    graphs = [graph for result in csv_results for graph in result.records]
    assert graphs
    sample_keys = [
        (graph.gpu_name, graph.model_name, graph.variant_name, graph.phase)
        for graph in graphs
    ]
    if len(sample_keys) != len(set(sample_keys)):
        raise ValueError("duplicate GPU/model/variant/phase samples across CSV files")
    total_gpu_quality_filtered_count = sum(
        result.gpu_quality_filtered_count for result in csv_results
    )
    total_high_memory_delta_count = sum(
        result.high_memory_delta_count for result in csv_results
    )
    source_row_count = sum(result.source_row_count for result in csv_results)
    excluded_before_gpu_quality_count = sum(
        result.excluded_before_gpu_quality_count for result in csv_results
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
        csv_dirs=csv_dirs,
        split_data=split_data,
        sample_count=len(graphs),
        total_record_count=len(graphs),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        target_names=target_names,
        gpu_names=tuple(sorted({graph.gpu_name for graph in graphs})),
        graph_capture_batch_sizes=tuple(
            sorted({graph.graph_capture_batch_size for graph in graphs})
        ),
        source_row_count=source_row_count,
        excluded_before_gpu_quality_count=excluded_before_gpu_quality_count,
        gpu_quality_filtered_count=total_gpu_quality_filtered_count,
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
    df = normalize_monitor_columns(df, csv_path)
    if df.is_empty():
        raise ValueError(f"monitor CSV is empty: {csv_path}")
    source_row_count = df.height
    if "gpu_node" not in df.columns:
        raise ValueError(f"monitor CSV missing gpu_node: {csv_path}")
    gpu_names = {normalize_gpu_name(value) for value in df["gpu_node"].to_list()}
    if len(gpu_names) != 1:
        raise ValueError(f"monitor CSV mixes GPU models: {csv_path}: {gpu_names}")
    gpu_name = next(iter(gpu_names))
    target_names = target_fields_for_gpu(gpu_name)
    required_columns = set(REQUIRED_COLUMNS) - set(SOURCE_TARGET_FIELDS)
    required_columns.update(
        field for field in target_names if field != RUN_DURATION_SOURCE_FIELD
    )
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"monitor CSV missing columns {sorted(missing_columns)}: {csv_path}"
        )
    df = df.filter(pl.col("phase_rounds") > 0).with_columns(
        (pl.col("duration_sec") / pl.col("phase_rounds")).alias(
            RUN_DURATION_SOURCE_FIELD
        ),
        pl.struct(["result_json", "variant_name", "phase"])
        .map_elements(
            lambda row: str(
                resolve_graph_path(
                    row["result_json"],
                    row["variant_name"],
                    row["phase"],
                    csv_path=csv_path,
                )
            ),
            return_dtype=pl.String,
        )
        .alias("variant_path"),
    )
    df = (
        df.drop_nans(subset=["batch_size", *target_names])
        .drop_nulls(subset=["batch_size", *target_names])
        .filter(
            pl.all_horizontal(
                [pl.col(field).is_finite() for field in target_names]
            )
        )
        .filter(
            pl.all_horizontal(
                [
                    pl.col("variant_name").str.len_chars() > 0,
                    (
                        ~pl.col("variant_name").str.to_lowercase().str.contains("_elu_")
                        if gpu_name != "bi-v150"
                        else pl.lit(True)
                    ),
                    pl.col("variant_path").map_elements(
                        lambda path: Path(path).is_file(),
                        return_dtype=pl.Boolean,
                    ),
                ]
            )
        )
    )
    assert df.height > 0, csv_path
    pre_gpu_quality_count = df.height
    excluded_before_gpu_quality_count = source_row_count - pre_gpu_quality_count
    high_memory_delta_count = df.filter(
        pl.col("memory_delta_gb_max") > MEMORY_DELTA_LOG_THRESHOLD_GB
    ).height
    df = df.filter(
        ~invalid_gpu_metric_expr(require_sm_occupancy=gpu_name != "bi-v150")
    )
    gpu_quality_filtered_count = pre_gpu_quality_count - df.height
    if df.is_empty():
        raise ValueError(f"no usable monitoring records in {csv_path}")
    assert df.unique(ID_FIELDS).height == df.height, csv_path
    records = [
        extract_feature_target(
            build_model_record_info(
                record,
                row_number=i,
                csv_path=csv_path,
                target_names=target_names,
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
        target_names=target_names,
        source_row_count=source_row_count,
        excluded_before_gpu_quality_count=excluded_before_gpu_quality_count,
    )


def normalize_monitor_columns(df: pl.DataFrame, csv_path: Path) -> pl.DataFrame:
    if "decode_output_length" in df.columns:
        return df
    if "phase" in df.columns and df.filter(pl.col("phase") == "decode").height > 0:
        raise ValueError(
            f"decode_output_length is required for decode rows: {csv_path}"
        )
    return df.with_columns(pl.lit(0, dtype=pl.Int64).alias("decode_output_length"))


def target_fields_for_gpu(gpu_name: str) -> tuple[str, ...]:
    return COREX_BI_V150_TARGET_FIELDS if gpu_name == "bi-v150" else TARGET_FIELDS


def invalid_gpu_metric_expr(*, require_sm_occupancy: bool = True) -> pl.Expr:
    gpu_mem = pl.col("gpu_mem_used_mb_max")
    gpu_util = pl.col("gpu_util_percent_max")
    invalid = (gpu_mem <= 0) | ((gpu_util <= 0) & (gpu_mem > 0))
    if require_sm_occupancy:
        gpu_sm_occupancy = pl.col("gpu_sm_occupancy_percent_max")
        invalid |= (gpu_util <= 0) & (
            gpu_sm_occupancy > GPU_SM_OCCUPANCY_CONFLICT_THRESHOLD
        )
        invalid |= (gpu_sm_occupancy <= 0) & (
            gpu_util >= GPU_UTIL_CONFLICT_THRESHOLD
        )
    return invalid


def build_model_record_info(
    record: Mapping[str, object],
    row_number: int,
    *,
    csv_path: Path,
    target_names: tuple[str, ...] = TARGET_FIELDS,
) -> ModelRecordInfo:
    phase = str(record["phase"]).strip()
    assert phase in {"training", "inference", "prefill", "decode"}
    batch_size = parse_int_value(record["batch_size"])
    assert batch_size > 0
    gpu_name = normalize_gpu_name(record["gpu_node"])
    target = tuple(parse_float_value(record[field]) for field in target_names)
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


def resolve_graph_path(
    result_json: object,
    variant_name: object,
    phase: str = "",
    *,
    csv_path: Path | None = None,
) -> Path:
    result_path = Path(str(result_json).strip())
    variant = str(variant_name).strip()
    suffix = f"_{phase.strip()}" if phase.strip() in {"prefill", "decode"} else ""
    file_name = f"{variant}{suffix}.pt2"
    if csv_path is None:
        return result_path.parent.parent / "fx_graphs" / file_name

    csv_dir = csv_path.resolve().parent
    candidates = []
    if not result_path.is_absolute():
        # New CSV format: result_json is relative to the CSV itself.
        candidates.append(csv_dir / result_path)
    # Old CSVs often contain an absolute path from the collection container,
    # or a path relative to the project root. Preserve the sibling layout after
    # moving the experiment directory to another machine.
    if csv_dir.name == result_path.parent.parent.name:
        candidates.append(csv_dir / "results" / result_path.name)
    candidates.append(result_path)
    for candidate in candidates:
        graph_path = candidate.parent.parent / "fx_graphs" / file_name
        if graph_path.is_file():
            return graph_path
    return candidates[0].parent.parent / "fx_graphs" / file_name


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
    from gnn_model.data.fx_graph import build_graph_data_from_fx

    assert info.model_path.exists()
    data = build_graph_data_from_fx(
        info.model_path,
        batch_size=info.batch_size,
        gpu_name=info.gpu_name,
        phase=info.phase,
        decode_output_length=parse_int_value(info.metadata["decode_output_length"]),
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
    group_keys = list(
        dict.fromkeys(
            (graph.gpu_name, graph.model_name, graph.variant_name)
            for graph in graphs
        )
    )
    train_count, val_count, _test_count = resolve_split_counts(
        len(group_keys),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    random.Random(seed).shuffle(group_keys)
    train_keys = set(group_keys[:train_count])
    val_keys = set(group_keys[train_count : train_count + val_count])
    test_keys = set(group_keys[train_count + val_count :])

    def group_key(graph: Data) -> tuple[str, str, str]:
        return graph.gpu_name, graph.model_name, graph.variant_name

    return (
        [graph for graph in graphs if group_key(graph) in train_keys],
        [graph for graph in graphs if group_key(graph) in val_keys],
        [graph for graph in graphs if group_key(graph) in test_keys],
    )


def build_manifest(
    *,
    csv_dirs: list[Path],
    split_data: dict[str, list[Data]],
    sample_count: int,
    total_record_count: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    target_names: tuple[str, ...] = TARGET_FIELDS,
    gpu_names: tuple[str, ...] = (),
    graph_capture_batch_sizes: tuple[int, ...] = (),
    source_row_count: int = 0,
    excluded_before_gpu_quality_count: int = 0,
    gpu_quality_filtered_count: int = 0,
) -> dict[str, Any]:
    return {
        "schema_version": "5.0.0",
        "feature_source": (
            "pytorch_export_inference_ir_static_metrics_shape_topology_v1"
        ),
        "csv_dirs": ",".join([str(Path(csv_dir).resolve()) for csv_dir in csv_dirs]),
        "id_fields": list(ID_FIELDS),
        "target_names": list(target_names),
        "split_unit": "gpu_model_variant",
        "gpu_names": list(gpu_names),
        "graph_capture_batch_sizes": list(graph_capture_batch_sizes),
        "quality_report": {
            "source_row_count": source_row_count,
            "excluded_before_gpu_quality_count": excluded_before_gpu_quality_count,
            "gpu_quality_filtered_count": gpu_quality_filtered_count,
            "retained_row_count": sample_count,
        },
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
