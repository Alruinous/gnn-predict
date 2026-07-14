"""Build workflow graph artifacts and serialized PyG cache entries."""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402
import yaml  # noqa: E402
from torch import nn  # noqa: E402

from common.log import get_logger  # noqa: E402
from gnn_model.data.fx_graph import build_graph_data_from_fx  # noqa: E402
from workflow.cache_config import (  # noqa: E402
    GraphCacheExportSpec,
    GraphCacheKey,
    WorkflowPhase,
    expand_graph_cache_specs,
    load_graph_cache_config,
)
from workflow.model import export_phase_cached_graph  # noqa: E402
from workflow.types import WorkflowModelFeatureKey  # noqa: E402

logger = get_logger("convert_cache")

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
GpuName = Literal["v100", "a100"]


def parse_dtype(name: str) -> torch.dtype:
    return _DTYPE_MAP[name]


def parse_phase_set(text: str | None) -> set[WorkflowPhase] | None:
    if text is None:
        return None
    phases: set[WorkflowPhase] = set()
    for raw in text.split(","):
        value = raw.strip().lower()
        if value not in ("prefill", "decode"):
            raise argparse.ArgumentTypeError(
                f"phase must be prefill or decode, got {value!r}"
            )
        phases.add(cast(WorkflowPhase, value))
    if not phases:
        raise argparse.ArgumentTypeError("phase must not be empty")
    return phases


def graph_digest(key: GraphCacheKey) -> str:
    return hashlib.sha256(key.digest_payload().encode()).hexdigest()[:16]


def load_model_from_config(model_path: Path, dtype: torch.dtype) -> nn.Module:
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config.use_cache = True
    model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    return model.cpu().eval()


def build_input_map(
    spec: GraphCacheExportSpec,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    if spec.key.phase == "prefill":
        input_ids = torch.zeros(
            (spec.key.batch_size, spec.key.sequence_length), dtype=torch.long
        )
        attention_mask = torch.ones(
            (spec.key.batch_size, spec.key.sequence_length), dtype=torch.long
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}, [
            "input_ids",
            "attention_mask",
        ]
    input_map: dict[str, torch.Tensor] = {
        "input_ids": torch.zeros((spec.key.batch_size, 1), dtype=torch.long),
        "attention_mask": torch.ones(
            (
                spec.key.batch_size,
                spec.key.sequence_length + spec.key.decode_output_length,
            ),
            dtype=torch.long,
        ),
    }
    return input_map, list(input_map.keys())


def command_export(args: argparse.Namespace) -> int:
    config = load_graph_cache_config(Path(args.cache_yaml))
    groups = set(args.group) if args.group else None
    phases = parse_phase_set(args.phase)
    specs = expand_graph_cache_specs(
        config,
        selected_group_names=groups,
        selected_phases=phases,
    )
    if not specs:
        logger.error("no graph cache specs selected")
        return 1

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    specs_by_model: dict[tuple[Path, str], list[GraphCacheExportSpec]] = {}
    for spec in specs:
        specs_by_model.setdefault((spec.model_path, spec.dtype), []).append(spec)

    entries: list[dict[str, object]] = []
    for (model_path, dtype_name), model_specs in specs_by_model.items():
        logger.info(f"loading {model_path} dtype={dtype_name}")
        model = load_model_from_config(model_path, parse_dtype(dtype_name))
        for spec in model_specs:
            digest = graph_digest(spec.key)
            graph_path = save_dir / f"{digest}.pt2"
            if graph_path.exists() and not args.overwrite:
                logger.info(f"skip existing {spec.key} -> {graph_path.name}")
                entries.append(_build_entry(digest, spec, graph_path))
                continue
            input_map, input_names = build_input_map(spec)
            runtime_input_names = export_phase_cached_graph(
                model=model,
                graph_path=graph_path,
                input_map=input_map,
                input_names=input_names,
                phase=spec.key.phase,
                sequence_length=spec.key.sequence_length,
                decode_output_length=spec.key.decode_output_length,
            )
            logger.info(
                f"exported {spec.key} -> {graph_path.name} "
                f"runtime_inputs={len(runtime_input_names)}"
            )
            entries.append(_build_entry(digest, spec, graph_path, runtime_input_names))
        del model

    manifest_path = write_manifest(save_dir, "graph", entries)
    logger.info(f"export finished: {len(entries)} graph -> {manifest_path}")
    return 0


def _build_entry(
    digest: str,
    spec: GraphCacheExportSpec,
    graph_path: Path,
    runtime_input_names: list[str] | None = None,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "digest": digest,
        "graph_path": str(graph_path.name),
        "cache_group": spec.group_name,
        "model_path": str(spec.model_path),
        "dtype": spec.dtype,
        "key": spec.key.model_dump(mode="python"),
    }
    if runtime_input_names is not None:
        entry["runtime_input_names"] = runtime_input_names
    return entry


def read_graph_manifest(graph_dir: Path) -> list[dict[str, object]]:
    manifest = graph_dir / "manifest.yaml"
    if not manifest.is_file():
        logger.error(f"graph manifest not found: {manifest}")
        raise SystemExit(1)
    with manifest.open() as f:
        raw = yaml.safe_load(f)
    entries = raw.get("entries", []) if isinstance(raw, dict) else []
    return [e for e in entries if isinstance(e, dict)]


def to_workflow_key(
    graph_key_data: Mapping[str, object],
    gpu_name: GpuName,
) -> WorkflowModelFeatureKey:
    return WorkflowModelFeatureKey(
        model_name=cast(str, graph_key_data["model_name"]),
        phase=cast(Literal["prefill", "decode"], graph_key_data["phase"]),
        gpu_name=gpu_name,
        batch_size=cast(int, graph_key_data["batch_size"]),
        sequence_length=cast(int, graph_key_data["sequence_length"]),
        decode_output_length=cast(int, graph_key_data["decode_output_length"]),
    )


def command_extract(args: argparse.Namespace) -> int:
    graph_dir = Path(args.graph_dir)
    geo_dir = Path(args.geo_dir)
    if not graph_dir.is_dir():
        logger.error(f"graph directory not found: {graph_dir}")
        return 1
    geo_dir.mkdir(parents=True, exist_ok=True)

    graph_entries = read_graph_manifest(graph_dir)
    logger.info(f"loaded {len(graph_entries)} entries from graph manifest")

    entries: list[dict[str, object]] = []
    built = 0
    for entry in graph_entries:
        digest = cast(str, entry["digest"])
        graph_rel = cast(str, entry["graph_path"])
        graph_file = graph_dir / Path(graph_rel).name
        if not graph_file.is_file():
            logger.warning(f"skip {digest}: graph file missing {graph_rel}")
            continue
        key_data = entry.get("key")
        if not isinstance(key_data, dict):
            logger.warning(
                f"skip {digest}: manifest entry missing `key` field "
                "(re-run `export` to refresh manifest)"
            )
            continue
        key = to_workflow_key(cast(Mapping[str, object], key_data), args.gpu_name)
        geo_path = geo_dir / f"{key.stable_digest}.pkl"
        if geo_path.exists() and not args.overwrite:
            logger.info(f"skip existing {key} -> {geo_path.name}")
            entries.append(_build_geo_entry(key, graph_file, geo_path))
            continue
        data = build_graph_data_from_fx(
            graph_file,
            phase=key.phase,
            gpu_name=args.gpu_name,
            batch_size=key.batch_size,
            decode_output_length=key.decode_output_length,
        )
        with geo_path.open("wb") as f:
            pickle.dump((key, data), f)
        entries.append(_build_geo_entry(key, graph_file, geo_path))
        built += 1
        logger.info(f"extracted {key} -> {geo_path.name}")

    manifest_path = write_manifest(geo_dir, "geo", entries)
    logger.info(f"extract finished: {built} features -> {manifest_path}")
    return 0 if entries else 1


def _build_geo_entry(
    key: WorkflowModelFeatureKey,
    graph_file: Path,
    geo_path: Path,
) -> dict[str, object]:
    return {
        "geo_path": geo_path.name,
        "graph_path": graph_file.name,
        "key": key.model_dump(mode="python"),
    }


def write_manifest(save_dir: Path, kind: str, entries: list[dict[str, object]]) -> Path:
    manifest = save_dir / "manifest.yaml"
    payload = {"version": 2, "kind": kind, "entries": entries}
    with manifest.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build WorkflowController graph-feature cache. Two stages: "
            "`export` HF model -> cache/graph; `extract` cache/graph -> cache/geo."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser(
        "export",
        help="read a cache YAML and export architecture-only graph for each spec",
    )
    p_export.add_argument(
        "--cache-yaml", required=True, help="example: config/workflow/cache.yaml"
    )
    p_export.add_argument("--save-dir", required=True, help="cache/graph output dir")
    p_export.add_argument("--group", action="append", help="cache group name to export")
    p_export.add_argument("--phase", help="comma-separated phase subset")
    p_export.add_argument("--overwrite", action="store_true")
    p_export.set_defaults(func=command_export)

    p_extract = sub.add_parser(
        "extract",
        help="rebuild Data pickles from existing cache/graph directory",
    )
    p_extract.add_argument("--graph-dir", required=True, help="cache/graph directory")
    p_extract.add_argument(
        "--geo-dir", required=True, help="cache/geo output directory"
    )
    p_extract.add_argument(
        "--gpu-name",
        default="a100",
        choices=["v100", "a100"],
    )
    p_extract.add_argument("--overwrite", action="store_true")
    p_extract.set_defaults(func=command_extract)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
