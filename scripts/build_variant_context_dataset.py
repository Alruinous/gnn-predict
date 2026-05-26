from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gnn_model.data.constants import GRAPH_FEATURE_DIM, GRAPH_FEATURE_NAMES
from gnn_model.data.variant_context import (
    VARIANT_CONTEXT_FEATURE_NAMES,
    build_variant_context_feature_vector,
)

VARIANT_CONTEXT_DIM = len(VARIANT_CONTEXT_FEATURE_NAMES)
VARIANT_CONTEXT_START = GRAPH_FEATURE_DIM - VARIANT_CONTEXT_DIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append variant-context graph features to extracted split data.",
    )
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_counts: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        graphs = torch.load(source_dir / f"{split_name}.pt", weights_only=False)
        for graph in graphs:
            base_features = graph.graph_features.float()
            if base_features.size(1) > VARIANT_CONTEXT_START:
                base_features = base_features[:, :VARIANT_CONTEXT_START]
            pad_width = VARIANT_CONTEXT_START - base_features.size(1)
            if pad_width < 0:
                raise ValueError("source graph features overlap variant context")
            middle = torch.zeros((1, pad_width), dtype=base_features.dtype)
            context = torch.tensor(
                build_variant_context_feature_vector(
                    model_name=str(graph.base_model_name),
                    variant_name=str(graph.variant_name),
                ),
                dtype=base_features.dtype,
            ).unsqueeze(0)
            graph.graph_features = torch.cat([base_features, middle, context], dim=1)
        torch.save(graphs, output_dir / f"{split_name}.pt")
        split_counts[split_name] = len(graphs)

    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = "3.1.0"
    manifest["feature_source"] = "peak_live_variant_context_features"
    manifest["graph_feature_names"] = list(GRAPH_FEATURE_NAMES)
    manifest["variant_context_feature_names"] = list(VARIANT_CONTEXT_FEATURE_NAMES)
    manifest["split_counts"] = split_counts
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
