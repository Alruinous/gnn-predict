from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common.graph_artifact import load_graph_artifact
from gnn_model.data.extract import resolve_graph_path
from gnn_model.data.fx_graph import (
    build_graph_data_from_exported_program,
    find_unsupported_tensor_operators,
)


def check_graphs(csv_dirs: list[Path]) -> dict[str, object]:
    """Check graph compatibility without changing graphs, CSVs or split datasets."""
    issues: list[dict[str, object]] = []
    cases: dict[Path, set[tuple[str, int, str, int]]] = {}
    csv_paths: list[str] = []
    for directory in csv_dirs:
        paths = sorted(directory.glob("*.csv"))
        if not paths:
            issues.append({"directory": str(directory), "error": "no CSV files"})
        for csv_path in paths:
            csv_paths.append(str(csv_path))
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if not rows:
                issues.append({"csv": str(csv_path), "error": "empty CSV"})
            for row_number, row in enumerate(rows, start=2):
                try:
                    path = resolve_graph_path(
                        row["result_json"],
                        row["variant_name"],
                        row["phase"],
                        csv_path=csv_path,
                    ).resolve()
                    case = (
                        row["gpu_node"], int(row["batch_size"]), row["phase"],
                        int(row.get("decode_output_length") or 0),
                    )
                    cases.setdefault(path, set()).add(case)
                except (KeyError, TypeError, ValueError) as exc:
                    issues.append({
                        "csv": str(csv_path), "row": row_number,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
    for index, (path, graph_cases) in enumerate(cases.items(), start=1):
        print(f"Checking graph {index}/{len(cases)}: {path}", flush=True)
        try:
            program, metadata = load_graph_artifact(path)
            unsupported = find_unsupported_tensor_operators(program)
            if unsupported:
                issues.append({"graph": str(path), "unsupported_operators": unsupported})
                continue
            for gpu_name, batch_size, phase, decode_length in sorted(graph_cases):
                try:
                    build_graph_data_from_exported_program(
                        program, runtime_input_names=metadata.runtime_input_names,
                        gpu_name=gpu_name, batch_size=batch_size, phase=phase,
                        decode_output_length=decode_length, graph_path=path,
                    )
                except Exception as exc:
                    issues.append({
                        "graph": str(path), "gpu_name": gpu_name,
                        "batch_size": batch_size, "phase": phase,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
            del program
        except Exception as exc:
            issues.append({"graph": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return {
        "check_scope": "graph_loading_operator_categories_and_feature_construction",
        "csv_files": csv_paths, "unique_graph_count": len(cases),
        "issue_count": len(issues), "issues": issues,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check all CSV-referenced graphs before extraction."
    )
    parser.add_argument("--csv_dirs", required=True, help="Comma-separated CSV directories.")
    parser.add_argument("--report", default="data/corex_bi_v150/graph_preflight.json")
    args = parser.parse_args(argv)
    report = check_graphs([Path(item.strip()) for item in args.csv_dirs.split(",")])
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Checked {report['unique_graph_count']} graphs; issues: {report['issue_count']}")
    print(f"Report: {report_path}")
    return 1 if report["issue_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
