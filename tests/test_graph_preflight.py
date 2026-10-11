from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import gnn_model.data.preflight as preflight


def test_preflight_reports_missing_csv_directory(tmp_path: Path) -> None:
    report = preflight.check_graphs([tmp_path / "missing"])
    assert report["unique_graph_count"] == 0
    assert report["issue_count"] == 1


def test_preflight_checks_later_graphs_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "monitor.csv"
    csv_path.write_text(
        "result_json,variant_name,phase,gpu_node,batch_size\n"
        "results/run.json,broken,training,bi-v150,16\n"
        "results/run.json,valid,inference,bi-v150,1\n",
        encoding="utf-8",
    )
    calls: list[str] = []

    def load(path: Path) -> tuple[object, SimpleNamespace]:
        calls.append(path.stem)
        if path.stem == "broken":
            raise ValueError("bad graph")
        return object(), SimpleNamespace(runtime_input_names=["inputs"])

    monkeypatch.setattr(preflight, "load_graph_artifact", load)
    monkeypatch.setattr(preflight, "find_unsupported_tensor_operators", lambda _: [])
    monkeypatch.setattr(
        preflight, "build_graph_data_from_exported_program", lambda *args, **kwargs: None
    )

    report = preflight.check_graphs([tmp_path])

    assert calls == ["broken", "valid"]
    assert report["unique_graph_count"] == 2
    assert report["issue_count"] == 1
