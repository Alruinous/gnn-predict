from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from PIL import Image

from experiment.workflow.plots import generate_workflow_plots


def test_complete_main_burst_groups_generate_four_pdf_png_pairs(
    tmp_path: Path,
) -> None:
    artifacts = generate_workflow_plots(_group_rows(), tmp_path / "figures")

    assert [path.name for path in artifacts] == [
        "qmsum_strategy_performance.pdf",
        "qmsum_strategy_performance.png",
        "qmsum_pipeline_bubble.pdf",
        "qmsum_pipeline_bubble.png",
        "mbpp_resource_tradeoff.pdf",
        "mbpp_resource_tradeoff.png",
        "mbpp_gpu_time_composition.pdf",
        "mbpp_gpu_time_composition.png",
    ]
    for path in artifacts:
        assert path.is_file()
        if path.suffix == ".pdf":
            assert path.read_bytes().startswith(b"%PDF")
            continue
        with Image.open(path) as image:
            assert image.mode in {"RGB", "RGBA"}
            assert image.width >= 1_200
            assert image.info["dpi"][0] == pytest.approx(300.0, abs=1.0)


def test_missing_metric_skips_only_the_dependent_figure(tmp_path: Path) -> None:
    rows = deepcopy(_group_rows())
    qmsum_cache = next(
        row
        for row in rows
        if row["scenario"] == "qmsum" and row["strategy"] == "wf-cache"
    )
    qmsum_metrics = qmsum_cache["metrics"]
    assert isinstance(qmsum_metrics, dict)
    qmsum_metrics = cast(dict[str, object], qmsum_metrics)
    qmsum_metrics.pop("session_latency_sec")

    artifacts = generate_workflow_plots(rows, tmp_path / "figures")
    names = {path.name for path in artifacts}

    assert "qmsum_strategy_performance.pdf" not in names
    assert "qmsum_strategy_performance.png" not in names
    assert "qmsum_pipeline_bubble.pdf" in names
    assert "mbpp_resource_tradeoff.pdf" in names
    assert "mbpp_gpu_time_composition.pdf" in names


def test_incomplete_group_and_component_do_not_create_fake_zero_plots(
    tmp_path: Path,
) -> None:
    rows = deepcopy(_group_rows())
    qmsum_fifo = next(
        row
        for row in rows
        if row["scenario"] == "qmsum" and row["strategy"] == "wf-fifo"
    )
    qmsum_fifo["trial_count"] = 4
    mbpp_cache_2 = next(
        row
        for row in rows
        if row["scenario"] == "mbpp"
        and row["strategy"] == "wf-cache"
        and row["gpu_count"] == 2
    )
    mbpp_metrics = mbpp_cache_2["metrics"]
    assert isinstance(mbpp_metrics, dict)
    mbpp_metrics = cast(dict[str, object], mbpp_metrics)
    mbpp_metrics.pop("idle_resident_gpu_seconds")

    artifacts = generate_workflow_plots(rows, tmp_path / "figures")
    names = {path.name for path in artifacts}

    assert names == {
        "mbpp_resource_tradeoff.pdf",
        "mbpp_resource_tradeoff.png",
    }


def _group_rows() -> list[dict[str, object]]:
    rows = [
        _group("qmsum", strategy, 2, scale)
        for strategy, scale in (
            ("lg-batch", 1.0),
            ("wf-fifo", 0.92),
            ("wf-cache", 0.76),
        )
    ]
    rows.extend(
        _group("mbpp", strategy, gpu_count, scale)
        for strategy, gpu_count, scale in (
            ("lg-batch", 3, 1.0),
            ("wf-cache", 1, 1.35),
            ("wf-cache", 2, 1.12),
            ("wf-cache", 3, 0.95),
        )
    )
    return rows


def _group(
    scenario: str,
    strategy: str,
    gpu_count: int,
    scale: float,
) -> dict[str, object]:
    return {
        "scenario": scenario,
        "strategy": strategy,
        "gpu_count": gpu_count,
        "workload": "burst",
        "max_num_seqs": 3,
        "queue_capacity": 16,
        "load_percent": None,
        "trial_count": 5,
        "metrics": {
            "makespan_sec": _interval(1_000.0 * scale),
            "session_latency_sec": {"p95": _interval(220.0 * scale)},
            "pipeline_bubble_ratio": _interval(0.2 * scale),
            "sessions_per_min": _interval(3.0 / scale),
            "resident_gpu_seconds": _interval(2_000.0 * scale),
            "loading_gpu_seconds": _interval(260.0 * scale),
            "active_gpu_seconds": _interval(1_100.0 * scale),
            "idle_resident_gpu_seconds": _interval(640.0 * scale),
        },
    }


def _interval(mean: float) -> dict[str, float]:
    return {
        "mean": mean,
        "std": mean * 0.03,
        "ci95_low": mean * 0.94,
        "ci95_high": mean * 1.06,
    }
