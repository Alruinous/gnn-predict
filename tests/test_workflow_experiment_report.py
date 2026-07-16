from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import pytest

from experiment.workflow.artifacts import (
    canonical_json,
    complete_trial,
    file_sha256,
    read_json,
    write_json_exclusive,
)
from experiment.workflow.config import EXPERIMENT_ID, TrialSpec
from experiment.workflow.plan import build_trial_matrix
from experiment.workflow.report import (
    EXPECTED_CAPACITY_KEYS,
    PREPARATION_EVIDENCE_SPECS,
    ReportInputs,
    build_report_summary,
    load_report_inputs,
    paired_trial_comparisons,
    publish_report,
    render_report_markdown,
)

Repetition = Literal[1, 2, 3, 4, 5]
REPETITIONS: tuple[Repetition, ...] = (1, 2, 3, 4, 5)


def test_load_report_inputs_reads_quality_capacity_and_failures(
    tmp_path: Path,
) -> None:
    trial = TrialSpec(
        scenario="qmsum",
        strategy="wf-cache",
        gpu_count=2,
        workload="burst",
        repetition=1,
    )
    trial_row = _trial_row(trial, makespan=10.0)
    _write_jsonl(tmp_path / "analysis" / "trial_metrics.jsonl", [trial_row])
    _write_jsonl(tmp_path / "analysis" / "group_metrics.jsonl", [])
    write_json_exclusive(
        tmp_path / "experiment_manifest.json",
        {"version": 1, "experiment_id": EXPERIMENT_ID, "config": {}},
    )
    trial_dir = tmp_path / "trials" / trial.trial_id
    quality_path = trial_dir / "quality_summary.json"
    write_json_exclusive(
        quality_path,
        {
            "scenario": "qmsum",
            "sample_count": 24,
            "rouge1_mean": 0.2,
            "rouge2_mean": 0.1,
            "rouge_l_mean": 0.15,
            "empty_output_count": 0,
        },
    )
    complete_trial(
        trial_dir,
        trial.trial_id,
        ("quality_summary.json",),
        completed_at=1.0,
    )
    write_json_exclusive(
        tmp_path / "calibration" / "qmsum__wf-cache__g2" / "capacity.json",
        _capacity("qmsum__wf-cache__g2", "qmsum", 2),
    )
    write_json_exclusive(
        tmp_path / "failed" / "trial" / "attempt" / "failure.json",
        {"scenario": "qmsum", "repetition": 1, "archive_reason": "interrupted"},
    )

    inputs = load_report_inputs(tmp_path)

    assert inputs.experiment_id == EXPERIMENT_ID
    assert inputs.quality_summaries[0]["metrics"] == {
        "sample_count": 24,
        "rouge1_mean": 0.2,
        "rouge2_mean": 0.1,
        "rouge_l_mean": 0.15,
        "empty_output_count": 0,
    }
    assert inputs.capacities[0]["capacity_sessions_per_sec"] == 1.0
    assert str(inputs.failures[0]["artifact_path"]).endswith("failure.json")
    trial_source = _mapping(inputs.source_artifacts["trial_metrics"])
    group_source = _mapping(inputs.source_artifacts["group_metrics"])
    assert trial_source == {
        "path": "analysis/trial_metrics.jsonl",
        "sha256": file_sha256(tmp_path / "analysis" / "trial_metrics.jsonl"),
    }
    assert group_source == {
        "path": "analysis/group_metrics.jsonl",
        "sha256": file_sha256(tmp_path / "analysis" / "group_metrics.jsonl"),
    }
    assert {row["status"] for row in inputs.preparation_evidence} == {"missing"}

    quality_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="completed trial artifact is invalid"):
        load_report_inputs(tmp_path)


def test_load_report_inputs_indexes_verified_preparation_evidence(
    tmp_path: Path,
) -> None:
    write_json_exclusive(
        tmp_path / "experiment_manifest.json",
        {"version": 1, "experiment_id": EXPERIMENT_ID, "config": {}},
    )
    _write_preparation_evidence(tmp_path)

    inputs = load_report_inputs(tmp_path)

    assert len(inputs.preparation_evidence) == len(PREPARATION_EVIDENCE_SPECS)
    assert {row["status"] for row in inputs.preparation_evidence} == {"verified"}
    capacity_reuse = next(
        row for row in inputs.preparation_evidence if row["name"] == "capacity_reuse"
    )
    assert "parent-60" in str(capacity_reuse["provenance"])
    assert inputs.preparation_environment["git_commit"] == "commit"


def test_paired_comparisons_use_repetition_level_ratio_and_delta() -> None:
    rows = []
    for repetition in REPETITIONS:
        reference = TrialSpec(
            scenario="qmsum",
            strategy="lg-batch",
            gpu_count=2,
            workload="burst",
            repetition=repetition,
        )
        candidate = TrialSpec(
            scenario="qmsum",
            strategy="wf-cache",
            gpu_count=2,
            workload="burst",
            repetition=repetition,
        )
        rows.extend(
            (
                _trial_row(reference, makespan=float(8 + repetition)),
                _trial_row(candidate, makespan=float(2 + repetition)),
            )
        )

    comparisons = paired_trial_comparisons(rows)

    assert len(comparisons) == 1
    makespan = _mapping(comparisons[0]["metrics"])["makespan_sec"]
    effect = _mapping(makespan)
    expected_speedup = sum(
        (8 + repetition) / (2 + repetition) for repetition in range(1, 6)
    ) / 5
    assert _mapping(effect["inverse_ratio"])["mean"] == pytest.approx(
        expected_speedup
    )
    assert _mapping(effect["inverse_ratio"])["mean"] != pytest.approx(11 / 5)
    assert _mapping(effect["relative_reduction"])["mean"] == pytest.approx(
        sum(6 / (8 + repetition) for repetition in range(1, 6)) / 5
    )
    assert _mapping(effect["delta"])["mean"] == pytest.approx(-6.0)


def test_paired_comparisons_reject_arrival_trace_drift() -> None:
    rows = []
    for repetition in REPETITIONS:
        reference = TrialSpec(
            scenario="qmsum",
            strategy="lg-batch",
            gpu_count=2,
            workload="burst",
            repetition=repetition,
        )
        candidate = TrialSpec(
            scenario="qmsum",
            strategy="wf-cache",
            gpu_count=2,
            workload="burst",
            repetition=repetition,
        )
        rows.append(_trial_row(reference, makespan=10.0))
        candidate_row = _trial_row(candidate, makespan=5.0)
        if repetition == 3:
            candidate_row["arrival_trace_sha256"] = "drifted"
        rows.append(candidate_row)

    with pytest.raises(ValueError, match="different arrival traces"):
        paired_trial_comparisons(rows)


def test_paired_comparisons_reject_workflow_prediction_cache_drift() -> None:
    rows = []
    for repetition in REPETITIONS:
        reference = TrialSpec(
            scenario="qmsum",
            strategy="wf-fifo",
            gpu_count=2,
            workload="burst",
            repetition=repetition,
        )
        candidate = TrialSpec(
            scenario="qmsum",
            strategy="wf-cache",
            gpu_count=2,
            workload="burst",
            repetition=repetition,
        )
        rows.append(_trial_row(reference, makespan=10.0))
        candidate_row = _trial_row(candidate, makespan=5.0)
        if repetition == 3:
            alignment = cast(dict[str, object], candidate_row["alignment"])
            alignment["prediction_cache_sha256"] = "drifted"
        rows.append(candidate_row)

    with pytest.raises(ValueError, match="prediction_cache_sha256"):
        paired_trial_comparisons(rows)


def test_report_summary_marks_complete_matrix_formal_and_partial() -> None:
    inputs = _formal_inputs()

    formal = build_report_summary(inputs)
    partial = build_report_summary(
        ReportInputs(
            output_root=inputs.output_root,
            experiment_id=inputs.experiment_id,
            trial_metrics=inputs.trial_metrics[:-1],
            group_metrics=inputs.group_metrics,
            quality_summaries=inputs.quality_summaries[:-1],
            capacities=inputs.capacities,
            failures=inputs.failures,
        )
    )

    assert formal["status"] == "formal"
    assert _mapping(formal["completeness"])["formal"] is True
    comparisons = formal["comparisons"]
    quality_groups = formal["quality_groups"]
    assert isinstance(comparisons, list) and comparisons
    assert isinstance(quality_groups, list) and len(quality_groups) == 15
    canonical_json(formal)
    assert partial["status"] == "partial"
    completeness = _mapping(partial["completeness"])
    assert _mapping(completeness["trials"])["actual"] == 74


def test_markdown_keeps_partial_results_non_conclusive() -> None:
    inputs = _formal_inputs()
    formal_summary = build_report_summary(inputs)
    partial_summary = dict(formal_summary)
    partial_summary["status"] = "partial"

    formal = render_report_markdown(formal_summary)
    partial = render_report_markdown(partial_summary)

    assert "## 核心结果" in formal
    assert "75 trials" in formal
    assert "固定 24 个样本" in formal
    assert "Student-t 区间(df=4)" in formal
    assert "session p95 先在每个 trial" in formal
    assert "不表示对完整数据集的抽样不确定性" in formal
    assert "## 主要绝对指标" in formal
    assert "### 性能" in formal
    assert "steady-state 吞吐" in formal
    assert "### 资源效率" in formal
    assert "sessions/min/GPU" in formal
    assert "idle-subtracted energy/completion" in formal
    expected_absolute_rows = (
        "| QMSum | LG-Batch | 2 |",
        "| QMSum | WF-FIFO | 2 |",
        "| QMSum | WF-Cache | 2 |",
        "| MBPP | WF-Cache | 1 |",
        "| MBPP | WF-FIFO | 2 |",
        "| MBPP | WF-History | 2 |",
        "| MBPP | WF-Cache | 2 |",
        "| MBPP | LG-Batch | 3 |",
        "| MBPP | WF-Cache | 3 |",
    )
    assert all(formal.count(row) == 2 for row in expected_absolute_rows)

    assert "## 主配对效应" in formal
    assert "makespan speedup" in formal
    assert "resident GPU-s reduction" in formal
    assert "candidate_over_reference" not in formal
    assert "## 核心机制" in formal
    assert "mean fan-in wait" in formal
    assert "wasted prefetches" in formal
    assert "ROUGE-1" in formal
    assert "initial pass@1" in formal
    assert "0.2000 [0.2000, 0.2000]" in formal
    assert "33.333% [33.333, 33.333]" in formal

    assert "## QMSum batching 与队列消融" in formal
    assert "默认配置 `(seq3,q16)`" in formal
    assert "## 开放到达负载扫描" in formal
    assert "0.7500000000" in formal
    assert "1.2500000000" in formal
    assert "## MBPP 2-GPU 策略对比" in formal
    assert "## 准备与复现证据索引" in formal
    assert "| preparation manifest |" in formal
    assert "| preparation evidence | 11 | 11 | complete |" in formal
    assert "| QMSum | WF-History | 2 |" not in formal
    assert "阶段性结果" in partial
    assert "## 主要绝对指标" in partial
    assert "## 主配对效应" in partial
    assert "## 质量" in partial
    assert partial.index("## 主要绝对指标") < partial.index("## 实验覆盖")
    assert "aligned groups" not in partial


def test_markdown_keeps_complete_skeleton_without_reportable_groups() -> None:
    inputs = _formal_inputs()
    summary = build_report_summary(
        ReportInputs(
            output_root=inputs.output_root,
            experiment_id=inputs.experiment_id,
            trial_metrics=inputs.trial_metrics[:1],
            group_metrics=(),
            quality_summaries=inputs.quality_summaries[:1],
            capacities=(),
            failures=(),
        )
    )

    markdown = render_report_markdown(summary)

    assert "当前尚无满足 n=5 且冻结输入 hash 对齐的实验组" in markdown
    assert "## 主要绝对指标" in markdown
    assert "## 主配对效应" in markdown
    assert "## 质量守护指标" in markdown
    assert "## 核心机制" in markdown
    assert "## QMSum batching 与队列消融" in markdown
    assert "## 开放到达负载扫描" in markdown
    assert "## MBPP 2-GPU 策略对比" in markdown
    assert "## 核心图" in markdown
    assert "## 分析与结论记录" in markdown
    assert "## 准备与复现证据索引" in markdown
    assert "待相关实验单元的 5 次 repetition" in markdown
    completeness = _mapping(summary["completeness"])
    reportable = _mapping(completeness["reportable_aligned_groups"])
    assert reportable == {"actual": 0, "ids": []}


def test_markdown_does_not_treat_single_main_group_as_ablation() -> None:
    inputs = _formal_inputs()
    main_group = next(
        row
        for row in inputs.group_metrics
        if row.get("scenario") == "qmsum"
        and row.get("strategy") == "wf-cache"
        and row.get("workload") == "burst"
        and row.get("max_num_seqs") == 3
        and row.get("queue_capacity") == 16
    )
    summary = build_report_summary(
        ReportInputs(
            output_root=inputs.output_root,
            experiment_id=inputs.experiment_id,
            trial_metrics=(),
            group_metrics=(main_group,),
            quality_summaries=(),
            capacities=(),
            failures=(),
        )
    )

    markdown = render_report_markdown(summary)

    assert "## 主要绝对指标" in markdown
    assert "## 核心机制" in markdown
    assert "## QMSum batching 与队列消融" in markdown
    assert "默认配置 `(seq3,q16)`" not in markdown
    assert "## 开放到达负载扫描" in markdown


def test_markdown_explains_stacked_component_intervals() -> None:
    summary = build_report_summary(_formal_inputs())
    summary["figures"] = [
        {
            "name": "mbpp_gpu_time_composition",
            "png": "composition.png",
            "pdf": "composition.pdf",
        }
    ]

    markdown = render_report_markdown(summary)

    assert "### MBPP 突发负载下的 GPU 时间构成" in markdown
    assert "误差条是各分量自身的 Student-t 95% CI" in markdown
    assert "不表示累计高度的置信区间" in markdown


def test_publish_report_overwrites_atomically_and_links_figure_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "output"
    _write_jsonl(output_root / "analysis" / "trial_metrics.jsonl", [])
    _write_jsonl(output_root / "analysis" / "group_metrics.jsonl", [])
    write_json_exclusive(
        output_root / "experiment_manifest.json",
        {"version": 1, "experiment_id": EXPERIMENT_ID, "config": {}},
    )
    document = tmp_path / "docs" / "results.md"
    document.parent.mkdir()
    document.write_text("stale", encoding="utf-8")

    import experiment.workflow.plots as plots_module

    def fake_plots(
        rows: list[Mapping[str, object]], output_dir: Path
    ) -> tuple[Path, ...]:
        assert rows == []
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf = output_dir / "qmsum_strategy_performance.pdf"
        png = output_dir / "qmsum_strategy_performance.png"
        pdf.write_bytes(b"pdf")
        png.write_bytes(b"png")
        return pdf, png

    monkeypatch.setattr(plots_module, "generate_workflow_plots", fake_plots)

    first = publish_report(output_root, document)
    first_summary = first.summary_path.read_text(encoding="utf-8")
    first_document = document.read_text(encoding="utf-8")
    second = publish_report(output_root, document)

    assert second.summary_path.read_text(encoding="utf-8") == first_summary
    assert document.read_text(encoding="utf-8") == first_document
    summary = _mapping(read_json(second.summary_path))
    figures = summary["figures"]
    assert figures == [
        {
            "name": "qmsum_strategy_performance",
            "png": "../output/analysis/figures/qmsum_strategy_performance.png",
            "pdf": "../output/analysis/figures/qmsum_strategy_performance.pdf",
        }
    ]
    assert "## 核心图" in first_document
    assert "![QMSum 突发负载下的端到端性能](../output/analysis/figures/" in first_document
    assert "固定的 24 个 sessions" in first_document
    assert "Student-t 95% CI" in first_document

    assert "[PDF 版本](../output/analysis/figures/" in first_document
    assert not [path for path in tmp_path.rglob("*") if path.name.endswith(".tmp")]


def _formal_inputs() -> ReportInputs:
    trials = build_trial_matrix()
    trial_rows = tuple(
        _trial_row(trial, makespan=10.0 + trial.repetition) for trial in trials
    )
    quality = tuple(_quality_row(trial) for trial in trials)
    groups: dict[str, dict[str, object]] = {}
    for row in trial_rows:
        trial_id = str(row["trial_id"])
        group_id = trial_id.rsplit("__rep", maxsplit=1)[0]
        groups.setdefault(
            group_id,
            {
                "group_id": group_id,
                **{
                    key: row[key]
                    for key in (
                        "experiment_id",
                        "scenario",
                        "strategy",
                        "gpu_count",
                        "workload",
                        "max_num_seqs",
                        "queue_capacity",
                        "load_percent",
                        "alignment_fingerprint",
                    )
                },
                "trial_count": 5,
                "trial_ids": [],
                "metrics": {
                    "makespan_sec": _interval(11.0),
                    "sessions_per_min": _interval(300.0),
                    "steady_state_sessions_per_min": _interval(320.0),
                    "sessions_per_min_per_gpu": _interval(150.0),
                    "session_latency_sec": {"p95": _interval(12.0)},
                    "active_gpu_seconds": _interval(18.0),
                    "resident_gpu_seconds": _interval(20.0),
                    "gpu_seconds_per_completion": _interval(0.3),
                    "idle_subtracted_energy_per_completion_joules": _interval(50.0),
                    "pipeline_bubble_ratio": _interval(0.25),
                    "fanin_wait_sec": {"mean": _interval(0.4)},
                    "backpressure_duration_sec": _interval(1.5),
                    "vllm_queue_time_sec": {"mean": _interval(0.2)},
                    "loading_gpu_seconds": _interval(4.0),
                    "model_load_count": _interval(3.0),
                    "model_reuse_count": _interval(57.0),
                    "prefetch_count": _interval(2.0),
                    "model_eviction_count": _interval(2.0),
                    "wasted_prefetch_count": _interval(0.0),
                },

            },
        )
        trial_ids = groups[group_id]["trial_ids"]
        assert isinstance(trial_ids, list)
        trial_ids = cast(list[str], trial_ids)
        trial_ids.append(trial_id)
    capacities = tuple(
        {
            "artifact_path": f"calibration/{key}/capacity.json",
            **_capacity(key, *(_capacity_dimensions(key))),
        }
        for key in sorted(EXPECTED_CAPACITY_KEYS)
    )
    return ReportInputs(
        output_root=Path("/experiment"),
        experiment_id=EXPERIMENT_ID,
        trial_metrics=trial_rows,
        group_metrics=tuple(groups.values()),
        quality_summaries=quality,
        capacities=capacities,
        failures=(),
        preparation_evidence=tuple(
            {
                "name": name,
                "label": label,
                "artifact_path": path,
                "sha256": "0" * 64,
                "status": "verified",
                "provenance": "test fixture",
            }
            for name, label, path in PREPARATION_EVIDENCE_SPECS
        ),
        preparation_environment={
            "hostname": "host",
            "python_version": "3.12",
            "platform": "linux",
            "git_commit": "commit",
            "git_dirty": False,
            "git_diff_sha256": "0" * 64,
            "package_versions": {"pytest": "9"},
        },
    )


def _trial_row(trial: TrialSpec, *, makespan: float) -> dict[str, object]:
    shared = (
        f"{trial.scenario}-{trial.gpu_count}-{trial.workload}-"
        f"{trial.load_percent}-{trial.max_num_seqs}-{trial.queue_capacity}-"
        f"{trial.repetition}"
    )
    alignment = {
        "sample_manifest_sha256": "samples",
        "prediction_cache_sha256": (
            None if trial.strategy == "lg-batch" else "cache"
        ),
        "capacity_calibration_sha256": "capacity",
        "absolute_arrival_rate": 1.0 if trial.load_percent is not None else None,
        "workflow_config_sha256": "workflow",
        "serving_environment_sha256": "serving",
        "telemetry_config_sha256": "telemetry",
        "environment_sha256": "environment",
    }
    return {
        "trial_id": trial.trial_id,
        "experiment_id": EXPERIMENT_ID,
        **trial.model_dump(mode="json"),
        "alignment_fingerprint": f"aligned-{trial.strategy}-{trial.scenario}",
        "alignment": alignment,
        "arrival_trace_sha256": shared,
        "metrics": {
            "makespan_sec": makespan,
            "sessions_per_min": 3600.0 / makespan,
            "sessions_per_min_per_gpu": 3600.0 / makespan / trial.gpu_count,
            "session_latency_sec": {"p95": makespan},
            "active_gpu_seconds": makespan,
            "resident_gpu_seconds": makespan * trial.gpu_count,
            "gpu_seconds_per_completion": makespan / 60.0,
            "idle_subtracted_energy_per_completion_joules": makespan,
            "pipeline_bubble_ratio": 0.25,
        },
    }


def _quality_row(trial: TrialSpec) -> dict[str, object]:
    if trial.scenario == "qmsum":
        metrics = {
            "sample_count": 24,
            "rouge1_mean": 0.2,
            "rouge2_mean": 0.1,
            "rouge_l_mean": 0.15,
            "empty_output_count": 0,
        }
    else:
        metrics = {
            "sample_count": 24,
            "initial_pass_count": 8,
            "final_pass_count": 12,
            "initial_pass_rate": 1 / 3,
            "final_pass_rate": 0.5,
            "repaired_count": 4,
            "regressed_count": 0,
            "initial_failure_types": {"assertion_error": 16},
            "final_failure_types": {"assertion_error": 12},
        }
    return {"trial_id": trial.trial_id, "scenario": trial.scenario, "metrics": metrics}


def _capacity(
    key: str, scenario: str, gpu_count: int
) -> dict[str, object]:
    return {
        "version": 1,
        "calibration_key": key,
        "scenario": scenario,
        "gpu_count": gpu_count,
        "steady_state_start_fraction": 0.2,
        "steady_state_end_fraction": 0.8,
        "capacity_sessions_per_sec": 1.0,
        "runs": [
            {
                "repetition": repetition,
                "trace_sha256": f"trace-{repetition}",
                "completion_slope_sessions_per_sec": 1.0,
            }
            for repetition in (1, 2, 3)
        ],
    }


def _capacity_dimensions(key: str) -> tuple[str, int]:
    scenario, _, gpu = key.split("__")
    return scenario, int(gpu.removeprefix("g"))


def _write_preparation_evidence(root: Path) -> None:
    trial_matrix_path = root / "prepared" / "trial_matrix.jsonl"
    _write_jsonl(
        trial_matrix_path,
        [trial.model_dump(mode="json") for trial in build_trial_matrix()],
    )
    sample_references: dict[str, dict[str, str]] = {}
    selection_references: dict[str, dict[str, str]] = {}
    for scenario in ("qmsum", "mbpp"):
        sample_path = root / "prepared" / "samples" / f"{scenario}.jsonl"
        sample_ids = [f"{scenario}-{index:02d}" for index in range(24)]
        _write_jsonl(sample_path, [{"sample_id": sample_id} for sample_id in sample_ids])
        sample_reference = _artifact_reference(root, sample_path)
        sample_references[scenario] = sample_reference
        selection_path = (
            root / "prepared" / "sample_selection" / f"{scenario}.json"
        )
        write_json_exclusive(
            selection_path,
            {
                "scenario": scenario,
                "parent_sample_count": 60,
                "formal_sample_count": 24,
                "cost_metric": (
                    "sum_chunk_input_tokens"
                    if scenario == "qmsum"
                    else "repair_input_tokens"
                ),
                "selected_sample_ids": sample_ids,
                "stratum_counts": {"short": 8, "medium": 8, "long": 8},
                "formal_manifest": sample_reference,
            },
        )
        selection_references[scenario] = _artifact_reference(root, selection_path)

    cache_path = root / "prepared" / "synthetic_prediction_cache.json"
    write_json_exclusive(cache_path, {"profiles": []})
    cache_reference = _artifact_reference(root, cache_path)
    generation_path = (
        root / "prepared" / "synthetic_prediction_cache_generation.json"
    )
    write_json_exclusive(
        generation_path,
        {
            "generator": "test.generator",
            "generator_version": 1,
            "cache": cache_reference,
        },
    )
    preflight_path = root / "prepared" / "reuse" / "parent60_preflight.json"
    write_json_exclusive(
        preflight_path,
        {
            "source_experiment_id": "system_20260713",
            "construction_sha256": "1" * 64,
        },
    )
    capacity_path = root / "calibration" / "qmsum__wf-cache__g2" / "capacity.json"
    write_json_exclusive(
        capacity_path,
        _capacity("qmsum__wf-cache__g2", "qmsum", 2),
    )
    reuse_path = (
        root / "calibration" / "qmsum__wf-cache__g2" / "reuse_manifest.json"
    )
    write_json_exclusive(
        reuse_path,
        {
            "source_experiment_id": "system_20260713",
            "parent_session_count": 60,
            "source_capacity": {
                "repository_path": "output/workflow/experiments/system_20260713/"
                "calibration/qmsum__wf-cache__g2/capacity.json",
                "sha256": "2" * 64,
            },
            "reused_capacity": _artifact_reference(root, capacity_path),
        },
    )
    write_json_exclusive(
        root / "prepared" / "preparation_manifest.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "trial_count": 75,
            "trial_matrix": _artifact_reference(root, trial_matrix_path),
            "sample_manifests": sample_references,
            "sample_selection_manifests": selection_references,
            "synthetic_cache": cache_reference,
            "synthetic_cache_generation": _artifact_reference(root, generation_path),
            "parent_preflight": _artifact_reference(root, preflight_path),
            "capacity_reuse": _artifact_reference(root, reuse_path),
            "environment": {
                "hostname": "host",
                "python_version": "3.12",
                "platform": "linux",
                "git_commit": "commit",
                "git_dirty": False,
                "git_diff_sha256": "3" * 64,
                "package_versions": {"pytest": "9"},
            },
        },
    )


def _artifact_reference(root: Path, path: Path) -> dict[str, str]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{canonical_json(row)}\n" for row in rows), encoding="utf-8")


def _interval(mean: float) -> dict[str, float]:
    return {
        "mean": mean,
        "std": 0.0,
        "ci95_low": mean,
        "ci95_high": mean,
    }


def _mapping(value: object) -> Mapping[str, Any]:
    assert isinstance(value, Mapping)
    return cast(Mapping[str, Any], value)
