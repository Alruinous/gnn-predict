from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from itertools import combinations
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from experiment.workflow.analysis import confidence_interval_95, read_jsonl
from experiment.workflow.artifacts import (
    canonical_json,
    file_sha256,
    read_json,
    stable_digest,
    validate_completion,
)
from experiment.workflow.calibration import CapacitySummary, calibration_key
from experiment.workflow.config import EXPERIMENT_ID, TrialSpec
from experiment.workflow.plan import build_trial_matrix
from experiment.workflow.quality import MbppQualitySummary, QmsumQualitySummary

EXPECTED_REPETITIONS = {1, 2, 3, 4, 5}
EXPECTED_CAPACITY_KEYS = {calibration_key("qmsum", 2)}
FORMAL_SESSION_COUNT = 24
FORMAL_TRIAL_COUNT = 75
FORMAL_GROUP_COUNT = 15
OPEN_LOOP_LOAD_PERCENTS = (75, 125)
PENDING_EVIDENCE = (
    "待相关实验单元的 5 次 repetition 全部完成且冻结输入 hash 对齐后生成;"
    "不使用单次 trial、不完整组或缺失值填充。"
)
PREPARATION_EVIDENCE_SPECS = (
    (
        "preparation_manifest",
        "preparation manifest",
        "prepared/preparation_manifest.json",
    ),
    ("trial_matrix", "75-trial matrix", "prepared/trial_matrix.jsonl"),
    ("qmsum_samples", "QMSum 24-sample manifest", "prepared/samples/qmsum.jsonl"),
    ("mbpp_samples", "MBPP 24-sample manifest", "prepared/samples/mbpp.jsonl"),
    (
        "qmsum_selection",
        "QMSum sample-selection provenance",
        "prepared/sample_selection/qmsum.json",
    ),
    (
        "mbpp_selection",
        "MBPP sample-selection provenance",
        "prepared/sample_selection/mbpp.json",
    ),
    (
        "parent_preflight",
        "parent-60 prompt preflight reuse",
        "prepared/reuse/parent60_preflight.json",
    ),
    (
        "capacity_reuse",
        "parent-60 QMSum capacity reuse",
        "calibration/qmsum__wf-cache__g2/reuse_manifest.json",
    ),
    (
        "synthetic_cache",
        "frozen synthetic prediction cache",
        "prepared/synthetic_prediction_cache.json",
    ),
    (
        "synthetic_cache_generation",
        "synthetic cache generation manifest",
        "prepared/synthetic_prediction_cache_generation.json",
    ),
    (
        "environment",
        "frozen repository and software environment",
        "prepared/preparation_manifest.json#environment",
    ),
)
EXPECTED_PREPARATION_EVIDENCE = {name for name, _, _ in PREPARATION_EVIDENCE_SPECS}
STRATEGY_ORDER = ("lg-batch", "wf-fifo", "wf-history", "wf-cache")
STRATEGY_LABELS = {
    "lg-batch": "LG-Batch",
    "wf-fifo": "WF-FIFO",
    "wf-history": "WF-History",
    "wf-cache": "WF-Cache",
}
SCENARIO_ORDER = ("qmsum", "mbpp")
SCENARIO_LABELS = {"qmsum": "QMSum", "mbpp": "MBPP"}
FIGURE_PRESENTATION = {
    "qmsum_strategy_performance": (
        "QMSum 突发负载下的端到端性能",
        "每个 trial 包含固定的 24 个 sessions; "
        "点和误差条分别表示 5 次 trial 的均值与 Student-t 95% CI。",
    ),
    "qmsum_pipeline_bubble": (
        "QMSum 突发负载下的流水线空泡",
        "每个 trial 包含固定的 24 个 sessions; "
        "柱和误差条分别表示 5 次 trial 的均值与 Student-t 95% CI。",
    ),
    "mbpp_resource_tradeoff": (
        "MBPP 突发负载下的吞吐与 GPU 驻留开销",
        "每个 trial 包含固定的 24 个 sessions; "
        "点和误差条分别表示 5 次 trial 的均值与 Student-t 95% CI。",
    ),
    "mbpp_gpu_time_composition": (
        "MBPP 突发负载下的 GPU 时间构成",
        "每个 trial 包含固定的 24 个 sessions;堆叠高度为分量均值。"
        "误差条是各分量自身的 Student-t 95% CI,不表示累计高度的置信区间。",
    ),
}

GROUP_DIMENSIONS = (
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
PAIR_DIMENSIONS = (
    "experiment_id",
    "scenario",
    "gpu_count",
    "workload",
    "max_num_seqs",
    "queue_capacity",
    "load_percent",
)
PAIR_ALIGNMENT_FIELDS = (
    "sample_manifest_sha256",
    "capacity_calibration_sha256",
    "absolute_arrival_rate",
    "workflow_config_sha256",
    "serving_environment_sha256",
    "telemetry_config_sha256",
    "environment_sha256",
)
COMPARISON_METRICS = (
    "makespan_sec",
    "sessions_per_min",
    "sessions_per_min_per_gpu",
    "session_latency_sec.p95",
    "active_gpu_seconds",
    "resident_gpu_seconds",
    "gpu_seconds_per_completion",
    "idle_subtracted_energy_per_completion_joules",
    "pipeline_bubble_ratio",
    "quality.rouge1_mean",
    "quality.rouge2_mean",
    "quality.rouge_l_mean",
    "quality.initial_pass_rate",
    "quality.final_pass_rate",
)


@dataclass(frozen=True, slots=True)
class ReportInputs:
    output_root: Path
    experiment_id: str
    trial_metrics: tuple[dict[str, object], ...]
    group_metrics: tuple[dict[str, object], ...]
    quality_summaries: tuple[dict[str, object], ...]
    capacities: tuple[dict[str, object], ...]
    failures: tuple[dict[str, object], ...]
    source_artifacts: dict[str, object] = dataclass_field(default_factory=dict)
    preparation_evidence: tuple[dict[str, object], ...] = ()
    preparation_environment: dict[str, object] = dataclass_field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PublishedReport:
    summary_path: Path
    results_document: Path
    figure_paths: tuple[Path, ...]


def load_report_inputs(output_root: Path) -> ReportInputs:
    root = output_root.resolve()
    manifest = _mapping(read_json(root / "experiment_manifest.json"))
    experiment_id = _string(manifest.get("experiment_id"))
    if experiment_id != EXPERIMENT_ID:
        raise ValueError(
            f"report expects experiment {EXPERIMENT_ID}, got {experiment_id}"
        )
    trial_metrics_path = root / "analysis" / "trial_metrics.jsonl"
    group_metrics_path = root / "analysis" / "group_metrics.jsonl"
    analysis_paths = (trial_metrics_path, group_metrics_path)
    existing_analysis = tuple(path.is_file() for path in analysis_paths)
    if any(existing_analysis) and not all(existing_analysis):
        raise FileNotFoundError("analysis inputs must exist together")
    trial_metrics = (
        tuple(read_jsonl(trial_metrics_path)) if all(existing_analysis) else ()
    )
    group_metrics = (
        tuple(read_jsonl(group_metrics_path)) if all(existing_analysis) else ()
    )
    quality_summaries = tuple(_load_quality_summary(root, row) for row in trial_metrics)
    capacities = tuple(
        _load_capacity(root, path)
        for path in sorted((root / "calibration").glob("*/capacity.json"))
    )
    failures = tuple(
        _load_failure(root, path)
        for path in sorted((root / "failed").rglob("failure.json"))
    )
    preparation_evidence, preparation_environment = _load_preparation_evidence(root)
    return ReportInputs(
        output_root=root,
        experiment_id=experiment_id,
        trial_metrics=trial_metrics,
        group_metrics=group_metrics,
        quality_summaries=quality_summaries,
        capacities=capacities,
        failures=failures,
        source_artifacts={
            "trial_metrics": _optional_source_artifact(root, trial_metrics_path),
            "group_metrics": _optional_source_artifact(root, group_metrics_path),
        },
        preparation_evidence=preparation_evidence,
        preparation_environment=preparation_environment,
    )


def build_report(output_root: Path) -> tuple[dict[str, object], str]:
    summary = build_report_summary(load_report_inputs(output_root))
    return summary, render_report_markdown(summary)


def publish_report(output_root: Path, results_document: Path) -> PublishedReport:
    summary, _ = build_report(output_root)
    root = output_root.resolve()
    document = results_document.resolve()
    from experiment.workflow.plots import generate_workflow_plots

    figure_paths = generate_workflow_plots(
        _mapping_rows(summary["group_metrics"]),
        root / "analysis" / "figures",
    )
    summary["figures"] = _figure_records(figure_paths, document.parent)
    summary_path = root / "analysis" / "report_summary.json"
    _atomic_write_text(summary_path, f"{canonical_json(summary)}\n")
    _atomic_write_text(document, render_report_markdown(summary))
    return PublishedReport(
        summary_path=summary_path,
        results_document=document,
        figure_paths=figure_paths,
    )


def build_report_summary(inputs: ReportInputs) -> dict[str, object]:
    if inputs.experiment_id != EXPERIMENT_ID:
        raise ValueError(
            f"report expects experiment {EXPERIMENT_ID}, got {inputs.experiment_id}"
        )
    quality_by_trial = _quality_by_trial(inputs.quality_summaries)
    trial_rows = tuple(
        _attach_quality(row, quality_by_trial.get(_string(row.get("trial_id"))))
        for row in inputs.trial_metrics
    )
    quality_groups = aggregate_quality_metrics(trial_rows)
    completeness = _build_completeness(
        inputs.trial_metrics,
        inputs.group_metrics,
        inputs.quality_summaries,
        quality_groups,
        inputs.capacities,
        inputs.failures,
        inputs.preparation_evidence,
    )
    return {
        "version": 1,
        "experiment_id": inputs.experiment_id,
        "status": "formal" if completeness["formal"] else "partial",
        "completeness": completeness,
        "capacities": list(inputs.capacities),
        "group_metrics": list(inputs.group_metrics),
        "quality_groups": quality_groups,
        "comparisons": paired_trial_comparisons(trial_rows),
        "failures": list(inputs.failures),
        "preparation_evidence": list(inputs.preparation_evidence),
        "preparation_environment": dict(inputs.preparation_environment),
        "source_artifacts": {
            "trial_metrics": inputs.source_artifacts.get(
                "trial_metrics", {"path": "analysis/trial_metrics.jsonl"}
            ),
            "group_metrics": inputs.source_artifacts.get(
                "group_metrics", {"path": "analysis/group_metrics.jsonl"}
            ),
            "quality": "trials/*/quality_summary.json",
            "capacity": "calibration/*/capacity.json",
            "failures": "failed/**/failure.json",
            "preparation": "prepared/preparation_manifest.json",
        },
    }


def paired_trial_comparisons(
    trial_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, dict[int, Mapping[str, object]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    dimensions_by_key: dict[str, dict[str, object]] = {}
    for row in trial_rows:
        dimensions = _dimensions(row, PAIR_DIMENSIONS)
        key = canonical_json(dimensions)
        strategy = _string(row.get("strategy"))
        repetition = _integer(row.get("repetition"))
        if repetition in grouped[key][strategy]:
            raise ValueError("comparison group contains duplicate repetitions")
        grouped[key][strategy][repetition] = row
        dimensions_by_key[key] = dimensions

    results = []
    for key in sorted(grouped):
        strategies = sorted(
            grouped[key], key=lambda strategy: STRATEGY_ORDER.index(strategy)
        )
        for reference_strategy, candidate_strategy in combinations(strategies, 2):
            reference = grouped[key][reference_strategy]
            candidate = grouped[key][candidate_strategy]
            if set(reference) != EXPECTED_REPETITIONS:
                continue
            if set(candidate) != EXPECTED_REPETITIONS:
                continue
            paired_rows = [
                (reference[repetition], candidate[repetition])
                for repetition in sorted(EXPECTED_REPETITIONS)
            ]
            for reference_row, candidate_row in paired_rows:
                _validate_pair(reference_row, candidate_row)
            metrics = _paired_metrics(paired_rows)
            if not metrics:
                continue
            dimensions = dimensions_by_key[key]
            identity = {
                **dimensions,
                "reference_strategy": reference_strategy,
                "candidate_strategy": candidate_strategy,
            }
            results.append(
                {
                    "comparison_id": stable_digest(identity),
                    **identity,
                    "trial_count": 5,
                    "ratio_direction": "candidate_over_reference",
                    "metrics": metrics,
                }
            )
    return results


def aggregate_quality_metrics(
    trial_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    dimensions_by_key: dict[str, dict[str, object]] = {}
    for row in trial_rows:
        metrics = _mapping(row.get("metrics"))
        if not isinstance(metrics.get("quality"), Mapping):
            continue
        dimensions = _dimensions(row, GROUP_DIMENSIONS)
        key = canonical_json(dimensions)
        grouped[key].append(row)
        dimensions_by_key[key] = dimensions

    results = []
    for key in sorted(grouped):
        rows = grouped[key]
        repetitions = {_integer(row.get("repetition")) for row in rows}
        if len(rows) != 5 or repetitions != EXPECTED_REPETITIONS:
            continue
        dimensions = dimensions_by_key[key]
        metrics = _aggregate_quality_rows(rows, _string(dimensions["scenario"]))
        results.append(
            {
                "group_id": stable_digest(dimensions),
                **dimensions,
                "trial_count": 5,
                "trial_ids": sorted(_string(row.get("trial_id")) for row in rows),
                "metrics": metrics,
            }
        )
    return results


def render_report_markdown(summary: Mapping[str, object]) -> str:
    experiment_id = _string(summary.get("experiment_id"))
    status = _string(summary.get("status"))
    completeness = _mapping(summary.get("completeness"))
    group_rows = _mapping_rows(summary.get("group_metrics"))
    main_groups = _main_burst_groups(group_rows)
    comparison_rows = _mapping_rows(summary.get("comparisons"))
    comparisons = _paper_comparisons(comparison_rows)
    open_loop_comparisons = _open_loop_comparisons(comparison_rows)
    all_quality_groups = _mapping_rows(summary.get("quality_groups"))
    quality_groups = _main_burst_groups(all_quality_groups)
    ablation_table = _qmsum_ablation_table(group_rows)
    open_loop_table = _open_loop_table(group_rows, _qmsum_capacity(summary))
    mbpp_two_gpu_table = _mbpp_two_gpu_table(group_rows)
    figure_value = summary.get("figures")
    figures = [] if figure_value is None else _mapping_rows(figure_value)
    trials = _mapping(completeness.get("trials"))
    groups = _mapping(completeness.get("groups"))
    trial_actual = _integer(trials.get("actual"))
    trial_expected = _integer(trials.get("expected"))
    group_actual = _integer(groups.get("actual"))
    group_expected = _integer(groups.get("expected"))

    has_results = bool(group_rows or comparison_rows or all_quality_groups)
    lines = [
        f"# Workflow 系统实验结果 ({experiment_id})",
        "",
        "## 核心结果",
        "",
    ]
    if status == "formal":
        lines.append(
            "完整实验矩阵已覆盖;下表报告绝对指标和基于同 repetition 配对的效应量。"
        )
    elif has_results:
        lines.append(
            "下表为已完成 n=5 实验组的阶段性结果;"
            "未完成配置不参与计算,当前结果不外推到完整矩阵。"
        )
    else:
        lines.append(
            "当前尚无满足 n=5 且冻结输入 hash 对齐的实验组;"
            "本文档保留完整证据骨架,不报告系统效应量。"
        )
    lines.extend(
        (
            "",
            "统计口径:每组包含 5 次独立 trial,均值的 95% CI 使用 "
            "Student-t 区间(df=4)。session p95 先在每个 trial 的 24 个 "
            "sessions 内计算,再跨 5 次 trial 汇总。",
            "",
            "质量 CI 仅反映固定 24 个样本工作负载在 5 次 trial 间的波动,"
            "不表示对完整数据集的抽样不确定性。",
            "",
            "## 实验口径",
            "",
            f"- 正式矩阵为 {FORMAL_GROUP_COUNT} 个实验单元,"
            f"每个单元 5 次重复,共 {FORMAL_TRIAL_COUNT} trials;"
            f"每个 trial 固定 {FORMAL_SESSION_COUNT} 个 sessions。",
            "- 本实验验证调度系统的性能、资源和机制行为,"
            "不将固定子集解释为算法级或数据集级 benchmark。",
            "- QMSum open-loop 负载以已冻结的 parent-60 QMSum g2 capacity 为基准;"
            "它是到达率输入,不是本轮 24-session workload 的新容量结论。",
            "",
            "## 准备与复现证据索引",
            "",
            _preparation_evidence_table(summary),
            "",
            "### 冻结环境与代码状态",
            "",
            _preparation_environment_table(summary),
            "",
            "## 论文证据地图",
            "",
            _evidence_map(group_rows, all_quality_groups),
            "",
            "## 主要绝对指标",
            "",
            "### 性能",
            "",
            _performance_table(main_groups) if main_groups else PENDING_EVIDENCE,
            "",
            "### 资源效率",
            "",
            _resource_efficiency_table(main_groups)
            if main_groups
            else PENDING_EVIDENCE,
            "",
            "## 主配对效应",
            "",
            _comparison_table(comparisons) if comparisons else PENDING_EVIDENCE,
            "",
            "## 核心机制",
            "",
            _mechanism_tables(main_groups) if main_groups else PENDING_EVIDENCE,
            "",
            "## 质量守护指标",
            "",
            _quality_tables(quality_groups) if quality_groups else PENDING_EVIDENCE,
            "",
            "## QMSum batching 与队列消融",
            "",
            ablation_table or PENDING_EVIDENCE,
            "",
            "## 开放到达负载扫描",
            "",
            open_loop_table or PENDING_EVIDENCE,
        )
    )
    if open_loop_comparisons:
        lines.extend(
            (
                "",
                "### 同 repetition 配对效应",
                "",
                _open_loop_comparison_table(open_loop_comparisons),
            )
        )
    lines.extend(
        (
            "",
            "## MBPP 2-GPU 策略对比",
            "",
            mbpp_two_gpu_table or PENDING_EVIDENCE,
            "",
            "## 核心图",
            "",
            _figure_markdown(figures) if figures else PENDING_EVIDENCE,
            "",
            "## 分析与结论记录",
            "",
            _conclusion_slots(status, group_rows),
            "",
            "## 写作边界",
            "",
            "- 只有 n=5 且输入 hash 对齐的实验单元可用于正式数值和配对效应。",
            "- 部分矩阵结果不外推到未运行配置,机制伴随指标不单独作因果证据。",
            "- 固定 24 个样本用于系统验证;质量指标只是运行时语义 guardrail。",
            "- synthetic prediction cache 是冻结调度输入,"
            "不支持 GNN 预测精度或算法收益结论。",
        )
    )
    pending_trials = len(_string_list(trials.get("missing")))
    pending_groups = len(_string_list(groups.get("missing")))
    lines.extend(
        (
            "",
            "## 实验覆盖",
            "",
            f"已完成 {trial_actual}/{trial_expected} trials 和 "
            f"{group_actual}/{group_expected} 个 n=5 实验组;"
            f"待完成 {pending_trials} trials、{pending_groups} 个实验组。",
            "",
            _completeness_table(completeness),
        )
    )
    lines.extend(("", "## 容量标定", "", _capacity_table(summary)))
    lines.extend(("", "## 失败与排除", "", _failure_table(summary)))
    lines.extend(
        (
            "",
            "## 机器可读产物",
            "",
            "- `analysis/trial_metrics.jsonl`",
            "- `analysis/group_metrics.jsonl`",
            "- `analysis/report_summary.json`",
            "",
        )
    )
    return "\n".join(lines)


def _load_quality_summary(
    root: Path, trial_row: Mapping[str, object]
) -> dict[str, object]:
    trial_id = _string(trial_row.get("trial_id"))
    scenario = _string(trial_row.get("scenario"))
    trial_dir = root / "trials" / trial_id
    completion = validate_completion(trial_dir)
    if "quality_summary.json" not in completion.artifact_sha256:
        raise ValueError(f"completion marker omits quality summary: {trial_dir}")
    value = read_json(trial_dir / "quality_summary.json")
    if scenario == "qmsum":
        summary = QmsumQualitySummary.model_validate(value)
    elif scenario == "mbpp":
        summary = MbppQualitySummary.model_validate(value)
    else:
        raise ValueError(f"unsupported quality scenario: {scenario}")
    return {
        "trial_id": trial_id,
        "scenario": scenario,
        "metrics": summary.model_dump(mode="json", exclude={"scenario"}),
    }


def _load_capacity(root: Path, path: Path) -> dict[str, object]:
    summary = CapacitySummary.model_validate(read_json(path))
    return {
        "artifact_path": str(path.relative_to(root)),
        **summary.model_dump(mode="json"),
    }


def _load_failure(root: Path, path: Path) -> dict[str, object]:
    return {
        "artifact_path": str(path.relative_to(root)),
        "data": dict(_mapping(read_json(path))),
    }


def _load_preparation_evidence(
    root: Path,
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    manifest_path = root / "prepared" / "preparation_manifest.json"
    if not manifest_path.is_file():
        return _missing_preparation_evidence(), {}
    manifest = _mapping(read_json(manifest_path))
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("preparation manifest has the wrong experiment ID")
    if manifest.get("trial_count") != FORMAL_TRIAL_COUNT:
        raise ValueError("preparation manifest has the wrong trial count")

    sample_references = _scenario_references(manifest, "sample_manifests")
    selection_references = _scenario_references(
        manifest,
        "sample_selection_manifests",
    )
    references = {
        "trial_matrix": manifest.get("trial_matrix"),
        "qmsum_samples": sample_references["qmsum"],
        "mbpp_samples": sample_references["mbpp"],
        "qmsum_selection": selection_references["qmsum"],
        "mbpp_selection": selection_references["mbpp"],
        "parent_preflight": manifest.get("parent_preflight"),
        "capacity_reuse": manifest.get("capacity_reuse"),
        "synthetic_cache": manifest.get("synthetic_cache"),
        "synthetic_cache_generation": manifest.get("synthetic_cache_generation"),
    }
    records = {
        "preparation_manifest": _preparation_record(
            "preparation_manifest",
            "prepared/preparation_manifest.json",
            file_sha256(manifest_path),
            f"experiment={EXPERIMENT_ID}; trials={FORMAL_TRIAL_COUNT}",
        )
    }
    paths: dict[str, Path] = {}
    for name, reference in references.items():
        record, path = _verified_preparation_reference(root, name, reference)
        records[name] = record
        paths[name] = path

    trial_specs = tuple(
        TrialSpec.model_validate(row) for row in read_jsonl(paths["trial_matrix"])
    )
    if {trial.trial_id for trial in trial_specs} != {
        trial.trial_id for trial in build_trial_matrix()
    }:
        raise ValueError("prepared trial matrix does not match the formal matrix")
    records["trial_matrix"]["provenance"] = (
        f"{len(trial_specs)} trials; {FORMAL_GROUP_COUNT} n=5 evidence groups"
    )

    for scenario in SCENARIO_ORDER:
        sample_name = f"{scenario}_samples"
        samples = tuple(read_jsonl(paths[sample_name]))
        if len(samples) != FORMAL_SESSION_COUNT:
            raise ValueError(f"{scenario} sample manifest must contain 24 samples")
        records[sample_name]["provenance"] = (
            f"{len(samples)} frozen samples for system validation"
        )

        selection_name = f"{scenario}_selection"
        selection = _mapping(read_json(paths[selection_name]))
        if (
            selection.get("scenario") != scenario
            or selection.get("parent_sample_count") != 60
            or selection.get("formal_sample_count") != FORMAL_SESSION_COUNT
        ):
            raise ValueError(f"{scenario} sample selection provenance is invalid")
        selected_ids = _string_list(selection.get("selected_sample_ids"))
        if len(selected_ids) != FORMAL_SESSION_COUNT:
            raise ValueError(f"{scenario} sample selection must contain 24 IDs")
        if dict(_mapping(selection.get("formal_manifest"))) != dict(
            _mapping(sample_references[scenario])
        ):
            raise ValueError(f"{scenario} formal sample reference does not match")
        records[selection_name]["provenance"] = (
            f"parent=60; formal=24; metric={_string(selection.get('cost_metric'))}; "
            f"strata={canonical_json(_mapping(selection.get('stratum_counts')))}"
        )

    preflight = _mapping(read_json(paths["parent_preflight"]))
    if preflight.get("source_experiment_id") != "system_20260713":
        raise ValueError("parent preflight has the wrong source experiment")
    records["parent_preflight"]["provenance"] = (
        "source=system_20260713; parent sessions=60; "
        f"construction={_string(preflight.get('construction_sha256'))}"
    )

    reuse = _mapping(read_json(paths["capacity_reuse"]))
    if (
        reuse.get("source_experiment_id") != "system_20260713"
        or reuse.get("parent_session_count") != 60
    ):
        raise ValueError("capacity reuse provenance is invalid")
    reused_capacity = _mapping(reuse.get("reused_capacity"))
    _verify_artifact_reference(root, reused_capacity)
    source_capacity = _mapping(reuse.get("source_capacity"))
    records["capacity_reuse"]["provenance"] = (
        "source=system_20260713 parent-60; "
        f"source_sha={_string(source_capacity.get('sha256'))}; "
        f"reused_sha={_string(reused_capacity.get('sha256'))}"
    )

    generation = _mapping(read_json(paths["synthetic_cache_generation"]))
    if dict(_mapping(generation.get("cache"))) != dict(
        _mapping(references["synthetic_cache"])
    ):
        raise ValueError("synthetic cache generation reference does not match")
    records["synthetic_cache_generation"]["provenance"] = (
        f"generator={_string(generation.get('generator'))}; "
        f"version={_integer(generation.get('generator_version'))}"
    )
    records["synthetic_cache"]["provenance"] = "frozen scheduler input"

    environment = dict(_mapping(manifest.get("environment")))
    _validate_preparation_environment(environment)
    records["environment"] = _preparation_record(
        "environment",
        "prepared/preparation_manifest.json#environment",
        stable_digest(environment),
        f"git={_string(environment.get('git_commit'))}; "
        f"dirty={environment.get('git_dirty')}; "
        f"diff={_string(environment.get('git_diff_sha256'))}",
    )
    ordered = tuple(records[name] for name, _, _ in PREPARATION_EVIDENCE_SPECS)
    return ordered, environment


def _scenario_references(
    manifest: Mapping[str, object],
    field: str,
) -> Mapping[str, object]:
    references = _mapping(manifest.get(field))
    if set(references) != set(SCENARIO_ORDER):
        raise ValueError(f"preparation manifest has invalid {field}")
    return references


def _verified_preparation_reference(
    root: Path,
    name: str,
    value: object,
) -> tuple[dict[str, object], Path]:
    reference = _mapping(value)
    expected_path = next(
        path for key, _, path in PREPARATION_EVIDENCE_SPECS if key == name
    )
    relative_path = _string(reference.get("relative_path"))
    if relative_path != expected_path:
        raise ValueError(f"unexpected preparation artifact path for {name}")
    path = _verify_artifact_reference(root, reference)
    return (
        _preparation_record(
            name,
            relative_path,
            _string(reference.get("sha256")),
            "hash verified",
        ),
        path,
    )


def _verify_artifact_reference(
    root: Path,
    reference: Mapping[str, object],
) -> Path:
    relative_path = _string(reference.get("relative_path"))
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("preparation artifact escapes the experiment root") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_sha = _string(reference.get("sha256"))
    if file_sha256(path) != expected_sha:
        raise ValueError(f"preparation artifact hash mismatch: {relative_path}")
    return path


def _validate_preparation_environment(environment: Mapping[str, object]) -> None:
    required = {
        "hostname",
        "python_version",
        "platform",
        "git_commit",
        "git_dirty",
        "git_diff_sha256",
        "package_versions",
    }
    if set(environment) != required:
        raise ValueError("preparation environment is incomplete")
    if not isinstance(environment.get("git_dirty"), bool):
        raise TypeError("preparation git_dirty must be a boolean")
    for field in required - {"git_dirty", "package_versions"}:
        _string(environment.get(field))
    _mapping(environment.get("package_versions"))


def _preparation_record(
    name: str,
    artifact_path: str,
    sha256: str,
    provenance: str,
) -> dict[str, object]:
    label = next(label for key, label, _ in PREPARATION_EVIDENCE_SPECS if key == name)
    return {
        "name": name,
        "label": label,
        "artifact_path": artifact_path,
        "sha256": sha256,
        "status": "verified",
        "provenance": provenance,
    }


def _missing_preparation_evidence() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "name": name,
            "label": label,
            "artifact_path": path,
            "sha256": None,
            "status": "missing",
            "provenance": "not prepared",
        }
        for name, label, path in PREPARATION_EVIDENCE_SPECS
    )


def _quality_by_trial(
    summaries: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    resolved: dict[str, Mapping[str, object]] = {}
    for summary in summaries:
        trial_id = _string(summary.get("trial_id"))
        if trial_id in resolved:
            raise ValueError("quality summaries contain duplicate trial IDs")
        resolved[trial_id] = _mapping(summary.get("metrics"))
    return resolved


def _attach_quality(
    row: Mapping[str, object], quality: Mapping[str, object] | None
) -> dict[str, object]:
    result = dict(row)
    metrics = dict(_mapping(row.get("metrics")))
    if quality is not None:
        metrics["quality"] = dict(quality)
    result["metrics"] = metrics
    return result


def _build_completeness(
    trial_rows: Sequence[Mapping[str, object]],
    group_rows: Sequence[Mapping[str, object]],
    quality_rows: Sequence[Mapping[str, object]],
    quality_groups: Sequence[Mapping[str, object]],
    capacities: Sequence[Mapping[str, object]],
    failures: Sequence[Mapping[str, object]],
    preparation_evidence: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    expected_trials = build_trial_matrix()
    expected_trial_ids = {trial.trial_id for trial in expected_trials}
    expected_group_ids = {_trial_group_id(trial) for trial in expected_trials}
    if len(expected_trial_ids) != FORMAL_TRIAL_COUNT:
        raise RuntimeError(
            f"formal report expects {FORMAL_TRIAL_COUNT} trials, "
            f"got {len(expected_trial_ids)}"
        )
    if len(expected_group_ids) != FORMAL_GROUP_COUNT:
        raise RuntimeError(
            f"formal report expects {FORMAL_GROUP_COUNT} groups, "
            f"got {len(expected_group_ids)}"
        )
    trial_ids = _unique_strings(trial_rows, "trial_id")
    group_ids = {_group_id(row) for row in group_rows}
    quality_ids = _unique_strings(quality_rows, "trial_id")
    quality_group_ids = {_group_id(row) for row in quality_groups}
    capacity_keys = _unique_strings(capacities, "calibration_key")
    preparation_names = {
        _string(value.get("name"))
        for value in preparation_evidence
        if value.get("status") == "verified"
    }
    formal = (
        trial_ids == expected_trial_ids
        and group_ids == expected_group_ids
        and len(group_rows) == len(expected_group_ids)
        and quality_ids == expected_trial_ids
        and quality_group_ids == expected_group_ids
        and len(quality_groups) == len(expected_group_ids)
        and capacity_keys == EXPECTED_CAPACITY_KEYS
        and preparation_names == EXPECTED_PREPARATION_EVIDENCE
    )
    return {
        "formal": formal,
        "trials": _set_completeness(expected_trial_ids, trial_ids),
        "groups": _set_completeness(expected_group_ids, group_ids),
        "reportable_aligned_groups": {
            "actual": len(group_ids),
            "ids": sorted(group_ids),
        },
        "quality_trials": _set_completeness(expected_trial_ids, quality_ids),
        "quality_groups": _set_completeness(expected_group_ids, quality_group_ids),
        "capacities": _set_completeness(EXPECTED_CAPACITY_KEYS, capacity_keys),
        "preparation_evidence": _set_completeness(
            EXPECTED_PREPARATION_EVIDENCE,
            preparation_names,
        ),
        "failures": {"actual": len(failures)},
    }


def _set_completeness(expected: set[str], actual: set[str]) -> dict[str, object]:
    return {
        "expected": len(expected),
        "actual": len(actual),
        "missing": sorted(expected - actual),
        "unexpected": sorted(actual - expected),
        "complete": actual == expected,
    }


def _paired_metrics(
    rows: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for path in COMPARISON_METRICS:
        reference = [_metric_number(left, path) for left, _ in rows]
        candidate = [_metric_number(right, path) for _, right in rows]
        if any(value is None for value in reference + candidate):
            continue
        reference_values = cast(list[float], reference)
        candidate_values = cast(list[float], candidate)
        ratios = (
            None
            if any(value == 0 for value in reference_values)
            else [
                candidate_value / reference_value
                for reference_value, candidate_value in zip(
                    reference_values, candidate_values, strict=True
                )
            ]
        )
        inverse_ratios = (
            None
            if any(value == 0 for value in candidate_values)
            else [
                reference_value / candidate_value
                for reference_value, candidate_value in zip(
                    reference_values, candidate_values, strict=True
                )
            ]
        )
        relative_reductions = (
            None
            if any(value == 0 for value in reference_values)
            else [
                (reference_value - candidate_value) / reference_value
                for reference_value, candidate_value in zip(
                    reference_values, candidate_values, strict=True
                )
            ]
        )
        result[path] = {
            "reference": confidence_interval_95(reference_values),
            "candidate": confidence_interval_95(candidate_values),
            "delta": confidence_interval_95(
                [
                    candidate_value - reference_value
                    for reference_value, candidate_value in zip(
                        reference_values, candidate_values, strict=True
                    )
                ]
            ),
            "ratio": confidence_interval_95(ratios) if ratios is not None else None,
            "inverse_ratio": (
                confidence_interval_95(inverse_ratios)
                if inverse_ratios is not None
                else None
            ),
            "relative_reduction": (
                confidence_interval_95(relative_reductions)
                if relative_reductions is not None
                else None
            ),
        }
    return result


def _validate_pair(
    reference: Mapping[str, object], candidate: Mapping[str, object]
) -> None:
    if reference.get("arrival_trace_sha256") != candidate.get("arrival_trace_sha256"):
        raise ValueError("paired trials use different arrival traces")
    reference_alignment = _mapping(reference.get("alignment"))
    candidate_alignment = _mapping(candidate.get("alignment"))
    mismatched = [
        field
        for field in PAIR_ALIGNMENT_FIELDS
        if reference_alignment.get(field) != candidate_alignment.get(field)
    ]
    strategies = {
        _string(reference.get("strategy")),
        _string(candidate.get("strategy")),
    }
    if "lg-batch" not in strategies and reference_alignment.get(
        "prediction_cache_sha256"
    ) != candidate_alignment.get("prediction_cache_sha256"):
        mismatched.append("prediction_cache_sha256")
    if mismatched:
        raise ValueError(f"paired trial alignment mismatch: {mismatched}")


def _aggregate_quality_rows(
    rows: Sequence[Mapping[str, object]], scenario: str
) -> dict[str, object]:
    quality = [_mapping(_mapping(row.get("metrics")).get("quality")) for row in rows]
    if scenario == "qmsum":
        keys = ("rouge1_mean", "rouge2_mean", "rouge_l_mean", "empty_output_count")
        return {
            key: confidence_interval_95([_number(value.get(key)) for value in quality])
            for key in keys
        }
    if scenario != "mbpp":
        raise ValueError(f"unsupported quality scenario: {scenario}")
    keys = (
        "initial_pass_count",
        "final_pass_count",
        "initial_pass_rate",
        "final_pass_rate",
        "repaired_count",
        "regressed_count",
    )
    result: dict[str, object] = {
        key: confidence_interval_95([_number(value.get(key)) for value in quality])
        for key in keys
    }
    for field in ("initial_failure_types", "final_failure_types"):
        counts = [_mapping(value.get(field)) for value in quality]
        error_types = sorted({key for count in counts for key in count})
        result[field] = {
            error_type: confidence_interval_95(
                [_number(count.get(error_type, 0)) for count in counts]
            )
            for error_type in error_types
        }
    return result


def _metric_number(row: Mapping[str, object], path: str) -> float | None:
    value: object = _mapping(row.get("metrics"))
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _dimensions(row: Mapping[str, object], names: Sequence[str]) -> dict[str, object]:
    missing = [name for name in names if name not in row]
    if missing:
        raise ValueError(f"report row lacks dimensions: {missing}")
    return {name: row[name] for name in names}


def _group_id(row: Mapping[str, object]) -> str:
    trial = TrialSpec.model_validate(
        {
            "scenario": row.get("scenario"),
            "strategy": row.get("strategy"),
            "gpu_count": row.get("gpu_count"),
            "workload": row.get("workload"),
            "repetition": 1,
            "max_num_seqs": row.get("max_num_seqs"),
            "queue_capacity": row.get("queue_capacity"),
            "load_percent": row.get("load_percent"),
        }
    )
    return _trial_group_id(trial)


def _trial_group_id(trial: TrialSpec) -> str:
    return trial.trial_id.rsplit("__rep", maxsplit=1)[0]


def _unique_strings(rows: Sequence[Mapping[str, object]], field: str) -> set[str]:
    values = [_string(row.get(field)) for row in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"report rows contain duplicate {field}")
    return set(values)


def _completeness_table(completeness: Mapping[str, object]) -> str:
    lines = ["| 产物 | 完成 | 期望 | 状态 |", "|---|---:|---:|---|"]
    for key, label in (
        ("trials", "trial"),
        ("groups", "aligned group"),
        ("quality_trials", "quality trial"),
        ("quality_groups", "quality group"),
        ("capacities", "capacity"),
        ("preparation_evidence", "preparation evidence"),
    ):
        value = _mapping(completeness.get(key))
        status = "complete" if value.get("complete") is True else "partial"
        lines.append(
            f"| {label} | {_integer(value.get('actual'))} | "
            f"{_integer(value.get('expected'))} | {status} |"
        )
    return "\n".join(lines)


def _capacity_table(summary: Mapping[str, object]) -> str:
    capacities = _mapping_rows(summary.get("capacities"))
    if not capacities:
        return "尚无完成的 capacity summary。"
    lines = [
        "| 场景 | GPU | capacity sessions/s |",
        "|---|---:|---:|",
    ]
    lines.extend(_capacity_row(value) for value in capacities)
    return "\n".join(lines)


def _preparation_evidence_table(summary: Mapping[str, object]) -> str:
    value = summary.get("preparation_evidence", [])
    records = {_string(record.get("name")): record for record in _mapping_rows(value)}
    if len(records) != len(_mapping_rows(value)):
        raise ValueError("preparation evidence contains duplicate names")
    lines = [
        "| 证据 | 产物 | SHA-256 | 状态 | 来源与语义 |",
        "|---|---|---|---|---|",
    ]
    for name, label, artifact_path in PREPARATION_EVIDENCE_SPECS:
        record = records.get(name)
        if record is None:
            status = "待生成"
            digest = "—"
            provenance = "未执行准备或尚未写入 preparation manifest"
        else:
            status = "已验证" if record.get("status") == "verified" else "待生成"
            digest_value = record.get("sha256")
            digest = "—" if digest_value is None else f"`{_string(digest_value)}`"
            provenance = str(record.get("provenance", ""))
            artifact_path = _string(record.get("artifact_path"))
            label = _string(record.get("label"))
        lines.append(
            f"| {_markdown(label)} | `{_markdown(artifact_path)}` | {digest} | "
            f"{status} | {_markdown(provenance)} |"
        )
    return "\n".join(lines)


def _preparation_environment_table(summary: Mapping[str, object]) -> str:
    value = summary.get("preparation_environment", {})
    environment = _mapping(value)
    fields = (
        ("hostname", "host"),
        ("python_version", "Python"),
        ("platform", "platform"),
        ("git_commit", "git commit"),
        ("git_dirty", "git dirty"),
        ("git_diff_sha256", "git diff SHA-256"),
    )
    lines = ["| 字段 | 冻结值 |", "|---|---|"]
    for field, label in fields:
        raw = environment.get(field, "—")
        lines.append(f"| {label} | `{_markdown(str(raw))}` |")
    packages_value = environment.get("package_versions", {})
    packages = _mapping(packages_value)
    if not packages:
        lines.append("| package versions | `—` |")
    else:
        lines.extend(
            (f"| package:{_markdown(name)} | `{_markdown(_string(packages[name]))}` |")
            for name in sorted(packages)
        )
    return "\n".join(lines)


def _qmsum_capacity(summary: Mapping[str, object]) -> float | None:
    matches = [
        value
        for value in _mapping_rows(summary.get("capacities"))
        if value.get("calibration_key") == calibration_key("qmsum", 2)
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("report contains duplicate QMSum capacity summaries")
    return _number(matches[0].get("capacity_sessions_per_sec"))


def _capacity_row(value: Mapping[str, object]) -> str:
    scenario = _string(value.get("scenario"))
    gpu_count = _integer(value.get("gpu_count"))
    capacity = _number(value.get("capacity_sessions_per_sec"))
    return f"| {scenario} | {gpu_count} | {capacity:.6f} |"


def _figure_markdown(figures: Sequence[Mapping[str, object]]) -> str:
    lines: list[str] = []
    for figure in figures:
        name = _string(figure.get("name"))
        presentation = FIGURE_PRESENTATION.get(name)
        if presentation is None:
            raise ValueError(f"missing report presentation for figure: {name}")
        title, caption = presentation
        png_path = _string(figure.get("png"))
        pdf_path = _string(figure.get("pdf"))
        if lines:
            lines.append("")
        lines.extend(
            (
                f"### {title}",
                "",
                f"![{title}]({png_path})",
                "",
                caption,
                "",
                f"[PDF 版本]({pdf_path})",
            )
        )
    return "\n".join(lines)


def _performance_table(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "| 场景 | 策略 | GPU | 端到端 makespan (s) | 全程吞吐 (sessions/min) | "
        "steady-state 吞吐 (sessions/min) | session p95 (s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = _mapping(row.get("metrics"))
        scenario = _scenario_label(row)
        strategy = _strategy_label(row)
        gpu_count = _integer(row.get("gpu_count"))
        makespan = _format_interval(metrics.get("makespan_sec"))
        throughput = _format_interval(metrics.get("sessions_per_min"))
        steady_state = _format_interval(metrics.get("steady_state_sessions_per_min"))
        p95 = _format_interval(_nested(metrics, "session_latency_sec", "p95"))
        lines.append(
            f"| {scenario} | {strategy} | {gpu_count} | {makespan} | "
            f"{throughput} | {steady_state} | {p95} |"
        )
    return "\n".join(lines)


def _resource_efficiency_table(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "| 场景 | 策略 | GPU | sessions/min/GPU | resident GPU-s | "
        "GPU-s/completion | idle-subtracted energy/completion (J) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = _mapping(row.get("metrics"))
        scenario = _scenario_label(row)
        strategy = _strategy_label(row)
        gpu_count = _integer(row.get("gpu_count"))
        normalized = _format_interval(metrics.get("sessions_per_min_per_gpu"))
        resident = _format_interval(metrics.get("resident_gpu_seconds"))
        per_completion = _format_interval(metrics.get("gpu_seconds_per_completion"))
        energy = _format_interval(
            metrics.get("idle_subtracted_energy_per_completion_joules")
        )
        lines.append(
            f"| {scenario} | {strategy} | {gpu_count} | {normalized} | "
            f"{resident} | {per_completion} | {energy} |"
        )
    return "\n".join(lines)


def _comparison_table(comparisons: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "每项先按同 repetition 配对,再对五个效应量报告 mean [95% CI]。speedup "
        "大于 1 表示 candidate 更快;reduction 大于 0 表示 candidate 的对应开销更低。",
        "",
        "| 场景 | GPU | 比较(candidate/reference) | makespan speedup | "
        "throughput speedup | p95 reduction | resident GPU-s reduction | "
        "idle-subtracted energy/completion reduction |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        metrics = _mapping(row.get("metrics"))
        scenario = _scenario_label(row)
        gpu_count = _integer(row.get("gpu_count"))
        candidate = _strategy_name(_string(row.get("candidate_strategy")))
        reference = _strategy_name(_string(row.get("reference_strategy")))
        makespan = _format_effect(metrics.get("makespan_sec"), "inverse_ratio")
        throughput = _format_effect(metrics.get("sessions_per_min"), "ratio")
        p95 = _format_effect(
            metrics.get("session_latency_sec.p95"),
            "relative_reduction",
            percent=True,
        )
        resident = _format_effect(
            metrics.get("resident_gpu_seconds"),
            "relative_reduction",
            percent=True,
        )
        energy = _format_effect(
            metrics.get("idle_subtracted_energy_per_completion_joules"),
            "relative_reduction",
            percent=True,
        )
        lines.append(
            f"| {scenario} | {gpu_count} | {candidate}/{reference} | "
            f"{makespan} | {throughput} | {p95} | {resident} | {energy} |"
        )
    return "\n".join(lines)


def _quality_tables(rows: Sequence[Mapping[str, object]]) -> str:
    qmsum = [row for row in rows if row.get("scenario") == "qmsum"]
    mbpp = [row for row in rows if row.get("scenario") == "mbpp"]
    sections: list[str] = []
    if qmsum:
        sections.extend(("### QMSum", "", _qmsum_quality_table(qmsum)))
    if mbpp:
        if sections:
            sections.append("")
        sections.extend(("### MBPP", "", _mbpp_quality_table(mbpp)))
    return "\n".join(sections)


def _qmsum_quality_table(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "| 策略 | GPU | ROUGE-1 | ROUGE-2 | ROUGE-L | empty outputs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = _mapping(row.get("metrics"))
        strategy = _strategy_label(row)
        gpu_count = _integer(row.get("gpu_count"))
        rouge1 = _format_interval(metrics.get("rouge1_mean"), precision=4)
        rouge2 = _format_interval(metrics.get("rouge2_mean"), precision=4)
        rouge_l = _format_interval(metrics.get("rouge_l_mean"), precision=4)
        empty = _format_interval(metrics.get("empty_output_count"))
        lines.append(
            f"| {strategy} | {gpu_count} | {rouge1} | {rouge2} | {rouge_l} | {empty} |"
        )
    return "\n".join(lines)


def _mbpp_quality_table(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "| 策略 | GPU | initial pass@1 | final pass@1 | repaired | regressed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = _mapping(row.get("metrics"))
        strategy = _strategy_label(row)
        gpu_count = _integer(row.get("gpu_count"))
        initial = _format_interval(
            metrics.get("initial_pass_rate"), scale=100.0, suffix="%"
        )
        final = _format_interval(
            metrics.get("final_pass_rate"), scale=100.0, suffix="%"
        )
        repaired = _format_interval(metrics.get("repaired_count"))
        regressed = _format_interval(metrics.get("regressed_count"))
        lines.append(
            f"| {strategy} | {gpu_count} | {initial} | {final} | "
            f"{repaired} | {regressed} |"
        )
    return "\n".join(lines)


def _mechanism_tables(rows: Sequence[Mapping[str, object]]) -> str:
    qmsum = [row for row in rows if row.get("scenario") == "qmsum"]
    mbpp = [row for row in rows if row.get("scenario") == "mbpp"]
    sections: list[str] = []
    if qmsum:
        sections.extend(("### QMSum", "", _qmsum_mechanism_table(qmsum)))
    if mbpp:
        if sections:
            sections.append("")
        sections.extend(("### MBPP", "", _mbpp_mechanism_table(mbpp)))
    return "\n".join(sections)


def _qmsum_mechanism_table(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "| 策略 | GPU | pipeline bubble | mean fan-in wait (s) | "
        "backpressure (s) | mean vLLM queue (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = _mapping(row.get("metrics"))
        strategy = _strategy_label(row)
        gpu_count = _integer(row.get("gpu_count"))
        bubble = _format_interval(
            metrics.get("pipeline_bubble_ratio"), scale=100.0, suffix="%"
        )
        fanin = _format_interval(_nested(metrics, "fanin_wait_sec", "mean"))
        backpressure = _format_interval(metrics.get("backpressure_duration_sec"))
        queue = _format_interval(_nested(metrics, "vllm_queue_time_sec", "mean"))
        lines.append(
            f"| {strategy} | {gpu_count} | {bubble} | {fanin} | "
            f"{backpressure} | {queue} |"
        )
    return "\n".join(lines)


def _mbpp_mechanism_table(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "| 策略 | GPU | loading GPU-s | loads | reuses | prefetches | "
        "evictions | wasted prefetches |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = _mapping(row.get("metrics"))
        strategy = _strategy_label(row)
        gpu_count = _integer(row.get("gpu_count"))
        loading = _format_interval(metrics.get("loading_gpu_seconds"))
        loads = _format_interval(metrics.get("model_load_count"))
        reuses = _format_interval(metrics.get("model_reuse_count"))
        prefetches = _format_interval(metrics.get("prefetch_count"))
        evictions = _format_interval(metrics.get("model_eviction_count"))
        wasted = _format_interval(metrics.get("wasted_prefetch_count"))
        lines.append(
            f"| {strategy} | {gpu_count} | {loading} | {loads} | {reuses} | "
            f"{prefetches} | {evictions} | {wasted} |"
        )
    return "\n".join(lines)


def _qmsum_ablation_table(rows: Sequence[Mapping[str, object]]) -> str:
    selected = _qmsum_ablation_groups(rows)
    keys = {
        (_integer(row.get("max_num_seqs")), _integer(row.get("queue_capacity")))
        for row in selected
    }
    if (3, 16) not in keys or not keys.intersection({(1, 16), (3, 1)}):
        return ""
    lines = [
        "默认配置 `(seq3,q16)` 复用 QMSum 主性能组;"
        "只列新矩阵的两个端点消融,每项为 mean [95% CI]。",
        "",
        "| 角色 | max_num_seqs | queue capacity | makespan (s) | sessions/min | "
        "session p95 (s) | pipeline bubble |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        metrics = _mapping(row.get("metrics"))
        max_num_seqs = _integer(row.get("max_num_seqs"))
        queue_capacity = _integer(row.get("queue_capacity"))
        role = "默认" if (max_num_seqs, queue_capacity) == (3, 16) else "端点消融"
        makespan = _format_interval(metrics.get("makespan_sec"))
        throughput = _format_interval(metrics.get("sessions_per_min"))
        p95 = _format_interval(_nested(metrics, "session_latency_sec", "p95"))
        bubble = _format_interval(
            metrics.get("pipeline_bubble_ratio"), scale=100.0, suffix="%"
        )
        lines.append(
            f"| {role} | {max_num_seqs} | {queue_capacity} | {makespan} | "
            f"{throughput} | {p95} | {bubble} |"
        )
    return "\n".join(lines)


def _qmsum_ablation_groups(
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    allowed = ((3, 16), (1, 16), (3, 1))
    selected: dict[tuple[int, int], Mapping[str, object]] = {}
    for row in rows:
        key = (_integer(row.get("max_num_seqs")), _integer(row.get("queue_capacity")))
        if (
            row.get("scenario") != "qmsum"
            or row.get("strategy") != "wf-cache"
            or row.get("gpu_count") != 2
            or row.get("workload") != "burst"
            or row.get("load_percent") is not None
            or row.get("trial_count") != 5
            or key not in allowed
        ):
            continue
        if key in selected:
            raise ValueError("QMSum ablation report contains duplicate groups")
        selected[key] = row
    return [selected[key] for key in allowed if key in selected]


def _open_loop_table(
    rows: Sequence[Mapping[str, object]], capacity_sessions_per_sec: float | None
) -> str:
    selected = _qmsum_open_loop_groups(rows)
    if not selected:
        return ""
    grouped: dict[tuple[int, str], Mapping[str, object]] = {}
    for row in selected:
        strategy = _string(row.get("strategy"))
        load_percent = _integer(row.get("load_percent"))
        key = (load_percent, strategy)
        if key in grouped:
            raise ValueError("open-loop report contains duplicate load groups")
        grouped[key] = row
    lines = [
        "offered rate 由冻结的 parent-60 QMSum g2 capacity 与 load factor 相乘;"
        "结果格为 throughput (sessions/min) / session p95 (s),均报告 mean [95% CI]。",
        "",
        "| load factor | offered sessions/s | LG-Batch | WF-Cache |",
        "|---:|---:|---:|---:|",
    ]
    for load_percent in OPEN_LOOP_LOAD_PERCENTS:
        rate = (
            "—"
            if capacity_sessions_per_sec is None
            else f"{capacity_sessions_per_sec * load_percent / 100:.10f}"
        )
        lines.append(
            f"| {load_percent}% | {rate} | "
            f"{_open_loop_cell(grouped.get((load_percent, 'lg-batch')))} | "
            f"{_open_loop_cell(grouped.get((load_percent, 'wf-cache')))} |"
        )
    return "\n".join(lines)


def _qmsum_open_loop_groups(
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    return [
        row
        for row in rows
        if row.get("scenario") == "qmsum"
        and row.get("strategy") in ("lg-batch", "wf-cache")
        and row.get("gpu_count") == 2
        and row.get("workload") == "open-loop"
        and row.get("max_num_seqs") == 3
        and row.get("queue_capacity") == 16
        and row.get("load_percent") in OPEN_LOOP_LOAD_PERCENTS
        and row.get("trial_count") == 5
    ]


def _open_loop_cell(row: Mapping[str, object] | None) -> str:
    if row is None:
        return "—"
    metrics = _mapping(row.get("metrics"))
    throughput = _format_interval(metrics.get("sessions_per_min"))
    p95 = _format_interval(_nested(metrics, "session_latency_sec", "p95"))
    return f"{throughput} / {p95}"


def _mbpp_two_gpu_table(rows: Sequence[Mapping[str, object]]) -> str:
    selected = _mbpp_two_gpu_groups(rows)
    if len(selected) < 2:
        return ""
    lines = [
        "同为 2 GPU 的 FIFO、History 和 Cache 策略;仅列已形成 n=5 的组。",
        "",
        "| 策略 | makespan (s) | sessions/min | p95 (s) | loading GPU-s | "
        "loads | reuses | prefetches | evictions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        metrics = _mapping(row.get("metrics"))
        makespan = _format_interval(metrics.get("makespan_sec"))
        throughput = _format_interval(metrics.get("sessions_per_min"))
        p95 = _format_interval(_nested(metrics, "session_latency_sec", "p95"))
        lines.append(
            f"| {_strategy_label(row)} | {makespan} | {throughput} | {p95} | "
            f"{_format_interval(metrics.get('loading_gpu_seconds'))} | "
            f"{_format_interval(metrics.get('model_load_count'))} | "
            f"{_format_interval(metrics.get('model_reuse_count'))} | "
            f"{_format_interval(metrics.get('prefetch_count'))} | "
            f"{_format_interval(metrics.get('model_eviction_count'))} |"
        )
    return "\n".join(lines)


def _mbpp_two_gpu_groups(
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    selected = [
        row
        for row in rows
        if _is_default_burst(row)
        and row.get("scenario") == "mbpp"
        and row.get("gpu_count") == 2
        and row.get("strategy") in ("wf-fifo", "wf-history", "wf-cache")
    ]
    return sorted(
        selected,
        key=lambda row: STRATEGY_ORDER.index(_string(row.get("strategy"))),
    )


def _failure_table(summary: Mapping[str, object]) -> str:
    failures = _mapping_rows(summary.get("failures"))
    if not failures:
        return "无归档失败记录。"
    lines = [
        "| 产物 | 场景 | repetition | 原因 |",
        "|---|---|---:|---|",
    ]
    for failure in failures:
        data = _mapping(failure.get("data"))
        lines.append(
            f"| `{_markdown(_string(failure.get('artifact_path')))}` | "
            f"{_markdown(str(data.get('scenario', '')))} | "
            f"{_markdown(str(data.get('repetition', '')))} | "
            f"{_markdown(str(data.get('archive_reason', '')))} |"
        )
    return "\n".join(lines)


def _main_burst_groups(
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    allowed = {
        ("qmsum", "lg-batch", 2),
        ("qmsum", "wf-fifo", 2),
        ("qmsum", "wf-cache", 2),
        ("mbpp", "lg-batch", 3),
        ("mbpp", "wf-cache", 1),
        ("mbpp", "wf-cache", 2),
        ("mbpp", "wf-cache", 3),
        ("mbpp", "wf-fifo", 2),
        ("mbpp", "wf-history", 2),
    }
    selected = [
        row
        for row in rows
        if _is_default_burst(row)
        and (
            row.get("scenario"),
            row.get("strategy"),
            row.get("gpu_count"),
        )
        in allowed
    ]
    return sorted(
        selected,
        key=lambda row: (
            SCENARIO_ORDER.index(_string(row.get("scenario"))),
            _integer(row.get("gpu_count")),
            STRATEGY_ORDER.index(_string(row.get("strategy"))),
        ),
    )


def _is_default_burst(row: Mapping[str, object]) -> bool:
    return (
        row.get("workload") == "burst"
        and row.get("max_num_seqs") == 3
        and row.get("queue_capacity") == 16
        and row.get("load_percent") is None
        and row.get("trial_count") == 5
    )


def _paper_comparisons(
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    selected = []
    for row in rows:
        if (
            row.get("workload") != "burst"
            or row.get("max_num_seqs") != 3
            or row.get("queue_capacity") != 16
            or row.get("candidate_strategy") != "wf-cache"
        ):
            continue
        reference = row.get("reference_strategy")
        gpu_count = row.get("gpu_count")
        is_qmsum = (
            row.get("scenario") == "qmsum"
            and gpu_count == 2
            and reference in ("lg-batch", "wf-fifo")
        )
        is_mbpp_two_gpu = (
            row.get("scenario") == "mbpp"
            and gpu_count == 2
            and reference in ("wf-fifo", "wf-history")
        )
        is_mbpp_static = (
            row.get("scenario") == "mbpp" and gpu_count == 3 and reference == "lg-batch"
        )
        if is_qmsum or is_mbpp_two_gpu or is_mbpp_static:
            selected.append(row)
    selected.sort(
        key=lambda row: (
            SCENARIO_ORDER.index(_string(row.get("scenario"))),
            _integer(row.get("gpu_count")),
            STRATEGY_ORDER.index(_string(row.get("reference_strategy"))),
        )
    )
    return selected


def _open_loop_comparisons(
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    selected = [
        row
        for row in rows
        if row.get("scenario") == "qmsum"
        and row.get("gpu_count") == 2
        and row.get("workload") == "open-loop"
        and row.get("max_num_seqs") == 3
        and row.get("queue_capacity") == 16
        and row.get("load_percent") in OPEN_LOOP_LOAD_PERCENTS
        and row.get("reference_strategy") == "lg-batch"
        and row.get("candidate_strategy") == "wf-cache"
    ]
    return sorted(selected, key=lambda row: _integer(row.get("load_percent")))


def _open_loop_comparison_table(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "| load factor | throughput speedup | p95 reduction | "
        "resident GPU-s reduction |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = _mapping(row.get("metrics"))
        throughput = _format_effect(metrics.get("sessions_per_min"), "ratio")
        p95 = _format_effect(
            metrics.get("session_latency_sec.p95"),
            "relative_reduction",
            percent=True,
        )
        resident = _format_effect(
            metrics.get("resident_gpu_seconds"),
            "relative_reduction",
            percent=True,
        )
        lines.append(
            f"| {_integer(row.get('load_percent'))}% | "
            f"{throughput} | {p95} | {resident} |"
        )
    return "\n".join(lines)


def _evidence_map(
    rows: Sequence[Mapping[str, object]],
    quality_rows: Sequence[Mapping[str, object]],
) -> str:
    main = _main_burst_groups(rows)
    qmsum_main = [row for row in main if row.get("scenario") == "qmsum"]
    mbpp_resource = [
        row
        for row in main
        if row.get("scenario") == "mbpp"
        and (
            (row.get("strategy") == "lg-batch" and row.get("gpu_count") == 3)
            or row.get("strategy") == "wf-cache"
        )
    ]
    evidence = (
        ("QMSum burst 主性能", len(qmsum_main), 3, "LG/FIFO/Cache 同资源对比"),
        (
            "QMSum 机制消融",
            len(_qmsum_ablation_groups(rows)),
            3,
            "默认点与两个端点消融",
        ),
        (
            "QMSum open-loop",
            len(_qmsum_open_loop_groups(rows)),
            4,
            "LG/Cache 在 75% 与 125% parent capacity 下的稳健性",
        ),
        (
            "MBPP 资源曲线",
            len(mbpp_resource),
            4,
            "LG-g3 与 Cache-g1/g2/g3",
        ),
        (
            "MBPP 2-GPU 策略",
            len(_mbpp_two_gpu_groups(rows)),
            3,
            "FIFO/History/Cache 同资源对比",
        ),
        (
            "质量 guardrail",
            len(quality_rows),
            FORMAL_GROUP_COUNT,
            "空输出、截断、执行失败与语义偏差",
        ),
    )
    lines = [
        "| 论文证据 | 已形成 n=5 组 | 期望 | 写作用途 |",
        "|---|---:|---:|---|",
    ]
    lines.extend(
        f"| {name} | {actual} | {expected} | {purpose} |"
        for name, actual, expected, purpose in evidence
    )
    return "\n".join(lines)


def _conclusion_slots(status: str, rows: Sequence[Mapping[str, object]]) -> str:
    if status == "formal":
        lead = (
            "全部证据单元已齐;下列结论应引用绝对值、同 repetition 配对效应"
            "与对应机制指标,不只引用单个均值。"
        )
    elif rows:
        lead = "当前仅可记录已完成组的阶段性观察;完整矩阵结论和论文摘要数字保持待填。"
    else:
        lead = "尚无可结论的 n=5 实验组;以下槽位在正式证据齐备前保持待填。"
    return "\n".join(
        (
            lead,
            "",
            "- QMSum 主性能结论:待由 LG/FIFO/Cache 绝对值和 "
            "Cache-vs-baseline 配对效应填写。",
            "- QMSum 机制结论:待由两个端点消融与 bubble、queue blocked、"
            "batching/inflight 同向变化填写。",
            "- QMSum 高负载结论:待由 75%/125% 的吞吐、p95 和配对效应填写。",
            "- MBPP 资源结论:待由资源曲线和 2-GPU 同资源策略对比填写。",
            "- 质量边界:只记录 guardrail 是否保持,不写为数据集级质量提升。",
        )
    )


def _scenario_label(row: Mapping[str, object]) -> str:
    return _scenario_name(_string(row.get("scenario")))


def _scenario_name(scenario: str) -> str:
    label = SCENARIO_LABELS.get(scenario)
    if label is None:
        raise ValueError(f"unknown report scenario: {scenario}")
    return label


def _strategy_label(row: Mapping[str, object]) -> str:
    return _strategy_name(_string(row.get("strategy")))


def _strategy_name(strategy: str) -> str:
    label = STRATEGY_LABELS.get(strategy)
    if label is None:
        raise ValueError(f"unknown report strategy: {strategy}")
    return label


def _format_effect(value: object, field: str, *, percent: bool = False) -> str:
    if not isinstance(value, Mapping):
        return "—"
    effect = _mapping(value).get(field)
    if not isinstance(effect, Mapping):
        return "—"
    effect = _mapping(effect)
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else "x"
    mean = _number(effect.get("mean")) * scale
    low = _number(effect.get("ci95_low")) * scale
    high = _number(effect.get("ci95_high")) * scale
    return f"{mean:.3f}{suffix} [{low:.3f}, {high:.3f}]"


def _format_interval(
    value: object,
    *,
    scale: float = 1.0,
    suffix: str = "",
    precision: int = 3,
) -> str:
    if not isinstance(value, Mapping):
        return "—"
    value = _mapping(value)
    mean = _number(value.get("mean")) * scale
    low = _number(value.get("ci95_low")) * scale
    high = _number(value.get("ci95_high")) * scale
    return f"{mean:.{precision}f}{suffix} [{low:.{precision}f}, {high:.{precision}f}]"


def _nested(value: Mapping[str, object], *path: str) -> object:
    current: object = value
    for part in path:
        if not isinstance(current, Mapping):
            return None
        current = _mapping(current).get(part)
    return current


def _markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _mapping_rows(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        raise TypeError("expected a list of mappings")
    if not all(isinstance(row, Mapping) for row in value):
        raise TypeError("expected a list of mappings")
    return cast(list[Mapping[str, object]], value)


def _source_artifact(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
    }


def _optional_source_artifact(root: Path, path: Path) -> dict[str, str]:
    if path.is_file():
        return _source_artifact(root, path)
    return {"path": path.relative_to(root).as_posix()}


def _figure_records(
    paths: Sequence[Path], document_parent: Path
) -> list[dict[str, str]]:
    grouped: dict[str, dict[str, Path]] = defaultdict(dict)
    for path in paths:
        file_format = path.suffix.removeprefix(".").lower()
        if file_format not in ("png", "pdf"):
            raise ValueError(f"unsupported report figure format: {path}")
        if file_format in grouped[path.stem]:
            raise ValueError(f"duplicate report figure format: {path}")
        grouped[path.stem][file_format] = path
    records = []
    for name in sorted(grouped):
        formats = grouped[name]
        if set(formats) != {"png", "pdf"}:
            raise ValueError(f"report figure lacks PNG/PDF pair: {name}")
        records.append(
            {
                "name": name,
                "png": _relative_path(formats["png"], document_parent),
                "pdf": _relative_path(formats["pdf"], document_parent),
            }
        )
    return records


def _relative_path(path: Path, parent: Path) -> str:
    return Path(os.path.relpath(path.resolve(), parent.resolve())).as_posix()


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as file:
            file.write(value)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError("expected a list of strings")
    return cast(list[str], value)


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("expected a mapping")
    return cast(Mapping[str, Any], value)


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("expected a string")
    return value


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("expected an integer")
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("expected a number")
    return float(value)
