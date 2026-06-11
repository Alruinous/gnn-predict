from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

import gnn_archs.monitoring.service as monitoring_service
from gnn_archs.monitoring import (
    CSV_COLUMNS,
    GPU_METRIC_DEFINITIONS,
    build_container_start_time_query,
    build_gpu_metrics_query,
    build_node_cpu_total_query,
    build_node_memory_total_query,
    build_pod_cpu_query,
    build_pod_info_query,
    build_pod_memory_query,
    extract_phase_records,
    load_monitor_settings,
    monitor_target,
)
from gnn_archs.monitoring.config import ResolvedMonitorSettings, ResolvedMonitorTarget
from gnn_archs.result import (
    InferenceResult,
    ResultDocument,
    TimeWindow,
    TrainingResult,
    VariantResult,
    write_result_document,
)

TEST_LOGGER = logging.getLogger(__name__)


class FakePrometheusClient:
    def __init__(
        self,
        *,
        instant_responses: dict[str, list[dict[str, Any]]] | None = None,
        range_responses: dict[str, list[list[dict[str, Any]]]] | None = None,
    ) -> None:
        self.instant_responses = instant_responses or {}
        self.range_responses = range_responses or {}
        self.instant_calls: list[tuple[str, float | None]] = []
        self.range_calls: list[tuple[str, float, float, int]] = []

    def instant_query(
        self, query: str, timestamp: float | None = None
    ) -> list[dict[str, Any]]:
        self.instant_calls.append((query, timestamp))
        if query not in self.instant_responses:
            raise AssertionError(f"unexpected instant query: {query}")
        return self.instant_responses[query]

    def range_query(
        self,
        query: str,
        start_ts: float,
        end_ts: float,
        step_seconds: int,
    ) -> list[dict[str, Any]]:
        self.range_calls.append((query, start_ts, end_ts, step_seconds))
        queue = self.range_responses.get(query)
        if not queue:
            raise AssertionError(f"unexpected range query: {query}")
        return queue.pop(0)


def test_load_monitor_settings_keeps_relative_paths_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result_json = tmp_path / "results.json"
    result_json.write_text("{}", encoding="utf-8")
    config_path = Path("monitor.yaml")
    config_path.write_text(
        "\n".join(
            [
                "defaults:",
                '  prometheus_url: "http://example:9090"',
                '  namespace: "crater-workspace"',
                '  cpu_rate_window: "2m"',
                "  query_step_seconds: 3",
                "targets:",
                "  bert_large:",
                "    enabled: true",
                f'    result_json: "{result_json.name}"',
                '    node_name: "dell-67"',
                '    pod_name: "sg-wangjh-260331-bbed6-default0-0"',
                '    gpu_id: "1"',
            ]
        ),
        encoding="utf-8",
    )

    settings = load_monitor_settings(config_path)

    assert settings.prometheus_url == "http://example:9090"
    assert settings.namespace == "crater-workspace"
    assert settings.cpu_rate_window == "2m"
    assert settings.cpu_rate_window_seconds == 120.0
    assert settings.query_step_seconds == 3
    assert settings.memory_baseline_window_seconds == 5
    assert len(settings.targets) == 1
    assert settings.targets[0].gpu_id == "1"
    assert settings.config_path == Path("monitor.yaml")
    assert settings.targets[0].result_json == Path("results.json")
    assert settings.targets[0].output_csv == Path("results_monitor.csv")


def test_load_monitor_settings_keeps_nested_config_relative_paths_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = Path("config/monitor/monitor.yaml")
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "\n".join(
            [
                "defaults:",
                '  prometheus_url: "http://example:9090"',
                '  namespace: "crater-workspace"',
                "targets:",
                "  vit:",
                "    enabled: true",
                '    result_json: "res/vit/results/results.json"',
                '    node_name: "dell-67"',
                '    pod_name: "pod-vit"',
                '    gpu_id: "1"',
                '    output_csv: "csv/vit_monitor.csv"',
            ]
        ),
        encoding="utf-8",
    )

    settings = load_monitor_settings(config_path)

    assert settings.config_path == Path("config/monitor/monitor.yaml")
    assert settings.targets[0].result_json == Path("res/vit/results/results.json")
    assert settings.targets[0].output_csv == Path("csv/vit_monitor.csv")


def test_load_monitor_settings_without_target_filter_keeps_all_enabled_targets(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "monitor.yaml"
    config_path.write_text(
        "\n".join(
            [
                "defaults:",
                '  prometheus_url: "http://example:9090"',
                '  namespace: "crater-workspace"',
                "targets:",
                "  bert_large:",
                "    enabled: true",
                '    result_json: "bert.json"',
                '    node_name: "dell-67"',
                '    pod_name: "pod-bert"',
                '    gpu_id: "1"',
                "  disabled_target:",
                "    enabled: false",
                '    result_json: "disabled.json"',
                '    node_name: "dell-67"',
                '    pod_name: "pod-disabled"',
                '    gpu_id: "2"',
                "  resnet50:",
                "    enabled: true",
                '    result_json: "resnet.json"',
                '    node_name: "dell-68"',
                '    pod_name: "pod-resnet"',
                '    gpu_id: "0"',
            ]
        ),
        encoding="utf-8",
    )

    settings = load_monitor_settings(config_path)

    assert [target.name for target in settings.targets] == ["bert_large", "resnet50"]


def test_load_monitor_settings_filters_targets_in_requested_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    absolute_result = (tmp_path / "bert.json").resolve()
    config_path = Path("monitor.yaml")
    config_path.write_text(
        "\n".join(
            [
                "defaults:",
                '  prometheus_url: "http://example:9090"',
                '  namespace: "crater-workspace"',
                "targets:",
                "  bert_large:",
                "    enabled: true",
                f'    result_json: "{absolute_result}"',
                '    node_name: "dell-67"',
                '    pod_name: "pod-bert"',
                '    gpu_id: "1"',
                "  resnet50:",
                "    enabled: true",
                '    result_json: "resnet.json"',
                '    node_name: "dell-68"',
                '    pod_name: "pod-resnet"',
                '    gpu_id: "0"',
            ]
        ),
        encoding="utf-8",
    )

    settings = load_monitor_settings(
        config_path,
        target_names=("resnet50", "bert_large"),
    )

    assert [target.name for target in settings.targets] == ["resnet50", "bert_large"]
    assert settings.targets[0].result_json == Path("resnet.json")
    assert settings.targets[1].result_json == absolute_result


def test_load_monitor_settings_rejects_unknown_requested_targets(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "monitor.yaml"
    config_path.write_text(
        "\n".join(
            [
                "defaults:",
                '  prometheus_url: "http://example:9090"',
                '  namespace: "crater-workspace"',
                "targets:",
                "  bert_large:",
                "    enabled: true",
                '    result_json: "bert.json"',
                '    node_name: "dell-67"',
                '    pod_name: "pod-bert"',
                '    gpu_id: "1"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="unknown: missing_target",
    ):
        load_monitor_settings(config_path, target_names=("missing_target",))


def test_load_monitor_settings_rejects_disabled_requested_targets(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "monitor.yaml"
    config_path.write_text(
        "\n".join(
            [
                "defaults:",
                '  prometheus_url: "http://example:9090"',
                '  namespace: "crater-workspace"',
                "targets:",
                "  bert_large:",
                "    enabled: false",
                '    result_json: "bert.json"',
                '    node_name: "dell-67"',
                '    pod_name: "pod-bert"',
                '    gpu_id: "1"',
                "  resnet50:",
                "    enabled: true",
                '    result_json: "resnet.json"',
                '    node_name: "dell-68"',
                '    pod_name: "pod-resnet"',
                '    gpu_id: "0"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="disabled: bert_large",
    ):
        load_monitor_settings(config_path, target_names=("bert_large",))


def test_load_monitor_settings_requires_gpu_id(tmp_path: Path) -> None:
    config_path = tmp_path / "monitor.yaml"
    config_path.write_text(
        "\n".join(
            [
                "defaults:",
                '  prometheus_url: "http://example:9090"',
                '  namespace: "crater-workspace"',
                "targets:",
                "  bert_large:",
                "    enabled: true",
                '    result_json: "results.json"',
                '    node_name: "dell-67"',
                '    pod_name: "sg-wangjh-260331-bbed6-default0-0"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_monitor_settings(config_path)


def test_load_monitor_settings_rejects_invalid_step(tmp_path: Path) -> None:
    config_path = tmp_path / "monitor.yaml"
    config_path.write_text(
        "\n".join(
            [
                "defaults:",
                '  prometheus_url: "http://example:9090"',
                '  namespace: "crater-workspace"',
                '  cpu_rate_window: "2m"',
                "  query_step_seconds: 0",
                "targets:",
                "  bert_large:",
                "    enabled: true",
                '    result_json: "results.json"',
                '    node_name: "dell-67"',
                '    pod_name: "sg-wangjh-260331-bbed6-default0-0"',
                '    gpu_id: "1"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_monitor_settings(config_path)


def test_load_monitor_settings_rejects_invalid_memory_baseline_window(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "monitor.yaml"
    config_path.write_text(
        "\n".join(
            [
                "defaults:",
                '  prometheus_url: "http://example:9090"',
                '  namespace: "crater-workspace"',
                "  memory_baseline_window_seconds: 0",
                "targets:",
                "  bert_large:",
                "    enabled: true",
                '    result_json: "results.json"',
                '    node_name: "dell-67"',
                '    pod_name: "sg-wangjh-260331-bbed6-default0-0"',
                '    gpu_id: "1"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_monitor_settings(config_path)


def test_extract_phase_records_reads_training_and_inference(tmp_path: Path) -> None:
    result_json = _write_result_document(tmp_path)

    document, records = extract_phase_records(
        "bert_large",
        result_json,
        namespace="crater-workspace",
        node_name="dell-67",
        pod_name="sg-wangjh-260331-bbed6-default0-0",
        gpu_id="1",
    )

    assert document.gpu_node == "v100"
    assert [record.phase for record in records] == ["training", "inference"]
    assert [record.gpu_id for record in records] == ["1", "1"]
    assert records[0].duration_sec == 7.0
    assert records[1].duration_sec == 6.0
    assert [record.deployment_duration_sec_avg for record in records] == [1.5, 1.5]
    assert [record.phase_rounds for record in records] == [3, 42]


def test_extract_phase_records_reads_prefill(tmp_path: Path) -> None:
    variant = VariantResult(
        name="qwen_prefill_smoke",
        base_model_name="qwen3_config",
        base_model_pretrained=False,
        source="variant_config_grid",
        group_total_variants_defined=1,
        variant_config={},
        mutations=[],
        timings={
            "model_build": TimeWindow(
                started_at_ts=118.0,
                ended_at_ts=119.5,
                started_at_text="2026-03-31T13:27:41+00:00",
                ended_at_text="2026-03-31T13:27:42.500000+00:00",
            )
        },
        prefill=InferenceResult(
            metrics={"iterations": 11, "batch_size": 2},
            timings=TimeWindow(
                started_at_ts=120.0,
                ended_at_ts=124.0,
                started_at_text="2026-03-31T13:27:33+00:00",
                ended_at_text="2026-03-31T13:27:37+00:00",
            ),
        ),
        metadata={"gpu_node": "v100"},
    )
    document = ResultDocument(
        config_path="/tmp/qwen3_variants.yaml",
        gpu_node="v100",
        variants=[variant],
        summary={"variant_count": 1, "prefill_count": 1},
    )
    result_json = tmp_path / "results.json"
    write_result_document(result_json, document)

    _, records = extract_phase_records(
        "qwen",
        result_json,
        namespace="crater-workspace",
        node_name="dell-67",
        pod_name="sg-wangjh-260331-bbed6-default0-0",
        gpu_id="1",
    )

    assert [record.phase for record in records] == ["prefill"]
    assert records[0].duration_sec == 4.0
    assert records[0].phase_rounds == 11
    assert records[0].batch_size == 2
    assert records[0].decode_output_length == 0


def test_extract_phase_records_reads_decode(tmp_path: Path) -> None:
    variant = VariantResult(
        name="qwen_decode_smoke",
        base_model_name="qwen3_config",
        base_model_pretrained=False,
        source="variant_config_grid",
        group_total_variants_defined=1,
        variant_config={},
        mutations=[],
        timings={
            "model_build": TimeWindow(
                started_at_ts=118.0,
                ended_at_ts=119.5,
                started_at_text="2026-03-31T13:27:41+00:00",
                ended_at_text="2026-03-31T13:27:42.500000+00:00",
            )
        },
        decode=InferenceResult(
            metrics={
                "iterations": 7,
                "batch_size": 2,
                "decode_max_output_length": 64,
            },
            timings=TimeWindow(
                started_at_ts=130.0,
                ended_at_ts=136.0,
                started_at_text="2026-03-31T13:27:43+00:00",
                ended_at_text="2026-03-31T13:27:49+00:00",
            ),
        ),
        metadata={"gpu_node": "v100"},
    )
    document = ResultDocument(
        config_path="/tmp/qwen3_variants.yaml",
        gpu_node="v100",
        variants=[variant],
        summary={"variant_count": 1, "decode_count": 1},
    )
    result_json = tmp_path / "results.json"
    write_result_document(result_json, document)

    _, records = extract_phase_records(
        "qwen",
        result_json,
        namespace="crater-workspace",
        node_name="dell-67",
        pod_name="sg-wangjh-260331-bbed6-default0-0",
        gpu_id="1",
    )

    assert [record.phase for record in records] == ["decode"]
    assert records[0].duration_sec == 6.0
    assert records[0].phase_rounds == 7
    assert records[0].batch_size == 2
    assert records[0].decode_output_length == 64


def test_extract_phase_records_fails_when_phase_timing_missing(tmp_path: Path) -> None:
    result_json = _write_result_document(tmp_path, missing_training_timing=True)

    with pytest.raises(ValueError, match="training timings missing started_at_ts"):
        extract_phase_records(
            "bert_large",
            result_json,
            namespace="crater-workspace",
            node_name="dell-67",
            pod_name="sg-wangjh-260331-bbed6-default0-0",
            gpu_id="1",
        )


def test_extract_phase_records_fails_when_model_build_timing_missing(
    tmp_path: Path,
) -> None:
    result_json = _write_result_document(tmp_path, include_model_build_timing=False)

    with pytest.raises(ValueError, match="model_build timings missing"):
        extract_phase_records(
            "bert_large",
            result_json,
            namespace="crater-workspace",
            node_name="dell-67",
            pod_name="sg-wangjh-260331-bbed6-default0-0",
            gpu_id="1",
        )


@pytest.mark.parametrize(
    (
        "include_training_total_steps",
        "include_inference_iterations",
        "expected_message",
    ),
    [
        (False, True, "training phase_rounds must be int"),
        (True, False, "inference phase_rounds must be int"),
    ],
)
def test_extract_phase_records_fails_when_phase_rounds_missing(
    tmp_path: Path,
    include_training_total_steps: bool,
    include_inference_iterations: bool,
    expected_message: str,
) -> None:
    result_json = _write_result_document(
        tmp_path,
        include_training_total_steps=include_training_total_steps,
        include_inference_iterations=include_inference_iterations,
    )

    with pytest.raises(AssertionError, match=expected_message):
        extract_phase_records(
            "bert_large",
            result_json,
            namespace="crater-workspace",
            node_name="dell-67",
            pod_name="sg-wangjh-260331-bbed6-default0-0",
            gpu_id="1",
        )


def test_monitor_target_writes_expected_csv_columns_and_rows(tmp_path: Path) -> None:
    result_json = _write_result_document(tmp_path)
    settings, target = _build_settings(tmp_path, result_json)
    client = _build_fake_client(
        namespace=settings.namespace,
        node_name=target.node_name,
        pod_name=target.pod_name,
        zero_gpu_values=False,
    )

    dataframe = monitor_target(settings, target, client, TEST_LOGGER)

    assert list(dataframe.columns) == CSV_COLUMNS
    assert dataframe.shape[0] == 2
    assert target.output_csv.exists()
    loaded = pd.read_csv(target.output_csv)
    assert loaded.shape[0] == 2
    assert set(loaded["phase"]) == {"training", "inference"}
    assert loaded["phase_rounds"].tolist() == [3, 42]
    assert loaded["gpu_id"].astype(str).tolist() == ["1", "1"]
    assert set(loaded["resolved_gpu_label"].astype(str)) == {"1"}
    assert set(loaded["resolved_device_label"]) == {"nvidia1"}
    assert loaded["sample_count"].tolist() == [3, 3]
    assert loaded["decode_output_length"].tolist() == [0, 0]
    assert loaded["deployment_duration_sec_avg"].tolist() == [1.5, 1.5]
    assert loaded["container_started_at_ts"].tolist() == [90.0, 90.0]
    assert loaded["memory_baseline_gb"].tolist() == [0.5, 0.5]
    assert loaded["memory_baseline_sample_count"].tolist() == [3, 3]
    assert loaded["memory_delta_gb_p95"].tolist() == pytest.approx([5.3, 1.4])
    assert loaded["gpu_util_percent_max"].tolist() == pytest.approx([30.0, 50.0])
    assert loaded["gpu_sm_active_percent_max"].tolist() == pytest.approx([30.0, 50.0])
    assert loaded["gpu_sm_occupancy_percent_max"].tolist() == pytest.approx(
        [30.0, 50.0]
    )
    assert dataframe["phase_rounds"].tolist() == [3, 42]


def test_monitor_target_offsets_cpu_queries_by_rate_window(tmp_path: Path) -> None:
    result_json = _write_result_document(tmp_path)
    settings, target = _build_settings(tmp_path, result_json)
    client = _build_fake_client(
        namespace=settings.namespace,
        node_name=target.node_name,
        pod_name=target.pod_name,
    )

    monitor_target(settings, target, client, TEST_LOGGER)

    training_cpu_call = client.range_calls[1]
    inference_cpu_call = client.range_calls[4]
    assert training_cpu_call[1:] == (103.0, 107.0, 1)
    assert inference_cpu_call[1:] == (113.0, 116.0, 1)


def test_monitor_target_keeps_zero_value_samples(tmp_path: Path) -> None:
    result_json = _write_result_document(tmp_path)
    settings, target = _build_settings(tmp_path, result_json)
    client = _build_fake_client(
        namespace=settings.namespace,
        node_name=target.node_name,
        pod_name=target.pod_name,
        zero_gpu_values=True,
    )

    dataframe = monitor_target(settings, target, client, TEST_LOGGER)

    assert dataframe["gpu_util_percent_avg"].tolist() == [0.0, 0.0]
    assert dataframe["gpu_power_watts_avg"].tolist() == [0.0, 0.0]


def test_monitor_target_keeps_rows_when_memory_baseline_is_missing(
    tmp_path: Path,
) -> None:
    result_json = _write_result_document(tmp_path)
    settings, target = _build_settings(tmp_path, result_json)
    client = _build_fake_client(
        namespace=settings.namespace,
        node_name=target.node_name,
        pod_name=target.pod_name,
    )
    client.instant_responses[
        build_container_start_time_query(target.pod_name, settings.namespace)
    ] = []
    memory_query = build_pod_memory_query(target.pod_name, settings.namespace)
    client.range_responses[memory_query].pop(0)

    dataframe = monitor_target(settings, target, client, TEST_LOGGER)

    assert dataframe.shape[0] == 2
    assert dataframe["memory_gb_p95"].tolist() == pytest.approx([5.8, 1.9])
    assert dataframe["container_started_at_ts"].isna().all()
    assert dataframe["memory_baseline_gb"].isna().all()
    assert dataframe["memory_baseline_sample_count"].tolist() == [0, 0]
    assert dataframe["memory_delta_gb_p95"].isna().all()


def test_monitor_target_keeps_rows_when_memory_baseline_samples_are_missing(
    tmp_path: Path,
) -> None:
    result_json = _write_result_document(tmp_path)
    settings, target = _build_settings(tmp_path, result_json)
    client = _build_fake_client(
        namespace=settings.namespace,
        node_name=target.node_name,
        pod_name=target.pod_name,
    )
    memory_query = build_pod_memory_query(target.pod_name, settings.namespace)
    client.range_responses[memory_query][0] = []

    dataframe = monitor_target(settings, target, client, TEST_LOGGER)

    assert dataframe.shape[0] == 2
    assert dataframe["container_started_at_ts"].tolist() == [90.0, 90.0]
    assert dataframe["memory_baseline_gb"].isna().all()
    assert dataframe["memory_baseline_sample_count"].tolist() == [0, 0]
    assert dataframe["memory_delta_gb_p95"].isna().all()


def test_monitor_target_clamps_negative_memory_delta(tmp_path: Path) -> None:
    result_json = _write_result_document(tmp_path)
    settings, target = _build_settings(tmp_path, result_json)
    client = _build_fake_client(
        namespace=settings.namespace,
        node_name=target.node_name,
        pod_name=target.pod_name,
    )
    memory_query = build_pod_memory_query(target.pod_name, settings.namespace)
    client.range_responses[memory_query][0] = [
        {
            "metric": {},
            "values": [[90.0, str(8 * 1024**3)]],
        }
    ]

    dataframe = monitor_target(settings, target, client, TEST_LOGGER)

    assert dataframe["memory_baseline_gb"].tolist() == [8.0, 8.0]
    assert dataframe["memory_delta_gb_p95"].tolist() == [0.0, 0.0]


def test_monitor_target_fails_when_result_file_is_missing(tmp_path: Path) -> None:
    missing_result = tmp_path / "missing.json"
    settings, target = _build_settings(tmp_path, missing_result)
    client = _build_fake_client(
        namespace=settings.namespace,
        node_name=target.node_name,
        pod_name=target.pod_name,
    )

    with pytest.raises(FileNotFoundError, match="result document does not exist"):
        monitor_target(settings, target, client, TEST_LOGGER)


def test_monitor_target_fails_when_node_total_is_missing(tmp_path: Path) -> None:
    result_json = _write_result_document(tmp_path)
    settings, target = _build_settings(tmp_path, result_json)
    client = _build_fake_client(
        namespace=settings.namespace,
        node_name=target.node_name,
        pod_name=target.pod_name,
    )
    client.instant_responses[build_node_cpu_total_query(target.node_name)] = []

    with pytest.raises(ValueError, match="expected exactly one series for node CPU total"):
        monitor_target(settings, target, client, TEST_LOGGER)


def test_monitor_target_fails_when_metric_samples_are_missing(tmp_path: Path) -> None:
    result_json = _write_result_document(tmp_path)
    settings, target = _build_settings(tmp_path, result_json)
    client = _build_fake_client(
        namespace=settings.namespace,
        node_name=target.node_name,
        pod_name=target.pod_name,
    )
    cpu_query = build_pod_cpu_query(
        target.pod_name,
        settings.namespace,
        settings.cpu_rate_window,
    )
    client.range_responses[cpu_query][0] = [{"metric": {}, "values": []}]

    with pytest.raises(ValueError, match="no samples returned for CPU usage"):
        monitor_target(settings, target, client, TEST_LOGGER)


def test_monitor_target_clamps_cpu_queries_when_phase_is_shorter_than_cpu_window(
    tmp_path: Path,
) -> None:
    result_json = _write_result_document(tmp_path)
    _, target = _build_settings(tmp_path, result_json)
    settings = ResolvedMonitorSettings(
        config_path=tmp_path / "monitor.yaml",
        prometheus_url="http://example:9090",
        namespace="crater-workspace",
        cpu_rate_window="8s",
        cpu_rate_window_seconds=8.0,
        query_step_seconds=1,
        targets=(target,),
    )
    client = _build_fake_client(
        namespace=settings.namespace,
        node_name=target.node_name,
        pod_name=target.pod_name,
        cpu_rate_window=settings.cpu_rate_window,
    )

    monitor_target(settings, target, client, TEST_LOGGER)

    training_cpu_call = client.range_calls[1]
    inference_cpu_call = client.range_calls[4]
    assert training_cpu_call[1:] == (107.0, 107.0, 1)
    assert inference_cpu_call[1:] == (116.0, 116.0, 1)


def test_monitor_target_fails_when_sample_count_is_too_small(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_json = _write_result_document(tmp_path)
    settings, target = _build_settings(tmp_path, result_json)
    client = _build_fake_client(
        namespace=settings.namespace,
        node_name=target.node_name,
        pod_name=target.pod_name,
    )
    monkeypatch.setattr(monitoring_service, "MIN_REQUIRED_PHASE_SAMPLES", 3)
    memory_query = build_pod_memory_query(target.pod_name, settings.namespace)
    client.range_responses[memory_query][1] = [
        {
            "metric": {},
            "values": [
                [100.0, str(2 * 1024**3)],
                [101.0, str(4 * 1024**3)],
            ],
        }
    ]

    with pytest.raises(ValueError, match="has too few samples"):
        monitor_target(settings, target, client, TEST_LOGGER)


def test_monitor_target_fails_when_configured_gpu_has_no_series(
    tmp_path: Path,
) -> None:
    result_json = _write_result_document(tmp_path)
    settings, target = _build_settings(tmp_path, result_json)
    client = _build_fake_client(
        namespace=settings.namespace,
        node_name=target.node_name,
        pod_name=target.pod_name,
        gpu_id=target.gpu_id,
    )
    gpu_query = build_gpu_metrics_query(
        tuple(definition.prometheus_name for definition in GPU_METRIC_DEFINITIONS),
        target.pod_name,
        settings.namespace,
        target.gpu_id,
    )
    client.range_responses[gpu_query][0] = []

    with pytest.raises(ValueError, match="expected exactly one GPU series"):
        monitor_target(settings, target, client, TEST_LOGGER)


def test_monitor_target_fails_when_multiple_gpu_series_are_returned(
    tmp_path: Path,
) -> None:
    result_json = _write_result_document(tmp_path)
    settings, target = _build_settings(tmp_path, result_json)
    client = _build_fake_client(
        namespace=settings.namespace,
        node_name=target.node_name,
        pod_name=target.pod_name,
    )
    gpu_query = build_gpu_metrics_query(
        tuple(definition.prometheus_name for definition in GPU_METRIC_DEFINITIONS),
        target.pod_name,
        settings.namespace,
        target.gpu_id,
    )
    multi_gpu_response = client.range_responses[gpu_query][0]
    client.range_responses[gpu_query][0] = multi_gpu_response + [
        {
            "metric": {
                "__name__": GPU_METRIC_DEFINITIONS[0].prometheus_name,
                "gpu": target.gpu_id,
                "device": "nvidia-duplicate",
                "pod": target.pod_name,
                "namespace": settings.namespace,
                "Hostname": target.node_name,
            },
            "values": [[100.0, "10"], [101.0, "10"]],
        }
    ]

    with pytest.raises(ValueError, match="expected exactly one GPU series"):
        monitor_target(settings, target, client, TEST_LOGGER)


def test_query_builders_lock_expected_cpu_and_memory_promql() -> None:
    assert build_node_cpu_total_query("dell-67") == 'max(machine_cpu_cores{node="dell-67"})'
    assert build_node_memory_total_query("dell-67") == (
        'max(machine_memory_bytes{node="dell-67"})'
    )
    assert build_container_start_time_query("pod-a", "ns-a") == (
        'min(container_start_time_seconds{pod="pod-a",namespace="ns-a",'
        'container!="",container!="POD"})'
    )
    assert build_pod_cpu_query("pod-a", "ns-a", "2m") == (
        'sum(max by (pod,namespace,node,container) '
        '(rate(container_cpu_usage_seconds_total{pod="pod-a",namespace="ns-a",'
        'container!="",container!="POD"}[2m])))'
    )
    assert build_pod_memory_query("pod-a", "ns-a") == (
        'sum(max by (pod,namespace,node,container) '
        '(container_memory_working_set_bytes{pod="pod-a",namespace="ns-a",'
        'container!="",container!="POD"}))'
    )
    assert build_pod_info_query("pod-a", "ns-a") == (
        'count by (pod,namespace,node) '
        '(kube_pod_info{pod="pod-a",namespace="ns-a"})'
    )
    assert build_gpu_metrics_query(
        tuple(definition.prometheus_name for definition in GPU_METRIC_DEFINITIONS),
        "pod-a",
        "ns-a",
        "1",
    ) == (
        'max by (__name__,pod,namespace,gpu,device,Hostname) '
        '({__name__=~"DCGM_FI_DEV_GPU_UTIL|DCGM_FI_PROF_SM_ACTIVE|'
        'DCGM_FI_PROF_SM_OCCUPANCY|DCGM_FI_DEV_FB_USED|DCGM_FI_DEV_FB_FREE|'
        'DCGM_FI_DEV_MEM_COPY_UTIL|DCGM_FI_PROF_DRAM_ACTIVE|'
        'DCGM_FI_PROF_PCIE_TX_BYTES|DCGM_FI_PROF_PCIE_RX_BYTES|'
        'DCGM_FI_DEV_POWER_USAGE|DCGM_FI_DEV_GPU_TEMP",pod="pod-a",'
        'namespace="ns-a",gpu="1"})'
    )


def _build_settings(
    tmp_path: Path,
    result_json: Path,
    *,
    gpu_id: str = "1",
) -> tuple[ResolvedMonitorSettings, ResolvedMonitorTarget]:
    output_csv = tmp_path / "out" / "monitor.csv"
    target = ResolvedMonitorTarget(
        name="bert_large",
        node_name="dell-67",
        pod_name="sg-wangjh-260331-bbed6-default0-0",
        gpu_id=gpu_id,
        result_json=result_json,
        output_csv=output_csv,
    )
    settings = ResolvedMonitorSettings(
        config_path=tmp_path / "monitor.yaml",
        prometheus_url="http://example:9090",
        namespace="crater-workspace",
        cpu_rate_window="3s",
        cpu_rate_window_seconds=3.0,
        query_step_seconds=1,
        targets=(target,),
    )
    return settings, target


def _write_result_document(
    tmp_path: Path,
    *,
    missing_training_timing: bool = False,
    include_training_total_steps: bool = True,
    include_inference_iterations: bool = True,
    include_model_build_timing: bool = True,
) -> Path:
    training_timings = (
        TimeWindow()
        if missing_training_timing
        else TimeWindow(
            started_at_ts=100.0,
            ended_at_ts=107.0,
            started_at_text="2026-03-31T13:27:13+00:00",
            ended_at_text="2026-03-31T13:27:20+00:00",
        )
    )
    inference_timings = TimeWindow(
        started_at_ts=110.0,
        ended_at_ts=116.0,
        started_at_text="2026-03-31T13:27:23+00:00",
        ended_at_text="2026-03-31T13:27:29+00:00",
    )
    training_metrics = {"total_steps": 3} if include_training_total_steps else {}
    inference_metrics = {"batch_size": 8}
    if include_inference_iterations:
        inference_metrics["iterations"] = 42
    timings = (
        {
            "model_build": TimeWindow(
                started_at_ts=98.0,
                ended_at_ts=99.5,
                started_at_text="2026-03-31T13:27:11+00:00",
                ended_at_text="2026-03-31T13:27:12.500000+00:00",
            )
        }
        if include_model_build_timing
        else {}
    )
    variant = VariantResult(
        name="bert-large-cased_ic1_oc2_no_mutations_large",
        base_model_name="bert-large-cased",
        base_model_pretrained=False,
        source="single_variant_define",
        group_total_variants_defined=1,
        variant_config={},
        mutations=[],
        timings=timings,
        training=TrainingResult(
            hyperparameters={"batch_size": 8},
            metrics=training_metrics,
            timings=training_timings,
        ),
        inference=InferenceResult(
            metrics=inference_metrics,
            timings=inference_timings,
        ),
        metadata={"gpu_node": "v100"},
    )
    document = ResultDocument(
        config_path="/tmp/bert_large_variants.yaml",
        gpu_node="v100",
        variants=[variant],
        summary={"variant_count": 1, "training_count": 1, "inference_count": 1},
    )
    result_json = tmp_path / "results.json"
    write_result_document(result_json, document)
    return result_json


def _build_fake_client(
    *,
    namespace: str,
    node_name: str,
    pod_name: str,
    gpu_id: str = "1",
    cpu_rate_window: str = "3s",
    zero_gpu_values: bool = False,
) -> FakePrometheusClient:
    pod_info_query = build_pod_info_query(pod_name, namespace)
    container_start_query = build_container_start_time_query(pod_name, namespace)
    node_cpu_query = build_node_cpu_total_query(node_name)
    node_memory_query = build_node_memory_total_query(node_name)
    cpu_query = build_pod_cpu_query(pod_name, namespace, cpu_rate_window)
    memory_query = build_pod_memory_query(pod_name, namespace)
    gpu_query = build_gpu_metrics_query(
        tuple(definition.prometheus_name for definition in GPU_METRIC_DEFINITIONS),
        pod_name,
        namespace,
        gpu_id,
    )

    gpu_training_values = ["0", "0", "0"] if zero_gpu_values else ["10", "20", "30"]
    gpu_inference_values = ["0", "0", "0"] if zero_gpu_values else ["30", "40", "50"]

    return FakePrometheusClient(
        instant_responses={
            pod_info_query: [
                {
                    "metric": {
                        "pod": pod_name,
                        "namespace": namespace,
                        "node": node_name,
                    },
                    "value": [100.0, "1"],
                }
            ],
            node_cpu_query: [{"metric": {}, "value": [100.0, "80"]}],
            node_memory_query: [
                {"metric": {}, "value": [100.0, str(256 * 1024**3)]}
            ],
            container_start_query: [{"metric": {}, "value": [100.0, "90.0"]}],
        },
        range_responses={
            cpu_query: [
                [
                    {
                        "metric": {},
                        "values": [[103.0, "1.0"], [105.0, "2.0"], [107.0, "3.0"]],
                    }
                ],
                [
                    {
                        "metric": {},
                        "values": [[113.0, "0.5"], [114.0, "0.25"], [116.0, "0.75"]],
                    }
                ],
            ],
            memory_query: [
                [
                    {
                        "metric": {},
                        "values": [
                            [90.0, str(int(0.5 * 1024**3))],
                            [92.0, str(int(0.75 * 1024**3))],
                            [95.0, str(1 * 1024**3)],
                        ],
                    }
                ],
                [
                    {
                        "metric": {},
                        "values": [
                            [100.0, str(2 * 1024**3)],
                            [101.0, str(4 * 1024**3)],
                            [107.0, str(6 * 1024**3)],
                        ],
                    }
                ],
                [
                    {
                        "metric": {},
                        "values": [
                            [110.0, str(1 * 1024**3)],
                            [113.0, str(1 * 1024**3)],
                            [116.0, str(2 * 1024**3)],
                        ],
                    }
                ],
            ],
            gpu_query: [
                _build_gpu_response(
                    pod_name=pod_name,
                    namespace=namespace,
                    node_name=node_name,
                    gpu_id=gpu_id,
                    values=gpu_training_values,
                ),
                _build_gpu_response(
                    pod_name=pod_name,
                    namespace=namespace,
                    node_name=node_name,
                    gpu_id=gpu_id,
                    values=gpu_inference_values,
                ),
            ],
        },
    )


def _build_gpu_response(
    *,
    pod_name: str,
    namespace: str,
    node_name: str,
    gpu_id: str,
    values: list[str],
) -> list[dict[str, Any]]:
    response: list[dict[str, Any]] = []
    for definition in GPU_METRIC_DEFINITIONS:
        metric_values = values
        if definition.prometheus_name in {
            "DCGM_FI_PROF_SM_ACTIVE",
            "DCGM_FI_PROF_SM_OCCUPANCY",
        }:
            metric_values = [str(float(value) / 100.0) for value in values]
        response.append(
            {
                "metric": {
                    "__name__": definition.prometheus_name,
                    "pod": pod_name,
                    "namespace": namespace,
                    "gpu": gpu_id,
                    "device": f"nvidia{gpu_id}",
                    "Hostname": node_name,
                },
                "values": [
                    [100.0, metric_values[0]],
                    [101.0, metric_values[1]],
                    [102.0, metric_values[2]],
                ],
            }
        )
    return response
