from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from gnn_archs.monitoring import (
    CSV_COLUMNS,
    GPU_METRIC_DEFINITIONS,
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


def test_load_monitor_settings_validates_and_resolves_paths(tmp_path: Path) -> None:
    result_json = tmp_path / "results.json"
    result_json.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "monitor.yaml"
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
            ]
        ),
        encoding="utf-8",
    )

    settings = load_monitor_settings(config_path)

    assert settings.prometheus_url == "http://example:9090"
    assert settings.namespace == "crater-workspace"
    assert settings.cpu_rate_window == "2m"
    assert settings.query_step_seconds == 3
    assert len(settings.targets) == 1
    assert settings.targets[0].result_json == result_json.resolve()
    assert settings.targets[0].output_csv == result_json.with_name("results_monitor.csv")


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
    )

    assert document.gpu_node == "v100"
    assert [record.phase for record in records] == ["training", "inference"]
    assert records[0].duration_sec == 2.0
    assert records[1].duration_sec == 0.5


def test_extract_phase_records_fails_when_phase_timing_missing(tmp_path: Path) -> None:
    result_json = _write_result_document(tmp_path, missing_training_timing=True)

    with pytest.raises(ValueError, match="training timings missing started_at_ts"):
        extract_phase_records(
            "bert_large",
            result_json,
            namespace="crater-workspace",
            node_name="dell-67",
            pod_name="sg-wangjh-260331-bbed6-default0-0",
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

    dataframe = monitor_target(settings, target, client)

    assert list(dataframe.columns) == CSV_COLUMNS
    assert dataframe.shape[0] == 2
    assert target.output_csv.exists()
    loaded = pd.read_csv(target.output_csv)
    assert loaded.shape[0] == 2
    assert set(loaded["phase"]) == {"training", "inference"}
    assert set(loaded["resolved_gpu_label"].astype(str)) == {"1"}
    assert set(loaded["resolved_device_label"]) == {"nvidia1"}
    assert loaded["sample_count"].tolist() == [2, 2]
    assert loaded["gpu_ids"].tolist() == ["[0]", "[0]"]


def test_monitor_target_keeps_zero_value_samples(tmp_path: Path) -> None:
    result_json = _write_result_document(tmp_path)
    settings, target = _build_settings(tmp_path, result_json)
    client = _build_fake_client(
        namespace=settings.namespace,
        node_name=target.node_name,
        pod_name=target.pod_name,
        zero_gpu_values=True,
    )

    dataframe = monitor_target(settings, target, client)

    assert dataframe["gpu_util_percent_avg"].tolist() == [0.0, 0.0]
    assert dataframe["gpu_power_watts_avg"].tolist() == [0.0, 0.0]


def test_monitor_target_fails_when_result_file_is_missing(tmp_path: Path) -> None:
    missing_result = tmp_path / "missing.json"
    settings, target = _build_settings(tmp_path, missing_result)
    client = _build_fake_client(
        namespace=settings.namespace,
        node_name=target.node_name,
        pod_name=target.pod_name,
    )

    with pytest.raises(FileNotFoundError, match="result document does not exist"):
        monitor_target(settings, target, client)


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
        monitor_target(settings, target, client)


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
        monitor_target(settings, target, client)


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
    )
    multi_gpu_response = client.range_responses[gpu_query][0]
    client.range_responses[gpu_query][0] = multi_gpu_response + [
        {
            "metric": {
                "__name__": GPU_METRIC_DEFINITIONS[0].prometheus_name,
                "gpu": "2",
                "device": "nvidia2",
                "pod": target.pod_name,
                "namespace": settings.namespace,
                "Hostname": target.node_name,
            },
            "values": [[100.0, "10"], [101.0, "10"]],
        }
    ]

    with pytest.raises(ValueError, match="expected exactly one GPU series"):
        monitor_target(settings, target, client)


def test_query_builders_lock_expected_cpu_and_memory_promql() -> None:
    assert build_node_cpu_total_query("dell-67") == 'max(machine_cpu_cores{node="dell-67"})'
    assert build_node_memory_total_query("dell-67") == (
        'max(machine_memory_bytes{node="dell-67"})'
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


def _build_settings(
    tmp_path: Path,
    result_json: Path,
) -> tuple[ResolvedMonitorSettings, ResolvedMonitorTarget]:
    output_csv = tmp_path / "out" / "monitor.csv"
    target = ResolvedMonitorTarget(
        name="bert_large",
        node_name="dell-67",
        pod_name="sg-wangjh-260331-bbed6-default0-0",
        result_json=result_json,
        output_csv=output_csv,
    )
    settings = ResolvedMonitorSettings(
        config_path=tmp_path / "monitor.yaml",
        prometheus_url="http://example:9090",
        namespace="crater-workspace",
        cpu_rate_window="2m",
        query_step_seconds=1,
        targets=(target,),
    )
    return settings, target


def _write_result_document(
    tmp_path: Path,
    *,
    missing_training_timing: bool = False,
) -> Path:
    training_timings = (
        TimeWindow()
        if missing_training_timing
        else TimeWindow(
            started_at_ts=100.0,
            ended_at_ts=102.0,
            started_at_text="2026-03-31T13:27:13+00:00",
            ended_at_text="2026-03-31T13:27:15+00:00",
        )
    )
    inference_timings = TimeWindow(
        started_at_ts=103.0,
        ended_at_ts=103.5,
        started_at_text="2026-03-31T13:27:16+00:00",
        ended_at_text="2026-03-31T13:27:16.500000+00:00",
    )
    variant = VariantResult(
        name="bert-large-cased_ic1_oc2_no_mutations_large",
        base_model_name="bert-large-cased",
        base_model_pretrained=False,
        source="single_variant_define",
        group_total_variants_defined=1,
        variant_config={},
        mutations=[],
        training=TrainingResult(timings=training_timings),
        inference=InferenceResult(timings=inference_timings),
        metadata={"gpu_node": "v100"},
    )
    document = ResultDocument(
        config_path="/tmp/bert_large_variants.yaml",
        gpu_node="v100",
        gpu_ids=[0],
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
    zero_gpu_values: bool = False,
) -> FakePrometheusClient:
    pod_info_query = build_pod_info_query(pod_name, namespace)
    node_cpu_query = build_node_cpu_total_query(node_name)
    node_memory_query = build_node_memory_total_query(node_name)
    cpu_query = build_pod_cpu_query(pod_name, namespace, "2m")
    memory_query = build_pod_memory_query(pod_name, namespace)
    gpu_query = build_gpu_metrics_query(
        tuple(definition.prometheus_name for definition in GPU_METRIC_DEFINITIONS),
        pod_name,
        namespace,
    )

    gpu_training_values = ["0", "0"] if zero_gpu_values else ["10", "20"]
    gpu_inference_values = ["0", "0"] if zero_gpu_values else ["30", "40"]

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
        },
        range_responses={
            cpu_query: [
                [{"metric": {}, "values": [[100.0, "1.0"], [101.0, "2.0"]]}],
                [{"metric": {}, "values": [[103.0, "0.5"], [103.5, "0.25"]]}],
            ],
            memory_query: [
                [
                    {
                        "metric": {},
                        "values": [
                            [100.0, str(2 * 1024**3)],
                            [101.0, str(4 * 1024**3)],
                        ],
                    }
                ],
                [
                    {
                        "metric": {},
                        "values": [
                            [103.0, str(1 * 1024**3)],
                            [103.5, str(1 * 1024**3)],
                        ],
                    }
                ],
            ],
            gpu_query: [
                _build_gpu_response(
                    pod_name=pod_name,
                    namespace=namespace,
                    node_name=node_name,
                    values=gpu_training_values,
                ),
                _build_gpu_response(
                    pod_name=pod_name,
                    namespace=namespace,
                    node_name=node_name,
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
    values: list[str],
) -> list[dict[str, Any]]:
    response: list[dict[str, Any]] = []
    for definition in GPU_METRIC_DEFINITIONS:
        response.append(
            {
                "metric": {
                    "__name__": definition.prometheus_name,
                    "pod": pod_name,
                    "namespace": namespace,
                    "gpu": "1",
                    "device": "nvidia1",
                    "Hostname": node_name,
                },
                "values": [[100.0, values[0]], [101.0, values[1]]],
            }
        )
    return response
