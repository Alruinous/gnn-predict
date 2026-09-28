from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricDefinition:
    prometheus_name: str
    output_prefix: str
    scale_factor: float = 1.0
    required: bool = True


GPU_METRIC_DEFINITIONS: tuple[MetricDefinition, ...] = (
    MetricDefinition("DCGM_FI_DEV_GPU_UTIL", "gpu_util_percent"),
    MetricDefinition("DCGM_FI_PROF_SM_ACTIVE", "gpu_sm_active_percent", 100.0),
    MetricDefinition("DCGM_FI_PROF_SM_OCCUPANCY", "gpu_sm_occupancy_percent", 100.0),
    MetricDefinition("DCGM_FI_DEV_FB_USED", "gpu_mem_used_mb"),
    MetricDefinition("DCGM_FI_DEV_FB_FREE", "gpu_mem_free_mb"),
    MetricDefinition("DCGM_FI_DEV_MEM_COPY_UTIL", "gpu_mem_copy_util_percent"),
    MetricDefinition("DCGM_FI_PROF_DRAM_ACTIVE", "gpu_dram_active_percent"),
    MetricDefinition(
        "DCGM_FI_PROF_PCIE_TX_BYTES",
        "gpu_pcie_tx_mb_per_sec",
        1 / (1024**2),
    ),
    MetricDefinition(
        "DCGM_FI_PROF_PCIE_RX_BYTES",
        "gpu_pcie_rx_mb_per_sec",
        1 / (1024**2),
    ),
    MetricDefinition("DCGM_FI_DEV_POWER_USAGE", "gpu_power_watts"),
    MetricDefinition("DCGM_FI_DEV_GPU_TEMP", "gpu_temp_celsius"),
)


# IX exporter uses node_name/name/uuid rather than DCGM's pod/device labels.
# ix_sm_utilization is intentionally not mapped to NVIDIA SM active or occupancy.
IX_GPU_METRIC_DEFINITIONS: tuple[MetricDefinition, ...] = (
    MetricDefinition("ix_gpu_utilization", "gpu_util_percent"),
    MetricDefinition("ix_sm_utilization", "gpu_sm_util_percent", required=False),
    MetricDefinition("ix_mem_used", "gpu_mem_used_mb"),
    MetricDefinition("ix_mem_free", "gpu_mem_free_mb", required=False),
    MetricDefinition(
        "ix_pcie_tx_throughput", "gpu_pcie_tx_mb_per_sec", 1 / 1024, False
    ),
    MetricDefinition(
        "ix_pcie_rx_throughput", "gpu_pcie_rx_mb_per_sec", 1 / 1024, False
    ),
    MetricDefinition("ix_power_usage", "gpu_power_watts", required=False),
    MetricDefinition("ix_temperature", "gpu_temp_celsius", required=False),
)


def build_node_cpu_total_query(node_name: str) -> str:
    return f'max(machine_cpu_cores{{node="{_escape_label_value(node_name)}"}})'


def build_node_memory_total_query(node_name: str) -> str:
    return f'max(machine_memory_bytes{{node="{_escape_label_value(node_name)}"}})'


def build_container_start_time_query(pod_name: str, namespace: str) -> str:
    return (
        "min(container_start_time_seconds"
        f'{{pod="{_escape_label_value(pod_name)}",'
        f'namespace="{_escape_label_value(namespace)}",'
        'container!="",container!="POD"})'
    )


def build_pod_cpu_query(pod_name: str, namespace: str, cpu_rate_window: str) -> str:
    return (
        "sum(max by (pod,namespace,node,container) "
        "(rate(container_cpu_usage_seconds_total"
        f'{{pod="{_escape_label_value(pod_name)}",'
        f'namespace="{_escape_label_value(namespace)}",'
        'container!="",container!="POD"}'
        f"[{cpu_rate_window}])))"
    )


def build_pod_memory_query(pod_name: str, namespace: str) -> str:
    return (
        "sum(max by (pod,namespace,node,container) "
        "(container_memory_working_set_bytes"
        f'{{pod="{_escape_label_value(pod_name)}",'
        f'namespace="{_escape_label_value(namespace)}",'
        'container!="",container!="POD"}))'
    )


def build_pod_info_query(pod_name: str, namespace: str) -> str:
    return (
        "count by (pod,namespace,node) "
        "(kube_pod_info"
        f'{{pod="{_escape_label_value(pod_name)}",'
        f'namespace="{_escape_label_value(namespace)}"}})'
    )


def build_gpu_metrics_query(
    metric_names: tuple[str, ...],
    pod_name: str,
    namespace: str,
    gpu_id: str,
) -> str:
    escaped_metric_names = "|".join(
        _escape_regex_value(metric_name) for metric_name in metric_names
    )
    return (
        "max by (__name__,pod,namespace,gpu,device,Hostname) "
        "({"
        f'__name__=~"{escaped_metric_names}",'
        f'pod="{_escape_label_value(pod_name)}",'
        f'namespace="{_escape_label_value(namespace)}",'
        f'gpu="{_escape_label_value(gpu_id)}"'
        "})"
    )


def build_ix_gpu_metrics_query(
    metric_names: tuple[str, ...],
    node_name: str,
    gpu_id: str,
    gpu_uuid: str | None = None,
) -> str:
    escaped_metric_names = "|".join(
        _escape_regex_value(metric_name) for metric_name in metric_names
    )
    uuid_matcher = (
        f',uuid="{_escape_label_value(gpu_uuid)}"' if gpu_uuid is not None else ""
    )
    return (
        "max by (__name__,gpu,uuid,name,node_name) "
        "({"
        f'__name__=~"{escaped_metric_names}",'
        f'node_name="{_escape_label_value(node_name)}",'
        f'gpu="{_escape_label_value(gpu_id)}"'
        f"{uuid_matcher}"
        "})"
    )


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _escape_regex_value(value: str) -> str:
    return _escape_label_value(value).replace(".", "\\.")
