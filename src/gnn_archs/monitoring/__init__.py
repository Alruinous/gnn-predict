"""Fail-fast Prometheus monitoring for result documents."""

from .config import (
    MonitorConfig,
    MonitorDefaults,
    MonitorTarget,
    ResolvedMonitorSettings,
    ResolvedMonitorTarget,
    load_monitor_settings,
)
from .queries import (
    GPU_METRIC_DEFINITIONS,
    IX_GPU_METRIC_DEFINITIONS,
    build_container_start_time_query,
    build_gpu_metrics_query,
    build_ix_gpu_metrics_query,
    build_node_cpu_total_query,
    build_node_memory_total_query,
    build_pod_cpu_query,
    build_pod_info_query,
    build_pod_memory_query,
)
from .service import (
    CSV_COLUMNS,
    extract_phase_records,
    monitor_target,
    run_monitoring,
)

__all__ = [
    "CSV_COLUMNS",
    "GPU_METRIC_DEFINITIONS",
    "IX_GPU_METRIC_DEFINITIONS",
    "MonitorConfig",
    "MonitorDefaults",
    "MonitorTarget",
    "ResolvedMonitorSettings",
    "ResolvedMonitorTarget",
    "build_container_start_time_query",
    "build_gpu_metrics_query",
    "build_ix_gpu_metrics_query",
    "build_node_cpu_total_query",
    "build_node_memory_total_query",
    "build_pod_cpu_query",
    "build_pod_info_query",
    "build_pod_memory_query",
    "extract_phase_records",
    "load_monitor_settings",
    "monitor_target",
    "run_monitoring",
]
