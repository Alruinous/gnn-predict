# Prometheus 指标查询参考

## 适用范围

本文档对集群做的实测快照整理，目标是给后续助手提供一份可直接复用的查询手册，减少重复确认以下内容的前置成本：

- 当前集群里 Prometheus 的入口、版本、抓取配置和规则评估配置
- 每类指标来自哪个 exporter 或 recording rule
- 每类指标的真实抓取频率、更新时间和常见标签
- 查询 CPU、内存、加速卡、Pod 元信息时应该优先选哪组指标
- 当前集群里已经存在的统一 `gpu_*` 规则指标和已知坑点

快照时间：`2026-04-11 20:52`（`Asia/Shanghai`）

## 当前 Prometheus 实例

| 项目 | 当前值 |
| --- | --- |
| Prometheus CR | `prometheus/prometheus-kube-prometheus-prometheus` |
| Prometheus 版本 | `v2.55.1` |
| Helm Chart | `kube-prometheus-stack 67.8.0` |
| 对外 Service | `prometheus/prometheus-kube-prometheus-prometheus` |
| 对外 NodePort | 未配置 |
| 全局默认抓取频率 | `scrape_interval: 1s` |
| 全局默认规则评估频率 | `evaluation_interval: 30s` |
| 保留期 | `365d` |
| 副本数 | `1` |
| 分片数 | `1` |
| 存储 | `2Ti`, `rook-cephfs` |
| Reconciled 状态 | `True` |
| ServiceMonitor 选择器 | `serviceMonitorSelector: {}` |
| ServiceMonitor 命名空间选择器 | `serviceMonitorNamespaceSelector: {}` |
| PodMonitor 数量 | `0` |
| ScrapeConfig 数量 | `0` |

当前这套 Prometheus 会抓取所有命名空间里被选中的 `ServiceMonitor`，因此除了 `prometheus` 命名空间里的标准监控组件，也会抓 `nvidia-gpu-operator`、`npu-exporter`、`kube-system`、`metax-monitor` 等命名空间中的 exporter。

当前 Prometheus CR 已恢复 `Reconciled=True`，Prometheus 本体的存储配置也和当前 PVC 一致。当前 Prometheus Operator 仍有 `PrometheusOperatorRejectedResources` 告警，直接原因是 `prometheus-grafana` 这个 `ServiceMonitor` 的 `scrapeTimeout: 30s` 大于当前生效的 `scrapeInterval: 1s`，因此该资源被拒绝。

## 核心概念

### 抓取频率

`scrape_interval` 表示 Prometheus 多久去目标的 `/metrics` 拉一次原始样本。

- `1s` 表示每秒抓一次
- 某个 `ServiceMonitor` 如果单独配置了 `interval`，就会覆盖全局默认值

### 规则评估频率

`evaluation_interval` 表示 Prometheus 多久执行一次规则表达式。

- `recording rule` 会把表达式结果写成新的时序指标
- `alerting rule` 会更新告警状态，不会生成新的业务指标名

### 查询步长

`query_range` 的 `step` 只是 API 返回数据时的采样步长，不等于 exporter 的真实抓取频率。

仓库里的临时脚本 [prometheus.py](/Users/yid11/project/tmp-python/monitor/prometheus.py:1) 会把 `query_range` 的最小步长固定到 `5s`，所以它返回的分辨率不代表 Prometheus 的原始采样分辨率。

## 推荐访问方式

### 从集群内直接查 Prometheus API

```bash
KUBECONFIG=backend/kubeconfig kubectl -n prometheus exec \
  prometheus-prometheus-kube-prometheus-prometheus-0 \
  -c prometheus -- \
  wget --post-data='query=up' -qO- http://127.0.0.1:9090/api/v1/query
```

### 从本地通过 port-forward 访问

当前 Prometheus Service 是 `ClusterIP`，没有直接暴露 NodePort。需要在本地访问时，直接做端口转发：

```bash
KUBECONFIG=backend/kubeconfig kubectl -n prometheus port-forward \
  svc/prometheus-kube-prometheus-prometheus 9090:9090
```

```text
http://127.0.0.1:9090
```

### 推荐查询 API

- 瞬时查询：`/api/v1/query`
- 区间查询：`/api/v1/query_range`
- 查看 target：`/api/v1/targets`
- 查看规则：`/api/v1/rules`

## 当前 source / job 总览

下表按“后续助手最可能直接使用”的角度整理。

| 数据源 | job 标签 | 目标数 | 实际抓取频率 | 实际抓取超时 | 按当前 job 可见的指标名数量 | 主要用途 |
| --- | --- | --- | --- | --- | --- | --- |
| kube-state-metrics | `kube-state-metrics` | `1` | `1s` | `1s` | `197` | Pod/Node 元信息、资源 requests/limits、allocatable/capacity |
| kubelet `/metrics` | `kubelet`, `metrics_path="/metrics"` | `43` | `1s` | `1s` | `379` | kubelet 自身运行状态、存储、设备插件、部分系统指标 |
| kubelet `/metrics/cadvisor` | `kubelet`, `metrics_path="/metrics/cadvisor"` | `43` | `1s` | `1s` | `62` | 容器 CPU、内存、网络、压力、machine 信息 |
| kubelet `/metrics/probes` | `kubelet`, `metrics_path="/metrics/probes"` | `43` | `1s` | `1s` | `12` | 探针相关 |
| node-exporter | `node-exporter` | `24` | `1s` | `1s` | `451` | 节点 CPU、内存、磁盘、文件系统、网络、Infiniband、hwmon |
| NVIDIA DCGM exporter | `nvidia-dcgm-exporter` | `10` | `1s` | `1s` | `47` | NVIDIA GPU 利用率、显存、时钟、温度、功耗、PCIe、Tensor/FP 管线 |
| Ascend NPU exporter | `npu-exporter` | `2` | `15s` | `1s` | `118` | Ascend NPU 利用率、HBM、HCCS、RoCE、功耗、温度、带宽 |
| Ix exporter | `ix-exporter` | `1` | `15s` | `1s` | `32` | Ix 加速卡利用率、显存、温度、功耗、PCIe |
| Metax exporter | `mx-exporter` | `1` | `15s` | `1s` | `46` | Metax 加速卡利用率、显存、时钟、温度、功耗、链路 |
| GPU operator | `gpu-operator` | `1` | `1s` | `1s` | `88` | GPU operator 自身健康、升级状态、reconcile 状态 |
| Alertmanager | `prometheus-kube-prometheus-alertmanager` | `2` | `1s` | `1s` | 未细分 | 告警组件自身指标 |
| Prometheus 自监控 | `prometheus-kube-prometheus-prometheus` | `2` | `1s` | `1s` | 未细分 | TSDB、规则评估、查询引擎、远端写入、自身状态 |
| 控制面组件 | `apiserver` / `kube-scheduler` / `kube-controller-manager` / `kube-etcd` / `kube-proxy` / `coredns` | 多个 | `1s` | `1s` | 未逐项展开 | 控制面与系统组件运行状态 |

补充说明：

- `monitoring` 命名空间里存在两个 `ServiceMonitor`，`kube-gpu-colocate-scheduler-monitor` 和 `lucid-scheduler-monitor`，配置的抓取频率是 `5s`，但这次快照里既没有出现在 `activeTargets`，也没有出现在 `droppedTargets`，说明当前 Prometheus 还没有实际发现到对应 target。
- `prometheus-grafana` 这个 `ServiceMonitor` 当前没有进入活跃抓取集合，因为它被 operator 拒绝了，拒绝原因是 `scrapeTimeout: 30s` 大于当前生效的 `scrapeInterval: 1s`。
- `up`、`scrape_duration_seconds`、`scrape_samples_scraped`、`scrape_series_added`、`ALERTS`、`ALERTS_FOR_STATE` 这类通用指标会出现在多个 job 中。
- 表中的指标名数量是按 `count by (__name__) ({job="..."})` 盘点的当前结果，可能包含共享 `up` / `scrape_*` 指标，以及保留了相同 `job` 标签的 recording metrics，不等于 exporter 原生指标数。

## 该用哪一类指标

| 查询目标 | 首选指标 | 原因 |
| --- | --- | --- |
| Pod 元信息 | `kube_pod_info`, `kube_pod_owner`, `kube_pod_status_*` | 来自 kube-state-metrics，语义稳定，适合做元信息和资源联结 |
| Pod requests / limits | `kube_pod_container_resource_requests`, `kube_pod_container_resource_limits` | 直接反映声明资源，不要用 cAdvisor 推断 |
| 节点 allocatable / capacity | `kube_node_status_allocatable`, `kube_node_status_capacity` | 直接来自 kube-state-metrics |
| 容器实时 CPU / 内存使用 | `container_cpu_usage_seconds_total`, `container_memory_usage_bytes` | 来自 cAdvisor，是真实运行时使用值 |
| 节点 CPU / 内存使用 | `node_cpu_seconds_total`, `node_memory_*` | 来自 node-exporter，原始语义最清楚 |
| NVIDIA GPU 原始指标 | `DCGM_FI_*` | 1s 更新，比统一 `gpu_*` 更细、更原始 |
| Ascend NPU 原始指标 | `container_npu_*`, `npu_chip_info_*` | 15s 更新，保留了 NPU 专有字段 |
| 跨厂商统一加速卡查询 | `gpu_*` recording metrics | Prometheus 已统一 vendor/model/node/gpu 维度，跨 NVIDIA/NPU/Ix/Metax 可直接查询 |
| 目标健康与抓取是否正常 | `up`, `scrape_*` | 最直接 |

## 各类指标详细说明

## 1. Pod / Node 元信息与声明资源

### 来源

- source job：`kube-state-metrics`
- 抓取路径：`/metrics`
- 抓取频率：`1s`
- 获取方式：kube-state-metrics 监听 Kubernetes API，把对象状态转成 Prometheus 指标

### 关键原始指标

| 指标 | 含义 | 常用标签 |
| --- | --- | --- |
| `kube_pod_info` | Pod 元信息 | `namespace`, `pod`, `node`, `host_ip`, `pod_ip`, `created_by_kind`, `created_by_name` |
| `kube_pod_owner` | Pod 所属工作负载 | `namespace`, `pod`, `owner_kind`, `owner_name` |
| `kube_pod_status_phase` | Pod Phase | `namespace`, `pod`, `phase` |
| `kube_pod_status_ready` | Pod Ready 状态 | `namespace`, `pod`, `condition` |
| `kube_pod_container_resource_requests` | 容器 requests | `namespace`, `pod`, `container`, `resource`, `unit`, `node` |
| `kube_pod_container_resource_limits` | 容器 limits | `namespace`, `pod`, `container`, `resource`, `unit`, `node` |
| `kube_node_status_allocatable` | 节点 allocatable | `node`, `resource`, `unit` |
| `kube_node_status_capacity` | 节点 capacity | `node`, `resource`, `unit` |
| `kube_node_info` | 节点元信息 | `node`, `kernel_version`, `os_image`, `container_runtime_version` |
| `kube_node_status_condition` | 节点条件 | `node`, `condition`, `status` |

### 常见查询模板

#### 查某个 Pod 当前所在节点与 owner

```promql
kube_pod_info{namespace="crater-workspace", pod="your-pod"}
```

```promql
kube_pod_owner{namespace="crater-workspace", pod="your-pod"}
```

#### 查某个 Pod 的 CPU / 内存 requests

```promql
kube_pod_container_resource_requests{
  namespace="crater-workspace",
  pod="your-pod",
  resource=~"cpu|memory"
}
```

#### 查某个节点的可分配 CPU / 内存 / 加速卡

```promql
kube_node_status_allocatable{node="your-node"}
```

#### 查运行中 Pod 的资源请求总和

```promql
sum by (node, resource) (
  kube_pod_container_resource_requests
  * on(pod, namespace) group_left()
  kube_pod_status_phase{phase="Running"}
)
```

### 说明

- 这组指标适合做元信息联结和资源声明统计，不适合表示真实使用量。
- 仓库中的临时脚本 [const.py](/Users/yid11/project/tmp-python/monitor/const.py:1) 和 [prometheus.py](/Users/yid11/project/tmp-python/monitor/prometheus.py:1) 查询 CPU、内存和 GPU 余量时，核心依赖的就是这组指标。

## 2. 容器实时 CPU / 内存使用

### 来源

- source job：`kubelet`, `metrics_path="/metrics/cadvisor"`
- 抓取路径：`/metrics/cadvisor`
- 抓取频率：`1s`
- 获取方式：Prometheus 抓 kubelet 暴露的 cAdvisor 指标

### 关键原始指标

| 指标 | 含义 | 常用标签 |
| --- | --- | --- |
| `container_cpu_usage_seconds_total` | 容器累计 CPU 使用秒数 | `namespace`, `pod`, `container`, `instance`, `metrics_path` |
| `container_memory_usage_bytes` | 容器当前内存使用 | `namespace`, `pod`, `container`, `instance`, `metrics_path` |
| `container_memory_working_set_bytes` | 工作集内存 | `namespace`, `pod`, `container` |
| `container_memory_rss` | RSS | `namespace`, `pod`, `container` |
| `container_start_time_seconds` | 容器启动时间 | `namespace`, `pod`, `container` |
| `container_network_receive_bytes_total` | 容器接收字节数 | `namespace`, `pod`, `container`, `interface` |
| `container_network_transmit_bytes_total` | 容器发送字节数 | `namespace`, `pod`, `container`, `interface` |
| `container_processes` | 进程数 | `namespace`, `pod`, `container` |
| `machine_cpu_cores` | 节点核心数 | `instance` |
| `machine_memory_bytes` | 节点内存容量 | `instance` |

### 常见查询模板

#### 某个 Pod 的 CPU 使用率

```promql
sum by (namespace, pod) (
  irate(container_cpu_usage_seconds_total{
    namespace="crater-workspace",
    pod="your-pod",
    container!="POD",
    container!=""
  }[30s])
)
```

#### 某个 Pod 的内存使用

```promql
sum by (namespace, pod) (
  container_memory_usage_bytes{
    namespace="crater-workspace",
    pod="your-pod",
    container!="POD",
    container!=""
  }
)
```

#### 某个 Pod 的容器启动时间

```promql
min(container_start_time_seconds{
  namespace="crater-workspace",
  pod="your-pod",
  container!="",
  container!="POD"
})
```

### 说明

- 这是 Crater 后端 [query.go](/Users/yid11/project/crater/limit/backend/pkg/monitor/query.go:17) 查询 Pod CPU / 内存使用时的主数据源。
- kubelet 的 cAdvisor 抓取配置设置了 `honor_timestamps: true`，直接看 `timestamp(container_*)` 容易受 exporter 自带时间戳影响，不适合反推抓取间隔。实际应以 target 配置里的 `1s` 为准。
- 当前项目用 `container_start_time_seconds` 后 5 秒内的 Pod 工作集内存最小值作为 `memory_delta_gb_*` 的启动基线。

## 3. 节点 CPU / 内存 / 磁盘 / 网络使用

### 来源

- source job：`node-exporter`
- 抓取路径：`/metrics`
- 抓取频率：`1s`
- 按当前 job 可见的指标名数量：`451`

### 关键原始指标家族

| 家族 | 示例 |
| --- | --- |
| CPU | `node_cpu_seconds_total`, `node_cpu_scaling_frequency_hertz` |
| 内存 | `node_memory_MemAvailable_bytes`, `node_memory_MemTotal_bytes`, `node_memory_Cached_bytes` |
| 磁盘 | `node_disk_read_bytes_total`, `node_disk_written_bytes_total`, `node_disk_io_time_seconds_total` |
| 文件系统 | `node_filesystem_avail_bytes`, `node_filesystem_size_bytes`, `node_filesystem_free_bytes` |
| 网络 | `node_network_receive_bytes_total`, `node_network_transmit_bytes_total` |
| Infiniband | `node_infiniband_*` |
| hwmon / 传感器 | `node_hwmon_temp_celsius`, `node_hwmon_power_average_watt` |

### 常见查询模板

#### 节点 CPU 使用率

```promql
100 - avg by (instance) (
  rate(node_cpu_seconds_total{mode="idle"}[5m])
) * 100
```

#### 节点内存使用率

```promql
(1 - avg by (instance) (node_memory_MemAvailable_bytes)
    / avg by (instance) (node_memory_MemTotal_bytes)) * 100
```

#### 节点磁盘吞吐

```promql
rate(node_disk_read_bytes_total[5m])
```

### 已有 recording rules

这类规则主要来自默认 node rule group，更新时间通常是 `30s`。

| recording metric | 说明 |
| --- | --- |
| `instance:node_cpu_utilisation:rate5m` | 节点 CPU 使用率 |
| `instance:node_memory_utilisation:ratio` | 节点内存使用率 |
| `instance:node_num_cpu:sum` | 节点 CPU 核数 |
| `instance:node_load1_per_cpu:ratio` | 每核 load1 |
| `instance_device:node_disk_io_time_seconds:rate5m` | 磁盘 IO time |
| `instance_device:node_disk_io_time_weighted_seconds:rate5m` | 加权 IO time |
| `instance:node_network_receive_bytes_excluding_lo:rate5m` | 去掉 lo 的接收流量 |
| `instance:node_network_transmit_bytes_excluding_lo:rate5m` | 去掉 lo 的发送流量 |
| `node:node_cpu_utilization:ratio_rate5m` | 按 node 聚合的 CPU 使用率 |
| `cluster:node_cpu:ratio_rate5m` | 集群级 CPU 使用率 |

## 4. kubelet 运行时与系统指标

### 来源

- source job：`kubelet`, `metrics_path="/metrics"`
- 抓取路径：`/metrics`
- 抓取频率：`1s`
- 按当前 job 可见的指标名数量：`379`

### 指标范围

这组指标不是给业务资源查询优先使用的，但在排查 kubelet、设备插件、CSI、镜像拉取、存储异常时很有用。

常见家族：

- `kubelet_*`
- `apiserver_*`
- `csi_*`
- `authentication_*`
- `go_*`
- `process_*`

### 常见示例

```promql
kubelet_active_pods
```

```promql
rate(kubelet_http_requests_total[5m])
```

```promql
histogram_quantile(
  0.99,
  sum by (le, instance) (
    rate(kubelet_http_requests_duration_seconds_bucket[5m])
  )
)
```

## 5. NVIDIA GPU 原始指标

### 来源

- source job：`nvidia-dcgm-exporter`
- 抓取路径：`/metrics`
- 抓取频率：`1s`
- 按当前 job 可见的指标名数量：`47`
- exporter：NVIDIA DCGM Exporter

### 关键原始指标

| 指标 | 含义 |
| --- | --- |
| `DCGM_FI_DEV_GPU_UTIL` | GPU 利用率 |
| `DCGM_FI_DEV_FB_USED` | 已用显存 |
| `DCGM_FI_DEV_FB_FREE` | 空闲显存 |
| `DCGM_FI_DEV_GPU_TEMP` | GPU 温度 |
| `DCGM_FI_DEV_POWER_USAGE` | 功耗 |
| `DCGM_FI_DEV_SM_CLOCK` | SM 时钟 |
| `DCGM_FI_DEV_MEM_CLOCK` | 显存时钟 |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | 显存拷贝利用率 |
| `DCGM_FI_PROF_SM_ACTIVE` | SM 活跃度 |
| `DCGM_FI_PROF_SM_OCCUPANCY` | SM 占用度 |
| `DCGM_FI_PROF_DRAM_ACTIVE` | DRAM 活跃度 |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | Tensor Core 活跃度 |
| `DCGM_FI_PROF_PIPE_FP16_ACTIVE` | FP16 管线活跃度 |
| `DCGM_FI_PROF_PIPE_FP32_ACTIVE` | FP32 管线活跃度 |
| `DCGM_FI_PROF_PIPE_FP64_ACTIVE` | FP64 管线活跃度 |
| `DCGM_FI_PROF_PCIE_TX_BYTES` | PCIe TX |
| `DCGM_FI_PROF_PCIE_RX_BYTES` | PCIe RX |
| `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION` | 累积能耗 |
| `DCGM_FI_DEV_XID_ERRORS` | XID 错误 |

### 常用标签

原始 NVIDIA 指标常见标签包括：

- `Hostname`
- `UUID`
- `modelName`
- `gpu`
- `instance`
- `pod`
- `namespace`
- `container`

### 常见查询模板

#### 某个 Pod 的 GPU 利用率

```promql
avg_over_time(DCGM_FI_DEV_GPU_UTIL{
  namespace="crater-workspace",
  pod="your-pod"
}[60s]) / 100
```

#### 某个 Pod 的显存峰值

```promql
max_over_time(DCGM_FI_DEV_FB_USED{
  namespace="crater-workspace",
  pod="your-pod"
}[60s])
```

#### 某个节点的 GPU 利用率

```promql
DCGM_FI_DEV_GPU_UTIL{Hostname="your-node"}
```

### 说明

- Crater 后端的 GPU 分析逻辑主要就是基于这组指标，见 [query.go](/Users/yid11/project/crater/limit/backend/pkg/monitor/query.go:17)。
- 如果你要做 NVIDIA 专有特征分析，优先用 `DCGM_FI_*` 原始指标，不要先用统一 `gpu_*`。

## 6. Ascend NPU 原始指标

### 来源

- source job：`npu-exporter`
- 抓取路径：`/metrics`
- 抓取频率：`15s`
- 按当前 job 可见的指标名数量：`118`

### 关键原始指标

| 指标家族 | 示例 |
| --- | --- |
| 容器视角 NPU 指标 | `container_npu_utilization`, `container_npu_used_memory`, `container_npu_total_memory` |
| 芯片基础信息 | `npu_chip_info_name`, `npu_chip_info_serial_number`, `npu_chip_info_health_status` |
| 内存 / HBM | `npu_chip_info_total_memory`, `npu_chip_info_used_memory`, `npu_chip_info_hbm_total_memory`, `npu_chip_info_hbm_used_memory` |
| 利用率 | `npu_chip_info_utilization`, `npu_chip_info_vector_utilization`, `npu_chip_info_hbm_utilization` |
| 功耗 / 温度 / 电压 | `npu_chip_info_power`, `npu_chip_info_temperature`, `npu_chip_info_voltage` |
| 互联与带宽 | `npu_chip_info_bandwidth_rx`, `npu_chip_info_bandwidth_tx`, `npu_chip_info_hccs_*`, `npu_chip_info_pcie_*` |
| 网络与 RoCE | `npu_chip_roce_*`, `npu_chip_mac_*` |

### 常用标签

NPU 原始指标常见标签与 kube-state / DCGM 不完全一致，后续助手需要特别注意：

- `pod_name`
- `container_name`
- `namespace`
- `id`
- `model_name`
- `vdie_id`
- `pcie_bus_info`

### 常见查询模板

#### 容器视角的 NPU 利用率

```promql
container_npu_utilization{
  namespace="crater-workspace",
  pod_name="your-pod"
}
```

#### NPU 芯片视角的功耗

```promql
npu_chip_info_power
```

### 说明

- NPU 原始指标里的 Pod 标签是 `pod_name`，不是 `pod`。
- 统一 `gpu_*` recording rule 会把这组指标映射成 `vendor="npu"` 的统一加速卡指标，但原始字段只在 NPU 指标里保留得最完整。

## 7. Ix / Metax 原始指标

### Ix exporter

- source job：`ix-exporter`
- 抓取频率：`15s`
- 按当前 job 可见的指标名数量：`32`

关键指标：

- `ix_gpu_utilization`
- `ix_mem_total`
- `ix_mem_used`
- `ix_mem_free`
- `ix_mem_utilization`
- `ix_temperature`
- `ix_power_usage`
- `ix_pcie_rx_throughput`
- `ix_pcie_tx_throughput`
- `ix_sm_clock`
- `ix_sm_utilization`
- `ix_process_info`

### Metax exporter

- source job：`mx-exporter`
- 抓取频率：`15s`
- 按当前 job 可见的指标名数量：`46`

关键指标：

- `mx_gpu_usage`
- `mx_memory_total`
- `mx_memory_used`
- `mx_memory_usage`
- `mx_gpu_clock`
- `mx_mem_clock`
- `mx_chip_hotspot_temp`
- `mx_board_power`
- `mx_pcie_bw`
- `mx_mxlk_bw`

## 8. 统一加速卡 recording metrics

### 基本事实

- 规则组：`node-gpu-unified`
- 规则位置：`prometheus/node-gpu-unified`
- 规则更新频率：`30s`
- 当前这组规则会把多厂商加速卡指标统一成 `gpu_*` 指标
- 当前统一来源包括：
  - NVIDIA：`DCGM_FI_*`
  - Ix：`ix_*`
  - Metax：`mx_*`
  - NPU：`npu_chip_info_*`

### 当前统一后的指标名

| recording metric | 主要来源 | 说明 |
| --- | --- | --- |
| `gpu_node_info` | `DCGM_FI_DEV_GPU_UTIL`, `ix_gpu_utilization`, `mx_gpu_usage`, `npu_chip_info_name` | 卡存在性和节点映射 |
| `gpu_utilization_percent` | `DCGM_FI_DEV_GPU_UTIL`, `ix_gpu_utilization`, `mx_gpu_usage`, `npu_chip_info_utilization` | 统一利用率 |
| `gpu_memory_used_bytes` | `DCGM_FI_DEV_FB_USED`, `ix_mem_used`, `mx_memory_used`, `npu_chip_info_*used_memory` | 统一显存 / HBM 已用量 |
| `gpu_memory_total_bytes` | `DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE`, `ix_mem_total`, `mx_memory_total`, `npu_chip_info_*total_memory` | 统一显存 / HBM 总量 |
| `gpu_temperature_celsius` | `DCGM_FI_DEV_GPU_TEMP`, `ix_temperature`, `mx_chip_hotspot_temp`, `npu_chip_info_*temperature` | 温度 |
| `gpu_power_watts` | `DCGM_FI_DEV_POWER_USAGE`, `ix_power_usage`, `mx_board_power / 1000`, `npu_chip_info_power` | 功耗 |
| `gpu_sm_clock_mhz` | `DCGM_FI_DEV_SM_CLOCK`, `ix_sm_clock`, `mx_gpu_clock`, `npu_chip_info_aicore_current_freq` | 核心时钟 |
| `gpu_tensor_core_utilization_percent` | `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`, `ix_gpu_utilization`, `mx_gpu_usage`, `npu_chip_info_vector_utilization` | 统一计算核心活跃度 |
| `gpu_memory_copy_utilization_percent` | `DCGM_FI_DEV_MEM_COPY_UTIL` | 仅 NVIDIA 有原始来源 |
| `gpu_total_energy_consumption_millijoules` | `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION` | 累积能耗 |
| `gpu_total_energy_consumption_millijoules_native` | `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION` | 原生累积能耗别名 |
| `gpu_available` | 基于存在性指标构造 | 卡存在即为 `1` |
| `gpu_energy_estimated_millijoules_30s` | 基于 `gpu_power_watts` 类似来源乘以 `30000` | 30 秒窗口估算能耗 |

### 统一标签

统一 `gpu_*` 指标中，后续助手优先使用这些标准标签：

- `node`
- `vendor`
- `model`
- `gpu`
- `uuid`（部分 vendor 不一定有）

### 说明

- 这组指标名虽然叫 `gpu_*`，但实际已经包含 `vendor="npu"` 的 NPU 数据。
- 这组规则的更新周期是 `30s`，所以它的时间粒度比 NVIDIA 原始 `1s` 指标和其他厂商常见的 `15s` 原始指标都更粗。
- `gpu_memory_used_bytes` / `gpu_memory_total_bytes` 这类名字看起来是 bytes，但跨 vendor 的原始 exporter 并不总是严格统一单位。做跨厂商比较前，建议先抽样验证各 vendor 的量纲。

## 9. 其他 recording metrics

除了统一加速卡规则，当前 Prometheus 里共有 `98` 条 recording rule，去重后形成 `72` 个唯一 recording metric 名称。

### 资源与工作负载聚合

- `node_namespace_pod_container:container_cpu_usage_seconds_total:sum_irate`
- `node_namespace_pod_container:container_memory_cache`
- `node_namespace_pod_container:container_memory_rss`
- `node_namespace_pod_container:container_memory_swap`
- `node_namespace_pod_container:container_memory_working_set_bytes`
- `cluster:namespace:pod_cpu:active:kube_pod_container_resource_requests`
- `cluster:namespace:pod_cpu:active:kube_pod_container_resource_limits`
- `cluster:namespace:pod_memory:active:kube_pod_container_resource_requests`
- `cluster:namespace:pod_memory:active:kube_pod_container_resource_limits`
- `namespace_cpu:kube_pod_container_resource_requests:sum`
- `namespace_cpu:kube_pod_container_resource_limits:sum`
- `namespace_memory:kube_pod_container_resource_requests:sum`
- `namespace_memory:kube_pod_container_resource_limits:sum`
- `namespace_workload_pod:kube_pod_owner:relabel`
- `node_namespace_pod:kube_pod_info:`

### 节点与集群聚合

- `instance:node_cpu:rate:sum`
- `instance:node_cpu:ratio`
- `instance:node_cpu_utilisation:rate5m`
- `instance:node_load1_per_cpu:ratio`
- `instance:node_memory_utilisation:ratio`
- `instance:node_network_receive_bytes:rate:sum`
- `instance:node_network_transmit_bytes:rate:sum`
- `instance:node_network_receive_bytes_excluding_lo:rate5m`
- `instance:node_network_transmit_bytes_excluding_lo:rate5m`
- `instance:node_network_receive_drop_excluding_lo:rate5m`
- `instance:node_network_transmit_drop_excluding_lo:rate5m`
- `instance:node_vmstat_pgmajfault:rate5m`
- `instance_device:node_disk_io_time_seconds:rate5m`
- `instance_device:node_disk_io_time_weighted_seconds:rate5m`
- `instance:node_num_cpu:sum`
- `node:node_num_cpu:sum`
- `node:node_cpu_utilization:ratio_rate5m`
- `cluster:node_cpu:sum_rate5m`
- `cluster:node_cpu:ratio`
- `cluster:node_cpu:ratio_rate5m`
- `:node_memory_MemAvailable_bytes:sum`

### API Server / Scheduler / Kubelet 规则产物

- `apiserver_request:availability30d`
- `apiserver_request:burnrate5m`
- `apiserver_request:burnrate30m`
- `apiserver_request:burnrate1h`
- `apiserver_request:burnrate2h`
- `apiserver_request:burnrate6h`
- `apiserver_request:burnrate1d`
- `apiserver_request:burnrate3d`
- `cluster_quantile:apiserver_request_sli_duration_seconds:histogram_quantile`
- `cluster_quantile:scheduler_e2e_scheduling_duration_seconds:histogram_quantile`
- `cluster_quantile:scheduler_scheduling_algorithm_duration_seconds:histogram_quantile`
- `cluster_quantile:scheduler_binding_duration_seconds:histogram_quantile`
- `node_quantile:kubelet_pleg_relist_duration_seconds:histogram_quantile`
- `count:up0`
- `count:up1`

### 规则更新时间

- 绝大多数 rule group：`30s`
- 例外：`kube-apiserver-availability.rules` 是 `180s`

## 10. alerting rules

### 当前规模

- 当前 alerting rule 数量：`145`
- 它们不会生成新的业务指标名
- Prometheus 内部会维护：
  - `ALERTS`
  - `ALERTS_FOR_STATE`

### 当前已观察到的告警状态

这次快照里以下告警在 firing，括号内是当前 firing 条数：

- `CPUThrottlingHigh` (`1`)
- `KubeAPIErrorBudgetBurn` (`1`)
- `KubeAggregatedAPIDown` (`1`)
- `KubeControllerManagerDown` (`1`)
- `KubeDaemonSetMisScheduled` (`11`)
- `KubeDaemonSetRolloutStuck` (`15`)
- `KubeDeploymentReplicasMismatch` (`9`)
- `KubeDeploymentRolloutStuck` (`5`)
- `KubeJobFailed` (`11`)
- `KubeJobNotCompleted` (`4`)
- `KubeNodeNotReady` (`1`)
- `KubeNodeUnreachable` (`1`)
- `KubePodCrashLooping` (`7`)
- `KubePodNotReady` (`17`)
- `KubeProxyDown` (`1`)
- `KubeSchedulerDown` (`1`)
- `KubeStatefulSetReplicasMismatch` (`2`)
- `NodeClockNotSynchronising` (`1`)
- `PrometheusMissingRuleEvaluations` (`1`)
- `PrometheusOperatorRejectedResources` (`1`)
- `PrometheusRuleFailures` (`2`)
- `TargetDown` (`5`)
- `Watchdog` (`1`)
- `etcdInsufficientMembers` (`1`)
- `etcdMembersDown` (`1`)

这意味着：

- 控制面多类 target 当前不可达，控制面指标不能当作稳定实时数据源
- 工作负载层面也有较多 rollout / crash / not ready 告警，集群本身处于不稳定状态
- 某些 recording rule 仍存在延迟或评估失败
- Prometheus Operator 当前还有被拒绝的监控资源

## 11. 当前健康状态与已知问题

### 与资源查询直接相关的 down target

| scrape pool | down 数量 | 影响 |
| --- | --- | --- |
| `serviceMonitor/nvidia-gpu-operator/nvidia-dcgm-exporter/0` | `1` | 某个 NVIDIA 节点 GPU 原始指标缺失 |
| `serviceMonitor/prometheus/prometheus-kube-prometheus-kube-controller-manager/0` | `3` | controller-manager 指标当前全部缺失 |
| `serviceMonitor/prometheus/prometheus-kube-prometheus-kube-etcd/0` | `3` | etcd 指标当前全部缺失 |
| `serviceMonitor/prometheus/prometheus-kube-prometheus-kube-proxy/0` | `24` | kube-proxy 指标当前全部缺失 |
| `serviceMonitor/prometheus/prometheus-kube-prometheus-kube-scheduler/0` | `3` | scheduler 指标当前全部缺失 |
| `serviceMonitor/prometheus/prometheus-kube-prometheus-kubelet/0` | `6` | 某些节点 kubelet `/metrics` 缺失 |
| `serviceMonitor/prometheus/prometheus-kube-prometheus-kubelet/1` | `6` | 某些节点 cAdvisor 容器指标缺失 |
| `serviceMonitor/prometheus/prometheus-kube-prometheus-kubelet/2` | `6` | 某些节点 probe 指标缺失 |
| `serviceMonitor/prometheus/prometheus-prometheus-node-exporter/0` | `2` | 某些节点 node-exporter 缺失 |

### 规则评估异常

当前 `PrometheusRuleFailures` 关联到的 rule group 包括：

- `kubelet.rules`
- `kubernetes-system-kubelet`

当前 `PrometheusMissingRuleEvaluations` 关联到的 rule group：

- `kube-apiserver-burnrate.rules`

当前 `PrometheusOperatorRejectedResources` 的直接原因：

- 被拒绝的资源是 `prometheus/prometheus-grafana`
- 该 `ServiceMonitor` 配置了 `scrapeTimeout: 30s`
- 当前 Prometheus 全局默认抓取频率是 `1s`
- operator 事件里的直接消息是 `scrapeTimeout "30s" greater than scrapeInterval "1s"`

### 结论

- 如果只查原始 kube-state-metrics、node-exporter、DCGM、NPU 指标，问题相对可控，但需要避开当前 down 掉的 target
- 如果直接依赖 rule 产物，尤其是 API Server burn rate / 部分 kubelet 规则，需要接受“结果可能延后或短时缺失”
- 如果要排查 Prometheus 配置本身，先关注 `PrometheusOperatorRejectedResources`，因为当前有无效 `ServiceMonitor` 被 operator 拒绝

## 12. 标签差异与踩坑记录

### 标签名并不统一

| 数据源 | 常用 Pod 标签 | 常用节点标签 | 其他关键标签 |
| --- | --- | --- | --- |
| kube-state-metrics | `pod` | `node` | `resource`, `unit`, `created_by_kind`, `created_by_name` |
| kubelet / cAdvisor | `pod` | 通过 `instance` 映射 | `container`, `metrics_path="/metrics/cadvisor"` |
| node-exporter | 无 Pod 维度 | `instance` | `device`, `mountpoint`, `fstype`, `cpu`, `mode` |
| NVIDIA DCGM | `pod` | `Hostname` | `UUID`, `modelName`, `gpu`, `container` |
| NPU exporter | `pod_name` | 原始不统一，统一规则补 `node` | `container_name`, `model_name`, `id`, `vdie_id` |
| Ix exporter | 无统一 Pod 维度 | `node_name` | `name`, `gpu` |
| Metax exporter | 无统一 Pod 维度 | `Hostname` | `modelName`, `deviceId` |
| 统一 `gpu_*` | 通常无工作负载 Pod 维度 | `node` | `vendor`, `model`, `gpu`, `uuid` |

### 不要混淆的三件事

- `scrape_interval` 是抓取频率
- `evaluation_interval` 是规则执行频率
- `query_range step` 是返回数据的采样步长

### cAdvisor 时间戳陷阱

直接查询：

```promql
timestamp(container_cpu_usage_seconds_total)
```

看到的时间差不一定是 `1s`，因为 kubelet/cAdvisor 抓取配置启用了 `honor_timestamps`。判断真实抓取频率应以 target API 返回的 `scrapeInterval` 为准。

### 统一 `gpu_*` 的适用边界

- 适合做跨 vendor 统一面板、统一巡检、统一粗粒度对比
- 不适合做 vendor 专有特性分析
- 不适合对跨 vendor 显存单位做盲目同量纲比较

## 13. 后续助手建议优先使用的 PromQL 模板

### 列出某个 source 当前有哪些指标名

```promql
count by (__name__) ({job="kube-state-metrics"})
```

```promql
count by (__name__) ({job="node-exporter"})
```

```promql
count by (__name__) ({job="kubelet", metrics_path="/metrics/cadvisor"})
```

```promql
count by (__name__) ({job="nvidia-dcgm-exporter"})
```

```promql
count by (__name__) ({job="npu-exporter"})
```

### 列出统一加速卡指标

```promql
count by (__name__) ({__name__=~"gpu_.*"})
```

### 查看某类 target 是否健康

```promql
up{job="nvidia-dcgm-exporter"}
```

```promql
up{job="kubelet", metrics_path="/metrics/cadvisor"}
```

### 查询节点剩余 CPU / 内存 / 加速卡余量

```promql
kube_node_status_allocatable{resource="cpu"}
- ignoring(container, endpoint, instance, job, namespace, pod, resource, service, unit)
sum by (node) (
  kube_pod_container_resource_requests{resource="cpu"}
  * on(pod, namespace) group_left()
  kube_pod_status_phase{phase="Running"}
)
```

```promql
kube_node_status_allocatable{resource="memory"}
- ignoring(container, endpoint, instance, job, namespace, pod, resource, service, unit)
sum by (node) (
  kube_pod_container_resource_requests{resource="memory"}
  * on(pod, namespace) group_left()
  kube_pod_status_phase{phase="Running"}
)
```

### 查询跨 vendor 加速卡利用率

```promql
gpu_utilization_percent
```

### 查询 NVIDIA 原始 GPU 利用率

```promql
DCGM_FI_DEV_GPU_UTIL
```

### 查询 Ascend NPU 原始利用率

```promql
container_npu_utilization
```

## 14. 当前 job 名称速查

当前 `up` 中可见的 job 标签如下：

- `apiserver`
- `coredns`
- `gpu-operator`
- `ix-exporter`
- `kube-controller-manager`
- `kube-etcd`
- `kube-proxy`
- `kube-scheduler`
- `kube-state-metrics`
- `kubelet`
- `node-exporter`
- `npu-exporter`
- `nvidia-dcgm-exporter`
- `mx-exporter`
- `prometheus-kube-prometheus-alertmanager`
- `prometheus-kube-prometheus-operator`
- `prometheus-kube-prometheus-prometheus`
