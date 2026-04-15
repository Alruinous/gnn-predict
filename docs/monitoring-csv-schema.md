# 监控 CSV Schema 说明

## 这张表记录什么

这份 CSV 记录的是“某个模型变体在某个实验阶段上的一条监控汇总结果”。

- 一行只对应一个 `variant_name` 和一个 `phase`
- 同一个变体通常会出现两行
  - 一行是 `training`
  - 一行是 `inference`
- 训练和推理共用同一张表，所以读取时必须先看 `phase`

这张表的数据来自两部分：

- 实验结果文档：提供变体名、模型名、阶段开始结束时间、训练轮数或推理迭代轮数
- Prometheus 监控数据：提供该阶段时间窗口内的 CPU、内存、GPU 指标，并在导出时汇总成统计值

## 读取这张表时先记住的规则

- `phase_rounds` 是统一的“阶段轮数”字段
  - `training` 行表示训练 epoch 数
  - `inference` 行表示推理测量阶段的 iteration 数
- `duration_sec / phase_rounds` 可以直接得到“该阶段平均每轮耗时”
- 所有带 `_avg`、`_max`、`_p95` 后缀的字段，都是对该阶段监控时间窗口内样本做聚合后的结果
- `cpu_cores_pct_of_total_avg` 和 `memory_gb_pct_of_total_avg` 是相对整机资源的平均占比，不是相对 Pod request/limit 的占比
- 只有拿到有效监控样本的阶段才会写入 CSV；没有 CPU 或内存数据的阶段不会生成最终行

## 统计口径

### 行粒度

- 一行表示一个变体的一个阶段
- 不表示单个 batch、单个 step、单次 GPU 采样

### 时间窗口

- `started_at_ts` 和 `ended_at_ts` 定义了该阶段的监控窗口
- `duration_sec` 等于 `ended_at_ts - started_at_ts`
- CPU 使用率查询会额外考虑 rate 窗口
  - 正常情况下，从“阶段开始时间 + CPU rate 窗口”开始统计
  - 如果阶段本身比 rate 窗口还短，就把 CPU 查询起点收缩到阶段结束时间
- 内存和 GPU 指标直接使用完整阶段窗口

### 样本处理

- `null`、`NaN` 这类无效样本会被忽略
- 每个指标族最终都会汇总成平均值、最大值和 95 分位
- `sample_count` 取 CPU、内存、GPU 各指标有效样本数中的最小值，表示这行数据共同可用的样本深度

### 统计后缀

| 后缀 | 含义 |
| --- | --- |
| `_avg` | 该阶段窗口内的平均值 |
| `_max` | 该阶段窗口内的最大值 |
| `_p95` | 该阶段窗口内的 95 分位值 |

## 字段分组说明

### 1. 关联与定位字段

| 字段 | 含义 | 如何获得 |
| --- | --- | --- |
| `target_name` | 当前监控目标的名字 | 来自监控配置里的目标名，用来区分不同导出对象 |
| `result_json` | 本行对应的实验结果文件路径 | 来自监控目标绑定的结果 JSON 路径 |
| `config_path` | 生成该结果文件时使用的实验配置路径 | 来自实验结果文档本身 |
| `variant_name` | 变体名 | 来自结果文档中的单个变体记录 |
| `base_model_name` | 该变体所属的基础模型名 | 来自结果文档中的变体信息 |
| `namespace` | 实验 Pod 所在命名空间 | 来自监控配置 |
| `node_name` | 预期运行节点名 | 来自监控配置，并会用 Pod 元信息做一次校验 |
| `pod_name` | 监控的 Pod 名称 | 来自监控配置 |
| `gpu_node` | 实验结果里记录的 GPU 节点类型或标签 | 来自结果文档 |
| `gpu_id` | 期望监控的 GPU 编号 | 来自监控配置，用来筛选 GPU 指标 |

### 2. 阶段与时间字段

| 字段 | 含义 | 如何获得 |
| --- | --- | --- |
| `phase` | 当前行属于 `training` 还是 `inference` | 根据结果文档中的阶段记录生成 |
| `started_at_ts` | 阶段开始时间，Unix 秒时间戳 | 来自该阶段的实验时间记录 |
| `ended_at_ts` | 阶段结束时间，Unix 秒时间戳 | 来自该阶段的实验时间记录 |
| `duration_sec` | 阶段总耗时，单位秒 | 由结束时间减开始时间得到 |
| `phase_rounds` | 统一的阶段轮数 | 训练行取 epoch 数，推理行取 measurement iteration 数 |

### 3. 监控样本与标签解析字段

| 字段 | 含义 | 如何获得 |
| --- | --- | --- |
| `sample_count` | 该行最终可共同使用的样本数 | 取 CPU、内存、GPU 各指标有效样本数的最小值 |
| `resolved_gpu_label` | 监控系统实际解析到的 GPU 标签 | 来自 GPU 指标返回的 `gpu` 标签 |
| `resolved_device_label` | 监控系统实际解析到的设备标签 | 来自 GPU 指标返回的 `device` 标签 |

这三个字段主要用于判断数据可靠性和标签匹配情况。

- `gpu_id` 是配置里想监控的 GPU
- `resolved_gpu_label` 是 Prometheus 查询真正返回的 GPU 标签
- 二者一致时，说明这行 GPU 数据确实来自目标 GPU

### 4. CPU 字段

| 字段 | 含义 | 如何获得 |
| --- | --- | --- |
| `cpu_cores_avg` | 阶段内 Pod CPU 平均占用核数 | 对阶段窗口内 Pod CPU 使用率样本求平均 |
| `cpu_cores_max` | 阶段内 Pod CPU 占用核数峰值 | 对阶段窗口内 Pod CPU 使用率样本求最大值 |
| `cpu_cores_p95` | 阶段内 Pod CPU 占用核数 95 分位 | 对阶段窗口内 Pod CPU 使用率样本求 95 分位 |
| `cpu_cores_pct_of_total_avg` | Pod 平均 CPU 占用占整机 CPU 总核数的百分比 | `cpu_cores_avg / 节点总 CPU 核数 * 100` |

这里的 CPU 数据反映的是该 Pod 在实验阶段里的实际 CPU 使用量，不是容器 request/limit。

### 5. 内存字段

| 字段 | 含义 | 如何获得 |
| --- | --- | --- |
| `memory_gb_avg` | 阶段内 Pod 平均工作集内存，单位 GB | 对阶段窗口内 Pod 内存样本求平均，并换算成 GB |
| `memory_gb_max` | 阶段内 Pod 工作集内存峰值，单位 GB | 对阶段窗口内 Pod 内存样本求最大值，并换算成 GB |
| `memory_gb_p95` | 阶段内 Pod 工作集内存 95 分位，单位 GB | 对阶段窗口内 Pod 内存样本求 95 分位，并换算成 GB |
| `memory_gb_pct_of_total_avg` | Pod 平均内存占整机总内存的百分比 | `memory_gb_avg / 节点总内存 * 100` |

这里的内存口径是 Pod 在阶段窗口内的实际工作集内存。

### 6. GPU 指标族

下列每个指标族都会生成三列：

- `<prefix>_avg`
- `<prefix>_max`
- `<prefix>_p95`

这些值都来自目标 Pod 在目标 GPU 上、且落在当前阶段时间窗口内的 GPU 监控样本。

| 指标族前缀 | 语义 | 常见理解 |
| --- | --- | --- |
| `gpu_util_percent` | GPU 总体利用率 | 越高表示 GPU 越忙 |
| `gpu_sm_active_percent` | SM 活跃比例 | 更偏向计算核心实际活跃程度 |
| `gpu_sm_occupancy_percent` | SM 占用率 | 更偏向内核占满程度 |
| `gpu_mem_used_mb` | 已用显存 | 按 exporter 返回值汇总，列名按 MB 命名 |
| `gpu_mem_free_mb` | 空闲显存 | 按 exporter 返回值汇总，列名按 MB 命名 |
| `gpu_mem_copy_util_percent` | 显存拷贝链路利用率 | 反映显存搬运忙碌程度 |
| `gpu_dram_active_percent` | DRAM 活跃比例 | 反映显存带宽使用强度 |
| `gpu_pcie_tx_mb_per_sec` | PCIe 发送带宽 | 由原始字节速率换算到 MB/s 量级 |
| `gpu_pcie_rx_mb_per_sec` | PCIe 接收带宽 | 由原始字节速率换算到 MB/s 量级 |
| `gpu_power_watts` | GPU 功耗 | 单位瓦特 |
| `gpu_temp_celsius` | GPU 温度 | 单位摄氏度 |

例如：

- `gpu_util_percent_avg` 表示阶段内 GPU 利用率平均值
- `gpu_power_watts_max` 表示阶段内 GPU 功耗峰值
- `gpu_temp_celsius_p95` 表示阶段内 GPU 温度 95 分位

## 推荐读取方式

后续智能体读取这张表时，建议按下面顺序理解：

1. 先看 `variant_name`、`phase`、`base_model_name`
2. 再看 `duration_sec` 和 `phase_rounds`
3. 根据需要计算 `duration_sec / phase_rounds`
4. 再看 `cpu_*`、`memory_*`、`gpu_*` 的资源使用画像
5. 最后用 `sample_count`、`resolved_gpu_label`、`resolved_device_label` 判断监控样本是否可靠

## 常见误读

- `phase_rounds` 只是统一的“阶段轮数”，训练和推理的轮数含义不同，不能把 epoch 和 iteration 当成同一种业务动作
- `cpu_cores_pct_of_total_avg` 不是 Pod CPU 利用率，也不是单核利用率，它是相对整机 CPU 总核数的平均占比
- `memory_gb_pct_of_total_avg` 不是显存占比，它是相对整机系统内存的平均占比
- `sample_count` 不是训练 batch 数，也不是推理 iteration 数，它只是监控样本深度
- GPU 的 `_avg/_max/_p95` 都是阶段内的时间聚合结果，不是单次模型调用的即时值
