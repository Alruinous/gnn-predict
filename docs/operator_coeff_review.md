# 算子性能表征重建设计

## 结论

旧项目里的 `OPERATOR_PERFORMANCE_COEFF`、`COMPUT_COEFF`、`MEMORY_COEFF`、
`BANDWIDTH_COEFF`、`LATENCY_COEFF` 应整体废弃。它们的问题不是数值不准，而是问题定义不准：
四个手工命名的相对分数无法稳定表达 GPU 上一个 ONNX 算子的真实行为。

新方案不再追求“把旧 dict 测准”。新目标是建立一套可观测、可复现、可扩展的算子性能画像：

- 静态事实：ONNX op 名、参数名、shape、参数量、逻辑读写字节、估算 FLOPs。
- 动态事实：在指定 GPU 和 shape 下实测 latency、throughput、显存流量、kernel 行为。
- 派生表征：由实验数据学习出来的 shape-cluster 画像和可用于 GNN 的连续特征。

最终仍可以输出 Python dict，方便其它项目读取；但 dict 的 schema 必须是新设计，不应兼容旧字段。

## 可信边界

旧 `src/data/const.py` 里可以保留或借鉴的内容只有三类：

- GPU 型号名：例如 `A100`、`V100`。
- 可查硬件规格：例如官方 FP32/TF32/FP16 峰值、HBM 带宽、显存大小、SM 数。
- ONNX 层面的名字常量：算子名、算子参数名、dtype 字节数这类可由标准或文档验证的信息。

应废弃的内容：

- `COMPUT_COEFF`、`MEMORY_COEFF`、`BANDWIDTH_COEFF`、`LATENCY_COEFF`。
- `OPERATOR_PERFORMANCE_COEFF` 的所有手工分数。
- `typical_efficiency`、`kernel_launch_overhead_us` 这类没有实验来源的手工填值。
- “计算密集型”“逐元素”“元算子”等作为性能事实使用的分类。
- 只按 op_type 给一个全局固定系数的设计。

算子性能不是 `op_type -> score`，而是：

```text
(gpu_type, op_type, shape, attrs) -> measured behavior
```

这条映射才是后续实验和建模的核心。

## 调研判断

| 来源 | 可用结论 | 对本项目的含义 |
| --- | --- | --- |
| [Roofline model, CACM 2009](https://cacm.acm.org/research/roofline-an-insightful-visual-performance-model-for-multicore-architectures/) | 用 operational intensity 和硬件峰值解释 compute-bound / memory-bound。 | 可作为解释框架，不能变成手工 roofline dict。 |
| [Nsight Compute Roofline Charts](https://docs.nvidia.com/nsight-compute/2025.2/ProfilingGuide/index.html#roofline-charts) | 官方 profile 能给 achieved value、memory boundary、peak boundary。 | 用于抽样校准，不作为全量自动流程的唯一依赖。 |
| [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | CUDA event 和 effective bandwidth 是低层性能测量基础。 | 自建 microbenchmark 时用它定义计时和带宽。 |
| [ONNX Runtime profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html) | 可输出 operator latency，CUDA profiling 可关联 kernel 耗时。 | 第一版计时工具，保留 ONNX 节点可见性。 |
| [TensorRT trtexec profiling](https://docs.nvidia.com/deeplearning/tensorrt/10.16.0/performance/best-practices.html) | 可输出 per-layer runtime 和 layer 信息。 | 说明更换部署后端需要重测，不进入第一版 schema。 |
| [CUTLASS profiler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html) | 可测 GEMM/Conv 的 bytes、FLOPs、runtime、memory、math throughput。 | 用于核心矩阵/卷积算子的上界校准。 |
| [DeepBench](https://github.com/baidu-research/DeepBench) | 低层 GEMM、Conv、RNN workload 设计成熟。 | 借鉴 shape sweep，不直接拿历史结果。 |
| [DNNMark](https://gem5.googlesource.com/public/gem5-resources/+/8a8193a69075b7baa1db438a41ef56a6bf2d4d5b/src/DNNMark) | primitive 可分离测量，也可组合测量。 | 证明单算子和组合子图应分开建模。 |
| [nn-Meter](https://github.com/microsoft/nn-Meter) | 真正可预测的单位常是 fused kernel，而不是框架 op。 | 第一版固定执行环境，避免把 ONNX op 系数伪装成跨后端规律。 |
| [DNNPerf](https://www.microsoft.com/en-us/research/publication/runtime-performance-prediction-for-deep-learning-models-with-graph-neural-network/) | 计算图节点/边语义特征可预测训练时间和显存。 | 本项目 GNN 路线合理，但节点特征应来自实测画像。 |
| [PerfSeer](https://www.ijcai.org/proceedings/2025/793) | 拓扑、节点、边、全局特征共同预测 execution time、memory、SM utilization。 | 单一 op score 不够，应构建多维画像。 |
| [NVIDIA A100 specs](https://www.nvidia.com/en-us/data-center/a100/) | A100 官方 FP32/TF32/FP16/INT8 和 HBM 带宽可查。 | 只作为硬件上界和元数据。 |
| [NVIDIA V100 datasheet](https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet.pdf) | V100 官方 FP32、Tensor Core、HBM 带宽可查。 | 只作为硬件上界和元数据。 |

核心判断：公开资料能支持实验设计，不能替代本项目实测。GPU 型号和模型 shape 分布共同决定结果。

## 新数据模型

### Hardware Catalog

硬件规格只存可查事实，不存经验效率：

```python
GPU_HARDWARE_CATALOG = {
    "A100_80GB_PCIE": {
        "vendor": "NVIDIA",
        "architecture": "Ampere",
        "memory_gb": 80,
        "sm_count": 108,
        "fp32_tflops": 19.5,
        "tf32_tensor_tflops": 156.0,
        "fp16_tensor_tflops": 312.0,
        "memory_bandwidth_gbs": 1935.0,
    },
    "V100_32GB_SXM2": {
        "vendor": "NVIDIA",
        "architecture": "Volta",
        "memory_gb": 32,
        "sm_count": 80,
        "fp32_tflops": 15.7,
        "fp16_tensor_tflops": 125.0,
        "memory_bandwidth_gbs": 900.0,
    },
}
```

### Operator Observation

一次实验样本是一条 observation，记录一个实际 ONNX 节点在目标 GPU 上的结构字段和 profiler 指标。

| 字段 | 来源 |
| --- | --- |
| `op_type` | ONNX node |
| `attrs` | ONNX node attribute |
| shape 信息 | 实际 ONNX shape inference |
| `device` | 实验机器 |
| FLOPs / bytes | 基于实际 shape 和参数计算 |
| latency / throughput / kernel 行为 | profiler 实测 |

### Operator Profile Table

最终供其它项目读取的是 profile table：

```python
OPERATOR_PROFILE_TABLE = {
    "A100_80GB_PCIE": {
        "<op_type>": {
            "schema_version": "2.0",
            "profiles": [],
        }
    }
}
```

这个表不要求每个 op 只有一个数。是否分 cluster 由实际 observation 分布决定；样本不足时不生成
结论。

### GNN Feature Schema

GNN 节点特征应从 profile table 派生，不暴露旧四系数：

```python
OPERATOR_FEATURE_SCHEMA = {
    "schema_version": "2.0",
    "features": [
        "static_flops_log2",
        "static_logical_bytes_log2",
        "static_arithmetic_intensity_log2",
        "measured_latency_us_p50_log2",
        "measured_latency_us_p95_log2",
        "measured_achieved_tflops_ratio",
        "measured_achieved_bandwidth_ratio",
        "measured_kernel_count_log2",
        "measured_zero_copy_rate",
        "measured_folded_rate",
    ],
}
```

这些字段对应真实物理含义。模型训练时可以做 ablation，决定哪些字段有效；不是先验认定四个
人工系数一定有价值。

## 实验路线

### Workload 来源

真实 workload 优先：

- 从当前 `res/` CSV 解析 ONNX 路径。
- 对每个 ONNX 做 shape inference。
- 提取 `op_type`、attrs、输入输出 shape、参数量和边连接。
- 统计每类 op 的 shape 分布。

合成 workload 用来补洞：

- 真实数据没有覆盖的 op 才生成 synthetic ONNX。
- 合成 shape 来自真实 shape 的外插，不做凭空网格爆炸。
- `Conv/Gemm/MatMul` 可以吸收 DeepBench/CUTLASS 的典型 shape sweep。
- `Reshape/Flatten/Squeeze/Unsqueeze/Shape/Constant` 必须验证是否被消除或 zero-copy。
- `Transpose/Concat/Slice/Gather/Where` 必须覆盖非连续访问和索引分布。

### 测量方式

第一版只使用 ONNX Runtime CUDA profile：

- 目标是保留 ONNX 节点级可见性。
- 输出每个 node 的 host latency 和 CUDA kernel latency。
- 禁止只记录模型总时延。

CUTLASS 或 Nsight Compute 只在核心算子结果异常时抽样校准，不作为默认 profile 生成流程。

### 实验元数据

每次 run 必须记录：

- GPU 型号、显存、SM 数、MIG 状态。
- driver、CUDA、cuDNN、ONNX Runtime 版本。
- batch size、dynamic shape profile。
- GPU clock、power limit、温度区间、是否独占 GPU。
- 原始 ONNX hash、输入数据生成种子、run id。

没有这些元数据的 profile 不进入最终表。

## 统计方法

### Shape Cluster

按 op_type 内部聚类，不跨 op 聚类。聚类特征只从实际 observation 自动派生，文档不预设具体
shape 字段或阈值。

每个 cluster 至少需要足够样本才产出 profile。样本不足的 cluster 只保留 observation，不产出结论。

### 指标计算

基础静态指标：

```text
logical_bytes = logical_read_bytes + logical_write_bytes
arithmetic_intensity = flops / max(logical_bytes, 1)
```

实测指标：

```text
achieved_tflops = flops / latency_us / 1e6
achieved_bandwidth_gbs = logical_bytes / latency_us / 1e3
```

相对硬件指标：

```text
compute_roof_ratio = achieved_tflops / selected_peak_tflops
bandwidth_roof_ratio = achieved_bandwidth_gbs / measured_memory_bandwidth_gbs
```

保留 p50/p95/IQR，不用一个均值代表全部。

### 模型验证

验证目标不是让旧系数看起来合理，而是证明新画像对 GNN 有信息增益：

- 单算子层：同 op 留出 shape，预测 latency rank 和 p50 latency。
- 子图层：验证融合前后 profile 聚合误差。
- 模型层：把 profile 派生特征接入 GNN，与不接入 profile 的 baseline 比较。
- 消融层：逐个移除 `latency`、`throughput`、`bandwidth`、`kernel_count`、`zero_copy` 类特征。

接受标准：

- profile 覆盖真实 ONNX 节点数 95% 以上。
- 主要 op 的 shape-cluster latency rank Spearman 大于 0.8。
- GNN 加入 profile 特征后，在验证集上至少一个核心目标显著提升，且其它目标不明显退化。

## 输出文件建议

建议后续生成：

```text
src/gnn_model/data/operator_hardware.py
src/gnn_model/data/operator_profiles.py
src/gnn_model/data/operator_feature_schema.py
```

职责分离：

- `operator_hardware.py` 只放硬件事实。
- `operator_profiles.py` 放实验生成的 profile table。
- `operator_feature_schema.py` 定义 GNN 如何消费 profile table。

原始实验输出放：

```text
data/operator_profile/raw/<run_id>/
data/operator_profile/processed/<profile_version>/
```

最终文档和代码只引用 `profile_version`，不把实验日志硬编码进源码。

## 废弃旧变量的迁移判断

`OPERATOR_PERFORMANCE_COEFF` 不迁移。

`OPERATOR_ROOFLINE_PARAMS_A100` 和 `OPERATOR_ROOFLINE_PARAMS_V100` 不迁移。硬件峰值进入
`GPU_HARDWARE_CATALOG`，实验效率进入 `OPERATOR_PROFILE_TABLE`。

`OP_TYPE_TO_INDEX` 可以保留，但来源应从可验证 op vocab 构建，而不是从旧性能系数字典的 key
顺序派生。

`OP_PARAMETERS` 和 `PARAMETERS_ORDER` 可保留或重建，但必须对齐 ONNX schema。参数名是静态
事实，参数值对性能的影响由 observation 学习。

## 第一版落地目标

第一版只做一件事：构建 `COMMON_OP_TYPES` 的 A100 或当前可用 GPU 上的 ONNX Runtime CUDA
profile table。

范围：

- `Conv`
- `Relu`
- `Gemm`
- `MatMul`
- `BatchNormalization`
- `Add`
- `Mul`
- `MaxPool`
- `AveragePool`
- `Reshape`
- `Transpose`
- `Concat`

产物：

- raw profiling JSON/CSV
- workload manifest
- processed observation table
- shape cluster report
- `operator_profiles.py`
- profile 特征接入前后的 GNN ablation 结果

这个版本完成后，再决定是否扩展 V100、Nsight Compute 抽样和旧项目完整 op 集。决策依据是
profile 特征是否提升预测质量，而不是旧变量是否被“修好”。
