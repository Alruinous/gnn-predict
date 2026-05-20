# GNN 预测器新数据训练评估报告

## 执行摘要

本次使用当前已生成的 `data/scaled` 重新训练 `src/gnn_model` 中的 GNN 性能预测器，目标从旧训练的 4 项扩展为 6 项：

- `duration_sec_avg`
- `cpu_cores_p95`
- `memory_gb_p95`
- `gpu_util_percent_p95`
- `gpu_sm_occupancy_percent_p95`
- `gpu_mem_used_mb_p95`

训练完整跑完 100 epoch，使用 `cuda:0` 上的 `Tesla V100-SXM2-32GB`。训练输出和临时配置均位于 `/tmp`，训练结束后已删除；本文档保留本次运行的关键结果。

核心结论：

- 新数据训练耗时合理，平均约 `21.0s/epoch`，旧训练约 `8.42s/epoch`；考虑样本数约翻倍且目标从 4 项增至 6 项，本次没有发现“训练异常过快”的运行迹象。
- 模型在当前 test split 上整体表现不错，尤其是 `duration_sec_avg`、`gpu_util_percent_p95`、`gpu_mem_used_mb_p95`。
- 当前评估口径偏乐观：split 间存在大量 `variant_name` 重叠，scaler 由 train/val/test 共同拟合，不能视为严格泛化评估。
- `cpu_cores_p95` 的 RobustScaler `scale` 约为 `0.001`，导致 normalized loss 被 CPU 目标强烈主导；新训练的总 loss 不适合直接和旧 4 目标模型比较。

## 运行配置

本次没有修改仓库配置文件。运行时使用临时 YAML，等价配置如下：

```yaml
experiment_name: gnn_model_scaled_train_newdata_eval
data:
  kind: split
  data_dir: data/scaled
  target_names:
    - duration_sec_avg
    - cpu_cores_p95
    - memory_gb_p95
    - gpu_util_percent_p95
    - gpu_sm_occupancy_percent_p95
    - gpu_mem_used_mb_p95
  scaler_dir: data/scalers
model:
  hidden_dim: 128
  num_layers: 2
  num_heads: 8
  dropout_rate: 0.1
training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
```

输入数据校验通过：

| split | 样本数 | target shape |
| --- | ---: | --- |
| train | 10,943 | `(1, 6)` |
| val | 3,648 | `(1, 6)` |
| test | 3,648 | `(1, 6)` |

`data/scalers/target_scalers.pkl` 包含 6 个目标的 scaler。

## 与旧结果对比

旧结果路径：

`output/gnn_model_scaled_train/results/gnn_model_scaled_train_1776751342_result.json`

| 项目 | 旧 4 目标训练 | 新 6 目标训练 |
| --- | ---: | ---: |
| train 样本数 | 5,365 | 10,943 |
| val 样本数 | 1,789 | 3,648 |
| test 样本数 | 1,789 | 3,648 |
| epoch 数 | 100 | 100 |
| batch size | 32 | 32 |
| 训练耗时 | 841.989s | 2,107.647s |
| 平均 epoch 耗时 | 8.420s | 21.000s |
| best epoch | 94 | 97 |
| best val loss | 0.102618 | 24.163776 |
| last train loss | 0.100498 | 31.388226 |

旧训练只包含 4 个目标：

- `duration_sec_avg`
- `gpu_util_percent_p95`
- `gpu_sm_occupancy_percent_p95`
- `gpu_mem_used_mb_p95`

新训练加入：

- `cpu_cores_p95`
- `memory_gb_p95`

因此，旧模型和新模型的 normalized aggregate loss 不能直接比较。

## 训练过程观察

训练日志显示 loss 起初很大，随后快速下降并进入震荡平台：

| epoch | train loss | val loss | 说明 |
| ---: | ---: | ---: | --- |
| 1 | 514.667368 | 389.803741 | CPU normalized 目标主导 loss |
| 10 | 49.492902 | 45.931503 | 快速下降后仍明显震荡 |
| 30 | 35.504347 | 28.819204 | 中期刷新低点 |
| 47 | 33.098418 | 27.012672 | 后半程继续改善 |
| 65 | 32.548918 | 24.225344 | 重要低点 |
| 97 | 30.966661 | 24.163776 | 最终 best checkpoint |
| 100 | 31.388226 | 28.339039 | 最后一轮未刷新 |

训练曲线特征：

- `val_loss` 波动较大，常在 `27-46` 区间跳动。
- best checkpoint 来自 epoch 97，说明 100 epoch 仍有实际收益。
- 大 loss 主要来自 `cpu_cores_p95` 的归一化尺度，不代表所有业务目标都很差。

## Test 指标

以下是现有代码在 test split 上生成的指标。官方性能以这些 test 指标为准。

### 整体指标

| 指标 | normalized | original-scale |
| --- | ---: | ---: |
| MAE | 24.857582 | 45.803040 |
| RMSE | 239.985245 | 206.203964 |
| WAPE | 0.046974 | 0.081658 |
| max abs error | 18,905.882812 | 7,369.947266 |

整体 original-scale 指标混合了秒、CPU 核、GB、百分比、MB 等不同单位，只适合粗略观察；分目标指标更有解释价值。

### 分目标 original-scale 指标

| target | MAE | RMSE | WAPE | max abs error |
| --- | ---: | ---: | ---: | ---: |
| `duration_sec_avg` | 1.166572s | 2.564277s | 0.087414 | 29.668396s |
| `cpu_cores_p95` | 0.148305 cores | 0.587868 | 0.035546 | 18.906767 |
| `memory_gb_p95` | 0.735548 GB | 1.602536 | 0.275455 | 14.856695 GB |
| `gpu_util_percent_p95` | 5.576684 | 9.086764 | 0.095764 | 91.084503 |
| `gpu_sm_occupancy_percent_p95` | 0.028414 | 0.047279 | 0.150980 | 0.318450 |
| `gpu_mem_used_mb_p95` | 267.162720 MB | 505.003326 MB | 0.081282 | 7,369.947266 MB |

### 分目标 normalized 指标

| target | MAE | RMSE | WAPE | max abs error |
| --- | ---: | ---: | ---: | ---: |
| `duration_sec_avg` | 0.058078 | 0.127662 | 0.087507 | 1.477041 |
| `cpu_cores_p95` | 148.297577 | 587.840393 | 0.046752 | 18,905.882812 |
| `memory_gb_p95` | 0.454322 | 0.989831 | 0.542450 | 9.176464 |
| `gpu_util_percent_p95` | 0.097837 | 0.159417 | 0.228465 | 1.597974 |
| `gpu_sm_occupancy_percent_p95` | 0.139283 | 0.231757 | 0.252028 | 1.561029 |
| `gpu_mem_used_mb_p95` | 0.098402 | 0.186005 | 0.178900 | 2.714529 |

`cpu_cores_p95` 的 normalized MAE/RMSE 非常大，但 original-scale MAE 只有 `0.1483` cores。原因是该目标的 scaler scale 极小。

## 数据诊断

### CSV 覆盖情况

当前 `csv/` 中进入数据集的记录数：

| CSV | 样本数 |
| --- | ---: |
| `beit_monitor.csv` | 885 |
| `bert_large_monitor.csv` | 272 |
| `bert_monitor.csv` | 651 |
| `convnext_monitor.csv` | 470 |
| `densenet_monitor.csv` | 793 |
| `mobilenet_monitor.csv` | 648 |
| `resnet101_monitor.csv` | 400 |
| `resnet152_monitor.csv` | 400 |
| `resnet_monitor.csv` | 1,086 |
| `vgg11_monitor.csv` | 1,115 |
| `vgg16_monitor.csv` | 991 |
| `vgg19_monitor.csv` | 988 |
| `vit_monitor.csv` | 244 |
| `yolov11_monitor.csv` | 7,030 |
| `yolov5_monitor.csv` | 1,789 |
| `yolov9_monitor.csv` | 477 |

当前已有 YOLO CSV：

- `yolov11_monitor.csv`
- `yolov5_monitor.csv`
- `yolov9_monitor.csv`

当前缺失 YOLO CSV：

- `yolov8_monitor.csv`
- `yolov10_monitor.csv`

YOLO 样本数：

| split | YOLO 样本数 |
| --- | ---: |
| train | 5,487 |
| val | 1,901 |
| test | 1,908 |

总 YOLO 样本数为 `9,296 / 18,239`，约 `51.0%`。

### phase 分布

| split | inference | training |
| --- | ---: | ---: |
| train | 5,526 | 5,417 |
| val | 1,806 | 1,842 |
| test | 1,803 | 1,845 |

训练和推理样本在各 split 中基本均衡。

### split 间 variant 重叠

| overlap | 重叠 `variant_name` 数 |
| --- | ---: |
| train / val | 2,107 |
| train / test | 2,147 |
| val / test | 720 |

这说明当前 `extract` 使用行级随机拆分，训练/推理阶段的同一变体或相近变体容易跨 split。这个会让 test 指标偏乐观，尤其是结构特征强相关的目标。

### 目标原始分布

| target | min | mean | median | max |
| --- | ---: | ---: | ---: | ---: |
| `duration_sec_avg` | 0.001190 | 13.173631 | 0.050174 | 255.045456 |
| `cpu_cores_p95` | 0.999000 | 4.286780 | 1.000000 | 20.125999 |
| `memory_gb_p95` | 0.820000 | 2.702714 | 1.533000 | 18.385000 |
| `gpu_util_percent_p95` | 0.000000 | 59.255653 | 57.000000 | 100.000000 |
| `gpu_sm_occupancy_percent_p95` | 0.000000 | 0.192226 | 0.164000 | 0.709000 |
| `gpu_mem_used_mb_p95` | 0.000000 | 3322.246176 | 2879.000000 | 14049.000000 |

`duration_sec_avg` 和 `cpu_cores_p95` 都有明显偏斜。耗时中位数只有 `0.050174s`，但最大值达到 `255.045456s`。

### scaler 参数

| target | center | scale |
| --- | ---: | ---: |
| `duration_sec_avg` | 0.050174 | 20.086379 |
| `cpu_cores_p95` | 1.000000 | 0.001000 |
| `memory_gb_p95` | 1.533000 | 1.619000 |
| `gpu_util_percent_p95` | 57.000000 | 57.000000 |
| `gpu_sm_occupancy_percent_p95` | 0.164000 | 0.204000 |
| `gpu_mem_used_mb_p95` | 2879.000000 | 2715.000000 |

`cpu_cores_p95` 的 scale 极小，原因是大量样本的 CPU p95 聚集在 `1.0` 附近。使用未加权 SmoothL1Loss 时，该目标在 normalized 空间会压倒其他目标。

## 对“过于优良”的判断

本次没有发现训练循环本身异常快或明显跳过数据：

- 新训练确实跑满 100 epoch。
- 每个 epoch 约 `19-22s`。
- 训练总耗时约 `35.1min`。
- 数据量、target dim、scaler 均通过训练前校验。

但当前结果不能视为严格泛化性能，主要风险如下：

- 行级随机拆分导致大量 `variant_name` 跨 split。
- 同一模型结构的 `training` 和 `inference` 行可能分属不同 split。
- scaler 由 `data/extracted/*.pt` 全部 split 共同拟合，test 分布信息进入了归一化参数。
- CPU target 的 RobustScaler scale 极小，使 normalized aggregate loss 和 best val loss 难解释。
- 当前 YOLO 数据只覆盖 YOLO11、YOLOv5、YOLOv9，尚未覆盖 YOLOv8、YOLOv10。

因此，旧结果“特别好”更可能来自数据拆分和归一化评估口径偏乐观，而不是训练代码直接存在跳过训练或标签泄漏到特征的明显漏洞。

## 建议

下一轮应优先修正评估口径：

- 按 `variant_name` 分组拆分，保证同一变体不会跨 train/val/test。
- 或按 `base_model_name` / 模型族做更严格的 holdout，评估跨模型族泛化。
- scaler 只在 train split 上拟合，再应用到 val/test。
- 对 `cpu_cores_p95` 单独处理：可考虑目标变换、裁剪异常 scale、按目标加权 loss，或改用 original-scale 友好的评价策略。
- 补齐 YOLOv8、YOLOv10 后重新 extract/scaler/train，避免检测模型域覆盖不完整。

当前模型在现有 split 下可用，但不能作为最终泛化结论。
