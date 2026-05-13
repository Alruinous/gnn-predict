# GNN 预测器新全量数据训练评估报告

## 执行摘要

本次基于当前 `data/extracted` 的新全量数据重新生成 `data/scaled`，并训练
`src/gnn_model` 的 GNN 性能预测器。当前有效样本数为 `19,774`，相比
2026-05-04 报告中的 `18,677` 增加 `1,097` 条。

本次采用 2026-05-04 报告短轮次探索推荐的 CPU 降权配置：

- 6 个预测目标保持不变。
- `cpu_cores_p95` 的 loss weight 设为 `0.1`。
- 其余目标 loss weight 为 `1.0`。
- 模型规模仍为 `hidden_dim=128, num_layers=2, num_heads=8`。

核心结果：

- 训练跑满 `100` epoch，best checkpoint 出现在 epoch `66`。
- best val loss 为 `3.227837`，last train loss 为 `3.536780`。
- test original-scale WAPE 为 `0.090008`，低于 2026-05-04 的 `0.100086`。
- `duration_sec_avg` 改善最明显，MAE 从 `4.038074s` 降到 `1.067241s`。
- 除 `memory_gb_p95` WAPE 轻微变差外，主要目标 WAPE 均优于 2026-05-04。
- `gpt2` 是当前 test 中最困难的新模型族，mean target WAPE 为 `3.523661`。

需要注意：本次使用了 CPU loss 降权，和 2026-05-04 的完整 unweighted 训练不是严格同配置对比。
但它正是上一份报告建议的下一轮完整训练配置。

## 运行产物

| 项目 | 路径 |
| --- | --- |
| 训练配置 | `config/gnn_model/scaled_training_6targets_cpu_w0p10.yaml` |
| result JSON | `output/gnn_model_newdata_20260513_cpu_w0p10/results/gnn_model_newdata_20260513_cpu_w0p10_1778654084_result.json` |
| 扩展指标 | `output/gnn_model_newdata_20260513_cpu_w0p10/results/extended_metrics.json` |
| checkpoint | `output/gnn_model_newdata_20260513_cpu_w0p10/checkpoints/best_model.pt` |
| 日志 | `output/gnn_model_newdata_20260513_cpu_w0p10/logs/run_1778654084.log` |

## 数据集

当前 `csv/` 原始行数为 `19,796`，`data/extracted` 有效样本数为 `19,774`。

| split | 样本数 | target shape |
| --- | ---: | --- |
| train | 11,864 | `(1, 6)` |
| val | 3,955 | `(1, 6)` |
| test | 3,955 | `(1, 6)` |

相比 2026-05-04：

| 项目 | 2026-05-04 | 2026-05-13 | 变化 |
| --- | ---: | ---: | ---: |
| train | 11,207 | 11,864 | +657 |
| val | 3,735 | 3,955 | +220 |
| test | 3,735 | 3,955 | +220 |
| total | 18,677 | 19,774 | +1,097 |

新增覆盖主要来自四类新模型：

| CSV | 有效样本数 |
| --- | ---: |
| `efficientnet_monitor.csv` | 576 |
| `gpt2_monitor.csv` | 456 |
| `inception_monitor.csv` | 289 |
| `swin_monitor.csv` | 216 |

这四类合计 `1,537` 条；当前数据中已没有上一轮报告里的 `*_02_monitor.csv`，少了 `440` 条。
因此净增为 `1,537 - 440 = 1,097`。

### CSV 覆盖

| CSV | 有效样本数 |
| --- | ---: |
| `yolov11_monitor.csv` | 7,030 |
| `yolov5_monitor.csv` | 1,789 |
| `vgg11_monitor.csv` | 1,115 |
| `resnet_monitor.csv` | 1,084 |
| `vgg19_monitor.csv` | 988 |
| `vgg16_monitor.csv` | 975 |
| `beit_monitor.csv` | 902 |
| `densenet_monitor.csv` | 793 |
| `bert_monitor.csv` | 651 |
| `mobilenet_monitor.csv` | 648 |
| `efficientnet_monitor.csv` | 576 |
| `yolov9_monitor.csv` | 477 |
| `convnext_monitor.csv` | 470 |
| `gpt2_monitor.csv` | 456 |
| `resnet101_monitor.csv` | 400 |
| `resnet152_monitor.csv` | 400 |
| `inception_monitor.csv` | 289 |
| `bert_large_monitor.csv` | 272 |
| `vit_monitor.csv` | 243 |
| `swin_monitor.csv` | 216 |

### phase 分布

| phase | 样本数 |
| --- | ---: |
| inference | 9,932 |
| training | 9,842 |

### split 间 variant 重叠

| overlap | 重叠 `variant_name` 数 |
| --- | ---: |
| train / val | 2,377 |
| train / test | 2,400 |
| val / test | 740 |

当前仍是行级随机拆分。同一变体可能跨 split 出现，test 仍不能视为严格的未见架构泛化评估。

### 原始目标分布

| target | min | mean | median | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `duration_sec_avg` | 0.001189 | 11.685823 | 0.036011 | 35.698720 | 255.045456 |
| `cpu_cores_p95` | 0.988000 | 4.388535 | 1.000000 | 21.200300 | 38.915001 |
| `memory_gb_p95` | 0.820000 | 2.537332 | 1.505000 | 6.748000 | 18.100000 |
| `gpu_util_percent_p95` | 0.000000 | 57.479031 | 56.000000 | 99.000000 | 100.000000 |
| `gpu_sm_occupancy_percent_p95` | 0.000000 | 0.185344 | 0.162000 | 0.448000 | 0.709000 |
| `gpu_mem_used_mb_p95` | 0.000000 | 3245.630808 | 2799.000000 | 6305.000000 | 14049.000000 |

### scaler 参数

| target | center | scale |
| --- | ---: | ---: |
| `duration_sec_avg` | 0.036011 | 18.504848 |
| `cpu_cores_p95` | 1.000000 | 0.001000 |
| `memory_gb_p95` | 1.505000 | 1.453000 |
| `gpu_util_percent_p95` | 56.000000 | 60.000000 |
| `gpu_sm_occupancy_percent_p95` | 0.162000 | 0.197000 |
| `gpu_mem_used_mb_p95` | 2799.000000 | 2606.000000 |

`cpu_cores_p95` 的 scale 仍约为 `0.001`，继续支持 CPU target 降权的必要性。

## 训练过程

| 阶段 | 起止时间 UTC | 耗时 |
| --- | --- | ---: |
| data load | 06:34:44 - 06:34:51 | 6.506s |
| model build | 06:34:51 - 06:34:51 | 0.596s |
| training | 06:34:51 - 07:07:30 | 1,959.231s |
| evaluation | 07:07:30 - 07:07:33 | 2.176s |
| full run | 06:34:44 - 07:07:33 | 1,968.509s |

平均训练耗时约 `19.592s/epoch`。设备为 `cuda:0`。

关键 epoch：

| epoch | train loss | val loss | 说明 |
| ---: | ---: | ---: | --- |
| 1 | 52.648848 | 38.691196 | CPU 降权后初始 loss 明显低于上一轮 unweighted |
| 10 | 5.228458 | 4.314149 | 前 10 轮快速收敛 |
| 20 | 4.817925 | 3.977223 | 进入低值震荡区间 |
| 33 | 4.225256 | 3.586039 | 中段刷新 |
| 46 | 4.012289 | 3.277813 | 后半程继续收益 |
| 66 | 4.012146 | 3.227837 | best checkpoint |
| 80 | 3.800267 | 3.373814 | 接近 best |
| 88 | 3.725995 | 3.245636 | late-stage 接近 best |
| 99 | 3.566686 | 3.309249 | 最后一段未刷新 |
| 100 | 3.536780 | 3.617651 | 使用 epoch 66 checkpoint 评估 |

## Test 指标

### split 稳定性

| split | original WAPE | mean target WAPE | R2 | P95 abs error |
| --- | ---: | ---: | ---: | ---: |
| train | 0.085869 | 0.125951 | 0.979853 | 283.999329 |
| val | 0.086861 | 0.131379 | 0.978308 | 284.545410 |
| test | 0.090008 | 0.129508 | 0.977378 | 291.013306 |

train/val/test 指标接近，没有明显训练集独好。但由于 split 仍有 `variant_name` 重叠，这只能说明当前行级
split 下稳定。

### 整体指标

| criterion | normalized | original-scale |
| --- | ---: | ---: |
| MAE | 29.565556 | 49.593859 |
| RMSE | 239.671234 | 212.725114 |
| WAPE | 0.054028 | 0.090008 |
| R2 | - | 0.977378 |
| mean target WAPE | - | 0.129508 |
| median abs error | - | 0.219472 |
| P90 abs error | - | 125.201685 |
| P95 abs error | - | 291.013306 |
| max abs error | 7146.368164 | 5782.589600 |

整体 original-scale 指标混合了秒、CPU 核、GB、百分比、MB 等不同单位；更适合看 WAPE、R2 和分目标指标。

### 分目标 original-scale 指标

| target | MAE | RMSE | WAPE | R2 | sMAPE | P95 abs error | max abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `duration_sec_avg` | 1.067241s | 2.870496s | 0.091107 | 0.981445 | 0.997060 | 4.368903s | 40.540802s |
| `cpu_cores_p95` | 0.176481 | 0.587098 | 0.041230 | 0.993211 | 0.010943 | 1.143950 | 7.146702 |
| `memory_gb_p95` | 0.717369 GB | 1.729132 GB | 0.282153 | 0.527508 | 0.191169 | 3.465764 GB | 16.370265 GB |
| `gpu_util_percent_p95` | 5.843985 | 10.069232 | 0.102087 | 0.885689 | 0.134520 | 21.038543 | 91.857979 |
| `gpu_sm_occupancy_percent_p95` | 0.031646 | 0.054520 | 0.170774 | 0.852630 | 0.257900 | 0.118786 | 0.518697 |
| `gpu_mem_used_mb_p95` | 289.726431 MB | 520.959574 MB | 0.089698 | 0.919849 | 0.119833 | 920.649463 MB | 5782.589600 MB |

`duration_sec_avg` 的 sMAPE 和 MAPE 口径偏大，原因是大量 inference 样本真实耗时极小，比例误差分母过小。
该目标更应同时看 MAE、RMSE、WAPE 和 phase 分组。

## 与 2026-05-04 对比

2026-05-04 是完整数据 unweighted 训练；本次是新全量数据加 `cpu_w0p10`。二者不能只用
best val loss 直接比较，但 original-scale test 指标可观察业务效果。

| target | 2026-05-04 MAE | 2026-05-13 MAE | 2026-05-04 WAPE | 2026-05-13 WAPE | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| `duration_sec_avg` | 4.038074s | 1.067241s | 0.331406 | 0.091107 | 明显改善 |
| `cpu_cores_p95` | 0.211776 | 0.176481 | 0.046838 | 0.041230 | 改善 |
| `memory_gb_p95` | 0.731422 GB | 0.717369 GB | 0.280431 | 0.282153 | MAE 略好，WAPE 略差 |
| `gpu_util_percent_p95` | 6.619610 | 5.843985 | 0.112553 | 0.102087 | 改善 |
| `gpu_sm_occupancy_percent_p95` | 0.035566 | 0.031646 | 0.187957 | 0.170774 | 改善 |
| `gpu_mem_used_mb_p95` | 319.218872 MB | 289.726431 MB | 0.098909 | 0.089698 | 改善 |

整体 original-scale：

| 指标 | 2026-05-04 | 2026-05-13 |
| --- | ---: | ---: |
| MAE | 55.142548 | 49.593859 |
| RMSE | 239.449371 | 212.725114 |
| WAPE | 0.100086 | 0.090008 |
| max abs error | 6601.460938 | 5782.589600 |

相比 2026-05-04 no-02 对照实验，本次新增了四类新模型并使用 CPU 降权：

- `duration_sec_avg` MAE 从 no-02 的 `1.348078s` 进一步降到 `1.067241s`。
- `cpu_cores_p95`、`memory_gb_p95`、`gpu_util_percent_p95`、`gpu_sm_occupancy_percent_p95` 也略好。
- `gpu_mem_used_mb_p95` MAE 从 no-02 的 `282.316772 MB` 升到 `289.726431 MB`，略差。
- overall WAPE 从 no-02 的 `0.087108` 到本次 `0.090008`，略差；考虑新增了 GPT-2、EfficientNet、
  Inception、Swin，这个退化幅度可接受。

## 分 phase 评估

| phase | count | original WAPE | R2 | duration MAE | duration WAPE | gpu_mem MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| inference | 1,990 | 0.074367 | 0.986219 | 0.130304s | 9.669785 | 232.612630 MB |
| training | 1,965 | 0.104974 | 0.969044 | 2.016099s | 0.085560 | 347.566870 MB |

inference 的 duration WAPE 极高，但 MAE 只有 `0.130304s`。这是因为许多 inference 样本真实耗时接近零，
比例误差被分母放大。training 阶段的整体 WAPE 更高，主要来自更重的耗时和显存尾部。

## 分模型族评估

按 test split 的 mean target WAPE 排序，最困难的模型族如下：

| family | count | overall WAPE | mean target WAPE | R2 |
| --- | ---: | ---: | ---: | ---: |
| `gpt2` | 86 | 0.514356 | 3.523661 | 0.621351 |
| `swin` | 35 | 0.067641 | 0.571774 | 0.993606 |
| `inception` | 62 | 0.091953 | 0.412157 | 0.967226 |
| `efficientnet` | 114 | 0.075086 | 0.361425 | 0.988694 |
| `resnet` | 187 | 0.254205 | 0.253819 | 0.888608 |

较稳定的模型族：

| family | count | overall WAPE | mean target WAPE | R2 |
| --- | ---: | ---: | ---: | ---: |
| `yolov11` | 1,418 | 0.079245 | 0.072897 | 0.978741 |
| `beit` | 159 | 0.102642 | 0.077358 | 0.973656 |
| `yolov5` | 363 | 0.097745 | 0.086878 | 0.980180 |
| `bert` | 129 | 0.074102 | 0.094233 | 0.988355 |

GPT-2 是新增域里最明显的弱点：

| target | GPT-2 MAE | GPT-2 WAPE |
| --- | ---: | ---: |
| `duration_sec_avg` | 0.518159s | 18.710000 |
| `cpu_cores_p95` | 0.000096 | 0.000096 |
| `memory_gb_p95` | 0.038276 GB | 0.034806 |
| `gpu_util_percent_p95` | 3.158355 | 0.276035 |
| `gpu_sm_occupancy_percent_p95` | 0.009712 | 1.600083 |
| `gpu_mem_used_mb_p95` | 257.493432 MB | 0.520947 |

GPT-2 的 duration 和 SM occupancy WAPE 主要受极小真实值影响，但显存 WAPE `0.520947` 说明该域仍需要更多样本或更好的结构特征。

## 判断

本次训练流程可信：

- 当前 `data/scaled` 已由新的 `data/extracted` 重新生成。
- 训练完整跑完 100 epoch。
- checkpoint、result JSON、扩展指标均已落盘。
- train/val/test original WAPE 和 R2 接近，没有观察到明显训练崩坏。

本次模型效果相对 2026-05-04 有实际改善：

- 新增四类模型后，总样本增加 `5.9%`，overall WAPE 仍从 `0.100086` 降到 `0.090008`。
- CPU 降权显著缓解了 normalized loss 被 `cpu_cores_p95` 主导的问题。
- `duration_sec_avg` 从上一轮最弱目标变成当前改善最大的目标。

主要风险仍然存在：

- 行级 split 造成同一 `variant_name` 跨 split，test 指标偏乐观。
- scaler 仍由 train/val/test 全量共同拟合，存在 test 分布信息进入归一化的问题。
- 新增模型族中 GPT-2 的 test 表现明显偏弱，尤其是显存和极小耗时比例误差。
- 当前对比混合了数据变化与 loss 权重变化，若要严格归因，需要在同一数据集上补一轮 unweighted 100 epoch 或固定 test/scaler 做 A/B。

当前推荐：

- 保留 `cpu_w0p10` 作为下一轮默认完整训练基线。
- 优先对 GPT-2 做独立诊断，至少看 GPT-2 的显存标签分布、图特征分布和训练/推理 phase 差异。
- 下一轮严格评估应改成按 `variant_name` 或模型族分组 split，并只在 train split 上拟合 scaler。
