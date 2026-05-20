# GNN 预测器困难族群过滤复训评估报告

## 执行摘要

本次根据 `current_blockers_2026-05-19.md` 做了一次临时数据消融实验：

- 过滤规则排除 `deepfm`、`dcn`、`edcn`、`gpt2`、`inception`、`swin`、`t5`。
- 保留 `dcnv2`，作为推荐模型族中的稳定对照。
- 参照 `.vscode/launch.json` 的 `extract` 和 `scaler` 任务，从当前 `csv/` 重新生成临时数据集和临时归一化器。
- 训练配置不设置 `loss_weights`，6 个目标仍然等权训练。
- 未使用 `cpu_p0`、`cpu_w0p10` 或其他 CPU 降权方案。

本次效果变化明显，已记录为 v2 复训结果。相对 2026-05-19 全量等权基线，test original-scale 指标变化如下：

| 指标 | 全量等权基线 | 过滤困难族群 | 变化 |
| --- | ---: | ---: | ---: |
| MAE | 89.404152 | 60.243626 | -32.62% |
| MSE | 476,781.968750 | 81,654.476562 | -82.87% |
| RMSE | 690.494003 | 285.752474 | -58.62% |
| WAPE | 0.131380 | 0.098012 | -25.40% |
| R2 | 0.888367 | 0.970299 | +0.081932 |
| mean target WAPE | 0.124826 | 0.127030 | +1.77% |
| P95 abs error | 280.232544 | 306.760986 | +9.47% |
| max abs error | 16,614.105469 | 7,203.835938 | -56.64% |

主要结论：

- 移除困难族群后，整体 MSE/RMSE 大幅下降，说明全量基线的平方误差主要被困难族群和显存尾部拉高。
- WAPE 和 R2 同时改善，说明不是只优化了单个大单位目标。
- mean target WAPE 和 P95 abs error 没有同步变好，说明小分母目标和中高分位误差仍然存在。
- 这不是一个可直接替代全量模型的结论，而是确认 blockers 中困难族群判断有效的消融证据。

## 运行产物

| 项目 | 路径 |
| --- | --- |
| 临时过滤脚本 | `/tmp/gnn_model_20260519_filtered_pipeline.py`，执行后已删除 |
| 临时工作目录 | `/tmp/gnn_model_filtered_20260519`，执行后已删除 |
| 临时训练配置 | `/tmp/gnn_model_filtered_20260519/filtered_unweighted_config.yaml`，随临时目录删除 |
| result JSON | `output/gnn_model_newdata_20260519_filtered_no_hard/results/gnn_model_newdata_20260519_filtered_no_hard_1779191638_result.json` |
| 扩展指标 | `output/gnn_model_newdata_20260519_filtered_no_hard/results/extended_metrics.json` |
| checkpoint | `output/gnn_model_newdata_20260519_filtered_no_hard/checkpoints/best_model.pt` |
| 日志 | `output/gnn_model_newdata_20260519_filtered_no_hard/logs/run_1779191638.log` |

## 过滤口径

| 项目 | 内容 |
| --- | --- |
| 过滤 family | `dcn`、`deepfm`、`edcn`、`gpt2`、`inception`、`swin`、`t5` |
| 实际移除 CSV | `dcn_monitor.csv`、`edcn_monitor.csv`、`inception_monitor.csv`、`swin_monitor.csv` |
| 保留推荐模型对照 | `dcnv2_monitor.csv` |

说明：本次按当前 `csv/` 目录重新执行 extract。过滤规则覆盖 blockers 中列出的七个 family，但当前
`csv/` 下实际存在并被移除的源 CSV 是 `dcn`、`edcn`、`inception`、`swin` 四个 family。

保留 CSV：

| CSV | 样本数 |
| --- | ---: |
| `beit_monitor.csv` | 902 |
| `bert_large_monitor.csv` | 272 |
| `bert_monitor.csv` | 651 |
| `convnext_monitor.csv` | 470 |
| `dcnv2_monitor.csv` | 768 |
| `densenet_monitor.csv` | 793 |
| `efficientnet_monitor.csv` | 576 |
| `mobilenet_monitor.csv` | 648 |
| `resnet101_monitor.csv` | 400 |
| `resnet152_monitor.csv` | 400 |
| `resnet_monitor.csv` | 1,084 |
| `vgg11_monitor.csv` | 1,115 |
| `vgg16_monitor.csv` | 975 |
| `vgg19_monitor.csv` | 988 |
| `vit_monitor.csv` | 243 |
| `yolov11_monitor.csv` | 7,030 |
| `yolov5_monitor.csv` | 1,789 |
| `yolov9_monitor.csv` | 477 |

## 临时数据集

| split | 样本数 | target shape |
| --- | ---: | --- |
| train | 11,749 | `(1, 6)` |
| val | 3,916 | `(1, 6)` |
| test | 3,916 | `(1, 6)` |
| total | 19,581 | - |

phase 分布：

| phase | 样本数 |
| --- | ---: |
| inference | 9,815 |
| training | 9,766 |

当前仍是行级随机 split，存在 `variant_name` 重叠：

| overlap | 重叠 `variant_name` 数 |
| --- | ---: |
| train / val | 2,389 |
| train / test | 2,347 |
| val / test | 760 |

### 原始目标分布

| target | min | mean | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `duration_sec_avg` | 0.001189 | 11.805155 | 0.152456 | 28.428055 | 35.812275 | 255.045456 |
| `cpu_cores_p95` | 0.999000 | 4.421937 | 1.000000 | 16.843000 | 21.643999 | 38.915001 |
| `memory_gb_p95` | 0.820000 | 2.577042 | 1.536000 | 5.099999 | 6.748000 | 18.100000 |
| `gpu_util_percent_p95` | 0.000000 | 60.612942 | 60.000000 | 98.000000 | 99.000000 | 100.000000 |
| `gpu_sm_occupancy_percent_p95` | 0.000000 | 0.209291 | 0.183000 | 0.441000 | 0.548000 | 0.861000 |
| `gpu_mem_used_mb_p95` | 0.000000 | 3,571.968750 | 2,951.000000 | 6,073.000000 | 7,755.000000 | 19,963.000000 |

### 临时 scaler 参数

| target | center | scale |
| --- | ---: | ---: |
| `duration_sec_avg` | 0.152456 | 18.527922 |
| `cpu_cores_p95` | 1.000000 | 0.001000 |
| `memory_gb_p95` | 1.536000 | 1.426000 |
| `gpu_util_percent_p95` | 60.000000 | 63.000000 |
| `gpu_sm_occupancy_percent_p95` | 0.183000 | 0.226000 |
| `gpu_mem_used_mb_p95` | 2,951.000000 | 2,884.000000 |

这里需要注意：过滤后 `cpu_cores_p95` 的 scaler scale 回到约 `0.001`。因此本次 normalized loss
不可直接与全量等权基线的 normalized loss 做优劣比较。

## 训练过程

| 项目 | 值 |
| --- | ---: |
| epoch | 100 |
| best epoch | 87 |
| best val loss | 28.481274 |
| last train loss | 37.680214 |
| batch size | 32 |
| learning rate | 0.001 |
| weight decay | 0.0001 |
| device | `cuda:0` |

| 阶段 | 起止时间 UTC | 耗时 |
| --- | --- | ---: |
| data load | 11:53:58 - 11:54:07 | 8.437s |
| model build | 11:54:07 - 11:54:07 | 0.250s |
| training | 11:54:07 - 12:27:22 | 1,995.028s |
| evaluation | 12:27:22 - 12:27:24 | 1.853s |
| full run | 11:53:58 - 12:27:24 | 2,005.568s |

## Test 指标

### 整体指标

| criterion | count | MAE | MSE | RMSE | WAPE | R2 | mean target WAPE | P95 abs error | max abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| normalized | 3,916 | 27.933926 | 44,979.945312 | 212.084760 | 0.050723 | 0.995557 | 0.230316 | 1.332833 | 4,821.834473 |
| original-scale | 3,916 | 60.243626 | 81,654.476562 | 285.752474 | 0.098012 | 0.970299 | 0.127030 | 306.760986 | 7,203.835938 |

normalized 指标被 `cpu_cores_p95` 的极小 scaler scale 放大，不适合单独判断模型质量。

### split 稳定性

| split | MAE | MSE | RMSE | WAPE | R2 | mean target WAPE | P95 abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 56.042534 | 71,384.406250 | 267.178604 | 0.092595 | 0.972827 | 0.122393 | 300.849884 |
| val | 59.060452 | 81,406.984375 | 285.319092 | 0.096407 | 0.969769 | 0.126709 | 312.621887 |
| test | 60.243626 | 81,654.476562 | 285.752474 | 0.098012 | 0.970299 | 0.127030 | 306.760986 |

train/val/test 的 original-scale WAPE 和 R2 接近，没有观察到训练集独好。

### 分目标 original-scale 指标

| target | MAE | MSE | RMSE | WAPE | R2 | P95 abs error | max abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `duration_sec_avg` | 1.127023 | 6.671934 | 2.583009 | 0.095409 | 0.984716 | 4.517703 | 36.651093 |
| `cpu_cores_p95` | 0.166689 | 0.269903 | 0.519522 | 0.038753 | 0.994776 | 1.138961 | 4.822060 |
| `memory_gb_p95` | 0.714556 | 2.813674 | 1.677401 | 0.277415 | 0.555540 | 2.857274 | 16.249506 |
| `gpu_util_percent_p95` | 5.549187 | 94.620247 | 9.727294 | 0.091774 | 0.894041 | 21.236107 | 95.844170 |
| `gpu_sm_occupancy_percent_p95` | 0.033872 | 0.003387 | 0.058196 | 0.160765 | 0.878740 | 0.133769 | 0.570083 |
| `gpu_mem_used_mb_p95` | 353.870422 | 489,822.468750 | 699.873180 | 0.098064 | 0.914590 | 1,207.914673 | 7,203.835938 |

显存目标的 MSE 从全量基线的 `2,860,482.750000` 降到 `489,822.468750`，仍是整体 MSE 的最大来源，
但尾部压力明显下降。

## 分 phase 评估

| phase | count | MAE | MSE | RMSE | WAPE | R2 | mean target WAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| inference | 1,957 | 51.883007 | 61,322.890625 | 247.634591 | 0.085979 | 0.977305 | 2.917467 |
| training | 1,959 | 68.595703 | 101,965.296875 | 319.320054 | 0.109602 | 0.963532 | 0.119262 |

inference 的 mean target WAPE 仍然很高，主要是小分母目标导致。training 的 MSE 更高，仍主要来自显存误差。

## 保留族群中的高误差样本

| family | test count | overall WAPE | mean target WAPE | R2 | MSE | duration MAE | gpu mem MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `efficientnet` | 113 | 0.083256 | 1.084257 | 0.988295 | 12,441.834961 | 0.819145 | 221.688889 |
| `dcnv2` | 174 | 0.133507 | 0.847145 | 0.960407 | 633,602.562500 | 1.003814 | 1,355.115479 |
| `resnet` | 192 | 0.200604 | 0.254672 | 0.932421 | 9,304.357422 | 0.624351 | 190.992752 |
| `mobilenet` | 144 | 0.082488 | 0.199471 | 0.988372 | 4,734.910156 | 0.607175 | 136.420731 |
| `yolov11` | 1,416 | 0.082620 | 0.075178 | 0.980434 | 34,383.593750 | 1.099879 | 260.255127 |

过滤后 `dcnv2` 不再被 `deepfm/dcn/edcn` 的大显存尾部遮蔽，反而成为保留族群中需要继续关注的推荐模型对照。

## 与全量等权基线对比

| run | 数据量 | 过滤 | best epoch | best val loss | test original MAE | test original MSE | test original RMSE | test original WAPE | test R2 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-05-19 全量等权 | 22,510 | 无 | 96 | 0.077473 | 89.404152 | 476,781.968750 | 690.494003 | 0.131380 | 0.888367 |
| 2026-05-19 v2 过滤困难族群 | 19,581 | blockers 困难族群 | 87 | 28.481274 | 60.243626 | 81,654.476562 | 285.752474 | 0.098012 | 0.970299 |

best val loss 的绝对值不能直接比较，因为本次临时 scaler 使 `cpu_cores_p95` 的 normalized scale 回到约 `0.001`。
更可靠的对比是 original-scale 的分目标指标、整体 MSE/RMSE/WAPE 和 R2。

## 判断

本次复训验证了 `current_blockers_2026-05-19.md` 的核心判断：

- 困难族群会显著拉高整体 MSE/RMSE，尤其是推荐模型显存尾部。
- 移除这些族群后，整体 original-scale MSE 从 `476,781.968750` 降到 `81,654.476562`。
- 过滤后的 `gpu_mem_used_mb_p95` max 从全量基线的 `28,347 MB` 数据分布压力降到 `19,963 MB`。
- 但过滤不能解决所有问题，`dcnv2`、`efficientnet` 等保留族群仍有较高 mean target WAPE。
- 过滤后 `cpu_cores_p95` scaler scale 回到极小值，说明数据分布和 scaler 策略仍需单独处理。

当前建议：

- 保留本次结果作为困难族群消融实验，不作为全量生产基线。
- 后续若继续训练全量模型，应优先处理 `gpu_mem_used_mb_p95` 尾部和 `cpu_cores_p95` scaler 稳定性。
- `dcnv2` 应继续作为推荐模型对照，而不是和 `deepfm/dcn/edcn` 一起移除。
- 小分母族群仍应拆出 MAE/RMSE 与相对误差口径，避免只用 WAPE 判断模型质量。
