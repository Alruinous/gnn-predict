# GNN 预测器 Qwen 数据集复训与 Stacker 评估记录

## 结论

- 本次基于旧 v3 数据增加 Qwen 样本，生成 `data/extracted_v3_qwen`、`data/scaled_v3_qwen` 和 `data/scalers_v3_qwen`。
- 旧 v3 数据为 `19,915` 条；本次纳入 Qwen `144` 条，总计 `20,059` 条。
- Qwen3.5 的 `22` 条 quality-valid 行未纳入：当前 `onnx_tool` 路径无法解析其 BFLOAT16 `Constant` tensor，且首个 Qwen3.5 ONNX 图膨胀到 `186,107` 个节点。
- GNN 完成 100 epoch，best epoch 为 `72`，best val loss 为 `0.08507294207811356`。
- Single GNN test original-scale overall WAPE 为 `0.101650`，R2 为 `0.967094`。
- Full-feature residual stacker test overall WAPE 为 `0.074885`，但它是 validation-fitted 后处理，不是单一 GNN predictor。
- Qwen test 子集只有 `29` 条，single GNN WAPE 为 `0.165638`，full-feature residual stacker WAPE 为 `0.160164`，明显难于 non-Qwen 子集。

## 产物

| 项目 | 路径 |
| --- | --- |
| extracted data | `data/extracted_v3_qwen` |
| scaled data | `data/scaled_v3_qwen` |
| scalers | `data/scalers_v3_qwen` |
| effective config | `output/gnn_qwen_full_retrain_20260606/effective_config.yaml` |
| result JSON | `output/gnn_qwen_full_retrain_20260606/gnn_model_scaled_v3_qwen_20260606/results/gnn_model_scaled_v3_qwen_20260606_1780741767_result.json` |
| stacker metrics JSON | `output/gnn_qwen_full_retrain_20260606/gnn_model_scaled_v3_qwen_20260606/results/stacker_evaluation_20260606.json` |
| checkpoint | `output/gnn_qwen_full_retrain_20260606/gnn_model_scaled_v3_qwen_20260606/checkpoints/best_model.pt` |
| train log | `output/gnn_qwen_full_retrain_20260606/gnn_model_scaled_v3_qwen_20260606/logs/run_1780741767.log` |

## 数据生成口径

- 参考 `.vscode/launch.json` 中 `extract` 和 `scaler` 任务。
- 旧数据来源复用 `data/extracted_v3`，这是当前主线旧数据的 extracted split，确认不含 Qwen。
- Qwen 监控来源为 `csv_v2/qwen_monitor.csv`；旧监控 CSV 当前位于 `csv_v2/v100/`。
- 临时 here-doc 脚本在运行期注册 `IsNaNNode`，用于补齐 Qwen2/Qwen2.5/Qwen3 ONNX 图中的 `IsNaN` shape/value inference。
- Qwen 样本按 seed `42` 切为 train/val/test，再追加到既有 v3 split，以保持旧 split 可比性。
- Scaler 使用现有 `gnn_model.data.scaler`，输入为 `data/extracted_v3_qwen` 下全部 `.pt` split。

Scaler 命令：

```bash
PYTHONPATH=src uv run python -m gnn_model.data.scaler \
  --data_dir data/extracted_v3_qwen \
  --scaler_output_path data/scalers_v3_qwen \
  --scaled_data_output_path data/scaled_v3_qwen \
  --target_fields duration_sec_avg,cpu_cores_p95,memory_delta_gb_p95,gpu_util_percent_p95,gpu_sm_active_percent_p95,gpu_sm_occupancy_percent_p95,gpu_mem_used_mb_p95
```

## 数据状态

| split | total | Qwen | training | inference | prefill |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 12,035 | 86 | 5,941 | 6,052 | 42 |
| val | 4,012 | 29 | 1,995 | 2,000 | 17 |
| test | 4,012 | 29 | 1,999 | 1,994 | 19 |

Qwen 纳入情况：

| 项目 | 数量 |
| --- | ---: |
| raw Qwen CSV rows | 182 |
| quality-valid rows | 166 |
| included Qwen rows | 144 |
| excluded Qwen3.5 rows | 22 |
| GPU quality filtered rows | 16 |

纳入的 Qwen base model：

| base model | rows |
| --- | ---: |
| `Qwen2-0.5B` | 22 |
| `Qwen2-1.5B` | 22 |
| `Qwen2.5-0.5B` | 22 |
| `Qwen2.5-1.5B` | 22 |
| `Qwen3-0.6B` | 22 |
| `Qwen3-1.7B` | 22 |
| `Qwen2.5-3B` | 6 |
| `Qwen3-4B` | 6 |

验证：

- `data/extracted_v3_qwen` 和 `data/scaled_v3_qwen` 均无 NaN。
- Graph feature shape 仍为 `[1, 30]`。
- Node feature dim 为 `22`，edge feature dim 为 `15`，target dim 为 `7`。

## 训练口径

训练命令：

```bash
PYTHONPATH=src uv run python -m gnn_model \
  --config output/gnn_qwen_full_retrain_20260606/effective_config.yaml \
  --output_dir output/gnn_qwen_full_retrain_20260606 \
  --device cuda:0
```

训练配置：

| 项目 | 值 |
| --- | ---: |
| device | `cuda:0` |
| batch_size | 32 |
| num_epochs | 100 |
| learning_rate | 0.0008 |
| weight_decay | 0.0001 |
| loss_weights | `[1, 1, 1, 1, 1, 2, 1]` |
| hidden_dim | 128 |
| num_layers | 2 |
| num_heads | 8 |
| readout_mode | `mean_sum_max` |
| structural_context_mode | `basic` |
| parameter_count | 979,328 |

训练结果：

| best epoch | best val loss | last train loss |
| ---: | ---: | ---: |
| 72 | 0.085073 | 0.083677 |

耗时：

| stage | start UTC | end UTC | duration sec |
| --- | --- | --- | ---: |
| data | `2026-06-06T10:29:27.616263+00:00` | `2026-06-06T10:29:36.726876+00:00` | 9.110612 |
| model_build | `2026-06-06T10:29:36.726979+00:00` | `2026-06-06T10:29:36.990259+00:00` | 0.263280 |
| training | `2026-06-06T10:29:36.990314+00:00` | `2026-06-06T11:07:49.458750+00:00` | 2,292.468436 |
| evaluation | `2026-06-06T11:07:49.458864+00:00` | `2026-06-06T11:07:52.707981+00:00` | 3.249116 |
| full | `2026-06-06T10:29:27.616263+00:00` | `2026-06-06T11:07:52.708318+00:00` | 2,305.092056 |

## 评估标准

- Single GNN 主标准：test split original-scale WAPE，越低越好。
- 辅助标准：MAE、RMSE、R2、max abs error，以及 per-target WAPE。
- Stacker 标准：与 single GNN 分开报告；residual regressor 只用 validation split 拟合，test split 只评估。
- Stacker 主候选：full-feature residual stacker；prediction-only residual stacker 作为轻量消融。
- Direct tree baseline 只作为诊断项；它没有使用 GNN prediction residual 结构。

Stacker 口径：

| 项目 | 值 |
| --- | --- |
| regressor | `HistGradientBoostingRegressor(loss="absolute_error", max_leaf_nodes=31, random_state=7)` |
| fit split | validation |
| eval split | test |
| prediction-only feature dim | 7 |
| full-feature dim | 841 |
| category maps | train split only |
| clipping | positive targets clip to `>=0`; utilization clip to `0..100`; SM active/occupancy clip to `0..1` |

## Test Overall

### Single GNN

| scale | MAE | MSE | RMSE | R2 | WAPE | max abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| normalized | 0.164204 | 1.025523 | 1.012681 | 0.878863 | 0.208518 | 37.054882 |
| original | 49.334240 | 63,259.808594 | 251.515030 | 0.967094 | 0.101650 | 10,422.500977 |

### GNN 与 Stacker 对比

| model | all WAPE | all R2 | Qwen WAPE | Qwen R2 | non-Qwen WAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw GNN | 0.101650 | 0.967094 | 0.165638 | 0.956339 | 0.099963 |
| prediction-only residual stacker | 0.080782 | 0.969610 | 0.163052 | 0.957902 | 0.078613 |
| full-feature residual stacker | 0.074885 | 0.970126 | 0.160164 | 0.958028 | 0.072637 |
| direct full-feature tree | 0.078187 | 0.966878 | 0.234801 | 0.921805 | 0.074058 |

结论：

- Full-feature residual stacker 相对 raw GNN 的 overall WAPE 从 `0.101650` 降到 `0.074885`，相对下降 `26.33%`。
- Prediction-only residual stacker 已有明显收益，说明 GNN 原始量纲预测里的 residual 结构可被 validation tree 学到。
- Direct tree 在整体上可竞争，但 Qwen WAPE 恶化到 `0.234801`，不适合作为 Qwen 场景主候选。

## Test By Target

### Single GNN Original Scale

| target | MAE | RMSE | R2 | WAPE | max abs error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `duration_sec_avg` | 0.005112 | 0.011428 | 0.982842 | 0.064194 | 0.287066 |
| `cpu_cores_p95` | 0.054186 | 0.246494 | 0.997997 | 0.022819 | 4.708128 |
| `memory_delta_gb_p95` | 0.248020 | 1.184385 | 0.723548 | 0.126070 | 16.711750 |
| `gpu_util_percent_p95` | 6.516607 | 10.533306 | 0.859187 | 0.113146 | 65.910622 |
| `gpu_sm_active_percent_p95` | 0.054153 | 0.089572 | 0.897623 | 0.112277 | 0.552168 |
| `gpu_sm_occupancy_percent_p95` | 0.029828 | 0.056259 | 0.835218 | 0.158184 | 0.461812 |
| `gpu_mem_used_mb_p95` | 338.431763 | 665.361755 | 0.888824 | 0.101490 | 10,422.500977 |

### Full-Feature Residual Stacker Original Scale

| target | MAE | RMSE | R2 | WAPE | max abs error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `duration_sec_avg` | 0.002930 | 0.010377 | 0.985851 | 0.036793 | 0.270722 |
| `cpu_cores_p95` | 0.038146 | 0.217268 | 0.998443 | 0.016064 | 4.769548 |
| `memory_delta_gb_p95` | 0.225970 | 1.144141 | 0.742015 | 0.114861 | 16.690563 |
| `gpu_util_percent_p95` | 5.713811 | 10.461716 | 0.861094 | 0.099208 | 66.909775 |
| `gpu_sm_active_percent_p95` | 0.043722 | 0.086336 | 0.904885 | 0.090650 | 0.579007 |
| `gpu_sm_occupancy_percent_p95` | 0.025589 | 0.055901 | 0.837307 | 0.135707 | 0.457830 |
| `gpu_mem_used_mb_p95` | 248.360107 | 633.958862 | 0.899071 | 0.074479 | 10,327.366211 |

### Per-Target WAPE Comparison

| target | raw GNN | pred-only stacker | full stacker | direct tree |
| --- | ---: | ---: | ---: | ---: |
| `duration_sec_avg` | 0.064194 | 0.039558 | 0.036793 | 0.040170 |
| `cpu_cores_p95` | 0.022819 | 0.016549 | 0.016064 | 0.578893 |
| `memory_delta_gb_p95` | 0.126070 | 0.125869 | 0.114861 | 0.129457 |
| `gpu_util_percent_p95` | 0.113146 | 0.103493 | 0.099208 | 0.100304 |
| `gpu_sm_active_percent_p95` | 0.112277 | 0.096357 | 0.090650 | 0.093691 |
| `gpu_sm_occupancy_percent_p95` | 0.158184 | 0.142857 | 0.135707 | 0.139303 |
| `gpu_mem_used_mb_p95` | 0.101490 | 0.080404 | 0.074479 | 0.077414 |

## Qwen 表现分析

Qwen test split：

| phase | count |
| --- | ---: |
| `prefill` | 19 |
| `training` | 10 |

| base model | count |
| --- | ---: |
| `Qwen2-0.5B` | 4 |
| `Qwen2-1.5B` | 5 |
| `Qwen2.5-0.5B` | 6 |
| `Qwen2.5-1.5B` | 4 |
| `Qwen3-0.6B` | 4 |
| `Qwen3-1.7B` | 2 |
| `Qwen3-4B` | 4 |

Qwen per-target：

| target | raw WAPE | full stacker WAPE | raw R2 | full stacker R2 |
| --- | ---: | ---: | ---: | ---: |
| `duration_sec_avg` | 0.324369 | 0.313253 | 0.559430 | 0.584278 |
| `cpu_cores_p95` | 0.012664 | 0.029615 | -4,277.649902 | -165,453.437500 |
| `memory_delta_gb_p95` | 0.322363 | 0.333179 | 0.664218 | 0.641461 |
| `gpu_util_percent_p95` | 0.347222 | 0.350618 | -0.039855 | -0.124215 |
| `gpu_sm_active_percent_p95` | 0.384092 | 0.381470 | 0.093629 | 0.082833 |
| `gpu_sm_occupancy_percent_p95` | 0.484255 | 0.500972 | -0.208516 | -0.340741 |
| `gpu_mem_used_mb_p95` | 0.164523 | 0.158983 | 0.731150 | 0.741554 |

Qwen by phase：

| phase | count | raw WAPE | full stacker WAPE | raw R2 | full stacker R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prefill` | 19 | 0.126053 | 0.122722 | 0.970217 | 0.970841 |
| `training` | 10 | 0.265494 | 0.254614 | 0.913599 | 0.918554 |

Qwen by base model：

| base | count | raw WAPE | full stacker WAPE | raw R2 | full stacker R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Qwen2-0.5B` | 4 | 0.327336 | 0.318440 | 0.845863 | 0.857795 |
| `Qwen2-1.5B` | 5 | 0.148510 | 0.144224 | 0.964159 | 0.965841 |
| `Qwen2.5-0.5B` | 6 | 0.182910 | 0.165055 | 0.944373 | 0.952300 |
| `Qwen2.5-1.5B` | 4 | 0.032414 | 0.025045 | 0.998529 | 0.998742 |
| `Qwen3-0.6B` | 4 | 0.304448 | 0.299282 | 0.881086 | 0.884024 |
| `Qwen3-1.7B` | 2 | 0.211544 | 0.215056 | 0.942806 | 0.941094 |
| `Qwen3-4B` | 4 | 0.112124 | 0.114586 | 0.982579 | 0.981349 |

判断：

- Qwen training 明显比 prefill 难，raw WAPE 为 `0.265494`，full stacker 后仍为 `0.254614`。
- Qwen 的主要误差来自 duration、memory delta、GPU util、SM active 和 SM occupancy；CPU WAPE 很低但 R2 极端负值，说明该子集 CPU 目标方差很小，R2 不稳定。
- Full-feature stacker 对 Qwen 只有轻微改善，说明当前 validation residual tree 的收益主要来自 non-Qwen 旧分布。
- Qwen2.5-1.5B 和 Qwen3-4B 在 test 子集上相对容易；Qwen2-0.5B、Qwen3-0.6B 更难，但每个 base 样本数只有 `2-6` 条，不能过度解释。

## 与旧 v3 复训对比

| 指标 | 旧 v3 复训 20260601 | 本次 Qwen single GNN | 变化 |
| --- | ---: | ---: | ---: |
| original-scale WAPE | 0.092068 | 0.101650 | +10.41% |
| original-scale R2 | 0.970300 | 0.967094 | -0.003205 |
| best val loss | 0.075801 | 0.085073 | +12.23% |

说明：

- 本次 single GNN 比旧 v3 复训差，主要是数据集新增了更难的 Qwen/prefill 分布，同时 scaler 也随新数据重拟合。
- 本次 non-Qwen WAPE 为 `0.099963`，仍高于旧 v3 复训整体 `0.092068`，说明差异不只来自 test 中的 29 条 Qwen。
- Full-feature residual stacker 的 `0.074885` 低于旧 single GNN，但不能作为单 GNN 主模型能力比较。

## Epoch Loss Trace 选段

| epoch | train loss | val loss |
| ---: | ---: | ---: |
| 1 | 0.286129 | 0.244615 |
| 2 | 0.203411 | 0.205878 |
| 3 | 0.179460 | 0.194089 |
| 4 | 0.176017 | 0.181673 |
| 5 | 0.172313 | 0.168064 |
| 6 | 0.150924 | 0.192716 |
| 7 | 0.136335 | 0.132267 |
| 8 | 0.147247 | 0.164003 |
| 9 | 0.145251 | 0.123742 |
| 10 | 0.128471 | 0.133237 |
| 21 | 0.108474 | 0.114695 |
| 31 | 0.101378 | 0.101760 |
| 41 | 0.091755 | 0.104628 |
| 51 | 0.085832 | 0.094755 |
| 61 | 0.084765 | 0.103090 |
| 71 | 0.081081 | 0.086730 |
| 72 | 0.081670 | 0.085073 |
| 81 | 0.077102 | 0.093412 |
| 91 | 0.078157 | 0.094155 |
| 100 | 0.083677 | 0.091197 |

完整 epoch 日志见 `output/gnn_qwen_full_retrain_20260606/gnn_model_scaled_v3_qwen_20260606/logs/run_1780741767.log`。

## 运行环境观察

- GPU：`cuda:0`，Tesla V100-PCIE-32GB。
- 训练期间出现 `torch-scatter` 未安装的 PyG 性能警告，不影响数值正确性。
- 运行期间出现 `pynvml` deprecation warning，不影响训练。
- Stacker 评估使用临时 here-doc 脚本，未新增源码或测试文件。
