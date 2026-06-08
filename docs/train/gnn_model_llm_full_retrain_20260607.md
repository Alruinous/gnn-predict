# GNN 预测器 LLM 数据集全量重训记录

## 结论

- 本次重新生成了带现代 LLM 变体的全量数据集：`data/extracted_llm`、`data/scaled_llm` 和 `data/scalers_llm`。
- 数据集总计 `20,143` 条，比旧 v3 的 `19,915` 条多 `228` 条 LLM 样本。
- LLM 样本构成为 Qwen `144`、Gemma `43`、LLaMA `41`。
- Qwen3.5 `36` 行按上一轮已确认口径在 extract 前排除；本轮不再重复验证 Qwen3.5。
- Single GNN 训练完成 100 epoch，best epoch 为 `79`，best val loss 为 `0.080125`。
- Single GNN test original-scale overall WAPE 为 `0.102238`，R2 为 `0.960808`。
- Full-feature residual stacker test overall WAPE 为 `0.074783`，但它是 validation-fitted 后处理，不是单一 GNN predictor。
- LLM test 子集只有 `41` 条，single GNN WAPE 为 `0.167591`；其中 Qwen `0.112953`、Gemma `0.096633`、LLaMA `0.410395`。
- 排除 LLaMA 后，Qwen+Gemma 子集 single GNN WAPE 为 `0.106187`，MAE 为 `268.594166`；相比 all LLM 的 WAPE `0.167591` 和 MAE `401.577140` 明显更好。
- LLaMA 当前表现明显偏弱，主要由 `Llama-3.2-3B` 的 4 条 test 样本拉高；该结论受样本量限制，不能过度外推。

## 产物

| 项目 | 路径 |
| --- | --- |
| extracted data | `data/extracted_llm` |
| scaled data | `data/scaled_llm` |
| scalers | `data/scalers_llm` |
| effective config | `output/gnn_llm_full_retrain_20260607/effective_config.yaml` |
| result JSON | `output/gnn_llm_full_retrain_20260607/gnn_model_scaled_llm_20260607/results/gnn_model_scaled_llm_20260607_1780812969_result.json` |
| stacker metrics JSON | `output/gnn_llm_full_retrain_20260607/gnn_model_scaled_llm_20260607/results/stacker_evaluation_20260607.json` |
| checkpoint | `output/gnn_llm_full_retrain_20260607/gnn_model_scaled_llm_20260607/checkpoints/best_model.pt` |
| train log | `output/gnn_llm_full_retrain_20260607/gnn_model_scaled_llm_20260607/logs/run_1780812969.log` |

## 数据生成口径

- 参考 `.vscode/launch.json` 中的 `extract` 和 `scaler` 任务。
- 旧主线 V100 CSV 使用 `csv_v2/v100/*.csv`。
- 新增 LLM CSV 使用 `csv_v2/qwen_monitor.csv`、`csv_v2/gemma_monitor.csv`、`csv_v2/llama_monitor.csv`。
- 临时 here-doc 脚本创建 `/tmp` 下 symlink CSV 目录，并在进程内注册 `IsNaNNode`，用于补齐普通 Qwen ONNX 图里的 `IsNaN` shape/value inference。
- `Qwen3.5-*` 行在 extract 前过滤，本轮沿用 20260606 记录中的已知不可用口径。
- 临时 CSV 目录已删除；`data/extracted_llm/manifest.json` 已记录真实输入来源。
- Scaler 使用 `gnn_model.data.scaler`，输入为 `data/extracted_llm` 下全部 `.pt` split。

Scaler 命令：

```bash
PYTHONPATH=src uv run python -m gnn_model.data.scaler \
  --data_dir data/extracted_llm \
  --scaler_output_path data/scalers_llm \
  --scaled_data_output_path data/scaled_llm \
  --target_fields duration_sec_avg,cpu_cores_p95,memory_delta_gb_p95,gpu_util_percent_p95,gpu_sm_active_percent_p95,gpu_sm_occupancy_percent_p95,gpu_mem_used_mb_p95
```

## 数据状态

| 项目 | 数量 |
| --- | ---: |
| raw CSV rows including Qwen3.5 | 20,359 |
| Qwen3.5 rows excluded before extract | 36 |
| effective extract input rows | 20,323 |
| included graph rows | 20,143 |
| GPU quality filtered rows | 162 |
| other pre-GPU filtered rows | 18 |
| high-memory-delta observations | 557 |

| split | total | non-LLM | Qwen | Gemma | LLaMA | training | inference | prefill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 12,085 | 11,939 | 92 | 29 | 25 | 5,964 | 6,039 | 82 |
| val | 4,029 | 3,988 | 30 | 5 | 6 | 1,984 | 2,023 | 22 |
| test | 4,029 | 3,988 | 22 | 9 | 10 | 2,027 | 1,984 | 18 |

LLM base model 总量：

| base model | rows |
| --- | ---: |
| `Qwen2-0.5B` | 22 |
| `Qwen2-1.5B` | 22 |
| `Qwen2.5-0.5B` | 22 |
| `Qwen2.5-1.5B` | 22 |
| `Qwen2.5-3B` | 6 |
| `Qwen3-0.6B` | 22 |
| `Qwen3-1.7B` | 22 |
| `Qwen3-4B` | 6 |
| `gemma-2-2b` | 21 |
| `gemma-3-1b-pt` | 22 |
| `Llama-3.2-1B` | 21 |
| `Llama-3.2-3B` | 20 |

LLM phase 总量：

| family | training | prefill |
| --- | ---: | ---: |
| Qwen | 66 | 78 |
| Gemma | 21 | 22 |
| LLaMA | 19 | 22 |

验证：

- `data/extracted_llm` 和 `data/scaled_llm` 均无 NaN。
- Graph feature shape 为 `[1, 30]`。
- Node feature dim 为 `22`，edge feature dim 为 `15`，target dim 为 `7`。

## 训练口径

训练命令：

```bash
PYTHONPATH=src uv run python -m gnn_model \
  --config output/gnn_llm_full_retrain_20260607/effective_config.yaml \
  --output_dir output/gnn_llm_full_retrain_20260607 \
  --device cuda:1
```

训练配置：

| 项目 | 值 |
| --- | ---: |
| device | `cuda:1` |
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
| 79 | 0.080125 | 0.075503 |

耗时：

| stage | start UTC | end UTC | duration sec |
| --- | --- | --- | ---: |
| data | `2026-06-07T06:16:09.017670+00:00` | `2026-06-07T06:16:18.113786+00:00` | 9.096116 |
| model_build | `2026-06-07T06:16:18.113886+00:00` | `2026-06-07T06:16:18.372703+00:00` | 0.258817 |
| training | `2026-06-07T06:16:18.372759+00:00` | `2026-06-07T06:55:09.817404+00:00` | 2,331.444645 |
| evaluation | `2026-06-07T06:55:09.817512+00:00` | `2026-06-07T06:55:13.131155+00:00` | 3.313643 |
| full | `2026-06-07T06:16:09.017670+00:00` | `2026-06-07T06:55:13.131222+00:00` | 2,344.113551 |

## 评估标准

- Single GNN 主标准：test split original-scale WAPE，越低越好。
- Single GNN 辅助标准：MAE、RMSE、R2、max abs error，以及 per-target WAPE。
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
| full-feature dim | 962 |
| category dim | 711 |
| clipping | positive targets clip to `>=0`; utilization clip to `0..100`; SM active/occupancy clip to `0..1` |

## Test Overall

### Single GNN

| scale | MAE | MSE | RMSE | R2 | WAPE | max abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| normalized | 0.171052 | 1.058510 | 1.028839 | 0.874717 | 0.214747 | 37.093468 |
| original | 50.899258 | 87,709.343750 | 296.157623 | 0.960808 | 0.102238 | 14,655.300781 |

### GNN 与 Stacker 对比

| model | all WAPE | all R2 | LLM WAPE | LLM R2 | Qwen WAPE | Gemma WAPE | LLaMA WAPE | non-LLM WAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw GNN | 0.102238 | 0.960808 | 0.167591 | 0.928794 | 0.112953 | 0.096633 | 0.410395 | 0.098872 |
| prediction-only residual stacker | 0.081339 | 0.963327 | 0.162954 | 0.930535 | 0.108439 | 0.088374 | 0.411414 | 0.077135 |
| full-feature residual stacker | 0.074783 | 0.963107 | 0.166969 | 0.924576 | 0.104593 | 0.095471 | 0.428573 | 0.070035 |
| direct full-feature tree | 0.082645 | 0.951369 | 0.269545 | 0.867591 | 0.149209 | 0.372119 | 0.379947 | 0.073020 |

结论：

- Full-feature residual stacker 相对 raw GNN 的 overall WAPE 从 `0.102238` 降到 `0.074783`，相对下降 `26.85%`。
- Prediction-only residual stacker 已有明显收益，说明 GNN 原始量纲预测 residual 本身含有可被 validation tree 学到的结构。
- Full-feature residual stacker 对 non-LLM WAPE 从 `0.098872` 降到 `0.070035`，收益主要来自旧分布。
- Full-feature residual stacker 对 LLM 子集没有改善，LLM WAPE 为 `0.166969`，几乎等于 raw GNN 的 `0.167591`。
- Direct tree 在整体上可竞争，但 LLM WAPE 恶化到 `0.269545`，不适合作为 LLM 场景主候选。

## Test By Target

### Single GNN Original Scale

| target | MAE | RMSE | R2 | WAPE | max abs error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `duration_sec_avg` | 0.004813 | 0.016473 | 0.965905 | 0.059772 | 0.477896 |
| `cpu_cores_p95` | 0.075602 | 0.320104 | 0.996653 | 0.031861 | 4.338432 |
| `memory_delta_gb_p95` | 0.254919 | 1.197550 | 0.709712 | 0.129058 | 16.729153 |
| `gpu_util_percent_p95` | 6.730355 | 10.474027 | 0.862769 | 0.116371 | 73.917488 |
| `gpu_sm_active_percent_p95` | 0.055352 | 0.094277 | 0.887784 | 0.114496 | 0.776480 |
| `gpu_sm_occupancy_percent_p95` | 0.030958 | 0.056654 | 0.835441 | 0.162955 | 0.453008 |
| `gpu_mem_used_mb_p95` | 349.142792 | 783.488403 | 0.892040 | 0.102028 | 14,655.300781 |

### Per-Target WAPE Comparison

| target | raw GNN | pred-only stacker | full stacker | direct tree |
| --- | ---: | ---: | ---: | ---: |
| `duration_sec_avg` | 0.059772 | 0.042720 | 0.039629 | 0.045090 |
| `cpu_cores_p95` | 0.031861 | 0.017037 | 0.015691 | 0.578577 |
| `memory_delta_gb_p95` | 0.129058 | 0.122494 | 0.116484 | 0.129881 |
| `gpu_util_percent_p95` | 0.116371 | 0.102721 | 0.100053 | 0.101646 |
| `gpu_sm_active_percent_p95` | 0.114496 | 0.096777 | 0.091210 | 0.094804 |
| `gpu_sm_occupancy_percent_p95` | 0.162955 | 0.140469 | 0.135004 | 0.140048 |
| `gpu_mem_used_mb_p95` | 0.102028 | 0.080994 | 0.074368 | 0.081949 |

## LLM 表现分析

LLM test split：

| family | count | raw WAPE | full stacker WAPE | raw R2 | full stacker R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen | 22 | 0.112953 | 0.104593 | 0.967243 | 0.967905 |
| Gemma | 9 | 0.096633 | 0.095471 | 0.982795 | 0.984938 |
| LLaMA | 10 | 0.410395 | 0.428573 | 0.696201 | 0.664050 |
| all LLM | 41 | 0.167591 | 0.166969 | 0.928794 | 0.924576 |

LLM by phase：

| phase | count | raw WAPE | full stacker WAPE | raw R2 | full stacker R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prefill` | 18 | 0.088725 | 0.087507 | 0.985268 | 0.985080 |
| `training` | 23 | 0.236678 | 0.236579 | 0.876415 | 0.868460 |

LLM by base model：

| base | count | raw WAPE | full stacker WAPE | raw R2 | full stacker R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Llama-3.2-1B` | 6 | 0.161855 | 0.175584 | 0.933375 | 0.919571 |
| `Llama-3.2-3B` | 4 | 0.730634 | 0.754544 | 0.522981 | 0.477419 |
| `Qwen2-0.5B` | 2 | 0.027920 | 0.025966 | 0.999231 | 0.999280 |
| `Qwen2-1.5B` | 2 | 0.062506 | 0.053812 | 0.995414 | 0.996645 |
| `Qwen2.5-0.5B` | 3 | 0.070114 | 0.042184 | 0.993989 | 0.997767 |
| `Qwen2.5-1.5B` | 4 | 0.048489 | 0.043032 | 0.996482 | 0.997187 |
| `Qwen2.5-3B` | 2 | 0.127875 | 0.115003 | 0.974943 | 0.982764 |
| `Qwen3-0.6B` | 2 | 0.436031 | 0.442309 | 0.714306 | 0.704976 |
| `Qwen3-1.7B` | 5 | 0.105594 | 0.091265 | 0.983734 | 0.986130 |
| `Qwen3-4B` | 2 | 0.119853 | 0.124026 | 0.981466 | 0.977897 |
| `gemma-2-2b` | 7 | 0.098850 | 0.096304 | 0.982210 | 0.984664 |
| `gemma-3-1b-pt` | 2 | 0.076925 | 0.088063 | 0.990703 | 0.986469 |

LLM per-target WAPE：

| target | Qwen raw | Qwen full | Gemma raw | Gemma full | LLaMA raw | LLaMA full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `duration_sec_avg` | 0.175061 | 0.170475 | 0.239208 | 0.261665 | 0.223788 | 0.217721 |
| `cpu_cores_p95` | 0.034519 | 0.010532 | 0.016337 | 0.003760 | 0.014937 | 0.000620 |
| `memory_delta_gb_p95` | 0.177783 | 0.177256 | 0.105105 | 0.078928 | 0.225623 | 0.226884 |
| `gpu_util_percent_p95` | 0.245995 | 0.253850 | 0.235539 | 0.197955 | 0.362444 | 0.356354 |
| `gpu_sm_active_percent_p95` | 0.314964 | 0.321113 | 0.190590 | 0.184208 | 0.681663 | 0.708964 |
| `gpu_sm_occupancy_percent_p95` | 0.409752 | 0.410263 | 0.366562 | 0.347459 | 0.629186 | 0.700693 |
| `gpu_mem_used_mb_p95` | 0.112271 | 0.103829 | 0.096215 | 0.095170 | 0.410706 | 0.428998 |

判断：

- Prefill 明显比 training 容易，LLM raw WAPE 为 `0.088725` vs `0.236678`。
- Qwen 在当前 test split 上优于上一轮 Qwen 记录，但当前 Qwen test 只有 `22` 条，且 split 已因全量重建改变，不能直接解释为模型泛化改善。
- Gemma 当前表现接近 non-LLM 整体，但 test 只有 `9` 条，且 `gemma-2-2b` 占 `7` 条。
- LLaMA 是当前 LLM 误差主要来源，尤其 `Llama-3.2-3B` 的 `4` 条 test 样本 raw WAPE 为 `0.730634`。
- LLaMA 的主要误差集中在 GPU util、SM active、SM occupancy 和 GPU memory；这说明图结构特征对当前 LLaMA 大参数 full-flow 运行形态覆盖不足。
- Full-feature stacker 对 Qwen 和 Gemma 有轻微收益或接近持平，但对 LLaMA 恶化，说明 validation residual tree 没学到可迁移到 LLaMA test 的稳定校正。
- 当前 LLaMA/Gemma 总量太小，特别是 train split 中 LLaMA 只有 `25` 条；LLaMA 弱表现很可能同时受到样本量和训练/显存分布跨度影响。

## 移除 LLaMA 后的指标

本节不是重新训练无 LLaMA 数据集，而是在同一次 20260607 checkpoint 和 test split 上过滤掉 LLaMA 样本后的评估。Stacker 指标使用同一 validation-fitted residual tree 口径重新拟合后评估；single GNN 仍是主判断口径。

移除后的数据规模：

| split | original total | removed LLaMA | remaining total | remaining LLM |
| --- | ---: | ---: | ---: | ---: |
| train | 12,085 | 25 | 12,060 | 121 |
| val | 4,029 | 6 | 4,023 | 35 |
| test | 4,029 | 10 | 4,019 | 31 |

### All Test Excluding LLaMA

| model | count | MAE | RMSE | R2 | WAPE | max abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw GNN | 4,019 | 49.000958 | 256.700484 | 0.969570 | 0.099161 | 10,567.829102 |
| prediction-only residual stacker | 4,019 | 38.528416 | 246.170686 | 0.972015 | 0.077968 | 10,372.655939 |
| full-feature residual stacker | 4,019 | 35.148396 | 240.109668 | 0.973376 | 0.071128 | 10,488.467004 |
| direct full-feature tree | 4,019 | 39.581915 | 307.256322 | 0.956404 | 0.080100 | 13,664.680847 |

与完整 test 对比：

| model | full test WAPE | excluding LLaMA WAPE | absolute change | relative change |
| --- | ---: | ---: | ---: | ---: |
| raw GNN | 0.102238 | 0.099161 | -0.003077 | -3.01% |
| full-feature residual stacker | 0.074783 | 0.071128 | -0.003655 | -4.89% |

### Qwen+Gemma Only

| model | count | MAE | RMSE | R2 | WAPE | max abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw GNN | 31 | 268.594166 | 1,055.858092 | 0.975919 | 0.106187 | 9,876.770020 |
| prediction-only residual stacker | 31 | 254.022355 | 1,043.911943 | 0.976461 | 0.100426 | 10,065.907367 |
| full-feature residual stacker | 31 | 254.539664 | 1,032.557276 | 0.976970 | 0.100631 | 10,073.538510 |
| direct full-feature tree | 31 | 639.082673 | 2,340.550611 | 0.881670 | 0.252657 | 13,664.680847 |

与 all LLM 对比：

| model | all LLM MAE | Qwen+Gemma MAE | MAE change | all LLM WAPE | Qwen+Gemma WAPE | WAPE change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw GNN | 401.577140 | 268.594166 | -33.12% | 0.167591 | 0.106187 | -36.64% |
| full-feature residual stacker | 400.087607 | 254.539664 | -36.38% | 0.166969 | 0.100631 | -39.73% |

Qwen+Gemma by phase：

| phase | model | count | MAE | RMSE | R2 | WAPE | max abs error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `prefill` | raw GNN | 14 | 181.762815 | 577.048808 | 0.992669 | 0.071417 | 3,216.748047 |
| `prefill` | full-feature residual stacker | 14 | 170.698507 | 551.711533 | 0.993299 | 0.067070 | 3,053.697022 |
| `training` | raw GNN | 17 | 340.102337 | 1,326.164994 | 0.962592 | 0.135146 | 9,876.770020 |
| `training` | full-feature residual stacker | 17 | 323.585322 | 1,301.357024 | 0.963978 | 0.128582 | 10,073.538510 |

Qwen+Gemma per-target raw GNN：

| target | MAE | WAPE | R2 |
| --- | ---: | ---: | ---: |
| `duration_sec_avg` | 0.039196 | 0.198015 | 0.655232 |
| `cpu_cores_p95` | 0.029240 | 0.029240 | 0.000000 |
| `memory_delta_gb_p95` | 2.250673 | 0.164457 | 0.859596 |
| `gpu_util_percent_p95` | 16.612103 | 0.242684 | 0.296710 |
| `gpu_sm_active_percent_p95` | 0.153632 | 0.274548 | 0.456885 |
| `gpu_sm_occupancy_percent_p95` | 0.106195 | 0.395489 | 0.492867 |
| `gpu_mem_used_mb_p95` | 1,860.968120 | 0.105605 | 0.866194 |

Qwen+Gemma per-target full-feature residual stacker：

| target | MAE | WAPE | R2 |
| --- | ---: | ---: | ---: |
| `duration_sec_avg` | 0.039402 | 0.199055 | 0.649395 |
| `cpu_cores_p95` | 0.009453 | 0.009453 | 0.000000 |
| `memory_delta_gb_p95` | 2.172172 | 0.158721 | 0.864097 |
| `gpu_util_percent_p95` | 16.780283 | 0.245141 | 0.285129 |
| `gpu_sm_active_percent_p95` | 0.155635 | 0.278129 | 0.478680 |
| `gpu_sm_occupancy_percent_p95` | 0.106051 | 0.394954 | 0.492464 |
| `gpu_mem_used_mb_p95` | 1,762.514650 | 0.100018 | 0.872035 |

判断：

- 排除 LLaMA 后，single GNN 的 all-test WAPE 从 `0.102238` 降到 `0.099161`，整体 test 只改善约 `3.01%`，因为 LLaMA test 只有 `10` 条。
- 只看 LLM 子集时改善很明显：Qwen+Gemma WAPE 为 `0.106187`，比 all LLM 的 `0.167591` 低 `36.64%`。
- Qwen+Gemma 的 MAE 为 `268.594166`，比 all LLM 的 `401.577140` 低 `33.12%`。
- Qwen+Gemma 仍然是 training 明显难于 prefill，WAPE 为 `0.135146` vs `0.071417`。
- Qwen+Gemma 的主要剩余误差仍集中在 GPU util、SM active、SM occupancy；GPU memory 的 WAPE 已接近 non-LLM 整体。

## LLaMA 问题补充

LLaMA 的问题不是单纯“模型家族不同”，而是样本量、静态图特征分布和监控目标分布同时不利。

### 样本量不足

| split | LLaMA count | `Llama-3.2-1B` | `Llama-3.2-3B` |
| --- | ---: | ---: | ---: |
| train | 25 | 13 | 12 |
| val | 6 | 2 | 4 |
| test | 10 | 6 | 4 |

其中 `Llama-3.2-3B` 的 test 只有 `4` 条，但 raw WAPE 为 `0.730634`，对 all LLM 指标影响很大。

### 静态图特征差异

LLaMA 的 ONNX 图在当前 architecture-only 口径下比 Qwen/Gemma 更小，和真实资源目标的关系更不稳定。

| feature, test mean | Qwen | Gemma | LLaMA |
| --- | ---: | ---: | ---: |
| node_count | 3,840.91 | 4,140.67 | 2,384.80 |
| edge_count | 4,498.27 | 4,838.67 | 2,826.60 |
| graph_memory_bytes | 9.065e9 | 7.808e9 | 4.625e9 |
| peak_live_activation_bytes | 245.531M | 482.672M | 150.425M |
| activation_elements_sum | 3.385e9 | 3.404e9 | 2.049e9 |
| max_tensor_dim | 151,936 | 257,365 | 128,256 |

这意味着模型看到的 LLaMA 静态图不像 Qwen/Gemma 那样处在高节点、高边、高 activation 的密集区域，但真实训练/Prefill 显存并没有按这些静态特征稳定下降。

### 目标分布异常

LLaMA test 中低 SM active/occupancy 的比例明显更高。

| condition | Qwen test | Gemma test | LLaMA test |
| --- | ---: | ---: | ---: |
| `gpu_sm_active_percent_p95 < 0.01` | 1/22 | 0/9 | 4/10 |
| `gpu_sm_occupancy_percent_p95 < 0.01` | 1/22 | 0/9 | 4/10 |
| `gpu_util_percent_p95 >= 50` 且 `gpu_sm_active_percent_p95 < 0.01` | 0/22 | 0/9 | 2/10 |
| `gpu_util_percent_p95 <= 5` 且 `gpu_mem_used_mb_p95 > 7000` | 0/22 | 0/9 | 2/10 |
| `gpu_mem_used_mb_p95 < 11000` | 7/22 | 0/9 | 3/10 |
| `gpu_mem_used_mb_p95 > 25000` | 0/22 | 7/9 | 1/10 |

这种组合对当前特征体系很难：有些 LLaMA 样本 GPU util 不低但 SM active/occupancy 接近 0；也有样本 GPU util 很低但显存仍很高。它们可能来自短训练 step、测量窗口、CUDA kernel 形态或监控采样差异。

### LLaMA 单样本误差

| base | phase | batch | sample_count | target mem MB | pred mem MB | abs mem error MB | target SM active | pred SM active | target SM occ | pred SM occ | target GPU util | pred GPU util |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Llama-3.2-1B` | training | 3 | 7 | 12,855 | 13,647.539 | 792.539 | 0.6600 | 0.6898 | 0.3490 | 0.3523 | 71.00 | 68.08 |
| `Llama-3.2-3B` | training | 1 | 5 | 8,013 | 19,993.467 | 11,980.467 | 0.5840 | 0.6485 | 0.4380 | 0.3992 | 10.00 | 50.14 |
| `Llama-3.2-3B` | prefill | 2 | 7 | 32,281 | 28,576.164 | 3,704.836 | 0.8830 | 0.9149 | 0.4330 | 0.5414 | 98.00 | 86.49 |
| `Llama-3.2-1B` | training | 4 | 11 | 12,795 | 16,878.805 | 4,083.805 | 0.0040 | 0.7139 | 0.0030 | 0.3433 | 83.00 | 70.71 |
| `Llama-3.2-1B` | prefill | 4 | 6 | 13,311 | 19,564.076 | 6,253.076 | 0.8940 | 0.8707 | 0.4490 | 0.4527 | 98.00 | 93.40 |
| `Llama-3.2-3B` | training | 1 | 18 | 9,633 | 23,436.674 | 13,803.674 | 0.0030 | 0.7795 | 0.0030 | 0.3742 | 3.00 | 53.93 |
| `Llama-3.2-1B` | prefill | 3 | 3 | 12,855 | 12,590.018 | 264.982 | 0.7290 | 0.6328 | 0.4540 | 0.4041 | 81.00 | 73.03 |
| `Llama-3.2-3B` | training | 1 | 11 | 10,569 | 25,224.320 | 14,655.320 | 0.0040 | 0.7774 | 0.0040 | 0.3758 | 3.00 | 66.33 |
| `Llama-3.2-1B` | training | 2 | 11 | 12,969 | 12,192.541 | 776.459 | 0.0030 | 0.6072 | 0.0020 | 0.3359 | 52.00 | 64.34 |
| `Llama-3.2-1B` | prefill | 1 | 15 | 12,851 | 13,267.436 | 416.436 | 0.8230 | 0.8058 | 0.5050 | 0.4652 | 98.00 | 87.65 |

判断：

- `Llama-3.2-1B` 的多数 prefill 样本预测尚可。
- `Llama-3.2-3B` 的 training 样本是主要问题：真实显存为 `8-10.6GB`，但 GNN 预测到 `20-25GB`。
- 多条 LLaMA training 样本真实 SM active/occupancy 接近 0，但预测值接近常规高负载训练样本。
- 这说明当前图特征没有表达“本次训练测量窗口里实际 kernel 活跃度很低”这类运行态信息。
- 如果后续继续保留 LLaMA，应优先补充 `Llama-3.2-3B` training 的重复采样和更多 batch/seq 组合，并复核这些低 SM/high util 或低 util/high memory 的监控记录。

## 与旧记录对比

| 指标 | 旧 v3 20260601 | Qwen 20260606 | 本次 LLM 20260607 |
| --- | ---: | ---: | ---: |
| dataset rows | 19,915 | 20,059 | 20,143 |
| single GNN WAPE | 0.092068 | 0.101650 | 0.102238 |
| single GNN R2 | 0.970300 | 0.967094 | 0.960808 |
| best val loss | 0.075801 | 0.085073 | 0.080125 |
| full stacker WAPE | - | 0.074885 | 0.074783 |

说明：

- 本次 single GNN overall WAPE 与 Qwen 20260606 接近，但比旧 v3 单模型差。
- 本次 best val loss 优于 Qwen 20260606，但 test R2 更低，说明新增 LLM split 的测试分布仍更难。
- Full-feature residual stacker 的整体 WAPE 与 Qwen 20260606 几乎相同，但收益主要仍来自 non-LLM。
- 本次是全量重建 split，不是把 LLM 样本追加到旧 split；与旧记录只能做趋势比较，不能视为严格 A/B。

## Epoch Loss Trace 选段

| epoch | train loss | val loss |
| ---: | ---: | ---: |
| 1 | 0.283558 | 0.249475 |
| 2 | 0.210405 | 0.211931 |
| 3 | 0.186545 | 0.171395 |
| 4 | 0.181530 | 0.176940 |
| 5 | 0.158843 | 0.190381 |
| 6 | 0.145084 | 0.129771 |
| 7 | 0.137222 | 0.121104 |
| 8 | 0.127303 | 0.124706 |
| 9 | 0.131651 | 0.120530 |
| 10 | 0.114422 | 0.130307 |
| 20 | 0.100936 | 0.100944 |
| 30 | 0.099930 | 0.096134 |
| 40 | 0.093743 | 0.090850 |
| 50 | 0.091472 | 0.127388 |
| 56 | 0.085011 | 0.083527 |
| 60 | 0.089236 | 0.091850 |
| 70 | 0.089797 | 0.098473 |
| 72 | 0.082605 | 0.081774 |
| 79 | 0.080308 | 0.080125 |
| 80 | 0.085910 | 0.081088 |
| 90 | 0.080897 | 0.086212 |
| 100 | 0.075503 | 0.090231 |

完整 epoch 日志见 `output/gnn_llm_full_retrain_20260607/gnn_model_scaled_llm_20260607/logs/run_1780812969.log`。

## 运行环境观察

- 数据抽取阶段主要使用 CPU；最大耗时来自 `yolov11_monitor.csv`，单 CSV 耗时约 `905` 秒。
- GNN 训练使用 `cuda:1`，避免与另一个助手可能使用的 GPU 冲突。
- 训练期间 `cuda:1` 显存占用约 `2.1-2.2GB`。
- 运行期间出现 `pynvml` deprecation warning，不影响训练。
- 训练期间出现 `torch-scatter` 未安装的 PyG 性能 warning，不影响数值正确性。
- Stacker 评估使用临时 here-doc 脚本，未新增源码或测试文件。
