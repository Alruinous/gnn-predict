# GNN 预测器 Qwen3/Gemma4 LLM-only 全量重训记录

## 结论

- 本次只使用新采集的 `csv/v100_llm` 和 `csv/a100_llm`，不混入旧 `csv/v100`、`csv/a100`。
- 产物使用当前 workflow 兼容的 9-target schema，与 `gnn_full_retrain_20260612` 的 target 顺序一致。
- 抽取后得到 `4,018` 条 LLM prefill/decode 样本，split 为 `2,410/804/804`。
- Single GNN 完成 100 epoch，best epoch 为 `33`，best val loss 为 `0.151482`。
- Test original-scale overall WAPE 为 `0.065814`，R2 为 `0.977144`。
- Test `gpu_mem_used_mb_max` WAPE 为 `0.062923`，Qwen3 decode 子集为 `0.085402`，低于 workflow 文档中的 `0.15` 重训门槛。
- Gemma4 test 显存预测明显优于 Qwen3：`gpu_mem_used_mb_max` WAPE 为 `0.037030` vs `0.086209`。
- Prefill 的 `run_duration_sec_avg` WAPE 很高，为 `5.557874`；该 checkpoint 不应把 prefill duration 作为硬 ETA 门控。
- 显存最大低估仍达到 `24,337.75 MB`，用于 OOM 风险控制时必须叠加 safety margin。

## 产物

| 项目 | 路径 |
| --- | --- |
| extracted data | `data/extracted_llm_only_20260702` |
| scaled data | `data/scaled_llm_only_20260702` |
| scalers | `data/scalers_llm_only_20260702` |
| effective config | `output/gnn_llm_only_full_retrain_20260702/effective_config.yaml` |
| checkpoint manifest | `output/gnn_llm_only_full_retrain_20260702/checkpoint_manifest.json` |
| checkpoint | `output/gnn_llm_only_full_retrain_20260702/gnn_model_llm_only_20260702/checkpoints/best_model.pt` |
| result JSON | `output/gnn_llm_only_full_retrain_20260702/gnn_model_llm_only_20260702/results/gnn_model_llm_only_20260702_1783011398_result.json` |
| supplemental eval JSON | `output/gnn_llm_only_full_retrain_20260702/gnn_model_llm_only_20260702/results/supplemental_evaluation_20260702.json` |
| train log | `output/gnn_llm_only_full_retrain_20260702/gnn_model_llm_only_20260702/logs/run_1783011398.log` |

| 路径 | 大小 |
| --- | ---: |
| `data/extracted_llm_only_20260702` | 2.4G |
| `data/scaled_llm_only_20260702` | 2.4G |
| `data/scalers_llm_only_20260702` | 5.0K |
| checkpoint | 4.4M |
| result JSON | 10K |
| supplemental eval JSON | 980K |

## 数据口径

输入 CSV：

| CSV | raw rows | included | GPU filtered | high memory delta |
| --- | ---: | ---: | ---: | ---: |
| `csv/v100_llm/gemma4_monitor.csv` | 687 | 601 | 86 | 18 |
| `csv/a100_llm/gemma4_monitor.csv` | 684 | 662 | 22 | 21 |
| `csv/v100_llm/qwen3_monitor.csv` | 1,428 | 1,344 | 84 | 0 |
| `csv/a100_llm/qwen3_monitor.csv` | 1,422 | 1,411 | 11 | 0 |
| total | 4,221 | 4,018 | 203 | 39 |

Split 分布：

| split | total | Qwen3 | Gemma4 | prefill | decode | A100 | V100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 2,410 | 1,641 | 769 | 469 | 1,941 | 1,259 | 1,151 |
| val | 804 | 558 | 246 | 147 | 657 | 410 | 394 |
| test | 804 | 556 | 248 | 150 | 654 | 404 | 400 |

Schema：

| feature | dim |
| --- | ---: |
| node feature | 22 |
| edge feature | 15 |
| graph feature | 31 |
| target | 9 |

Target 顺序：

```text
deployment_duration_sec_avg
run_duration_sec_avg
cpu_cores_max
memory_delta_gb_max
gpu_util_percent_max
gpu_sm_active_percent_max
gpu_sm_occupancy_percent_max
gpu_mem_used_mb_max
gpu_power_watts_avg
```

## 训练口径

训练命令：

```bash
PYTHONPATH=src uv run python -m gnn_model \
  --config output/gnn_llm_only_full_retrain_20260702/effective_config.yaml \
  --output_dir output/gnn_llm_only_full_retrain_20260702 \
  --device cuda:0
```

训练配置：

| 项目 | 值 |
| --- | ---: |
| GPU | Tesla V100-SXM2-32GB |
| batch_size | 32 |
| num_epochs | 100 |
| learning_rate | 0.0008 |
| weight_decay | 0.0001 |
| loss_weights | `[1,1,1,1,1,1,2,1,1]` |
| hidden_dim | 128 |
| num_layers | 2 |
| num_heads | 8 |
| dropout_rate | 0.1 |
| readout_mode | `mean_sum_max` |
| structural_context_mode | `basic` |
| parameter_count | 1,112,322 |

训练结果：

| best epoch | best val loss | last train loss |
| ---: | ---: | ---: |
| 33 | 0.151482 | 0.084558 |

耗时：

| stage | start UTC | end UTC | duration sec |
| --- | --- | --- | ---: |
| data | `2026-07-02T16:56:38.206663+00:00` | `2026-07-02T16:56:43.276099+00:00` | 5.069436 |
| model_build | `2026-07-02T16:56:43.276203+00:00` | `2026-07-02T16:56:43.639219+00:00` | 0.363017 |
| training | `2026-07-02T16:56:43.639282+00:00` | `2026-07-02T17:13:08.154461+00:00` | 984.515179 |
| evaluation | `2026-07-02T17:13:08.154632+00:00` | `2026-07-02T17:13:09.691789+00:00` | 1.537157 |
| full | `2026-07-02T16:56:38.206663+00:00` | `2026-07-02T17:13:09.691859+00:00` | 991.485196 |

## Test 指标

Overall：

| scale | MAE | RMSE | R2 | WAPE | max abs error |
| --- | ---: | ---: | ---: | ---: | ---: |
| normalized | 0.308095 | 0.763108 | 0.623369 | 0.424494 | 8.524085 |
| original | 52.994896 | 412.870789 | 0.977144 | 0.065814 | 24,337.751953 |

Per-target original-scale：

| target | MAE | RMSE | R2 | WAPE | max abs error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `deployment_duration_sec_avg` | 4.066590 | 5.994773 | 0.986841 | 0.061935 | 38.883118 |
| `run_duration_sec_avg` | 0.679950 | 1.224210 | 0.986030 | 0.102025 | 10.417921 |
| `cpu_cores_max` | 0.000212 | 0.000318 | 0.574317 | 0.000212 | 0.001001 |
| `memory_delta_gb_max` | 0.588687 | 1.114582 | 0.387967 | 0.253507 | 7.697710 |
| `gpu_util_percent_max` | 6.242064 | 14.426762 | 0.498887 | 0.167176 | 91.112534 |
| `gpu_sm_active_percent_max` | 6.320857 | 14.166135 | 0.533942 | 0.343656 | 89.297371 |
| `gpu_sm_occupancy_percent_max` | 2.597434 | 6.189841 | 0.567827 | 0.379672 | 44.953899 |
| `gpu_mem_used_mb_max` | 442.390747 | 1,238.123779 | 0.934795 | 0.062923 | 24,337.750000 |
| `gpu_power_watts_avg` | 14.067531 | 26.908808 | 0.597279 | 0.180037 | 164.209152 |

By model：

| model | count | WAPE | gpu mem WAPE | run WAPE | mem under rate | mem under p95 | mem under max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma4 | 248 | 0.038092 | 0.037030 | 0.104738 | 0.455645 | 781.754456 | 5,233.007812 |
| Qwen3 | 556 | 0.090142 | 0.086209 | 0.099528 | 0.517986 | 1,334.103516 | 24,337.750000 |

By phase：

| phase | count | WAPE | gpu mem WAPE | run WAPE | mem under rate | mem under p95 | mem under max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decode | 654 | 0.062876 | 0.060471 | 0.094715 | 0.503058 | 1,187.243408 | 11,807.022461 |
| prefill | 150 | 0.079233 | 0.074219 | 5.557874 | 0.480000 | 863.257080 | 24,337.750000 |

By GPU：

| GPU | count | WAPE | gpu mem WAPE | run WAPE | mem under rate | mem under p95 | mem under max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A100 | 404 | 0.062469 | 0.060327 | 0.115139 | 0.537129 | 934.998292 | 24,337.750000 |
| V100 | 400 | 0.069522 | 0.065833 | 0.093750 | 0.460000 | 1,184.536622 | 10,888.109375 |

By model and phase:

| group | count | WAPE | gpu mem WAPE | run WAPE | mem under rate | mem under p95 | mem under max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma4 decode | 223 | 0.038070 | 0.037061 | 0.100787 | 0.457399 | 788.003967 | 5,233.007812 |
| Gemma4 prefill | 25 | 0.038271 | 0.036767 | 7.262478 | 0.440000 | 660.473694 | 962.001953 |
| Qwen3 decode | 431 | 0.088668 | 0.085402 | 0.089117 | 0.526682 | 1,477.119629 | 11,807.022461 |
| Qwen3 prefill | 125 | 0.094691 | 0.088705 | 5.138010 | 0.488000 | 897.504456 | 24,337.750000 |

By decode output bucket:

| bucket | count | WAPE | gpu mem WAPE | run WAPE |
| --- | ---: | ---: | ---: | ---: |
| prefill | 150 | 0.079233 | 0.074219 | 5.557874 |
| 1-64 | 325 | 0.074947 | 0.070595 | 0.232351 |
| 65-128 | 65 | 0.057961 | 0.057432 | 0.122591 |
| 129-256 | 119 | 0.041693 | 0.041058 | 0.112620 |
| 257-512 | 96 | 0.047769 | 0.047377 | 0.082073 |
| >512 | 49 | 0.073857 | 0.074353 | 0.049859 |

## Workflow 使用判断

- 可以把该 checkpoint 作为 Qwen3/Gemma4 prefill/decode 的 workflow 资源预测候选。
- Qwen3 decode `gpu_mem_used_mb_max` WAPE 为 `0.085402`，满足 `docs/workflow/experiment_20260629.md` 中低于 `0.15` 的重训判定门槛。
- 显存预测不能裸用作 OOM 硬门控。Test split 显存低估率为 `0.498756`，低估 p95 为 `1,131.786621 MB`，最大低估为 `24,337.75 MB`。
- Prefill duration 不适合直接做硬 ETA：prefill `run_duration_sec_avg` WAPE 为 `5.557874`，其中 Qwen3 prefill 为 `5.138010`、Gemma4 prefill 为 `7.262478`。
- 用于调度时建议优先使用 `gpu_mem_used_mb_max`、`deployment_duration_sec_avg` 和 decode duration；prefill duration 应保留历史统计或 safety margin。

## 验证

- 输入预检：4 个 CSV、4,221 raw rows、0 missing ONNX、文件内 `(variant_name, phase)` 无重复。
- 抽取验证：`data/extracted_llm_only_20260702` manifest 写出，split 为 `2,410/804/804`。
- Scaler 验证：`node_feature_scaler.pkl`、`edge_feature_scaler.pkl`、`graph_feature_scaler.pkl`、`target_scalers.pkl` 均写出。
- Scaled split 验证：node `[*,22]`、edge `[*,15]`、graph `[1,31]`、target `[1,9]`，全量有限值检查通过。
- Checkpoint 验证：supplemental evaluation 成功加载 `best_model.pt` 并完成 train/val/test 预测。
- 临时 `/tmp` 并行抽取脚本和评估脚本已删除。
