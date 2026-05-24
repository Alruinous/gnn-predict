# GNN 预测器 refreshed v2 7 目标训练评估报告

## 执行摘要

本次基于你重新执行 extract 和 scale 后刷新的 `data/extracted_v2`、`data/scaled_v2` 和 `data/scalers_v2` 运行全量 GNN 预测器训练与评估。训练配置为 `config/gnn_model/scaled_training_v2_20260522.yaml`，实验名为 `gnn_model_scaled_v2_20260522`，训练设备为 `cuda:0`。

核心结果：

- 当前 refreshed v2 数据集 split 为 `11,852 / 3,950 / 3,950`，总样本 `19,752`。
- 本次是 7 目标训练，新增并启用了 `gpu_sm_active_percent_p95`；训练前校验确认 `data/scaled_v2/*.pt` 的 `y.shape == (1, 7)`，`data/scalers_v2/target_scalers.pkl` 包含全部 7 个目标。
- schema 为 `3.0.0`，feature source 为 `onnx_tool_profile_p0_features`；首样本 feature shape 为 node `(137, 22)`、edge `(146, 15)`、graph `(1, 30)`。
- best checkpoint 出现在 epoch `87`，best val loss 为 `0.072692`。
- test original-scale WAPE 为 `0.091725`，MAE 为 `44.068172`，RMSE 为 `266.308048`，R2 为 `0.851137`。
- 目标 WAPE 中，`gpu_mem_used_mb_p95` 为 `0.091083`，`gpu_util_percent_p95` 为 `0.130569`，`gpu_sm_active_percent_p95` 为 `0.121448`，`gpu_sm_occupancy_percent_p95` 为 `0.168727`。
- training phase 仍明显比 inference phase 更难：test WAPE 分别为 `0.111097` 和 `0.071943`。

注意：当前仍是行级随机 split，存在 `variant_name` 跨 split 重叠；本报告的 test 指标适合作为当前数据口径下的训练评估，不应直接解释为严格的 unseen-variant 泛化能力。

## 运行产物

| 项目 | 路径 |
| --- | --- |
| 训练配置 | `config/gnn_model/scaled_training_v2_20260522.yaml` |
| 数据目录 | `data/scaled_v2` |
| scaler 目录 | `data/scalers_v2` |
| result JSON | `output/gnn_model_newdata_training_eval_v2_20260522/gnn_model_scaled_v2_20260522/results/gnn_model_scaled_v2_20260522_1779448914_result.json` |
| 扩展指标 | `output/gnn_model_newdata_training_eval_v2_20260522/gnn_model_scaled_v2_20260522/results/extended_metrics.json` |
| report sidecar | `output/gnn_model_newdata_training_eval_v2_20260522/gnn_model_scaled_v2_20260522/results/report_sidecar.json` |
| checkpoint | `output/gnn_model_newdata_training_eval_v2_20260522/gnn_model_scaled_v2_20260522/checkpoints/best_model.pt` |
| 训练日志 | `output/gnn_model_newdata_training_eval_v2_20260522/gnn_model_scaled_v2_20260522/logs/run_1779448914.log` |
| 报告 | `docs/train/gnn_model_newdata_training_eval_v2_2026-05-22.md` |

## 运行命令

```bash
PYTHONPATH=src ./.venv/bin/python -m gnn_model \
  --config config/gnn_model/scaled_training_v2_20260522.yaml \
  --output_dir output/gnn_model_newdata_training_eval_v2_20260522 \
  --device cuda:0
```

训练参数：

| 参数 | 值 |
| --- | ---: |
| batch size | 32 |
| epoch | 100 |
| learning rate | 0.001000 |
| weight decay | 0.000100 |
| hidden dim | 128 |
| num layers | 2 |
| num heads | 8 |
| dropout | 0.1 |
| parameter count | 873,471 |

## 数据集与目标分布

### split 与目标

| split | 样本数 | target shape |
| --- | ---: | --- |
| train | 11,852 | `(1, 7)` |
| val | 3,950 | `(1, 7)` |
| test | 3,950 | `(1, 7)` |

目标字段：

- `duration_sec_avg`
- `cpu_cores_p95`
- `memory_delta_gb_p95`
- `gpu_util_percent_p95`
- `gpu_sm_active_percent_p95`
- `gpu_sm_occupancy_percent_p95`
- `gpu_mem_used_mb_p95`

### source CSV 分布

| source CSV | 样本数 |
| --- | ---: |
| `yolov11_monitor.csv` | 7,030 |
| `yolov5_monitor.csv` | 1,791 |
| `vgg11_monitor.csv` | 1,115 |
| `densenet_monitor.csv` | 1,079 |
| `vgg19_monitor.csv` | 993 |
| `vgg16_monitor.csv` | 974 |
| `resnet_monitor.csv` | 941 |
| `beit_monitor.csv` | 915 |
| `bert_monitor.csv` | 653 |
| `mobilenet_monitor.csv` | 647 |
| `efficientnet_monitor.csv` | 576 |
| `gpt2_monitor.csv` | 480 |

phase 分布：

| phase | 样本数 |
| --- | ---: |
| `inference` | 9,964 |
| `training` | 9,788 |

variant overlap：

| overlap | variant 数 |
| --- | ---: |
| `train_val` | 2,340 |
| `train_test` | 2,362 |
| `val_test` | 775 |

### 原始目标分布

| target | min | mean | median | p90 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `duration_sec_avg` | 0.001275 | 0.078986 | 0.028021 | 0.204746 | 0.259922 | 0.915653 |
| `cpu_cores_p95` | 0.997000 | 2.395967 | 1.000000 | 1.001000 | 18.921051 | 38.234001 |
| `memory_delta_gb_p95` | 0.780000 | 1.889757 | 1.461000 | 2.861000 | 3.990000 | 18.096001 |
| `gpu_util_percent_p95` | 1.000000 | 58.012627 | 56.000000 | 98.000000 | 99.000000 | 100.000000 |
| `gpu_sm_active_percent_p95` | 0.005000 | 0.486165 | 0.482000 | 0.855000 | 0.887000 | 0.976000 |
| `gpu_sm_occupancy_percent_p95` | 0.002000 | 0.190200 | 0.170000 | 0.404000 | 0.449000 | 0.688000 |
| `gpu_mem_used_mb_p95` | 449.000000 | 3,282.626465 | 2,781.000000 | 5,597.000000 | 6,473.000000 | 14,049.000000 |

### scaler 参数

| target | center | scale |
| --- | ---: | ---: |
| `duration_sec_avg` | 0.028021 | 0.123773 |
| `cpu_cores_p95` | 1.000000 | 1.000000 |
| `memory_delta_gb_p95` | 1.461000 | 0.452000 |
| `gpu_util_percent_p95` | 56.000000 | 53.000000 |
| `gpu_sm_active_percent_p95` | 0.482000 | 0.550000 |
| `gpu_sm_occupancy_percent_p95` | 0.170000 | 0.196000 |
| `gpu_mem_used_mb_p95` | 2,781.000000 | 2,594.000000 |

## 训练过程

| 项目 | 值 |
| --- | ---: |
| epoch | 100 |
| best epoch | 87 |
| best val loss | 0.072692 |
| last train loss | 0.085606 |
| batch size | 32 |
| learning rate | 0.001000 |
| weight decay | 0.000100 |
| device | `cuda:0` |

| 阶段 | 起止时间 UTC | 耗时 |
| --- | --- | ---: |
| data | 11:21:54.159711 - 11:22:03.277139 | 9.117428s |
| model_build | 11:22:03.277257 - 11:22:03.655680 | 0.378422s |
| training | 11:22:03.655746 - 11:57:50.056992 | 2,146.401246s |
| evaluation | 11:57:50.057091 - 11:57:53.137996 | 3.080905s |
| full | 11:21:54.159710 - 11:57:53.138060 | 2,158.978349s |

关键 epoch：

| epoch | train loss | val loss |
| ---: | ---: | ---: |
| 1 | 0.254088 | 0.208263 |
| 10 | 0.136459 | 0.116897 |
| 25 | 0.091265 | 0.097431 |
| 50 | 0.076631 | 0.095875 |
| 75 | 0.075017 | 0.075810 |
| 80 | 0.069601 | 0.072718 |
| 87 | 0.070446 | 0.072692 |
| 100 | 0.085606 | 0.077809 |

训练曲线中段有波动，epoch `44` val loss 曾升至 `0.136192`，epoch `67` val loss 曾升至 `0.120624`；但后段仍继续刷新 best，最终 best checkpoint 使用 epoch `87`。

## Test 指标

### split 稳定性

以下指标均为 original-scale 扩展评估结果。

| split | count | MAE | MSE | RMSE | WAPE | R2 | mean target WAPE | P95 abs error | max abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 11,852 | 38.447689 | 44,515.343750 | 210.986596 | 0.080920 | 0.904967 | 0.093692 | 193.227264 | 8,443.535156 |
| val | 3,950 | 41.874825 | 54,540.945312 | 233.540029 | 0.086530 | 0.885741 | 0.100234 | 199.017059 | 7,700.065430 |
| test | 3,950 | 44.068172 | 70,919.976562 | 266.308048 | 0.091725 | 0.851137 | 0.099957 | 204.681351 | 10,677.783203 |

默认 runner 输出的 test aggregate：

| criterion | MAE | MSE | RMSE | WAPE | max abs error |
| --- | ---: | ---: | ---: | ---: | ---: |
| normalized | 0.151933 | 0.671663 | 0.819551 | 0.202945 | 36.780331 |
| original | 44.068169 | 70,919.976562 | 266.308044 | 0.091725 | 10,677.783203 |

### test 分目标 original-scale 指标

| target | MAE | MSE | RMSE | WAPE | R2 | sMAPE | P95 abs error | max abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `duration_sec_avg` | 0.004643 | 0.000189 | 0.013763 | 0.058436 | 0.975517 | 0.098239 | 0.015712 | 0.316215 |
| `cpu_cores_p95` | 0.043534 | 0.044835 | 0.211743 | 0.018493 | 0.998526 | 0.006364 | 0.151390 | 4.083372 |
| `memory_delta_gb_p95` | 0.206340 | 0.903275 | 0.950408 | 0.110947 | 0.689186 | 0.080440 | 0.637248 | 16.624710 |
| `gpu_util_percent_p95` | 7.496808 | 116.154541 | 10.777502 | 0.130569 | 0.851545 | 0.166936 | 23.852646 | 65.739693 |
| `gpu_sm_active_percent_p95` | 0.058538 | 0.008460 | 0.091976 | 0.121448 | 0.892736 | 0.200826 | 0.191661 | 1.003101 |
| `gpu_sm_occupancy_percent_p95` | 0.031792 | 0.003078 | 0.055484 | 0.168727 | 0.839768 | 0.282714 | 0.111610 | 0.526869 |
| `gpu_mem_used_mb_p95` | 300.635529 | 496,322.750000 | 704.501774 | 0.091083 | 0.851136 | 0.091492 | 1,086.271729 | 10,677.783203 |

### test 分 phase 指标

| phase | count | MAE | MSE | RMSE | WAPE | R2 | duration MAE | memory delta MAE | SM active MAE | gpu mem MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `inference` | 1,975 | 34.202599 | 49,924.285156 | 223.437430 | 0.071943 | 0.889083 | 0.001760 | 0.316316 | 0.040397 | 233.088867 |
| `training` | 1,975 | 53.933739 | 91,915.664062 | 303.175962 | 0.111097 | 0.817027 | 0.007525 | 0.096364 | 0.076679 | 368.182220 |

training phase 的整体 WAPE 和 RMSE 均高于 inference phase；显存和 SM active 误差也更高。`memory_delta_gb_p95` 例外，inference phase 的 MAE 更高。

## 分 family 评估

按 test overall WAPE 排序前 15：

| family | test count | overall WAPE | mean target WAPE | R2 | MSE | duration MAE | memory delta MAE | SM active MAE | gpu mem MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpt2` | 97 | 0.237637 | 0.155813 | 0.424551 | 233,758.515625 | 0.013433 | 0.070790 | 0.087032 | 503.629242 |
| `convnext` | 94 | 0.217256 | 0.171447 | 0.355702 | 667,127.062500 | 0.038745 | 0.785283 | 0.087639 | 1,230.851440 |
| `resnet` | 160 | 0.150752 | 0.153765 | -4.223925 | 5,509.430176 | 0.001356 | 1.991256 | 0.096134 | 163.080475 |
| `vit` | 56 | 0.133892 | 0.102443 | 0.720446 | 58,587.304688 | 0.007069 | 0.161380 | 0.056967 | 365.599945 |
| `vgg19` | 204 | 0.119123 | 0.084101 | 0.041986 | 206,082.281250 | 0.003375 | 0.125112 | 0.054772 | 684.646240 |
| `vgg11` | 239 | 0.115317 | 0.075240 | 0.342454 | 242,632.359375 | 0.002402 | 0.087340 | 0.036817 | 546.956482 |
| `yolov5` | 371 | 0.109553 | 0.118896 | 0.889752 | 29,180.826172 | 0.002367 | 0.060475 | 0.078855 | 234.643570 |
| `bert` | 129 | 0.106564 | 0.091045 | 0.684305 | 17,578.527344 | 0.001989 | 0.025017 | 0.075663 | 233.510025 |
| `bert_large` | 66 | 0.102554 | 0.085016 | 0.781194 | 94,649.570312 | 0.004223 | 0.174258 | 0.045078 | 567.594666 |
| `beit` | 154 | 0.097504 | 0.071741 | 0.914585 | 27,004.492188 | 0.003282 | 0.044375 | 0.054416 | 270.310059 |
| `mobilenet` | 142 | 0.089639 | 0.171540 | 0.090277 | 4,482.444336 | 0.001551 | 0.481975 | 0.052453 | 148.679214 |
| `resnet101` | 74 | 0.081878 | 0.122282 | 0.398495 | 20,973.718750 | 0.002163 | 0.078315 | 0.119697 | 258.902496 |
| `vgg16` | 188 | 0.076567 | 0.081382 | 0.467789 | 109,035.085938 | 0.001846 | 0.593299 | 0.033709 | 454.013550 |
| `resnet152` | 76 | 0.072958 | 0.090911 | 0.589998 | 23,900.617188 | 0.004704 | 0.177072 | 0.080808 | 310.740753 |
| `yolov9` | 102 | 0.069456 | 0.094290 | 0.942070 | 15,957.199219 | 0.002197 | 0.050771 | 0.079367 | 212.653824 |

WAPE 最低的 5 个 family：

| family | test count | overall WAPE |
| --- | ---: | ---: |
| `yolov9` | 102 | 0.069456 |
| `yolov11` | 1,407 | 0.065010 |
| `efficientnet` | 112 | 0.055190 |
| `swin` | 37 | 0.050134 |
| `densenet` | 242 | 0.047098 |

## 结论

本次 refreshed v2 7 目标训练正常完成，模型在 test split 上的 original-scale WAPE 为 `0.091725`，明显低于 2026-05-21 旧 6 目标 delta v2/v3 报告中的约 `0.112` 到 `0.114`。但两者数据集口径不同：refreshed v2 样本数为 `19,752`，去掉了此前旧数据中的 `yolov8`、`dcn`、`dcnv2`、`edcn`、`deepfm` 等样本，且新增了 `gpu_sm_active_percent_p95` 目标。因此这次改善不能单独归因于模型或目标数变化。

从误差结构看，`gpu_sm_occupancy_percent_p95`、`gpu_util_percent_p95`、`gpu_sm_active_percent_p95` 和 `memory_delta_gb_p95` 仍是相对更难的目标；family 维度上 `gpt2`、`convnext`、`resnet` 的 test WAPE 较高。后续如果要评估泛化能力，应增加按 `variant_name` 或 model family 分组的 split，避免当前行级随机 split 的 variant 重叠影响判断。

## 追加实验：预测器容量调参

### 实验设计

本次追加只探索预测器容量，不修改模型代码和训练代码。多卡使用方式是并行运行多个独立单卡训练进程，每个进程指定一个 `--device cuda:N`、一个临时 YAML 配置和一个独立 output 目录；没有使用 DDP，也没有改 runner。

探索策略：

- baseline：沿用本报告主实验 `hidden_dim=128, num_layers=2, num_heads=8, dropout=0.1`。
- stage-1：用临时脚本 `/tmp/gnn_v2_capacity_stage1.py` 跑 30 epoch 容量筛选。
- final：只把 stage-1 中最有希望的 `hidden_dim=160, num_layers=2` 跑满 100 epoch。
- `hidden_dim=192` 两组短跑趋势明显落后，为节省 GPU 时间提前中止，仅作为排除证据。

临时脚本和配置：

| 项目 | 路径 |
| --- | --- |
| stage-1 临时脚本 | `/tmp/gnn_v2_capacity_stage1.py` |
| stage-1 临时配置目录 | `/tmp/gnn_capacity_stage1_configs` |
| stage-1 输出目录 | `output/gnn_model_v2_capacity_stage1_e30_20260522` |
| final 临时配置 | `/tmp/gnn_v2_capacity_h160_l2_e100.yaml` |
| final 输出目录 | `output/gnn_model_v2_capacity_final_20260522` |
| sweep 汇总 | `output/gnn_model_v2_capacity_final_20260522/capacity_sweep_summary.json` |

### 候选配置

| 配置 | hidden dim | layers | heads | dropout | epoch | 参数量 | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 128 | 2 | 8 | 0.1 | 100 | 873,471 | 完成 |
| h160_l2 | 160 | 2 | 8 | 0.1 | 30 | 1,358,047 | stage-1 完成 |
| h160_l3 | 160 | 3 | 8 | 0.1 | 30 | 1,668,447 | stage-1 完成 |
| h192_l2 | 192 | 2 | 8 | 0.1 | 15 observed | 1,949,119 | 提前中止 |
| h192_l3 | 192 | 3 | 8 | 0.1 | 8 observed | 2,395,327 | 提前中止 |
| h160_l2_e100 | 160 | 2 | 8 | 0.1 | 100 | 1,358,047 | 完成 |

### stage-1 结果

stage-1 只用于筛选，不作为最终模型选择依据。

| 配置 | observed epoch | best epoch | best val loss | test WAPE | test MAE | test RMSE | gpu mem WAPE | SM active WAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| h160_l2 | 30 | 26 | 0.094616 | 0.108242 | 52.003662 | 274.209503 | 0.108017 | 0.130106 |
| h160_l3 | 30 | 24 | 0.100849 | 0.107943 | 51.859848 | 282.543518 | 0.107617 | 0.123734 |
| h192_l2 | 15 | 14 | 0.104919 | - | - | - | - | - |
| h192_l3 | 8 | 8 | 0.172053 | - | - | - | - | - |

stage-1 观察：

- `h160_l2` 是短跑里 validation 最好的增容配置，但 30 epoch test WAPE 为 `0.108242`，明显差于 baseline 100 epoch 的 `0.091725`。
- `h160_l3` 比 `h160_l2` 更深，但 best val loss 和 RMSE 都更差，没有体现加深收益。
- `h192_l2`、`h192_l3` 参数量更大，早期收敛更慢；到中止点仍明显落后 `h160_l2`，因此没有继续跑满。

### final h160_l2 100 epoch 结果

| 配置 | 参数量 | best epoch | best val loss | test WAPE | test MAE | test RMSE | test R2 | gpu mem WAPE | SM active WAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline h128_l2 | 873,471 | 87 | 0.072692 | 0.091725 | 44.068172 | 266.308048 | 0.851137 | 0.091083 | 0.121448 |
| h160_l2_e100 | 1,358,047 | 71 | 0.078274 | 0.095848 | 46.048992 | 269.321340 | 0.847750 | 0.095518 | 0.115500 |

`h160_l2_e100` 的主要产物：

| 项目 | 路径 |
| --- | --- |
| result JSON | `output/gnn_model_v2_capacity_final_20260522/gnn_model_v2_capacity_h160_l2_heads8_d0p10_e100/results/gnn_model_v2_capacity_h160_l2_heads8_d0p10_e100_1779456808_result.json` |
| 扩展指标 | `output/gnn_model_v2_capacity_final_20260522/gnn_model_v2_capacity_h160_l2_heads8_d0p10_e100/results/extended_metrics.json` |
| checkpoint | `output/gnn_model_v2_capacity_final_20260522/gnn_model_v2_capacity_h160_l2_heads8_d0p10_e100/checkpoints/best_model.pt` |

### 容量调参结论

按 test original-scale overall WAPE、best val loss、MAE、RMSE 综合判断，本轮最优配置仍是主实验 baseline：

```yaml
hidden_dim: 128
num_layers: 2
num_heads: 8
dropout_rate: 0.1
batch_size: 32
learning_rate: 0.001
weight_decay: 0.0001
num_epochs: 100
```

增大到 `hidden_dim=160` 后，参数量从 `873,471` 增至 `1,358,047`，但 overall test WAPE 从 `0.091725` 退化到 `0.095848`，test RMSE 从 `266.308048` 退化到 `269.321340`。唯一明确改善的是 `gpu_sm_active_percent_p95` WAPE，从 `0.121448` 降到 `0.115500`，但不足以抵消整体指标退化。

本轮没有证据支持单纯提高预测器参数量。下一步若继续调参，优先方向不应是更宽/更深，而应考虑：

- 对大模型降低 learning rate 或加 warmup，解决早期收敛慢和 validation spike。
- 对 `gpu_sm_active_percent_p95` 单独做 loss weight 或多任务权重实验。
- 增加正则或 early stopping，对 `h160_l2` 后段 validation 波动做控制。
