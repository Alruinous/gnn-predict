# GNN P0 特征工程消融实验记录

## 结论

本次没有撤销 P0 特征工程代码。基于 `data/scaled_v2` 的受控遮蔽消融，结果不支持
“P0 整体让预测器变差”的判断，但 P0 内部收益不均衡：

- `shape` 特征是最稳定有用的部分，尤其改善 `memory_delta_gb_p95`。
- `op_type` one-hot / histogram 没有显示稳定收益，`no_op` 在本次 probe 中整体 WAPE
  最低。
- `recommender` 专项特征对整体 WAPE 没有稳定收益；在推荐模型 family 上也不是一致改善。
- 因此更合理的后续动作是保留 shape，继续把 op/recommender 作为可开关消融项，而不是
  直接回滚全部 P0。

## 未暂存改动归类

P0 特征工程本体：

- `src/gnn_model/data/constants.py`
  - 新增 `OP_TYPE_FEATURE_NAMES`、`NODE_SHAPE_FEATURE_NAMES`、
    `EDGE_SHAPE_FEATURE_NAMES`、`GRAPH_SHAPE_FEATURE_NAMES`、
    `RECOMMENDER_GRAPH_FEATURE_NAMES`。
  - 维度变为 `node=37`、`edge=15`、`graph=78`。
- `src/gnn_model/data/onnx_graph.py`
  - 新增 op category one-hot、graph op histogram。
  - 新增 node/edge/graph tensor shape 摘要。
  - 新增 recommender graph feature vector。
- `src/gnn_model/data/variant_context.py`
  - 从 result JSON 读取 `variant_config` 和 `metadata.model_kind`，供 recommender
    特征使用。
- `src/gnn_model/data/extract.py`
  - 从 result JSON 解析 variant context。
  - manifest 升级为 schema `3.0.0`，写入 feature names。
  - 其中 GPU metric 质量过滤和目标字段调整是相邻的数据清洗/目标改动，不是纯 P0 特征。
- `src/gnn_model/data/prepared_dataset.py`
  - prepared manifest 校验 feature names，防止旧 schema 混用。
- 相关测试：
  - `tests/test_gnn_model_data.py`
  - `tests/test_gnn_model_cli.py`
  - `tests/test_gnn_model_pipeline.py`
  - `tests/gnn_model_test_utils.py`

本次新增的消融工具：

- `scripts/gnn_p0_feature_ablation.py`
- `tests/test_gnn_p0_feature_ablation.py`

非 P0 或仅间接相关：

- `.gitignore`
- `config/monitor/monitor.yaml`
- `config/monitor/monitor_v2.yaml`
- `config/gnn_model/config.yaml`
- `config/gnn_model/scaled_training.yaml`
- `config/gnn_model/scaled_training_6targets_delta.yaml`
- 删除的旧训练配置
- `docs/train/gnn_model_newdata_training_eval*_2026-05-21.md`

其中 `config/monitor/monitor.yaml` 当前有 `query_step_seconds: 3a`，这看起来像手误，和
P0 特征工程无关。

## 数据与实验设置

使用当前 v2 数据：

| split | count | shape |
| --- | ---: | --- |
| train | 13,286 | `x=(?,37)` |
| val | 4,429 | `edge_attr=(?,15)` |
| test | 4,429 | `graph_features=(1,78)` |

manifest:

- `schema_version`: `3.0.0`
- `feature_source`: `onnx_tool_profile_p0_features`
- target: 6 个 delta 目标

probe 训练设置：

- 数据切片：`train=4096`、`val=1024`、`test=2048`
- 训练：20 epoch，batch size 32，`hidden_dim=128`，`num_layers=2`，`num_heads=8`
- 设备：`cuda:0`，Tesla V100
- seeds：`7,11,19`
- 消融方式：不改磁盘数据，在内存中把对应 P0 feature 列置零。

实验矩阵：

| experiment | 保留内容 | 遮蔽内容 |
| --- | --- | --- |
| `p0_all` | 全部 P0 | 无 |
| `p0_off` | 非 P0 基础特征 | op + shape + recommender |
| `op_only` | op | shape + recommender |
| `shape_only` | shape | op + recommender |
| `recommender_only` | recommender | op + shape |
| `no_op` | shape + recommender | op |
| `no_shape` | op + recommender | shape |
| `no_recommender` | op + shape | recommender |

输出目录：

`output/gnn_p0_ablation_v2_probe_20260522`

## 整体指标

按 `original_scale_wape` 从低到高排序：

| experiment | n | WAPE mean | WAPE stdev | best val loss | duration WAPE | memory delta WAPE | gpu mem WAPE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_op` | 3 | 0.153346 | 0.007706 | 0.134140 | 0.249747 | 0.175412 | 0.153362 |
| `recommender_only` | 3 | 0.153611 | 0.008049 | 0.136888 | 0.255920 | 0.213620 | 0.153491 |
| `no_recommender` | 3 | 0.153700 | 0.002078 | 0.146430 | 0.239145 | 0.173333 | 0.153724 |
| `shape_only` | 3 | 0.155340 | 0.004034 | 0.130269 | 0.245715 | 0.169539 | 0.155355 |
| `p0_all` | 3 | 0.156471 | 0.011451 | 0.133980 | 0.244360 | 0.172357 | 0.156432 |
| `no_shape` | 3 | 0.157120 | 0.008238 | 0.154710 | 0.236996 | 0.206891 | 0.157200 |
| `p0_off` | 3 | 0.158371 | 0.005503 | 0.163362 | 0.268330 | 0.214781 | 0.158356 |
| `op_only` | 3 | 0.160364 | 0.013514 | 0.159184 | 0.250987 | 0.209723 | 0.160387 |

关键对比：

- `shape_only` vs `p0_off`: memory delta WAPE 从 `0.214781` 降到 `0.169539`。
- `no_shape` vs `p0_all`: memory delta WAPE 从 `0.172357` 升到 `0.206891`。
- `no_op` vs `p0_all`: overall WAPE 从 `0.156471` 降到 `0.153346`，说明 op 特征未显示收益。
- `no_recommender` vs `p0_all`: overall WAPE 从 `0.156471` 降到 `0.153700`，说明 recommender
  专项在本 probe 中不是整体收益来源。

## Family 切片

test 子集 family 样本数前列：

| family | count |
| --- | ---: |
| `yolov11` | 638 |
| `yolov5` | 167 |
| `vgg11` | 107 |
| `densenet` | 102 |
| `resnet` | 88 |
| `dcnv2` | 83 |
| `edcn` | 46 |
| `deepfm` | 43 |
| `dcn` | 41 |
| `gpt2` | 41 |

推荐模型相关 family WAPE：

| family | p0_all | p0_off | shape_only | no_shape | no_op | no_recommender | recommender_only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `deepfm` | 0.2266 | 0.2519 | 0.2444 | 0.2725 | 0.2515 | 0.2434 | 0.2530 |
| `dcn` | 0.1632 | 0.1822 | 0.1944 | 0.2019 | 0.1571 | 0.1840 | 0.1877 |
| `dcnv2` | 0.1279 | 0.1259 | 0.1332 | 0.1082 | 0.1105 | 0.1362 | 0.0789 |
| `edcn` | 0.3006 | 0.3215 | 0.3259 | 0.2940 | 0.2945 | 0.3165 | 0.3082 |

这里没有看到 recommender 特征对所有推荐 family 的一致收益。`dcnv2` 在
`recommender_only` 下很好，但 `deepfm/dcn/edcn` 没有同步改善，所以不能把 recommender
专项视为已验证有效。

## 与 2026-05-21 全 P0 报告的关系

`docs/train/gnn_model_newdata_training_eval_v2_2026-05-21.md` 与
`docs/train/gnn_model_newdata_training_eval_v3_2026-05-21.md` 是全 P0、100 epoch、
全量数据训练：

| report | test original WAPE | R2 |
| --- | ---: | ---: |
| v2 | 0.112127 | 0.936799 |
| v3 | 0.114206 | 0.937057 |

这些结果说明全 P0 模型可以训练到较好指标，但没有无 P0 对照。因此本次 probe 只用于判断
P0 内部 feature block 的相对贡献，不能直接替代全量 100 epoch 结论。

## 决策

暂不撤销全部 P0。建议后续按下面顺序处理：

1. 保留 shape 特征，作为 P0 的核心收益项。
2. 把 op 特征改成可开关或从默认 schema 中拆出，除非更长训练/full data 证明其有效。
3. recommender 专项继续保留为实验项，但不要宣称它已稳定改善推荐模型 family。
4. 下一轮做 `shape_only`、`no_op`、`p0_all` 的更长训练对照，优先用 full test 评估。
5. 在 group split 和 train-only scaler 修复前，不把当前行级 split 指标当作泛化证明。
