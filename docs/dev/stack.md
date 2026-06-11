# GNN 后处理 Stacker 复现交接文档

本文给后续助手交接 stacked 方案的历史证据、风险判断和复现实验路线。本文不是实现记录，也不表示当前分支已经复现该方案。

## 核心判断

- 历史 stacked 方案是 GNN 推理后的 residual tree 后处理，不是 GNN 主体结构改造。
- 决策树输入不是整张图本身，而是 GNN 预测值、图/节点/边摘要统计、op histogram 和类别 one-hot 拼成的 tabular feature。
- 历史 residual 模型是 `HistGradientBoostingRegressor(loss="absolute_error", max_leaf_nodes=31, random_state=7)`，7 个 target 分别拟合。
- 输出公式是 `final_prediction = clip(base_prediction + residual_prediction)`。
- 主线不建议复原旧 174 维数据集；它的问题不是维度高，而是特征口径混杂且有伪信息风险。
- 若后续继续 stacked，应优先在当前 30 维主线数据上重新建立同口径 GNN baseline，再评估 residual tree 是否仍有收益。
- 旧 174 维路线只适合作为 `legacy/name-context experiment` 消融，不应作为主线数据集或默认复现目标。

## 当前事实

本轮已核对的当前仓库状态：

- 仓库：`/home/wangjh/gnn_predict`
- 当前 `main` 和本地 `dev/stacked` 指向同一提交：`011dc6ee719abe3cea7c01034e18ed2e7e67222d`
- 当前 `data/scaled_v3` split 数量：`11949 / 3983 / 3983`
- 当前 `data/scaled_v3` graph feature shape：`[1, 30]`
- 当前代码常量：`GRAPH_FEATURE_DIM=30`、`NODE_FEATURE_DIM=22`、`EDGE_FEATURE_DIM=15`、`OP_TYPE_COUNT=15`

当前可见但不能直接作为主项目 v3 baseline 的 30 维 checkpoint：

- checkpoint：`output/gnn_main_baseline_20260601/gnn_model_scaled_main_20260601/checkpoints/best_model.pt`
- graph input dim：`30`
- 风险：result JSON 指向 `data/scaled_main_20260601`，不是本任务指定的 `data/scaled_v3`

当前可见但缺少 checkpoint 的历史结果：

- result JSON：`output/gnn_model_v3_20260531/gnn_model_scaled_v3_20260531/results/gnn_model_scaled_v3_20260531_1779771864_result.json`
- 风险：对应 checkpoint 不在该 output 目录中

用户补充的临时实验：

- result JSON：`/home/wangjh/gnn_tmp/output/gnn_model_v3_peak_live_no_name_20260601/gnn_model_scaled_v3_peak_live_no_name_20260601/results/gnn_model_scaled_v3_peak_live_no_name_20260601_1780294000_result.json`
- worktree commit：`0ee1a738068029c616f6845b83f297926fff5777`
- branch：`experiment/tmp-v3-peak-live-no-name-train-20260601`
- data：`/home/wangjh/gnn_predict/data/scaled_v3`
- scaler：`/home/wangjh/gnn_predict/data/scalers_v3`
- checkpoint：`/home/wangjh/gnn_tmp/output/gnn_model_v3_peak_live_no_name_20260601/gnn_model_scaled_v3_peak_live_no_name_20260601/checkpoints/best_model.pt`
- graph input dim：`174`
- best epoch：`85`
- best val loss：`0.07513626664876938`
- test original-scale WAPE：`0.09796569496393204`

关键风险：同名 `data/scaled_v3` 不足以证明 schema 相同。用户给出的临时 checkpoint 是 174 维，而当前主线数据检查为 30 维，说明数据或代码口径曾发生变化。

## 为什么不复原 174 维主线

旧 174 维数据集不适合继续作为主线。核心问题是特征口径混杂和伪信息风险。

`variant_context` 有命名泄漏风险：

- 它来自 `model_name` / `variant_name` 字符串解析。
- 它学习的是命名规则和家族/变体名对应的性能模式，不完全是架构图本身。
- 对重命名、未见命名规则、真实部署输入不稳定。
- 如果部署时没有同等命名约定，该特征会使离线指标高估泛化能力。

runtime profile 字段缺少真实样例支撑：

- 旧代码里 runtime profile 依赖 `result_json.profile_summary`。
- 本轮未找到能证明这些字段真实存在且稳定填充的样例。
- 按旧逻辑，读不到 profile 就全 0。
- 因此那组 runtime profile 维度很可能只是零填充占位，不是有效测量信号。

`profile` 术语曾被混用：

- ONNX 静态估算的 `macs`、`memory`、`params` 是可复现架构特征。
- runtime profiler 字段没有同等证据。
- 174 维容易让后续助手误以为数据包含真实 runtime profiling 信号。

padding 兼容会掩盖 schema 错误：

- 旧代码存在向后兼容补 0 行为。
- 缺失特征被静默补 0 后，训练流程可能继续运行。
- 这会让数据版本不匹配难以及时暴露。

重新 profile 会污染数据口径：

- 如果为了复原 174 维而在当前机器重新采集 profiler 信号，会引入当前硬件、驱动、容器和运行环境口径。
- 原始 benchmark 的采集环境和当前环境不一定一致。
- 这种补采不是复原，而是混入新的测量来源。

结论：

- 不要把 174 维恢复为主线。
- 174 维最多作为 legacy/name-context 消融。
- 若做消融，必须明确标注 `variant_context` 是命名上下文，不是纯架构图特征。
- 若做消融，必须明确标注 runtime profile 列缺失或全 0，不能宣称它提供真实 runtime profiling 信号。

## 历史 stacker 证据

关键提交：

```text
4d20c8b6546979039f7faf8bdba5b588b07bd7ff
Add posthoc best stacker implementation and corresponding tests
```

历史文件：

- `src/gnn_model/posthoc/best_stacker_20260525.py`
- `tests/test_best_stacker_20260525.py`
- `docs/train/gnn_model_v3_variant_context_gpu1_training_eval_2026-05-26.md`

查看方式：

```bash
git show 4d20c8b:src/gnn_model/posthoc/best_stacker_20260525.py
git show 4d20c8b:tests/test_best_stacker_20260525.py
git show 4d20c8b:docs/train/gnn_model_v3_variant_context_gpu1_training_eval_2026-05-26.md
```

历史归档文件 `docs/refactor/20260531.md` 也提到这些后处理路线已被删除：

- `residual stacker`
- `out-of-fold residual stacker`
- `device-aware specialized stacker`
- `repeated second-level out-of-fold ridge stack`
- `group switch / group blend`
- `target-specific checkpoint pool`

删除原因不是指标无效，而是它们不是单一 GNN predictor，并且引入部署复杂度或 test-coupling 风险。

## 历史数据流

历史 stacker 脚本流程：

- 读取 `train.pt`、`val.pt`、`test.pt`。
- 读取 `target_scalers.pkl`。
- 用 GNN checkpoint 对 validation/test 推理。
- 将 GNN 输出 inverse transform 到原始量纲。
- 在 validation split 计算 residual：`target_original - base_prediction`。
- 用 validation feature 拟合每个 target 的 residual regressor。
- 在 test split 预测 residual。
- 将 test residual 加回 base prediction。
- 对输出做 target-domain clipping。
- 输出 overall 和 per-target WAPE。

target：

- `deployment_duration_sec_avg`
- `run_duration_sec_avg`
- `cpu_cores_p95`
- `memory_delta_gb_p95`
- `gpu_util_percent_p95`
- `gpu_sm_active_percent_p95`
- `gpu_sm_occupancy_percent_p95`
- `gpu_mem_used_mb_p95`

bounded target：

- `gpu_util_percent_p95`：`0..100`
- `gpu_sm_active_percent_p95`：`0..100`
- `gpu_sm_occupancy_percent_p95`：`0..100`

positive target：

- `deployment_duration_sec_avg`
- `run_duration_sec_avg`
- `cpu_cores_p95`
- `memory_delta_gb_p95`
- `gpu_mem_used_mb_p95`

## 历史特征构造

历史 stacker 将图样本转成 tabular feature。

GNN prediction feature：

- checkpoint 的原始量纲预测值。
- checkpoint 预测值相对 base prediction 的差值。
- 单目标或 hard4 checkpoint 会通过 `expand_predictions` 填回 7 目标矩阵，缺失目标用 base prediction fallback。

graph feature：

- `graph_features` 展平。

node/edge summary：

- `min`
- `mean`
- `std`
- `p90`
- `max`

op histogram：

- `log1p(counts)`
- `ratios[:-1]`

category one-hot：

- `family`
- `base`
- `phase`
- `device`
- `batch`
- `family_phase`
- `base_phase`
- `family_batch`
- `base_batch`
- `base_device`
- `family_device`
- `base_phase_device`
- `batch_device`

历史 `CategoryMaps` 只用 train split 建类别表。validation/test 未见类别输出全零 one-hot。

当前 30 维 `data/scaled_v3/train.pt` 上已核对的类别维度：

| field | count |
| --- | ---: |
| `family` | 20 |
| `base` | 53 |
| `phase` | 2 |
| `device` | 4 |
| `batch` | 5 |
| `family_phase` | 40 |
| `base_phase` | 106 |
| `family_batch` | 44 |
| `base_batch` | 109 |
| `base_device` | 53 |
| `family_device` | 20 |
| `base_phase_device` | 106 |
| `batch_device` | 15 |

合计 category dim 为 `577`。如果只使用一个当前 7-target GNN checkpoint，并拼接 GNN 预测、30 维 graph features、node/edge 统计、op histogram 和类别特征，预计 stacker feature dim 为 `828`。

## 历史 checkpoint pool

`4d20c8b` 中记录的 checkpoint pool：

| name | 来源 | target |
| --- | --- | --- |
| `family` | 2026-05-24 all7 log/logit family checkpoint | all7 |
| `base_phase` | 2026-05-24 all7 log/logit base+phase checkpoint | all7 |
| `family_base_phase` | 2026-05-24 all7 log/logit family+base+phase checkpoint | all7 |
| `hard4_log` | 2026-05-24 hard4 log/logit family checkpoint | hard4 |
| `hard4_smooth` | 2026-05-24 hard4 smooth family checkpoint | hard4 |
| `sm_occ_single` | 2026-05-24 SM occupancy single-target family checkpoint | SM occupancy |
| `all7_device` | 2026-05-25 all7 base+phase+device checkpoint | all7 |
| `sm_device` | 2026-05-25 SM occupancy base+phase+device checkpoint | SM occupancy |
| `current_v3vc` | 2026-05-26 v3 variant-context rerun checkpoint | all7 |

历史默认 checkpoint root：

```text
/home/wangjh/gnn_predict/output/gnn_wape_opt_current_v2_20260524
```

当前仓库未确认这些历史 checkpoint 是否仍存在。后续若研究 historical pool，应先核对 checkpoint 文件、graph feature dim、target transform mode 和 split 对齐。

## 历史指标

指标来源：

```text
git show 4d20c8b:docs/train/gnn_model_v3_variant_context_gpu1_training_eval_2026-05-26.md
```

该指标属于 legacy v3 variant-context / posthoc stacker 口径，不应直接视为当前 30 维主线能力。

raw checkpoint 对比：

| model | WAPE | duration | cpu | memory | gpu util | SM active | SM occupancy | gpu mem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current v3vc raw | 0.085519 | 0.061608 | 0.023755 | 0.107358 | 0.105217 | 0.099961 | 0.149403 | 0.085200 |
| historical pool stack baseline | 0.062576 | 0.028941 | 0.019284 | 0.104335 | 0.094782 | 0.085366 | 0.132126 | 0.062010 |

stack 候选结果：

| candidate | WAPE | duration | cpu | memory | gpu util | SM active | SM occupancy | gpu mem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| historical pool, `base_phase` anchor | 0.062576 | 0.028941 | 0.019284 | 0.104335 | 0.094782 | 0.085366 | 0.132126 | 0.062010 |
| historical pool + current, `base_phase` anchor | 0.061270 | 0.028994 | 0.017633 | 0.095922 | 0.093376 | 0.084606 | 0.130260 | 0.060710 |
| historical pool + current, `current_v3vc` anchor | 0.060983 | 0.032270 | 0.015328 | 0.096983 | 0.093491 | 0.084663 | 0.130094 | 0.060417 |
| current only, `current_v3vc` anchor | 0.061668 | 0.034150 | 0.015111 | 0.097803 | 0.094003 | 0.084840 | 0.130004 | 0.061106 |

主 stack 结果：

| target | single GNN WAPE | stacked WAPE | absolute change | relative change |
| --- | ---: | ---: | ---: | ---: |
| overall | 0.085519 | 0.060983 | -0.024536 | -28.69% |
| `run_duration_sec_avg` | 0.061608 | 0.032270 | -0.029338 | -47.62% |
| `cpu_cores_p95` | 0.023755 | 0.015328 | -0.008427 | -35.47% |
| `memory_delta_gb_p95` | 0.107358 | 0.096983 | -0.010376 | -9.66% |
| `gpu_util_percent_p95` | 0.105217 | 0.093491 | -0.011726 | -11.14% |
| `gpu_sm_active_percent_p95` | 0.099961 | 0.084663 | -0.015298 | -15.30% |
| `gpu_sm_occupancy_percent_p95` | 0.149403 | 0.130094 | -0.019309 | -12.92% |
| `gpu_mem_used_mb_p95` | 0.085200 | 0.060417 | -0.024783 | -29.09% |

解读：

- 历史 stacker 明显降低 WAPE，但它是 validation-fitted 后处理结果。
- `current_only_current_anchor=0.061668` 表明单个 GNN 的 residual correction 特征已有强信号。
- 加入 historical pool 后为 `0.060983`，说明 checkpoint diversity 有额外收益。
- 这些结果不能解释为单一 GNN 主模型能力。
- 这些结果也不能直接证明 174 维 name-context 数据集适合作为主线。

## 当前推荐复现路线

推荐主线：当前 30 维 v3 数据 + current-only residual stacker。

目标：

- 判断当前简化架构特征口径下，GNN 后接 residual tree 是否仍有稳定收益。
- 避免恢复 name-context、runtime profile 占位和 schema padding。
- 让 stacked 实验成为后处理能力评估，而不是旧 schema 复活。

前置条件：

- 训练或定位一个明确使用当前 `data/scaled_v3`、`data/scalers_v3`、30 维 graph feature 的 GNN checkpoint。
- checkpoint 的 `graph_encoder.weight.shape[1]` 必须等于当前 dataset graph dim。
- result JSON 的 `metadata.data_dir` 和 `metadata.scaler_dir` 必须与实验数据匹配。
- 若没有逐样本 prediction 文件，先生成 validation/test 原始量纲预测。

主实验：

- raw GNN baseline。
- direct tree baseline，不加 residual。
- prediction-only residual tree。
- full-feature residual tree。

默认主候选：

- `current_only_current_anchor`
- anchor 为当前 30 维 GNN prediction。
- residual fit split 为 validation。
- eval split 为 test。
- 不使用 historical pool。

可选消融：legacy/name-context experiment。

- 仅用于理解历史指标来源。
- 不推荐作为主线。
- 必须标注 `variant_context` 是命名上下文。
- 必须标注 runtime profile 字段缺失、全 0 或无法证实。
- 不允许用当前机器重新 profile 来补齐旧 runtime 字段。

## 后续助手核对清单

进入实现或实验前先核对：

- 当前分支和 commit。
- `data/scaled_v3/train.pt` 的 graph feature dim。
- checkpoint 的 `graph_encoder.weight.shape[1]`。
- result JSON 的 `metadata.data_dir`、`metadata.scaler_dir`。
- scaler 文件是否与 target 顺序匹配。
- 是否有逐样本 val/test prediction。
- 是否存在静默 padding 或 schema 兼容逻辑。
- validation/test split 是否只用于各自职责。

实验输出至少记录：

- raw GNN overall/per-target WAPE。
- stacked overall/per-target WAPE。
- feature dim。
- anchor checkpoint。
- fit split 和 eval split。
- clipping policy。
- graph dim 核对结果。
- result JSON 路径。

不要接受以下结果：

- 只报告 overall，不报告 per-target。
- 用 test residual 训练或调参。
- graph feature dim 不一致但强行推理。
- 将 174 维临时 checkpoint 误当当前 30 维主线 checkpoint。
- 将 stacker 指标表述为单一 GNN 主模型能力。
- 将 runtime profile 占位当作真实 profiling 信号。
- 将 `variant_name` 解析特征当作纯架构图特征。

## 外部思路定位

相关外部思想是 GNN 与树模型互补。BGNN / “Boost then Convolve: Gradient Boosting Meets Graph Neural Networks” 属于 GBDT 与 GNN 结合的更复杂路线，其中树模型擅长 tabular feature，GNN 擅长图结构。

本项目历史方案更简单：先用 GNN 预测图性能，再用树模型学习原始量纲 residual。后续不应把它升级为联合训练 BGNN，除非另开研究任务。
