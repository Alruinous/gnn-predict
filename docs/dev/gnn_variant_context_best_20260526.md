# GNN variant-context best overall 方案 20260526

## 目标

本方案固化 2026-05-26 多轮实验中 overall WAPE 最好的 GNN predictor 路径：

- 数据：`data/scaled_v3_variant_context`
- 配置：`config/gnn_model/scaled_training_v3_variant_context_20260526.yaml`
- result JSON：
  `output/gnn_model_v3_variant_context_20260526/gnn_model_scaled_v3_variant_context_20260526/results/gnn_model_scaled_v3_variant_context_20260526_1779771864_result.json`
- overall WAPE：`0.085933`

这不是 `SM occupancy` 最优路径。当前正式 `SM occupancy` best 仍是旧的 repeated OOF stacker：
`0.119378`。本方案选择依据是用户指定的“总体 WAPE 表现最好一次”。

## 实现范围

核心代码：

- `src/gnn_model/data/variant_context.py`
- `src/gnn_model/data/constants.py`
- `src/gnn_model/data/onnx_graph.py`
- `src/gnn_model/data/extract.py`
- `src/gnn_model/models/predictor.py`
- `src/gnn_model/config.py`
- `src/gnn_model/runner.py`

固化脚本：

- `scripts/build_variant_context_dataset.py`

训练配置：

- `config/gnn_model/scaled_training_v3_variant_context_20260526.yaml`

## 特征设计

新增 `VARIANT_CONTEXT_FEATURE_NAMES`，共 106 维，追加到 graph-level features 末尾。

特征分组：

- architecture family one-hot：`resnet`、`vgg16`、`yolov11`、`bert` 等。
- size word：`tiny`、`small`、`base`、`large`、`wide`、`deep` 等。
- activation word：`relu`、`gelu`、`silu`、`hardswish` 等。
- mutation word：`c2f`、`kernel`、`depth`、`channel`、`prune`、`sppf` 等。
- 数字摘要：token count、number count、sum、mean、std、min、max、first、last。
- 数字前缀：`ic`、`oc`、`kernel`、`depth`、`channel`、`head`、`layer`、`scale` 等。

`GRAPH_FEATURE_DIM` 从 68 增加到 174。variant-context 放在最后，旧 v2/v3/v4 数据仍可通过末尾补零加载。

## 数据生成

完整 extract 路径已经传入 `model_name` 和 `variant_name`，重新 extract 时会直接生成 variant-context。

为了复用已经生成好的 v3 peak-live extracted 数据，本方案也固化了一个轻量派生脚本：

```bash
PYTHONPATH=src uv run python scripts/build_variant_context_dataset.py \
  --source_dir data/extracted_v3_peak_live \
  --output_dir data/extracted_v3_variant_context
```

随后重新 scaler/scale：

```bash
PYTHONPATH=src uv run python -m gnn_model.data.scaler \
  --data_dir data/extracted_v3_variant_context \
  --scaler_output_path data/scalers_v3_variant_context \
  --scaled_data_output_path data/scaled_v3_variant_context \
  --target_fields duration_sec_avg,cpu_cores_p95,memory_delta_gb_p95,gpu_util_percent_p95,gpu_sm_active_percent_p95,gpu_sm_occupancy_percent_p95,gpu_mem_used_mb_p95
```

## 训练配置

```yaml
model:
  hidden_dim: 128
  num_layers: 2
  num_heads: 8
  dropout_rate: 0.1
  readout_mode: mean_sum_max
  structural_context_mode: basic
training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.0008
  weight_decay: 0.0001
  loss_weights: [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0]
```

训练命令：

```bash
PYTHONPATH=src uv run python -m gnn_model \
  --config config/gnn_model/scaled_training_v3_variant_context_20260526.yaml \
  --output_dir output/gnn_model_v3_variant_context_20260526 \
  --device cuda:0
```

## 指标

| metric | WAPE |
| --- | ---: |
| overall | 0.085933 |
| duration | 0.069336 |
| cpu | 0.030306 |
| memory delta | 0.114935 |
| gpu util | 0.112200 |
| SM active | 0.106593 |
| SM occupancy | 0.161136 |
| gpu mem | 0.085488 |

训练摘要：

| item | value |
| --- | --- |
| best epoch | 81 |
| best val loss | 0.069185 |
| last train loss | 0.047406 |
| parameter count | 997,760 |

## 设计取舍

- 这版代码不把完整 `variant_name` 当作 ID 特征使用，而是拆成研发人员能读懂的 family、mutation、activation 和数字前缀。
- `variant_context.py` 保留直接解析逻辑，避免引入 HashingVectorizer 这类不透明依赖。
- 数据不做运行时容错修补；缺少必要 graph 属性时直接失败，要求重新生成正确数据。
- 这版方案提升 overall，但没有解决 `SM occupancy`。若后续目标转回 `SM occupancy`，应继续做 target-specific checkpoint selection 或接入 runtime/kernel 信息。

## 验证

已运行：

- `uv run ruff check src/gnn_model tests/test_gnn_model_data.py tests/test_gnn_model_model.py tests/test_gnn_model_config.py`
- `PYTHONPATH=src uv run python -m pytest tests/test_gnn_model_data.py -q`
- `PYTHONPATH=src uv run python -m pytest tests/test_gnn_model_model.py tests/test_gnn_model_config.py -q`
- `uv run ty check src/gnn_model/data/variant_context.py src/gnn_model/data/constants.py src/gnn_model/data/onnx_graph.py src/gnn_model/data/extract.py src/gnn_model/data/dataset.py src/gnn_model/models/predictor.py`
