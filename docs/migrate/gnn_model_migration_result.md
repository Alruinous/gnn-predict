# GNN 预测器 V1 迁移结果

> 更新时间：2026-04-15  
> 依据：`/home/wangjh/gnn_predict` 当前仓库落地代码、旧项目分析文档、已新增测试与本轮日志收敛结果。

## 1. 结论

`gnn_model` 已完成第一版主链迁移，当前仓库已经具备一条独立于 `gnn_archs` 的最小可运行 GNN 性能预测流程：

- `ONNX -> PyG Data`
- prepared/split dataset 读取
- `IntelliGraphLargeModelPredictor` 前向
- 单卡训练
- 测试集评估
- 结构化 JSON 结果导出
- 独立 CLI 运行

这次迁移的目标不是复刻旧项目全部行为，而是先把旧 `gnn_model` 中真正有价值且仍可维护的主链收敛为一套干净实现。旧项目里的错误数据、复杂优化器、历史增强线、缓存兼容层、恢复逻辑和废弃模型都没有进入新主链。

## 2. 当前已落地的代码

### 2.1 目录与入口

当前 `gnn_model` 相关代码已经落地到以下位置：

- `src/common/onnx_initializer.py`
- `src/gnn_model/config.py`
- `src/gnn_model/result.py`
- `src/gnn_model/runner.py`
- `src/gnn_model/__main__.py`
- `src/gnn_model/data/`
- `src/gnn_model/models/`
- `src/gnn_model/training/`
- `src/gnn_model/evaluation/`

当前运行方式为：

```bash
PYTHONPATH=src python -m gnn_model --config config/gnn_model/scaled_training.yaml --output_dir output --device cpu
```

该入口不会并入当前根 `main.py`，因此不会影响现有 `gnn_archs` 主链。

### 2.2 共享 ONNX 初始化能力

原先只服务 `gnn_archs` 的 architecture-only ONNX 参数初始化逻辑，现已抽到：

- `src/common/onnx_initializer.py`

`src/gnn_archs/util/onnx_initializer.py` 现在只做兼容转发，避免两套实现继续分叉。

这意味着：

- `gnn_archs` 仍可继续使用原有导出与随机初始化能力
- `gnn_model` 可以直接复用同一套参数注入逻辑
- 后续 ONNX 初始化修复只需要维护一处

### 2.3 数据主链

当前数据主链已经具备以下能力：

- `build_graph_data_from_onnx(...)`
  - 读取 ONNX
  - 做 shape inference
  - 提取节点、边、全图、图指标特征
  - 输出 `torch_geometric.data.Data`
- `load_split_graph_datasets(...)`
  - 读取 `data/scaled/train.pt`
  - 读取 `data/scaled/val.pt`
  - 读取 `data/scaled/test.pt`
  - 校验 PyG 图样本字段与目标维度

当前特征维度与目标名统一收口在：

- `src/gnn_model/data/constants.py`

其中包含：

- `NODE_FEATURE_DIM`
- `EDGE_FEATURE_DIM`
- `GRAPH_FEATURE_DIM`
- `GRAPH_METRIC_DIM`

### 2.4 模型与训练

当前只保留一个公开模型：

- `gnn_model.models.IntelliGraphLargeModelPredictor`

这版实现不是机械搬运旧项目，而是保留其核心输入契约与多目标回归形态，收敛成可维护的最小实现：

- 输入为 `PyG Data`
- 必需字段：
  - `x`
  - `edge_index`
  - `edge_attr`
  - `graph_features`
  - `graph_metrics`
- 可选字段：
  - `node_op_token_id`
- 输出为 `[B, num_targets]` 的多任务回归结果

训练侧已经重写为单卡优先的简化实现：

- 优化器：`AdamW`
- 损失：weighted `SmoothL1`
- 只保留必要的 epoch 循环、验证、checkpoint 与指标汇总

没有迁移以下旧逻辑：

- DDP
- Lookahead
- DynamicLossWeighter
- 复杂 scheduler
- 预测后处理修补逻辑
- 旧 `src/optim` 中的复杂优化器体系

### 2.5 结果导出与输出目录

当前结果模型位于：

- `src/gnn_model/result.py`

结果根对象已包含：

- `schema_version`
- `experiment_name`
- `config_path`
- `device`
- `target_names`
- 数据集切分摘要
- 训练摘要
- 评估摘要
- 各阶段时间戳
- 运行元数据

输出目录统一为：

- `logs/`
- `results/`
- `checkpoints/`

没有为旧项目保留 `N/A` 字段，也没有继续兼容旧结果结构。

## 3. 日志初始化现状

`gnn_model` 的日志初始化已经收敛为与当前 `gnn_archs` 主链一致的直接方式，位置在：

- `src/gnn_model/runner.py::configure_logging`

当前实现特点：

- 直接使用 `logging.getLogger(...)`
- 显式清理旧 handler
- 同时输出到控制台和日志文件
- 不再保留额外的 `logger.py` 包装层

这次收敛的目的，是让 `gnn_model` 的日志初始化方式与当前仓库主风格一致，避免再引入单独的日志抽象层。

## 4. 已完成验证

迁移过程中已补充以下测试：

- `tests/test_gnn_model_config.py`
- `tests/test_gnn_model_data.py`
- `tests/test_gnn_model_model.py`
- `tests/test_gnn_model_pipeline.py`
- `tests/test_gnn_model_cli.py`

覆盖点包括：

- 配置解析
- ONNX 转图
- split dataset 读取
- 模型前向
- 训练与结果落盘
- CLI 端到端运行

已执行的检查包括：

```bash
uv run ruff check src/common/onnx_initializer.py src/gnn_archs/util/onnx_initializer.py src/gnn_model gnn_model tests/test_gnn_model_config.py tests/test_gnn_model_data.py tests/test_gnn_model_model.py tests/test_gnn_model_pipeline.py tests/test_gnn_model_cli.py
```

```bash
uv run python -m pytest tests/test_gnn_model_config.py tests/test_gnn_model_data.py tests/test_gnn_model_model.py tests/test_gnn_model_pipeline.py tests/test_gnn_model_cli.py -q
```

```bash
uv run python -m pytest tests/test_variant_runner.py -q -k "test_image_variant_runner_can_export_architecture_only_onnx or test_randomized_architecture_only_onnx_runs_with_runtime_inputs_only"
```

其中：

- `gnn_model` 新增测试当前通过
- `gnn_archs` 的 ONNX architecture-only 初始化回归当前通过

## 5. 明确未迁移的内容

这次迁移明确没有做以下事情：

- 不迁旧项目里的错误数据与历史缓存
- 不迁 `src/data/data` 中的旧图数据资产
- 不迁旧 `src/optim` 的复杂优化器体系
- 不迁 `IntelliGraphPredictor` 及其他历史废弃模型
- 不迁旧入口中的恢复、缓存、检查点参数兼容逻辑
- 不迁旧评估后处理和 CSV 报表链路

这些内容没有进入新主链，是刻意设计，不是遗漏。

## 6. 当前仍未完成

当前 `gnn_model` 仍然是 V1，不是完整迁移终态。主要缺口有三类：

### 6.1 真实数据训练已接入

当前支持从 `data/scaled` 读取归一化后的真实 split 数据，并在评估阶段通过 `data/scalers/target_scalers.pkl` 追加原始量纲指标：

- `src/gnn_model/data/dataset.py`
- `src/gnn_model/data/prepared_dataset.py`
- `config/gnn_model/scaled_training.yaml`

### 6.2 旧项目更深的模型语义尚未恢复

当前模型是基于迁移目标重建的最小在线版，不等于旧项目全部实验能力的完整恢复。尤其未包含：

- 旧多分支增强损失体系
- 旧 MoE 及相关辅助损失链路
- 旧训练稳定性补丁与长周期训练策略

### 6.3 真实生产数据契约还未定型

当前结果结构已经是新格式，但真实数据 manifest、prepared dataset wire shape、长期训练元数据字段还没有最终定型。

## 7. 当前判断

截至 2026-04-15，`gnn_model` 的状态可以判断为：

- 已完成 V1 主链迁移
- 已具备最小可运行、可测试、可扩展的独立子系统形态
- 已与 `gnn_archs` 在 ONNX 初始化能力上完成共享
- 已完成真实数据接入，尚未完成更完整训练语义恢复

当前最合理的下一步不是回头搬旧复杂逻辑，而是继续沿着新主链补：

- 长周期真实数据训练配置
- 更完整的训练与评估字段
- 面向真实数据的回归测试
