# gnn_predict 迁移完成与 TODO 总结

> 更新时间：2026-03-29  
> 依据：`/home/wangjh/gnn_predict` 当前目录现状、`gen_archs_gnn_model_merge_plan_v2.md`、`gen_archs_project_analysis.md`、全量配置展开核对、当前 pytest 覆盖情况。

## 0. 更新日志

### 2026-03-29（第六轮更新）

- 已完成第六轮 `P0` 配置清洗：
  - 收紧文本模型识别逻辑，`resnet50` 不再被 `t5` 子串误判。
  - 为 `ConvNeXt` 增加专用 `ConvNeXtModifyDropout` mutation。
  - 清理 `convnext_variants.yaml` 和 `densenet_variants.yaml` 中已暴露的命名污染。
- 已补充配置审计与回归测试：
  - `tests/test_arch_configs.py`
  - `tests/test_variant_expander.py`
  - `tests/test_variant_runner.py`
- 已重新核对当前真实状态：
  - `config/arch/*.yaml` 现为 15/15 份都可通过当前 `ArchConfig + expand_arch_config` 展开。
  - 当前可展开的变体总数为 4912。
  - `data` 目录仍只有根目录，没有 `data/archs`、`data/archs/yolo` 或其他已迁移数据目录。
- 已执行当前主链回归：
  - `cd /home/wangjh/gnn_predict && PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_main.py tests/test_config_migration.py tests/test_variant_expander.py tests/test_variant_runner.py tests/test_arch_configs.py -q`
  - 结果：`62 passed`
- 本轮重新判定后的结论：
  - 配置侧已基本齐全，不再是当前主阻塞项。
  - 代码侧已经具备“最小可运行主链”，但还不是迁移计划要求的完整形态。
  - 数据迁移和真实数据准备仍明显未完成，因此整体迁移计划不能判定为完成。

### 2026-03-29（第五轮更新，历史）

- 当时确认主链实际落地为：
  - `main.py`
  - `src/gnn_archs/config.py`
  - `src/gnn_archs/util/config_migration.py`
  - `src/gnn_archs/util/variant_expander.py`
  - `src/gnn_archs/mutations.py`
  - `src/gnn_archs/variant_runner.py`
  - `src/gnn_archs/result.py`
- 当时识别出的三个配置问题：
  - `resnet50_variants.yaml` 无法展开
  - `convnext_variants.yaml` 里仍有图像侧不成立的 `DropoutModification`
  - `densenet_variants.yaml` 里仍有非法 mutation 名 `ActivationSworkload_iterationsp`
- 说明：
  - 上述三项都已经在第六轮更新中修复，不应再视为“当前仍存在的问题”。
  - 第五轮里大量使用“当前仍...”的表述，在本文件中应理解为“第五轮核对时的状态”，不是第六轮后的当前状态。

### 2026-03-28

- 已将 `ChannelPruning`、`GlobalChannelPruning`、`ChannelPruningByIndex` 迁移进 `src/gnn_archs/mutations.py`，并接入 `variant_runner` 主链。
- 已补充三类 CNN pruning 聚焦测试。

### 2026-03-28（第二轮更新）

- 已将 `ConvToDepthwiseSeparable`、`ConvToGroupedConv`、`AddNormalizationLayer`、`AddSEBlock`、`ModifyDropout` 迁移进主链。
- 已补充通用 CNN / VGG mutation 聚焦测试。

### 2026-03-28（第三轮更新）

- 已将 `ViTBlockReduction`、`ViTModifyAttentionHeads`、`ViTModifyMLPDimension`、`ViTModifyDropout`、`ViTModifyEmbedDim` 迁移进主链。
- 已补充 ViT 聚焦测试。

### 2026-03-28（第四轮更新）

- 已将 `BEiT` / `ConvNeXt` 高频结构 mutation 迁移进主链。
- 已补充对应聚焦测试。

## 1. 当前已完成

### 1.1 主链代码与结构

- 当前主链实际落地为：
  - `main.py`
  - `src/gnn_archs/config.py`
  - `src/gnn_archs/util/config_migration.py`
  - `src/gnn_archs/util/variant_expander.py`
  - `src/gnn_archs/mutations.py`
  - `src/gnn_archs/variant_runner.py`
  - `src/gnn_archs/result.py`
- `main.py` 已支持：
  - 读取 YAML
  - 解析 `--config`、`--output_dir`、`--gpu_node`
  - 创建统一输出目录结构
  - 顺序执行多配置
  - 为每个配置写结果 JSON
- `variant_runner.py` 已支持：
  - 加载基础模型
  - 应用 mutation
  - 前向验证
  - ONNX 导出
  - 使用 fake data 做训练
  - 推理
  - 返回结构化结果
- `result.py` 已提供 `pydantic` 结果模型和 JSON 导出函数。

### 1.2 配置现状

- `config/arch` 下已有 15 份迁移后的 YAML：
  - `beit_variants.yaml`
  - `bert_large_variants.yaml`
  - `bert_variants.yaml`
  - `convnext_variants.yaml`
  - `densenet_variants.yaml`
  - `inception_v3_variants.yaml`
  - `mobilenet_variants.yaml`
  - `resnet101_variants.yaml`
  - `resnet152_variants.yaml`
  - `resnet50_variants.yaml`
  - `resnet_variants.yaml`
  - `vgg11_variants.yaml`
  - `vgg16_variants.yaml`
  - `vgg19_variants.yaml`
  - `vit_variants.yaml`
- 当前这 15 份配置都能通过统一 schema 校验与组合展开。
- 当前总变体数为 4912。
- 第五轮里列出的配置命名与展开问题已经清完，因此“配置清洗与全量可展开性”不再是当前第一优先级。

### 1.3 已迁移 mutation 范围

- 图像模型当前主链已支持：
  - `ActivationSwap`
  - `ActivationFunctionSwap`
  - `AddIntermediateFCLayer`
  - `AddNormalizationLayer`
  - `AddSEBlock`
  - `BEiTAttentionHeadPruning`
  - `BEiTBlockReduction`
  - `BEiTChannelPruning`
  - `BEiTGlobalChannelPruning`
  - `BEiTLayerPruning`
  - `BEiTModifyAttentionHeads`
  - `BEiTModifyDropout`
  - `BEiTModifyMLPDimension`
  - `ChannelPruning`
  - `ChannelPruningByIndex`
  - `ConvKernelReplacement`
  - `ConvNeXtKernelSizeModification`
  - `ConvNeXtMLPExpansionRatio`
  - `ConvNeXtModifyDropout`
  - `ConvNeXtStageReduction`
  - `ConvToDepthwiseSeparable`
  - `ConvToGroupedConv`
  - `GlobalChannelPruning`
  - `ModifyDropout`
  - `ReplaceFCStructure`
  - `ViTBlockReduction`
  - `ViTModifyAttentionHeads`
  - `ViTModifyDropout`
  - `ViTModifyEmbedDim`
  - `ViTModifyMLPDimension`
- 文本模型当前主链已支持：
  - `ActivationFunctionSwap`
  - `AttentionHeadsModification`
  - `BertAttentionHeadPruning`
  - `BertHiddenSizePruning`
  - `BertLayerPruning`
  - `DropoutModification`
  - `HiddenSizeModification`
  - `IntermediateSizeModification`
  - `PositionEncodingModification`
  - `TransformerLayerReduction`
  - `VocabSizeModification`

### 1.4 验证与测试

- 当前 pytest 已覆盖：
  - 配置解析
  - 配置迁移
  - 配置展开
  - 全量配置文件审计
  - 单变体执行
  - ONNX 导出
  - 训练
  - 推理
  - 结果序列化
  - CLI 主入口
- 当前主链回归结果为：
  - `62 passed`

### 1.5 当前真实完成度判断

- 配置：基本齐了。
- 代码：最小可运行主链已经齐了，但完整迁移版代码还没齐。
- 数据：明显没齐。

## 2. 当前 TODO

从第六轮更新后的真实状态看，后续 TODO 可以分成五层：

- `P0`：落地真实数据目录与数据准备模块。
- `P1`：补齐 pretrained 语义与真实 base model 加载能力。
- `P2`：补齐训练 / 推理 / 结果结构的完整语义。
- `P3`：补齐主入口编排、运行元数据和全局汇总。
- `P4`：补齐面向真实数据、预训练和多配置运行的回归测试。

### 2.1 数据迁移与目录落地

当前数据侧是最明显的未完成项。

- `/home/wangjh/gnn_predict/data` 目录目前只有根目录。
- 计划里要求的下面这些目录都还没有形成：
  - `data/archs`
  - `data/archs/yolo`
  - 各模型族数据子目录
- 当前仍未迁移：
  - 真实图像数据集读取与组织逻辑
  - 真实文本数据集读取与组织逻辑
  - 独立的数据准备模块
  - YOLO 数据目录约定的实际落地

### 2.2 基础模型加载与 pretrained 语义

当前主链的基础模型加载仍是“最小可跑版”，还不是旧项目语义的完整迁移。

- 图像模型当前直接走 `timm.create_model(..., pretrained=False)`。
- 文本模型当前直接基于 `BertConfig` / `BertForSequenceClassification` 重新构造。
- `base_model.pretrained` 当前并没有真正驱动预训练权重加载。
- 导出结果里 `pretrained_weights_loaded` 当前固定为 `False`。

因此仍未迁移的内容包括：

- 从 Hugging Face / timm 加载真实预训练权重。
- 更接近旧 `get_base_model(...)` 的多模型族分发逻辑。
- 不同模型族的真实 base model 适配，而不是“只构造同家族骨架”。

### 2.3 训练与推理语义

当前训练 / 推理链路能跑，但仍是最小实现。

- 真实数据仍不支持：
  - 文本训练逻辑只要不是 `use_fake_text_dataset=True` 就会 `NotImplementedError`
  - 图像训练逻辑只要不是 `use_fake_imagenet=True` 就会 `NotImplementedError`
- `training_batch_sizes` 当前只实际使用第一个 batch size。
- 运行链路当前固定使用第一个容器可见 GPU（`cuda:0`）；监控查询使用 monitor 配置中的 `gpu_id`。
- 仍未迁移：
  - 更完整的训练超参数控制
  - 多 batch size 调度或探测策略
  - 更完整的 optimizer / scheduler 配置
  - 训练 history、epoch 级指标、checkpoint 元信息
  - 更接近旧 workload 语义的推理评估
  - 多 GPU 训练能力

### 2.4 结果结构与主入口编排

当前结果结构和主入口都属于“干净但偏最小”的版本。

- `result.py` 已比旧项目干净，但仍未补齐：
  - 更完整的训练超参数导出
  - 更完整的优化器配置导出
  - 更细的阶段统计和流程摘要
  - 更接近长期分析用途的字段设计
- `main.py` 当前虽然支持多配置顺序运行，但仍未补齐：
  - 更完整的多配置全局 summary
  - base model group 级别结果组织
  - 更完整的运行元数据
  - 更清晰的失败报告

### 2.5 依赖清理

当前 `pyproject.toml` 还没有完全收敛到“只保留主链必需依赖”的状态。

- 当前仍保留了一些更偏 notebook / 可视化 / 监控的依赖。
- 这不影响当前最小主链运行，但和迁移计划里“主链依赖收敛”的目标不完全一致。

### 2.6 测试覆盖

当前测试已经足够支撑“最小主链 + 配置审计”，但还不够支撑完整迁移完成判定。

仍缺少或还不完整的测试包括：

- 真实图像数据集路径测试
- 真实文本数据集路径测试
- 预训练权重加载测试
- 多配置运行与全局汇总测试
- 更复杂结果结构的回归测试
- 更完整的失败路径测试

### 2.7 已不再属于当前 TODO 的事项

下面这些内容在第五轮时还是当前问题，但在第六轮后不应再继续列为“当前剩余 TODO”：

- `resnet50_variants.yaml` 展开失败
- `convnext_variants.yaml` 使用文本侧 `DropoutModification`
- `densenet_variants.yaml` 中的 `ActivationSworkload_iterationsp`
- “全量配置文件无法全部展开”
- “缺少配置文件与 mutation 实现集合一致性的测试”

## 3. 按计划暂不迁移 / 不纳入当前主链

下面这些内容不应和“尚未完成但要继续做”混在一起，它们属于按计划明确不迁移，或当前不纳入主链：

- `gen_archs/arch_config copy`
- `gen_archs/res`
- `gen_archs/data_analysic` 下的历史 csv / notebook / 报表脚本
- `performance_monitor.py` 及其旧结果结构依赖链
- `gen_archs/yolo` 主流程
  - `validate_yolo_variants.py`
  - `validate_yolo_variants_v2.py`
  - `gen_yolo_variants_config.py`
  - `yolo_config` 历史配置
  - `yolo_validation_results`
- `gen_archs/model_trainer.py`
- 旧 `config_loader.py`
- 旧 `logger_setup.py`
- 旧 `result_manager.py` 的兼容结构

说明：

- 上述内容如果后续要做，应作为新的独立迁移任务。
- 当前主链只需要保留 YOLO 数据目录约定，而不是迁移 YOLO 主流程代码。

## 4. 一句话结论

`gnn_predict` 当前更适合定义为：

- 配置基本齐
- 主链最小闭环可运行
- 数据迁移与完整运行语义仍未完成

因此，`gen_archs_gnn_model_merge_plan_v2.md` 整体**还不能判定为完成**。
