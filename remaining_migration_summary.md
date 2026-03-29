# gnn_predict 迁移完成与 TODO 总结

> 更新时间：2026-03-28  
> 依据：`/home/wangjh/gnn_predict` 当前代码现状、`gen_archs_gnn_model_merge_plan_v2.md`、已迁移配置与测试覆盖情况。

## 0. 更新日志

### 2026-03-28

- 已将 `ChannelPruning`、`GlobalChannelPruning`、`ChannelPruningByIndex` 迁移进 `src/gnn_archs/mutations.py`，并接入 `variant_runner` 主链。
- 已补充三类 CNN pruning 的聚焦测试，验证了模型构造、前向可运行、输出类别保持不变，以及剪枝后结构/参数量确实变化。
- 已执行聚焦验证：
  - `./.venv/bin/ruff check src/gnn_archs/mutations.py src/gnn_archs/variant_runner.py tests/test_variant_runner.py`
  - `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_variant_runner.py -q`
- 结论：迁移优先级最高的三类 CNN pruning 已不再属于“尚未迁移内容”，后续优先级应顺延到第二梯队图像 mutation 和模型族专用 mutation。

### 2026-03-28（第二轮更新）

- 已将 `ConvToDepthwiseSeparable`、`ConvToGroupedConv`、`AddNormalizationLayer`、`AddSEBlock`、`ModifyDropout` 迁移进 `src/gnn_archs/mutations.py`，并接入 `variant_runner` 主链。
- 已补充 5 类通用 CNN / VGG mutation 的聚焦测试，验证了：
  - 深度可分离卷积替换后参数量下降且前向可运行。
  - 分组卷积替换后 `groups` 配置生效且前向可运行。
  - 归一化层插入和 SE Block 插入后结构位置正确且前向可运行。
  - Dropout 修改后目标层概率确实变化且前向可运行。
- 已执行聚焦验证：
  - `./.venv/bin/ruff check src/gnn_archs/mutations.py tests/test_variant_runner.py`
  - `./.venv/bin/python -m pytest tests/test_variant_runner.py -q`
- 本轮结论：
  - 第二梯队通用 CNN mutation 已从“待迁移”转为“已迁移”。
  - 当前高优先级缺口已进一步收敛到 `ViT`、`BEiT`、`ConvNeXt` 等模型族专用 mutation，以及 pretrained / 数据准备 / 训练推理增强等后续能力。

### 2026-03-28（第三轮更新）

- 已将 `ViTBlockReduction`、`ViTModifyAttentionHeads`、`ViTModifyMLPDimension`、`ViTModifyDropout`、`ViTModifyEmbedDim` 迁移进 `src/gnn_archs/mutations.py`，并接入 `variant_runner` 主链。
- 已补充 5 个 ViT 聚焦测试，验证了：
  - block reduction 后深度下降且前向可运行。
  - attention heads / MLP hidden dim 修改后结构字段生效且前向可运行。
  - embed dim 组合修改后 patch embedding、attention、MLP、head 结构保持一致且前向可运行。
  - attention / MLP dropout 可以分别修改。
  - 非法 attention head 配置会直接抛错，不再静默跳过。
- 已执行聚焦验证：
  - `./.venv/bin/ruff check src/gnn_archs/mutations.py tests/test_variant_runner.py`
  - `PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_variant_runner.py -q`
- 本轮结论：
  - `ViT` 高频专用 mutation 已从“待迁移”转为“已迁移”。
  - 当前高优先级结构缺口进一步收敛到 `BEiT` 与 `ConvNeXt` 两个模型族。

### 2026-03-28（第四轮更新）

- 已将以下 `BEiT` / `ConvNeXt` 高频专用 mutation 迁移进 `src/gnn_archs/mutations.py`，并接入 `variant_runner` 主链：
  - `BEiTBlockReduction`
  - `BEiTLayerPruning`
  - `BEiTModifyMLPDimension`
  - `BEiTModifyDropout`
  - `BEiTModifyAttentionHeads`
  - `BEiTAttentionHeadPruning`
  - `BEiTChannelPruning`
  - `BEiTGlobalChannelPruning`
  - `ConvNeXtStageReduction`
  - `ConvNeXtMLPExpansionRatio`
  - `ConvNeXtKernelSizeModification`
- 同步补充了 7 个聚焦测试，覆盖了：
  - `BEiT` 的 block / layer / MLP / dropout / attention heads / head pruning / channel pruning / global pruning。
  - `ConvNeXt` 的 stage depth、MLP expansion ratio、depthwise kernel size。
- 已执行聚焦验证：
  - `cd /home/wangjh/gnn_predict && ./.venv/bin/ruff check src/gnn_archs/mutations.py tests/test_variant_runner.py`
  - `cd /home/wangjh/gnn_predict && PYTHONPATH=src ./.venv/bin/python -m pytest tests/test_variant_runner.py -q`
- 本轮结论：
  - 先前总结中的 `BEiT` / `ConvNeXt` 高频结构缺口已经补齐。
  - 当前 TODO 的重心已从“缺少主链结构 mutation”转移到失败路径与配置清洗、pretrained / 数据准备语义，以及训练推理增强。

## 1. 当前已完成

当前已经完成的迁移内容，可以概括为下面四类。

### 1.1 主链与基础结构已完成

- 已迁移统一配置 schema、配置迁移脚本、配置展开逻辑。
- 已迁移主入口 `main.py`，支持读取 YAML、解析 `--config`、`--output_dir`、`--gpu_node`、`--gpu_ids`。
- 已迁移单变体入口 `src/gnn_archs/variant_runner.py`，负责：
  - 加载基础模型
  - 应用一小批 mutation
  - 前向验证
  - ONNX 导出
  - 训练
  - 推理
  - 结果序列化
- 已迁移结果模型 `src/gnn_archs/result.py`，并使用 `pydantic` 约束导出结构。
- 已补充最小测试，覆盖配置解析、配置展开、单变体运行、ONNX 导出、训练、推理、结果序列化、CLI 主入口。

### 1.2 已迁移 mutation 范围

当前主链已支持的 mutation 范围如下。

- 图像模型：
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
- 文本模型：
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

### 1.3 已完成的测试覆盖

- 已覆盖最小主链闭环：
  - 配置解析
  - 配置展开
  - 单变体执行
  - ONNX 导出
  - 训练
  - 推理
  - 结果序列化
  - CLI 主入口
- 已覆盖 24 类图像 mutation：
  - `ChannelPruning`
  - `ChannelPruningByIndex`
  - `GlobalChannelPruning`
  - `ConvToDepthwiseSeparable`
  - `ConvToGroupedConv`
  - `AddNormalizationLayer`
  - `AddSEBlock`
  - `ModifyDropout`
  - `ViTBlockReduction`
  - `ViTModifyAttentionHeads`
  - `ViTModifyDropout`
  - `ViTModifyEmbedDim`
  - `ViTModifyMLPDimension`
  - `BEiTBlockReduction`
  - `BEiTLayerPruning`
  - `BEiTModifyMLPDimension`
  - `BEiTModifyDropout`
  - `BEiTModifyAttentionHeads`
  - `BEiTAttentionHeadPruning`
  - `BEiTChannelPruning`
  - `BEiTGlobalChannelPruning`
  - `ConvNeXtStageReduction`
  - `ConvNeXtMLPExpansionRatio`
  - `ConvNeXtKernelSizeModification`

### 1.4 当前已完成结论

- `gnn_predict` 已经具备“配置可解析、单变体主链可运行、结果可导出、最小测试可验证”的基础闭环。
- 通用 CNN 高频 mutation、`ViT` 高频专用 mutation、`BEiT` 高频专用 mutation、`ConvNeXt` 高频专用 mutation 已完成迁移。
- 当前剩余工作重点已经不在主入口壳子，而在失败路径与配置清洗、真实模型语义、真实数据准备，以及训练推理增强。

## 2. 当前 TODO（核心缺口）

从当前状态看，后续 TODO 可以分成四层：

- `P0`：补齐失败路径、参数校验和配置清洗。
- `P1`：补齐 pretrained 语义、真实 base model、真实数据准备、训练推理增强。
- `P2`：补齐结果结构、多配置汇总、运行元数据。
- `P3`：补齐更完整测试和长期回归覆盖。

### 2.1 高频 mutation 家族 TODO

上一版总结里的高频结构缺口已经收敛并补齐：

- `ChannelPruning` / `GlobalChannelPruning` / `ChannelPruningByIndex`
  - 是图像模型配置里的主力 mutation。
  - 这三类已经进入当前主链，不再属于“待迁移”。
- `ConvToDepthwiseSeparable` / `ConvToGroupedConv` / `AddNormalizationLayer` / `AddSEBlock` / `ModifyDropout`
  - 是当前配置里第二梯队的高频通用 CNN mutation。
  - 这五类也已经进入当前主链，不再属于“待迁移”。
- `ViTBlockReduction` / `ViTModifyAttentionHeads` / `ViTModifyMLPDimension` / `ViTModifyDropout` / `ViTModifyEmbedDim`
  - 是当前配置中最主要的一组 ViT 专用 mutation。
  - 这五类已经进入当前主链，不再属于“待迁移”。
- `BEiTBlockReduction` / `BEiTLayerPruning` / `BEiTModifyMLPDimension` / `BEiTModifyDropout` / `BEiTModifyAttentionHeads` / `BEiTAttentionHeadPruning` / `BEiTChannelPruning` / `BEiTGlobalChannelPruning`
  - 是当前配置中最主要的一组 `BEiT` 专用 mutation。
  - 这八类已经进入当前主链，不再属于“待迁移”。
- `ConvNeXtStageReduction` / `ConvNeXtMLPExpansionRatio` / `ConvNeXtKernelSizeModification`
  - 是当前配置中最主要的一组 `ConvNeXt` 专用 mutation。
  - 这三类已经进入当前主链，不再属于“待迁移”。

当前最高优先级 TODO：

- 高频 mutation 的失败路径和参数校验
  - 虽然高频结构 mutation 已基本接入主链，但当前测试仍主要覆盖成功路径。
  - 针对非法层名、非法分组数、非法插层位置等参数错误，仍需要单独补齐失败路径测试。
- 配置清洗与别名收敛
  - 迁移后 schema 虽已统一，但部分历史配置仍可能混有旧命名或旧参数语义。
  - 这部分需要继续清洗，避免出现“配置可解析但语义未完全对齐”的灰区。

### 2.2 基础模型加载能力还没有迁齐

当前主链的基础模型加载明显是“最小可跑版”，还没有达到旧项目覆盖面：

- 图像模型当前直接走 `timm.create_model(..., pretrained=False)`。
- 文本模型当前直接基于 `BertConfig` / `BertForSequenceClassification` 重新构造。
- `base_model.pretrained` 目前并没有真正驱动预训练权重加载。

因此仍未迁移的能力包括：

- 从 Hugging Face / timm 加载真实预训练权重。
- 更接近旧 `get_base_model(...)` 的多模型族分发逻辑。
- 不同模型族的真实 base model 适配，而不只是“能构造出一个同家族骨架”。

这意味着当前结果更接近“结构变体工具验证”，还不是“完整复现旧实验语义”。

### 2.3 真实数据集准备逻辑还没有迁移

当前只支持假数据：

- 图像训练/推理只支持 `use_fake_imagenet=True`。
- 文本训练/推理只支持 `use_fake_text_dataset=True`。

仍未迁移的内容包括：

- 真实图像数据集读取与组织逻辑。
- 真实文本数据集读取与组织逻辑。
- 独立的数据准备模块。
- `data/archs` 目录约定的真正落地。

目前 `/home/wangjh/gnn_predict/data` 目录只有根目录，尚未形成：

- `data/archs`
- `data/archs/yolo`
- 各模型族数据子目录

### 2.4 训练与推理流程仍然是最小实现

当前训练/推理逻辑只是一个可验证主链的最小版，不等于旧项目能力已迁齐。

仍未迁移的内容包括：

- 更完整的训练超参数控制。
- 多 batch size 探测或调度策略。
- 更完整的 optimizer / scheduler 配置体系。
- 训练 history、epoch 级指标、checkpoint 元信息。
- 更接近旧项目 workload 语义的推理评估。
- 多 GPU 训练能力。
- `gpu_ids` 的真正多卡使用，当前只是取第一个设备。

### 2.5 ONNX 导出能力仍然是通用版本

当前 ONNX 导出已经可用，但仍是统一通用路径，不是旧项目的模型族特化实现。

仍未迁移的内容包括：

- 针对不同模型族的导出路径分支。
- 更丰富的导出参数控制。
- opset 自动协商或多版本策略。
- 更细的图结构统计。
- 导出失败后的明确分类与诊断信息。

补充说明：

- 当前为了适配本地 PyTorch 环境，显式使用了 legacy exporter（`dynamo=False`）。
- 这解决了当前环境里缺少 `onnxscript` 的问题，但不代表导出链路已经完全定型。

### 2.6 结果结构与旧样例之间仍有差距

当前结果结构已经比旧项目干净，但还没有完全扩展到计划中期待的完整程度。

仍未迁移或未完成的部分包括：

- 更完整的训练超参数导出。
- 更完整的优化器配置导出。
- 更细的阶段统计和流程摘要。
- 多配置运行的全局 summary 文件。
- 更接近长期分析用途的字段设计。

当前已经明确没有保留旧结构里的无效字段和 `N/A` 字符串，这是符合迁移计划的；但“新结构更完整的最终形态”还没全部做完。

### 2.7 主入口编排能力仍然偏最小化

当前主入口可以跑，但还没有把旧项目中一些编排级能力重新设计完成。

仍待补充的内容包括：

- 多配置运行的更完整全局汇总。
- 更清晰的失败隔离与失败报告。
- 面向集群作业的更完整运行元数据。
- 对 base model group 级别执行结果的更细粒度组织。

### 2.8 测试覆盖 TODO

当前虽然已经有最小闭环测试和一批图像 mutation 测试，但大量高风险区域还没有测试。

仍缺少的测试主要包括：

- 不同模型族 mutation 的参数校验测试。
- 真实数据集路径的测试。
- 多配置运行测试。
- 失败路径测试。
- 预训练权重加载测试。
- 更复杂结果结构的序列化与回归测试。

### 2.9 配置质量清理 TODO

当前已迁移配置中存在至少一个异常 mutation 名：

- `ActivationSworkload_iterationsp`

这明显不是合法的目标 schema 内容，说明：

- 迁移后的 YAML 里还残留旧配置噪声或迁移污染。
- 需要单独做一轮配置清洗和校验，而不只是依赖运行时报错。

## 3. 按计划暂不迁移 / 不迁移

下面这些内容不应当和“尚未完成但要继续做”混在一起，它们属于按计划明确不迁移，或本轮不纳入主链：

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

这些内容如果后续要做，应作为新的独立迁移任务，而不是继续塞回当前主链。

## 4. 建议的下一阶段优先级

如果继续按照当前迁移计划推进，建议优先级如下：

1. 先补高频 mutation 的失败路径、参数校验测试和配置清洗
2. 独立实现真实数据准备模块，并落地 `data/archs`
3. 扩展 pretrained 语义、真实 base model 适配，以及训练 / 推理增强
4. 再完善结果结构、多配置全局 summary 和运行元数据
5. 最后再评估是否需要迁移额外的旧项目辅助能力

## 5. 一句话结论

`gnn_predict` 目前已经从“只有配置 schema”推进到“主链可跑”，并且已经补完通用 CNN、`ViT`、`BEiT`、`ConvNeXt` 的高频结构 mutation；后续 TODO 的核心，已经收敛到：

- 高频 mutation 的失败路径与参数校验
- 配置清洗与 schema 收敛
- 真实 base model / pretrained 语义
- 真实数据准备
- 更完整的训练 / 推理 / 结果导出能力
