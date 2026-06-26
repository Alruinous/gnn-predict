# Workflow 图文理解 Agent 临时实验记录

## 结论

本次完成了一次临时、零落盘的 workflow agent 实验。实验目标是验证当前
`src/workflow` 的 LLM ReAct 主节点能否把本地部署在加速卡上的 CV/NLP 模型封装为工具节点调用，并把工具执行结果交回 LLM 汇总回答。

实验结果：

- 外部 LLM provider 的 ReAct workflow invoke 成功。
- LLM 实际调用了两个工具节点：`text_classifier` 和 `scene_classifier`。
- 两个工具节点都在 `Tesla V100-SXM2-32GB` 上完成真实 PyTorch forward。
- 最终回答由 LLM 汇总，且明确区分了“加速卡部署证据来自工具节点元数据”，不是来自真实图片解码。
- 本次按确认口径暂时不接入 GNN 预测器，不做资源预测、不做 adaptive scheduling。

## 背景

项目当前有两条相关主线：

- `src/workflow`：定义 workflow DAG、节点 schema、运行结果结构、tool node 包装和 LangGraph ReAct agent 接入。
- `src/gnn_archs`：负责模型变体构造、训练/推理/ONNX 导出和监控数据采集，本次复用其中的模型构造与 forward 工具函数。

文档层面，未发现 `dcs/` 目录；本次按 `docs/` 作为项目背景文档来源。重点参考：

- `docs/workflow/schema_runtime_20260626.md`
- `docs/workflow/research_plan_20260626.md`
- `docs/train/gnn_model_v3_op_reclass_full_retrain_20260607.md`
- `docs/train/gnn_model_llm_full_retrain_20260607.md`

相关已有判断：

- Workflow 配置只描述 DAG、节点模型先验和运行输入参数。
- 运行层通过 `NodeHandler` 执行节点，通过 `BaseToolNode` 把 tool 节点包装成 LLM 可调用工具。
- 首版 workflow 的边只表示依赖关系，不承担 tensor 或业务数据流。
- `adaptive` 才应接入 GNN 预测器；本次实验验证 agent-tool-handler 闭环，不提前引入预测器。

## 样例任务

本次采用“部署证据问答”作为图文理解样例。

用户输入：

```text
样例图片路径：/samples/lab_gpu_demo.jpg
用户文本：这张图能否证明模型已经部署在加速卡上？
```

样例图片来源：

- `/samples/lab_gpu_demo.jpg` 是临时实验中的虚拟占位路径。
- 该路径不对应本仓库中的真实图片文件。
- 图片不是从网络获取的，因此没有外部链接。
- 本次 CV 工具节点没有解码图片文件，只运行了 MobileNetV2 的样例 forward，并返回部署元数据和样例标签。

任务要求：

- LLM 主节点理解用户问题。
- LLM 选择并调用 CV 工具节点和 NLP 工具节点。
- 工具节点在加速卡上执行模型 forward。
- 工具节点返回模型、设备、latency、输出 shape、样例分类标签等元数据。
- LLM 基于工具结果汇总回答。

选择这个任务的原因：

- 它直接贴合当前项目的 workflow + 加速卡资源调度背景。
- 不依赖真实图像文件或外部数据集即可验证 agent 闭环。
- 证据重点是“工具节点真实部署和执行位置”，不是视觉模型本身的业务准确率。

## 模型选择

### CV 模型

使用 `mobilenetv2_100`。

选择依据：

- 来自现有 `config/arch/mobilenet_variants.yaml` 和 `gnn_archs` 支持范围。
- 模型轻量，适合快速临时验证。
- 在 20260607 op-reclass 训练记录中，`mobilenetv2` family 的 row-random test WAPE 为 `0.055901`，属于较稳定模型族。
- 真实 V100 forward 已验证，输出 shape 为 `[1, 1000]`。

本次配置：

| 字段 | 值 |
| --- | --- |
| base model | `mobilenetv2_100` |
| input shape | `[1, 3, 224, 224]` |
| output classes | `1000` |
| batch size | `1` |
| phase | `inference` |

### NLP 模型

使用 `gpt2_balanced_h256_l4_a4_i1024_medium_oc2`。

选择依据：

- 来自现有 `config/arch/gpt2_variants.yaml` 和 `gnn_archs.gpt2_builder` 支持范围。
- 小型 GPT-2 sequence classification 结构能代表文本工具节点。
- `gpt2` 在 20260607 op-reclass 训练记录中 WAPE 较高，为 `0.210201`，说明它是更有挑战的文本族；但本次不接入 GNN，只用它验证真实工具执行。
- 真实 V100 forward 已验证，输出 shape 为 `[1, 2]`。

本次配置：

| 字段 | 值 |
| --- | --- |
| base model | `gpt2` |
| vocab size | `4096` |
| hidden size | `256` |
| layers | `4` |
| heads | `4` |
| inner dim | `1024` |
| input shape | `[1, 64]` |
| output classes | `2` |
| phase | `inference` |

## Workflow 结构

本次在临时脚本内构造如下 DAG：

```text
input
  -> main_llm
      -> scene_classifier
      -> text_classifier
  -> output
```

节点定义：

| 节点 | 角色 | model.task | 作用 |
| --- | --- | --- | --- |
| `input` | `input` | - | 用户请求入口 |
| `main_llm` | `agent` | `react_agent` | 外部 LLM provider 提供的 ReAct 主节点 |
| `scene_classifier` | `tool` | `image_classification` | MobileNetV2 工具节点 |
| `text_classifier` | `tool` | `text_classification` | GPT-2 工具节点 |
| `output` | `output` | - | 逻辑出口 |

边仍只表示依赖关系。实际 LLM tool calling 由 LangGraph ReAct agent 决定，不把边解释为 tensor 数据流。

## 临时实现思路

实验没有修改源码，也没有创建配置文件。实现以 here-doc 临时 Python 脚本完成，核心组件如下。

### AcceleratorModelHandler

临时实现 `NodeHandler`：

- 根据 `node.name` 分派到 CV 或 NLP tool 节点。
- 使用 `gnn_archs.variant_runner.build_variant_model()` 构建模型。
- 使用 `build_example_batch()` 构造项目内一致的样例 batch。
- 使用 `forward_model()` 和 `extract_logits()` 执行 forward 并读取 logits。
- 模型部署到 `cuda:0`。
- 每个模型首次调用时先 warmup 一次。
- 使用 `torch.cuda.Event` 记录单次 forward latency。
- 返回结构化 metadata。

返回字段包括：

| 字段 | 含义 |
| --- | --- |
| `model` | base model 名称 |
| `variant` | 本次临时 variant 名称 |
| `task` | LLM 传入的工具任务 |
| `device` | 实际 CUDA 设备名 |
| `latency_ms` | 单次 forward latency |
| `logits_shape` | logits shape |
| `parameter_count` | 模型参数量 |
| `task_result` | 样例标签、置信度和语义边界说明 |

### ExperimentToolNode

临时继承 `BaseToolNode`：

- 覆盖 `description()`，让 LLM 知道工具节点用途和边界。
- 覆盖 `format_node_output()`，把 `WorkflowNodeResult` 转为 JSON 字符串返回给 LLM。
- 不改变 `ToolNodeInput` 接口，仍使用 `task`、`input_path` 和 `payload`。

### Agent prompt

本次 prompt 明确约束：

- 必须调用 `scene_classifier` 和 `text_classifier`。
- 只能使用工具 metadata 作为“部署到加速卡”的证据。
- 不得声称图片文件被真实解码。
- 使用中文回答。

这个约束很关键。首次探索时，如果没有给出明确图片路径和工具调用要求，LLM 会倾向于要求用户补充输入，而不是主动调用工具。

## 运行环境

| 项目 | 值 |
| --- | --- |
| 日期 | `2026-06-10` |
| CUDA | 可用 |
| GPU | `Tesla V100-SXM2-32GB` |
| 可见 GPU 数 | `3` |
| LLM 配置 | `.env` 中存在 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_NAME` |
| GNN 预测器 | 未接入 |

实验过程中出现 `pynvml` deprecation warning，不影响执行结果。

## 预验证

正式 workflow invoke 前，先在内存中验证真实模型 forward。

| 模型 | 设备 | logits shape | 参数量 |
| --- | --- | ---: | ---: |
| `mobilenetv2_100_ic3_oc1000_no_mutations_probe` | `Tesla V100-SXM2-32GB` | `[1, 1000]` | `3,504,872` |
| `gpt2_balanced_h256_l4_a4_i1024_medium_oc2_probe` | `Tesla V100-SXM2-32GB` | `[1, 2]` | `4,274,176` |

该验证说明 `gnn_archs` builder、batch 构造、forward 路径和 V100 部署均可用。

## Workflow invoke 结果

正式临时实验执行成功，返回状态为 `succeeded`。

实际工具调用顺序：

| 顺序 | node | task | input_path | device |
| ---: | --- | --- | --- | --- |
| 1 | `text_classifier` | `classify` | `null` | `Tesla V100-SXM2-32GB` |
| 2 | `scene_classifier` | `classify` | `/samples/lab_gpu_demo.jpg` | `Tesla V100-SXM2-32GB` |

工具节点结果摘要：

| 工具 | 模型 | 结果类型 | latency | 关键证据 |
| --- | --- | --- | ---: | --- |
| `scene_classifier` | `mobilenetv2_100` | 样例场景标签 | `4.58 ms` | `device=Tesla V100-SXM2-32GB` |
| `text_classifier` | `gpt2` | 样例意图标签 | `6.90 ms` | `device=Tesla V100-SXM2-32GB` |

LLM 最终回答要点：

- 可以证明模型部署到了加速卡。
- 有力证据来自两个工具节点返回的 `device` 元数据。
- 两个工具节点都显示运行设备为 `Tesla V100-SXM2-32GB`。
- latency 字段说明模型确实完成了前向推理。
- 图片文件没有被真实解码，因此不能把图片内容本身作为直接证据。

## 验证断言

临时脚本内置断言：

- `scene_classifier` 必须被调用。
- `text_classifier` 必须被调用。
- 两个工具调用返回的 `device` 都必须包含 `V100`。
- 最终回答必须包含加速卡相关证据。

所有断言均通过。

## 结果汇总

本次实验验证的是 workflow agent 的最小真实闭环：

```text
用户输入
  -> 外部 LLM provider
  -> LangGraph ReAct agent
  -> workflow tool node
  -> 本地 NodeHandler
  -> gnn_archs 构造的 PyTorch 模型
  -> V100 forward
  -> WorkflowNodeResult
  -> LLM 汇总回答
```

已经证明：

- 当前 `src/workflow` 的 tool node 抽象可以承载本地模型执行。
- 外部 LLM 能基于工具描述选择并调用工具。
- 工具节点可以复用 `src/gnn_archs` 的模型构造和 forward 路径。
- CV/NLP 两类模型均可在加速卡上作为工具节点执行。
- 工具返回的部署元数据可以被 LLM 用于最终回答。

没有证明：

- 没有验证真实图像解码或真实 NLP tokenizer/pipeline。
- 没有验证模型业务准确率。
- 没有接入 GNN 预测器。
- 没有验证 `adaptive` 调度。
- 没有验证并发、多 GPU、OOM 或资源装箱策略。

## 与类似工作的区别

HuggingGPT、MM-ReAct、Chameleon、ViperGPT 等工作都验证了 LLM 调用外部工具或视觉专家模型的能力。它们的工具节点通常更接近 API 调用或任务模块编排。

本实验更贴近当前项目的问题：

- 工具节点是本地真实模型。
- 模型在数据中心加速卡上执行。
- 节点执行会产生资源元数据。
- 后续可把 GNN predictor 接入为 `ResourceEstimator`，用于排序、风险标记或调度策略。

因此，本实验不是为了证明“LLM 能理解图片”，而是为了证明“LLM 可以编排部署在加速卡上的模型工具节点，并消费节点执行结果”。

## 后续建议

短期可做：

- 把临时 `AcceleratorModelHandler` 收敛为正式的 `GnnArchsNodeHandler`。
- 为 image/text tool 节点增加真实输入适配：图片读取、resize、tokenizer。
- 将本次 here-doc 脚本固化为可复现实验脚本或 pytest 集成测试。
- 记录 `WorkflowNodeResult.metadata` 的最小稳定字段。

中期可做：

- 加入 GNN predictor，但先只作为 metadata 记录，不影响调度。
- 对比 LLM 是否会根据资源摘要选择工具。
- 把 `adaptive` 限定为 ready queue 排序，不做显存装箱。

暂不建议：

- 直接引入 YOLO/Qwen3 作为首个正式实验组合。
- 在 workflow schema 中加入运行后指标字段。
- 在首版 adaptive 中实现多 GPU 装箱、retry 或复杂 OOM 策略。

## 产物与清理

本次用户要求保留本文档，因此新增持久产物：

```text
docs/workflow/experiment_20260610.md
```

临时实验脚本没有保存到仓库。除本文档外，不应留下其他实验文件。
