# Workflow Schema 与运行边界（20260626）

本文记录 20260626 时点的 workflow 模块边界。旧日期文档里的 schema 示例只作历史参考。

## Schema

Workflow YAML 当前描述一张 DAG：

- `nodes` 是节点列表。
- `edges` 是依赖边列表。
- 节点名在一个 workflow 内唯一。
- 边只表达依赖关系，不承载 tensor 或业务数据。

节点当前常见字段：

- `name`：节点名。
- `type`：节点类型。
- `task`：任务名，可为空。
- `description`：给工具或调度理解用的短描述。
- `prompt_template`：可选 prompt 模板。
- `model`：模型名和参数。
- `runtime`：运行前可知的输入形状、batch、phase。
- `execution`：本地执行配置，例如模型路径、设备和解码参数。

`model` 当前倾向采用：

```yaml
model:
  name: qwen3
  parameters: {}
```

新配置倾向沿用上面的嵌套结构，旧文档里的 `model.task` 和平铺模型字段只作历史参考。

## Runtime

`runtime` 当前只保存运行前可知的信息：

- `batch_size`
- `phase`
- `input_shape`
- `sequence_length`
- `decode_max_output_length`

一般不建议把真实运行结果、监控统计、ONNX 图特征或输出路径写进 workflow YAML。
这些信息更适合放在 profile、结果记录或训练数据里。

## Execution

`execution` 面向可执行 workflow。

常见字段：

- `model_path`
- `devices`
- `dtype`
- `max_new_tokens`
- `do_sample`
- `temperature`
- `use_chat_template`
- `enable_thinking`

设备通常使用 `cuda:N` 形式。具体模型路径容易随机器变化，这里不记录固定路径。

## 运行边界

20260626 时点，workflow 运行层主要服务两类需求：

- 固定 DAG 的本地实验。
- ReAct 工具节点包装。

`gnn_archs` 仍主要承担模型变体生成、ONNX、训练、推理和监控采集链路。当前不建议把
`variant_runner.run_variant()` 直接当作 workflow DAG 节点执行器。

## GNN 的位置

GNN 预测器当前主要服务本地 tool 节点的资源估计和调度辅助。

一般不把 GNN 预测用于：

- 主 agent 的语义质量。
- 外部 provider 延迟。
- benchmark 答案正确率。
- 工具是否会被调用。

如果 agent 已经决定调用某个工具，GNN 可以预测这个工具在候选设备上的成本和风险。
