# Workflow 研究路线（20260626）

本文记录 20260626 时点的 workflow 方向和下一步。更早的调研、路线图和步骤文档只作历史参考。

## 当前判断

当前不把重点放在证明“小模型 workflow 优于大模型”，而是在质量约束下判断：

```text
什么时候用 direct
什么时候用 workflow
怎样调度 workflow 的本地工具节点
```

GNN 预测器当前只参与资源预测。任务质量仍交给公开 benchmark 和 evaluator 判断。

## 数据边界

公开任务数据集用于质量评估，不用于直接训练 GNN。

当前可优先观察：

- GSM8K：数学题，numeric exact match。
- MBPP：代码题，pass@1。
- QMSum / Multi-News：摘要任务，可用于观察分块 workflow。

不建议一开始把 MATH、BigCodeBench、VideoMME、真实多租户 trace 一起混进来。变量太多，定位困难。

## Workflow 边界

主 agent 可以是外部 provider，也可以是本地模型实验节点。GNN 调度当前主要看本地 tool 节点。

应分清三类数据：

| 数据 | 作用 |
| --- | --- |
| 任务 benchmark | 判断答案质量 |
| workflow YAML | 描述 DAG、模型和运行参数 |
| resource profile | 训练或评估资源预测与调度 |

一般不把 benchmark label、gold answer 或真实运行结果写进 workflow YAML。

## 调度方向

当前优先做静态 workflow，再考虑动态 ReAct。

静态 workflow 更适合当前阶段：

- DAG 固定。
- 实验可复现。
- direct / weak / strong workflow 可以直接对比。
- 调度目标容易定义为 makespan、显存、OOM、缓存和能耗代理。

动态 ReAct 更有系统价值，但需要模拟或接入真实 tool call 决策。GNN 在其中更适合回答
“如果调用这个工具，成本是多少”，不适合回答“模型会不会调用这个工具”。

## 最小下一步

当前可以先做：

- 固定一组小样本，先做 direct / weak / strong workflow 对照。
- 记录机器可读结果，避免只写 markdown。
- 分开报告质量、推理时延和 judge 成本。
- 在质量达标的候选里再做调度。

暂缓：

- 复杂 MILP 或强化学习调度。
- 大规模 runtime registry。
- 多轮自适应 repair。
- 把所有数据集一次性铺开。
