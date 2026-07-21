# Workflow 文档索引

本文用于区分现行实现、冻结实验和历史方案。日期化文档中的“当前”只代表文档生成时的
代码状态，不自动代表当前仓库。

## 现行实现

- [当前实现说明](implementation.md)：`src/workflow` 的 schema、并发、调度、模型副本和
  资源契约边界。继续扩展系统时先读此文档。
- [多 workflow 调度器与资源契约重构方案](system_plan_20260720.md)：`WorkflowFleet`、
  跨 workflow 公平性和资源契约（`ResourceContract`）设计决策的定稿记录，是
  `implementation.md` 里"多 workflow 编排"和"资源契约与调度策略"两节的背景依据。
- [Phase 1 系统目标](../dev/workflow_system_phase1.md)：需求、研究边界和首阶段约束。
- [系统实验方案](system_experiment_plan_20260713.md)：已冻结并完成的正式实验设计。
- [系统实验结果](system_experiment_results_20260713.md)：75 次正式 trial 的冻结结果。
- [vLLM V100 验证](vllm_v100_validation.md)：当前 serving 版本和 V100 实机门禁记录。

推荐阅读顺序是当前实现说明、Phase 1 系统目标、系统实验方案和系统实验结果。

## 历史研究与实验

以下文档保存研究演进和实验依据，不作为当前 runtime 契约：

- [早期论文调研](survey_20260619.md)
- [早期研究路线](research_plan_20260626.md)
- [2026-06-26 动机实验](experiment_20260626.md)
- [2026-06-29 runtime 实验规划](experiment_20260629.md)
- [2026-07-03 动机实验方案](motivation_20260703.md)
- [2026-07-04 动机实验结果](motivation_20260704.md)
- [2026-07-05 MBPP 动机实验改造方案](motivation_plan_20260705.md)

这些文档中的动态 token budget、在线 GNN、旧 trace schema 和旧实验矩阵均为历史研究内容。
是否继续实现其中的未完成方向，应以新的扩展方案为准。

## 历史实现方案

- [2026-07-02 运行时骨架方案](archive/claude_plan_20260702.md)
- [2026-07-08 调度系统方案](archive/codex_plan_20260708.md)

两份方案保留设计演进记录，但包含已经被替换的静态 eager-load、单任务 worker、Hugging
Face backend、NVML 轮询、ONNX、独立 deployment profile 和动态 token budget 设计。

派生执行清单 `codex_execution_plan_20260710.md` 和旧 schema 说明
`schema_runtime_20260626.md` 已从当前文档树移除；需要追溯时使用 Git 历史。

## 其他文档

[论文草稿](../draft/workflow.tex) 同时包含研究目标和论文算法，不等同于当前实现说明。
论文中的目标架构在落地前不得反向解释为 runtime 已有能力。

## 信息优先级

发生冲突时按以下顺序判断：

1. `src/workflow`、`src/experiment/workflow` 及对应测试。
2. 当前实现说明和 Phase 1 系统目标。
3. 冻结实验方案、结果和环境验证记录。
4. 日期化研究文档和历史实现方案。
