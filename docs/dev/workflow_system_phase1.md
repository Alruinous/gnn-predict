# 面向多智能体 workflow 的调度系统

实现一个面向多智能体 workflow 的调度系统。系统面向普遍的 workflow 图，不要把它实现成某个数据集或某个固定 workflow 的专用流水线。

现行代码边界见 [`docs/workflow/implementation.md`](../workflow/implementation.md)。本文同时保留研究目标；带有“希望”或“后续实现”的内容不表示 runtime 已经具备相应能力。

## 背景

在现有 LangGraph 框架中，因为其图执行器无法实现细粒度的并发控制，导致 workflow 在处理一个请求时，无法基于图上流水线技术，进行批量请求处理，带来模型空闲、空闲资源占用等情况。同时也无法方便地做流水线、背压、资源调度和模型生命周期管理。

## 创新点


本项目打算完全重新实现一个 workflow 调度系统，把 Agent Workflow 从粗粒度的步骤执行，推进到预测驱动的数据流执行。

传统框架在 workflow 运行时通常只能在全图维度做调度，如 LangGraph 中，一批请求进入 DAG 后，必须等待这批请求完全结束，才能发送下一批请求。DAG 中缺少更细粒度的并行。比如在静态 workflow 的一种长上下文总结任务中， workflow 内一般将文章切片后，交给有多个 agent 负责总结切片的内容，切片内容汇总交给下游 agent 负责整合输出。切片会有不同长度和复杂度，导致负责处理切片内容的 agent 之间会有完成推理任务的快慢，造成 bubble 空闲。同时 LangGraph 中，上游节点在处理完其推理任务后，也不能立刻开始处理后续的请求，必须等待下游节点完全将这批请求处理完，才能继续处理下一批请求。也就是说，LangGraph 的调度器只能在节点级别做调度，而无法在数据流级别做调度。

本系统希望，提供更细粒度的调度能力，结合预测器的信息，感知 workflow 负载状态和数据流情况，以支持流水线、模型驱逐、模型预取等功能。

第一个创新点是面向 Agent Workflow 的细粒度流水线执行。对于文档分块、批量工具调用和动态任务生成等场景，上游节点可能持续产生多个数据项，下游节点的处理速度也可能显著不同。 节点 通过 session 聚合、有界缓冲和背压机制，把原本受 fan-in barrier 限制的执行过程转化为可流水线推进的数据流过程。后续实现应重点验证这种机制能否降低节点空闲时间、提高吞吐，并保持 workflow 语义依赖不被破坏。

第二个创新点是 workflow-aware 的模型生命周期管理。系统借助已有预测器提供的离线资源评估缓存，估计固定请求在一种模型部署方式下的显存占用、部署成本、推理耗时和功耗。系统据此选择加速卡、安排模型预取并决定驱逐对象，减少模型驻留和切换开销，优化有限资源场景下的资源利用效率。预测结果只影响调度，不改变 workflow 声明的模型、prompt、图结构、路由、采样参数或输出上限。

运行时只读取一份统一资源预测缓存。缓存以 `WorkflowModelFeatureKey` 为索引，每个 input/output length 与 GPU bucket 的条目同时保存模型加载时间、推理时间、峰值显存、功耗和来源元数据。当前阶段使用外部提供的冻结 synthetic cache，只验证 workflow 调度机制；不接入 `src/gnn_model`，不训练或评估 GNN，也不维护独立的部署 profile。

## Workflow 语义与系统职责边界

Workflow 配置负责声明模型、prompt、图结构、路由、采样参数和固定的 `max_new_tokens`。系统只负责排队与背压、并发准入、等价加速卡上的放置、模型加载与复用、预取和驱逐。

暂时没有可用资源时，请求保持 pending；任何部署都无法满足固定请求时，系统显式失败。系统不得静默截断输入、降低输出上限、替换模型或改变采样语义。`hit_token_limit` 只作为运行结果记录，不触发自动调整。

## 核心功能

- 通用 workflow 图执行器（必须）
- 跨 session 和 session 内的流水线并行（必须）
- 有界缓冲和背压（不必须，只是辅助）
- session 级 fan-in（必须）
- 预测器缓存（我自行准备）
- 固定生成配置的可行性检查（必须）
- 模型加载与复用（必须）
- ETA 模型预取（必须）
- workflow-aware 驱逐（必须）
- 功耗优化（不用）

## vLLM serving 与批量语义

Agent 模型副本使用 vLLM `AsyncLLMEngine` 接收并发请求，由引擎执行 continuous batching。主运行环境负责 chat template、分词和最终解码，副本只接收与返回 token IDs，避免主环境 Transformers 5.x 与 vLLM 隔离环境 Transformers 4.x 产生模板或重分词差异。

每个 agent 的 `ExecutionConfig.max_new_tokens` 是 workflow 业务语义。worker 和 serving backend 必须原样使用该值，调度器不得根据资源状态扩大或缩小它。

每个副本的 `max_model_len`、`max_num_seqs`、`max_num_batched_tokens` 和 `gpu_memory_utilization` 属于部署身份。调度器可以在副本处于 `BUSY` 时继续授予请求，直到达到 `max_num_seqs`；每个请求拥有独立 lease，可以乱序完成。驱逐仅允许在 lease 全部释放后发生，`SUSPECT` 副本停止接收新请求并先排空现有请求。

synthetic cache 为 batch `k` 提供精确测试条目，用于验证 batching 准入和调度路径。候选请求加入已有 `k` 个请求时，调度器查询 batch `k+1` 的精确缓存条目，并采用覆盖实际固定请求的最小 input/output bucket。bucket 只用于调度，不改变实际输入或输出上限。联合包络不可行但 batch 1 可行时，请求保持 pending；batch 1 缺少覆盖条目或不可行时显式失败。运行时不外推缺失 batch。

为控制 serving 变量，首阶段固定 vLLM 0.10.2 V0、XFormers、FP16、eager mode、TP/PP 1，并关闭 prefix caching、chunked prefill、speculative decoding、量化、CPU offload 和 swap。V100 实机结果见 `docs/workflow/vllm_v100_validation.md`。
