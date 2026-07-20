# Workflow Runtime 当前实现

本文记录 `src/workflow` 的现行实现边界。代码、schema 或测试发生变化时，应同步更新本文；
历史方案中的“当前实现”不再作为依据。

## Workflow 契约

- Workflow 是非空静态 DAG，必须只有一个入口节点和一个终端节点。
- 当前节点类型只有 `agent` 和 `function`。
- 每个静态节点在一个 session 中最多执行一次；同一 source 对同一 target 最多贡献一个 item。
- fan-in 按 `session_id` 和静态依赖集合聚合，等待每个依赖 source 的 item 到齐。
- 每个节点拥有一个有界输入队列。所有入边共享目标节点队列，队列满时上游产生背压。
- function 节点支持 broadcast 和 targeted routing；终端 function 必须使用 broadcast。

当前实现不支持运行时新增节点、任意环、同一 source 连续发出不定数量 item、流结束标记或
分布式 exactly-once。

## 组件职责

- `WorkflowController` 校验配置，创建 Ray 队列、存储 actor、调度器和 worker，并提供
  `start`、`submit`、状态、结果、drain 和立即停止接口。
- `NodeWorker` 负责取队列、session fan-in、prompt 或函数调用、输出路由和背压传播。
- `SchedulerCore` 保存 session、task、acquire、模型副本和加速卡账本，并产生加载、授予、
  预取和驱逐决策。
- `SchedulerActor` 以单一 mailbox 串行修改调度状态，并异步观察副本生命周期操作。
- `ModelReplica` 封装一个单 GPU vLLM engine，处理加载、并发生成、取消和关闭。
- `ResultStoreActor` 和 `TraceWriterActor` 分别单写终端结果与 trace。

数据通过 Ray Queue 流动；session、acquire、模型副本和停止操作通过 actor 控制面流动。调度
决策不编码为队列数据项。

## Worker 并发与 vLLM batching

`NodeWorker` 不是单任务 worker。worker 内部维护多个异步 session task：

- agent 节点并发上限为 `execution.serving.max_num_seqs`；
- function 节点并发上限为 `max_concurrency`；
- 同一 worker 内不同 session 可以并发执行，同一 session 不会在同一节点重复执行。

Agent 副本使用 vLLM `AsyncLLMEngine` continuous batching。副本处于 `BUSY` 时仍可授予新
请求，直到达到 `max_num_seqs`。每个请求拥有独立 lease，允许乱序完成；只有所有 lease
释放后，副本才能被驱逐。

调度器按副本当前并发数查询精确 batch `k` 缓存条目。缺少 batch 1 条目表示固定请求不可
调度；缺少更高 batch 条目只表示当前联合 batch 不可准入，不会外推预测值。

当前 serving 固定为 vLLM 0.10.2、V0、XFormers、FP16、eager、TP/PP 1，并关闭 prefix
caching、chunked prefill、speculative decoding、量化、CPU offload 和 swap。V100 实机
边界见 [vLLM V100 验证](vllm_v100_validation.md)。

## 固定生成语义

Agent 节点使用 `ExecutionConfig.max_new_tokens` 声明固定输出上限。调度器不得根据资源状态
放大或缩小它，也不存在 `TokenBudgetConfig`、granted token range 或自动 upscale/downscale。

输入长度、输出上限和 prediction bucket 用于可行性检查，但 bucket 不改变实际请求。请求在
所有候选部署上都不可行时显式失败；暂时没有空闲资源时保持 pending。

## 模型副本与加速卡

部署身份由 backend 版本、模型名、模型路径、dtype 和完整 `ServingConfig` 共同决定。只有
部署身份完全一致的节点才能共享副本。

- 同一 `(model_key, gpu_kind)` 最多有一个 live 或 loading 副本。
- 当前每个副本绑定一张 GPU，每张 GPU 同时只归属一个模型副本。
- `gpu_kind` 是非空字符串，不限于 V100/A100 枚举。
- 副本状态为 `LOADING`、`IDLE`、`BUSY`、`EVICTING` 或 `SUSPECT`。
- OOM 或 engine failure 会使副本进入 `SUSPECT`；它停止接收新请求，排空后优先清理。

当前资源账本不在一张 GPU 上打包多个模型，也不通过空闲显存扫描推导新的部署容量。

## 预测缓存与调度策略

运行时读取一份统一 `PredictionCache`。每个 decode key 包含模型、GPU kind、batch、输入
长度和输出长度，value 同时保存：

- `predicted_load_sec`
- `predicted_run_sec`
- `predicted_peak_vram_mb`
- 可选 `predicted_power_watts`
- predictor metadata

当前没有独立 `DeploymentProfile`。所有 agent 策略都需要 prediction cache 完成请求可行性、
GPU 选择、显存门禁和 batch 准入；策略差异如下：

- `fifo`：按 acquire 到达顺序处理，不生成 near-ready prefetch，不用缓存 ETA 维护运行完成
  时间或 reload cost。
- `history`：使用当前运行中观察到的执行和加载历史估计 ETA、预取与 reload cost；没有加载
  历史时不做 near-ready prefetch。缓存仍用于固定请求可行性和 batch 准入。
- `cache`：使用冻结缓存中的执行和加载耗时维护 ETA、安排预取，并结合 workflow 未来复用
  距离选择驱逐对象。

预测缓存是只读调度输入。运行时不在线执行 GNN、不更新预测器，也不使用本轮结果校正
`cache` 策略的冻结预测值。

## 调度循环与 GPU 观测

`SchedulerActor` 通过异步 mailbox 接收命令，批量串行应用状态变化。没有命令时，它等待至
最近的 prefetch deadline 或 `max_tick_interval_sec`，而不是固定 `Sleep(Δt)` 后扫描全部状态。

核心调度器不轮询 NVML。GPU 放置依据静态 `AcceleratorConfig`、prediction cache 和调度器
维护的副本所有权账本；模型副本加载后会报告实际加载耗时、空闲显存和 vLLM KV capacity。
正式实验的 GPU 功率、显存和利用率采样属于外部 telemetry sidecar，不参与核心调度决策。

## 离线图与缓存链路

Workflow 图特征链路使用 PyTorch Export/FX：

1. `scripts/workflow/convert.py` 展开 `cache.yaml` 中的模型和运行 bucket。
2. `src/workflow/model.py` 捕获 `torch.export.ExportedProgram`。
3. FX 图转换为 PyG 图特征并写入图缓存。
4. 离线预测或 profiling 结果最终转换为 scheduler-facing `PredictionCache`。

当前 workflow 链路不生成或读取 ONNX 文件。`scripts/workflow/profile_workflow_cache.py` 可以
直接实测加载、运行、显存和功耗并生成统一 PredictionCache；正式 Phase 1 实验使用冻结的
synthetic cache，没有接入 `src/gnn_model` 在线推理。

## 结果与 trace

运行时分别持久化终端 session 结果和统一 trace。trace 使用连续 `event_seq`，覆盖队列、
fan-in、task、acquire、placement、batch admission、模型生命周期、prefetch、inference、失败
和 run shutdown。动态 token budget 事件不属于当前 trace 契约。

## 当前扩展边界

下一阶段扩展应显式选择新的系统契约，不从历史方案恢复旧接口。当前可复用边界包括：

- 在 `schema.py` 扩展新的静态节点配置；
- 在 `policy.py` 增加纯调度决策；
- 在 `SchedulerCore` 扩展 session、资源或生命周期状态；
- 在 `ModelReplica`/backend 边界扩展 serving 能力；
- 在 `src/experiment/workflow` 增加新的实验场景和验证。

在线 GNN、动态 DAG、多模型 GPU packing、功耗优化和新的 token 控制机制均不属于现行实现。
