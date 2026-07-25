# Workflow Runtime 当前实现

本文记录 `src/workflow` 的现行实现边界。代码、schema 或测试发生变化时，应同步更新本文；
历史方案中的"当前实现"不再作为依据。

## Workflow 契约

- 每个 `Workflow` 有一个全局唯一的 `workflow_name`，是非空静态 DAG，必须只有一个入口节点
  和一个终端节点。
- 当前节点类型只有 `agent` 和 `function`。
- 每个静态节点在一个 session 中最多执行一次；同一 source 对同一 target 最多贡献一个 item。
- fan-in 按 `session_id` 和静态依赖集合聚合，等待每个依赖 source 的 item 到齐。
- 每个节点拥有一个有界输入队列。所有入边共享目标节点队列，队列满时上游产生背压。
- function 节点支持 broadcast 和 targeted routing；终端 function 必须使用 broadcast。

当前实现不支持运行时新增节点、任意环、同一 source 连续发出不定数量 item、流结束标记或
分布式 exactly-once。

## 组件职责

- `WorkflowFleet`（`fleet.py`）拥有 Ray 生命周期、GPU 加速卡池、共享 `_SchedulerActor`、
  共享 `TraceWriterActor`，并提供 `register_workflow`/`submit`/`drain_workflow`/`shutdown`
  等 fleet 级接口；一个 fleet 可以同时注册多个 workflow，共享同一个调度器和 GPU 池。
- `WorkflowController`（`controller.py`）是单 workflow 场景下的薄 facade：内部持有一个只
  注册了自己这一个 workflow 的私有 `WorkflowFleet`，`start`/`submit`/`get_session_state`/
  `has_result`/`get_result`/`drain_and_stop`/`stop_now` 均逐方法委托给它。
- `NodeWorker` 负责取队列、session fan-in、prompt 或函数调用、输出路由和背压传播。
- `SchedulerCore` 保存 session、task、acquire、模型副本和加速卡账本，并产生加载、授予、
  预取和驱逐决策；`register_workflow`/`deregister_workflow` 支持运行时增减已注册的
  workflow。
- `SchedulerActor` 以单一 mailbox 串行修改调度状态，并异步观察副本生命周期操作。
- `ModelReplica` 封装一个单 GPU vLLM engine，处理加载、并发生成、取消和关闭；与
  `baseline/vllm_engine.py::StaticVLLMEngine` 共享 `GenerationEngine` 基类。
- `ResultStoreActor` 和 `TraceWriterActor` 分别单写终端结果与 trace。

数据通过 Ray Queue 流动；session、acquire、模型副本和停止操作通过 actor 控制面流动。调度
决策不编码为队列数据项。

## 多 workflow 编排

一个 `WorkflowFleet` 可以注册任意多个 `Workflow`，共享同一个 `_SchedulerActor` 和同一个
`AcceleratorConfig` 池：

- **全局共享状态**：`accelerators`、`replicas`、`_replica_pairs`（`(model_key, gpu_kind)`
  → replica_id）、`oom_penalties`、`pending_acquires`、`grants`、`tasks`、`sessions`、
  `load_history` 均不按 workflow 拆分。`ModelDeploymentConfig.model_key` 本身与 workflow
  身份无关（只依赖 backend 版本、模型名、模型路径、dtype、完整 `ServingConfig`），因此两个
  workflow 请求字节级相同的部署配置时会自动共享同一个副本。`session_id`/`task_id`/
  `acquire_id` 是 fleet 范围内的全局唯一 UUID，不是每个 workflow 各自独立的命名空间。
- **按 workflow 拆分的状态**：`SchedulerCore._workflows`/`_nodes`/`_node_order` 按
  `workflow_name` 分桶；`duration_history` 的键包含 `workflow_name`（避免不同 workflow
  即使节点重名、部署配置相同，prompt/输出长度分布不同导致 EMA 估计互相污染）。
  `SessionRecord.workflow_name` 是判断"这个 session 属于哪个 workflow"的唯一锚点。
- **Trace 与结果的作用域不对称**：trace 是 fleet 级别的单一文件（`TraceEvent.workflow_name`
  用于消歧），结果存储按 workflow 各自一个 `ResultStoreActor`/`output_dir`（避免跨
  workflow 的 `session_id` 落在同一份结果文件里造成误判）。所有已注册 workflow 共享 fleet
  的 `run_id`。
- **准入公平性**：每个已注册 workflow 有一个 `priority_weight`（默认 1.0，
  `WorkflowFleet.register_workflow(..., priority_weight=...)` 传入）。`_ready_decisions`
  先按 `sessions[task.session_id].workflow_name` 分组，组内沿用现有策略排序规则，组间按
  权重加权轮转交织（`_interleave_by_workflow`）；权重相等时退化为按 workflow 轮转。
- **驱逐保护**：`_future_reuse_distance` 对每个贡献需求的 workflow 来源单独加权
  （`distance / priority_weight`），再取所有来源的最小值；高权重 workflow 的需求会拉低
  有效距离，从而保护对应副本不被驱逐。
- **drain/停止语义拆成两条独立路径**：`WorkflowFleet.drain_workflow(name)` 只等这个
  workflow 的 session 排空、停这个 workflow 自己的 worker、`deregister_workflow`、关闭
  这个 workflow 自己的 result store——不触碰 `evict_all`、不触碰共享调度循环。
  `WorkflowFleet.shutdown()` 才会把仍注册的 workflow 逐个 drain 完、`evict_all`、等待
  `replicas_empty`、`stop_loop`、关闭 Ray。已 drain 的 workflow 状态移入
  `_drained_bindings`（不删除），因此 drain 后仍可查询 `get_session_state`/`get_result`；
  `register_workflow` 的重复注册检查只看仍存活的 `_bindings`，允许同名 workflow 在 drain
  后重新注册。

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
部署身份完全一致的节点才能共享副本——这个身份与 workflow 身份无关，因此跨 workflow 的
模型复用是共享池架构的自然结果，不需要额外代码。

- 同一 `(model_key, gpu_kind)` 最多有一个 live 或 loading 副本，且这个约束是 fleet 全局的
  （不是每个 workflow 各自独立的）。
- 当前每个副本绑定一张 GPU，每张 GPU 同时只归属一个模型副本。
- `gpu_kind` 是非空字符串，不限于 V100/A100 枚举；`SchedulerConfig.accelerators` 允许在
  同一个 fleet 里混合多种 `gpu_kind`（一个 Ray hostname 只能暴露一种 `gpu_kind`，见
  `SchedulerConfig.validate_accelerators`），这是单节点异构 GPU 池的落地方式，参考
  `config/workflow/multi_workflow_demo/`。
- 副本状态为 `LOADING`、`IDLE`、`BUSY`、`EVICTING` 或 `SUSPECT`。
- OOM 或 engine failure 会使副本进入 `SUSPECT`；它停止接收新请求，排空后优先清理。

当前资源账本不在一张 GPU 上打包多个模型，也不通过空闲显存扫描推导新的部署容量。

## 资源契约与调度策略

运行时读取一份统一 `ResourceContractCache`（`artifacts.py`）。每个 decode key 包含模型、
GPU kind、batch、输入长度和输出长度，value（`ResourceContract`）同时保存：

- `predicted_load_sec` / `predicted_run_sec` / `predicted_peak_vram_mb`：点估计 ŷ，只用于
  软决策（placement 排序 tiebreak、预取 ETA）。
- `peak_vram_mb_upper_bound`（必填）+ `peak_vram_mb_evidence`（必填 `ResourceEvidence`：
  `method` + `sample_count` + 可选 `margin_fraction`）：校准过的显存上界 U，是唯一的硬性
  OOM 安全闸门——`policy.py::_feasible_placement` 和 `scheduler.py::_admission_prediction`
  的可行性判断都读 `peak_vram_mb_upper_bound`，不读 `predicted_peak_vram_mb`。
- `run_sec_upper_bound` / `run_sec_evidence`：可选，预留给未来 SLA/deadline 策略，当前
  没有消费者。
- `source`（`ResourceContractSource`：`empirical_profile` / `synthetic_fixture` /
  `gnn_predicted`）：数据来源标注，只影响 trace/可观测性，不参与调度决策。
- 可选 `predicted_power_watts`（不参与调度决策）、predictor metadata。

这条"资源知识精细度匹配决策风险"的原则只落在显存这一个硬性维度上：`predicted_run_sec`
的两个消费点（bin-packing tiebreak、预取 ETA）都是软决策，不强制要求置信区间。

当前没有独立 `DeploymentProfile`。所有 agent 策略都需要 resource contract cache 完成请求
可行性、GPU 选择、显存门禁和 batch 准入；策略差异如下：

- `fifo`：按 acquire 到达顺序处理，不生成 near-ready prefetch，不用缓存 ETA 维护运行完成
  时间或 reload cost。
- `history`：使用当前运行中观察到的执行和加载历史估计 ETA、预取与 reload cost；没有加载
  历史时不做 near-ready prefetch。缓存仍用于固定请求可行性和 batch 准入。
- `cache`：使用冻结缓存中的执行和加载耗时维护 ETA、安排预取，并结合 workflow 未来复用
  距离选择驱逐对象。

资源契约缓存是只读调度输入。运行时不在线执行 GNN、不更新预测器，也不使用本轮结果校正
`cache` 策略的冻结预测值——`gnn_predicted` 这个 `source` 枚举值当前也不产出，只保留接口。

`scripts/workflow/compose_profile_cache.py`（`ProfileSource`/`parse_profile_source`/
`compose_prediction_cache`/`write_prediction_cache`）把多个单一 GPU kind 的
`ResourceContractCache` 合并成一份覆盖异构池的缓存：每个来源只保留声明的 `gpu_kind` 对应
条目（丢弃混入的其他型号数据），来源间 `gpu_kind` 重复或 `version` 不一致时 fail fast。

## 调度循环与 GPU 观测

`SchedulerActor` 通过异步 mailbox 接收命令，批量串行应用状态变化。没有命令时，它等待至
最近的 prefetch deadline 或 `max_tick_interval_sec`，而不是固定 `Sleep(Δt)` 后扫描全部状态。

核心调度器不轮询 NVML。GPU 放置依据静态 `AcceleratorConfig`、resource contract cache 和
调度器维护的副本所有权账本；模型副本加载后会报告实际加载耗时、空闲显存和 vLLM KV
capacity。正式实验的 GPU 功率、显存和利用率采样属于外部 telemetry sidecar，不参与核心
调度决策。

## 离线图与缓存链路

Workflow 图特征链路使用 PyTorch Export/FX：

1. `scripts/workflow/convert.py` 展开 `cache.yaml` 中的模型和运行 bucket。
2. `src/gnn_model/data/causal_lm_graph.py` 捕获 `torch.export.ExportedProgram`（与通用的
   `src/gnn_model/data/fx_graph.py::build_graph_data_from_fx` 是姊妹关系：后者是通用
   PT2/FX→PyG 转换，前者是 causal LM 专属的捕获编排）。
3. FX 图转换为 PyG 图特征并写入图缓存。
4. 离线预测或 profiling 结果最终转换为 scheduler-facing `ResourceContractCache`
   （`scripts/workflow/profile_workflow_cache.py` 对实测显存按
   `peak_vram_mb_upper_bound = predicted_peak_vram_mb * 1.10`、
   `evidence.method="fixed_margin_fallback"` 填充 U；`src/experiment/workflow/cache.py`
   的 synthetic fixture 用 `evidence.method="point_estimate_only"`、零余量填充 U，如实
   标注两者都不是真正校准过的置信区间）。

当前 workflow 链路不生成或读取 ONNX 文件。`scripts/workflow/profile_workflow_cache.py` 可以
直接实测加载、运行、显存和功耗并生成统一 resource contract cache；正式 Phase 1 实验使用
冻结的 synthetic cache，没有接入 `src/gnn_model` 在线推理。

## 结果与 trace

运行时分别持久化终端 session 结果和统一 trace。trace 使用连续 `event_seq`，覆盖队列、
fan-in、task、acquire、placement、batch admission、模型生命周期、prefetch、inference、失败
和 run shutdown。动态 token budget 事件不属于当前 trace 契约。多 workflow 场景下
`TraceEvent.workflow_name` 用于消歧，但 `event_seq` 仍是 fleet（单一 `_SchedulerActor`）
范围内连续的。

## master 驱动与实验部署

`WorkflowFleet`/`submit` 之上的**实验驱动 master**（`src/workflow/master.py`，`python -m
workflow.master`）把库级 API 封装成可在 Crater DDP 作业里跑实验的主任务：attach 已有 Ray Head
→ 注册 `--workflow-files` 的 workflow → 运行 `--experiment` 指定的实验 → 写完 trace →
`shutdown` 退出。核心 CLI 不依赖任何数据集/实验层，二者接缝是**点路径解析**（`module:attr`）。

关键参数：`--workflow-files a.yaml,b.yaml`、`--functions module:attr`（→ name→callable 注册
表）、`--experiment module:attr`（→ 实验可调用）、`--experiment-config`、`--scheduler-config`、
`--vllm-python`、`--predictions`、`--gpu-mem kind=mb,...`、`--min-gpus`、`--output-dir`、
`--run-id`、`--ray-address`。

- **Ray attach**：master 在 `fleet.start()` 之前自行 `ray.init(address=...)`，因此
  `WorkflowFleet._owns_ray` 保持 `False`，`shutdown()` 绝不 `ray.shutdown()`（不拆 worker 集群）。
- **加速卡自动发现**：`--min-gpus>0` 时先等 worker 加入，再读 `ray.nodes()` 的
  `NodeManagerHostname`/`GPU`/`accelerator_type:X` 构造 `SchedulerConfig.accelerators`；
  `gpu_kind` 由 `accelerator_type` 或 `--host-gpu-kind` 决定、显存由 `--gpu-mem` 映射，免手写
  Volcano pod DNS。`scheduler.yaml` 里 accelerators 留空、`vllm_python_executable` 由
  `--vllm-python` 覆盖，部署无需改动已提交配置。
- **实验插件约定**：实验 = `run(fleet, *, output_dir, run_id, config) -> None`，只驱动 session
  （submit/poll/get_result），生命周期归 master。新增实验 = 新增一个导出 `run` 的模块，用
  `--experiment new.module:run` 引用，核心零改动。已提供 `experiments/static_inputs.py`（config
  显式 session，dev-sim/冒烟用）与 `experiments/dataset_replay.py`（QMSum/MBPP 数据集回放）。

QMSum/MBPP 接线（`src/experiment/workflow/`）：`scenario_functions.build_registry` 并集
`qmsum_functions`+`mbpp_functions`；`scripts/workflow/export_scenario_workflows.py` 由 builder
再生 `config/workflow/serve/{qmsum,mbpp}.yaml`；`scripts/workflow/write_serve_predictions.py`
写覆盖 Qwen3-4B/8B/14B × v100+a100 的合成 `ResourceContractCache`。

trace 目录：fleet 目录 = `<output-dir>/<run-id>`（须每次全新，trace/summary/results 都 `"x"`
独占）；产物 `workflow_trace.jsonl`（运行中持续追加）、`run_summary.json`（shutdown 写）、每
workflow `<name>/session_results.jsonl`，`analysis.summarize_trace` 直接消费。**policy 对比** =
改 `scheduler.yaml` 的 `policy`（fifo|history|cache）+ 换 `--run-id` 重投作业，离线逐个比对
trace（一个 fleet 只有一个共享调度器、只跑一种 policy）。

部署脚本分两条路径。**旧一次性流**：`scripts/workflow/serve_master.sh`（ddp.md master 模板 +
`ray start --head` + `python -m workflow.master`，起 Head 与跑实验一体，作业平台一次性作业）。
**常驻集群流**（Head/worker 跨实验复用，见 `experiment_runbook_resident_20260725.md`）：开发容器
`serve_head.sh`（每 `RAY_PORT` 一个常驻 Head，`--num-gpus=0`、关 dashboard、`temp-dir=/tmp/ray_<port>`
且附属端口按 port 派生，故多 Head 同机隔离共存）；作业平台 `serve_worker.sh`（各持一卡
`ray start --address $RAY_HEAD_ADDR:$RAY_PORT --block`，地址优先级 `RAY_HEAD_ADDR` > `MASTER_ADDR` >
`localhost`，`RAY_HEAD_ADDR` 绕开 Volcano 注入的 `MASTER_ADDR`）；开发容器 `serve_submit.sh`（env 接口
同 serve_master，attach `--ray-address 127.0.0.1:$RAY_PORT` 跑一个实验后 detach，不拆集群，下一条
submit 复用同集群）。约束：同机多 Head 时 driver 必须显式 address（`auto` 遇多个 GCS 会报错）；一个
集群同时只能跑一个实验（每次 `serve_submit` 结束 `evict_all` 整个共享 GPU 池），要并行须用不同集群；
`ray stop` 是全机全局（无 address 作用域），选择性停某个 Head 用 `pkill -f /tmp/ray_<port>`。本地无 GPU
冒烟：`scripts/workflow/dev_sim.py` 用多进程 Ray（loopback head + workers）+ function-only echo workflow
端到端验证 master 全链路。

注册前置（fail-fast）：agent workflow 需 predictions、accelerators、每个 `(model, gpu_kind)`
覆盖且有 ≥ `max_new_tokens` 的 output bucket、`model_path` 存在、`vllm_python_executable`
可执行；纯 function workflow 无此要求（dev-sim 即纯 function）。

## 当前扩展边界

下一阶段扩展应显式选择新的系统契约，不从历史方案恢复旧接口。当前可复用边界包括：

- 在 `schema.py` 扩展新的静态节点配置；
- 在 `policy.py` 增加纯调度决策；
- 在 `SchedulerCore` 扩展 session、资源或生命周期状态；
- 在 `ModelReplica`/backend 边界扩展 serving 能力；
- 在 `src/experiment/workflow` 增加新的实验场景和验证；
- 在 `WorkflowFleet` 增加新的跨 workflow 编排策略（例如按 workflow 硬性预留加速卡——已有
  设计但默认不启用，见 `docs/workflow/system_plan_20260720.md`）。

在线 GNN 预测器接入（`ResourceContractSource.GNN_PREDICTED` 的实际产出链路）、动态 DAG、
多模型 GPU packing、功耗优化、新的 token 控制机制、跨节点分布式调度均不属于现行实现。
