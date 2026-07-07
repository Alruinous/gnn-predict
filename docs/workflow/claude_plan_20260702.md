# src/workflow 运行时骨架修复与补全（Phase 1）

## Context

`src/workflow/` 是一个通用 LLM-agent workflow 运行时（Ray actor + Ray Queue），当前处于"重写中"状态：用户此前删除了一套更重的实现（`ResourcePool`/`GnnPredictionService`/`TraceRecorder`/`TokenBudgetController`/`WorkflowScheduler` 等，见 commit `a154463`），并手写了一个极简骨架作为重新开始的起点（commit `8145dc3`，当前 HEAD，仅 `controller.py`/`worker.py`/`types.py`/`profile.py` 四个文件）。

根 目录 `dev.md` 是本次任务的权威说明，明确要求：LangChain/LangGraph 仅用于单节点内部构建 agent（`worker.py::build_agent`），图级调度/队列/并发/GPU 分配/模型加载驱逐/prefetch 全部由本项目自己的 `WorkflowController` + `NodeWorker` 承担；每条流转数据是 `WorkflowDataItem`（`session_id`/`item_id`/`source_node`/`target_node`/`message`）；边是有界队列；不做无界流、任意环、分布式 exactly-once、复杂窗口；并明确优先级——**先把通用运行时骨架做对，再逐步补资源预测/驱逐/prefetch/token budget**（Phase 2+，不在本次范围）。

用户已确认一个关键设计约束：**当前静态 DAG 中，一个上游节点到一个下游节点在同一 session 下只发送恰好一个 item**（哪怕是 batch/动态任务场景，产出方节点也要把批量结果打包进一个 item 里）。这意味着 fan-in 是固定 arity 的 join（等所有配置的依赖 source_node 各到齐一个 item），不需要引入 end-of-stream 标记这类新概念——只需要修正现有 join 逻辑里的 key 使用错误。

我通读了 `dev.md`、`docs/dev/workflow_plugin_20260626.md`、`docs/workflow/` 下全部日期化实验文档、当前 `src/workflow/` 全部源码、git 历史（含已删除的 `plan_20260639.md` 和已删除的重架构 `.pyc` 残留）、`src/common/validate.py`/`log.py` 现有约定，以及 `python-norms`/`feature-iteration` 两个 SKILL。逐行核对当前代码后，发现的问题比最初预期更根本——**当前骨架实际上完全不能跑通任何一条边**，原因见下。本计划只覆盖"把骨架修对/补全"这一层，不涉及 GNN 预测器/资源池/prefetch/驱逐/token budget（Phase 2+，仅在文末给出路线图级别的说明）。

## 已核实的关键 bug（按修复优先级排列，均已直接读源码确认，非推测）

1. **队列没有真正连通任何一条边**：`controller.py::prepare_queues_for_workflow` 用两个独立的 dict comprehension 各自 `Queue()`，`input_queue_map[target]` 和 `output_queues_map[source][target]` 是两个不同的 `Queue` 实例。上游 `put()` 进的队列，下游从来不会读——数据完全不流动，与 bug 4（fan-in key 错误）无关，是更底层的阻断。
2. **同一原因导致入口/终端节点直接 KeyError**：`input_queue_map`/`output_queues_map` 分别只对 `len(dependencies) > 0`/`len(adjacency) > 0` 的节点建 key，但 `prepare_node_workers` 对每个节点无条件 `input_queue_map[node_name]`/`output_queues_map[node_name]`——入口节点（无依赖）和终端节点（无下游）会直接抛 `KeyError`。
3. **`stop_workflow()` 永远无法真正停止 worker**：Ray actor 默认单线程顺序执行任务，`worker.loop.remote()` 是一个 `while True` 的无限循环，之后再 `worker.stop.remote()` 只会排在 actor 任务队列里永远等不到执行。唯一可行的停止路径是 `loop()` 里已经写好但从未被触发的哨兵逻辑：`WorkerQueueItem(worker_state=STOPPED)` 从 `input_queue` 读到时才会 `self.stop(); break`。
4. **没有任何地方调用过 `NodeWorker.load()`**（已用 grep 核实，全仓库零引用）：即便前三个 bug 修好，`self.agent` 也永远是 `None`，所有 item 会永远堆在 `pending_items` 里，骨架仍然跑不起来。
5. **`_process_item` 对入口节点的处理是错的**：`self.dependencies == []` 时，`content = [self.store[...][source] for source in self.dependencies]` 求值为空列表，而不是 `item.message` 本身——入口数据被丢弃成 `[]`。
6. **fan-in store 的 key 用错**（原始已识别 bug）：写入用 `item.item_id` 做 key，读取判断到齐/取值用 `source`（`self.dependencies` 里的 source_node 名）——多依赖 fan-in 场景下必现 `KeyError`。
7. **`_process_item` 无条件假设每个节点都要跑一次模型推理**（`assert self.agent is not None; result = self.agent.invoke(prompt)`），没有"无 `execution_config` 的纯透传/dispatcher 节点"这条路径——与 `experiment_20260626.md` 里"dispatcher 不使用 LLM"的场景矛盾。
8. **终端节点的输出被静默丢弃**：`output_queues` 为空时 `for output_name, output_queue in self.output_queues.items()` 循环零次，计算结果不落地到任何地方，controller 也没有办法查询某个 session 是否已经跑完。
9. `WorkflowController.from_yaml` 是残缺 stub（读了 YAML 存进局部变量 `dc`，没有构造/返回任何东西）；`get_workflow_status()` 是空 `pass`。
10. `prepare_node_workers` 硬编码 `tools=[]`、`system_prompt=""`（用户自己标注的 TODO：`NodeConfig` 还没有 `system_prompt` 字段）。
11. 无任何有界队列/背压配置——与 dev.md"边是有界队列"的明确要求相悖。
12. 无任何重试/失败记录机制——与 `workflow_plugin_20260626.md` 的失败/重试要求相悖。
13. `types.py` 死代码：`BOUNDARY_NODE_TYPES`/`MODEL_NODE_TYPES`/`EVALUATOR_TASKS`/`EDGE_CONDITIONS`/`WorkerAction`、未使用的 `from collections import deque`（已用 grep 核实在 `src/workflow`、`tests/`、`main.py` 均无引用；`worker.py` 里 `deque` 的导入是另一处，正常使用，不动）。
14. 零测试覆盖；`config/workflow/summarization_compare_20260621/*.yaml` 与当前 schema 不兼容（`execution` 现在是必填字段，旧 YAML 的 input 节点没有），不能直接拿来做冒烟测试。
15. `main.py` 没有任何入口接入 `src/workflow/`。

## Sub-stage 0 —— 修队列连通性 + 真正的停止信号（bug 1/2/3/4）

- `controller.py::prepare_queues_for_workflow`：先为**每一个**节点（不再按 `len(...) > 0` 过滤）建一个共享 `Queue`，作为该节点的 input queue；`output_queues_map[source][target]` 直接引用（同一对象，不是新建）`input_queue_map[target]`。
- `controller.py::prepare_node_workers`：入口节点判定改为显式 `entry_nodes = [n for n in node_names if len(dependencies[n]) == 0]`，`assert len(entry_nodes) == 1`（fail-fast，多入口节点场景明确不在本阶段支持范围）。
- `controller.py::WorkflowController.__init__`：保留 `self.input_queue_map: dict[str, Queue]`（供 `stop_workflow`/状态查询使用）。
- `controller.py::WorkflowController.stop_workflow`：改为对每个节点 `self.input_queue_map[node_name].put(WorkerQueueItem(worker_state=WorkerState.STOPPED))`，不再直接调 `worker.stop.remote()`。
- `worker.py::NodeWorker.stop`：补上 `self.status = WorkerState.STOPPED`（现在只 evict，不转状态）。
- `worker.py::NodeWorker.load`：改签名为 `load(self) -> None`，内部用 `self.execution_config.devices[0]` 取设备，不再要求调用方传 `device_name`。
- `controller.py::WorkflowController.start_workflow`：对每个 `execution_config is not None` 的节点，在 `worker.loop.remote()` 之前先 `ray.get(worker.load.remote())`（同步等待加载完成，Phase 1 用最简单的"启动时全量 eager load"策略，不做任何 GPU 池/驱逐/prefetch 决策——那是 Phase 2+ 范围）。

## Sub-stage 1 —— 修 fan-in key 错误 + 补透传节点路径（bug 5/6/7/8）

- 在 `worker.py` 里抽出纯函数（不依赖 Ray/GPU，可独立单测）：
  `resolve_session_content(store, dependencies, item) -> tuple[bool, Any]`：
  - `dependencies` 为空（入口/无依赖节点）：直接返回 `(True, item.message)`，不经过 store。
  - 否则：`store[item.session_id][item.source_node] = item.message`（修正 key，用 `item.source_node` 而不是 `item.item_id`）；`assert item.source_node not in` 已写入的旧值（同一 session 同一上游只应到一次，fail-fast 校验固定 arity 假设）；未到齐返回 `(False, None)`；到齐后返回 `(True, [store[session][d] for d in dependencies])` **并清空 `store[session_id]`**（避免长跑 session 下 store 无限增长——当前代码 join 触发后从不清理）。
- 重写 `NodeWorker._process_item`：调用上面的辅助函数拿到 `content`；若 `self.execution_config is None`（纯透传/dispatcher 节点），直接把 `content` 作为输出，不调用 `self.agent.invoke`；否则走现有的 prompt 拼接 + `self.agent.invoke` 路径。
- 输出分发：若 `self.output_queues` 非空，按现有逻辑广播；若为空（终端节点），改为写入 `self.terminal_results: dict[str, AgentState]`（新增实例属性），不再静默丢弃。
- `NodeWorker` 新增 actor 方法 `has_result(self, session_id) -> bool`、`get_result(self, session_id) -> AgentState`，供 controller 查询终端结果。

## Sub-stage 2 —— 有界队列（bug 11）

- `types.py::NodeConfig` 新增 `queue_capacity: PositiveInt = Field(default=16)`（复用 `src/common/validate.py::PositiveInt`）。放在 `NodeConfig` 而不是 `EdgeConfig`：因为运行时是"每个节点一个共享 input queue"，容量本质是节点属性；放在 `EdgeConfig` 会引出"多条边容量不一致时听谁的"这种没有必要的歧义。
- `controller.py::prepare_queues_for_workflow`：`Queue(maxsize=node_map[node_name].queue_capacity)`。
- 背压行为：直接依赖 `ray.util.queue.Queue.put()` 默认 `block=True` 语义——下游队列满时，上游 worker 的 `put()` 会阻塞在自己的 actor 循环里，从而自然停止消费自己的 input queue，背压逐级向上游传导，不需要额外的显式 `Full` 处理逻辑。

## Sub-stage 3 —— 重试 / 失败记录（bug 12）

- `types.py` 新增：
  ```
  class RetryConfig(BaseModel):
      max_attempts: PositiveInt = 1
      retry_delay_sec: NonNegativeInt = 0
      on_exhausted: Literal["fail_workflow", "skip_item"] = "fail_workflow"

  class FailureRecord(BaseModel):
      node_name: NonEmptyStr
      session_id: NonEmptyStr
      item_id: NonEmptyStr
      attempt: PositiveInt
      error_type: NonEmptyStr
      error_message: str
  ```
  `NodeConfig` 新增 `retry: RetryConfig = Field(default_factory=RetryConfig)`。
- `worker.py::NodeWorker.__init__` 新增 `failure_queue: Queue | None = None` 参数（控制面队列，与数据面队列分开，controller 持有另一端供 `get_workflow_status` 读取）。
- 新增 `_invoke_agent_with_retry(self, prompt)`：只在 `self.agent.invoke(prompt)` 这一调用外包一层有界重试（`except Exception` 范围严格限定在这一行第三方调用上，不做 catch-all）；耗尽后必然构造 `FailureRecord` 推到 `failure_queue` 并用 `get_logger` 记日志（绝不静默丢弃）；`on_exhausted == "fail_workflow"` 时重新抛出，让 `loop()` 捕获后调用 `self.stop()` 并退出循环；`"skip_item"` 时记录后跳过、不产出下游 item（已知限制：被跳过的 item 会让等它的下游 fan-in 永远等不到，Phase 1 不做超时/悬挂检测，留给后续阶段）。

## Sub-stage 4 —— `system_prompt` 字段 + `execution` 可选化 + 死代码清理（bug 9/10/13）

- `types.py::NodeConfig` 新增 `system_prompt: NonEmptyStr | None = None`；`controller.py::prepare_node_workers` 用 `node_info.system_prompt` 替换硬编码 `""`，删掉对应 TODO 注释。`tools=[]` 保持硬编码——目前没有 `ToolConfig` schema，明确不在本阶段范围内，不做任何铺垫性设计。
- `types.py::NodeConfig.execution` 改为 `ExecutionConfig | None = None`，新增 `model_validator`：`type in (NodeType.AGENT, NodeType.TOOL)` 时必须非空，否则 raise（fail-fast，不做反向约束，不为 `OUTPUT`/`EVALUATOR` 强加规则）。
- `types.py::Workflow` 新增一个 `model_validator`，校验每条 `EdgeConfig.source`/`target` 都是真实存在的节点名（否则现在会在 `prepare_queues_for_workflow` 深处报出一个很难懂的 `KeyError`）。
- 删除 `types.py` 中的 `BOUNDARY_NODE_TYPES`/`MODEL_NODE_TYPES`/`EVALUATOR_TASKS`/`EDGE_CONDITIONS`/`WorkerAction`/未使用的 `deque` import。
- `controller.py::WorkflowController.from_yaml`：修复为 `raw = yaml.safe_load(f); return cls(Workflow.model_validate(raw))`；`import yaml` 挪到模块顶部。
- `controller.py::WorkflowController.get_workflow_status`：实现为返回一个新增的 `types.py::WorkflowStatus`（`node_states: dict[str, WorkerState]` 通过新增的 `NodeWorker.get_state` 方法 `ray.get`；`queue_sizes: dict[str, int]` 直接调 `Queue.qsize()`，非 remote 调用；`pending_failures: int`，从 `failure_queue` 非阻塞排空计数）。

## Sub-stage 5 —— 可测试性 seam + 示例 workflow + 测试

- `worker.py::NodeWorker.__init__` 新增 `agent_factory: Callable[..., CompiledStateGraph] = build_agent`；`load()` 内部改调用 `self.agent_factory(...)`。测试可注入一个返回固定 `.invoke()` 结果的 stub，绕开真实 HuggingFace 模型加载/GPU 依赖，同时仍然完整走真实的 actor/queue/controller 连线逻辑。
- 新增示例 workflow YAML（schema 合法，覆盖 fan-out + fan-in + 固定 arity join），例如 `config/workflow/runtime_smoke_20260702/fan_in_smoke.yaml`：`reader`（`input` 类型，无 `execution`）→ `chunk_a`、`chunk_b`（`agent` 类型，fan-out）→ `reducer`（`agent` 类型，fan-in，终端节点）。不复用 `config/workflow/summarization_compare_20260621/` 下的旧文件（已确认 schema 不兼容）。
- 新增测试（遵循 `AGENTS.md` 行为导向命名，复用 `tests/conftest.py` 已有的 `sys.path` 设置）：
  - `tests/test_workflow_types.py`：唯一节点名校验、`execution` 按类型必填校验、边引用不存在节点报错、`queue_capacity`/`retry` 默认值。
  - `tests/test_workflow_worker.py`：`resolve_session_content` 按 `source_node` 而非 `item_id` 记录（bug 6 回归）、未到齐不触发、到齐后清空 store、入口节点（空依赖）直接透传 `item.message`、终端节点结果可通过 `has_result`/`get_result` 取到、重试耗尽后记录 `FailureRecord` 且分别验证 `fail_workflow`/`skip_item` 两种行为。
  - `tests/test_workflow_controller.py`：`prepare_queues_for_workflow` 对同一条边的两端返回同一个 `Queue` 对象（bug 1 回归）、`from_yaml` 能正确构造、`stop_workflow` 通过 `input_queue_map` 而非 `.stop.remote()` 发送哨兵（bug 3 回归）、`get_workflow_status` 正确汇总。
  - `tests/test_workflow_end_to_end_smoke.py`：用 `agent_factory` stub + 上面的示例 YAML，启动 `WorkflowController`，提交若干 `session_id` 进入口队列，轮询 `is_session_complete`/`get_result` 确认 `chunk_a`/`chunk_b` 两路都被 `reducer` 消费到，再 `stop_workflow()` 并确认所有 worker 转为 `STOPPED`。

## 明确不做的事

- 不接入 GNN 预测器（`src/gnn_model/models/predictor.py`）、不做 `ResourcePool`/驱逐/prefetch/token budget——dev.md 明确要求先把骨架做对。
- 不改 `main.py`（纯 `gnn_archs` CLI，混入 workflow 概念不合适）；如需要 CLI，作为后续 `scripts/run_workflow.py`（符合 `AGENTS.md` "一次性维护脚本放 scripts/" 的约定），本次不做。
- 不修改 `docs/workflow/experiment_20260626.md` 除实验内容/结果外的任何部分（用户已在文档内显式禁止）。
- 不恢复任何已删除的重架构（`ResourcePool`/`GnnPredictionService`/`TraceRecorder`/`TokenBudgetController`/`WorkflowScheduler`/`scheduled_tool_nodes.py`/`model_export.py` 等）。
- 不实现"上游对同一 session 发送数量不固定的多个 item"这种能力——已被用户明确否定；批量场景由产出节点自己在一个 item 内打包。

## Phase 2+ 路线图（仅作说明，不在本次实现范围）

按 `docs/workflow/experiment_20260629.md` 的阶段划分，本计划完成后（骨架具备真实的 fan-out/fan-in、背压、重试、drain 能力）再进入：Stage A（BufferNode 真实流水线吞吐实验，验证本计划是否真的达成了此前离线模拟的 1.68x~1.82x 理论加速）；Stage B（Qwen 场景下 GNN checkpoint 的验证/重训决策、token budget/OOM 控制、GNN-aware 在线调度）；Stage C（基于 ETA 的模型 prefetch、驱逐/复用生命周期）；Stage D（预测误差鲁棒性、动态任务生成/批量工具调用、能耗感知调度）。在 Qwen 推理场景下的 GNN checkpoint 验证结果达标前，不应把预测器当作 OOM 硬性门控条件。

## 验证方式

1. `uv run python -m pytest tests/test_workflow_types.py tests/test_workflow_worker.py tests/test_workflow_controller.py tests/test_workflow_end_to_end_smoke.py -q`
2. `uv run python -m pytest -q`（确保未破坏其他模块）
3. `uv run ruff check src/workflow tests/test_workflow_types.py tests/test_workflow_worker.py tests/test_workflow_controller.py tests/test_workflow_end_to_end_smoke.py`
4. `uv run ruff format` 同上路径
5. `uv run ty check src/workflow main.py tests`
6. 端到端冒烟测试本身（`test_workflow_end_to_end_smoke.py`）即验证真实 Ray actor + Queue 连线跑通 fan-out/fan-in/drain，不依赖真实 GPU/HuggingFace 模型（通过 `agent_factory` stub 注入）。

# Phase 2：预测感知的 Workflow 调度系统（workflow.tex Algorithm 1 落地）

## Context

Phase 1 骨架（`docs/workflow/claude_plan_20260702.md`）已全部实现并通过测试：队列连通、fan-in、有界队列、重试、终端结果、agent_factory 测试缝、4 个测试文件。当前 `src/workflow/` 仅有静态 eager-load 模式——每个 NodeWorker 独占加载模型、无资源池/预测/驱逐/prefetch。

本阶段实现论文核心系统（`docs/draft/workflow.tex` Algorithm 1 + claims 节）：面向 workflow DAG，用 GNN 预测器（`output/gnn_llm_only_full_retrain_20260702` checkpoint）预测各节点推理耗时、显存峰值、部署耗时，由 controller 侧调度器集中决定模型实例的**加载 / 预取 / 复用 / 驱逐**与加速卡放置（本机 4× V100-SXM2-32GB）。

**文献调研结论**（支撑设计与后续论文写作，报告已由调研 agent 产出）：
- 新颖性确认：现有 workflow 感知服务系统（Parrot/Autellix/Kairos/SAGA/Pythia/KVFlow/PBKV/Helium）生命周期客体均为 KV cache 或请求路由，模型权重一律预驻留；模型级生命周期系统（ServerlessLLM/Torpor/Aegaeon/MuxServe/Prism）均 workflow 无关。"GNN 结构性能预测 → 整模型实例生命周期 → 受限本地 GPU"是空白交点。最接近的对比对象：PBKV（GNN 预测控制流、客体 KV）、ServerlessLLM（解析式装载时间成本模型、无 DAG 前瞻）、KVFlow/Continuum（结构距离驱逐评分模板，KV 层）。
- 机制借鉴：ServerlessLLM 的显式装载时间成本模型择卡；KVFlow 的 steps-to-execution 驱逐评分 + 后台预取；Continuum 的 TTL=f(空闲预测, 重装成本)；Kairos 的显存时间线 + 在线偏差修正。
- 基线阶梯（决定可插拔要求）：static eager-load / on-demand+LRU 无预取 / 历史均值预测 / GNN 完整系统 / oracle 回放。

## 已确认的用户决策与关键假设

1. **【用户已确认】模型实例独立为 ModelInstance Ray actor**：与 NodeWorker 解耦，每个部署副本一个 actor 绑定单卡；多个节点可共享同一实例（引用语义）；NodeWorker 保留 fan-in/prompt 组装，经 LangChain ChatModel shim 转发推理。
2. **【超时默认，采用推荐项】覆盖缺口分阶段**：系统开发立即开始；实验先用 Qwen3-0.6B/1.7B/4B（与合成变体 h1024_l28/h2048_l28/h2560_l36 结构完全同构，vocab 一致已核实）；超出覆盖的 token 长度用最近 ONNX 桶 clamp + 显式 flag。**Qwen3-8B（h4096_l36）导出与长序列（s>1024）数据采集是论文正式实验的前置任务，单独排期，不在本次实现范围**。
3. **【超时默认，采用推荐项】交付范围**：tex 算法完整闭环 + 可插拔策略（驱逐 max_vram/min_vram/longest_idle/shortest_idle；放置 best_fit/worst_fit；预测源 gnn/history_mean/static）+ 对齐 `experiment_20260629.md` 50 字段 schema 的统一 trace。实验脚本本身为后续单独任务。
4. **【超时默认，采用推荐项】T_deploy 来源**：离线 profiling 脚本实测各真实模型加载耗时/空闲显存写入模型注册表；运行时用实测值在线更新；GNN deployment 预测仅作未 profile 模型的兜底（训练分布是合成 model_build，与真实 from_pretrained 非同一分布）。

预测器使用边界（写入代码注释与 flags，来自 manifest usage_notes）：**prefill run_duration WAPE 5.56 永不使用**（用历史均值+静态兜底）；decode run_duration（WAPE 0.095）与 gpu_mem（WAPE 0.063）可用但显存最大低估 24.3GB，只作带 eps_mem 安全边际的软门控，OOM 硬防线在实例边界 catch。

## 架构

三平面分离，保持 dev.md 约束（controller 不侵入 prompt 逻辑、worker 不知全局调度）：

- **数据面（不变）**：NodeWorker.loop() 拉队列、`resolve_session_content` fan-in、组装 prompt。
- **控制面（新）**：命名 Ray actor `SchedulerActor` 持有模型池、加速卡账本、预测源、ETA 表、trace writer；tick 循环是纯决策函数 `plan_tick`（状态入→动作出）的薄壳，tex 算法可无 Ray/GPU 单测。
- **执行面（新）**：`ModelInstance` actor（每副本一个、绑单卡、generation 级 API）；NodeWorker 经 `InstanceChatModel`（BaseChatModel 子类，`_generate` 内 `ray.get(instance.generate.remote(...))`）接入，`create_agent`/`agent_factory` 缝保持。
- **协议**：worker fan-in 就绪后 `ray.get(scheduler.acquire.remote(node, session, item, input_tokens))` 阻塞至授予 `GrantInfo(instance_id, actor_name, device)`（内部 threading.Event + 超时 raise）；用完 `release(..., result, oom)` 上报实测值。ETA 无需额外 RPC：授予即记 RunningTask(t_start=t_grant, est_duration=同 tick 预测值)。
- **向后兼容**：`Workflow.scheduler is None` → Phase-1 静态路径逐字节不变，现有测试全部保持通过。

## 分阶段实现（每阶段可独立提交并有测试门）

### Sub-stage 0 — 配置 schema、模型注册表、profiling 脚本

文件：`src/workflow/types.py`（扩展）、`src/workflow/registry.py`（新）、`config/workflow/model_registry_v100.yaml`（新）、`scripts/profile_workflow_models.py`（新）。

- `types.py` 新增：`GpuDeviceConfig(device, gpu_name="v100", total_mem_mb)`、`GnnPredictorConfig(checkpoint_path, effective_config_path, scaler_dir, target_names, prewarm)`、`SchedulerConfig(tick_interval_sec=0.5, eps_time_sec=1.0, eps_mem_mb=2048, placement_policy, eviction_policy, prediction_source, prefetch_enabled=True, adaptive_max_new_tokens=False, acquire_timeout_sec=600, devices, model_registry_path, gnn|None, trace_path|None)`；`Workflow.scheduler: SchedulerConfig | None`；`ExecutionConfig.devices` 改 optional（validator：静态模式必填、managed 模式忽略）；`InstanceState(LOADING/IDLE/BUSY/EVICTING)`、`GrantInfo`、`GenerateResult(text, input_tokens, output_tokens, duration_sec)`、`PredictionResult(duration_sec, vram_mb, deploy_sec, source, flags)`。
- `registry.py`：`ModelSpec(name, path, onnx_template, onnx_dir, param_count_b, profiled_load_sec|None, profiled_idle_vram_mb|None, static_prefill_sec, static_decode_sec, static_vram_mb)`；`load_model_registry(path)`；`discover_onnx_buckets(spec)`（glob prefill `{tpl}_bs1_s*.onnx` 与 decode `{tpl}_decode_bs1_s*_o*.onnx`，空则 fail-fast）。注册表含 qwen3-0.6b/1.7b/4b 三条映射；8b 条目留注释说明前置采集任务。
- profiling 脚本：每模型 ×3 次子进程实测（pynvml used 前后差 + from_pretrained 墙钟），取中位数回写注册表 YAML。

验证：`tests/test_workflow_types.py` 扩展 + 新 `tests/test_workflow_registry.py`（tmp_path 假 onnx 目录）。

### Sub-stage 1 — ModelInstance actor + ChatModel shim（手动分派可用）

文件：`src/workflow/model_instance.py`、`src/workflow/chat_shim.py`（新）、`tests/test_workflow_model_instance.py`、`scripts/smoke_model_instance.py`（GPU 冒烟）。

- `ModelInstance`（`@ray.remote(max_concurrency=4)`）：`__init__(instance_id, spec, device, dtype, pipeline_factory|None)`（pipeline_factory 为测试缝，镜像 agent_factory）；`load() -> InstanceLoadReport(load_sec, idle_vram_mb)`（真实路径 AutoTokenizer + AutoModelForCausalLM.from_pretrained(device_map={"":idx})，pynvml 测占用）；`generate(messages, max_new_tokens, ...) -> GenerateResult`（实例侧 `tokenizer.apply_chat_template` 复现 ChatHuggingFace 的 prompt 格式化，返回精确 token 计数与耗时）；串行执行，BUSY 状态标志；**唯一窄异常边界**：`except torch.cuda.OutOfMemoryError` 仅包 `model.generate` → empty_cache + 重抛类型化 `InstanceOomError`。
- `chat_shim.py`：`InstanceChatModel(BaseChatModel)`（`bound_actor_name` 每 item 由 worker 绑定；单线程 `_process_item` 保证安全）+ `build_managed_agent(config, system_prompt, tools, device)`（复用 `langchain.agents.create_agent`）。

验证：CPU fake 测试 + Qwen3-0.6B 单卡真实冒烟脚本。

### Sub-stage 2 — ModelPool 与加速卡账本（纯 Python）

文件：`src/workflow/resource.py`（新）、`tests/test_workflow_resource.py`。

- `InstanceRecord(instance_id, spec_name, actor_name, device, state, idle_since, busy_task, measured_load_sec, measured_idle_vram_mb, reserved_vram_mb, suspect)`；`AcceleratorState(device, gpu_name, total_mem_mb, nvml_free_mb)`；`ModelPool`（普通类）：`idle_instances/loading_instances/predicted_free_mb(=min(nvml视图, 账本视图))/evictable`。
- pynvml 封装 `probe_devices(...)`（调度器进程无 CUDA 上下文，pynvml 可见全局占用，优于 mem_get_info）。
- 可插拔策略为**纯函数**（python-norms，不搞类层次）：`select_accelerator(pool, need_mb, policy)`（best_fit=最小满足空闲 / worst_fit=最大空闲）；`select_eviction_victim(pool, device, deficit_mb, policy)`（仅 IDLE、实测显存 ≥ 缺口——tex 必要条件；四种 victim 策略）。驱逐用**实测**空闲显存而非预测值（per tex）。

验证：纯单测覆盖账本运算、两种放置 × 四种驱逐策略、只逐空闲不变量。

### Sub-stage 3 — 调度核心：纯决策函数（tex Algorithm 1 逐行映射）

文件：`src/workflow/scheduler.py`（本阶段只有函数）、`tests/test_workflow_scheduler_core.py`。

`plan_tick(state, t_now, config, predict, deploy_time) -> list[Action]`，Action = `Grant | Load(spec, device, prefetch) | Evict(instance_id, reason)`：

- `C_ready` = 已挂起的 acquire 请求（fan-in 完整性由构造保证——worker 只在 resolve 就绪后才 acquire）；`C_near` = 所有依赖要么已完成、要么正在上游 RunningTask 中的 (session, node)，`t_ready_hat = max(t_start + est_duration)`（tex fan-in max 规则）。
- 按拓扑序（Kahn，init 时一次）遍历，节点内 FIFO（tex 固定）。
- `PredictTokens`：ready 用 worker 传来的精确 input_tokens；near 用上游历史平均输出 token（冷启动兜底 max_new_tokens）+ prompt 模板 token 开销（init 时 tokenize 一次）；L_out 同理。
- `predict(spec, in, out, gpu)` → PredictionResult；ready 且有 idle 实例 → Grant（规划账本内立即置 BUSY 防同 tick 重复授予）。
- `T_deploy` 兜底链：profiled → 运行时实测滑动均值 → GNN deployment 预测。
- `t_now + T_deploy + eps_time >= t_ready_hat` 才走加载分支（ready 任务 t_ready_hat=0 恒真）→ `select_accelerator(M_peak + eps_mem)` → 不足则 `select_eviction_victim` → Evict 后重试 → `Load(prefetch=x∈C_near)`。
- **副本泛滥防护**（tex 循环的必要细化）：同 tick 内 LOADING 实例先 FIFO 匹配未服务候选，只有未匹配者才触发新 Load；`prefetch_enabled=False` 时加载分支仅对 C_ready 开放。
- 无法服务的任务留在 pending，下一 tick 重扫（tex FIFO buffer）。
- Token budget 最小实现：`vram_mb + eps_mem > max(total_mem)` → fail-fast（acquire raise 显式原因，绝不静默截断）；`adaptive_max_new_tokens` 开关下折半重预测并在 GrantInfo 携带 clamp 值（默认关，研究旋钮）。

验证：无 Ray/GPU 纯单测——ready 分派、near-ready 预取时机（早于阈值不触发）、驱逐链式重选卡、全策略组合、FIFO/拓扑序、副本防护、ETA fan-in max。**这是论文算法的直接测试载体**。

### Sub-stage 4 — SchedulerActor 壳、acquire/grant 协议、managed 模式接线

文件：`scheduler.py`（加 actor）、`worker.py`（managed 路径）、`controller.py`（接线）、`tests/test_workflow_scheduler_actor.py`、`tests/test_workflow_managed_end_to_end.py`。

- `SchedulerActor`（`@ray.remote(max_concurrency=32)`，threading.Lock 守护共享态）：`acquire()`（注册 PendingAcquire + Event，`event.wait(timeout)` 超时 raise TimeoutError → 走 worker 现有重试边界成 FailureRecord）；`release()`（关 RunningTask、置 IDLE+idle_since、喂历史统计、更新下游 done_deps）；`run()`（Δt tick：pynvml 刷新 → plan_tick → 执行动作）；`stop()`（停循环、逐全部实例、flush trace）；`instance_factory` 与 `clock` 注入为测试缝。
- 实例生命周期执行：Load → `ModelInstance.options(name=...).remote()` + `load.remote()`，每 tick `ray.wait(timeout=0)` 轮询完成 → LOADING→IDLE、记实测 load_sec/idle_vram（在线更新 T_deploy）；Evict → `ray.kill`（进程退出释放 CUDA 内存，简单可靠路径）。
- `worker.py` managed 路径（增量小）：构造参数加 `scheduler_name|None`；`load()` 在 managed 模式建 shim agent + CPU tokenizer（精确 input token 计数，worker 永不持 GPU 权重）；`_invoke_agent_with_retry` 每次尝试 = acquire → 绑 shim → invoke → finally release（OOM 标志在 except 路径上报）。
- `controller.py`：`workflow.scheduler is not None` 时建命名 SchedulerActor、workers 传 `scheduler_name` 与 `build_managed_agent`、start_workflow 跳过 GPU 权重 eager-load 改 `scheduler.run.remote()`、stop_workflow 追加 `ray.get(scheduler.stop.remote())`；静态模式路径不动。

验证：FakeInstance（sleep+固定 vram）CPU 集成测试（acquire 阻塞至加载完成、双节点共享单实例串行、积压触发第二副本、超时 raise）；managed 端到端（fan-in 图 + scheduler 段）；**全量回归 `uv run python -m pytest -q` 确保 Phase-1 不破**。

### Sub-stage 5 — 预测源（gnn / history_mean / static）

文件：`src/workflow/prediction.py`（新）、`tests/test_workflow_prediction.py`。

- Protocol：`predict(spec, input_tokens, output_tokens, gpu_name) -> PredictionResult`。
- `StaticPredictor`：注册表常量。`HistoryMeanPredictor`：release 喂的 (node, spec) 实测滑动均值，显存取观测 max，冷启动回落 static。
- `GnnPredictor` 初始化（一次，CPU）：直接构造 `IntelliGraphLargeModelPredictor`（`src/gnn_model/models/predictor.py:20`，dims 取 `src/gnn_model/data/constants.py` 常量，targets 取 manifest 9 目标序）→ `load_state_dict(strict=True)` 即结构契约校验 → 加载 3 个特征 scaler pickle + `load_target_scalers`（`src/gnn_model/data/scaler.py:128`）→ prewarm 缓存本 workflow 所用 spec 全部桶的**未缩放** PyG Data（把昂贵的 onnx_tool shape-infer 移出 tick）。
- 查询路径：桶选择（decode：最小 s≥input_tokens，clamp+flag `input_bucket_clamped`；再最小 o≥output_tokens，clamp+flag）→ 缓存 Data `.clone()`、改 `graph_features[0,3]=output_tokens`（已核实 index 3=decode_output_length）→ 本地 `scale_features_only`（镜像 `scaler.py::scale_data` 特征段，无 data.y）→ forward `[1,9]` → `inverse_transform_targets`（`src/gnn_model/training/metrics.py`）。
- **组合规则显式化（不静默）**：`vram_mb` = decode 图 col 7；`duration_sec` = decode 图 col 1 + prefill 历史均值（冷启动 static），flag `prefill_from_history/prefill_static_fallback`——**GNN prefill 时长永不使用**；`deploy_sec` = col 0 仅作兜底链末端。结果缓存 keyed (spec, s桶, o桶, output_tokens, gpu)。

验证：桶选择纯单测（精确/clamp/flag）；组合规则 stub 测试；`@pytest.mark.slow` 真实 checkpoint 单查询 sanity 测试（CPU）。

### Sub-stage 6 — 驱逐执行、OOM 处理、token-budget fail-fast 收尾

- 驱逐后链式 Load 前 pynvml 复测（ray.kill 进程退出异步，轮询 nvml_free 至反映释放或 5s 超时 + trace 警告）。
- OOM：`release(oom=True)` → 实例标 suspect → 下一 tick 无条件 Evict（分配器状态可能已污染）→ 重试重新 acquire；FailureRecord 复用现有 `failure_queue`（`worker.py:161-171`）。

验证：actor 测试覆盖 OOM→驱逐→重试成功、超大预测快速失败、adaptive clamp。

### Sub-stage 7 — Trace 记录 + 真实 GPU 闭环冒烟

文件：`src/workflow/trace.py`（新）、`config/workflow/managed_smoke_20260703/fan_in_managed.yaml`（新）、`scripts/smoke_managed_workflow.py`（新）。

- `TraceWriter`：SchedulerActor 单写者，JSON lines 到 `output/<run>/trace.jsonl`，字段名严格取自 `experiment_20260629.md` 50 字段 schema（DCGM-only 字段缺省不改名）；`prediction_metrics`=PredictionResult dump（含 source/flags），`schedule_decision`∈{dispatch, load, prefetch, evict, reuse, reject_vram}。事件点：task_submit / acquire_request / predict / grant / load_start/end / prefetch_start/end / evict / infer_start/end / oom / session_complete。worker/instance 不直接写，经调度器。
- 冒烟 YAML：reader → chunk_a/chunk_b（qwen3-0.6b，验证同 spec 共享实例）→ reducer（qwen3-1.7b），带 scheduler 段与真实 GNN manifest 路径，execution 不写 devices。
- 冒烟脚本：4 sessions 真跑 4× V100，断言 trace 含 ≥1 reuse、≥1 load、全部 session 完成、零失败。**这是论文闭环 demo；正式实验脚本为后续任务**。

## 复用的现有代码（不重复造）

- `worker.py::resolve_session_content`、重试边界、`agent_factory` 缝、STOPPED 哨兵——全部保留。
- `controller.py::prepare_queues_for_workflow` 的邻接/依赖构建模式（拓扑序复用其结构）。
- `src/gnn_model/models/predictor.py::IntelliGraphLargeModelPredictor`、`data/onnx_graph.py::build_graph_data_from_onnx`、`data/scaler.py::load_target_scalers`、`training/metrics.py::inverse_transform_targets`、`data/constants.py` 维度常量——gnn_model 包零修改。
- `common/validate.py` 类型、`common/log.py::get_logger`、`tests/conftest.py::ray_session/wait_until`。
- pynvml（pyproject 已有依赖）。

## 验证方式

1. 每阶段：`uv run python -m pytest -q tests/test_workflow_*.py` + 新增测试。
2. 全量回归：`uv run python -m pytest -q`（Phase-1 四个测试文件必须原样通过）。
3. `uv run ruff check main.py src scripts tests && uv run ruff format ... && uv run ty check src main.py tests`。
4. GPU 冒烟（人工两步）：`scripts/smoke_model_instance.py`（Sub-stage 1 后）与 `scripts/smoke_managed_workflow.py`（Sub-stage 7 后，验证真实加载/复用/预取/驱逐闭环 + trace 完整性）。
5. profiling 脚本一次性运行填充注册表（Sub-stage 0 后、GPU 冒烟前）。

## 风险与已知边界（写入实现注释/文档）

1. **账本与现实漂移**：共享机器上外部进程可在 tick 与加载完成之间侵占显存；`predicted_free_mb` 取账本/nvml 双视图 min + eps_mem + 实例边界 OOM 路径缓解，不根治。
2. **prefill 冷启动**：历史累积前 ETA 用 static prefill + GNN decode，首轮预取时机粗糙——预取只是优化不影响正确性，flags 可审计。
3. **4B（h2560_l36）覆盖薄**（prefill s≤512、decode o≤32）：真实摘要输入会持续 `input_bucket_clamped`，4B 预测属外推。前置采集任务（8B 导出 + 三模型长序列行）是论文正式数据的先决条件。
4. **decode 图作显存代理**：假设满 KV 的 decode 峰值 ≥ prefill 峰值；长输入短输出场景可能反转，预留 `max(prefill_vram, decode_vram)` 一行扩展，先按 decode-only 并对照冒烟 trace 检查。
5. infer_start 语义定义为 grant 时刻（与 worker 实际 invoke 相差微秒级，可忽略）。

## 明确不做的事

- 不写正式实验脚本/对照组 runner（后续任务，本计划交付其依赖的可插拔策略与 trace）。
- 不采集新监控数据、不重训 GNN（单独排期）。
- 不做跨机分布式、任意环、exactly-once、复杂窗口（dev.md 红线）。
- 不恢复已删除重架构代码；文件名自然重合（scheduler.py/resource.py/trace.py/prediction.py）但全部重新设计。
- buffer 消费策略固定 FIFO（tex 明确不研究）。
