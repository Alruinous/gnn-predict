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
