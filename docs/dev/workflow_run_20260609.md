# Workflow 运行设计探索

当前 Workflow 配置只描述 DAG、节点模型先验和运行输入参数。运行层不要反向污染配置 schema，也不要把 `serial`、`parallel`、`adaptive` 做成三套互相独立的执行代码。

更合适的抽象是：

```text
WorkflowConfig
  -> Scheduler
  -> HandlerPool
  -> NodeHandler
  -> NodeResult
  -> WorkflowRunResult
```

`Scheduler` 只负责依赖和提交策略。`NodeHandler` 负责真正执行节点。worker 不需要一开始设计成很具体的远程 worker，可以先是一个本地 handler 调用，后续再换成线程池、进程池、Ray、Kubernetes Job 或其他执行后端。

## 核心判断

- `serial`、`parallel`、`adaptive` 是三种调度策略，不是三套运行器。
- 首版运行时应把边理解为依赖关系，不做 tensor 或业务数据流传递。
- 节点运行失败是正常结果，OOM 也是允许发生的节点失败，不需要在朴素 `parallel` 阶段提前规避。
- `parallel` 不等于每一层 BFS 全量同时运行，而是依赖满足后进入 ready queue，再按 worker capacity 提交。
- `adaptive` 才引入 GNN 预测器，用预测结果影响 ready node 的选择、排序或资源组合。
- 不要在 `parallel` 中提前实现显存估计、GPU 利用率装箱、重试策略或复杂资源模型。

## Serial

`serial` 是正确性基线，应该最先实现。

行为：

- 对 DAG 做拓扑排序。
- 每次只运行一个 ready node。
- 当前节点结束后记录 `NodeResult`。
- 上游失败时，下游默认 `skipped`。
- 只要 serial 跑通，节点执行、状态记录、失败传播和结果汇总的基本语义就明确了。

`serial` 的实现不需要 worker pool。可以直接调用：

```text
handler.run(node_name, node, context)
```

## Parallel

`parallel` 首版采用朴素 DAG 调度：

```text
ready queue + max_workers + handler.submit()
```

调度算法应使用 Kahn indegree 逻辑：

- 初始化所有 indegree 为 0 的节点到 ready queue。
- worker slot 有空时，从 ready queue 取节点提交。
- 节点完成后，更新后继节点 indegree。
- 后继节点所有上游完成后进入 ready queue。
- 所有节点进入终态后结束。

不要按“BFS 层级一次性全提交”实现。层级执行虽然容易理解，但会过度限制并发，也容易在多上游依赖时把“访问过”误当成“依赖已完成”。ready queue 更接近实际 DAG scheduler。

首版 `parallel` 需要的参数可以很少：

- `max_workers`
- `fail_fast`

`max_workers` 控制最高并发数。它不是 GPU 资源保证，只是提交上限。

`fail_fast` 建议默认 `false`，更适合实验场景。某个分支 OOM 或失败后，不影响无依赖的其他分支继续运行。依赖失败节点的下游标记为 `skipped`。

OOM 处理：

- handler 捕获运行失败并返回失败状态。
- 如果能识别 OOM，记录 `reason: oom`。
- 不在 `parallel` 中做重试或自动降并发。
- 不把 OOM 当成调度器 bug。

## Adaptive

`adaptive` 复用 `parallel` 的执行框架，只替换 ready queue 的选择策略。

GNN 预测器可以作为 `ResourceEstimator`：

```text
node -> graph/build features -> predictor -> ResourceEstimate
```

估计值可以包括：

- `duration_sec_avg`
- `gpu_mem_used_mb_p95`
- `gpu_util_percent_p95`
- `gpu_sm_active_percent_p95`
- `gpu_sm_occupancy_percent_p95`
- 置信度或风险标记

首版 adaptive 不必直接做复杂装箱。可以先只影响排序：

- 关键路径优先。
- 预计短任务用于填空。
- 预计高风险或高显存节点单独运行。
- 预测不可靠时回退到普通 parallel。

后续再考虑资源组合：

- 显存预算。
- 多节点同时运行的 GPU 利用率互补。
- 长短任务混排。
- OOM 风险阈值。

## 建议的数据结构

运行结果至少要保留：

```text
NodeResult:
  node_name
  status
  started_at
  ended_at
  duration_sec
  error_type
  error_message
  metadata

WorkflowRunResult:
  mode
  status
  node_results
  levels_or_events
```

`status` 可以先使用：

```text
pending
running
succeeded
failed
skipped
```

如果要支持并发运行，`WorkflowRunResult` 最好记录事件或节点结果，而不是只记录静态层级。静态层级适合计划展示，不足以表达节点实际开始和结束时间。

## Handler 边界

`NodeHandler` 不要绑定具体执行后端：

```text
run(node_name, node, context) -> NodeResult
```

后续可以有多个实现：

- `NoopNodeHandler`：测试调度语义。
- `SubprocessNodeHandler`：用命令运行节点。
- `GnnArchsNodeHandler`：把 workflow node 转成现有模型 workload。
- `KubernetesNodeHandler`：提交 Pod 或 Job。

首版建议先做 `NoopNodeHandler` 或 fake handler，把 scheduler 行为测清楚。真实模型执行再接。

## 与现有项目的关系

现有 `gnn_archs` 更像“变体生成、训练/推理/ONNX/监控样本采集”链路，不是通用 DAG runtime。Workflow runner 不应直接复用 `variant_runner.run_variant()` 当成 DAG 节点运行器。

更稳妥的路线是：

- 先实现 workflow 自己的 handler 抽象。
- 再写适配层，把单个 workflow model node 映射到可运行 workload。
- 最后再接 GNN predictor 做 adaptive scheduling。

Workflow 配置里的边只表达依赖。不要在首版假设节点输出会成为下游输入。

## 开发顺序

先实现最小可运行闭环：

- `NoopNodeHandler`
- `SerialScheduler`
- `ParallelScheduler`
- `WorkflowRunResult`
- 基于小 DAG 的单元测试

然后再接真实节点执行：

- 节点状态记录。
- 失败和 skipped 传播。
- OOM reason 归类。
- 本地 handler 或 subprocess handler。

最后再做 adaptive：

- `ResourceEstimator` 接口。
- 静态 estimator。
- GNN estimator。
- adaptive ready queue 排序。
- 高风险节点独占运行策略。

## 不要过度设计

不要在首版加入：

- 多后端 worker 插件系统。
- GPU 显存装箱。
- 自动 retry。
- checkpoint/resume。
- 动态修改 DAG。
- 节点间真实数据传递。
- Kubernetes/Ray 绑定。
- GNN predictor 强依赖。

这些都可以作为后续能力，但不是把 `serial` 和朴素 `parallel` 跑通的前提。
