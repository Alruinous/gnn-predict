# 20260629 workflow runtime 实验规划

## 目的

本文把 `docs/dev/workflow_plugin_20260626.md` 中的系统设计目标，和
`docs/workflow/experiment_20260626.md` 中已经完成的动机实验，整理成后续
实现与评估的实验路线。

当前实验不应只证明“BufferNode 可以运行”，而应回答以下问题：

- 粗粒度 DAG 执行中是否存在可被 Buffered runtime 回收的等待损失。
- BufferNode 的有界缓冲和背压是否能在不破坏 workflow 依赖语义的情况下提高吞吐。
- GNN 预测信号是否能在线影响 token budget、选卡、模型加载、预取和驱逐。
- 预测误差存在时，调度策略是否仍能降低 OOM、冷启动和尾部延迟风险。

## 已有实验基础

### 端到端并行度实验

已有实验使用 QMSum test 的 24 条分层样本，比较了 32B direct、2-way chunk
workflow 和 3-way chunk workflow。

| 实验项 | 主耗时 total | 主耗时 mean | p95 | ROUGE-L | LLM score | pass rate | 输入上限命中 | OOM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 32B 直连 | 731.547 | 30.481 | 44.912 | 0.1991 | 2.750 | 0.292 | 23 | 0 |
| 切成 2 份 | 669.934 | 27.914 | 39.464 | 0.1888 | 2.875 | 0.250 | 15 | 0 |
| 切成 3 份 | 721.839 | 30.077 | 41.956 | 0.1880 | 3.125 | 0.333 | 7 | 0 |

该实验说明：

- 切分可以降低单个 chunk agent 的输入压力。
- 切成 2 份在当前设置下平均耗时最低。
- 切成 3 份输入截断更少、LLM score 和 pass rate 更高，但 merge 更重。
- 单纯增加并行度不能保证端到端更快。

### 节点耗时分析实验

已有实验记录了每个 model/evaluator 节点的 `started_at`、`ended_at`、
`duration_sec`、`input_token_count` 和 `output_token_count`。

| 实验项 | 节点 | 调用数 | input mean | output mean | duration mean | corr(input,duration) | corr(output,duration) |
|---|---|---:|---:|---:|---:|---:|---:|
| 32B 直连 | summarizer | 24 | 4087.1 | 103.0 | 30.479 | 0.206 | 1.000 |
| 切成 2 份 | chunk agents | 48 | 5889.8 | 116.4 | 15.930 | 0.782 | 0.769 |
| 切成 2 份 | merge | 24 | 325.5 | 149.8 | 10.605 | 0.769 | 1.000 |
| 切成 3 份 | chunk agents | 72 | 4496.2 | 107.9 | 15.606 | 0.632 | 0.772 |
| 切成 3 份 | merge | 24 | 425.9 | 180.8 | 12.756 | 0.665 | 1.000 |

该实验说明：

- chunk agent 的输入 token、输出 token 与耗时存在明显相关性。
- merge 的耗时与输出长度高度相关。
- token 统计可以作为 ETA、token budget 和资源预测的输入信号。

### 粗粒度 DAG 等待损失实验

已有实验分析了 LangGraph 原生 parallel workflow 在 fan-out/fan-in 后的等待损失。

| 实验项 | trace 样本数 | chunk wait mean | chunk wait p95 | merge wait mean | idle ratio mean | barrier gap mean |
|---|---:|---:|---:|---:|---:|---:|
| 切成 2 份 | 24 | 2.747 | 6.698 | 2.747 | 0.083 | 0.001 |
| 切成 3 份 | 24 | 5.124 | 12.081 | 3.501 | 0.098 | 0.001 |

跨样本流水线离线模拟结果：

| 实验项 | 当前主耗时 total | 当前 chunk+merge total | 流水线模拟 total | 理论加速比 |
|---|---:|---:|---:|---:|
| 切成 2 份 | 669.934 | 669.813 | 399.560 | 1.676 |
| 切成 3 份 | 721.839 | 721.675 | 395.680 | 1.824 |

该实验说明：

- 同一样本内最后一个 chunk 完成后，merge 基本会立刻启动。
- 主要损失不是本地调度空隙，而是 fan-out/fan-in barrier 和样本间串行。
- Buffered Node 更适合优化批量吞吐，不会消除同一样本的真实依赖。

### 有限资源下 OOM 与 token 实验

`experiment_20260626.md` 已经提出该实验，但尚未完成。目标是验证生成式节点的输出
token 上限是否会显著影响显存峰值、OOM 风险和任务质量。

该实验应作为后续 GNN token budget 控制的前置实验。

### 当前 GNN checkpoint 的 Qwen 子集现状

当前 workflow 默认 GNN checkpoint 为
`output/gnn_full_retrain_20260612/gnn_model_scaled_20260612/checkpoints/best_model.pt`，
对应配置为 `output/gnn_full_retrain_20260612/effective_config.yaml`。它可以作为
runtime 接入和离线调度回放的默认预测器，但还不能直接证明 Qwen 推理过程中的资源开销
已经被稳定预测。

对 `data/scaled/test.pt` 做只读检查时，当前 test split 中包含 `361` 条 `qwen3`
样本，其中推理相关的 `prefill` 和 `decode` 共 `299` 条，覆盖 V100 和 A100。使用当前
checkpoint 在 Qwen 推理子集上的原始量纲指标如下：

| 子集 | 样本数 | overall WAPE | run WAPE | gpu mem WAPE | gpu mem MAE | gpu mem max error | power WAPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen prefill+decode | 299 | 0.1994 | 0.2219 | 0.2005 | 1561.7 MB | 12664.5 MB | 0.1914 |
| Qwen prefill | 72 | 0.1283 | 0.9845 | 0.1274 | 1398.6 MB | 4879.4 MB | 0.2285 |
| Qwen decode | 227 | 0.2361 | 0.2206 | 0.2381 | 1613.4 MB | 12664.5 MB | 0.1703 |

这说明当前 checkpoint 对 Qwen 推理有一定排序和粗估价值，但不能直接作为严格 OOM
边界或 `max_new_tokens` 控制器。尤其是 decode 阶段显存误差较高，prefill 阶段
运行耗时 WAPE 很高，最大显存误差达到 12GB 量级。

因此当前结论是：

- BufferNode、runtime、trace 和 controller 接入可以继续使用当前 checkpoint 做 smoke test。
- `gnn_aware` 调度可以先使用当前 checkpoint，但必须加安全 margin，并保留静态或历史 baseline。
- 若要正式声明“预测器能很好预测 Qwen 推理资源开销”，需要补 Qwen 推理验证集。
- 若要把预测器用于 `predictor_budget`、OOM 门禁或论文核心实验，应补充 Qwen
  prefill/decode/token-budget 数据并重训或校准。

## 需要补充的实验

### P0: BufferNode 真实流水线实验

#### 目标

验证真实 BufferNode runner 能否接近离线流水线模拟收益，并明确收益来自跨样本流水线、
队列解耦还是更高 GPU 利用率。

#### 对照组

| 组别 | 含义 |
|---|---|
| native_parallel | 当前 LangGraph 粗粒度 parallel workflow |
| oracle_pipeline | 20260626 中的离线流水线模拟 |
| buffered_no_gnn | 只使用 BufferNode、有界队列和固定并发 |
| buffered_eta | 使用历史 token/耗时统计提供 ETA |
| buffered_gnn | 使用 GNN 预测和 runtime 统计共同调度 |

#### 数据与配置

- 继续使用 QMSum 24 条分层样本。
- 优先复用切成 2 份和切成 3 份的 workflow。
- 保持模型、显卡、输入上限和输出上限与 20260626 实验一致。

#### 指标

- 生成阶段 total、mean、p50、p95。
- samples/min 或 items/sec。
- chunk worker busy time、idle time、blocked time。
- merge worker busy time、idle time。
- 每个 BufferNode 的 queue length、enqueue wait、dequeue wait。
- backpressure 触发次数与持续时间。
- ROUGE-L、LLM score、pass rate、空输出和 judge error。

#### 成功标准

- `buffered_no_gnn` 至少回收一部分 oracle pipeline 收益。
- `buffered_gnn` 在 total 或 p95 上优于 `buffered_no_gnn`。
- 质量指标不出现明显退化。

### P0: 有界队列与背压实验

#### 目标

验证 BufferNode 的容量限制不是形式功能，而是可以在下游变慢时限制上游生产，避免无限积压。

#### 对照组

| 组别 | 含义 |
|---|---|
| unbounded | 不限制队列容量 |
| bounded_1 | 容量为 1 |
| bounded_2 | 容量为 2 |
| bounded_4 | 容量为 4 |
| bounded_8 | 容量为 8 |

#### 实验设置

- 构造 fast producer 和 slow consumer。
- consumer 可以使用真实 merge agent，也可以先使用可控 sleep 模拟。
- producer 必须记录被下游阻塞的时间。

#### 指标

- 最大队列长度。
- 上游 blocked time。
- 下游 GPU busy/idle time。
- 端到端 total、mean、p95。
- 峰值内存占用。
- 是否有 item 丢失、重复或未 drain。

#### 成功标准

- bounded 组最大积压受容量约束。
- 队列满时上游确实暂停，而不是继续生成隐藏积压。
- drain 结束后所有已接收 item 都完成或有失败记录。

### P0: token budget 与 OOM 风险控制实验

#### 目标

验证 token budget 是资源控制变量，而不只是 prompt 文案约束。

#### 对照组

| 组别 | 含义 |
|---|---|
| fixed_high | 使用较高 `max_new_tokens`，观察 OOM 或理论 OOM 风险 |
| fixed_safe | 使用人工保守 token 上限 |
| prompt_only | 在 prompt 中要求短输出，但不改 generation 上限 |
| predictor_budget | 根据 prompt 长度、模型和设备预测 token 上限 |
| oracle_budget | 使用实测安全边界回放得到的理想上限 |

#### 数据与配置

- 使用 QMSum 长样本和 20260626 预检中的最长样本。
- 覆盖 direct summarizer、chunk agent 和 merge agent。
- 优先记录真实显存峰值；如果无法稳定复现 OOM，则使用理论显存阈值标记风险。

#### 指标

- OOM 次数或理论 OOM 次数。
- 输入上限命中、输出上限命中。
- 实际输出 token 数。
- 质量指标。
- latency 和 tokens/sec。
- predictor budget 与 oracle budget 的差距。

#### 成功标准

- `predictor_budget` 相比 `fixed_high` 显著降低 OOM 或理论 OOM 风险。
- 相比 `fixed_safe`，`predictor_budget` 保留更多有效输出或更高质量。
- `prompt_only` 不能替代 generation 上限控制。

### P0: Qwen checkpoint 验证与重训判定实验

#### 目标

验证当前 GNN checkpoint 是否足以预测 Qwen 本地推理节点的资源开销，并给出是否需要
重训或校准的明确门槛。该实验应在 `predictor_budget` 和正式 GNN-aware 调度实验前完成。

#### 对照组

| 组别 | 含义 |
|---|---|
| current_checkpoint | 当前默认 `gnn_full_retrain_20260612` checkpoint |
| current_checkpoint_margin | 当前 checkpoint 加验证集分位数 safety margin |
| qwen_retrained | 补充 Qwen 推理数据后重训的 checkpoint |
| qwen_calibrated | 不重训，只用 Qwen validation residual 做目标级校准 |
| oracle | 使用实测资源开销 |

#### 数据与配置

- 覆盖 Qwen prefill 和 decode，不把 training 指标混入主推理结论。
- 覆盖 direct summarizer、chunk agent 和 merge agent。
- 覆盖多档 `sequence_length`、`max_new_tokens`、batch size、模型大小和设备类型。
- 至少保留独立 Qwen validation/test split，不能只报告全局 test 指标。
- 若目标是 token budget，必须显式改变 `max_new_tokens` 并记录真实输出长度。

#### 采集指标

指标采集参考 `docs/spec/dcgm.md` 和 `docs/spec/dev.md`。每次节点执行至少记录：

- Workflow trace：`node_name`、`session_id`、`item_id`、phase、device、start/end、
  duration、model load/unload、queue wait、blocked duration、status/error。
- Token：prompt token、input token limit、output token、`max_new_tokens`、
  truncation hit、output limit hit。
- GPU memory：`DCGM_FI_DEV_FB_USED`、`DCGM_FI_DEV_FB_FREE`、
  `DCGM_CUSTOM_PROCESS_MEM_USED`。
- GPU utilization：`DCGM_FI_DEV_GPU_UTIL`、`DCGM_FI_DEV_MEM_COPY_UTIL`、
  `DCGM_FI_PROF_GR_ENGINE_ACTIVE`。
- SM 与 Tensor Core：`DCGM_FI_PROF_SM_ACTIVE`、`DCGM_FI_PROF_SM_OCCUPANCY`、
  `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`。
- Memory/IO pressure：`DCGM_FI_PROF_DRAM_ACTIVE`、
  `DCGM_FI_PROF_PCIE_TX_BYTES`、`DCGM_FI_PROF_PCIE_RX_BYTES`。
- Power/energy：`DCGM_FI_DEV_POWER_USAGE`、
  `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION`。
- 诊断字段：`DCGM_FI_DEV_SM_CLOCK`、`DCGM_FI_DEV_MEM_CLOCK`、
  `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS`、`DCGM_FI_DEV_XID_ERRORS`。

#### 指标

- 原始量纲 WAPE、MAE、RMSE、R2 和 max abs error。
- 按 phase、设备、模型大小、`sequence_length`、`max_new_tokens` 分组的误差。
- 显存低估率、低估 p95、低估 max error。
- OOM 风险分类的 precision、recall、false negative rate。
- `predictor_budget` 与 oracle budget 的差距。
- 加 safety margin 后的吞吐损失和 OOM 风险下降幅度。

#### 重训判定

满足任一条件时，应重训或至少做 Qwen-specific calibration：

- Qwen 推理子集 `gpu_mem_used_mb_max` WAPE 高于 `0.15`。
- Qwen decode 子集 `gpu_mem_used_mb_max` WAPE 高于 `0.15`。
- 显存低估 p95 超过可用显存 margin。
- OOM 风险 false negative rate 不能接受。
- `max_new_tokens` 改变后，预测值对显存或耗时变化不敏感。
- Qwen 子集误差显著高于全局 test 或非 Qwen test。

当前只读检查已经触发前两个条件，因此在正式资源闭环实验前，建议补充 Qwen
prefill/decode/token-budget 数据并重训；在实现联调阶段，可以继续使用当前 checkpoint
作为保守 baseline。

### P1: GNN-aware 在线资源调度实验

#### 目标

验证 GNN 预测器从离线估计进入在线 runtime 后，是否能改善有限资源下的选卡、
执行顺序和失败风险。

#### 对照组

| 组别 | 含义 |
|---|---|
| fifo | ready item 按到达顺序执行 |
| static_size | 只使用模型大小、参数量或静态图规模 |
| history_eta | 使用历史 token/耗时统计 |
| gnn_mean | 直接使用 GNN 预测均值 |
| gnn_margin | 使用 GNN 预测值加安全 margin |
| oracle | 使用实测运行成本和显存 |

#### 工作负载

- 多 session QMSum pipeline。
- 批量工具调用 workflow。
- 至少一种显存紧张配置，例如减少可用 GPU 数或人为限制可用显存。

#### 指标

- total、mean、p95 latency。
- OOM、retry、skip。
- GPU busy/idle time。
- cache hit、cold start sec。
- 平均显存利用率和峰值显存利用率。
- 调度决策与 oracle 的差距。

#### 成功标准

- `gnn_margin` 相比 `static_size` 或 `history_eta` 降低 OOM/retry。
- `gnn_margin` 不因过度保守导致明显吞吐下降。

### P1: ETA-aware prefetch 实验

#### 目标

验证上游 ETA 能否帮助下游在合适时间提前加载模型，降低输入齐全后的等待时间。

#### 对照组

| 组别 | 含义 |
|---|---|
| no_prefetch | 输入齐全后才加载模型 |
| eager_prefetch | workflow 开始时尽早加载所有可能下游模型 |
| eta_prefetch | 根据上游 ETA 和模型部署成本预取 |
| oracle_prefetch | 使用真实完成时间和真实部署耗时 |

#### 指标

- input-ready 到 node-start 的等待时间。
- 冷启动被隐藏的时间。
- 预取后未使用的模型数量。
- 预取导致的驱逐次数。
- OOM/retry。
- 端到端 p95 latency。

#### 成功标准

- `eta_prefetch` 相比 `no_prefetch` 降低 ready 后等待时间。
- `eta_prefetch` 相比 `eager_prefetch` 减少无效加载和错误驱逐。

### P1: 模型驱逐与复用实验

#### 目标

验证 workflow-aware 的模型生命周期管理是否优于 LRU 或 size-based cache。

#### 对照组

| 组别 | 含义 |
|---|---|
| no_cache | 每次执行都重新加载 |
| lru | 最近最少使用驱逐 |
| size_based | 优先驱逐显存占用大的模型 |
| deploy_cost | 优先保留重部署成本高的模型 |
| workflow_future | 结合 ETA、下游依赖和未来复用概率 |

#### 工作负载

- 多 session 混合 workflow。
- 部分 session 共享模型，部分 session 使用独占模型。
- 显存容量设置为不能同时保留所有模型。

#### 指标

- cache hit rate。
- cold start sec。
- eviction count。
- invalid eviction count，即很快又被加载回来的模型。
- OOM/retry。
- total、p95 latency。

#### 成功标准

- `workflow_future` 降低无效驱逐和冷启动时间。
- 延迟收益不能只来自保留更多显存，而应在相同容量约束下成立。

### P1: 预测误差鲁棒性实验

#### 目标

验证调度策略不是在完美预测下才有效，而是在 GNN 预测有误差时仍然可用。

#### 对照组

| 组别 | 含义 |
|---|---|
| gnn_raw | 原始 GNN 预测 |
| gnn_margin | 固定安全 margin |
| gnn_calibrated | 按验证集误差分位数校准 margin |
| perturbed_low | 人为低估耗时或显存 |
| perturbed_high | 人为高估耗时或显存 |
| oracle | 实测值 |

#### 指标

- OOM/retry 对预测误差的敏感度。
- p95 latency 对预测误差的敏感度。
- backpressure 触发是否过早或过晚。
- 预取命中率和无效预取率。
- 调度策略在不同误差水平下的退化曲线。

#### 成功标准

- 校准后的策略在低估显存时显著减少 OOM。
- 高估成本时不会让吞吐退化到明显低于保守 baseline。

### P2: 动态任务生成与批量工具调用实验

#### 目标

验证 BufferNode 不只适用于固定摘要 DAG，也适用于 agent 持续发现子任务的场景。

#### 实验设置

- planner 连续产生多个 tool task。
- tool worker 通过 BufferNode 并发处理。
- 下游 aggregator 按 session 汇总结果。

#### 指标

- task 生成速率和消费速率。
- 队列积压与背压。
- 每个 session 的完成时间。
- 工具失败、重试和跳过记录。

#### 成功标准

- 动态任务不会造成无限积压。
- session 级输出完整且可追踪。

### P2: 能耗与功耗感知实验

#### 目标

验证功耗预测是否能作为辅助目标，而不是只停留在报告指标。

#### 对照组

| 组别 | 含义 |
|---|---|
| latency_only | 只优化延迟 |
| energy_proxy | 在延迟约束下优化功耗代理 |
| balanced | 延迟、显存风险和能耗联合打分 |

#### 指标

- total energy proxy。
- p95 latency。
- OOM/retry。
- 每张 GPU 的负载分布。

#### 成功标准

- 在 p95 latency 不明显变差的前提下降低能耗代理。

## 推荐执行顺序

| 阶段 | 实验 | 目的 |
|---|---|---|
| A | BufferNode 真实流水线实验 | 验证最核心吞吐收益 |
| A | 有界队列与背压实验 | 验证 BufferNode 运行时语义 |
| B | Qwen checkpoint 验证与重训判定实验 | 验证预测器是否足以进入资源闭环 |
| B | token budget 与 OOM 风险控制实验 | 验证生成式节点资源闭环 |
| B | GNN-aware 在线资源调度实验 | 验证预测器进入 runtime 后的收益 |
| C | ETA-aware prefetch 实验 | 验证节点联动和模型预取 |
| C | 模型驱逐与复用实验 | 验证模型生命周期管理 |
| D | 预测误差鲁棒性实验 | 验证策略不是 oracle-only |
| D | 动态任务生成与批量工具调用实验 | 扩展到更接近 Agent Workflow 的场景 |
| D | 能耗与功耗感知实验 | 作为辅助方向 |

最小闭环应优先完成 A 和 B。若 A 不能取得明显收益，后续 prefetch、eviction 和
energy 实验的优先级应降低。

## 统一记录字段

后续所有实验应尽量记录以下字段，避免每轮实验重新补日志：

- `workflow_name`
- `sample_id`
- `session_id`
- `item_id`
- `node_name`
- `source_node`
- `enqueue_at`
- `dequeue_at`
- `started_at`
- `ended_at`
- `duration_sec`
- `queue_size_before`
- `queue_size_after`
- `blocked_duration_sec`
- `input_token_count`
- `output_token_count`
- `max_new_tokens`
- `prediction_metrics`
- `schedule_decision`
- `device`
- `gpu_name`
- `phase`
- `sequence_length`
- `truncation_hit`
- `output_limit_hit`
- `model_load_started_at`
- `model_load_ended_at`
- `model_unload_at`
- `gpu_mem_used_mb`
- `gpu_mem_free_mb`
- `gpu_util_percent`
- `gpu_mem_copy_util_percent`
- `gpu_sm_active_percent`
- `gpu_sm_occupancy_percent`
- `gpu_tensor_active_percent`
- `gpu_dram_active_percent`
- `pcie_tx_bytes_per_sec`
- `pcie_rx_bytes_per_sec`
- `gpu_power_watts`
- `gpu_energy_joules`
- `gpu_sm_clock_mhz`
- `gpu_mem_clock_mhz`
- `gpu_clock_throttle_reasons`
- `gpu_xid_errors`
- `status`
- `error_type`
- `error_message`

## 判断标准

当前项目是否继续推进为完整系统，应优先看以下结果：

- 真实 BufferNode 能否回收 20260626 oracle pipeline 的主要收益。
- Qwen 推理资源预测误差是否足以支撑 token budget、OOM 门禁和在线调度。
- GNN 控制是否比静态启发式更稳定地降低 OOM、retry、冷启动和 p95 latency。
- token budget 是否能降低资源风险且不显著伤害质量。
- 预测误差存在时，调度策略是否仍优于保守 baseline。

如果这些实验无法成立，系统应收缩为工程插件和 workflow trace 分析工具，不应继续把主线定义为预测驱动的 Agent Workflow runtime。
