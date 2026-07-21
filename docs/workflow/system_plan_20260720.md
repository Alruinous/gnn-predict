# 多智能体协作调度系统重构方案（20260720）

本文记录多 workflow 调度器与资源契约重构的最终方案与落地结果。规划阶段的完整讨论稿（含
逐条决策记录）在 Claude Code 会话中产出；本文是收敛后的定稿版本，替代那份会话内文档。

## 背景

`dev.md`（仓库根目录设计纲要）的核心诉求是"基于工作流负载感知的模型生命周期与资源管理"。
三路调研（`src/workflow/` 代码、`docs/workflow/*` 历史决策文档、`paper/hpca2027-sagepilot/`
论文草稿——由同学根据本项目代码撰写，参考价值较大但不是需要代码反向对齐的外部规范）交叉
确认：多个 workflow 并发共享一个调度器/GPU 池/模型实例池，在当时的代码、测试、历史实验
里都不存在——`Workflow` schema 没有身份字段，`SchedulerCore`/`WorkflowController` 从构造
函数开始就只认一个 `Workflow` 对象。这是唯一被交叉验证过"确实是空白"的地方。

同时，07-13 号端到端实验（唯一一次真实调度器在环的统计对比，见
[system_experiment_results_20260713.md](system_experiment_results_20260713.md)）证明：
`cache` 策略在单 workflow 内提高并发负载时，尾延迟明显恶化，根因是"省下的驻留时间变成了
应用层排队"。该实验只在单节点、单 GPU 型号（V100）、多进程隔离下跑过，没有测试过 V100+
A100 组合缓存或分布式平台。多 workflow 并发从调度器视角看是同类压力，必须显式设计公平性
机制，不能假设"支持多 workflow"会自动带来好的多 workflow 表现。

## 范围决策

1. 目标资源池形态：单个 Ray 集群、V100+A100 异构 GPU 混合池；不做跨节点分布式。
2. 采用论文提出的"资源契约" `K(c) = ⟨ŷ, L, U, ρ, Ω, a⟩` 抽象作为调度决策的设计语言：
   点估计、校准下界、校准上界、置信证据、适用范围、运行时动作；核心原则是"资源知识的
   精细度匹配决策风险"——软决策用点估计，硬性安全决策（OOM 防护）用校准上界。
3. 本轮不接入 GNN 预测器，只保证契约 schema 对预测来源无感知（`ResourceContractSource`
   预留 `gnn_predicted`，不产出）。
4. 顺带清理已识别的代码质量问题：命名冲突模块、`ModelReplica`/`StaticVLLMEngine` 重复
   代码、缺失的 `compose_profile_cache.py`。

## 架构：哪些状态保持全局共享

`SchedulerCore` 的 `accelerators`、`replicas`、`_replica_pairs`、`oom_penalties`、
`pending_acquires`、`grants`、`tasks`、`sessions`、`load_history` 都不按 workflow 拆分。
`ModelDeploymentConfig.model_key` 本来就与 workflow 身份无关（哈希 backend 版本、模型名、
模型路径、dtype、完整 `ServingConfig`），两个 workflow 请求字节级相同的部署配置时就已经
能共享一个副本——这是共享池设计"免费"获得的能力，不需要额外代码。`model_key` 比 dev.md
伪代码里的简化三元组（模型名+最大 token 长度+GPU 类型）更严格：共享的副本必须对所有消费者
行为一致（`max_num_seqs`/`max_model_len` 等），放宽会让一个 workflow 悄悄继承另一个
workflow 的服务限制。

`session_id`/`task_id`/`acquire_id` 是 fleet 范围内的全局唯一 UUID，不是每个 workflow
各自独立的命名空间——这是一个需要注意的既有约束，不是本轮引入的限制。

## 架构：新增的编排层

`WorkflowFleet`（`src/workflow/fleet.py`）拥有 Ray 进程生命周期、共享 `SchedulerConfig`、
共享 `_SchedulerActor`。`register_workflow` 校验+建队列+起 worker 的实现体基本是原
`WorkflowController.start()` 里"每 workflow 一份"的那半段逻辑，区别是最后调用共享
scheduler 的 `register_workflow`/`register_workers`，而不是新起一个 scheduler；调度器的
`.run()` 只在 fleet 第一次成功注册后启动一次。某个 workflow 注册失败时的回滚只清理这个
workflow 自己的 actor/队列/目录，不牵连整个 Ray 运行时或已注册的其他 workflow。

`WorkflowController` 保留完全不变的公开签名，内部改成持有一个私有的、只注册自己这一个
workflow 的 `WorkflowFleet`，逐方法委托。`tests/test_workflow_controller.py` 全套用例
零改动通过，验证了 facade 行为保持不变。

drain/停止语义拆成两条独立路径（详见 [implementation.md](implementation.md) "多 workflow 编排" 一节）：
`drain_workflow(name)` 只影响这一个 workflow；`shutdown()` 才会驱逐全部副本、停止共享
调度循环、关闭 Ray。这是最容易做错的地方——如果 `evict_all`/`stop_loop` 被单个 workflow
的 drain 触发，会把其他 workflow 还在用的热副本和调度循环一起关掉。

## 跨 workflow 公平性

07-13 号实验暴露的问题本质是"副本复用把突发需求集中到一个副本的内部队列上，节省下来的
驻留时间变成了应用层排队"。公平性机制控制的是**谁排在谁前面**，不会减少**总排队延迟**在
聚合过载下的存在——这一点需要如实记录，不能暗示"支持多 workflow 顺带解决了尾延迟问题"。

复用一个旋钮解决两个问题：每个已注册 workflow 一个 `priority_weight`（默认 1.0）。

1. **准入排序**：按 `sessions[task.session_id].workflow_name` 把待处理请求分组，组内
   继续用现有策略排序规则不变，组间按 `priority_weight` 加权轮转交织
   （`SchedulerCore._interleave_by_workflow`）。权重相等时退化为按 workflow 轮转——即使
   不配置任何权重，也比"一个全局 `request_seq` 计数器"更公平。
2. **驱逐保护**：对每个贡献需求的 workflow 来源单独加权，`distance_i / weight_i` 换算
   有效距离，再取所有来源的最小值（`_future_reuse_distance`）。`select_eviction_victim`
   的函数签名和打分公式不变，只是收到的 `reuse_distance_sec` 已经是加权后的结果。

评估但未默认采用：按 workflow 硬性预留加速卡。能给出硬隔离保证，代价是被预留但空闲的卡
不能被其他 workflow 借用，直接违背共享池的初衷；定位为面向真正 SLA 关键场景的小众可选项，
留作后续如果加权交织不够用时的备选，当前未实现。

## 资源契约落地

`artifacts.py`：`PredictionEntry`→`ResourceContract`，`PredictionCache`→
`ResourceContractCache`，新增 `ResourceContractSource`（`empirical_profile`/
`synthetic_fixture`/`gnn_predicted`）与 `ResourceEvidence`（`method`+`sample_count`+可选
`margin_fraction`）。`peak_vram_mb_upper_bound`+`peak_vram_mb_evidence` 必填，
`run_sec_upper_bound`+`run_sec_evidence` 可选（`run_sec` 的两个消费点都是软决策，不强制
置信区间）。新增 `validate_bounds` 校验器，fail fast 拒绝 `U < ŷ` 的不一致数据。

`policy.py::_feasible_placement` 与 `scheduler.py::_admission_prediction`（已并发副本上
第 k+1 个请求的批量准入复核）的硬性内存闸门统一改用 `peak_vram_mb_upper_bound`；排序
tiebreak（`select_placement` 的 bin-packing、预取 ETA）仍用 `predicted_run_sec` 点估计
不变。

置信证据 `ρ` 的落地是诚实但保守的：`profile_workflow_cache.py` 对 `run_sec`/`peak_vram`/
`power` 只采样一次，真正有统计意义的置信区间需要重复采样，而 `config/workflow/cache.yaml`
网格约几千个点/GPU 型号，重复采样是实打实的 GPU 时间开销。本轮默认用
`fixed_margin_fallback`（`peak_vram_mb_upper_bound = predicted_peak_vram_mb * 1.10`），
如实标注不是真正校准过的区间。提高采样次数、计算真正的残差分位数，留作后续可单独排期的
任务（GPU 时间成本需要单独评估）。`src/experiment/workflow/cache.py` 的 synthetic
fixture 没有真实测量噪声，如实设 `peak_vram_mb_upper_bound = predicted_peak_vram_mb`
（零余量）、`method="point_estimate_only"`。

动作 `a`（Use/Fallback/Defer/...）不存进缓存，运行时现算：`select_placement` 是纯函数，
只区分 `USE`/`FALLBACK`；`DEFER` 是 `tick_once` 层面的概念，作为 trace 标注记录，不硬塞进
`PlacementDecision`。

`scripts/workflow/compose_profile_cache.py`（新增）把多个单一 GPU kind 的
`ResourceContractCache` 合并成异构池缓存：每个来源只保留声明 `gpu_kind` 对应的条目，
来源间 `gpu_kind` 重复或 `version` 不一致时 fail fast，合并结果按
`(gpu_name, model_name, phase, batch_size, sequence_length, decode_output_length)`
确定性排序。这个脚本不需要感知契约里 `U`/`ρ`/`source` 字段的具体内容，架构上与"资源契约"
改造彻底解耦。

## 代码清理

- `workflow/model.py`（PT2/FX 图捕获）→ `src/gnn_model/data/causal_lm_graph.py`；
  `workflow/cache_config.py` → `src/gnn_model/data/causal_lm_cache_config.py`（与已有的
  `fx_graph.py` 是姊妹关系，不合并——`fx_graph.py` 是通用 PT2/FX→PyG 转换，
  `causal_lm_graph.py` 是 causal LM 专属的捕获编排）。`workflow/profile.py`（零引用死
  代码）→ `src/common/profile.py`。
- `ModelReplica`/`StaticVLLMEngine` 抽出共享基类 `GenerationEngine`
  （`load`/`invoke`/`abort`/`shutdown`/`get_stats`），`enforce_capacity` 作为
  `ClassVar[bool]` 区分二者：`ModelReplica` 有调度器准入控制在前，`enforce_capacity=True`；
  `StaticVLLMEngine` 是基线路径，没有调度器准入控制、依赖 vLLM 自身 batching，
  `enforce_capacity=False`——这是刻意的行为差异，不是需要补齐的缺口，落地前用测试证据
  （`test_actor_forwards_concurrent_requests_without_capacity_rejection` 等）确认过。
- 三份独立实现的 actor-or-plain-object 适配器（`controller.py`、`worker.py`、
  `scheduler.py` 各自的"有 `.remote()` 就调，没有就直接调"模式）收敛进
  `src/workflow/actor_support.py`（`dispatch`/`resolve`/`await_value`/`invoke`）。

## 明确不做的事

- GNN 预测器接入缓存产出链路（本轮只保证 schema/接口对预测来源无感知）。
- HTTP 对外请求接口（单/批量打到某 workflow）——仍不做。
  **更新（0721）**：`submit` 之上的**实验驱动 master**（`src/workflow/master.py`，`python -m
  workflow.master`，跑完即停、无 HTTP）已落地，含 Crater DDP 启动脚本与加速卡自动发现，见
  `docs/workflow/implementation.md` 的"master 驱动与实验部署"。本轮补的是脚本化实验运行，
  不是 HTTP 常驻服务。
- 跨节点分布式 Ray 集群。
- `predicted_power_watts` 参与调度决策。
- 按 workflow 硬性预留加速卡（设计已给出，默认不启用）。
- 提高 `profile_workflow_cache.py` 采样次数以获得真正校准过的置信区间。

## 验证

全量测试（`ruff check`/`ty check`/`pytest`）在每个里程碑后保持绿——4 个失败用例
（`test_gemma4_arch_configs_expand_to_expected_counts`、3 个 `test_monitor_cli.py` 用例）
在本轮改动之前就存在，与本次改动无关，不在本次范围内修复。新增测试文件：
`tests/test_workflow_actor_support.py`、`tests/test_workflow_scheduler_multiworkflow.py`、
`tests/test_workflow_fleet.py`、`tests/test_workflow_compose_profile_cache.py`、
`tests/test_workflow_multi_workflow_demo.py`。

`config/workflow/multi_workflow_demo/` 下有两个小 workflow（`quick_qa.yaml` 用 Qwen3-4B
配 v100，`long_report.yaml` 用 Qwen3-32B 配 a100）、一份跨 hostname/gpu_kind 的
`scheduler_config.yaml`、一份覆盖两个模型的 `predictions.yaml`，留给未来在真实 V100+A100
节点上做一次冒烟测试。规划阶段没有真实 GPU 可用，`tests/test_workflow_fleet.py` 用真实
Ray + 纯函数节点验证了两个 workflow 共享一个调度器/GPU 池、注册失败互不影响、drain 互不
干扰的核心编排语义。
