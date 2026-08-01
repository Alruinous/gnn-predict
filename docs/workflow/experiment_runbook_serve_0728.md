# 实验 Runbook：serve_0728 六因子矩阵（2026-07-28）

集群开关由人工执行，本文只规定**每个因子对应哪个开关**、**跑哪些 run**、**每个 run 怎么验收**。
实验本身不由本文的作者执行。

系统侧的改造已经落在 `main`：`SchedulerConfig.enable_prefetch`、
`SchedulerConfig.cross_workflow_lifecycle`、`run_manifest.json`、
`config/workflow/serve/serve_0728/` 与 `scripts/workflow/run_serve_0728.sh`。

---

## 1. 因子与开关的对应

| 因子 | 取值 | 落到哪里 |
|---|---|---|
| 到达模式 | `burst` / `poisson_r050` / `poisson_r025` / `poisson_r0125` | `config/workflow/serve/serve_0728/replay_<ARRIVAL>.yaml` |
| 加速卡环境 | `a1v2` / `a1v3` / `a1v4` | `RAY_PORT` 选集群，`master.discover_accelerators` 自动发现整台 head 的卡 |
| 预测算法 | SageRadar / Analytical / GBDT | `--predictions` 指向 `cache/gnn_v2` / `cache/static` / `cache/tabular` |
| 节点融合 | 开 / 关 | `master.py --fuse-nodes` |
| 预取 | 开 / 关 | `enable_prefetch`（`scheduler_sagepilot_noprefetch.yaml`） |
| 跨 workflow 生命周期 | 开 / 关 | `cross_workflow_lifecycle`（`scheduler_sagepilot_noxwf.yaml`） |

`ARM` 把后四个因子压成一个名字，映射写死在 `run_serve_0728.sh` 里：

| ARM | scheduler yaml | `--predictions` | `--fuse-nodes` | 相对参考格改了什么 |
|---|---|---|---|---|
| `sagepilot` | `scheduler_sagepilot` | `cache/profile_v2` | ✓ | —（参考格） |
| `analytical` | `scheduler_sagepilot` | `cache/static` | ✓ | 预测器换成解析公式 |
| `gbdt` | `scheduler_sagepilot` | `cache/tabular` | ✓ | 预测器换成梯度提升树 |
| `nofuse` | `scheduler_sagepilot` | `cache/profile_v2` | ✗ | 去掉节点融合 |
| `noprefetch` | `scheduler_sagepilot_noprefetch` | `cache/profile_v2` | ✓ | 去掉预取 |
| `noxwf` | `scheduler_sagepilot_noxwf` | `cache/profile_v2` | ✓ | 生命周期按 workflow 分开决策 |

三个消融臂与参考格共用同一份预测缓存，这是硬性要求：每个臂必须只与参考格差一个机制，
否则差异无法归因到该机制。`analytical` 与 `gbdt` 换缓存，因为预测算法本身就是它们的自变量。
| `parrot` | `scheduler_parrot` | `cache/profile` | ✗ | 先到先服务排序基线 |
| `kairos` | `scheduler_kairos` | `cache/profile` | ✗ | 最短剩余关键路径排序基线 |

两个基线用 `cache/profile`（原始实测）而不是 `profile_v2`/`gnn_v2`：后者含本项目自己的
加载时长校准，白送给基线不合适。它们仍然需要缓存做显存可行性和批处理准入，跑不掉。

### 三个开关各自关掉了什么

- **`enable_prefetch: false`** 只门住 `tick_once` 里的预取动作。near-ready 集合照常计算，
  因为复用距离驱逐、扩容可行性预留和装载收益排序都读它——把集合整个清空会顺带把驱逐降级成
  静态深度回退，这一臂就不再是"只去掉预取"。
- **`cross_workflow_lifecycle: false`** 收敛三处决策输入到单条 workflow：复用距离只看属主
  workflow 的需求、准入退回按 workflow 加权公平交错（局部性排序下沉到每条 workflow 内部）、
  装载收益按 workflow 计需求。**副本仍然物理共享**（`model_key` 不变），因为拆键会把这套
  负载的 5 个模型键变成 11 个（5+4+2），池子装不下。可行性预留
  （`_demand_requirements` / `_scale_out_preserves_feasibility`）刻意不收窄。
- **`--fuse-nodes`** 是注册前的图重写，只对 `chain_qmsum` 有作用（10 节点 → 4 节点，
  即每会话 3 个融合节点）。另外两条工作流没有相邻同模型边，融合与否完全一样。

---

## 2. 工作负载（沿用 eval_20260727，不改）

三条工作流各 20 会话共 60 个：

| 工作流 | 图形状 | 模型 |
|---|---|---|
| `moa_gsm8k` | 分发 → 5 个求解 → 投票 → 精修 | 0.6B / 1.7B / 4B / 8B / 14B |
| `repair_mbpp` | 写码 → 测试 → 诊断 → 修复 → 再测 | 14B / 1.7B / 4B / 8B |
| `chain_qmsum` | 分发 → 两条 3 段滚动摘要 → 合并 → 补充 → 定稿 | 4B / 8B |

三条共解析出 **5 个共享 `model_key`**。Qwen3-14B 在 8192 上下文下装不进 V100，只能落 A100，
所以每个集群至少要有 1 张 A100。

**种子固定 `arrival_seed: 42`，三轮重复不改种子。** `poisson_arrival_offsets` 的 RNG 只由
`(seed, "arrival", workflow_name)` 决定，所以全部 288 个 run 共享同一条到达序列——臂之间是
成对（paired）对照，差异不含到达抖动；三轮重复量的是系统自身的非确定性。

三档泊松更进一步：`expovariate(λ) = -ln(U)/λ` 消耗同一条 U 序列，所以
`replay_poisson_r0125` / `r025` / `r050` 是**同一条到达序列的三种速度**，不是三次不同的抽样。

---

## 3. 硬约束

1. **必须从主 checkout `/home/wangjh/gnn_predict` 提交。** 常驻 Ray 的 `_SchedulerActor` 和
   `NodeWorker` 没有 `runtime_env`，永远 import 主 checkout 的 `src`。
2. **开跑前必须重启三个 head 的常驻 actor。** 本次给 `SchedulerConfig` 加了
   `enable_prefetch` 和 `cross_workflow_lifecycle` 两个字段；旧 actor 的类定义不认，
   不重启会在 tick 中途挂掉。
3. **一个集群同时只能跑一个 run**（每个 run 会 `evict_all` 共享 GPU 池）。并行靠三个端口。
4. **同一硬件格的所有 run 必须用同一批物理卡**，否则不可比。中途重组过卡，该格全部重跑。
5. **不要改 `acquire_timeout_sec`。** `_order_pending` 和 `_order_by_load_yield` 都用
   `0.5 * acquire_timeout_sec` 当饥饿阈值，改它会同时把 cache 策略的老化行为重新调参。
   全部臂统一 1800.0。

### 集群清点（`ray.nodes()` 实查，2026-07-28）

| 端口 | 卡 | 代号 | `--min-gpus` |
|---|---|---|---|
| 6661 | 1×A100 + 2×V100 | `a1v2` | 3 |
| 6662 | 1×A100 + 3×V100 | `a1v3` | 4 |
| 6663 | 1×A100 + 4×V100 | `a1v4` | 5 |

`run_serve_0728.sh` 从 `RAY_PORT` 推出 `HW` 与 `--min-gpus`，端口不在这三个里直接退出。

---

## 4. 规模与机时

满矩阵 **8 臂 × 4 到达 × 3 硬件 × 3 轮 = 288 个 run**，每个集群 96 个串行。

| 到达 | 到达跨度 | 单 run 墙钟（估） | 每集群 24 run |
|---|---|---|---|
| `burst` | 0s | ~15 min | ~6.0 h |
| `poisson_r050` | ~460s | ~18 min | ~7.2 h |
| `poisson_r025` | ~918s | ~20 min | ~8.0 h |
| `poisson_r0125` | ~1836s | ~35 min | ~14.0 h |

**每个集群约 35 小时，三集群并行即约 35 小时墙钟。** 要压缩就砍到达维度：
`burst` 有 makespan 区分度，`poisson_r025` 可与 eval_20260727 直接对照，这两档优先。

---

## 5. 输出布局

```
output/serve_0728/
  a1v2__burst__sagepilot/
    r1/  run_manifest.json  run_summary.json  workflow_trace.jsonl  <每条workflow一个子目录>
    r2/
    r3/
  a1v2__burst__noprefetch/
  ...
  logs/a1v2__burst__sagepilot__r1.log
```

目录名 `<hw>__<arrival>__<arm>` 自带三个因子；余下三个因子（预测缓存、融合、两个开关）
在每个 run 的 `run_manifest.json` 里，字段包括 git commit、解析后的 `scheduler_config`
（**含发现到的 accelerators，即硬件指纹**）、预测缓存路径与 sha256、工作流文件与 sha256、
`fuse_nodes`、实验配置与 `arrival_seed`。分析时按清单里的因子归组，不要靠目录名。

---

## 6. 每个 run 的验收

`run_summary.json` 里：

- `completed_session_count == 60`、`failed_session_count == 0`
- `request_infeasible_count == 0`、`oom_count == 0`

`workflow_trace.jsonl` 里（开关真的接上了的直接证据）：

| 臂 | 必须成立 |
|---|---|
| `noprefetch` | 没有 `prefetch_started` 事件；`prefetch_skipped` 的 `reason` 全是 `disabled` |
| 其余 cache 臂 | 从不出现 `reason="disabled"` 的 `prefetch_skipped` |
| 带 `--fuse-nodes` 的臂 | `task_execution_finished` 里带 `stages` 的有 60 次（20 个 `chain_qmsum` 会话 × 3 个融合节点） |
| `nofuse` / `parrot` / `kairos` | 带 `stages` 的执行为 0 |
| `noxwf` | `model_load_started` 的 `owner_workflow` 有多条 workflow；副本仍被跨 workflow 复用（同一 `replica_id` 上相邻的 `acquire_granted` 出现不同 `workflow_name`）；驱逐决策的 `reuse_distance_sec` 只反映属主 workflow 的需求（`null` 表示范围内无人再要该模型） |

驱逐相关的 `scheduler_decision` 事件带 `owner_workflow` / `reuse_distance_sec` /
`reload_cost_sec`，这是复用距离驱逐唯一可审计的出口——`reuse_distance_sec` 为 `null`
表示原始值是 `inf`（JSON 里 `Infinity` 不是合法 token，严格读取器会拒绝）。

任何一条不成立，该 run 作废重跑，不要进表。

**不要拿 `prefetch_started > 0` 当验收条件。** 实测（`a1v2__burst__sagepilot/r1`）该值为 0：
涌入负载下 60 个会话把 5 个模型全部顶成常驻，near-ready 任务要的模型已经在卡上，
预取无事可做（`prefetch_skipped` 里 29 次 `replica_resident`、7 次 `already_loading`）。
这是设计行为——预取在 `tick_once` 里优先级最低，只在没有就绪工作需要装载时才出手。
区分 `noprefetch` 与其他臂的证据是 `reason="disabled"`，只有该臂会发。

---

## 7. 提交命令

每条是一条自包含单行命令，一条 = 一个 run，逐条提交，不要用 for 循环。
同一集群内串行；三个集群之间可并行。

### 集群 6661（a1v2，1×A100 + 2×V100）

#### a1v2 · 涌入

```sh
ARM=sagepilot ARRIVAL=burst RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=sagepilot ARRIVAL=burst RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=sagepilot ARRIVAL=burst RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=burst RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=burst RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=burst RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=burst RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=burst RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=burst RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=burst RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=burst RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=burst RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=burst RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=burst RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=burst RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=burst RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=burst RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=burst RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=burst RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=burst RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=burst RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=burst RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=burst RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=burst RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
```

#### a1v2 · 泊松 0.050/wf

```sh
ARM=sagepilot ARRIVAL=poisson_r050 RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=sagepilot ARRIVAL=poisson_r050 RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=sagepilot ARRIVAL=poisson_r050 RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r050 RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r050 RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r050 RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r050 RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r050 RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r050 RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r050 RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r050 RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r050 RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r050 RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r050 RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r050 RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r050 RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r050 RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r050 RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r050 RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r050 RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r050 RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r050 RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r050 RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r050 RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
```

#### a1v2 · 泊松 0.025/wf

```sh
ARM=sagepilot ARRIVAL=poisson_r025 RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=sagepilot ARRIVAL=poisson_r025 RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=sagepilot ARRIVAL=poisson_r025 RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r025 RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r025 RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r025 RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r025 RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r025 RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r025 RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r025 RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r025 RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r025 RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r025 RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r025 RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r025 RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r025 RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r025 RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r025 RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r025 RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r025 RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r025 RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r025 RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r025 RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r025 RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
```

#### a1v2 · 泊松 0.0125/wf

```sh
ARM=sagepilot ARRIVAL=poisson_r0125 RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=sagepilot ARRIVAL=poisson_r0125 RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=sagepilot ARRIVAL=poisson_r0125 RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r0125 RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r0125 RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r0125 RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r0125 RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r0125 RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r0125 RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r0125 RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r0125 RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r0125 RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r0125 RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r0125 RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r0125 RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r0125 RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r0125 RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r0125 RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r0125 RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r0125 RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r0125 RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r0125 RAY_PORT=6661 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r0125 RAY_PORT=6661 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r0125 RAY_PORT=6661 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
```

### 集群 6662（a1v3，1×A100 + 3×V100）

#### a1v3 · 涌入

```sh
ARM=sagepilot ARRIVAL=burst RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=sagepilot ARRIVAL=burst RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=sagepilot ARRIVAL=burst RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=burst RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=burst RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=burst RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=burst RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=burst RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=burst RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=burst RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=burst RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=burst RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=burst RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=burst RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=burst RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=burst RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=burst RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=burst RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=burst RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=burst RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=burst RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=burst RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=burst RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=burst RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
```

#### a1v3 · 泊松 0.050/wf

```sh
ARM=sagepilot ARRIVAL=poisson_r050 RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=sagepilot ARRIVAL=poisson_r050 RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=sagepilot ARRIVAL=poisson_r050 RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r050 RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r050 RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r050 RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r050 RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r050 RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r050 RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r050 RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r050 RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r050 RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r050 RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r050 RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r050 RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r050 RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r050 RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r050 RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r050 RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r050 RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r050 RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r050 RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r050 RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r050 RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
```

#### a1v3 · 泊松 0.025/wf

```sh
ARM=sagepilot ARRIVAL=poisson_r025 RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=sagepilot ARRIVAL=poisson_r025 RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=sagepilot ARRIVAL=poisson_r025 RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r025 RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r025 RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r025 RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r025 RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r025 RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r025 RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r025 RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r025 RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r025 RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r025 RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r025 RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r025 RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r025 RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r025 RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r025 RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r025 RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r025 RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r025 RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r025 RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r025 RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r025 RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
```

#### a1v3 · 泊松 0.0125/wf

```sh
ARM=sagepilot ARRIVAL=poisson_r0125 RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=sagepilot ARRIVAL=poisson_r0125 RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=sagepilot ARRIVAL=poisson_r0125 RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r0125 RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r0125 RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r0125 RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r0125 RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r0125 RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r0125 RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r0125 RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r0125 RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r0125 RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r0125 RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r0125 RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r0125 RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r0125 RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r0125 RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r0125 RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r0125 RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r0125 RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r0125 RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r0125 RAY_PORT=6662 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r0125 RAY_PORT=6662 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r0125 RAY_PORT=6662 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
```

### 集群 6663（a1v4，1×A100 + 4×V100）

#### a1v4 · 涌入

```sh
ARM=sagepilot ARRIVAL=burst RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=sagepilot ARRIVAL=burst RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=sagepilot ARRIVAL=burst RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=burst RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=burst RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=burst RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=burst RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=burst RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=burst RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=burst RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=burst RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=burst RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=burst RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=burst RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=burst RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=burst RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=burst RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=burst RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=burst RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=burst RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=burst RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=burst RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=burst RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=burst RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
```

#### a1v4 · 泊松 0.050/wf

```sh
ARM=sagepilot ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=sagepilot ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=sagepilot ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
```

#### a1v4 · 泊松 0.025/wf

```sh
ARM=sagepilot ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=sagepilot ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=sagepilot ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r025 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
```

#### a1v4 · 泊松 0.0125/wf

```sh
ARM=sagepilot ARRIVAL=poisson_r0125 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=sagepilot ARRIVAL=poisson_r0125 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=sagepilot ARRIVAL=poisson_r0125 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r0125 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r0125 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=analytical ARRIVAL=poisson_r0125 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r0125 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r0125 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=gbdt ARRIVAL=poisson_r0125 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r0125 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r0125 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=nofuse ARRIVAL=poisson_r0125 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r0125 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r0125 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=noprefetch ARRIVAL=poisson_r0125 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r0125 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r0125 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=noxwf ARRIVAL=poisson_r0125 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r0125 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r0125 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=parrot ARRIVAL=poisson_r0125 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r0125 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r0125 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0728.sh
ARM=kairos ARRIVAL=poisson_r0125 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0728.sh
```
