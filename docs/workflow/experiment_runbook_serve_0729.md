# serve_0729 提交手册

`output/serve_0729` 是为 GPU 时间分布图重跑的一组实验。它与 `output/serve_0728` 的区别只有一处，
但很关键：**这里的 GBDT 和 Formula 是独立基线，不是 SagePilot 换预测器**。

配套文档：`docs/workflow/serve_0728_reference.md` 解释了工作负载、目录命名、trace 事件和结果
文件的含义，那些内容 serve_0729 完全沿用，本文不重复。

---

## 1. 与 serve_0728 的差别

| | serve_0728 | serve_0729 |
|---|---|---|
| GBDT / Formula 的含义 | SagePilot 换掉预测器（消融项） | 独立基线：不用 SagePilot 任何调度机制 |
| 显卡环境 | a1v2、a1v3、a1v4 三种 | 只有 a1v3（1×A100 + 3×V100） |
| 到达方式 | 四种 | 先做 burst 与 poisson_r050 两种 |
| 方案数 | 8 | 5 + 2 |
| 重复次数 | 3 | 3 |

工作流定义、到达配置、随机种子、模型与服务参数**完全相同**：
到达配置直接引用 `config/workflow/serve/serve_0728/replay_*.yaml` 原文件而不是复制，所以
`run_manifest.json` 里记录的文件哈希可以证明两个数据集投递的会话序列逐字节一致
（`arrival_seed: 42`，每次运行 60 个会话）。

---

## 2. 七个方案

映射写死在 `scripts/workflow/run_serve_0729.sh` 的 case 表里，是唯一的真相来源；
`tests/test_serve_0729_configs.py` 钉住这张表。

### 阶段一：四个基线 + 完整系统（30 次运行）

| 方案代号 | 中文 | 调度器配置 | 预测数据 | 节点融合 |
|---|---|---|---|---|
| `parrot` | Parrot 排序基线 | `scheduler_baseline_fifo` | `cache/profile` | 关 |
| `kairos` | Kairos 排序基线 | `scheduler_kairos` | `cache/profile` | 关 |
| `formula_baseline` | 解析公式预测器基线 | `scheduler_baseline_fifo` | `cache/static` | 关 |
| `gbdt_baseline` | 梯度提升树预测器基线 | `scheduler_baseline_fifo` | `cache/tabular` | 关 |
| `sagepilot` | 完整系统 | `scheduler_sagepilot` | `cache/profile_v2` | 开 |

前四个方案一律**没有**预取、没有代价感知驱逐、没有装载收益排序、没有多开实例、没有防抖、
没有节点融合。这不是靠开关关掉的，而是 `policy` 不等于 `cache` 时这些机制在代码里根本没有输入：

| 机制 | 在 `src/workflow/scheduler.py` 里为什么失效 |
|---|---|
| 预取 | `_collect_near_ready_tasks` 对 `fifo`/`kairos` 直接返回空列表 |
| 代价感知驱逐 | `_reload_cost` 对 `fifo`/`kairos` 返回 `None` |
| 装载收益排序 | `_order_by_load_yield` 非 `cache` 策略时原样返回输入 |
| 就近优先准入 | `_order_pending` 的 locality-first 分支只在 `cache` 策略下进入 |
| 多开实例 | `SchedulerConfig.validate_elastic_policy` 禁止 `elastic_replicas` 与非 `cache` 策略共存 |

`parrot`、`formula_baseline`、`gbdt_baseline` **共用同一份调度器配置**
（`scheduler_baseline_fifo.yaml`），所以这三者之间**只差预测数据**一项。

预测数据在最简调度下仍然实质影响结果，不是摆设：`src/workflow/policy.py::select_placement`
用 `peak_vram_mb_upper_bound` 判断某模型能否放进某张卡。实测 8192 序列桶、批 1 时 Qwen3-4B 的
上界为 `profile`/`profile_v2` 33227 MB、`tabular` 32862 MB（**都超过 32768 MB 的 V100**）、
`static` 11613 MB（**放得下**）。换预测数据会改变 4B 落在哪种卡上。

### 阶段二：同两个预测器的「含 SagePilot 机制」版本（12 次运行）

| 方案代号 | 中文 | 调度器配置 | 预测数据 | 节点融合 |
|---|---|---|---|---|
| `formula_with_sagepilot` | 解析公式预测器 + 完整调度 | `scheduler_sagepilot` | `cache/static` | 开 |
| `gbdt_with_sagepilot` | 梯度提升树预测器 + 完整调度 | `scheduler_sagepilot` | `cache/tabular` | 开 |

这两个方案就是 serve_0728 里 `analytical` 和 `gbdt` 的定义，搬到同一批卡上重跑。
配上阶段一，同一个预测器有了两个读数，可以把「换预测器值多少」和「换整套调度值多少」分开。

**阶段一全部跑完并确认合格后才启动阶段二**，两者不并行（一个集群同一时刻只能跑一次实验）。

---

## 3. 提交命令

一条命令一次运行，不写循环。四个变量写错会立即报错退出，不会静默用默认值。

```sh
ARM=parrot ARRIVAL=burst RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0729.sh
```

结果落到 `output/serve_0729/<显卡环境>__<到达方式>__<方案>/r<REPEAT>/`，
日志落到 `output/serve_0729/logs/`。

**显卡环境这一段不再由端口推断，而是开跑前向 Ray head 实测**（`ray status -v` 的 A100/V100
计数）。三个集群已重新配卡，端口和卡数的旧对应关系（6661→3 卡、6663→5 卡）已经不成立；
读不到卡型、缺 A100、或 A100 + V100 之和对不上总卡数，脚本立即退出。
`run_serve_0728.sh` 已做同样处理。

必须从主检出目录 `/home/wangjh/gnn_predict` 提交：常驻的 Ray actor 没有独立运行环境，
永远导入这个目录下的代码。

### 阶段一（30 条）

```sh
ARM=parrot ARRIVAL=burst RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0729.sh
ARM=kairos ARRIVAL=burst RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0729.sh
ARM=formula_baseline ARRIVAL=burst RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0729.sh
ARM=gbdt_baseline ARRIVAL=burst RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0729.sh
ARM=sagepilot ARRIVAL=burst RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0729.sh
ARM=parrot ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0729.sh
ARM=kairos ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0729.sh
ARM=formula_baseline ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0729.sh
ARM=gbdt_baseline ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0729.sh
ARM=sagepilot ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0729.sh
ARM=parrot ARRIVAL=burst RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0729.sh
ARM=kairos ARRIVAL=burst RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0729.sh
ARM=formula_baseline ARRIVAL=burst RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0729.sh
ARM=gbdt_baseline ARRIVAL=burst RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0729.sh
ARM=sagepilot ARRIVAL=burst RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0729.sh
ARM=parrot ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0729.sh
ARM=kairos ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0729.sh
ARM=formula_baseline ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0729.sh
ARM=gbdt_baseline ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0729.sh
ARM=sagepilot ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0729.sh
ARM=parrot ARRIVAL=burst RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0729.sh
ARM=kairos ARRIVAL=burst RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0729.sh
ARM=formula_baseline ARRIVAL=burst RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0729.sh
ARM=gbdt_baseline ARRIVAL=burst RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0729.sh
ARM=sagepilot ARRIVAL=burst RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0729.sh
ARM=parrot ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0729.sh
ARM=kairos ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0729.sh
ARM=formula_baseline ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0729.sh
ARM=gbdt_baseline ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0729.sh
ARM=sagepilot ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0729.sh
```

### 阶段二（12 条）

```sh
ARM=formula_with_sagepilot ARRIVAL=burst RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0729.sh
ARM=gbdt_with_sagepilot ARRIVAL=burst RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0729.sh
ARM=formula_with_sagepilot ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0729.sh
ARM=gbdt_with_sagepilot ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=1 sh scripts/workflow/run_serve_0729.sh
ARM=formula_with_sagepilot ARRIVAL=burst RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0729.sh
ARM=gbdt_with_sagepilot ARRIVAL=burst RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0729.sh
ARM=formula_with_sagepilot ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0729.sh
ARM=gbdt_with_sagepilot ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=2 sh scripts/workflow/run_serve_0729.sh
ARM=formula_with_sagepilot ARRIVAL=burst RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0729.sh
ARM=gbdt_with_sagepilot ARRIVAL=burst RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0729.sh
ARM=formula_with_sagepilot ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0729.sh
ARM=gbdt_with_sagepilot ARRIVAL=poisson_r050 RAY_PORT=6663 REPEAT=3 sh scripts/workflow/run_serve_0729.sh
```

---

## 4. 合格判据与开关生效证据

一次运行是否合格只看一个条件：`run_summary.json` 存在，且
`completed_session_count == 60` 且 `failed_session_count == 0`。不满足就是废运行。

除此之外，读结果前应当先确认开关真的生效（判据取自 `serve_0728_reference.md` 第 6 节）：

| 要确认的事 | 看什么 |
|---|---|
| 基线方案没有节点融合 | `workflow_trace.jsonl` 中 `task_execution_finished` 的 `payload.stages` 非空计数为 0 |
| 基线方案没有预取 | 没有 `prefetch_started` 事件 |
| 基线方案没有多开实例 | `run_summary.json` 的 `scale_out_count == 0` |
| 完整系统方案融合开着 | `payload.stages` 非空计数为 60（每个 chain_qmsum 会话 3 次 × 20 个会话） |

自变量应当从 `run_manifest.json` 读，不要解析目录名——清单里记录了实际发现到的显卡清单、
预测数据的哈希、到达配置的哈希和节点融合开关。

---

## 5. 跑实验时用的临时脚本

在 `tmp/serve_0729_driver/`，**不属于代码库正式产物**：

| 脚本 | 用途 |
|---|---|
| `drive.sh <端口> baselines` | 串行扫一遍阶段一的 30 格，不合格的删掉重跑 |
| `drive.sh <端口> with_sagepilot` | 同上，扫阶段二的 12 格 |
| `status.py` | 打印两组实验的完成度，加 `--with-sagepilot` 连阶段二一起打 |
| `progress_6663.tsv` | 每次尝试的流水：时刻、格子、第几次重复、合格与否、耗时 |

判定合格沿用 `tmp/serve_0728_driver/valid.py`。
