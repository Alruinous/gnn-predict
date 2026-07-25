# 多 workflow 实验 Runbook（2026-07-24）· 三数据集 Cache showcase

面向 HPCA。本轮因**新增 GSM8K workflow**(自一致性 fan-out → refine[14B])需**重新跑**，
目标是**用最能体现 Cache 方案的配置**跑一遍 QMSum+MBPP+GSM8K 三数据集同池实验，
使 Cache 的**模型复用 / reuse-distance 驱逐 / near-ready 预加载**三机制与结果指标同时可见。

上一轮冻结结果见 `docs/workflow/hpca2027_sagepilot_experiment_analysis_20260723.md`
（数据在 `output/serve_0723/`），**不重跑**，仅作对照与补缺口。

发作业方式沿用 0721 runbook：先 1 个 master 作业（起 Ray Head + 跑实验），再 N 个 worker 作业（每卡一个）。

---

## 为什么这样配（选点判据，务必先读）

冻结 profile 给出的两条硬事实决定了"哪个配置最能体现 Cache"：

| 模型 | 落卡 | 加载(s) | 运行512(s) | 显存(MB) |
|---|---|---:|---:|---:|
| Qwen3-1.7B | V100/A100 | 54 / 49 | 30 | 4.9k |
| Qwen3-4B | V100/A100 | **191** / 165 | 39 | 10k |
| Qwen3-8B | V100/A100 | **200** / 276 | 39 | 18k |
| Qwen3-14B | **A100 only**(30GB) | **257** | 28 | 30k |

1. **加载(191–257s) ≫ 单节点窗口(28–39s)**：预加载**无法完全隐藏加载**（窗口远小于加载）。
   Cache 最强、最稳的杠杆是**用冻结估计把"保谁/逐谁"判对，少一次 ~200s 重载**（复用 + reuse-distance 驱逐）。
2. **工作集 = 4 个模型，14B 只能上 A100**。GPU 槽位(1 副本=1 卡)：A1V1=2、A1V2=3、A1V3=4。
   要体现 Cache，必须让**工作集模型数 > 槽位**以强制驱逐。**GSM8K 的关键作用**：refine[14B] 占死 A100，
   把 {1.7B,4B,8B} 三个模型全逼到受限的 V100 池抢槽位——正是 reuse-distance 驱逐相对 LRU 发光处。
3. **burst vs Poisson**：burst（t=0 全灌）使系统全程饱和、模型始终驻留（历史实测四策略 prefetch=0、
   load 仅 4–19 次）→ 只体现**冷启动排序 + 均衡**；**开环 Poisson** 制造活跃-空闲交替 → 模型被逐后又被需要
   → **prefetch/eviction churn 才发生（prefetch_count 首次 >0）**。故两者都要：Poisson 讲机制，burst 讲冷启动/均衡。

**结论**：主打 **6w（三数据集×2）× 受限 A1V2/A1V1 × {Poisson r050/r075, burst}**；
**明确避免** A1V3+（槽位≥模型数→无驱逐→Cache 无优势）与 2w-burst（低争用，FIFO 反而最快）——只作边界对照。

---

## 前置（一次性）

- **模型 + A100**：`/data/Models/Qwen/Qwen3-{1.7B,4B,8B,14B}` 就位；worker 中**至少 1 张 A100**（14B 只落 A100）。
- **数据集**：`dataset/{QMSum/data/ALL, mbpp/sanitized-mbpp.json, gsm8k/data/test.jsonl}` 就位（gsm8k 已下载）。
- **预测缓存**：`cache/profile/predictions.yaml` 已含 Qwen3-{0.6,1.7,4,8,14}B ×{a100,v100}× seq1024–8192 × out128–4096，
  三数据集全部 in-grid，**无需重新 profiling**（`PREDICTIONS` 默认指向它）。
- **配置状态**：三数据集 workflow/replay YAML 已就绪并校验（见文末清单）；本轮**新建** r075/r100 Poisson 四档。
  现有多 workflow YAML **无问题，未改动**。

---

## Worker 作业（所有实验通用）

每张卡一个 Pod，改 `PROJECT_DIR`/`VENV_DIR` 后提交，各申请一张卡：

```sh
sh scripts/workflow/serve_worker.sh   # 重复 N 次，N = MIN_GPUS
```

**硬件配比通过提交的 worker 组合 + `MIN_GPUS` 控制**：
- **A1V2**（主）：1×A100 + 2×V100，`MIN_GPUS=3`
- **A1V1**（放大受限增益）：1×A100 + 1×V100，`MIN_GPUS=2`
- `GPU_MEM` 默认 `v100=32768,a100=81920`；`PREDICTIONS` 默认 `cache/profile/predictions.yaml`。

> **命令格式（适配你的作业平台：一次提交 = 一个 master）**：下面每条命令是一个用 `\` 续行的
> **单条自包含命令**，env 全部**内联**——不用 `for` 循环、不依赖跨提交的 `export`。逐条提交，
> **一条命令 = 一个 RUN**。若一次提交里要顺序跑多个 master，用 `;` 串接并在中间插 `ray stop`
> （serve_submit 起 Ray Head 后不自动停）：`CMD1 ; ray stop ; CMD2`。
> 公共项：`MIN_GPUS`=A1V2 用 `3` / A1V1 用 `2`；`GPU_MEM=v100=32768,a100=81920`。
> `WORKFLOW_FILES` 只需是 replay 里用到的 workflow 的**超集**（多注册不用的 workflow 无害）。

---

## E1 —（主·机制 showcase）6w Poisson · A1V2 · 四策略

**最能体现 Cache 三机制**：Poisson 的空闲期触发驱逐/重载 → Cache 的 reuse-distance 驱逐 + near-ready 预加载
（`prefetch_count>0`）+ 跨-workflow 复用同时可见。

```sh
WORKFLOW_FILES=config/workflow/serve/qmsum1.yaml,config/workflow/serve/qmsum2.yaml,config/workflow/serve/mbpp1.yaml,config/workflow/serve/mbpp2.yaml,config/workflow/serve/gsm8k1.yaml,config/workflow/serve/gsm8k2.yaml \
  RAY_PORT=6661 \
  OUTPUT_DIR=output/serve/6w_poisson EXPERIMENT_CONFIG=config/workflow/serve/replay_6w_poisson_r150.yaml GPU_MEM=v100=32768,a100=81920 \
  RUN_ID=hetero_r150_cache_a1v2   SCHED_CONFIG=config/workflow/serve/scheduler_cache.yaml   MIN_GPUS=3 sh scripts/workflow/serve_submit.sh

WORKFLOW_FILES=config/workflow/serve/qmsum1.yaml,config/workflow/serve/qmsum2.yaml,config/workflow/serve/mbpp1.yaml,config/workflow/serve/mbpp2.yaml,config/workflow/serve/gsm8k1.yaml,config/workflow/serve/gsm8k2.yaml \
  RAY_PORT=6662 \
  OUTPUT_DIR=output/serve/6w_poisson EXPERIMENT_CONFIG=config/workflow/serve/replay_6w_poisson_r150.yaml GPU_MEM=v100=32768,a100=81920 \
  RUN_ID=hetero_r150_history_a1v2 SCHED_CONFIG=config/workflow/serve/scheduler_history.yaml MIN_GPUS=3 sh scripts/workflow/serve_submit.sh

WORKFLOW_FILES=config/workflow/serve/qmsum1.yaml,config/workflow/serve/qmsum2.yaml,config/workflow/serve/mbpp1.yaml,config/workflow/serve/mbpp2.yaml,config/workflow/serve/gsm8k1.yaml,config/workflow/serve/gsm8k2.yaml \
  RAY_PORT=6663 \
  OUTPUT_DIR=output/serve/6w_poisson EXPERIMENT_CONFIG=config/workflow/serve/replay_6w_poisson_r150.yaml GPU_MEM=v100=32768,a100=81920 \
  RUN_ID=hetero_r150_fifo_a1v2    SCHED_CONFIG=config/workflow/serve/scheduler_fifo.yaml    MIN_GPUS=3 sh scripts/workflow/serve_submit.sh

WORKFLOW_FILES=config/workflow/serve/qmsum1.yaml,config/workflow/serve/qmsum2.yaml,config/workflow/serve/mbpp1.yaml,config/workflow/serve/mbpp2.yaml,config/workflow/serve/gsm8k1.yaml,config/workflow/serve/gsm8k2.yaml \
  RAY_PORT=6664 \
  OUTPUT_DIR=output/serve/6w_poisson EXPERIMENT_CONFIG=config/workflow/serve/replay_6w_poisson_r150.yaml GPU_MEM=v100=32768,a100=81920 \
  RUN_ID=hetero_r150_kairos_a1v2  SCHED_CONFIG=config/workflow/serve/scheduler_kairos.yaml  MIN_GPUS=3 sh scripts/workflow/serve_submit.sh
```

四策略共用同一 `arrival_seed=42` → 逐字节相同到达轨迹（paired）。**sanity**：cache/history `prefetch_count>0`；
kairos/fifo `prefetch_count==0`。

## E2 —（主·结果/冷启动 + 受限放大）6w burst · A1V2

burst 讲**冷启动排序 + 均衡**。A1V2 为主
但 makespan 会明显变长。

A1V2 四条（逐条提交）。**A1V1 版本**：只提交 1×A100+1×V100，把每条的 `MIN_GPUS=3`→`2`、
`OUTPUT_DIR=output/serve/6w`→`output/serve/6w_a1v1`、`RUN_ID` 的 `_a1v2`→`_a1v1`（共 8 条）。

```sh
WORKFLOW_FILES=config/workflow/serve/qmsum1.yaml,config/workflow/serve/qmsum2.yaml,config/workflow/serve/mbpp1.yaml,config/workflow/serve/mbpp2.yaml,config/workflow/serve/gsm8k1.yaml,config/workflow/serve/gsm8k2.yaml \
  RAY_PORT=6661 \
  OUTPUT_DIR=output/serve/6w EXPERIMENT_CONFIG=config/workflow/serve/replay_6w.yaml GPU_MEM=v100=32768,a100=81920 \
  RUN_ID=hetero_cache_a1v2   SCHED_CONFIG=config/workflow/serve/scheduler_cache.yaml   MIN_GPUS=3 sh scripts/workflow/serve_submit.sh

WORKFLOW_FILES=config/workflow/serve/qmsum1.yaml,config/workflow/serve/qmsum2.yaml,config/workflow/serve/mbpp1.yaml,config/workflow/serve/mbpp2.yaml,config/workflow/serve/gsm8k1.yaml,config/workflow/serve/gsm8k2.yaml \
  RAY_PORT=6662 \
  OUTPUT_DIR=output/serve/6w EXPERIMENT_CONFIG=config/workflow/serve/replay_6w.yaml GPU_MEM=v100=32768,a100=81920 \
  RUN_ID=hetero_history_a1v2 SCHED_CONFIG=config/workflow/serve/scheduler_history.yaml MIN_GPUS=3 sh scripts/workflow/serve_submit.sh

WORKFLOW_FILES=config/workflow/serve/qmsum1.yaml,config/workflow/serve/qmsum2.yaml,config/workflow/serve/mbpp1.yaml,config/workflow/serve/mbpp2.yaml,config/workflow/serve/gsm8k1.yaml,config/workflow/serve/gsm8k2.yaml \
  RAY_PORT=6663 \
  OUTPUT_DIR=output/serve/6w EXPERIMENT_CONFIG=config/workflow/serve/replay_6w.yaml GPU_MEM=v100=32768,a100=81920 \
  RUN_ID=hetero_fifo_a1v2    SCHED_CONFIG=config/workflow/serve/scheduler_fifo.yaml    MIN_GPUS=3 sh scripts/workflow/serve_submit.sh

WORKFLOW_FILES=config/workflow/serve/qmsum1.yaml,config/workflow/serve/qmsum2.yaml,config/workflow/serve/mbpp1.yaml,config/workflow/serve/mbpp2.yaml,config/workflow/serve/gsm8k1.yaml,config/workflow/serve/gsm8k2.yaml \
  RAY_PORT=6664 \
  OUTPUT_DIR=output/serve/6w EXPERIMENT_CONFIG=config/workflow/serve/replay_6w.yaml GPU_MEM=v100=32768,a100=81920 \
  RUN_ID=hetero_kairos_a1v2  SCHED_CONFIG=config/workflow/serve/scheduler_kairos.yaml  MIN_GPUS=3 sh scripts/workflow/serve_submit.sh
```

## E3 —（干净机制对比）3w burst · A1V2 · Cache vs History

workflow 少、噪声小，专讲 "Cache 冻结估计在冷启动阶段避免一次 ~200s 迁移/重载" 的因果故事（doc 的 trace 案例）。

```sh
WORKFLOW_FILES=config/workflow/serve/qmsum1.yaml,config/workflow/serve/mbpp1.yaml,config/workflow/serve/gsm8k1.yaml \
  RAY_PORT=6661 \
  OUTPUT_DIR=output/serve/3w EXPERIMENT_CONFIG=config/workflow/serve/replay_3w.yaml GPU_MEM=v100=32768,a100=81920 \
  RUN_ID=hetero_cache_a1v2   SCHED_CONFIG=config/workflow/serve/scheduler_cache.yaml   MIN_GPUS=3 sh scripts/workflow/serve_submit.sh

WORKFLOW_FILES=config/workflow/serve/qmsum1.yaml,config/workflow/serve/mbpp1.yaml,config/workflow/serve/gsm8k1.yaml \
  RAY_PORT=6662 \
  OUTPUT_DIR=output/serve/3w EXPERIMENT_CONFIG=config/workflow/serve/replay_3w.yaml GPU_MEM=v100=32768,a100=81920 \
  RUN_ID=hetero_history_a1v2 SCHED_CONFIG=config/workflow/serve/scheduler_history.yaml MIN_GPUS=3 sh scripts/workflow/serve_submit.sh

WORKFLOW_FILES=config/workflow/serve/qmsum1.yaml,config/workflow/serve/mbpp1.yaml,config/workflow/serve/gsm8k1.yaml \
  RAY_PORT=6663 \
  OUTPUT_DIR=output/serve/3w EXPERIMENT_CONFIG=config/workflow/serve/replay_3w.yaml GPU_MEM=v100=32768,a100=81920 \
  RUN_ID=hetero_fifo_a1v2    SCHED_CONFIG=config/workflow/serve/scheduler_fifo.yaml    MIN_GPUS=3 sh scripts/workflow/serve_submit.sh

WORKFLOW_FILES=config/workflow/serve/qmsum1.yaml,config/workflow/serve/mbpp1.yaml,config/workflow/serve/gsm8k1.yaml \
  RAY_PORT=6664 \
  OUTPUT_DIR=output/serve/3w EXPERIMENT_CONFIG=config/workflow/serve/replay_3w.yaml GPU_MEM=v100=32768,a100=81920 \
  RUN_ID=hetero_kairos_a1v2  SCHED_CONFIG=config/workflow/serve/scheduler_kairos.yaml  MIN_GPUS=3 sh scripts/workflow/serve_submit.sh
```


## 重复与冷启动（拿置信区间）

- 每配置 **≥5 次独立 trial**：Poisson 把 `arrival_seed` 42→46 同步用于四策略（paired）；burst 用 `RUN_ID` 加 `_r1…_r5`。
- **随机化策略运行顺序**（或拉丁方），避免运行顺序与冷缓存混杂。
- **冷文件缓存起始**：每个 trial 前 `sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`（由你执行），
  暴露 History 的冷启动劣势、放大 Cache 冻结估计优势。首轮可先各单跑一次拿信号。
- 失败/超时/异常冷加载**不从分母剔除**。

---

## 读结果

每个 RUN 出指标（沿用 0721 runbook）：

```sh
PYTHONPATH=src uv run python -c "
from pathlib import Path; import json
from experiment.workflow.analysis import summarize_trial
s = summarize_trial(Path('output/serve/6w_poisson/hetero_r075_cache_a1v2'))
print(json.dumps({k: s[k] for k in ('makespan_sec','completed_session_count','session_latency_sec',
  'resident_gpu_seconds','idle_resident_gpu_seconds','pipeline_bubble_ratio',
  'model_load_count','model_reuse_count','model_eviction_count',
  'prefetch_count','wasted_prefetch_count')}, indent=2))
"
```

**每个 RUN 的 sanity check**：`completed_session_count==120`、`oom_count==0`、
`model_load_count + model_reuse_count == placement_count`；kairos/fifo `prefetch_count==0`；
Poisson 下 cache/history `prefetch_count>0`（否则 policy 或到达轨迹未生效）。

---

## 机制消融（需代码，单列，不在本 runbook 的 YAML 范围）

`SchedulerConfig.policy` 是固定枚举 `fifo|history|cache|kairos`，**无预取/驱逐开关**。因此：
- `Cache − 冻结估计` = **History**（已可跑，天然消融）。
- `Cache − reuse-distance 驱逐(退化 LRU)` 与 `Cache − near-ready 预加载` **需新增 policy 变体 + scheduler 代码**，
  不是 YAML 能配的。若要做机制归因图，另开一个代码任务（在 `SchedulerPolicy` 加 `cache_lru`/`cache_noprefetch`
  两个变体，`scheduler.py`/`policy.py` 对应关掉一条路径）。

---

## 配置清单

**已就绪（校验通过，未改）**：
- workflow：`config/workflow/serve/{qmsum1..4,mbpp1..4,gsm8k1..4}.yaml`
- burst replay：`replay_{2w,4w,6w,8w}.yaml`、`replay_3w.yaml`
- Poisson r050：`replay_{2w,4w}_poisson_r050.yaml`、`replay_{3w,6w}_poisson_r050.yaml`
- 调度：`scheduler_{fifo,history,cache,kairos}.yaml`
- 预测缓存：`cache/profile/predictions.yaml`（Qwen3 全网格，三数据集 in-grid）

**本轮新建**：
- `replay_3w_poisson_r075.yaml`（λ 0.0125/wf）、`replay_3w_poisson_r100.yaml`（0.016667/wf）
- `replay_6w_poisson_r075.yaml`（0.00625/wf）、`replay_6w_poisson_r100.yaml`（0.008333/wf）

> λ 为**名义 ρ**（相对旧 2 数据集 burst 饱和 ≈0.05 session/s）；三数据集含更轻的 gsm8k，真实饱和更高，
> 可先跑一组 6w burst 读 makespan→吞吐后回校 `arrival_rate_per_sec`（可选，不阻塞首轮）。

## 指标实现入口

- Trace 汇总：`src/experiment/workflow/analysis.py::summarize_trial`
- 调度与 lifecycle：`src/workflow/scheduler.py`；驱逐 victim：`src/workflow/policy.py`
- 数据回放入口：`src/experiment/workflow/experiments/dataset_replay.py`（scenario→workflow_name→serve YAML）
