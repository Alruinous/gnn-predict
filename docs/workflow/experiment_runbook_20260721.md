# 多 workflow 实验 Runbook（2026-07-21）

面向 HPCA（截止 2026-08-01）。**要跑的新实验 = 异构 A100+V100 上 QMSum+MBPP 同池的多 workflow 全系统**
（论文 RQ4「Full system」的唯一缺口）。单机 V100 的 75-trial 矩阵已冻结、已进论文
（`docs/workflow/system_experiment_results_20260713.md`），**不重跑**。

范围锚定到已实现的系统（预测感知生命周期 + 共享池 + 显存安全准入）；论文里 KV
preparation-path / composition-failure / transition 等 `% TODO(exp)` 机制未实现，本轮不做。

发作业方式：先提交 1 个 master 作业（起 Ray Head + 跑实验），再提交 N 个 worker 作业（每 worker 一张卡）。

---

## 前置（一次性）

- **模型 + A100**：`/data/Models/Qwen/Qwen3-{4B,8B,14B}` 就位、且 worker 中**至少 1 张 A100**
  （MBPP 的 Qwen3-14B 只能落 A100）——由你保证。
- **数据集**：已写入 `config/workflow/serve/replay.yaml`，无需再改。
- **预测缓存**（先合成，立即可开跑）：

  ```sh
  PYTHONPATH=src uv run python scripts/workflow/write_serve_predictions.py output/serve/predictions.json
  ```

  可选实测升级（每种卡各测一次再合并，用同一 `output/serve/predictions.json` 无缝替换重跑）：

  ```sh
  # 在 V100 Pod / A100 Pod 各跑一次
  PYTHONPATH=src uv run python scripts/workflow/profile_workflow_cache.py \
    --cache-yaml config/workflow/cache.yaml --gpu-kind v100 --gpu-count 1 \
    --output-dir cache/profile/v100 --resume
  PYTHONPATH=src uv run python scripts/workflow/profile_workflow_cache.py \
    --cache-yaml config/workflow/cache.yaml --gpu-kind a100 --gpu-count 1 \
    --output-dir cache/profile/a100 --resume
  PYTHONPATH=src uv run python scripts/workflow/compose_profile_cache.py \
    --source v100=cache/profile/v100/predictions.yaml \
    --source a100=cache/profile/a100/predictions.yaml \
    --output cache/profile/predictions.yaml
  ```

---

## Worker 作业（所有实验通用）

每张卡一个 Pod，改 `PROJECT_DIR`/`VENV_DIR` 后原样提交，各申请一张卡（A100 或 V100）：

```sh
sh scripts/workflow/serve_worker.sh
```

`MIN_GPUS` 取你实际提交的 worker 总数；`GPU_MEM` 默认 `v100=32768,a100=81920`。

---

## E1 — 多 workflow 共享池 · 策略扫（主力）

四次 master 作业，只换 `SCHED_CONFIG` 和 `RUN_ID`；`fifo/history/cache` 是本项目内部消融，
`kairos` 是对外 prior-art 基线（Kairos 复现：全局剩余延迟 SRPT 排序 + 显存感知放置，无预取、
无预测式驱逐），复用同一批 worker/workload/predictions，与 `cache` 做系统级对比：

```sh
RUN_ID=hetero_fifo    SCHED_CONFIG=config/workflow/serve/scheduler_fifo.yaml    MIN_GPUS=2 sh scripts/workflow/serve_master.sh
RUN_ID=hetero_history SCHED_CONFIG=config/workflow/serve/scheduler_history.yaml MIN_GPUS=2 sh scripts/workflow/serve_master.sh
RUN_ID=hetero_cache   SCHED_CONFIG=config/workflow/serve/scheduler_cache.yaml   MIN_GPUS=2 sh scripts/workflow/serve_master.sh
RUN_ID=hetero_kairos  SCHED_CONFIG=config/workflow/serve/scheduler_kairos.yaml  MIN_GPUS=2 sh scripts/workflow/serve_master.sh
```

产物：`output/serve/<RUN_ID>/{workflow_trace.jsonl,run_summary.json,<qmsum|mbpp>/session_results.jsonl}`。
拿到置信区间需每配置重复 5 次（`RUN_ID` 加 `_r1`…`_r5` 后缀）；首轮先各 1 次拿信号。

### Kairos 基线 · 完整启动 recipe（自包含）

`kairos` 依赖 `config/workflow/serve/scheduler_kairos.yaml` 与 `kairos` 策略（本分支已含）。从零
起一次 Kairos run，复用与 `fifo/history/cache` 完全相同的 worker/workload/predictions：

```sh
# 1) 预测缓存（已在「前置」生成过可跳过）
PYTHONPATH=src uv run python scripts/workflow/write_serve_predictions.py output/serve/predictions.json

# 2) 每张卡各起 1 个 worker（改 PROJECT_DIR/VENV_DIR 后逐个提交；至少 1 张 A100 承载 Qwen3-14B）
sh scripts/workflow/serve_worker.sh   # 重复 N 次，N = 下面的 MIN_GPUS

# 3) Kairos master 作业（起 Ray Head + 跑 QMSum+MBPP 同池回放）
RUN_ID=hetero_kairos SCHED_CONFIG=config/workflow/serve/scheduler_kairos.yaml MIN_GPUS=2 sh scripts/workflow/serve_master.sh

# 4) 读指标 + sanity check（Kairos 无预取，两个计数必须为 0）
PYTHONPATH=src uv run python -c "
from pathlib import Path; import json
from experiment.workflow.analysis import summarize_trial
s = summarize_trial(Path('output/serve/hetero_kairos'))
assert s['prefetch_count'] == 0 and s['wasted_prefetch_count'] == 0, 'kairos 不应有预取'
print(json.dumps({k: s[k] for k in ('makespan_sec','completed_session_count','session_latency_sec',
  'resident_gpu_seconds','idle_resident_gpu_seconds','pipeline_bubble_ratio',
  'model_load_count','model_reuse_count','model_eviction_count','prefetch_count')}, indent=2))
"
```

置信区间：`RUN_ID=hetero_kairos_r1…_r5` 重复 5 次。与 `hetero_cache` 同法对照
resident/idle/pipeline-bubble/makespan/尾延迟，形成 trade-off 表。

## E2 — 加权公平性

在 `cache` 策略上加 `PRIORITY_WEIGHT`，对比 qmsum 优先与轮转：

```sh
RUN_ID=hetero_w21 SCHED_CONFIG=config/workflow/serve/scheduler_cache.yaml PRIORITY_WEIGHT=qmsum=2,mbpp=1 MIN_GPUS=2 sh scripts/workflow/serve_master.sh
RUN_ID=hetero_w11 SCHED_CONFIG=config/workflow/serve/scheduler_cache.yaml PRIORITY_WEIGHT=qmsum=1,mbpp=1 MIN_GPUS=2 sh scripts/workflow/serve_master.sh
```

## E3 — 异构配比

固定 `cache` 策略与负载。1A+3V 入口要求向同一个 Ray Head 提交 1 个 A100 worker 和
3 个 V100 worker：

```sh
RUN_ID=hetero_cache_a1v3 SCHED_CONFIG=config/workflow/serve/scheduler_cache.yaml MIN_GPUS=4 sh scripts/workflow/serve_master.sh
RUN_ID=hetero_kairos_a1v3 SCHED_CONFIG=config/workflow/serve/scheduler_kairos.yaml MIN_GPUS=4 sh scripts/workflow/serve_master.sh
```

观察 14B 是否稳定落 A100、小模型是否走最小安全卡，以及 makespan/驻留随配比的变化。

## E4 — 开环泊松到达（负载扫描）

E1–E3 都是 burst（`arrival_gap_sec=0`，所有 session 在 t=0 一次性灌入）。burst 使系统全程饱和、
模型始终驻留：实测四策略 `prefetch_count` 全为 0、`model_load` 仅 13–19 次，cache 的预取与预测式驱逐
没机会触发，优势只剩 makespan/尾延迟。**开环泊松**让 session 按指数间隔流入，制造「活跃-空闲」交替
→ 模型被驱逐后又被需要 → cache 的预取/保留才发挥（`prefetch_count` 首次 >0）。

**λ 锚点**：实测 burst 聚合饱和吞吐 ≈ 0.05 session/s（cache 3.36/min、fifo 2.52/min，见「读结果」）。
按 ρ×0.05 得聚合 λ，再均分到各 workflow。已备三档配置（`arrival_process: poisson`、每场景
`arrival_rate_per_sec`、顶层 `arrival_seed`；同一 seed 跨策略 → 逐字节相同到达轨迹）：

| ρ | 聚合 λ | 2wf 每场景 λ | 4wf 每场景 λ | 配置 |
|---|---|---|---|---|
| 0.50 | 0.025 | 0.0125 | 0.00625 | `replay_{2w,4w}_poisson_r050.yaml` |
| 0.75 | 0.0375 | 0.01875 | 0.009375 | `replay_{2w,4w}_poisson_r075.yaml` |
| 1.00 | 0.050 | 0.025 | 0.0125 | `replay_{2w,4w}_poisson_r100.yaml` |

> ρ 越低到达窗口越长：2wf 各档到达窗口 ≈79/52/39 min，总 wall-clock ≈ 到达窗口 + drain。首轮各单跑一次。

启动（复用 `serve_master.sh`，公共 env 固定，只换 `SCHED_CONFIG`/`RUN_ID`；2wf 另设 `WORKFLOW_FILES`/
`OUTPUT_DIR`，4wf 用默认 4 文件）：

```sh
# 2wf · ρ=0.75：四策略共用同一 EXPERIMENT_CONFIG（= 同 arrival_seed = 同到达轨迹，paired 对比）
export WORKFLOW_FILES=config/workflow/serve/qmsum1.yaml,config/workflow/serve/mbpp1.yaml
export OUTPUT_DIR=output/serve_2workflow_poisson
export EXPERIMENT_CONFIG=config/workflow/serve/replay_2w_poisson_r075.yaml
RUN_ID=pois2w_r075_fifo    SCHED_CONFIG=config/workflow/serve/scheduler_fifo.yaml    MIN_GPUS=2 sh scripts/workflow/serve_master.sh
RUN_ID=pois2w_r075_history SCHED_CONFIG=config/workflow/serve/scheduler_history.yaml MIN_GPUS=2 sh scripts/workflow/serve_master.sh
RUN_ID=pois2w_r075_cache   SCHED_CONFIG=config/workflow/serve/scheduler_cache.yaml   MIN_GPUS=2 sh scripts/workflow/serve_master.sh
RUN_ID=pois2w_r075_kairos  SCHED_CONFIG=config/workflow/serve/scheduler_kairos.yaml  MIN_GPUS=2 sh scripts/workflow/serve_master.sh

# 4wf · ρ=0.75（默认 4 文件 + output/serve_4workflow_poisson）
OUTPUT_DIR=output/serve_4workflow_poisson EXPERIMENT_CONFIG=config/workflow/serve/replay_4w_poisson_r075.yaml \
  RUN_ID=pois4w_r075_cache SCHED_CONFIG=config/workflow/serve/scheduler_cache.yaml MIN_GPUS=2 sh scripts/workflow/serve_master.sh
```

- **置信区间**：每个 repeat 把配置里 `arrival_seed` 42→46 同步用于四策略（paired）；首轮先 seed=42 单跑。
- **预期信号**（用「读结果」的 `summarize_trial`）：
  - cache `prefetch_count` 首次 >0（burst 恒为 0）→ 直接证明预取价值；预取/驱逐优势预计 ρ≈0.5–0.75 最强，
    makespan/吞吐优势 ρ→1.0 最强。
  - cache vs fifo/history/kairos：更低 `model_load_count`、更高 `model_reuse_count`、更低
    `idle_resident_gpu_seconds`、更低 `session_latency_sec` p95/max，差距随 ρ 上升扩大。
  - kairos sanity：`prefetch_count==0 且 wasted_prefetch_count==0`。

---

## 读结果

每个 RUN 出指标：

```sh
PYTHONPATH=src uv run python -c "
from pathlib import Path; import json
from experiment.workflow.analysis import summarize_trial
s = summarize_trial(Path('output/serve/hetero_cache'))
print(json.dumps({k: s[k] for k in ('makespan_sec','completed_session_count','session_latency_sec',
  'resident_gpu_seconds','idle_resident_gpu_seconds','pipeline_bubble_ratio',
  'model_load_count','model_reuse_count','model_eviction_count','prefetch_count')}, indent=2))
"
```

可直接得到：makespan、JCT/延迟 p50/p95、resident/idle/pipeline-bubble GPU-s、模型 load/reuse/evict/prefetch 计数、TTFT、token 量。

**Kairos 对比 sanity check**：`hetero_kairos` 的 `prefetch_count` 与 `wasted_prefetch_count` 必须为 0
（Kairos 无预取），否则说明 policy 未生效。核心对照是 `hetero_kairos` vs `hetero_cache` 的
`resident_gpu_seconds`/`idle_resident_gpu_seconds`/`pipeline_bubble_ratio`/`makespan_sec` 与
`session_latency_sec`{p50,p95}，预期是"cache 省驻留、Kairos 可能占尾延迟"的 trade-off。

> 能耗/利用率、LangGraph 基线、n=5 自动聚合出图暂未接入 master 路径，由后续 agent 补
> （复用 `telemetry.py`/`aggregate.py`/`report.py`）。首轮用上面的 trace 指标即可支撑多 workflow 故事。
