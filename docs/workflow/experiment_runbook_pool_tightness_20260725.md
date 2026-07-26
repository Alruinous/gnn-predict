# 实验 Runbook：3-workflow × pool 紧度（2026-07-25）

面向 `paper/hpca2027-sagepilot/sections/05-evaluation.tex` 的调度实验。集群开关由人工执行，
本文只规定**每个集群配多少张什么卡**、**跑哪些 run**、**每个 run 怎么验收**。

代码基线：`main` @ `c342d1d`（含 `becb1c5` 的 cache 修复）。

---

## 1. 为什么是这个矩阵

三条已测量的事实决定了矩阵形状。

**(a) 系统是 load-bound，不是算力受限。** 6w r150 A1V2 实测：总 acquire 等待 41332s vs 总执行
4116s，即 **90.9% 的排队延迟是模型加载延迟**（66% 等自己的模型加载、33% 头阻塞在别的 key 的加载
后面、仅 1% 是算力争抢）。所以调度收益全部来自减少/重排加载。

**(b) 收益只在"模型工作集装不下"时存在。** 同一份实测工作量，只改 pool 的离线回放：

| pool | 全程加载次数 | 四臂 mean 极差 |
|---|---|---|
| 1×A100 + 2×V100 | 37–40 | 真机 42%（cache 408s vs fifo 703s） |
| 1×A100 + 3×V100 | 37–40 | 2.7% |
| 1×A100 + **4**×V100 | **5**（纯暖机） | **0.5%** |

到 4 张 V100 整个工作集一次性驻留，没有驱逐、没有 prefetch，四种策略无法区分。**这不是缺陷，
是适用范围**——把它做成一条曲线，比回避它更有说服力。

**(c) 3 workflow 与 6 workflow 的模型压力完全相同。** 实测两者解析出**逐项相同**的 5 个
`model_key`（4 个驻留 V100），Qwen3-4B 都占 71% 的 agent 节点——6 个 workflow 本就是这 3 个的
成对复制。所以改用 3w 是**表述简化**（更好讲、更省卡），不改变任何机制压力。

---

## 2. 硬件清算

| 端口 | 现状 | 本 runbook 用途 |
|---|---|---|
| 6661 | 1×A100 + 2×V100 | 各阶段主力 |
| 6662 | 1×A100 + 2×V100 | 各阶段主力 |
| 6663 | 2×A100 + 2×V100 | 各阶段主力 |
| 6664 | Ray 报 3 张（口述为 4×V100，**卡型待确认**） | 各阶段主力 |

合计 **4×A100 + 9×V100 = 13 张**（若 6664 第 4 张 V100 存在则为 10×V100）。

Head 常驻 dev 容器且 `--num-gpus=0`，**每个 worker Pod 恰好一张卡**，靠
`RAY_HEAD_ADDR + RAY_PORT` 加入指定 head。所以「集群配多少卡」= 让多少张卡的 Pod 指向该端口。

> **开始前请确认 6664 的卡型**：`ray status --address 127.0.0.1:6664`，或看某个 run 的
> `accelerator_id`（形如 `.../v100:0`）。若它含 A100，Phase 2/3 的卡账需要重排。

---

## 3. 硬约束（违反会直接失败或不可比）

1. **Qwen3-14B 必须放 A100。** 按实测输入长度算的有效显存需求：`mbpp1.coder` 34592 MB、
   `gsm8k1.refine` 36506 MB，均 > V100 的 32768 MB。因此
   **纯 V100 pool 跑不了这个工作负载**（会得到 `request_infeasible`），每个集群至少 1 张 A100。
2. **每个集群至少 1×A100 + 1×V100。** 4B/8B/1.7B 在实测上下文下占 11.5–28.6 GB，V100 装得下。
3. **不要把 Qwen3-4B 的两个 key 合并。** 当前配置里 4B 按 `max_model_len` 分成 1024/8192 两个
   key，这让它能同时占住两张 V100（实测双副本驻留 653s）。调度器每个 `(模型, 卡型)` 只允许一个
   副本，所以合并 4B 会让承担 71% 需求的模型掉回单卡——实测 mean 从 287s 恶化到 408s。
   （14B 的 key 已经统一，那是纯赚：A100 加载 14→1 次、加载时间 1143s→228s。）
4. **一个集群同时只能跑一个实验**（每个 run 会 `evict_all` 共享 GPU 池）。并行靠多端口。
5. **同一 pool 下所有 arm 必须用同一批物理卡**，否则不可比。若中途重组过卡，该 pool 的所有 arm
   要重跑。

---

## 5. 实验矩阵

工作负载固定：3w = `qmsum1 + mbpp1 + gsm8k1`，各 40 session = 120，Poisson 聚合 0.075/s
（到达跨度 ~1600s，与 6w r150 同速率同总量，因此两者可直接对照）。单 run 墙钟约 **35–50 分钟**。

**种子固定为配置里的 `arrival_seed: 42`，不做重复跑。** 所有 arm 共用同一条到达序列，因此是
成对（paired）对照——策略之间的差异不含到达抖动。代价是没有 run 间方差，见 §8 的口径说明。

### Phase 1 — 主表：紧 pool（必做）

**集群配置：4 个集群各 1×A100 + 2×V100** → 用掉 4×A100 + 8×V100（余 1 张 V100 备用）。

| 端口 | 卡 |
|---|---|
| 6661 | 1×A100 + 2×V100 |
| 6662 | 1×A100 + 2×V100 |
| 6663 | 1×A100 + 2×V100 |
| 6664 | 1×A100 + 2×V100 |

4 arm 各占一个集群并行 → **1 轮 ≈ 45 分钟，4 个 run**。这是论文主表（makespan / mean / p50 /
p95 + 生命週期指标）。

### Phase 2 — pool 紧度曲线（强烈建议）

| 子阶段 | 每集群配卡 | 可并行集群数 | 卡账 | run 数 |
|---|---|---|---|---|
| N=3 | 1×A100 + 3×V100 | 3 | 3×A100 + 9×V100 | 4（2 轮） |
| N=4 | 1×A100 + 4×V100 | 2 | 2×A100 + 8×V100 | 4（2 轮） |

与 Phase 1 的 N=2 合成三点曲线：**cache 相对 fifo 的增益随 pool 收紧单调增长，并在工作集装得下
时收敛到 0**。约 3 小时。

**核心矩阵合计 12 个 run、5 轮、约 3.75 小时**（Phase 1 一轮 + Phase 2 四轮），期间需要重组卡
两次（N=2 → N=3 → N=4）。

### Phase 3 — 匹配 pool 的异构对照（可选，论文 §5.1 提到）

计数匹配（各 3 张卡）：`3×A100` vs `1×A100 + 2×V100`。前者用掉 3 张 A100。
只跑 `fifo` 与 `cache` 两臂即可说明硬件构成的影响，4 个 run，约 1 小时。

---

## 6. 提交命令

每条是一条自包含单行命令（env 全内联，不用 for 循环，一条 = 一个 run）。逐条提交。
`OUTPUT_DIR` 按 pool 分目录，便于分析脚本按 cohort 归组。

### Phase 1（N=2）

```sh
WORKFLOW_FILES=config/workflow/serve/qmsum1.yaml,config/workflow/serve/mbpp1.yaml,config/workflow/serve/gsm8k1.yaml,config/workflow/serve/qmsum2.yaml,config/workflow/serve/mbpp2.yaml,config/workflow/serve/gsm8k2.yaml \
RAY_PORT=6661 \
PYTHONHASHSEED=0 MIN_GPUS=3 GPU_MEM=v100=32768,a100=81920 \
EXPERIMENT_CONFIG=config/workflow/serve/replay_6w_poisson_r150.yaml \
PREDICTIONS=/home/wangjh/gnn_predict/cache/profile_v2/predictions.yaml \
SCHED_CONFIG=config/workflow/serve/scheduler_cache.yaml \
OUTPUT_DIR=output/serve_0726/6w_poisson_a1v2 RUN_ID=cache \
sh scripts/workflow/serve_submit.sh

WORKFLOW_FILES=config/workflow/serve/qmsum1.yaml,config/workflow/serve/mbpp1.yaml,config/workflow/serve/gsm8k1.yaml,config/workflow/serve/qmsum2.yaml,config/workflow/serve/mbpp2.yaml,config/workflow/serve/gsm8k2.yaml \
RAY_PORT=6662 \
PYTHONHASHSEED=0 MIN_GPUS=3 GPU_MEM=v100=32768,a100=81920 \
EXPERIMENT_CONFIG=config/workflow/serve/replay_6w_poisson_r150.yaml \
PREDICTIONS=/home/wangjh/gnn_predict/cache/profile_v2/predictions.yaml \
SCHED_CONFIG=config/workflow/serve/scheduler_fifo.yaml  \
OUTPUT_DIR=output/serve_0726/6w_poisson_a1v2 RUN_ID=fifo \
sh scripts/workflow/serve_submit.sh

WORKFLOW_FILES=config/workflow/serve/qmsum1.yaml,config/workflow/serve/mbpp1.yaml,config/workflow/serve/gsm8k1.yaml,config/workflow/serve/qmsum2.yaml,config/workflow/serve/mbpp2.yaml,config/workflow/serve/gsm8k2.yaml \
RAY_PORT=6663 \
PYTHONHASHSEED=0 MIN_GPUS=3 GPU_MEM=v100=32768,a100=81920 \
EXPERIMENT_CONFIG=config/workflow/serve/replay_6w_poisson_r150.yaml \
PREDICTIONS=/home/wangjh/gnn_predict/cache/profile_v2/predictions.yaml \
SCHED_CONFIG=config/workflow/serve/scheduler_kairos.yaml \
OUTPUT_DIR=output/serve_0726/6w_poisson_a1v2 RUN_ID=kairos \
sh scripts/workflow/serve_submit.sh

WORKFLOW_FILES=config/workflow/serve/qmsum1.yaml,config/workflow/serve/mbpp1.yaml,config/workflow/serve/gsm8k1.yaml,config/workflow/serve/qmsum2.yaml,config/workflow/serve/mbpp2.yaml,config/workflow/serve/gsm8k2.yaml \
RAY_PORT=6664 \
PYTHONHASHSEED=0 MIN_GPUS=3 GPU_MEM=v100=32768,a100=81920 \
EXPERIMENT_CONFIG=config/workflow/serve/replay_6w_poisson_r150.yaml \
PREDICTIONS=/home/wangjh/gnn_predict/cache/tabular/predictions.yaml \
SCHED_CONFIG=config/workflow/serve/scheduler_cache.yaml \
OUTPUT_DIR=output/serve_0726/6w_poisson_a1v2 RUN_ID=gbdt \
sh scripts/workflow/serve_submit.sh
```

3 workflow 效果十分不好，且有嚴重問題，不適合。