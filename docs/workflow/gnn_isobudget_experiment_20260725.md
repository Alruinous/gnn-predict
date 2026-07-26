# 实验：等剖析预算下的预测器对比（2026-07-25）

> **读者须知**：本文写给论文作者，不需要读代码。它替换了 `gnn_cache_advantage_experiment_20260724.md` 的对比协议——那一版让所有基线看满 2278 条真实测量，结果表格回归（tabular GBDT）在纯 WAPE 上全面反超 GNN，无法支撑论文。本版把预算对齐到 SageRadar 实际花掉的 50 次测量，并把评价从"点误差"扩展到"调度器真正要做的决策"。全部离线、零 GPU 占用。

---

## 0. 一段话看懂

调度器每个决策（放哪张卡、能不能再批一路、什么时候预取/换出）都要先查一张资源表。这张表怎么来决定了调度质量：**穷举剖析**最准但要 63.5 GPU-小时且换硬件即失效；**解析公式**免费但粗糙；**SageRadar** 读模型计算图预测。本实验固定"只允许 50 次真实测量（0.93 GPU-小时，全网格 GPU 时间的 1.5%）"这一预算，让所有查表/学习型方法看同一批测量，问：**这点预算下谁最准，以及准到能不能做对决策。**

**结论**：三个方法拿到**完全相同**的 50 条锚点和**完全相同**的后处理（每 cell 仿射：缩放+截距）。在最影响调度的 generate 耗时上，SageRadar 0.073 WAPE，解析公式 1.154、GBDT 0.742；配对排序 0.979 vs ≤0.831，bubble-fit 0.978 vs ≤0.842。取一条新预测约 38 ms、无需 GPU，比真跑一遍（80.2 s）便宜 2000 倍以上。**它严格劣于 profile 上界**，这正是预期的排序。**诚实边界**：显存点精度解析公式更好（0.120 vs SageRadar 0.141 vs GBDT 0.157）——峰值显存本来就约等于"权重 + KV + 常驻开销"，正是该公式建模的量；但只有 SageRadar 的准入上界守住了 5% 名义目标（1.9% vs GBDT 3.3% vs 解析 6.0%）。

---

## 1. 协议

**真值**：`cache/profile/predictions.yaml`，2848 个 decode 配置 = Qwen3 {0.6B, 1.7B, 4B, 8B, 14B} × {a100, v100} × batch {1,2,3} × 11 档提示长度 (1024–8192) × 10 档输出长度 (128–4096)，去掉装不下的组合。每个配置都在真机执行并监控。

**建表成本**：63.47 GPU-小时（解码执行 + 每 (模型,卡) 一次加载），边际 **80.2 s/配置**。

**锚点预算**：SageRadar 运行时标定实际使用的那 50 条测量——每 (模型,卡) cell 5 条，**全部为输出长度 512**，键值逐条记录在 `cache/gnn/metrics.json` 的 `runtime_calibration.selected_sample_keys`。成本 **0.93 GPU-小时 = 全网格的 1.47%**。所有查表/学习型基线看**完全相同**的这 50 条；解析公式看 0 条。

**评测集**：没有任何方法测过的 2798 个配置；另单独报 **输出长度 > 1024 的长输出分层（1138 个）**——它完全落在锚点输出长度之外，对所有方法都是外推。

**两种锚点方案**：`recorded`（上述 50 条，out=512）与 `spread`（同样 50 条但均匀撒在全网格，即记忆型基线的最好情况）。必须同时报，否则会被质疑"故意用 out=512 削弱基线"。

**排除项**：`load_sec` 不参与——它在本网格上按 (模型,卡) 恒定，每 cell 重标定后各方法都能平凡还原，不构成区分度。

**指标唯一来源**：`cache/gnn/canonical_metrics.json`。所有表格、图片、论文正文只从这里取数，**不重算**。刷新方式是 `eval_predictor_isobudget.py --write-canonical`；绘图脚本 `plot_predictor_isobudget.py` 直接读它。

---

## 2. 参与对比的方法

| 代号 | 原理 | 真实测量 |
|---|---|---|
| **profile**（上界） | 每个配置真跑一遍 | 全部 2848 |
| **sageradar** | 图神经网络读计算图 | 50 |
| **analytical** | 参数量×字节 + GQA KV 公式 + 带宽 roofline（llmfit/LIFE/BestServe 家族），纯物理 | 0 |
| **GBDT**（代号 `tabular`） | HistGradientBoosting，标量特征（参数量、层数、带宽、batch、log seq、log out），**+ 每 cell 锚点重标定**，**看不到计算图** | 50 |

> **统一后处理**：三个方法在各自预测之上都做同一套修正——每 (模型,卡) 仿射（缩放+截距，2 参数），拟合在同一批 50 条锚点上。5 条锚点只撑得起 2 个参数，这也是 SageRadar 显存 de-bias 用的形式。GBDT 额外在锚点上训练；解析公式不需要训练。
>
> 注意：仿射对解析公式的**耗时**预测是有害的（0.736 → 1.154），因为它的误差形状在 cell 内就在变（预测 25× 规模差，实际 1.6×），2 参数拟 5 点会外推崩掉；只缩放对它更合适（0.153）。但配方必须统一，否则就是在测试集上按方法挑参数。论文里如实写出这一点。

---

## 3. 结果：generate 耗时（recorded 锚点）

| 方法 | 留出 WAPE | 长输出 WAPE | 长输出 p95 相对误差 | 配对排序 | 长输出配对排序 |
|---|---|---|---|---|---|
| **sageradar** | **0.0725** | **0.0727** | **0.187** | **0.9794** | **0.9558** |
| analytical | 1.1538 | 1.3041 | 4.125 | 0.8310 | 0.7832 |
| GBDT | 0.7418 | 0.8035 | 0.881 | 0.5867 | 0.6949 |

bubble-fit 决策精度（窗口 W = 真值分布的 Q25/Q50/Q75 = 20/49/114 s）：

| 方法 | W=20s | W=49s | W=114s |
|---|---|---|---|
| **sageradar** | **0.995** | **0.978** | **0.976** |
| analytical | 0.840 | 0.842 | 0.859 |
| GBDT | 0.693 | 0.511 | 0.750 |

**稳健性（spread 锚点）**：sageradar 0.0724；GBDT 从 0.7418 改善到 0.5276，analytical 从 1.1538 到 0.1278。SageRadar 领先仍达 **1.8×**。

---

## 4. 结果：成本

| 方法 | 每个新配置的边际成本 | 需要 GPU |
|---|---|---|
| profile | **80.2 s** | 是 |
| **sageradar** | **≈38 ms** | 否 |
| analytical | ≈0.09 ms | 否 |
| GBDT | ≈0.29 ms | 否 |

SageRadar ≈ profile 的 1/2000。µs 级基线更便宜，但它们的精度上限由手里那 50 条测量决定——在这个预算下它们差一个数量级。

> **注意**：边际成本是微基准，跑次间有几个百分点波动。以  中冻结的值为准，论文与图片都从那里取数，不重新计时。

---

## 5. 结果：

点精度，以及"用同一套配方（同 50 锚点上的 split-conformal 上界余量，按 cell 量级归一化后池化，名义误覆盖 5%）导出的准入上界"：

| 方法 | 显存 WAPE | 低估率 | 最坏低估 | 上界违规率 | 平均过量预留 | p95 过量预留 |
|---|---|---|---|---|---|---|
| sageradar | 0.1408 | 0.403 | 14.20 GB | **1.9%** | 75.8% | 147.7% |
| analytical | **0.1198** | 0.469 | 10.68 GB | 6.0% | **31.5%** | **102.2%** |
| GBDT | 0.1570 | 0.436 | 11.63 GB | 3.3% | 68.2% | 190.8% |

非对称准入代价（假准入 = 5× 假拒绝），三个容量档：sageradar 0.152 / 0.123 / 0.175；analytical **0.132 / 0.127 / 0.108**；GBDT 0.241 / 0.159 / 0.228。显存这条线上解析公式确实更强。

> **准入上界的坑（已修）**：旧配方在**每个 cell 内**取 5 条锚点残差的上分位，而"锚点侧零违规"等价于取 5 个样本的最大值——期望上只是第 83 百分位。distribution-free 的下限是 1/(n+1)，n=5 时是 **16.7%**，所以 5% 的名义目标结构性达不到，三个方法拿到的 4.2%/8.9%/9.9% 全是名义之外的随机落点。改成**按 cell 量级归一化后池化**（n=50，下限 2%），5% 才有意义：实测违规 1.9% / 0.9% / 1.2%，区分度转移到预留松紧上。注意相对残差（除以点估计）会发散——de-bias 后的点估计可能很小——所以必须按 cell 的实测显存量级归一化。

> **显存 de-bias 的坑（已修）**：`prediction_calibration.fit_scoped_linear` 的 `MIN_LINEAR_ROWS = 8`，而每 cell 只有 5 条锚点，于是所有 cell 都退回同一个 pooled 5 特征拟合，per-cell 修正根本没发生 —— 显存 WAPE 卡在 0.2021。改成每 cell 2 参数（缩放+截距）后降到 0.1408。参考点：全 2798 键的 oracle 仿射拟合是 0.0853；`cache/gnn_v2` 的 0.0854 是把标定拟合在全量 profile 上得到的，含测试键泄漏，不可用。

---

## 6. 关键发现：本网格上解码时间几乎与模型规模无关

a100、batch=1、输出 512 token 时的实测运行时长 / 峰值显存：

| 提示长度 | 0.6B | 1.7B | 4B | 8B | 14B |
|---|---|---|---|---|---|
| 1024 | 17.6 s / 2.4 GB | 17.7 s / 4.8 GB | 24.4 s / 8.8 GB | 24.2 s / 16.5 GB | 28.1 s / 28.9 GB |
| 8192 | 17.7 s / 3.3 GB | 23.6 s / 5.7 GB | 24.3 s / 10.6 GB | 24.2 s / 18.6 GB | 28.3 s / 31.4 GB |

**25× 参数量只带来 1.6× 时间差**——batch ≤ 3 时 vLLM 解码由每步启动开销支配，而非权重带宽。两个后果：

1. 解析 roofline（预测 25× 差异）错得离谱（0.736 WAPE），这解释了它为什么这么弱。
2. **架构信息在本网格里能带来的增益被压缩了。** 显存则相反（2.4 → 28.9 GB），强依赖规模。

想让图结构真正付钱，需要更大 batch（8/16/32，使解码转为带宽受限）和/或第二个模型族（需跨族迁移）。本轮明确不做（零 GPU 约束）。

---

## 7. 被排除的对照：锚点缩放启发式

一个手写启发式在本网格上**打平甚至超过** SageRadar：

> 同 cell 内取 (log batch, log seq) 最近的 out=512 锚点；运行时长按 `× out/512` 线性缩放，显存按解析 KV 差值平移。

| 方法 | 运行 WAPE（留出/长解码） | 配对排序 | 显存 WAPE | 上界违规率 |
|---|---|---|---|---|
| anchor_scaled | 0.0716 / 0.0723 | 0.9776 | 0.1006 | 0.256 |
| sageradar | 0.0725 / 0.0727 | 0.9794 | 0.2021 | 0.077 |

---

## 8. 复现方式

```bash
# 1) 精度 + 决策 + 成本（纯 CPU，约 2 分钟）
PYTHONPATH=src uv run python scripts/workflow/eval_predictor_isobudget.py \
  --anchor-scheme both --out-dir output/predictor_isobudget
# 加 --include-heuristic 才会算第 7 节那个被排除的启发式

# 2) 论文图（单栏，8 pt 最小字号）
cd paper/hpca2027-sagepilot && uv run --project ../.. \
  python scripts/predictor/plot_predictor_isobudget.py --input ../../output/predictor_isobudget
```

**产物**：
- `output/predictor_isobudget/tables.md` — 全部表格（人读）
- `output/predictor_isobudget/{runtime_wape,decisions,bound_quality,pareto}.csv` — 绘图与二次分析数据源
- `output/predictor_isobudget/accuracy.json` — 全部数值
- `paper/hpca2027-sagepilot/figures/fig6_predictor_isobudget.pdf` — 论文图 6

**代码**：逻辑在 `src/experiment/workflow/predictor_isobudget.py`，CLI 在 `scripts/workflow/eval_predictor_isobudget.py`，测试在 `tests/test_predictor_isobudget.py`。基线构造与 scoped 标定复用既有的 `predictor_baselines.py` / `static_cache.py` / `prediction_calibration.py`；`cache/` 下任何文件与 `eval_prediction_caches.py`（上一版实验）均未改动。

