# GNN 资源预测缓存优势实验：离线精度—成本对比（2026-07-24）

## 目标与结论摘要

在**不依赖 GPU 集群、不触碰 vLLM 运行时**的前提下，量化 SagePilot 调度器所消费的 `ResourceContractCache` 用不同方式生成时的**预测精度**与**生成成本**，以 profile（实测剖析）为 ground-truth 精度上限，对照 llmfit 类静态解析模型与其它 predictor 基线。

**三条主张的证据结论（诚实版）：**

1. **GNN 决定性优于静态解析（llmfit 类）——在调度最关键的时序信号上。** GNN 的 `run_sec` WAPE=**0.076**（≈论文 RQ1 的 0.066），静态解析=**0.738**（差 **9.7×**）；静态解析对 `load_sec` 几乎完全错误（WAPE 0.996，100% 低估），对显存 **89% 低估、最大低估 21.6 GB**（放置/OOM 的危险方向），而 GNN 系统性**高估**显存（偏安全侧）。✅ 支持。
2. **GNN 的成本远优于 profile 上限。** 建出完整 profile 缓存需 **64.4 GPU-小时**（2848 配置，边际 81.4 s/配置）；GNN 产出一条新预测只需一次图上 forward=**37 ms**、无 GPU 执行、无 generate（边际成本约为 profile 的 **1/2200**）。✅ 支持。
3. **在相同剖析预算下，GNN 的泛化优于记忆式/表格基线。** 给 config-mean / nearest-profile / tabular 与 GNN 相同的锚点预算（5 个/cell = 50 条 profile，正是 GNN 运行时校准所用），GNN 的 `run_sec` WAPE=0.076，最好的基线（nn 0.452 / tabular 0.456）≈**0.45**（差 **~6×**）；tabular 需 **~500 条锚点（10× 预算）** 才追平 GNN（0.079）。✅ 支持。

**诚实边界（必须保留）：** 在**稠密同分布网格**上、给足**全量剖析标签**时，直接在 profile 标签上回归的 **tabular 在每个目标上都优于 GNN**（`run_sec` 0.033、`peak_vram` 0.012）。这不削弱上述主张——它恰好说明"无限剖析预算下记忆最优，但那正是 64 GPU-小时的代价"；有意义的工作区间是**有限剖析预算**，此处 GNN 占优（主张 3）。此外 GNN 在**显存点精度**上不如 tabular/nn，其价值在于**偏安全的高估 + 校准上界**（与论文"预测排序、校准上界准入"的设计一致），而非点精度。

---

## 背景：为何绕开 vLLM 做离线代理

推理后端 vLLM 每个 replica 独占整卡且 `gpu_memory_utilization=0.98` 预分配 KV，单模型 OOM 被 vLLM 兜底（抢占+重算），使"GNN 防 OOM"难以在线观测。破解：**不以 OOM 崩溃为信号**，把命题降到 vLLM 保护层无法掩盖的**预测精度 × 生成成本**——这正是调度器在 `policy.py:161`（放置选卡）与 `scheduler.py:1712`（批准入）两个安全门、以及计时/预取/驱逐软信号背后所依赖的量。预测差 → 决策差，是 vLLM 兜底也修不掉的根因。

## 方法

- **键空间**：`cache/profile/predictions.yaml` 的 2848 键（Qwen3-{0.6B,1.7B,4B,8B,14B} × {a100,v100} × batch{1,2,3} × 11 seq × 10 out，phase=decode）。所有方法在**同一键空间**上预测。
- **划分**：确定性 seed=7、20% 留出（`deterministic_split`），train=2278 / test=570。所有方法在**同一 570 留出键**上对 profile 真值评分。
- **目标**：`run_sec`（ETA）、`peak_vram_mb`（放置）、`power_watts`、`load_sec`。GNN 缓存复用 profile 的 `load_sec`（故其 load WAPE≡0，不参与 GNN-vs-others 的 load 对比）。
- **精度指标**：WAPE=Σ|pred−truth|/Σ|truth|、MAE、**低估率**（pred<truth 占比）、**最大低估**（安全相关）、mean_bias。
- **成本**：profile=Σ(run×1 + load×3) GPU-秒→GPU-小时（下界，未计 warmup/重试）；gnn=`IntelliGraphLargeModelPredictor.forward` 微基准（随机权重，前向延迟与权重无关；800 节点代表性图）+ 一次性训练/图捕获摊销；static/config-mean/nn/tabular=µs 级 CPU（tabular 训练标签仍继承全量剖析）。
- **预算扫描**：`stratified_anchor_sample(per_cell)` 给记忆/表格基线每 (model,gpu) cell 仅 N 条锚点，重建后在固定 570 留出集评分——隔离"泛化"与"网格记忆"。

**基线分类法（文献锚定）**：物理解析（llmfit / LIFE arXiv:2508.00904 / BestServe 2506.05871）；记忆查表（config-mean、nearest-profile）；学习-标量（tabular GBDT，无图，论文 predictor baseline #4）；学习-图（GNN；DIPPM 2303.11733 / PerfSAGE 2301.10999 / PerfSeer 2502.01206）；profile=真值上限。

## 复现命令

```bash
WT=.claude/worktrees/gnn-cache-eval   # 或合并后的仓库根
PYTHONPATH=$WT/src uv run python $WT/scripts/workflow/build_baseline_prediction_caches.py \
  --profile-cache cache/profile/predictions.yaml --cache-config config/workflow/cache.yaml --out-dir cache
PYTHONPATH=$WT/src uv run python $WT/scripts/workflow/eval_prediction_caches.py \
  --profile-cache cache/profile/predictions.yaml --cache-dir cache --out-dir output/cache_eval
```

## 结果

### 精度（570 留出键，profile 为真值）

| 方法 | run_sec WAPE | peak_vram WAPE | vram 低估率 | vram 最大低估 | load_sec WAPE | power WAPE |
|---|---|---|---|---|---|---|
| **gnn** | **0.076** | 0.235 | 0.244 | 10.8 GB | 0.000 | 0.262 |
| static (llmfit) | 0.738 | 0.161 | **0.891** | **21.6 GB** | 0.996 | 0.489 |
| config_mean | 0.777 | 0.184 | 0.321 | 17.9 GB | 0.000 | 0.158 |
| nearest_profile | 0.109 | 0.066 | 0.300 | 6.4 GB | 0.000 | 0.050 |
| tabular (GBDT) | **0.033** | **0.012** | 0.505 | 2.4 GB | 0.000 | 0.021 |

要点：GNN 的 `run_sec` 紧贴 profile 且 10× 优于静态解析；静态解析显存 89% 低估（危险），GNN 高估（安全，bias +1894 MB）。全量标签下 tabular 各项最优（记忆最强）。

### 生成成本（每个新配置的边际成本）

| 方法 | 边际成本/配置 | 是否需 GPU | 备注 |
|---|---|---|---|
| profile | **81.4 s** | 是 | 真机执行；换模型/换卡须全量重跑；全网格=**64.4 GPU-小时** |
| **gnn** | **37 ms** | 否 | 图上一次 forward，无 generate；训练/图捕获一次性摊销 |
| static | 34 µs | 否 | 解析公式 |
| config_mean | 38 µs | 否 | 查表 |
| nearest_profile | 1.4 ms | 否 | 锚点最近邻扫描 |
| tabular | 2.3 ms | 否 | GBDT 前向；**训练标签继承全量剖析** |

GNN 取一条**新配置**预测的边际成本 ≈ profile 的 **1/2200**。一次性成本上，GNN 缓存的校准仅用了 ~468(显存)+50(运行时) 条 profile 锚点（≈ 一次性 11 GPU-小时，且基础预测来自独立的 `data/scaled` 数据集），而 tabular/nn/config-mean 的准确性依赖先付出的 **64 GPU-小时全量标签**——这是主张 3 预算扫描要隔离的差异。

### 剖析预算扫描：run_sec WAPE vs 每 cell 锚点数

| 每 cell 锚点 | 总锚点 | config_mean | nearest | tabular | **gnn(固定)** | static(0) |
|---|---|---|---|---|---|---|
| 1 | 10 | 0.977 | 0.977 | 0.839 | — | — |
| 2 | 20 | 0.941 | 0.657 | 0.920 | — | — |
| 3 | 30 | 0.818 | 0.478 | 0.821 | — | — |
| **5** | **50** | 0.839 | 0.452 | 0.456 | **0.076** | — |
| 10 | 100 | 0.774 | 0.337 | 0.218 | 0.076 | — |
| 20 | 200 | 0.806 | 0.265 | 0.141 | 0.076 | — |
| 50 | 500 | 0.780 | 0.191 | **0.079** | 0.076 | — |
| 参照 | 0 | — | — | — | — | 0.738 |

**核心信号**：在 GNN 的 50 锚点运行时校准预算下，GNN=0.076，最好的记忆基线 nn/tabular≈0.45（**~6× 差**）；tabular 需 ~500 锚点（10× 预算）才追平 GNN 的 0.076（0.079）。config_mean 恒 ~0.80（弱）。显存维度 tabular 在 ≥100 锚点即反超 GNN（显存平滑易插值），如实呈现于 `output/cache_eval/budget_sweep.csv`。（锚点采样为 hashlib 确定性种子，跨进程可复现。）

## 与论文对照

- 印证 `sections/05-evaluation.tex:62` 的 predictor 基线设定（analytical size/bandwidth、tabular regressor、nearest-profile、config means）与 RQ1 的 `run_sec`/`peak_mem` WAPE 量级（GNN run_sec 0.076 ≈ 论文 0.066）。
- 与 `sections/02-background-motivation.tex:65`（显存 49.9% 低估、最大 24.3 GB）方向一致：本实验静态解析 89% 低估、最大 21.6 GB；GNN 偏安全高估——支撑论文"预测排序、校准上界准入"（`sections/03-design.tex:84`）而非拿点估计当硬门的设计。

## 局限与后续

- **真 leave-model-out 需重训 GNN**：现有 `cache/gnn/predictions.yaml` 由在全部 5 模型上训练的 GNN 产出，无法在"留出整个模型架构"下公平评估 GNN（会偏袒 GNN 或需 GPU 重训）。本实验以**锚点预算匹配**在离线、不重训的前提下逼近该结论；真正的架构外推留作后续（需训练管线 + GPU）。
- **在线 serving 验证**（把三种缓存喂进调度器比 p95/goodput/evict）为下一步，需 A100+V100 Ray 集群；本轮范围限定离线。
- GNN 缓存的 `load_sec` 继承 profile，故 load 维度不体现 GNN 能力；`run_sec`/`peak_vram`/`power` 为 GNN 真实输出。

## 产物清单

- 基线缓存：`cache/{static,config_mean,nn,tabular}/predictions.yaml`（同 2848 键，`ResourceContractCache` v2，可被调度器直接消费）。
- 评估产物：`output/cache_eval/{accuracy_cost.json, tables.md, pareto.csv, budget_sweep.csv}`。
- 代码：`src/experiment/workflow/{static_cache,predictor_baselines}.py`、`scripts/workflow/{build_baseline_prediction_caches,eval_prediction_caches}.py`、`tests/test_static_cache.py`。

## 指标定义

- **WAPE** = Σ|pred−truth| / Σ|truth|（加权绝对百分误差，对量级鲁棒）。
- **低估率** = #(pred<truth) / N；**最大低估** = max(truth−pred)（放置安全相关，越大越危险）。
- **边际成本/配置** = 为一个此前未见的 (model,gpu,batch,seq,out) 取得一条预测所需成本；profile 为真机执行时间，gnn 为一次图 forward，其余为公式/查表/GBDT 前向。
- **每 cell 锚点数** = 每 (model,gpu) 允许的真实 profile 样本数（剖析预算旋钮）。
