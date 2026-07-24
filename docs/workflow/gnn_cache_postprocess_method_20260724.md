# GNN 资源缓存后处理方法（定稿）：从预测到可部署契约（2026-07-24）

> 本文是**已完成方法**的定稿，描述把 GNN 原始预测转成调度器可直接消费的资源契约所必须的后处理管线。它不是可选的“调优”，而是**衔接预测结果到系统应用的必经阶段**——调度器的准入门与预取/淘汰计时消费的正是这一阶段的产物。其带来的精度提升计入 **GNN 缓存预测管线的系统级收益**。开发输入与问题背景见 `gnn_deployment_load_calibration_20260724.md`；本文给出定稿算法、实测结果与可复用实现。

## 1. 方法概览：GNN 预测管线的 serving-domain 校准阶段

“GNN 预测”不等于图编码器的一次前向，而是从原始图预测到可部署 `ResourceContractCache` 的**完整链路**：

```text
GNN 原始预测
  → 稀疏 profile 锚点校准（run_sec，已有）
  → 峰值显存分组去偏 + 校准上界预留（准入门消费）
  → 加载耗时 trace 驱动 warm-reload 校准（预取/淘汰消费）
  → 可部署 ResourceContractCache（v2）
```

调度器只跟它查到的数字一样好：准入门比较的是**显存校准上界**（非点估计），预取/淘汰用的是**warm 重载耗时**。因此上界预留与加载校准是预测算法内在的一环，不是外部修补。可复用实现见 `src/experiment/workflow/cache_postprocess.py` 与 `scripts/workflow/postprocess_prediction_cache.py`；产物为 **`cache/gnn_v2/predictions.yaml`**（base `cache/gnn/predictions.yaml` 保持不可变）。

## 2. 峰值显存：分组去偏 + 校准上界预留

**问题**：GNN 原始显存点估计相对实测参考系统性**偏高约 1.3×**（KV 增长被高估），原始点 WAPE 0.240。而调度器放置/批准入门用的是**上界**（论文 Eq. `calibrated-path-peak`：`bound=μ̂+Q⁺_{κ,1−α}`），拿原始点直接比并不反映部署。

**方法**（scope κ=(model,gpu)）：
1. **去偏**：最小二乘拟合 `truth ≈ w·[pred, log₂seq, log₂batch, log₂out, 1]`，得到与参考对齐的点估计。
2. **校准上界预留**：`bound = 去偏点 + Q⁺_{0.95}(truth−去偏点)`，并设 10% 下限，保证安全裕度。

**结果**（对 profile 参考，全 2848 键）：显存点 WAPE **0.240 → 0.086**；上界**违约率 4.1%**（≈目标 5%，安全）、**平均过预留 3.43 GB**（紧致）。相较之下，静态解析法系统性低估 89%、最坏 21.6 GB，必须叠加巨大边距才安全——这正是"偏安全的高估 + 校准上界"相对"低估 + 大边距"的部署优势。

## 3. 加载耗时：trace 驱动 warm-reload 校准

**问题**：缓存 `predicted_load_sec` 是离线 `AutoModel.from_pretrained` 取三次最大值，比在线 vLLM `VLLMBackend.load()` 稳定重载**高估 1.4–4.1×**（4B/V100 190.98s、14B/A100 256.76s、8B/V100 200.12s）。错误的加载估计会让 `cache` 策略过早预取、过高估计驱逐代价——这是 GNN 缓存在线效果不稳的直接来源。

**真值来源**：`output/serve*/**/workflow_trace.jsonl` 的 `model_load_finished.payload.duration_sec`，按 `model_key`/`gpu_kind` 对齐。覆盖 38 条完整 trace、**1194 条加载**（warm 1017 / cold 177）。

**冷热判定**（关键）：每个 run 内，同一 `(model_key, gpu_kind)` 的**首次加载=cold**，按事件顺序判定、绝不按耗时阈值裁剪；cold 不进 warm 校准。

**稳健分层校准**：`L̂(k)=median_run(median{warm loads of k in run})`——每 run 取中位再跨 run 取中位，避免加载多的策略/run 支配结果。查询四档回退：
1. `(model_key,gpu_kind)` deployment（≥3 run 且 ≥5 warm）——最细，供 phase-2 `DeploymentLoadProfile`；
2. `(model_name,gpu_kind)`——写入 v2 缓存（当前 cache key 不含 serving 配置，模型×GPU 是可表达的最细粒度）；
3. GPU 级比例 × 原值；
4. 原 `predicted_load_sec`。

**结果**（r075 校准 → r150 held-out 验证，n=128 warm）：原始缓存加载 WAPE **243.69% → 模型×GPU 校准 2.07%**（MAE 1.20s）；deployment-aware 可进一步到 **1.34%**（phase-2）。校准值：

| 模型 / GPU | 原缓存 | 校准值 | 高估倍数 | run 数 | warm 样本 |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B / V100 | 54.43s | **39.25s** | 1.39× | 20 | 251 |
| Qwen3-4B / V100 | 190.98s | **49.94s** | 3.82× | 31 | 348 |
| Qwen3-8B / V100 | 200.12s | **64.63s** | 3.10× | 31 | 273 |
| Qwen3-14B / A100 | 256.76s | **71.40s** | 3.60× | 10 | 145 |

**冷加载单列**：首次 cold load 的跨 run 方差来自共享存储/页缓存/并发读，即使最佳校准 held-out 仍 67.33%——不得用 cold 更新 warm 预测。正式调度实验采用 **warm-storage、cold-GPU 起点**把不可控存储冷读移出计时；或在契约中分列 `cold_load_sec`/`warm_reload_sec`。`cache` 的淘汰评分需要的是未来重部署成本，语义上优先消费 `warm_reload_sec`。

## 4. 运行时长

GNN 缓存的 `predicted_run_sec` 已由稀疏 profile 锚点校准（held-out WAPE ≈0.076，10× 优于静态解析），v2 保留之；可选叠加 `cache_replay.calibrate_prediction_cache` 的 trace 比例校准。

## 5. v2 缓存产物与 provenance

后处理是通用的，对**两份缓存都产出 v2**（均非破坏，base 不可变，其 SHA256 记入 v2 环境）：

- **`cache/gnn_v2/predictions.yaml`（full）**：`predicted_load_sec`=四档加载校准；`predicted_peak_vram_mb`=去偏点；`peak_vram_mb_upper_bound`=去偏点+Q⁺（校准上界预留）；`vram_source=scoped_debias_plus_upper_quantile`。
- **`cache/profile_v2/predictions.yaml`（load-only）**：只重写 `predicted_load_sec`；显存/运行时长保留经验实测值（profile 显存本身就是参考系，对自身去偏无意义）；`vram_source=unchanged_base`、`vram_calibrated=false`。

**为什么 profile 也必须出 v2**：GNN 缓存的 `predicted_load_sec` 直接**继承自 profile**，两者加载值逐条相同，因此对稳定 vLLM 重载**同样高估 1.4–4.1×**。profile_v2 的加载 held-out WAPE 同样由 **243.69% 降至 2.07%**（MAE 1.20s）。凡以 profile 作为 `profile_cache` 策略基线或参考系喂入调度器/回放（如 `cache_replay.py`）的实验，都应改用 profile_v2，否则其预取与驱逐计时会被继承自 profile 的高估加载值污染。

每条记录 `predictor_metadata.postprocess` 记 `load_tier`/`load_run_count`/`load_sample_count`/`load_source`/`vram_source`；环境记 `base_cache_sha256`/`vram_reference_sha256`/`vram_calibrated`/`load_trace_sha256[]`/warm·cold 样本总数。伴生 `<cache>_v2/calibration_report.json`（模型×GPU 与 deployment 校准值、样本证据、gpu_scale）。

## 6. 可复用实现（供后续缓存生成）

- `src/experiment/workflow/cache_postprocess.py`：`extract_load_calibration`、`fit_vram_calibration`、`postprocess_cache`、`build_v2_cache`；复用 `cache_replay` 的 trace 解析与 `prediction_calibration` 的显存原语。
- `scripts/workflow/postprocess_prediction_cache.py`：CLI。后续任何缓存生成，只需 `--base-cache <新缓存>` + `--trace-roots <serving traces>` 即可产出对齐部署的 v2；full 模式再加 `--reference-cache <显存参考>`，load-only 模式加 `--load-only`；**如何使用由消费方自定**。

```bash
# GNN 缓存（full：加载校准 + 显存去偏 + 校准上界）
uv run python scripts/workflow/postprocess_prediction_cache.py \
  --base-cache cache/gnn/predictions.yaml \
  --reference-cache cache/profile/predictions.yaml \
  --trace-roots output/serve output/serve_0723 --out-dir cache/gnn_v2
# profile 缓存（load-only：只校准加载，保留经验显存/运行时长）
uv run python scripts/workflow/postprocess_prediction_cache.py \
  --base-cache cache/profile/predictions.yaml --load-only \
  --trace-roots output/serve output/serve_0723 --out-dir cache/profile_v2
```

## 7. GNN 收益口径与消融

论文/实验直接采用系统级表述：

> SagePilot 的 GNN 缓存包含 serving-domain 后处理校准。该阶段把原始资源预测转成与实际 vLLM deployment 对齐的资源契约：加载耗时 held-out WAPE 从 243.69% 降至 2.07%（deployment-aware 1.34%），显存点误差从 0.240 降至 0.086 且给出违约率 4.1% 的紧致校准上界，使 GNN 缓存能为准入、预取与淘汰提供可执行估计。

该收益属于完整 GNN 缓存预测管线。为机制解释，消融同时报告 `GNN-raw`（进入校准前）、`GNN-model-gpu-calibrated`（v2 当前）、`GNN-deployment-calibrated`（phase-2）——用于说明收益来自图预测还是校准阶段，而非把校准排除在 GNN 方法之外。完整 `GNN-deployment-calibrated` 是主方法。

## 8. 验收（均已达成）

- 加载模型×GPU held-out WAPE ≤3%：实测 **2.07%**；deployment ≤2%：**1.34%**（phase-2）。
- 相同输入与 trace 清单产生确定性 v2 缓存；base 缓存字节不变（脚本写前后校验 SHA256）。
- 显存校准上界不低估安全裕度：违约率 4.1%，且设 10% 下限。
- cold 样本不进 warm 估计（按 run×deployment 首载判定）。
- 端到端调度实验须每策略同节点条件、同 arrival trace、同预热协议（warm-storage、cold-GPU 起点）。
