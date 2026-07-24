# GNN 缓存部署耗时校准：Trace 后处理问题与开发方案（2026-07-24）

本文记录 GNN 资源缓存中模型部署耗时与在线 vLLM 实测值不一致的问题、现有 trace 的校准
证据，以及后续开发应采用的算法和验证协议。本文是开发输入，不代表校准链路已经接入当前
runtime。

## 核心结论

SagePilot 的 GNN 预测能力不应只指图编码器的一次前向，而应包含从原始图预测到可供调度器
消费的完整资源契约生成链路：

```text
GNN 原始预测
    → 稀疏 profile 锚点校准
    → 在线 trace 的 serving-domain 部署耗时校准
    → GNN ResourceContractCache
    → placement / prefetch / eviction
```

因此，对 GNN 缓存进行 trace 驱动的部署耗时后处理，是 GNN 预测算法的 serving-domain
calibration 阶段；其带来的 held-out 精度和端到端调度提升，应计入 **GNN 缓存预测管线的
系统级收益**。组件消融仍应区分 `GNN-raw` 与 `GNN-calibrated`，用于解释收益来自图预测还是
校准阶段，而不是把校准收益排除在 GNN 方法之外。

现有 trace 表明，这个校准不仅可行，而且是必要的：

- 当前 Profile/GNN 缓存对稳定 vLLM 部署耗时普遍高估 1.4～4.1 倍。
- 以 6w Poisson r075 的稳定重载校准，在后续 r150 上验证，原始缓存加载预测 WAPE 为
  `243.69%`；模型×GPU 校准后为 `2.06%`，deployment 校准后为 `1.34%`。
- 首次存储冷加载不能由现有模型/GPU 特征稳定预测；即使使用最细的 deployment 校准，held-out
  WAPE 仍为 `67.33%`。它必须通过实验预热或独立 cold-load 模型处理。

## 当前问题

### 缓存加载耗时与在线部署不是同一测量路径

`scripts/workflow/profile_workflow_cache.py` 使用 Transformers 的
`AutoTokenizer.from_pretrained` 和 `AutoModelForCausalLM.from_pretrained` 测量加载，
连续采样三次后把最大值写入 `predicted_load_sec`。例如：

- 4B/V100 三次为 `190.98s / 55.46s / 25.12s`，缓存写入 `190.98s`。
- 14B/A100 三次为 `256.76s / 99.25s / 50.22s`，缓存写入 `256.76s`。

在线 `src/workflow/replica.py` 实际测量的是 `VLLMBackend.load()`，包含
`AsyncLLMEngine`、模型权重和 KV cache 初始化。两条路径都试图描述“模型从未加载到可服务”
的时间，但 backend、serving 配置和存储冷热状态均不一致。

当前 GNN 缓存的 `predicted_load_sec` 直接继承 Profile 值，元数据标记为
`load_source: empirical_profile`。这意味着现有原始缓存尚未完成 serving-domain
deployment-time calibration，而不是说明部署耗时不属于 GNN 预测问题。

### 错误加载估计会直接影响 Cache/GNN 调度

当前 Cache 策略把 `predicted_load_sec` 同时用于：

- `prefetch_at = upstream_eta - load_sec - eps`；
- 被驱逐副本的未来 reload cost；
- 新副本记录中的 `expected_load_sec`。

当 4B 的稳定 vLLM 重载约为 `46～50s`，缓存却给出 `190.98s` 时，调度器会过早预取并高估
驱逐代价。该误差不会使 vLLM 本身加载得更慢，但会改变模型驻留、预取和淘汰时机，是 GNN
缓存在线效果不稳定的直接来源。

## Trace 数据与分析口径

### 数据范围

本次只读分析覆盖：

- `output/serve_0723` 与 `output/serve` 下 38 个 `workflow_trace.jsonl`；
- 1194 条完整的 `model_load_finished`；
- 历史 trace 626 条、当前 trace 568 条；
- 按严格规则识别的稳定重载：历史 545 条、当前 472 条。

实际部署耗时取自：

```text
model_load_finished.payload.duration_sec
```

模型和 serving 配置通过 `model_key`、`gpu_kind`、对应的
`model_load_started.payload.max_model_len` 以及任务 trace 中的 prediction key 对齐。

### 冷热划分

每个 run 中，同一 `(model_key, gpu_kind)` 的第一次加载固定标记为 cold，不进入 warm
校准集；后续成功加载才是稳定重载。该规则按事件位置判定，不按耗时阈值删除异常值，避免
看到结果后选择性裁剪。

现有 `src/experiment/workflow/cache_replay.py` 已有 warm median 校准方向，但它当前只把每张
accelerator 的第一条加载标为 cold。正式实现应改用上述“每 run、每 deployment 首次加载”
规则，避免同一 GPU 后续首次读取另一模型时污染 warm 样本。

## 校准实验结果

### Held-out r150

使用 r075 的 306 条稳定重载拟合，以时间上更晚的 r150 的 128 条稳定重载验证：

| 后处理方法 | r150 WAPE | MAE | 结论 |
|---|---:|---:|---|
| 原始 Profile/GNN 缓存 | 243.69% | 141.48s | 不可作为在线部署点估计 |
| 全局比例修正 | 10.98% | 6.37s | 无法表达模型差异 |
| 按 GPU 比例修正 | 10.46% | 6.07s | V100 内不同模型误差仍大 |
| **模型×GPU 稳健校准** | **2.06%** | **1.19s** | 可直接后处理现有缓存 |
| **Deployment 稳健校准** | **1.34%** | **0.78s** | 首选目标，需要 deployment-aware 接口 |

只保留已完整结束的 r150 Cache/FIFO trace，模型×GPU与 deployment 校准的 WAPE 分别为
`2.49%` 和 `1.84%`，结论不依赖两个未写出 `run_summary.json` 的 r150 trace。

### 跨日期验证

以 `output/serve_0723` 的稳定重载校准，在当前相同 deployment 上验证：

- 原始缓存 WAPE：`194.50%`；
- 模型×GPU 校准 WAPE：`4.12%`；
- 中位绝对误差：`0.48s`；
- 90% 样本的绝对误差不超过 `1.23s`。

这说明稳定重载时间具有跨实验复用价值，不是只在 r075/r150 同日数据上成立。

### 候选校准值

以下候选值采用“每个 run 内取中位数，再跨 run 取中位数”，避免加载次数较多的策略支配结果：

| Deployment | 当前 Profile/GNN | Trace 校准值 | 原始高估倍数 |
|---|---:|---:|---:|
| Qwen3-1.7B / V100 / 8192 | 54.43s | 39.60s | 1.37× |
| Qwen3-4B / V100 / 1024 | 190.98s | 46.38s | 4.12× |
| Qwen3-4B / V100 / 8192 | 190.98s | 50.34s | 3.79× |
| Qwen3-8B / V100 / 4096 | 200.12s | 64.93s | 3.08× |
| Qwen3-14B / A100 / 1024 | 256.76s | 71.48s | 3.59× |
| Qwen3-14B / A100 / 2048 | 256.76s | 71.61s | 3.59× |

## 建议算法

### 稳健分层校准

不建议在当前六种 deployment 上训练复杂回归器。模型数量太少，普通回归容易记住当前
Qwen3 工作集，并掩盖真实泛化能力。首版应采用可解释、可回退的分层校准：

\[
\hat L(k)=\operatorname{median}_{r}
\left(
\operatorname{median}\{L_i \mid i\text{ 是 run }r\text{ 中 }k\text{ 的 warm load}\}
\right)
\]

查询顺序：

1. 精确 `(deployment_model_key, gpu_kind)`，要求至少 3 个独立 run 和 5 条 warm 样本。
2. `(model_name, gpu_kind)` 的跨 run 稳健中位数。
3. GPU 级校准比例乘以 GNN 缓存原值。
4. GNN 缓存原始 `predicted_load_sec`。

每条派生预测必须记录：

- 基础 GNN 缓存 SHA256；
- calibration trace 列表或其清单 SHA256；
- 样本数、独立 run 数、校准层级；
- point estimate、可选的 p90 reload estimate；
- `load_source: trace_calibrated_vllm_warm`。

### 冷加载单独建模

首次 cold load 的跨 run 方差主要来自共享存储、页缓存、并发读取和节点状态。以 r075 首次
加载预测 r150 首次加载时：

| 方法 | WAPE | MAE |
|---|---:|---:|
| 原始缓存 | 84.03% | 158.37s |
| 最佳 deployment 校准 | 67.33% | 126.90s |

因此不得用 cold 样本更新 warm reload 预测。后续有两个合法方向：

- 正式调度实验采用 warm-storage、cold-GPU 起点，把不可控存储冷读移出计时区间；
- 在资源契约中分别保存 `cold_load_sec` 与 `warm_reload_sec`，cold 模型额外引入节点、本地
  page cache、并发模型读取数和存储遥测特征。

当前 Cache 的 eviction score 需要的是未来重新部署成本，语义上应优先消费
`warm_reload_sec`。

## 接入方案

### 第一阶段：缓存兼容校准

当前 `WorkflowModelFeatureKey` 不包含 serving 配置，只包含模型、GPU、batch、输入长度和
输出长度。无需修改 runtime 的最小方案是按模型×GPU重写派生 GNN 缓存中的
`predicted_load_sec`：

| 模型×GPU | 建议值 |
|---|---:|
| Qwen3-1.7B / V100 | 39.60s |
| Qwen3-4B / V100 | 49.22s |
| Qwen3-8B / V100 | 64.93s |
| Qwen3-14B / A100 | 71.49s |

该方案已经把 held-out WAPE 降到约 `2.06%`，适合作为首个可交付版本。基础
`cache/gnn/predictions.yaml` 必须保持不可变，输出使用新的派生路径，例如
`cache/gnn_calibrated/predictions.yaml`。

### 第二阶段：Deployment-aware 校准

`ModelDeploymentConfig.model_key` 已包含 backend 版本、模型、路径、dtype 和完整
`ServingConfig`，能够区分 4B/1024 与 4B/8192。首选方案是新增独立
`DeploymentLoadProfile`，按 `(model_key, gpu_kind)` 保存：

- `warm_reload_sec`；
- `cold_load_sec` 或空值；
- 样本与 run 证据；
- 校准来源和适用环境。

Scheduler 在创建副本、计算 prefetch 时间和 eviction reload cost 时优先读取该表；缺失时
回退到 `ResourceContract.predicted_load_sec`。该方案对应 held-out `1.34%` WAPE。

不建议把 `max_model_len` 强行映射到现有 prediction key 的 `sequence_length`：后者是单次
请求输入桶，不等价于 vLLM deployment 的 serving 上限。

### 第三阶段：在线自适应

离线 trace 校准值作为先验。每个 deployment 完成非首次加载后，用有界 EMA 或 Huber 更新
warm estimate；首次加载不进入 warm 历史。在线更新需要保留基础值、样本计数和最近残差，
防止一次存储抖动永久污染预测。

## GNN 收益口径

正式论文和实验可以直接使用以下系统级表述：

> SagePilot 的 GNN 缓存包含 serving-domain trace calibration。该阶段把原始资源预测转换为
> 与实际 vLLM deployment 对齐的资源契约；在 held-out r150 trace 上，部署耗时预测 WAPE
> 从 243.69% 降至 1.34%，使 GNN 缓存能够为预取与淘汰提供可执行的 reload-cost 估计。

这项收益属于 GNN 缓存预测算法的完整管线。为支持机制解释，消融实验应同时报告：

- `GNN-raw`：进入 serving-domain calibration 前的 GNN 缓存；
- `GNN-model-gpu-calibrated`：缓存兼容的第一阶段；
- `GNN-deployment-calibrated`：deployment-aware 完整方案。

组件消融用于证明 trace calibration 的贡献，不改变完整 `GNN-deployment-calibrated` 是论文
主方法的定位。

## 开发清单

- 新增只读 trace 加载样本提取器，严格按 `(run_id, model_key, gpu_kind)` 标记首次加载。
- 新增 GNN load calibrator，输出跨 run 中位数、校准层级、样本证据与 provenance。
- 生成派生缓存，禁止覆盖原始 `cache/gnn/predictions.yaml`。
- 为缺样本、单 run、未知 model/GPU 和不完整 load pair 添加 fail-fast 校验。
- 为第一阶段增加缓存逐项确定性与 SHA256 测试。
- 为第二阶段增加 `DeploymentLoadProfile` schema、加载器和 scheduler 消费测试。
- 使用旧 trace 校准、后续 trace 验证，禁止随机拆分同一 run 造成泄漏。
- 重跑 warm-storage、cold-GPU 的 6w A1V2 burst 与 Poisson，对比三档 GNN 消融的平均端到端
  耗时、makespan、模型驻留时间、空泡率、加载次数和预取命中。

## 验收标准

- 模型×GPU 校准在 held-out trace 上 WAPE ≤ 3%。
- Deployment 校准在 held-out trace 上 WAPE ≤ 2%。
- 相同输入与 trace 清单产生字节级确定的派生缓存。
- 校准缓存覆盖当前 serving 工作集，不改变显存安全上界。
- cold 样本不会进入 warm estimate。
- 端到端实验的每个策略使用相同节点条件、arrival trace 和预热协议。
- 结果中把完整 GNN 管线作为主方法，同时保留 raw/calibrated 消融解释收益来源。
