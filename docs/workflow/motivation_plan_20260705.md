# MBPP 动机实验改造方案：n=2 → n=3 Sequential Chain

## 1. 动机

当前 MBPP 动机实验使用 `coder→tester→repair→final_tester` 链（n=2 LLM 节点），
资源缺口 ρ=2.01×，已初步验证 `ρ=n` 的理论。

将其替换为 Kairos 论文中 survey 到的 sequential execution 模式（3 个 agent 的
序列链），使 n=3 从而展示更大的结构性缺口（ρ≥3×），并使实验设计与已知论文的
workflow 形态建立直接对应。

## 2. 改造后 Workflow

```
coder (Qwen3-14B) → tester (CPU) → reviewer (Qwen3-8B) → repair (Qwen3-14B) → final_tester (CPU)
```

n = 3 个 LLM 节点，m = 60 个 session。

节点职责：

| 节点 | 类型 | 模型 | 输入 | 输出 |
|---|---|---|---|---|
| coder | LLM | Qwen3-14B | 编程题 + 测试用例 | 初始代码 |
| tester | deterministic | CPU | 初始代码 | 测试结果（pass/fail + 错误信息） |
| reviewer | LLM | **Qwen3-8B** | 编程题 + 初始代码 + 测试结果 | 代码审查意见 |
| repair | LLM | Qwen3-14B | 编程题 + 代码 + 审查意见 + 测试结果 | 修复后的代码 |
| final_tester | deterministic | CPU | 修复后的代码 | pass@1 判定 |

### 2.1 设计依据

Reviewer 使用 Qwen3-8B（小于 coder/repair 的 14B），基于以下考虑：

- Kairos 论文的 core observation：不同 agent 角色的输出长度、推理延迟和显存需求
  存在显著差异（Section 2.1）。heterogeneous 模型分配更真实地反映这一观察。
- Reviewer 的任务（分析已有代码）比 coder/repair（生成/修复代码）认知负荷更低，
  8B 是合理选择。
- 纯同构 3×14B 链在单 V100 32GB 上**不可部署**（3×~28GB > 32GB），正好说明
  本地资源受限是真实约束。

## 3. 指标定义

沿用与当前实验一致的口径：

| 指标 | 定义 |
|---|---|
| `active_model_seconds` | Σ 各 LLM 节点耗时 = coder dur + reviewer dur + repair dur |
| `resident_model_seconds` | 3 × session 跨度 | 
| `resource_gap ρ` | resident / active ≥ 3 |
| `node_idle_time` | resident − active |
| `pass@1` | final_tester 通过率 |

## 4. 与当前实验的指标对比

| | 当前 (n=2) | 改造后 (n=3) |
|---|---|---|
| Workflow | coder→tester→repair→final_tester | coder→tester→reviewer→repair→final_tester |
| LLM 节点数 | 2 | 3 |
| 理论 ρ | 2 | 3 |
| 预期 ρ 实测 | ~2.01 | ~3.01 |
| reviewer 模型 | — | Qwen3-8B |

## 5. 需要修改的文件

### 5.1 runner.py：新增 reviewer，改用 3 GPU 独立实例

当前 `run_mbpp_chain` 使用单个 `LocalQwenGenerator`（所有节点共享同一 GPU）。
改造后改为 3 个独立 generator，各绑定一张 GPU，与 QMSum 3-way 的部署模式一致。

```python
generators = {
    "coder": LocalQwenGenerator(model_name="qwen3-14b", model_path=QWEN3_14B, device="cuda:0", ...),
    "reviewer": LocalQwenGenerator(model_name="qwen3-8b", model_path=QWEN3_8B, device="cuda:1", ...),
    "repair": LocalQwenGenerator(model_name="qwen3-14b", model_path=QWEN3_14B, device="cuda:2", ...),
}
```

在 `run_mbpp_sample` 中，于 tester 和 repair 之间插入 reviewer 调用。

当前 flow（`run_mbpp_sample` 第 180-197 行）：

```
coder_result = generator.generate(mbpp_coder_prompt(sample))
first_eval = evaluate_mbpp(...)
repair_result = generator.generate(mbpp_repair_prompt(sample, coder_result.text, first_eval))
final_eval = evaluate_mbpp(...)
```

改造后 flow（串行执行，每个节点用对应 generator）：

```
coder_result = generators["coder"].generate(mbpp_coder_prompt(sample))
first_eval = evaluate_mbpp(...)
reviewer_result = generators["reviewer"].generate(mbpp_reviewer_prompt(sample, coder_result.text, first_eval))
repair_result = generators["repair"].generate(mbpp_repair_prompt(sample, coder_result.text, first_eval, reviewer_result.text))
final_eval = evaluate_mbpp(...)
```

串行而非并行执行——这模拟了 LangGraph 默认行为（节点按拓扑顺序执行），
与 QMSum 的并行 fan-out 不同。串行 + 3 GPU 独立实例正是产生 ρ≥3 的原因：
3 张 GPU 全部被占用，但任意时刻只有 1 张在做计算。

新增函数 `mbpp_reviewer_prompt`：

```python
def mbpp_reviewer_prompt(sample: TaskSample, code: str, evaluation: Any) -> str:
    tests = "\n".join(sample.test_imports + sample.test_list)
    return (
        "Review the Python solution and identify issues that cause test failures.\n"
        "Focus on: (1) logic errors, (2) edge cases, (3) API misuse.\n"
        "Return only the review analysis, no code.\n\n"
        f"Task:\n{sample.input_text}\n\n"
        f"Current solution:\n{code}\n\n"
        f"Test result:\n{evaluation.model_dump_json()}\n\nTests:\n{tests}"
    )
```

修改 `mbpp_repair_prompt` 签名，新增 `review: str` 参数：

```python
def mbpp_repair_prompt(sample: TaskSample, code: str, evaluation: Any, review: str) -> str:
    tests = "\n".join(sample.test_imports + sample.test_list)
    return (
        "Repair the Python solution so that it passes the tests.\n"
        "Return only Python code in one fenced code block.\n\n"
        f"Task:\n{sample.input_text}\n\nCurrent solution:\n{code}\n\n"
        f"Code review:\n{review}\n\n"
        f"Test result:\n{evaluation.model_dump_json()}\n\nTests:\n{tests}"
    )
```

trace event 发送修改：在 event list 中插入 reviewer 的 `model_event(...)`，
residency 参数与 coder/repair 一致（resident_started_at=session_start, 
resident_ended_at=session_end）。

### 5.2 analysis.py：更新汇总逻辑

直接修改 `summarize_mbpp` 和 `mbpp_sample_metrics`：

- 新增 reviewer 的 `single_event(events, "reviewer")`
- 新增 reviewer 时长计入 `active_model_seconds`
- `resident_model_seconds` 因子从 2.0 改为 3.0
- `frontier_gap`: 2.0 → 3.0
- `resident_model_count_peak`: 2 → 3

```python
def mbpp_sample_metrics(sample_id, events):
    coder = single_event(events, "coder")
    reviewer = single_event(events, "reviewer")
    repair = single_event(events, "repair")
    final_tester = single_event(events, "final_tester")
    start = float(coder["started_at"])
    end = float(final_tester["ended_at"])
    active = float(coder["duration_sec"]) + float(reviewer["duration_sec"]) + float(repair["duration_sec"])
    resident = 3.0 * (end - start)
    return { ... "resource_gap": resident / active }
```

### 5.3 plotting.py：更新图函数

`plot_mbpp_gap` 当前硬编码了 n=2 标签（"Static residency" vs "Active frontier"）。
改造后需要更新：

- 标签不变（抽象描述，不依赖具体 n 值）
- 数据来源调整为新的 summary key（如果 field 名不变则无需改）

### 5.4 report.py：更新文字

图注中的 n=2 相关数值和文字改为 n=3：
- "两个 14B 模型实例" → "三个模型实例（两个 14B、一个 8B）"
- "资源浪费为 2.010x" → 更新为实测值
- "frontier gap: 2.0" → "frontier gap: 3.0"

### 5.5 run_motivation.py：新增命令行选项

为了向后兼容，建议新增 `--mbpp-chain-length 3` 参数或独立入口：

- 新增 `--mbpp-chain-length`（默认 2，支持 2/3）
- 根据该值选择调用 `run_mbpp_chain`（n=2）或 `run_mbpp_3chain`（n=3）
- 输出文件路径区分：`mbpp_chain_trace.jsonl` vs `mbpp_3chain_trace.jsonl`

### 5.6 workflow.tex：更新理论分析对照

表格中 MBPP 行更新为 n=3 的实测值。

## 6. 执行计划

| 步骤 | 改动 | 验证 |
|---|---|---|
| 1 | runner.py: 新增 `mbpp_reviewer_prompt`, 修改 `run_mbpp_sample` | `uv run python -c "from scripts.motivation.runner import mbpp_reviewer_prompt; ..."` |
| 2 | runner.py: 修改 `run_mbpp_chain` → `run_mbpp_3chain`，trace 包含 reviewer | preflight 1 sample 跑通 |
| 3 | analysis.py: 新增 `summarize_mbpp_3chain` | 对 preflight trace 跑分析 |
| 4 | plotting.py: 确认 `plot_mbpp_gap` 兼容性 | 生成图检查 |
| 5 | report.py: 更新文字 | 生成 report 检查 |
| 6 | run_motivation.py: 新增入口 + 选项 | 通过 --skip-preflight 验证完整流程 |
| 7 | workflow.tex: 更新表格 | LaTeX 编译检查 |
| 8 | 正式跑 60 样本 | ~1 小时 |

## 7. 已确认决策

| 决策 | 选择 | 理由 |
|---|---|---|
| [A] Reviewer 模型规格 | **Qwen3-8B（异构）** | 与 Kairos inter-agent difference 对齐 |
| [B] Reviewer 接收测试结果 | **是** | 使 reviewer 分析更有针对性 |
| [C] 模型实例分配 | **3 张 GPU 独立部署** | 暴露 static residency 问题 |
| [D] 实验命名 | **直接覆盖 `mbpp_chain`** | 不需要向后兼容 |
| [E] pass@1 跟踪 | **仅最终 pass@1** | 动机实验关注资源缺口 |
