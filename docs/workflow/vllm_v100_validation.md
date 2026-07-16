# vLLM V100 验证记录

## 环境与配置

- 日期：2026-07-13
- GPU：NVIDIA Tesla V100-SXM2-32GB，compute capability 7.0
- 驱动：580.126.16
- serving 环境：Python 3.12、vLLM 0.10.2、Torch 2.8.0+cu128、Transformers 4.55.2、XFormers 0.0.32.post1、Ray 2.55.1
- 模型：`/data/Models/Qwen/Qwen3-0.6B`
- 引擎：V0、XFormers、FP16、eager、TP=1、PP=1
- 容量：`max_model_len=1024`、`max_num_seqs=3`、`max_num_batched_tokens=1536`、`gpu_memory_utilization=0.98`
- 生成：greedy，三个请求各生成 32 tokens

环境安装命令：

```bash
uv sync --project envs/vllm-v100 --frozen
```

基准命令：

```bash
PYTHONPATH=src envs/vllm-v100/.venv/bin/python \
  scripts/workflow/benchmark_vllm_v100.py --gpu-id 2 --max-new-tokens 32
```

## 结果

| 指标 | 顺序执行 | 三请求并发 |
|---|---:|---:|
| 总输出 tokens | 96 | 96 |
| 总耗时 | 2.3568 s | 0.8967 s |
| 吞吐 | 40.73 tok/s | 107.06 tok/s |

观测吞吐提升为 2.63×。该比值只作记录，不作为跨机器硬性验收阈值。

结构性验收结果：

- `peak_active_requests=3`
- 三个请求执行区间重叠
- 三个请求均成功，`finish_reason=length`
- vLLM 明确报告 V0、XFormers 和 FP16
- 0.98 配置预留约 29.84 GiB KV cache
- worker 退出后 GPU 显存从 0 MiB 回到 0 MiB

Ray `runtime_env.py_executable` 也使用同一环境完成验证。主环境 Transformers 5.x 生成 prompt token IDs，Ray 副本峰值并发为 3；显式 shutdown 后 kill actor，GPU 显存回到 0 MiB，未残留 ModelReplica 或 vLLM 进程。

## 边界

本次只验证 Qwen3 与 V100。A100 沿用相同 V0/XFormers/FP16 配置，但在实际部署前仍需运行相同冒烟测试。vLLM 新版本要求更高 compute capability，因此该 serving 环境必须保持锁文件固定，升级前重新执行 V100 门禁。
