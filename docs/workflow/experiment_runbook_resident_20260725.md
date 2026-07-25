# 常驻 Ray 集群 · workflow 实验提交 Runbook（2026-07-25）

把"起集群"和"跑实验"解耦：**开发容器常驻 N 个 Ray Head**（每集群一端口），**作业平台常驻
worker**（每卡一 Pod），实验从**开发容器串行提交**、跑完一个自动接下一个，Head 与 worker 全程常驻。

与旧 `serve_master.sh`（`ray start --head` + 跑实验一体、作业平台一次性作业）的区别：这里 Head/worker
跨实验复用，不必每换一个 `RUN_ID`/`SCHED_CONFIG` 就重投作业重起集群。`serve_master.sh` 仍保留，供旧
一次性流使用。

底层保证：`workflow.master` 用 `ray.init(address=...)` **attach** 已有 Head（`_owns_ray=False`），结束时
`fleet.shutdown()` 只驱逐自己的副本 + 杀自己的 scheduler/trace actor，**绝不** `ray.shutdown()`，故 Head
与 worker raylet 存活、下一个实验从干净集群复用。

---

## ① 开发容器：起常驻 Head（每集群一次）

```sh
cd /home/wangjh/gnn_predict && . ./.venv/bin/activate
RAY_PORT=6661 sh scripts/workflow/serve_head.sh
RAY_PORT=6662 sh scripts/workflow/serve_head.sh   # 第二个集群（可选，用于并行）
hostname -i                                        # 记下开发容器 IP，worker 要用（如 10.244.18.19）
```

- Head 无 GPU（`--num-gpus=0`，不征用开发卡）、关 dashboard、`temp-dir=/tmp/ray_<port>`，附属端口按
  `RAY_PORT` 派生 → 多 Head 同机互不冲突。
- `ray start --head` 起完守护进程即返回，Head 常驻。**幂等**：该端口已有 Head 时打印 `already up` 跳过。

## ② 作业平台：起常驻 worker（每卡一 Pod）

每个 worker Pod 的启动脚本（**单条前缀内联**赋值，务必别用 `&&` 断开 `RAY_HEAD_ADDR=... && ...`——
未导出的 shell 变量不会传进 `sh` 子进程，且会被 Volcano 注入的 `MASTER_ADDR` 盖掉，导致静默连错地址）：

```sh
cd /home/wangjh/gnn_predict && . ./.venv/bin/activate && RAY_HEAD_ADDR=10.244.18.19 RAY_PORT=6661 sh scripts/workflow/serve_worker.sh
```

- `RAY_HEAD_ADDR` = 开发容器 IP；`RAY_PORT` = 目标集群端口（连 6662 集群就整套换成 6662）。
- 地址优先级 `RAY_HEAD_ADDR` > `MASTER_ADDR` > `localhost`；`RAY_HEAD_ADDR` 专为绕开 Volcano 注入的
  `MASTER_ADDR`（它指向 DDP 作业自己的 master 角色，不是常驻 Head）。
- worker `--block` 常驻，服务该集群的**所有**实验；A100/V100 各按需提交，硬件配比由 worker 组合 +
  提交实验时的 `MIN_GPUS` 控制（14B 只落 A100，至少 1 张 A100）。

## ③ 开发容器：串行提交实验

env 接口与 `serve_master.sh` **完全一致**，只多一个 `RAY_PORT` 选目标集群：

```sh
RAY_PORT=6661 \
  WORKFLOW_FILES=config/workflow/serve/qmsum1.yaml,config/workflow/serve/mbpp1.yaml,config/workflow/serve/gsm8k1.yaml \
  OUTPUT_DIR=output/serve/3w EXPERIMENT_CONFIG=config/workflow/serve/replay_3w.yaml GPU_MEM=v100=32768,a100=81920 \
  RUN_ID=hetero_cache_a1v2 SCHED_CONFIG=config/workflow/serve/scheduler_cache.yaml MIN_GPUS=3 \
  sh scripts/workflow/serve_submit.sh
```

- 跑完 fleet 驱逐副本 + 清 actor + detach，**不拆集群**；下一条 `serve_submit` 复用同 Head/worker。
- **串行接力**：`CMD1 ; CMD2 ; CMD3`（中间**不要** `ray stop`）。要 trial 间冷文件缓存，在中间插
  `sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`。整批可 `nohup ... &` 后台跑。

---

## 约束（重要）

- **一集群同时只能跑一个实验**：每次 `serve_submit` 建独立 fleet 并在结束 `evict_all` 整个共享 GPU 池，
  两实验并发会互相驱逐副本、污染彼此账本。→ 同集群**串行**；要**并行**就用**不同集群**（6661/6662），
  各接自己的 worker 池——这正是"多 Head"的用途。
- **网络**：worker Pod 必须能路由到开发容器 `IP:RAY_PORT` 及 Ray 附属端口（同 pod 网段通常可达，沿用旧
  worker↔master 的连通前提）。

## 集群停用

- `ray stop` 是**全机全局**（只有 `-f/-g`，无 `--address`/`--temp-dir` 作用域），会停掉本机**所有** Ray
  （含所有 Head 与本机 worker）。
- **只停某一个集群**：`pkill -f /tmp/ray_<port>`（该 Head 的进程都带此 temp-dir 标识）；作业平台 worker
  直接停/删对应 Pod。

## 已验证（2026-07-25 · ray 2.55.1 · 开发容器 loopback）

- 同机 `serve_head.sh` 起 6691/6692 两 Head：各自独立、`ray status --address` 各只见自身 1 节点、Head
  显示 0 GPU；重复执行命中 `already up`；`/tmp/ray_6691`、`/tmp/ray_6692` temp-dir 隔离；6 个进程带
  `/tmp/ray_6691` 标识（选择性停用依据）。
- worker 经 `RAY_HEAD_ADDR=127.0.0.1` 连 6691（节点数→2）；连续两次 function-only `workflow.master`
  提交均写出 trace，**两次之间 Head+worker 节点数恒为 2**（复用坐实）。
- agent（vLLM/GPU）实验路径与 `serve_master.sh` 完全相同，沿用其实机门禁与前置（predictions/accelerators/
  model_path/vllm_python）。

## 读结果

沿用现有：`experiment.workflow.analysis.summarize_trial`（指标口径与 sanity check 见
`experiment_runbook_20260724.md` 的"读结果"节）。
