# 弹性模型副本 + 串行节点融合（2026-07-27）

两个新机制的实现记录。融合已有真机结论，弹性已实现但本轮未做对照实验。
代码：`src/workflow/{scheduler,policy,artifacts,fusion,worker}.py`。

面向论文动机章节的写法见 `motivation_20260727_adjacent_same_model_calls.md`，
它和本文用的是同一批 18 个运行记录，只是换了提问角度。

## 1. 基线重新冻结（前置改动，无行为变化）

`select_eviction_victim`（`policy.py`）用 `replica_id` 做 tiebreak，而 `replica_id` 原本是无种子的 `uuid4()`。`_future_reuse_distance` 在「没有 session 要这个模型」时返回 `inf`，打平很常见，**因此此前的实验在驱逐决策上本来就不是 run-to-run 确定的**。

- `replica_id` 改为 `SchedulerCore` 内单调计数器（`r0001`、`r0002`…）。
- `SchedulerConfig` 新增四个弹性字段，`scheduler_config_sha256` 因此改变。

**影响：** 新 run 与 `output/serve_0726` 归档 trace 的 `replica_id` 命名不同、对齐指纹不同；所有指标口径不变。基线自本次提交起重新冻结。

## 2. 串行节点融合

### 2.1 实现

`FusedAgentNodeConfig` **继承** `AgentNodeConfig` 并覆盖判别字段，因此 `scheduler.py` 中十余处 `isinstance(node, AgentNodeConfig)` 分支（placement、model_key、预取、关键路径）全部无需改动。

`src/workflow/fusion.py::fuse_workflow` 是纯 `Workflow → Workflow` 重写，由 `master.py --fuse-nodes` / `serve_submit.sh` 的 `FUSE_NODES=1` 触发。融合条件：极大链上每一条内部边都是 1→1、`model_key` 相同、`ExecutionConfig` 除 `max_new_tokens` 外全等。**链头入度与链尾出度不设限**。

运行时（`worker.py::execute_fused_agent`）**一次 `begin_node`、一次 `request_acquire`、一个 grant 跑完全链**，各 stage 用 `f"{acquire_id}#{i}"` 作副本内请求 id。准入档位取各 stage 上界（`estimate_chain_input_tokens`），ETA 取各 stage 契约之和（`_chain_run_sec`）。逐 stage 记录进 `AgentTaskRuntimeReport.stage_reports` → trace payload `stages` → `analysis.py::_stage_intervals` 展开回原始节点名，融合臂与未融合臂的 per-node 指标因此仍可比。

落地时踩到两个坑，都已修并有回归测试：

1. `NodeTaskRecord.node_kind: Literal["agent","function"]` 曾直接取 `node.type`，融合节点传入 `"fused_agent"` 直接 `ValidationError`。改为按 `isinstance(node, FunctionNodeConfig)` 派生。
2. 融合会把链头改名，**`routing: targeted` 的函数节点按原始后继名返回的 key 就对不上了**。`worker.py::_resolve_target` 现在按 `FUSED_NAME_SEPARATOR` 前缀唯一匹配回落。

### 2.2 workflow 形状决定融合收益

融合量是编译期确定的，不需要跑实验。三种 QMSum 形状的静态对比（每 workflow 10 session）：

| 形状 | 可融合链 | 节点数 | 每 session 省 acquire/跳 | 会话内 4B 并发 |
|---|---|---|---|---|
| fan-out 6 + 8B 链 3（旧设计） | 1 条 (3→1) | 10→8 | 2 | 6 |
| `qmsum_roll1` 纯串行 6 + 8B 链 3 | 2 条 (6→1, 3→1) | 9→2 | 7 | 1 |
| **`qmsum_lane1` 2 lane × 深 3 + 8B 链 3** | **3 条 (3→1 ×2, 3→1)** | **10→4** | **6** | **2** |

旧的 fan-out 设计只有 2 次可省——**fan-out 的兄弟节点不是融合对象**，融合严格只吃串行链。`qmsum_lane1` 拿到纯串行 86% 的收益，同时保留 2 路并发。实验用 `qmsum_lane1`。

滚动 lane 的每个 stage 从 session inputs 读自己的 `slice_i`（`build_qmsum_session_inputs` 现在额外产出 `slice_0..5`），因此不需要在链中间传切片。

### 2.3 真机结果

三个集群（6661 / 6662 / 6664，各 1×A100 + 2×V100）。负载 `replay_lane_small.yaml`：`qmsum_lane1` + `mbpp1` 各 10 session，Poisson λ=0.02/s，`arrival_seed: 42`。**3 种调度策略 × 2 臂（融合开/关）× 3 次重复 = 18 个 run**，每个 run 20/20 session 完成、0 失败。trace 全部保留在 `output/fusion_exp/<run_id>/`。

三种策略的定位：`fifo` 是纯到达序准入（Parrot 在「请求排序」这一维度上的对照），`kairos` 是 Kairos 复现（critical-path SRPT + 显存感知放置、无预取、LRU 驱逐），`cache` 是本系统全量方法。**融合是编译期图重写，与调度策略正交**，所以三种策略下都能开关。

#### 编译期就能算出、18 个 run 全部一致（零方差）

| | base | fuse |
|---|---|---|
| qmsum acquire 次数 | 90 | **30** |
| qmsum 队列传输 | 110 | **50** |

10 session × 每 session 省 6 次。这两个数字不提供新信息，只确认算术。

#### 必须实测的

| 策略 | 指标 | base（3 run） | fuse（3 run） | Δ | 区间 | 配对胜 |
|---|---|---|---|---|---|---|
| **fifo** | qmsum mean | 144 [143–146] | 125 [120–129] | **−13%** | 不重叠 | 3/3 |
| | qmsum p95 | 240 [223–250] | 215 [210–217] | **−11%** | 不重叠 | 3/3 |
| | acquire 等待合计 | 1338 [1321–1370] | 1070 [981–1187] | **−20%** | 不重叠 | 3/3 |
| **kairos** | qmsum mean | 177 [175–179] | 127 [106–142] | **−28%** | 不重叠 | 3/3 |
| | qmsum p95 | 342 [293–366] | 209 [179–225] | **−39%** | 不重叠 | 3/3 |
| | acquire 等待合计 | 1791 [1784–1801] | 1071 [823–1239] | **−40%** | 不重叠 | 3/3 |
| **cache** | qmsum mean | 160 [145–173] | 124 [107–158] | −22% | 重叠 | 3/3 |
| | qmsum p95 | 281 [222–313] | 191 [181–210] | **−32%** | 不重叠 | 3/3 |
| | acquire 等待合计 | 1447 [1277–1689] | 1106 [802–1692] | −24% | 重叠 | 2/3 |

握手延迟（grant → 开始生成）合计：cache 27s → 7s（−74%）；fifo 与 kairos 本来就只有 2–5s，融合后仍是 2s。

#### 站得住的结论

1. **融合的收益与调度策略无关。** 三种策略、九组对照，融合在每一组的配对比较里都赢，且 fifo 与 kairos 下三个指标的区间全部不重叠。这直接回答了「收益是不是靠你自己的调度器堆出来的」——不是。
2. **收益大小取决于该策略本身有多少排队损失。** kairos 的 base 排队最重（1791s），融合收益最大（−40%）；fifo 的 base 排队最轻（1338s），收益最小（−20%）。融合削掉的是「排队事件的个数」，所以基线排队越重，省得越多。
3. **fifo 与 kairos 的 base 近乎确定性**（fifo 143/145/146，kairos 175/179/178），而 cache 的 base 抖动明显（145/173/161）。cache 的预取与预测式驱逐是方差来源，这也是 cache 组唯一出现区间重叠的原因。

#### 不能写的

- **makespan 18 个 run 全部落在 695–728s**，负载是到达速率受限而非吞吐受限，makespan 区分不了任何东西。
- **cache 组的 mean 与 acquire 等待区间重叠**：`fuse_r2` 是离群点（等待合计 1692s），它和 `base_r2` 一样加载了 13 次模型（其余 run 10–11 次）。模型加载次数是该组的主导噪声源，与融合无关。
- **本实验不比较策略之间的优劣。** 顺带可见 fifo 的 base（144s）优于 cache（160s）与 kairos（177s），但这与 `pool_tightness_results_20260726.md` 的已知问题同源，需要单独实验才能下结论，不在本文范围内。

## 3. 弹性模型副本（已实现，本轮未做对照实验）

`_replica_pairs: dict[pair, str]` → `ReplicaGroups`。`elastic_replicas=False` 时组容量硬为 1，路径与此前逐字相同；`elastic_replicas=True` 时强制 `policy == "cache"`。

扩容判据是队列排空时间与冷启时间的比较：

```
best_existing_ect = 队列排空时间 + run_sec
new_ect           = load_sec + run_sec
warranted  iff  best_existing_ect - new_ect > scale_out_margin_sec
```

`slots` 按缓存实际覆盖的 batch 档位算（不是 `max_num_seqs`），`depth` 为组内在飞 lease + 排在本请求之前的 pending。真机缓存下的算术（Qwen3-4B/V100，`run=29.3s`、`load=49.9s`、`slots=3`）：

| 并发 session | 排队请求 | rounds_ahead | gain | 扩容 |
|---|---|---|---|---|
| 1 | 6 | 1 | −49.9s | 否 —— 等一轮 29s 比冷启 50s 划算 |
| 2 | 12 | 3 | **+37.8s** | **是** → 2 副本、6 slots |
| 扩容后 | 12 | 1 | −49.9s | 否 —— 边际收益自动收敛 |

最后一行是副本数不靠配额就能收敛的原因。反垄断三重闸门：空闲卡直接放行 / 抢占需 `gain > victim.reload_cost_sec` / 扩容后每个有需求且无驻留的模型仍须保留至少一张可行卡。缩容用 `EvictionCandidate.redundant` 作首要排序键 + `drain_requested_at` 排空回收。

**注意：** `gain > reload_cost` 这一条在常见情形下是唯一起作用的约束——`_future_reuse_distance` 对无人需要的模型返回 `inf`，所以 `reuse_distance > gain` 恒真。

一次早期联合实验（融合 × 弹性 2×2）显示 `both` 臂并不优于 `fuse` 单独臂，两个机制互相污染。该结果已删除，不作为结论。**弹性要单独设计对照实验**，且需要一个能持续制造深队列的 fan-out 负载——这与融合偏好串行链正好相反，是两个机制的固有张力。

## 4. 复现

```sh
# 静态融合分析（不需要 GPU）
PYTHONPATH=src uv run python -c "
import yaml; from pathlib import Path
from workflow.fusion import find_fusable_chains
from workflow.schema import Workflow
wf = Workflow.model_validate(yaml.safe_load(Path('config/workflow/serve/qmsum_lane1.yaml').read_text()))
print([[n.name for n in c] for c in find_fusable_chains(wf)])"

# 一次真机 run（融合臂；去掉 FUSE_NODES 即基线臂）
WORKFLOW_FILES=config/workflow/serve/qmsum_lane1.yaml,config/workflow/serve/mbpp1.yaml \
RAY_PORT=6662 PYTHONHASHSEED=0 MIN_GPUS=3 GPU_MEM=v100=32768,a100=81920 \
EXPERIMENT_CONFIG=config/workflow/serve/replay_lane_small.yaml \
PREDICTIONS=/home/wangjh/gnn_predict/cache/profile_v2/predictions.yaml \
SCHED_CONFIG=config/workflow/serve/scheduler_cache.yaml FUSE_NODES=1 \
OUTPUT_DIR=output/fusion_exp RUN_ID=fuse_r1 sh scripts/workflow/serve_submit.sh
```

三种策略各跑一遍融合开关（`SCHED_CONFIG` 换成 `scheduler_fifo.yaml` / `scheduler_kairos.yaml` / `scheduler_cache.yaml`），每种 3 次重复，即本文的 18 个 run。

## 5. 下一步

1. cache 组的 mean 区间要分开，需控制模型加载次数或加重复次数（fifo 与 kairos 已经分开，不需要）。
2. 把负载压到吞吐受限，makespan 才有区分度。
3. 弹性的对照实验单独设计，配 fan-out 负载。
