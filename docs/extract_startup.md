# 特征提取启动说明

## 当前状态

当前仓库已经支持单进程执行 CSV 到 PyG prepared dataset 的转换：

```bash
uv run python src/gnn_model/data/extract.py \
  --csv_dir csv \
  --output_dir output/gnn_model_prepared \
  --val_ratio 0.2 \
  --test_ratio 0.2 \
  --seed 42
```

生成结果位于 `output/gnn_model_prepared/`：

```text
manifest.json
train.pt
val.pt
test.pt
```

每行 CSV 通过 `result_json` 和 `variant_name` 定位 ONNX：`result_json` 所在 target 目录下的 `onnx_models/<variant_name>.onnx`。
`id_fields` 和 `target_fields` 由提取代码内置，不再读取 YAML 配置。

训练时使用已归一化 split dataset：

```yaml
experiment_name: gnn_model_scaled_train
data:
  kind: split
  data_dir: data/scaled
  target_names:
    - duration_sec_avg
    - gpu_util_percent_p95
    - gpu_sm_occupancy_percent_p95
    - gpu_mem_used_mb_p95
  scaler_dir: data/scalers
model:
  hidden_dim: 64
  num_layers: 2
  num_heads: 4
  dropout_rate: 0.1
training:
  batch_size: 4
  num_epochs: 1
  learning_rate: 0.001
  weight_decay: 0.0001
```

```bash
PYTHONPATH=src uv run python -m gnn_model \
  --config config/gnn_model/scaled_training.yaml \
  --output_dir output \
  --device cpu
```

## 单进程模式

单进程模式适合调试、抽样验证和首次确认数据契约。它不依赖 Ray，也不需要提前启动任何集群。

推荐先用一个较小的 CSV 目录或临时输出目录验证：

```bash
uv run python src/gnn_model/data/extract.py \
  --csv_dir csv \
  --output_dir output/gnn_model_prepared_smoke
```

确认 `manifest.json` 中的关键字段：

```text
target_names
split_files
split_counts
sample_count
```

## Ray 模式

当前仓库的 `src/gnn_model/data/extract.py` 尚未接入 Ray 参数；下面记录的是源项目使用习惯和后续接入时的运行约定。

旧项目中的使用方式是先准备 Ray worker，再由 Ray master 进入数据转换主入口。这样 worker 先处于可连接状态，master 侧主流程负责枚举 CSV、分发行级或 batch 级转换任务，并收集 `Data` 列表落盘。

### Worker 节点

在每台 worker 上先清理旧 Ray 进程：

```bash
ray stop
```

启动 worker 并连接 master：

```bash
ray start \
  --address <master_ip>:6379 \
  --num-cpus <worker_cpu_count>
```

如果 worker 和 master 不在同一网络命名空间，先确认端口和防火墙：

```bash
nc -vz <master_ip> 6379
```

### Master 节点

在 master 上清理旧 Ray 进程：

```bash
ray stop
```

启动 head 节点：

```bash
ray start \
  --head \
  --port 6379 \
  --dashboard-host 0.0.0.0 \
  --num-cpus <master_cpu_count>
```

确认集群节点已注册：

```bash
ray status
```

### 转换主入口

Ray 模式接入后，主入口应仍由 master 执行，worker 不直接读取 CLI 参数：

```bash
MASTER_ADDR=<master_ip>:6379 \
uv run python src/gnn_model/data/extract.py \
  --csv_dir csv \
  --output_dir output/gnn_model_prepared_ray
```

后续如果为当前仓库补 Ray 参数，建议最小接口保持为：

```bash
uv run python src/gnn_model/data/extract.py \
  --csv_dir csv \
  --output_dir output/gnn_model_prepared_ray \
  --process_mode ray \
  --ray_address <master_ip>:6379
```

## 输出检查

无论单进程还是 Ray，输出契约都应一致：

```text
output/gnn_model_prepared/
  manifest.json
  train.pt
  val.pt
  test.pt
```

最小验收标准：

- `manifest.json` 存在且可被 `PreparedDataConfig` 加载。
- `sample_count > 0`。
- `split_counts.train`、`split_counts.val`、`split_counts.test` 都大于 0。
- 每个 `.pt` 文件内都是 `list[torch_geometric.data.Data]`。
- 每个 `Data.y` 的最后一维等于 `manifest.target_names` 数量。
