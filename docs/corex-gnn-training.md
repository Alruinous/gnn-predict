# BI-V150 数据提取与 GNN 训练

本流程只适用于已经取得 BI-V150 变体结果 JSON、`.pt2` 图和监控 CSV 的实验。GNN 在另一台机器训练；BI-V150 只负责模型测量和图导出。V100/A100 的九目标数据集与本流程分开制作，不要混在同一数据集中。

## 1. 检查采集结果

一个完成的变体通常有训练和推理两条 CSV 记录，共用一份 `fx_graphs/<变体名>.pt2`。监控可能跳过缺样本的阶段，所以 83 个变体不保证得到 166 条有效记录。提取器会忽略无图、缺少目标值或 GPU 指标不合理的记录；`manifest.json` 中的 `quality_report` 记录原始、过滤和保留行数。`split_unit: gpu_model_variant` 表示同一变体的所有阶段不会跨训练、验证和测试集。

新生成的 `monitor.csv` 会把 `result_json` 写成**相对 CSV 所在目录**的路径。例如 CSV 位于 `output/resnet50/monitor.csv` 时，值通常是 `results/xxx_results.json`。提取器据此查找同级 `fx_graphs/<变体名>.pt2`。复制到另一台机器时，保持 `monitor.csv`、`results/`、`fx_graphs/` 三者的相对目录结构即可，不需要保留原容器的 `/home/...` 路径。旧 CSV 中的项目根目录相对路径或原容器绝对路径也会尝试映射到 CSV 同级的 `results/`、`fx_graphs/`；如果你把 CSV 单独移到别处，则仍需更正路径或恢复目录结构。不要把多个重复运行的同名变体直接混合进同一数据集，提取器会拒绝重复的“卡／模型／变体／阶段”样本。

## 2. 在原 BI-V150 容器提取图特征（当前可执行步骤）

BI-V150 的 `.pt2` 是由 `torch 2.4.1+corex.4.3.0` 写出的。项目的图加载器要求 PyTorch 主、次版本相同；**不能假定**在默认的 PyTorch 2.9 训练环境里可直接读取。先用一份真实 `.pt2` 在拟用于提取的环境中测试读取和特征构建。如果失败，使用与导图版本匹配的 PyTorch 2.4 环境提取，而不是关闭版本校验或覆盖厂商版 torch。

完整 ResNet50 配置用 `example_input_shape` 的 batch 1 导出静态推理图，并以 batch 1 测推理、以 `batch_size: 16` 测训练。因此，`.pt2` 中节点形状、MACs、激活内存等**静态图指标是导图 batch 1 的参考值**，并不冒充 batch 16 训练图。提取时会分别保留 `graph_capture_batch_size=1` 与监控行的实测 `batch_size`；图级特征中的 `batch_size` 使用实测值（训练 16、推理 1）。训练阶段的实际耗时和资源目标仍来自 batch 16 的监控，不能把 CSV 的训练 batch 改成 1。当前还没有导出反向传播图，因此训练预测的静态结构信息仅来自共用的推理图；评估预测误差时应考虑这一限制。

提取环境还需 `polars`、`torch-geometric`、`pydantic`、`PyYAML`。它可以是原 BI-V150 容器，也可以是另一台可读取该图的 CPU 机器；提取计算主要使用 CPU，**不要求在天数卡上训练 GNN**。**不要在 CoreX 容器执行 `uv sync` 或不加限制地执行 `pip install -r ...`**，它们可能按项目配置安装标准 PyTorch 2.9，覆盖已验证的厂商版 torch。若重新申请容器，先确认它仍将同一份 Ceph Home 挂载到 `/home/luoruian26`，且能看见本次 `monitor.csv`、`results/` 和 `fx_graphs/`。更换提取容器不会改变之前在 BI-V150 上测得的时间和资源标签，但新镜像的软件环境不会因 Home 持久化而自动一致。

以下命令都在**原 BI-V150 容器的终端**执行；每次打开新终端，至少重新执行 `cd` 和 `export PYTHONPATH`，除非已把这些路径写入 shell 启动配置。先确认使用的是包含数据提取修改的最新项目代码，并检查输入文件和当前 Python：

```bash
cd ~/gnn-predict
findmnt -T "$PWD/output/resnet50/monitor.csv" -o TARGET,SOURCE,FSTYPE
ls -lh output/resnet50/monitor.csv
find output/resnet50/fx_graphs -maxdepth 1 -type f -name '*.pt2' | wc -l
ls output/resnet50/results/*_results.json
python3 -c 'import torch; print(torch.__version__)'
```

预期最后一行显示 `2.4.1+corex.4.3.0`。如果 `.pt2` 数量为 0，先停止；没有图无法提取 GNN 输入。补充缺失的依赖时，安装到 Ceph Home 下的单独目录，并禁止 pip 自动更换 torch。已安装的包不必重复安装；下例仅适用于前面确认过的 Python 3.10 / x86_64 CoreX 容器：

```bash
deps_dir="$HOME/.local/gnn-predict-extract-deps"
python3 -m pip install --no-deps --target "$deps_dir" \
  'polars==1.39.3' 'polars-runtime-32==1.39.3' 'torch-geometric==2.6.1'
export PYTHONPATH="$PWD/src:$deps_dir:$HOME/.local/gnn-predict-deps${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'import torch, polars, torch_geometric, pydantic, yaml; print("torch:", torch.__version__, "polars:", polars.__version__, "PyG:", torch_geometric.__version__)'
```

这里使用 `--no-deps` 是为了保护 CoreX torch；如果最后一条命令报缺少其他依赖，需按报错逐项补装到同一个目录，不能因此改装普通版 torch。PyG 仅用于表示和保存图数据，无需为本次提取安装它的可选 CUDA 扩展；[PyG 安装说明](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)将这些扩展列为可选组件。Polars 主包和运行时包需要一起安装。若下载或导入失败，先排查该问题，不要直接开始整批提取。

然后先读取一份真实图验证兼容性，不必重新运行 83 个变体：

```bash
python3 - <<'PY'
from pathlib import Path
from gnn_model.data.fx_graph import build_graph_data_from_fx

graph_path = next(Path('output/resnet50/fx_graphs').glob('*no_mutations.pt2'))
graph = build_graph_data_from_fx(
    graph_path, batch_size=16, gpu_name='bi-v150', phase='training'
)
print('graph:', graph_path, 'nodes:', graph.num_nodes)
print('graph capture batch:', graph.graph_capture_batch_size)
PY
```

这里的 `PYTHONPATH` 同时包含项目 `src` 和用户目录中的额外依赖。若真实图读取失败，不要继续批量提取；先检查版本和错误信息。

例如在项目根目录中，保证 `output/resnet50/monitor.csv` 和其对应 `results/`、`fx_graphs/` 都在后执行：

```bash
time python3 -m gnn_model.data.extract \
  --csv_dirs output/resnet50 \
  --output_dir data/corex_bi_v150/raw
```

`--csv_dirs` 指向包含 `monitor.csv` 的目录，`--output_dir` 是**新数据集**的输出目录；不会重新测量模型或重新查询 Prometheus。保持当前终端打开，等待进程退出且退出码为 0 后，再检查产物：

```bash
ls -lh data/corex_bi_v150/raw/
python3 - <<'PY'
import json
from pathlib import Path

path = Path('data/corex_bi_v150/raw/manifest.json')
manifest = json.loads(path.read_text(encoding='utf-8'))
for key in ('gpu_names', 'graph_capture_batch_sizes', 'sample_count', 'split_counts', 'quality_report', 'target_names'):
    print(key, manifest[key])
for name in ('train.pt', 'val.pt', 'test.pt'):
    print(name, (path.parent / name).is_file())
PY
```

预期有 `manifest.json` 和三个 `.pt` 文件，`gpu_names` 为 `['bi-v150']`，三个 `split_counts` 均大于 0，且其和等于 `sample_count`。只有监控 CSV 里的合格记录、对应 `.pt2` 图均存在并成功解析时，才会计入样本。若批量提取中途报错，不能把目录中已有的 `.pt` 当作这次完整成功的数据集；先看错误并核对 `manifest.json` 的生成时间。

提取会逐行读取 `.pt2` 并将生成的图样本保存在内存中，再写出三个划分；128 行监控数据不等于 128 份不同的源图。单核、8 GiB 可先做上述单图试读，但完整提取的耗时和内存是否足够不能事先保证；如果可以重新申请，优先选 4 核以上、16 GiB 以上的容器。完成后检查 `raw/manifest.json` 的 `sample_count`、`quality_report`，不要把监控 CSV 的 128 行直接当作最终可训练样本数。

多个 YAML 的 CSV 可以用逗号分隔目录，例如 `--csv_dirs output/resnet50,output/vgg16`。本命令仅接受同一套七目标 BI-V150 数据；不同型号或 NVIDIA 的 CSV 应单独制作。至少要有三个不同的成功变体，才能生成非空的 train/val/test 划分。

生成的 `data/corex_bi_v150/raw/` 中有 `train.pt`、`val.pt`、`test.pt` 和 `manifest.json`。它们是 CPU 图特征及实测标签，不是原始 `.pt2`，也不是 GNN 权重。首次跨环境转移时，先用目标训练环境读取一个小样本确认 PyTorch/PyG 序列化兼容，再转移整批。只读取自己生成、可信的 `.pt` 文件。

## 3. 在 GNN 训练机器归一化并训练

训练机器使用本项目 Python 3.12、PyTorch 2.9 和 PyG 环境。将 `raw/` 文件复制到同名目录后，从项目根目录执行：

```bash
PYTHONPATH=src uv run python -m gnn_model.data.scaler \
  --data_dir data/corex_bi_v150/raw \
  --scaler_output_path data/corex_bi_v150/scalers \
  --scaled_data_output_path data/corex_bi_v150/scaled

PYTHONPATH=src uv run python -m gnn_model \
  --config config/gnn_model/corex_bi_v150.yaml \
  --output_dir output \
  --device cuda:0
```

没有 NVIDIA GPU 的训练机器可改为 `--device cpu`。scaler 从 `raw/manifest.json` 读取七个目标名称，只用 `train.pt` 拟合，再变换三个划分，避免测试集信息泄漏；归一化后的 `scaled/manifest.json` 会记录目标顺序，训练时自动核对配置。GNN 的最佳权重在 `output/gnn_model_corex_bi_v150/checkpoints/best_model.pt`，结果 JSON 在同目录的 `results/`，scaler 单独保存在 `data/corex_bi_v150/scalers/`。三者和配置需要一起保留才能复现预测。训练配置默认 100 轮，最终使用验证误差最小的权重在测试集上评估。

## 重要边界

- 此版本是**BI-V150 单卡档案**：旧 NVIDIA 图特征维度保留，但 NVIDIA 专属硬件参数槽在 BI-V150 数据中置为中性常量 0，不代表实际硬件值。BI-V150 与 V100/A100 数据不得合训或共用权重；将来做跨卡共享 GNN 时必须设计并验证跨厂商硬件特征。
- BI-V150 的监控数据没有 NVIDIA `SM active` 和 `SM occupancy`；七目标配置不预测这两项，也不会将缺失值填为 0。
- 采集容器的 CPU、内存配额会影响 CPU 与部分耗时标签。多次采样应固定资源条件，并在实验记录中保存配额。
- 一份 ResNet50 YAML 可以验证“未见过的 ResNet50 变体”，不能证明对其他模型架构泛化。应继续采集其他模型，并做按模型族留出的外部测试。
