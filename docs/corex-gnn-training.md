# BI-V150 数据提取与 GNN 训练

本流程只适用于已经取得 BI-V150 变体结果 JSON、`.pt2` 图和监控 CSV 的实验。GNN 在另一台机器训练；BI-V150 只负责模型测量和图导出。V100/A100 的九目标数据集与本流程分开制作，不要混在同一数据集中。

## 1. 检查采集结果

一个完成的变体通常有训练和推理两条 CSV 记录，共用一份 `fx_graphs/<变体名>.pt2`。监控可能跳过缺样本的阶段，所以 83 个变体不保证得到 166 条有效记录。提取器会忽略无图、缺少目标值或 GPU 指标不合理的记录；`manifest.json` 中的 `quality_report` 记录原始、过滤和保留行数。`split_unit: gpu_model_variant` 表示同一变体的所有阶段不会跨训练、验证和测试集。

新生成的 `monitor.csv` 会把 `result_json` 写成**相对 CSV 所在目录**的路径。例如 CSV 位于 `output/resnet50/monitor.csv` 时，值通常是 `results/xxx_results.json`。提取器据此查找同级 `fx_graphs/<变体名>.pt2`。复制到另一台机器时，保持 `monitor.csv`、`results/`、`fx_graphs/` 三者的相对目录结构即可，不需要保留原容器的 `/home/...` 路径。旧 CSV 中的项目根目录相对路径或原容器绝对路径也会尝试映射到 CSV 同级的 `results/`、`fx_graphs/`；如果你把 CSV 单独移到别处，则仍需更正路径或恢复目录结构。不要把多个重复运行的同名变体直接混合进同一数据集，提取器会拒绝重复的“卡／模型／变体／阶段”样本。

## 2. 用兼容的 PyTorch 提取图特征

BI-V150 的 `.pt2` 是由 `torch 2.4.1+corex.4.3.0` 写出的。项目的图加载器要求 PyTorch 主、次版本相同；**不能假定**在默认的 PyTorch 2.9 训练环境里可直接读取。先用一份真实 `.pt2` 在拟用于提取的环境中测试读取和特征构建。如果失败，使用与导图版本匹配的 PyTorch 2.4 环境提取，而不是关闭版本校验或覆盖厂商版 torch。

提取环境还需 `polars`、`torch-geometric`、`pydantic`、`PyYAML`。它可以是 BI-V150 容器，也可以是另一台可读取该图的 CPU 机器；不要求在天数卡上训练 GNN。**不要在 CoreX 容器执行 `uv sync`**，它会按项目锁文件安装标准 PyTorch 2.9。

例如在项目根目录中，保证 `output/resnet50/monitor.csv` 和其对应 `results/`、`fx_graphs/` 都在后执行：

```bash
PYTHONPATH=src python3 -m gnn_model.data.extract \
  --csv_dirs output/resnet50 \
  --output_dir data/corex_bi_v150/raw
```

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
