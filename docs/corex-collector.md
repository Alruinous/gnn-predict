# BI-V150 数据采集（CoreX 4.3）

本文介绍如何在 BI-V150 上运行模型变体，并采集运行耗时和 Prometheus 监控数据。这里不训练 GNN，因此卡上不需要安装 `torch-geometric`。以下步骤是待执行的验证流程，**不代表项目已在 BI-V150 上完成端到端实测**。

## 保留容器中的 CoreX 环境

已检查的容器使用 Python 3.10.18、天数版 PyTorch `2.4.1+corex.4.3.0`，以及天数版 torchvision `0.19.1a0+corex.4.3.0`。请保留这些厂商构建的包。

**不要在此容器中执行 `uv sync`，也不要直接安装项目 `pyproject.toml` 中的全部依赖。**整个项目要求 Python 3.12 和标准 PyTorch 2.9，与当前容器的 CoreX 环境不同。容器中的 `python3` 应始终指向能够导入天数版 PyTorch 的解释器。

进入容器中的项目根目录后，将图像模型采集入口缺少的两个包安装到当前用户可写的独立目录。使用 `--no-deps`，避免 pip 因依赖解析而替换厂商版 PyTorch 或 torchvision；使用 `--target`，避免容器将用户安装错误地写到无权限的 `/usr/local`：

```bash
deps_dir="$HOME/.local/gnn-predict-deps"
python3 -m pip install --no-deps --target "$deps_dir" 'timm==1.0.20' 'torch-pruning==1.6.1'
export PYTHONPATH="$deps_dir${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'import torch, torchvision, timm, torch_pruning; print(torch.__version__, torch.cuda.get_device_name(0))'
```

每次进入新的终端会话，都需要重新设置上述 `PYTHONPATH`，或者在运行命令前临时设置它。不要把该路径写入全局 shell 配置，以免影响其他 Python 环境。如果导入失败，请停止并记录完整报错；不要通过安装普通版本的 `torch` 或 `torchvision` 来尝试修复。YOLO 等其他模型类型可能需要额外依赖及单独验证。上述操作只是从 ResNet 开始验证的最小准备，不表示所有变体都已得到支持。

## 先运行一个 ResNet50 变体

```bash
python3 main.py --config config/arch/corex_resnet50_smoke.yaml --output_dir output --gpu_node bi-v150 --device_backend corex
```

程序会在日志中记录识别到的设备名称和后端，并在 `output/corex_resnet50_smoke/results/` 下写入结果 JSON。JSON 中的训练、推理时间区间供后续监控采集使用。

此单变体配置关闭了计算图导出：CoreX PyTorch 2.4 生成的图文件，尚未验证能否由另一台使用 PyTorch 2.9 的机器读取和处理。**请先确认模型构建、运行和计时成功，再单独验证导图。**

## 配置并采集监控数据

复制 `config/monitor/monitor_corex_example.yaml`，将其中的结果 JSON 路径、Pod 名称、节点名称和 GPU UUID 替换成实际值。可以通过以下命令查询 Pod 所在节点：

```bash
kubectl get pod -n crater-workspace POD_NAME -o jsonpath='{.spec.nodeName}'
```

在 Prometheus 中查询 `ix_gpu_utilization{node_name="NODE_NAME",gpu="0"}`，查看 IX exporter 上报的 GPU UUID。当前集群查到的 IX 指标包含 `node_name`、`gpu`、`name`、`uuid` 标签，但没有 `pod`、`namespace` 标签。尤其在多卡节点上，必须核实查到的 UUID 确实属于当前 Pod 分配到的 GPU；如果节点上有多个作业，不能只凭 GPU 编号推断对应关系。

在能够访问 Prometheus、也能读取结果 JSON 的机器上执行：

```bash
python monitor.py --config config/monitor/monitor_corex_example.yaml
```

生成的 CSV 包含 GPU 利用率、显存占用等通用字段；如果 IX exporter 提供相应指标，还会包含功率、温度及 `gpu_sm_util_percent_*`。BI-V150 的 NVIDIA 专属 SM active 和 occupancy 字段保持为空，不会伪造为 0，也不会将 IX 的 SM 利用率视为与它们等价。V100/A100 的监控配置仍默认使用 DCGM。
