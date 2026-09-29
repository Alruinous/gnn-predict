# BI-V150 数据采集（CoreX 4.3）

本文介绍如何在 BI-V150 上运行模型变体，并采集运行耗时和 Prometheus 监控数据。这里不训练 GNN，因此卡上不需要安装 `torch-geometric`。

当前实测进度：三个 ResNet50 smoke 变体及其监控采集已完成；用户报告完整的 83 变体 ResNet50 运行也已结束，接下来需核对这次的结果 JSON、计算图并采集正式监控 CSV。跨 PyTorch 版本读图、正式数据提取及 GNN 训练仍未验证。

## 保留容器中的 CoreX 环境

已检查的容器使用 Python 3.10.18、天数版 PyTorch `2.4.1+corex.4.3.0`，以及天数版 torchvision `0.19.1a0+corex.4.3.0`。请保留这些厂商构建的包。

**不要在此容器中执行 `uv sync`，也不要直接安装项目 `pyproject.toml` 中的全部依赖。**整个项目要求 Python 3.12 和标准 PyTorch 2.9，与当前容器的 CoreX 环境不同。容器中的 `python3` 应始终指向能够导入天数版 PyTorch 的解释器。

进入容器中的项目根目录后，将图像模型采集入口缺少的两个包安装到当前用户可写的独立目录。使用 `--no-deps`，避免 pip 因依赖解析而替换厂商版 PyTorch 或 torchvision；使用 `--target`，避免容器将用户安装错误地写到无权限的 `/usr/local`：

```bash
deps_dir="$HOME/.local/gnn-predict-deps"
python3 -m pip install --no-deps --target "$deps_dir" 'timm==1.0.20' 'torch-pruning==1.6.1'
python3 -m pip install --target "$deps_dir" 'prometheus-api-client==0.7.2'
export PYTHONPATH="$deps_dir${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'import torch, torchvision, timm, torch_pruning, prometheus_api_client; print(torch.__version__, torch.cuda.get_device_name(0))'
```

`prometheus-api-client` 是监控脚本使用的 Prometheus Python 客户端；以上安装命令不需要安装普通版 PyTorch。上述包只需安装一次，但每次进入新的终端会话都需要重新设置 `PYTHONPATH`。当前容器有 `python3`，没有 `python` 命令。

如果希望新终端自动提供专用启动命令，可在用户的 `~/.bashrc` 中加入以下函数；这样依赖路径只对通过 `gnnpython` 启动的进程生效，不会改变其他 Python 程序的环境：

```bash
gnnpython() {
    PYTHONPATH="$HOME/.local/gnn-predict-deps${PYTHONPATH:+:$PYTHONPATH}" python3 "$@"
}
```

保存后执行一次 `source ~/.bashrc`，以后将下文命令中的 `python3` 换成 `gnnpython` 即可。如果容器重建且用户主目录未保留，安装的包和 `~/.bashrc` 修改也可能需要重做。若导入失败，请停止并记录完整报错；不要通过安装普通版本的 `torch` 或 `torchvision` 来尝试修复。YOLO 等其他模型类型可能需要额外依赖及单独验证。

## 先运行三个 ResNet50 测试变体

```bash
gnnpython main.py --config config/arch/corex_resnet50_smoke.yaml --output_dir output --gpu_node bi-v150 --device_backend corex
```

程序会在日志中记录识别到的设备名称和后端，并在 `output/corex_resnet50_smoke/results/` 下写入结果 JSON。早期单变体实验生成过 `corex_resnet50_smoke_bi-v150_1790615496_results.json`；再次运行时必须使用本次新生成的 JSON。JSON 中的训练、推理时间区间供后续监控采集使用。

运行较大的变体 YAML 时，可在 `main.py` 命令末尾加 `--continue_on_variant_error`。单个变体失败会写入结果 JSON 的 `failures`（变体名、失败阶段、异常类型与原因），记录完整错误日志，并继续后续变体；成功的变体仍保存在 `variants` 中。结果 JSON 在每个成功或失败的变体后原子更新，因此中途停止时已经完成的记录不会因尚未运行到最后一步而全部丢失。若有失败，命令最终返回非零状态，不能将整批视为全部成功。默认不加此参数时仍在首个变体异常后停止，但已成功变体也会逐个保存。设备清理失败、存储权限/空间异常或用户中断不会被当成普通变体失败继续运行。监控采集应在模型运行结束后，针对这次生成的 JSON 执行；失败变体没有性能阶段可供采集。

当前只给第一个测试变体开启了计算图导出，其余两个仍关闭。CoreX PyTorch 2.4 生成的图文件，尚未验证能否由另一台使用 PyTorch 2.9 的机器读取和处理。若要将三个变体全部用于后续 GNN 数据制作，每个变体都需要对应的 `.pt2` 图。

## 配置并采集监控数据

`config/monitor/monitor_corex_example.yaml` 目前填入了这次实测的结果 JSON、Pod、节点、物理 GPU 编号和 UUID。以后重新申请容器或重跑实验时，必须将这些值换成当次的实际值。可以在有 Kubernetes 权限的终端查询 Pod 所在节点：

```bash
kubectl get pod -n crater-workspace POD_NAME -o jsonpath='{.spec.nodeName}'
```

在容器中运行 `/usr/local/corex/bin/ixsmi -q`，读取分配给容器的 GPU UUID。然后查询 Prometheus 的 `ix_gpu_utilization{node_name="NODE_NAME"}`，找到 **UUID 相同** 的时间序列，将其 `gpu` 标签填入 `gpu_id`。不要将容器内显示的 `GPU 0` 直接当成节点上的 `gpu="0"`。

此次容器的 UUID 是 `GPU-8241332d-37cb-5585-8482-a32f429b4bdc`，与 `inspur-01` 节点上的 `gpu="7"` 匹配，因此示例配置使用 `gpu_id: "7"`。当前查询到的 IX 指标可用 `node_name`、`gpu`、`uuid` 标签定位，但不能仅凭容器内 GPU 编号推断物理卡。新 Pod 必须重新匹配。

在能够访问 Prometheus、也能读取结果 JSON 的机器上执行：

```bash
gnnpython monitor.py --config config/monitor/monitor_corex_example.yaml
```

若使用前述 `~/.bashrc` 函数，可以把命令中的 `python3` 换成 `gnnpython`。三个 smoke 变体已取得监控 CSV；正式 ResNet50 必须另用下面的配置和本次运行的结果 JSON，不要复用 smoke 的旧 JSON。CSV 包含 GPU 利用率、显存占用等通用字段；如果 IX exporter 提供相应指标，还会包含功率、温度及 `gpu_sm_util_percent_*`。BI-V150 的 NVIDIA 专属 SM active 和 occupancy 字段保持为空，不会伪造为 0，也不会将 IX 的 SM 利用率视为与它们等价。V100/A100 的监控配置仍默认使用 DCGM。

## 完整 ResNet50 运行结束后采集正式监控

在**运行模型的容器**中进入项目根目录，列出本次运行生成的结果 JSON。不要直接沿用 smoke 测试文件或较早的 ResNet50 文件：

```bash
cd ~/gnn-predict
ls -lt output/resnet50/results/*_bi-v150_*_results.json
```

确认最新文件确实属于这次完整运行，然后设置路径并核对成功/失败变体数。若有多个文件，必须人工确认，不要仅凭时间戳猜测：

```bash
result_json='output/resnet50/results/将这里替换为本次的_results.json'
gnnpython -c 'import json,sys; d=json.load(open(sys.argv[1])); s=d.get("summary",{}); print("planned:",s.get("planned_variant_count"),"success:",len(d.get("variants",[])),"failures:",len(d.get("failures",[])))' "$result_json"
```

将 `config/monitor/monitor_corex_resnet50.yaml` 的 `result_json` 改成上述**真实路径**。该配置暂填先前验证过的 Pod `jpt-luoruian26-260928-01fe9-default0-0`、节点 `inspur-01`、物理 `gpu_id: "7"` 和 UUID。请确认本次 83 变体就是在**这个 Pod 和这张卡**上运行；如果期间换过 Pod，应填运行当时的 Pod、节点和 GPU 标识，而非盲目填写当前新 Pod。确认配置后执行：

```bash
gnnpython monitor.py --config config/monitor/monitor_corex_resnet50.yaml
```

成功时生成 `output/resnet50/monitor.csv`。检查行数和阶段分布：

```bash
gnnpython -c 'import csv,collections; r=list(csv.DictReader(open("output/resnet50/monitor.csv",newline=""))); print("rows:",len(r),"phases:",dict(collections.Counter(x["phase"] for x in r)))'
find output/resnet50/fx_graphs -maxdepth 1 -type f -name '*.pt2' | wc -l
```

理论上 83 个变体各有训练、推理阶段，最多约 166 行；失败变体或监控样本不足的阶段会减少行数。`monitor.py` 会对缺失阶段打印警告，不会为其编造数据。还应确认每个**准备纳入训练**的变体有对应 `.pt2` 图；仅有 CSV 的变体在后续提取时不能成为图样本。监控依赖 Prometheus 保存的历史时间序列，建议运行结束后尽快采集；若历史数据已过保留期，重新执行监控命令也无法恢复原始样本。

## 正式采集时的监控完整性

早期快速连通性测试只运行约 12 秒，CPU `rate` 窗口仅 3 秒；这种配置**不能直接用来生成 GNN 训练数据**。在本次集群实测中，Pod CPU 原始指标约每 10–18 秒出现一个新样本；3 秒窗口无法计算 CPU 使用率，22 秒窗口也曾间断。当前三个测试变体已改为每个训练、推理阶段至少 60 秒，对应监控 YAML 也已设置：

```yaml
cpu_rate_window: "30s"
min_phase_coverage_ratio: 0.8
```

`min_phase_coverage_ratio` 是每个阶段的 CPU、内存及必需 GPU 指标在查询时间点上的最低覆盖率。它不代表独立硬件采样的比例，因为 `query_step_seconds: 1` 可能重复使用同一次 Prometheus 抓取。默认值为 `0.0`，以保持旧的 V100/A100 配置行为；正式采集应显式设定阈值，并检查 CSV 中的 `cpu_sample_coverage`、`memory_sample_coverage`、`gpu_min_sample_coverage`。

监控程序现在允许节点容量指标在阶段边界短暂缺失：它会在各阶段边界寻找 Pod 身份和节点总核数、总内存，并对容量指标回看最多 30 秒。某个阶段缺少必需数据或覆盖率低于阈值时，程序会警告并跳过该**阶段**，继续处理其他阶段和变体；若一个完整阶段都没有，则报错，不再写出只有表头的 CSV。指标标签不匹配或返回多条含糊的 GPU 序列仍会报错，避免把其他卡的数据误认成本卡数据。数据缺失不会被填成 0。正式使用前仍需核对 GPU UUID、结果 JSON 路径以及实际导图能力；完整 ResNet50 虽已报告运行结束，其监控质量与图覆盖情况仍需以上述检查结果为准。
