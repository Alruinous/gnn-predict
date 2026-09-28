# BI-V150 collection (CoreX 4.3)

This path only runs model variants and collects timing/Prometheus data. It does
not train the GNN, and it does not require `torch-geometric` on the card.

The verified container uses Python 3.10.18, CoreX PyTorch
`2.4.1+corex.4.3.0`, and CoreX torchvision `0.19.1a0+corex.4.3.0`. Keep those
vendor packages. **Do not run `uv sync` or install the repository's complete
`pyproject.toml` in this container:** that environment requires Python 3.12 and
standard PyTorch 2.9. The container's `python3` must still point to the
interpreter that imports CoreX PyTorch.

From the repository root inside the BI-V150 container, install only the two
missing imports needed for image-model collection, without permitting pip to
replace the vendor PyTorch or torchvision:

```bash
python3 -m pip install --no-deps 'timm==1.0.20' 'torch-pruning==1.6.1'
python3 -c 'import torch, torchvision, timm, torch_pruning; print(torch.__version__, torch.cuda.get_device_name(0))'
```

If that import fails, stop and record the traceback; do not fix it by
installing generic `torch` or `torchvision`. Other model families (for example,
YOLO) may need their own dependencies and validation. The above is the
minimal ResNet starting point, not a claim that every variant is supported.

Run one unmutated variant first:

```bash
python3 main.py --config config/arch/corex_resnet50_smoke.yaml --output_dir output --gpu_node bi-v150 --device_backend corex
```

The command logs the detected device name and backend and writes a result JSON
under `output/corex_resnet50_smoke/results/`. Its training and inference timing
windows are used by the monitoring step. Graph export is disabled in this
smoke configuration because the CoreX 2.4 graph artifact has not yet been
validated for the separate PyTorch 2.9 graph-processing environment.

For monitoring, copy `config/monitor/monitor_corex_example.yaml` and replace
the result JSON path, Pod name, node name, and GPU UUID. The Pod's node can be
checked with:

```bash
kubectl get pod -n crater-workspace POD_NAME -o jsonpath='{.spec.nodeName}'
```

Query `ix_gpu_utilization{node_name="NODE_NAME",gpu="0"}` in Prometheus to
find the IX-exporter UUID. The current cluster's IX series contain `node_name`,
`gpu`, `name`, and `uuid`, but not `pod` or `namespace`; verify that this UUID
really belongs to the Pod's allocated GPU, especially on a multi-GPU node.
Do not infer that mapping from GPU index alone if multiple jobs share the node.

Run monitoring on a machine with access to Prometheus and the result JSON:

```bash
python monitor.py --config config/monitor/monitor_corex_example.yaml
```

The CSV contains shared columns (utilization, memory, optional power and
temperature) and `gpu_sm_util_percent_*` when the IX exporter provides it.
NVIDIA-only SM-active and occupancy columns remain empty for BI-V150; they
are not silently set to zero or treated as equivalent to IX SM utilization.
The V100/A100 configurations continue to use DCGM by default.
