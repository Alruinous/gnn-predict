from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from .fusion import GraphFusionLayer, RegressionHead

NODE_ONNX_TOOL_METRIC_COUNT = 3
STRUCTURAL_TOPOLOGY_FEATURE_COUNT = 8
ReadoutMode = Literal["mean", "mean_sum_max"]
StructuralContextMode = Literal["none", "basic"]


class IntelliGraphLargeModelPredictor(nn.Module):
    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        graph_dim: int,
        op_type_count: int,
        op_type_embedding_dim: int,
        hidden_dim: int,
        targets: list[str],
        num_heads: int,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        readout_mode: ReadoutMode = "mean",
        target_readout_modes: Mapping[str, ReadoutMode] | None = None,
        structural_context_mode: StructuralContextMode = "none",
    ) -> None:
        super().__init__()
        self.targets = targets
        if readout_mode not in {"mean", "mean_sum_max"}:
            raise ValueError(f"unsupported readout_mode: {readout_mode}")
        self.readout_mode = readout_mode
        if structural_context_mode not in {"none", "basic"}:
            raise ValueError(
                f"unsupported structural_context_mode: {structural_context_mode}"
            )
        self.structural_context_mode = structural_context_mode
        target_readout_modes = target_readout_modes or {}
        unknown_targets = set(target_readout_modes) - set(targets)
        if unknown_targets:
            raise ValueError(f"unknown target readout modes: {sorted(unknown_targets)}")
        if any(
            mode not in {"mean", "mean_sum_max"}
            for mode in target_readout_modes.values()
        ):
            raise ValueError("target readout modes must be mean or mean_sum_max")
        self.target_readout_modes = {
            target_name: target_readout_modes.get(target_name, readout_mode)
            for target_name in targets
        }
        if op_type_count <= 0:
            raise ValueError("op_type_count must be positive")
        if op_type_embedding_dim <= 0:
            raise ValueError("op_type_embedding_dim must be positive")
        self.op_type_embedding = nn.Embedding(op_type_count, op_type_embedding_dim)
        self.node_encoder = nn.Linear(node_dim + op_type_embedding_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.graph_encoder = nn.Linear(graph_dim, hidden_dim)
        structural_dim = structural_context_dim(op_type_count)
        self.structural_encoder = (
            nn.Sequential(
                nn.Linear(structural_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if structural_context_mode == "basic"
            else nn.Identity()
        )
        self.structural_scale = (
            nn.Parameter(torch.tensor(0.05))
            if structural_context_mode == "basic"
            else None
        )
        self.structural_norm = (
            nn.LayerNorm(hidden_dim)
            if structural_context_mode == "basic"
            else nn.Identity()
        )
        self.layers = nn.ModuleList(
            [
                GraphFusionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )
        self.needs_multi_pool = any(
            mode == "mean_sum_max" for mode in self.target_readout_modes.values()
        )
        self.node_readout = (
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.LayerNorm(hidden_dim),
            )
            if self.needs_multi_pool
            else nn.Identity()
        )
        self.heads = nn.ModuleDict(
            {
                target_name: RegressionHead(hidden_dim=hidden_dim, dropout=dropout_rate)
                for target_name in targets
            }
        )

    def forward(self, data: Data) -> torch.Tensor:
        x_input = data.x
        assert isinstance(x_input, torch.Tensor)
        edge_index = data.edge_index
        assert isinstance(edge_index, torch.Tensor)
        edge_attr_input = data.edge_attr
        assert isinstance(edge_attr_input, torch.Tensor)
        graph_features = getattr(data, "graph_features", None)
        assert isinstance(graph_features, torch.Tensor)
        op_type_ids = getattr(data, "op_type_ids", None)
        assert isinstance(op_type_ids, torch.Tensor)
        op_type_embeddings = self.op_type_embedding(op_type_ids.long())
        x_input = torch.cat([x_input.float(), op_type_embeddings], dim=-1)

        batch_value = getattr(data, "batch", None)
        batch = (
            batch_value
            if isinstance(batch_value, torch.Tensor)
            else torch.zeros(x_input.size(0), dtype=torch.long, device=x_input.device)
        )
        if graph_features.dim() == 1:
            graph_features = graph_features.unsqueeze(0)

        x = self.node_encoder(x_input)
        edge_attr = self.edge_encoder(edge_attr_input.float())
        graph_state = self.graph_encoder(graph_features.float())
        if self.structural_context_mode == "basic":
            assert self.structural_scale is not None
            graph_state = self.structural_norm(
                graph_state
                + self.structural_scale
                * self.structural_encoder(
                    build_structural_context(
                        data_x=x_input,
                        edge_index=edge_index,
                        edge_attr=edge_attr_input.float(),
                        op_type_ids=op_type_ids,
                        batch=batch,
                        graph_count=graph_features.size(0),
                        op_type_count=self.op_type_embedding.num_embeddings,
                    )
                )
            )
        for layer in self.layers:
            x, edge_attr, graph_state = layer(
                x,
                edge_index,
                edge_attr,
                graph_state,
                batch,
            )

        pooled_nodes = global_mean_pool(x, batch)
        multi_pooled_nodes = None
        if self.needs_multi_pool:
            multi_pooled_nodes = self.node_readout(
                torch.cat(
                    [
                        pooled_nodes,
                        global_add_pool(x, batch),
                        global_max_pool(x, batch),
                    ],
                    dim=-1,
                )
            )
        outputs = []
        for target_name, head in self.heads.items():
            target_nodes = (
                multi_pooled_nodes
                if self.target_readout_modes[target_name] == "mean_sum_max"
                else pooled_nodes
            )
            assert isinstance(target_nodes, torch.Tensor)
            outputs.append(head(target_nodes, graph_state))
        predictions = torch.cat(outputs, dim=-1)
        if not torch.isfinite(predictions).all():
            raise ValueError("predictor produced non-finite outputs")
        return predictions


def structural_context_dim(op_type_count: int) -> int:
    return (
        op_type_count * 2
        + op_type_count * NODE_ONNX_TOOL_METRIC_COUNT
        + op_type_count * op_type_count
        + STRUCTURAL_TOPOLOGY_FEATURE_COUNT
    )


def build_structural_context(
    *,
    data_x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    op_type_ids: torch.Tensor,
    batch: torch.Tensor,
    graph_count: int,
    op_type_count: int,
) -> torch.Tensor:
    graph_count_int = int(graph_count)
    device = data_x.device
    dtype = data_x.dtype
    op_ids = op_type_ids.long().clamp(min=0, max=op_type_count - 1)
    node_count = torch.bincount(batch, minlength=graph_count_int).to(dtype).unsqueeze(1)
    op_one_hot = F.one_hot(op_ids, num_classes=op_type_count).to(dtype)
    op_counts = torch.zeros(
        graph_count_int,
        op_type_count,
        dtype=dtype,
        device=device,
    )
    op_counts.index_add_(0, batch, op_one_hot)
    op_count_features = op_counts.log1p()
    op_ratios = op_counts / node_count.clamp_min(1.0)
    metrics = data_x[:, :NODE_ONNX_TOOL_METRIC_COUNT]
    op_metrics = torch.zeros(
        graph_count_int * op_type_count,
        NODE_ONNX_TOOL_METRIC_COUNT,
        dtype=dtype,
        device=device,
    )
    op_metrics.index_add_(0, batch * op_type_count + op_ids, metrics)
    op_metrics = signed_log1p(op_metrics.reshape(graph_count_int, -1))
    transitions, edge_count, edge_batch = build_transition_context(
        edge_index=edge_index,
        op_ids=op_ids,
        batch=batch,
        graph_count=graph_count_int,
        op_type_count=op_type_count,
        dtype=dtype,
    )
    topology = build_topology_context(
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_batch=edge_batch,
        batch=batch,
        graph_count=graph_count_int,
        edge_count=edge_count,
        dtype=dtype,
    )
    return torch.cat(
        [op_count_features, op_ratios, op_metrics, transitions, topology],
        dim=1,
    )


def build_transition_context(
    *,
    edge_index: torch.Tensor,
    op_ids: torch.Tensor,
    batch: torch.Tensor,
    graph_count: int,
    op_type_count: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = batch.device
    if edge_index.numel() == 0:
        edge_count = torch.zeros(graph_count, dtype=dtype, device=device)
        edge_batch = torch.zeros(0, dtype=torch.long, device=device)
        return (
            torch.zeros(
                graph_count,
                op_type_count * op_type_count,
                dtype=dtype,
                device=device,
            ),
            edge_count,
            edge_batch,
        )
    source, target = edge_index
    edge_batch = batch[source]
    transition_index = (
        edge_batch * op_type_count * op_type_count
        + op_ids[source] * op_type_count
        + op_ids[target]
    )
    flat = torch.zeros(
        graph_count * op_type_count * op_type_count,
        dtype=dtype,
        device=device,
    )
    flat.index_add_(0, transition_index, torch.ones_like(transition_index, dtype=dtype))
    transitions = flat.reshape(graph_count, op_type_count * op_type_count)
    edge_count = torch.bincount(edge_batch, minlength=graph_count).to(dtype)
    return transitions / edge_count.unsqueeze(1).clamp_min(1.0), edge_count, edge_batch


def build_topology_context(
    *,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_batch: torch.Tensor,
    batch: torch.Tensor,
    graph_count: int,
    edge_count: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    device = batch.device
    node_count = torch.bincount(batch, minlength=graph_count).to(dtype)
    outdegree = torch.zeros(batch.size(0), dtype=dtype, device=device)
    indegree = torch.zeros(batch.size(0), dtype=dtype, device=device)
    if edge_index.numel() > 0:
        source, target = edge_index
        outdegree.index_add_(0, source, torch.ones_like(source, dtype=dtype))
        indegree.index_add_(0, target, torch.ones_like(target, dtype=dtype))
    source_nodes = scatter_graph_count((indegree == 0).to(dtype), batch, graph_count)
    sink_nodes = scatter_graph_count((outdegree == 0).to(dtype), batch, graph_count)
    max_in = scatter_graph_max(indegree, batch, graph_count)
    max_out = scatter_graph_max(outdegree, batch, graph_count)
    edge_density = edge_count / node_count.clamp_min(1.0)
    edge_bytes = edge_attr[:, 0] if edge_attr.numel() else edge_attr.new_zeros(0)
    edge_bytes_sum = scatter_graph_count(edge_bytes, edge_batch, graph_count)
    edge_bytes_max = scatter_graph_max(edge_bytes, edge_batch, graph_count)
    return torch.stack(
        [
            node_count.log1p(),
            edge_count.log1p(),
            edge_density,
            source_nodes / node_count.clamp_min(1.0),
            sink_nodes / node_count.clamp_min(1.0),
            max_in.log1p(),
            max_out.log1p(),
            signed_log1p(edge_bytes_sum + edge_bytes_max),
        ],
        dim=1,
    )


def signed_log1p(values: torch.Tensor) -> torch.Tensor:
    return values.sign() * values.abs().log1p()


def scatter_graph_count(
    values: torch.Tensor,
    index: torch.Tensor,
    graph_count: int,
) -> torch.Tensor:
    output = values.new_zeros(graph_count)
    if values.numel() > 0:
        output.index_add_(0, index, values)
    return output


def scatter_graph_max(
    values: torch.Tensor,
    index: torch.Tensor,
    graph_count: int,
) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(graph_count)
    output = values.new_full((graph_count,), 0.0)
    output.scatter_reduce_(0, index, values, reduce="amax", include_self=True)
    return output
