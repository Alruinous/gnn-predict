from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import TransformerConv, global_mean_pool


class AdaptiveFeatureFusion(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        emb_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_proj = nn.Linear(feat_dim, output_dim)
        self.context_proj = nn.Linear(emb_dim, output_dim)
        self.gate = nn.Sequential(
            nn.Linear(emb_dim, output_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, features: Tensor, context: Tensor) -> Tensor:
        projected_features = self.feature_proj(features)
        projected_context = self.context_proj(context)
        gate = self.gate(context)
        fused = projected_features + gate * projected_context
        return self.norm(self.dropout(fused))


class GraphFusionLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.node_conv = TransformerConv(
            hidden_dim,
            hidden_dim // num_heads,
            heads=num_heads,
            edge_dim=hidden_dim,
            dropout=dropout,
            beta=True,
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.graph_fusion = AdaptiveFeatureFusion(
            hidden_dim,
            hidden_dim,
            hidden_dim,
            dropout,
        )
        self.graph_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        graph_state: Tensor,
        batch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        updated_nodes = self.node_conv(x, edge_index, edge_attr)
        x = self.node_norm(x + updated_nodes)
        src_index, dst_index = edge_index
        edge_inputs = torch.cat([x[src_index], x[dst_index], edge_attr], dim=-1)
        edge_attr = self.edge_norm(edge_attr + self.edge_update(edge_inputs))
        pooled_nodes = global_mean_pool(x, batch)
        graph_state = self.graph_norm(self.graph_fusion(pooled_nodes, graph_state))
        return x, edge_attr, graph_state


class RegressionHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fusion = AdaptiveFeatureFusion(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pooled_nodes: Tensor, graph_state: Tensor) -> Tensor:
        return self.mlp(self.fusion(pooled_nodes, graph_state))
