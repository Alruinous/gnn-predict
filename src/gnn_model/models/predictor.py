from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from .fusion import GraphFusionLayer, RegressionHead


class IntelliGraphLargeModelPredictor(nn.Module):
    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        graph_dim: int,
        graph_metric_dim: int,
        hidden_dim: int,
        targets: list[str],
        num_heads: int,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.targets = targets
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.graph_encoder = nn.Linear(graph_dim + graph_metric_dim, hidden_dim)
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
        self.heads = nn.ModuleDict(
            {
                target_name: RegressionHead(hidden_dim=hidden_dim, dropout=dropout_rate)
                for target_name in targets
            }
        )

    def forward(self, data: Data) -> torch.Tensor:
        if not hasattr(data, "graph_features") or not hasattr(data, "graph_metrics"):
            raise ValueError(
                "graph batch must contain graph_features and graph_metrics"
            )
        if not hasattr(data, "edge_attr"):
            raise ValueError("graph batch must contain edge_attr")

        batch = getattr(
            data,
            "batch",
            torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device),
        )
        graph_features = data.graph_features
        if graph_features.dim() == 1:
            graph_features = graph_features.unsqueeze(0)
        graph_metrics = data.graph_metrics
        if graph_metrics.dim() == 1:
            graph_metrics = graph_metrics.unsqueeze(0)

        x = self.node_encoder(data.x.float())
        edge_attr = self.edge_encoder(data.edge_attr.float())
        graph_state = self.graph_encoder(
            torch.cat([graph_features.float(), graph_metrics.float()], dim=-1)
        )
        for layer in self.layers:
            x, edge_attr, graph_state = layer(
                x,
                data.edge_index,
                edge_attr,
                graph_state,
                batch,
            )

        pooled_nodes = global_mean_pool(x, batch)
        outputs = [head(pooled_nodes, graph_state) for head in self.heads.values()]
        predictions = torch.cat(outputs, dim=-1)
        if not torch.isfinite(predictions).all():
            raise ValueError("predictor produced non-finite outputs")
        return predictions
