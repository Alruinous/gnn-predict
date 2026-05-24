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
        op_type_count: int,
        op_type_embedding_dim: int,
        hidden_dim: int,
        targets: list[str],
        num_heads: int,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.targets = targets
        if op_type_count <= 0:
            raise ValueError("op_type_count must be positive")
        if op_type_embedding_dim <= 0:
            raise ValueError("op_type_embedding_dim must be positive")
        self.op_type_embedding = nn.Embedding(op_type_count, op_type_embedding_dim)
        self.node_encoder = nn.Linear(node_dim + op_type_embedding_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        self.graph_encoder = nn.Linear(graph_dim, hidden_dim)
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
        for layer in self.layers:
            x, edge_attr, graph_state = layer(
                x,
                edge_index,
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
