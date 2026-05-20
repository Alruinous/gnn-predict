from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
from torch_rechub.models.ranking import DCNv2

from gnn_archs.recommender.common import (
    build_dense_features,
    build_mlp_params,
    build_sparse_features,
    get_dcnv2_config,
    validate_recommender_spec,
)

if TYPE_CHECKING:
    from gnn_archs.config import ResolvedVariantSpec


def build_dcnv2_model(spec: ResolvedVariantSpec) -> nn.Module:
    validate_recommender_spec(spec)
    config = get_dcnv2_config(spec.variant_config)
    if config.n_cross_layers is None or config.n_cross_layers <= 0:
        raise ValueError("dcnv2_config.n_cross_layers must be positive")
    if config.low_rank is None or config.low_rank <= 0:
        raise ValueError("dcnv2_config.low_rank must be positive")
    if config.num_experts is None or config.num_experts <= 0:
        raise ValueError("dcnv2_config.num_experts must be positive")

    return DCNv2(
        features=build_sparse_features(config) + build_dense_features(config),
        n_cross_layers=config.n_cross_layers,
        mlp_params=build_mlp_params(config),
        model_structure=config.model_structure,
        use_low_rank_mixture=config.use_low_rank_mixture,
        low_rank=config.low_rank,
        num_experts=config.num_experts,
    )
