from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
from torch_rechub.models.ranking import DCN

from gnn_archs.recommender.common import (
    build_dense_features,
    build_mlp_params,
    build_sparse_features,
    get_dcn_config,
    validate_recommender_spec,
)

if TYPE_CHECKING:
    from gnn_archs.config import ResolvedVariantSpec


def build_dcn_model(spec: ResolvedVariantSpec) -> nn.Module:
    validate_recommender_spec(spec)
    config = get_dcn_config(spec.variant_config)
    if config.n_cross_layers is None or config.n_cross_layers <= 0:
        raise ValueError("dcn_config.n_cross_layers must be positive")

    return DCN(
        features=build_sparse_features(config) + build_dense_features(config),
        n_cross_layers=config.n_cross_layers,
        mlp_params=build_mlp_params(config),
    )
