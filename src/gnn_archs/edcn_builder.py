from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
from torch_rechub.models.ranking import EDCN

from gnn_archs.recommender.common import (
    build_dense_features,
    build_edcn_mlp_params,
    build_sparse_features,
    get_edcn_config,
    validate_recommender_spec,
)

if TYPE_CHECKING:
    from gnn_archs.config import ResolvedVariantSpec


def build_edcn_model(spec: ResolvedVariantSpec) -> nn.Module:
    validate_recommender_spec(spec)
    config = get_edcn_config(spec.variant_config)
    if config.n_cross_layers is None or config.n_cross_layers <= 0:
        raise ValueError("edcn_config.n_cross_layers must be positive")
    if config.temperature is None or config.temperature <= 0:
        raise ValueError("edcn_config.temperature must be positive")

    features = build_sparse_features(config) + build_dense_features(config)
    return EDCN(
        features=features,
        n_cross_layers=config.n_cross_layers,
        mlp_params=build_edcn_mlp_params(config, features),
        bridge_type=config.bridge_type,
        use_regulation_module=config.use_regulation_module,
        temperature=config.temperature,
    )
