from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
from torch_rechub.basic.features import SparseFeature
from torch_rechub.models.ranking import DeepFM

from gnn_archs.recommender.common import (
    build_dense_features,
    build_mlp_params,
    build_sparse_features,
    get_deepfm_config,
    validate_recommender_spec,
)

if TYPE_CHECKING:
    from gnn_archs.config import ResolvedVariantSpec


def build_deepfm_model(spec: ResolvedVariantSpec) -> nn.Module:
    validate_recommender_spec(spec)
    config = get_deepfm_config(spec.variant_config)
    sparse_features = build_sparse_features(config)
    sparse_features_by_name = {feature.name: feature for feature in sparse_features}
    unknown_names = sorted(set(config.fm_feature_names) - set(sparse_features_by_name))
    if unknown_names:
        raise ValueError(
            "deepfm_config.fm_feature_names must reference sparse "
            f"features: {unknown_names}"
        )

    fm_features = [sparse_features_by_name[name] for name in config.fm_feature_names]
    validate_matching_sparse_embed_dims(fm_features)
    return DeepFM(
        deep_features=sparse_features + build_dense_features(config),
        fm_features=fm_features,
        mlp_params=build_mlp_params(config),
    )


def validate_matching_sparse_embed_dims(features: list[SparseFeature]) -> None:
    embed_dims = {feature.embed_dim for feature in features}
    if len(embed_dims) != 1:
        raise ValueError("DeepFM fm_features must use the same embed_dim")
