from __future__ import annotations

from typing import Any

import torch
from torch_rechub.basic.features import DenseFeature, SparseFeature

from gnn_archs.config import (
    DCNConfigOverride,
    DeepFMConfigOverride,
    ResolvedVariantSpec,
    VariantConfig,
)

type RecommenderModelConfig = DeepFMConfigOverride | DCNConfigOverride


def validate_recommender_spec(spec: ResolvedVariantSpec) -> None:
    if spec.mutations:
        raise ValueError("recommender variants do not support mutations")
    if spec.variant_config.target_output_classes != 1:
        raise ValueError("recommender variants require target_output_classes=1")


def get_deepfm_config(variant_config: VariantConfig) -> DeepFMConfigOverride:
    config = variant_config.deepfm_config
    if config is None:
        raise ValueError("deepfm variants require variant_config.deepfm_config")
    return config


def get_dcn_config(variant_config: VariantConfig) -> DCNConfigOverride:
    config = variant_config.dcn_config
    if config is None:
        raise ValueError("dcn variants require variant_config.dcn_config")
    return config


def get_recommender_config(variant_config: VariantConfig) -> RecommenderModelConfig:
    config = variant_config.deepfm_config or variant_config.dcn_config
    if config is None:
        raise ValueError(
            "recommender variants require variant_config.deepfm_config "
            "or variant_config.dcn_config"
        )
    return config


def build_sparse_features(config: RecommenderModelConfig) -> list[SparseFeature]:
    return [
        SparseFeature(
            name=feature.name,
            vocab_size=feature.vocab_size,
            embed_dim=feature.embed_dim,
        )
        for feature in config.sparse_features
    ]


def build_dense_features(config: RecommenderModelConfig) -> list[DenseFeature]:
    return [
        DenseFeature(name=feature.name, embed_dim=feature.embed_dim)
        for feature in config.dense_features
    ]


def build_mlp_params(config: RecommenderModelConfig) -> dict[str, Any]:
    return {
        "dims": config.mlp_dims,
        "activation": config.activation,
        "dropout": config.dropout,
    }


def get_recommender_feature_names(variant_config: VariantConfig) -> list[str]:
    config = get_recommender_config(variant_config)
    return [feature.name for feature in config.sparse_features] + [
        feature.name for feature in config.dense_features
    ]


def build_recommender_batch(
    variant_config: VariantConfig,
    batch_size: int,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    config = get_recommender_config(variant_config)
    batch: dict[str, torch.Tensor] = {}
    for feature in config.sparse_features:
        batch[feature.name] = torch.randint(
            0,
            feature.vocab_size,
            (batch_size,),
            generator=generator,
            dtype=torch.long,
        )
    for feature in config.dense_features:
        shape = (
            (batch_size,) if feature.embed_dim == 1 else (batch_size, feature.embed_dim)
        )
        batch[feature.name] = torch.randn(shape, generator=generator)
    return batch
