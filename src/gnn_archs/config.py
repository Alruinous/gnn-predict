from __future__ import annotations

from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TEXT_MODEL_PREFIXES = (
    "albert",
    "bert",
    "deberta",
    "distilbert",
    "electra",
    "flan-t5",
    "gpt",
    "mt5",
    "qwen",
    "roberta",
    "t5",
    "xlnet",
    "xlm",
)


def normalize_model_identifier(model_name: str) -> str:
    return model_name.strip().lower().split("/")[-1]


def is_text_model_name(model_name: str) -> bool:
    normalized_name = normalize_model_identifier(model_name)
    return any(normalized_name.startswith(prefix) for prefix in TEXT_MODEL_PREFIXES)


def is_qwen_model_name(model_name: str) -> bool:
    return normalize_model_identifier(model_name).startswith("qwen")


DETECTION_MODEL_PREFIXES = (
    "yolov3",
    "yolov5",
    "yolov6",
    "yolov7",
    "yolov8",
    "yolov9",
    "yolov10",
    "yolo11",
    "yoloe",
)


def is_detection_model_name(model_name: str) -> bool:
    """判断是否为 YOLO 检测模型。"""
    normalized = normalize_model_identifier(model_name)
    return any(normalized.startswith(prefix) for prefix in DETECTION_MODEL_PREFIXES)


RECOMMENDER_MODEL_NAMES = {"deepfm", "dcn", "dcnv2", "edcn"}


def is_recommender_model_name(model_name: str) -> bool:
    normalized_name = normalize_model_identifier(model_name)
    return normalized_name in RECOMMENDER_MODEL_NAMES


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MutationConfig(StrictModel):
    type: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("mutation type must not be empty")
        return normalized_value


class MutationSet(StrictModel):
    name: str
    mutations: list[MutationConfig] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("mutation set name must not be empty")
        return normalized_value


class Gpt2ConfigOverride(StrictModel):
    vocab_size: int
    n_positions: int
    n_embd: int
    n_layer: int
    n_head: int
    n_inner: int | None = None
    activation_function: str = "gelu_new"
    resid_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    scale_attn_by_inverse_layer_idx: bool = False
    reorder_and_upcast_attn: bool = False


class T5ConfigOverride(StrictModel):
    vocab_size: int
    d_model: int
    d_ff: int
    num_layers: int
    num_decoder_layers: int
    num_heads: int
    d_kv: int | None = None
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float = 0.0
    classifier_dropout: float = 0.0
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 1.0
    feed_forward_proj: str = "relu"


class RecommenderSparseFeatureConfig(StrictModel):
    name: str
    vocab_size: int
    embed_dim: int

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("sparse feature name must not be empty")
        return normalized_value

    @model_validator(mode="after")
    def validate_positive_fields(self) -> RecommenderSparseFeatureConfig:
        if self.vocab_size <= 0:
            raise ValueError("sparse feature vocab_size must be positive")
        if self.embed_dim <= 0:
            raise ValueError("sparse feature embed_dim must be positive")
        return self


class RecommenderDenseFeatureConfig(StrictModel):
    name: str
    embed_dim: int = 1

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("dense feature name must not be empty")
        return normalized_value

    @field_validator("embed_dim")
    @classmethod
    def validate_embed_dim(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("dense feature embed_dim must be positive")
        return value


class RecommenderBaseConfigOverride(StrictModel):
    config_field_name: ClassVar[str] = "recommender_model_config"

    sparse_features: list[RecommenderSparseFeatureConfig]
    dense_features: list[RecommenderDenseFeatureConfig] = Field(default_factory=list)
    activation: str = "relu"
    dropout: float = 0.0

    @model_validator(mode="after")
    def validate_common_config(self) -> Self:
        if not self.sparse_features:
            raise ValueError(
                f"{self.config_field_name}.sparse_features must not be empty"
            )
        if not 0 <= self.dropout < 1:
            raise ValueError(f"{self.config_field_name}.dropout must be in [0, 1)")

        sparse_names = [feature.name for feature in self.sparse_features]
        dense_names = [feature.name for feature in self.dense_features]
        feature_names = sparse_names + dense_names
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("recommender feature names must be unique")
        return self


class RecommenderMlpConfigOverride(RecommenderBaseConfigOverride):
    mlp_dims: list[int]

    @model_validator(mode="after")
    def validate_mlp_config(self) -> Self:
        if not self.mlp_dims:
            raise ValueError(f"{self.config_field_name}.mlp_dims must not be empty")
        if any(dim <= 0 for dim in self.mlp_dims):
            raise ValueError(f"{self.config_field_name}.mlp_dims must be positive")
        return self


class DeepFMConfigOverride(RecommenderMlpConfigOverride):
    config_field_name: ClassVar[str] = "deepfm_config"

    fm_feature_names: list[str]

    @model_validator(mode="after")
    def validate_deepfm_config(self) -> DeepFMConfigOverride:
        sparse_names = {feature.name for feature in self.sparse_features}
        if not self.fm_feature_names:
            raise ValueError("deepfm_config.fm_feature_names must not be empty")
        unknown_names = sorted(set(self.fm_feature_names) - sparse_names)
        if unknown_names:
            raise ValueError(
                "deepfm_config.fm_feature_names must reference sparse "
                f"features: {unknown_names}"
            )
        return self


class DCNConfigOverride(RecommenderMlpConfigOverride):
    config_field_name: ClassVar[str] = "dcn_config"

    n_cross_layers: int

    @field_validator("n_cross_layers")
    @classmethod
    def validate_n_cross_layers(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("dcn_config.n_cross_layers must be positive")
        return value


class DCNv2ConfigOverride(RecommenderMlpConfigOverride):
    config_field_name: ClassVar[str] = "dcnv2_config"

    n_cross_layers: int
    low_rank: int
    num_experts: int
    model_structure: Literal["crossnet_only", "stacked", "parallel"] = "parallel"
    use_low_rank_mixture: bool = True

    @model_validator(mode="after")
    def validate_dcnv2_config(self) -> DCNv2ConfigOverride:
        if self.n_cross_layers <= 0:
            raise ValueError("dcnv2_config.n_cross_layers must be positive")
        if self.low_rank <= 0:
            raise ValueError("dcnv2_config.low_rank must be positive")
        if self.num_experts <= 0:
            raise ValueError("dcnv2_config.num_experts must be positive")
        return self


class EDCNConfigOverride(RecommenderBaseConfigOverride):
    config_field_name: ClassVar[str] = "edcn_config"

    n_cross_layers: int
    bridge_type: Literal[
        "hadamard_product",
        "pointwise_addition",
        "concatenation",
        "attention_pooling",
    ] = "hadamard_product"
    use_regulation_module: bool = True
    temperature: float = 1.0

    @model_validator(mode="after")
    def validate_edcn_config(self) -> EDCNConfigOverride:
        if self.n_cross_layers <= 0:
            raise ValueError("edcn_config.n_cross_layers must be positive")
        if self.temperature <= 0:
            raise ValueError("edcn_config.temperature must be positive")
        return self


def has_recommender_config(variant_config: VariantConfig) -> bool:
    return (
        variant_config.deepfm_config is not None
        or variant_config.dcn_config is not None
        or variant_config.dcnv2_config is not None
        or variant_config.edcn_config is not None
    )


def count_recommender_configs(variant_config: VariantConfig) -> int:
    return sum(
        config is not None
        for config in (
            variant_config.deepfm_config,
            variant_config.dcn_config,
            variant_config.dcnv2_config,
            variant_config.edcn_config,
        )
    )


class VariantConfig(StrictModel):
    target_input_channels: int | None = None
    target_output_classes: int | None = None
    example_input_shape: list[int]
    run_training: bool = False
    run_inference: bool = False
    run_prefill: bool = False
    pre_inference_cooldown_seconds: float = 3.0
    pre_prefill_cooldown_seconds: float = 3.0
    inference_measurement_min_seconds: float = 5.0
    prefill_measurement_min_seconds: float = 5.0
    export_onnx: bool = False
    onnx_export_mode: Literal["full", "architecture_only"] = "full"
    training_batch_sizes: list[int] = Field(default_factory=lambda: [32])
    training_measurement_min_seconds: float = 5.0
    use_fake_imagenet: bool = False
    use_fake_text_dataset: bool = False
    use_fake_recommender_dataset: bool = False
    use_real_text_dataset: bool = False
    max_sequence_length: int = 128
    gpt2_config: Gpt2ConfigOverride | None = None
    t5_config: T5ConfigOverride | None = None
    deepfm_config: DeepFMConfigOverride | None = None
    dcn_config: DCNConfigOverride | None = None
    dcnv2_config: DCNv2ConfigOverride | None = None
    edcn_config: EDCNConfigOverride | None = None

    @field_validator("training_batch_sizes", mode="before")
    @classmethod
    def normalize_training_batch_sizes(cls, value: int | list[int]) -> list[int]:
        if isinstance(value, int):
            return [value]
        return value

    @field_validator("example_input_shape")
    @classmethod
    def validate_example_input_shape(cls, value: list[int]) -> list[int]:
        if len(value) not in (1, 2, 4):
            raise ValueError(
                "example_input_shape must be [batch], [batch, seq_length] "
                "or [batch, channels, height, width]"
            )
        if any(item <= 0 for item in value):
            raise ValueError("example_input_shape values must be positive integers")
        return value

    @model_validator(mode="after")
    def validate_positive_fields(self) -> VariantConfig:
        positive_optional_fields = (
            ("target_input_channels", self.target_input_channels),
            ("target_output_classes", self.target_output_classes),
        )
        for field_name, value in positive_optional_fields:
            if value is not None and value <= 0:
                raise ValueError(f"{field_name} must be positive when provided")

        if self.training_measurement_min_seconds <= 0:
            raise ValueError("training_measurement_min_seconds must be positive")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if self.pre_inference_cooldown_seconds < 0:
            raise ValueError("pre_inference_cooldown_seconds must be non-negative")
        if self.pre_prefill_cooldown_seconds < 0:
            raise ValueError("pre_prefill_cooldown_seconds must be non-negative")
        if self.inference_measurement_min_seconds <= 0:
            raise ValueError("inference_measurement_min_seconds must be positive")
        if self.prefill_measurement_min_seconds <= 0:
            raise ValueError("prefill_measurement_min_seconds must be positive")
        if any(batch_size <= 0 for batch_size in self.training_batch_sizes):
            raise ValueError("training_batch_sizes must only contain positive integers")
        recommender_config_count = count_recommender_configs(self)
        if recommender_config_count > 1:
            raise ValueError("variant_config must define only one recommender config")
        if recommender_config_count == 0 and len(self.example_input_shape) == 1:
            raise ValueError("non-recommender variants require 2D or 4D input shapes")
        if recommender_config_count == 1:
            if len(self.example_input_shape) != 1:
                raise ValueError(
                    "recommender variants require example_input_shape [batch]"
                )
            if self.target_output_classes != 1:
                raise ValueError("recommender variants require target_output_classes=1")

        return self


class BaseModelConfig(StrictModel):
    name: str
    pretrained: bool = True

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("base model name must not be empty")
        return normalized_value


class SingleVariantDefinition(StrictModel):
    name: str
    variant_config: VariantConfig
    mutations: list[MutationConfig] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("variant name must not be empty")
        return normalized_value


class CombinatorialVariantGrid(StrictModel):
    input_channels: list[int]
    output_classes: list[int]
    base_variant_config_template: VariantConfig
    mutation_sets: list[MutationSet]

    @model_validator(mode="after")
    def validate_template_targets(self) -> CombinatorialVariantGrid:
        if self.base_variant_config_template.target_input_channels is not None:
            raise ValueError(
                "combinatorial_variant_grid.base_variant_config_template."
                "target_input_channels must be omitted"
            )
        if self.base_variant_config_template.target_output_classes is not None:
            raise ValueError(
                "combinatorial_variant_grid.base_variant_config_template."
                "target_output_classes must be omitted"
            )
        if not self.input_channels:
            raise ValueError("input_channels must not be empty")
        if not self.output_classes:
            raise ValueError("output_classes must not be empty")
        if not self.mutation_sets:
            raise ValueError("mutation_sets must not be empty")
        return self


class VariantConfigGridAxisValue(StrictModel):
    name: str
    overrides: dict[str, Any]

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("variant_config_grid axis value name must not be empty")
        return normalized_value

    @field_validator("overrides")
    @classmethod
    def validate_overrides(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError(
                "variant_config_grid axis value overrides must not be empty"
            )
        for field_path in value:
            if not field_path.strip():
                raise ValueError("variant_config_grid override path must not be empty")
            if any(not part for part in field_path.split(".")):
                raise ValueError(
                    f"variant_config_grid override path is invalid: {field_path}"
                )
        return value


class VariantConfigGridAxis(StrictModel):
    name: str
    values: list[VariantConfigGridAxisValue]

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("variant_config_grid axis name must not be empty")
        return normalized_value

    @model_validator(mode="after")
    def validate_values(self) -> VariantConfigGridAxis:
        if not self.values:
            raise ValueError("variant_config_grid axis values must not be empty")
        value_names = [value.name for value in self.values]
        if len(set(value_names)) != len(value_names):
            raise ValueError(
                f"variant_config_grid axis '{self.name}' value names must be unique"
            )
        return self


class VariantConfigGrid(StrictModel):
    base_variant_config_template: VariantConfig
    axes: list[VariantConfigGridAxis]
    mutations: list[MutationConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_axes(self) -> VariantConfigGrid:
        if not self.axes:
            raise ValueError("variant_config_grid.axes must not be empty")
        axis_names = [axis.name for axis in self.axes]
        if len(set(axis_names)) != len(axis_names):
            raise ValueError("variant_config_grid axis names must be unique")
        return self


class BaseModelGroup(StrictModel):
    base_model: BaseModelConfig
    single_variant_define: list[SingleVariantDefinition] = Field(default_factory=list)
    combinatorial_variant_grid: CombinatorialVariantGrid | None = None
    variant_config_grid: VariantConfigGrid | None = None
    fc_mutation_sets: list[MutationSet] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_variant_source(self) -> BaseModelGroup:
        if (
            self.combinatorial_variant_grid is not None
            and self.variant_config_grid is not None
        ):
            raise ValueError(
                "base_model_group must not define both combinatorial_variant_grid "
                "and variant_config_grid"
            )
        if (
            not self.single_variant_define
            and self.combinatorial_variant_grid is None
            and self.variant_config_grid is None
        ):
            raise ValueError(
                "base_model_group must define single_variant_define or a variant grid"
            )
        for variant in self.single_variant_define:
            self.validate_single_variant_targets(variant)
        return self

    def validate_single_variant_targets(
        self,
        variant: SingleVariantDefinition,
    ) -> None:
        if has_recommender_config(variant.variant_config):
            return
        if is_qwen_model_name(self.base_model.name):
            if variant.variant_config.target_input_channels is not None:
                raise ValueError("qwen variants must omit target_input_channels")
            if variant.variant_config.target_output_classes is not None:
                raise ValueError("qwen variants must omit target_output_classes")
            return
        if variant.variant_config.target_input_channels is None:
            raise ValueError(
                "single_variant_define.variant_config.target_input_channels is required"
            )
        if variant.variant_config.target_output_classes is None:
            raise ValueError(
                "single_variant_define.variant_config.target_output_classes is required"
            )


class ArchConfig(StrictModel):
    base_model_groups: list[BaseModelGroup]

    @model_validator(mode="after")
    def validate_groups(self) -> ArchConfig:
        if not self.base_model_groups:
            raise ValueError("base_model_groups must not be empty")
        return self


class ResolvedVariantSpec(StrictModel):
    name: str
    base_model: BaseModelConfig
    variant_config: VariantConfig
    mutations: list[MutationConfig]
    source: str
    group_total_variants_defined: int
