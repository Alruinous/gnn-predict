from __future__ import annotations

from typing import Any, Literal

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


# ============== 新增：Detection 模型识别 ==============

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


class VariantConfig(StrictModel):
    target_input_channels: int | None = None
    target_output_classes: int | None = None
    example_input_shape: list[int]
    run_training: bool = False
    run_inference: bool = False
    pre_inference_cooldown_seconds: float = 3.0
    inference_measurement_min_seconds: float = 5.0
    export_onnx: bool = False
    onnx_export_mode: Literal["full", "architecture_only"] = "full"
    training_batch_sizes: list[int] = Field(default_factory=lambda: [32])
    training_epochs: int = 1
    use_fake_imagenet: bool = False
    fake_dataset_size: int = 1000
    use_fake_text_dataset: bool = False
    use_real_text_dataset: bool = False
    max_sequence_length: int = 128

    @field_validator("training_batch_sizes", mode="before")
    @classmethod
    def normalize_training_batch_sizes(cls, value: int | list[int]) -> list[int]:
        if isinstance(value, int):
            return [value]
        return value

    @field_validator("example_input_shape")
    @classmethod
    def validate_example_input_shape(cls, value: list[int]) -> list[int]:
        if len(value) not in (2, 4):
            raise ValueError(
                "example_input_shape must be [batch, seq_length] "
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

        if self.training_epochs <= 0:
            raise ValueError("training_epochs must be positive")
        if self.fake_dataset_size <= 0:
            raise ValueError("fake_dataset_size must be positive")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if self.pre_inference_cooldown_seconds < 0:
            raise ValueError("pre_inference_cooldown_seconds must be non-negative")
        if self.inference_measurement_min_seconds <= 0:
            raise ValueError("inference_measurement_min_seconds must be positive")
        if any(batch_size <= 0 for batch_size in self.training_batch_sizes):
            raise ValueError("training_batch_sizes must only contain positive integers")

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

    @model_validator(mode="after")
    def require_explicit_targets(self) -> SingleVariantDefinition:
        if self.variant_config.target_input_channels is None:
            raise ValueError(
                "single_variant_define.variant_config.target_input_channels is required"
            )
        if self.variant_config.target_output_classes is None:
            raise ValueError(
                "single_variant_define.variant_config.target_output_classes is required"
            )
        return self


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


class BaseModelGroup(StrictModel):
    base_model: BaseModelConfig
    single_variant_define: list[SingleVariantDefinition] = Field(default_factory=list)
    combinatorial_variant_grid: CombinatorialVariantGrid | None = None
    fc_mutation_sets: list[MutationSet] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_variant_source(self) -> BaseModelGroup:
        if not self.single_variant_define and self.combinatorial_variant_grid is None:
            raise ValueError(
                "base_model_group must define single_variant_define "
                "or combinatorial_variant_grid"
            )
        return self


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
