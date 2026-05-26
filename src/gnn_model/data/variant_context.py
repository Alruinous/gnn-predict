from __future__ import annotations

import math
import re

ARCH_FAMILY_NAMES = (
    "beit",
    "bert_large",
    "bert",
    "convnext",
    "densenet",
    "efficientnet",
    "gpt2",
    "mobilenet",
    "resnet152",
    "resnet101",
    "resnet50",
    "resnet",
    "swin",
    "vgg19",
    "vgg16",
    "vgg11",
    "vit",
    "yolov11",
    "yolov9",
    "yolov5",
    "deepfm",
    "dcnv2",
    "dcn",
    "edcn",
)

SIZE_TOKENS = (
    "pico",
    "micro",
    "nano",
    "tiny",
    "small",
    "medium",
    "base",
    "large",
    "wide",
    "narrow",
    "deep",
    "light",
    "heavy",
)

ACTIVATION_TOKENS = (
    "relu",
    "gelu",
    "silu",
    "swish",
    "hardswish",
    "leaky",
    "elu",
    "selu",
    "sigmoid",
    "tanh",
)

MUTATION_TOKENS = (
    "activation",
    "backbone",
    "blocks",
    "c2f",
    "c3",
    "c3k2",
    "channel",
    "combo",
    "combo2",
    "combo3",
    "conv",
    "depth",
    "double",
    "dropout",
    "fc",
    "front",
    "global",
    "head",
    "kernel",
    "layer",
    "layers",
    "multiple",
    "pool",
    "prune",
    "pruning",
    "repeat",
    "scale",
    "size",
    "sppf",
    "sppf7",
    "to",
    "transition",
)

NUMERIC_PREFIX_NAMES = (
    "input_channels",
    "output_classes",
    "kernel",
    "depth",
    "channel",
    "head",
    "hidden",
    "layers",
    "attention_heads",
    "scale",
    "sppf",
    "dropout",
    "repeat",
    "blocks",
    "combo",
    "c_block",
)

NUMERIC_SUMMARY_FEATURE_NAMES = (
    "variant_token_count_log",
    "variant_unique_token_count_log",
    "variant_number_count_log",
    "variant_number_sum_log",
    "variant_number_mean_log",
    "variant_number_std_log",
    "variant_number_min_log",
    "variant_number_max_log",
    "variant_number_first_log",
    "variant_number_last_log",
)

VARIANT_CONTEXT_FEATURE_NAMES = (
    *(f"variant_family_{name}" for name in ARCH_FAMILY_NAMES),
    "variant_family_known",
    *(f"variant_size_{name}" for name in SIZE_TOKENS),
    *(f"variant_activation_{name}" for name in ACTIVATION_TOKENS),
    *(f"variant_mutation_{name}" for name in MUTATION_TOKENS),
    *NUMERIC_SUMMARY_FEATURE_NAMES,
    *(f"variant_numeric_{name}_log" for name in NUMERIC_PREFIX_NAMES),
)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_NUMBER_RE = re.compile(r"\d+(?:[p.]\d+)?")
_PREFIX_NUMBER_RE = re.compile(r"([a-zA-Z]+)(\d+(?:p\d+)?)")

_PREFIX_ALIASES = {
    "ic": "input_channels",
    "oc": "output_classes",
    "kernel": "kernel",
    "k": "kernel",
    "depth": "depth",
    "channel": "channel",
    "head": "head",
    "h": "hidden",
    "layer": "layers",
    "layers": "layers",
    "l": "layers",
    "a": "attention_heads",
    "scale": "scale",
    "sppf": "sppf",
    "dropout": "dropout",
    "repeat": "repeat",
    "blocks": "blocks",
    "combo": "combo",
    "c": "c_block",
}


def build_variant_context_feature_vector(
    *,
    model_name: str = "",
    variant_name: str = "",
) -> list[float]:
    tokens = tokenize_variant_context(model_name, variant_name)
    family = resolve_arch_family(tokens)
    family_features = [1.0 if family == name else 0.0 for name in ARCH_FAMILY_NAMES]
    family_known = [1.0 if family is not None else 0.0]
    size_features = build_token_flags(tokens, SIZE_TOKENS)
    activation_features = build_token_flags(tokens, ACTIVATION_TOKENS)
    mutation_features = build_token_flags(tokens, MUTATION_TOKENS)
    numeric_features = build_numeric_summary(tokens)
    prefix_features = build_numeric_prefix_features(tokens)
    features = [
        *family_features,
        *family_known,
        *size_features,
        *activation_features,
        *mutation_features,
        *numeric_features,
        *prefix_features,
    ]
    assert len(features) == len(VARIANT_CONTEXT_FEATURE_NAMES)
    return features


def tokenize_variant_context(*values: str) -> tuple[str, ...]:
    text = "_".join(
        value.strip().lower() for value in values if value.strip()
    )
    return tuple(match.group(0) for match in _TOKEN_RE.finditer(text))


def resolve_arch_family(tokens: tuple[str, ...]) -> str | None:
    joined = "_".join(tokens)
    if joined.startswith("yolo11"):
        return "yolov11"
    if joined.startswith("yolov9"):
        return "yolov9"
    if joined.startswith("yolov5"):
        return "yolov5"
    token_set = set(tokens)
    for family in ARCH_FAMILY_NAMES:
        if family in token_set or joined.startswith(family):
            return family
    return None


def build_token_flags(
    tokens: tuple[str, ...],
    vocabulary: tuple[str, ...],
) -> list[float]:
    token_set = set(tokens)
    return [1.0 if token in token_set else 0.0 for token in vocabulary]


def build_numeric_summary(tokens: tuple[str, ...]) -> list[float]:
    numbers = [number for token in tokens for number in extract_numbers(token)]
    if not numbers:
        return [math.log1p(float(len(tokens))), math.log1p(float(len(set(tokens))))] + [
            0.0
        ] * 8
    values = [float(value) for value in numbers]
    count = float(len(values))
    total = sum(values)
    mean = total / count
    variance = sum((value - mean) ** 2 for value in values) / count
    return [
        math.log1p(float(len(tokens))),
        math.log1p(float(len(set(tokens)))),
        math.log1p(count),
        math.log1p(total),
        math.log1p(mean),
        math.log1p(math.sqrt(variance)),
        math.log1p(min(values)),
        math.log1p(max(values)),
        math.log1p(values[0]),
        math.log1p(values[-1]),
    ]


def build_numeric_prefix_features(tokens: tuple[str, ...]) -> list[float]:
    values = {name: 0.0 for name in NUMERIC_PREFIX_NAMES}
    for token in tokens:
        for prefix, raw_number in _PREFIX_NUMBER_RE.findall(token):
            canonical = _PREFIX_ALIASES.get(prefix.lower())
            if canonical is None:
                continue
            values[canonical] = max(values[canonical], parse_number(raw_number))
    return [math.log1p(values[name]) for name in NUMERIC_PREFIX_NAMES]


def extract_numbers(token: str) -> list[float]:
    return [parse_number(match.group(0)) for match in _NUMBER_RE.finditer(token)]


def parse_number(value: str) -> float:
    return float(value.replace("p", "."))
