from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from gnn_model.config import (
    ExperimentConfig,
    FakeDataConfig,
    PreparedDataConfig,
    load_experiment_config,
)
from gnn_model.data import GraphDatasetBundle, build_fake_graph_datasets
from gnn_model.data.constants import (
    DEFAULT_TARGET_NAMES,
    EDGE_FEATURE_DIM,
    GRAPH_FEATURE_DIM,
    GRAPH_METRIC_DIM,
    NODE_FEATURE_DIM,
)
from gnn_model.data.prepared_dataset import load_prepared_graph_datasets
from gnn_model.models import IntelliGraphLargeModelPredictor
from gnn_model.result import (
    DatasetSummary,
    EvaluationSummary,
    ExperimentResult,
    TimeWindow,
    TrainingSummary,
    write_result_document,
)
from gnn_model.training import evaluate_model, train_model


@dataclass(frozen=True)
class OutputLayout:
    root: Path
    logs_dir: Path
    results_dir: Path
    checkpoints_dir: Path


def run_experiment(
    *,
    config_path: Path,
    output_dir: Path,
    requested_device: str | None = None,
) -> tuple[ExperimentResult, Path]:
    config = load_experiment_config(config_path)
    output_layout = prepare_output_layout(output_dir, config.experiment_name)
    logger = configure_logging(output_layout, config.experiment_name)
    device = resolve_device(requested_device)
    torch.manual_seed(resolve_random_seed(config))
    timings: dict[str, TimeWindow] = {}

    experiment_started_at = time.time()
    data_started_at = time.time()
    datasets = load_dataset_bundle(config)
    timings["data"] = build_time_window(data_started_at, time.time())

    model_started_at = time.time()
    model = build_model(config)
    model = model.to(device)
    timings["model_build"] = build_time_window(model_started_at, time.time())

    training_started_at = time.time()
    training_artifacts = train_model(
        model=model,
        train_data=datasets.train_data,
        val_data=datasets.val_data,
        training_config=config.training,
        checkpoint_dir=output_layout.checkpoints_dir,
        device=device,
        logger=logger,
    )
    timings["training"] = build_time_window(training_started_at, time.time())

    evaluation_started_at = time.time()
    evaluation_metrics = evaluate_model(
        model=model,
        dataset=datasets.test_data,
        batch_size=config.training.batch_size,
        device=device,
        target_names=list(DEFAULT_TARGET_NAMES),
    )
    timings["evaluation"] = build_time_window(evaluation_started_at, time.time())
    timings["full"] = build_time_window(experiment_started_at, time.time())

    result = ExperimentResult(
        experiment_name=config.experiment_name,
        config_path=str(Path(config_path).resolve()),
        device=str(device),
        target_names=list(DEFAULT_TARGET_NAMES),
        dataset=DatasetSummary(
            train_count=len(datasets.train_data),
            val_count=len(datasets.val_data),
            test_count=len(datasets.test_data),
        ),
        timings=timings,
        training=TrainingSummary(
            batch_size=config.training.batch_size,
            num_epochs=config.training.num_epochs,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            best_epoch=training_artifacts.best_epoch,
            best_val_loss=training_artifacts.best_val_loss,
            last_train_loss=training_artifacts.last_train_loss,
            checkpoint_path=str(training_artifacts.checkpoint_path),
        ),
        evaluation=EvaluationSummary(
            split="test",
            metrics=evaluation_metrics,
        ),
        metadata={
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "hidden_dim": config.model.hidden_dim,
            "num_layers": config.model.num_layers,
        },
    )
    result_path = (
        output_layout.results_dir
        / f"{config.experiment_name}_{int(experiment_started_at)}_result.json"
    )
    write_result_document(result_path, result)
    logger.info("wrote result document to %s", result_path)
    return result, result_path


def prepare_output_layout(output_root: Path, experiment_name: str) -> OutputLayout:
    resolved_root = Path(output_root).resolve() / experiment_name
    layout = OutputLayout(
        root=resolved_root,
        logs_dir=resolved_root / "logs",
        results_dir=resolved_root / "results",
        checkpoints_dir=resolved_root / "checkpoints",
    )
    for directory in (
        layout.root,
        layout.logs_dir,
        layout.results_dir,
        layout.checkpoints_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return layout


def configure_logging(
    output_layout: OutputLayout,
    experiment_name: str,
) -> logging.Logger:
    logger = logging.getLogger(f"gnn_model.{experiment_name}")
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log_path = output_layout.logs_dir / f"run_{int(time.time())}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def load_dataset_bundle(config: ExperimentConfig) -> GraphDatasetBundle:
    if isinstance(config.data, FakeDataConfig):
        return build_fake_graph_datasets(
            dataset_size=config.data.dataset_size,
            seed=config.data.random_seed,
            input_shape=config.data.input_shape,
            val_ratio=config.data.val_ratio,
            test_ratio=config.data.test_ratio,
        )
    if isinstance(config.data, PreparedDataConfig):
        load_prepared_graph_datasets(config.data)
    raise TypeError(f"unsupported data config: {type(config.data)}")


def build_model(config: ExperimentConfig) -> IntelliGraphLargeModelPredictor:
    return IntelliGraphLargeModelPredictor(
        node_dim=NODE_FEATURE_DIM,
        edge_dim=EDGE_FEATURE_DIM,
        graph_dim=GRAPH_FEATURE_DIM,
        graph_metric_dim=GRAPH_METRIC_DIM,
        hidden_dim=config.model.hidden_dim,
        targets=list(DEFAULT_TARGET_NAMES),
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        dropout_rate=config.model.dropout_rate,
    )


def resolve_random_seed(config: ExperimentConfig) -> int:
    if isinstance(config.data, FakeDataConfig):
        return config.data.random_seed
    return 7


def resolve_device(requested_device: str | None) -> torch.device:
    if requested_device is not None:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_time_window(started_at: float, ended_at: float) -> TimeWindow:
    return TimeWindow(
        started_at_ts=started_at,
        ended_at_ts=ended_at,
        started_at_text=format_timestamp(started_at),
        ended_at_text=format_timestamp(ended_at),
    )


def format_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, tz=UTC).isoformat()
