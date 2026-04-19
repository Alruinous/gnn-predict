from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from gnn_model.evaluation.metrics import (
    TargetScaler,
    compute_original_scale_metrics,
    compute_regression_metrics,
)

if TYPE_CHECKING:
    import logging

    from gnn_model.config import TrainingConfig


@dataclass(frozen=True)
class TrainingArtifacts:
    checkpoint_path: Path
    best_epoch: int
    best_val_loss: float
    last_train_loss: float


class WeightedSmoothL1Loss(torch.nn.Module):
    def __init__(self, weights: list[float], device: torch.device) -> None:
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32).to(device))

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if predictions.shape != targets.shape:
            raise ValueError(
                f"prediction shape mismatch: {predictions.shape} != {targets.shape}"
            )
        loss = F.smooth_l1_loss(predictions, targets, reduction="none")
        return (loss * self.weights).mean()


def train_model(
    *,
    model: torch.nn.Module,
    train_data: list[object],
    val_data: list[object],
    training_config: TrainingConfig,
    checkpoint_dir: Path,
    device: torch.device,
    logger: logging.Logger,
) -> TrainingArtifacts:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    criterion = WeightedSmoothL1Loss(
        resolve_loss_weights(
            training_config.loss_weights,
            target_dim=train_data[0].y.size(-1),
        ),
        device=device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=training_config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=training_config.batch_size,
        shuffle=False,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    best_epoch = 0
    last_train_loss = float("inf")

    for epoch in range(1, training_config.num_epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_examples = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            predictions = model(batch)
            targets = batch.y.float()
            if targets.dim() == 1:
                targets = targets.unsqueeze(0)
            loss = criterion(predictions, targets)
            if not torch.isfinite(loss):
                raise ValueError("training loss became non-finite")
            loss.backward()
            optimizer.step()
            batch_examples = targets.size(0)
            total_train_loss += float(loss.item()) * batch_examples
            total_examples += batch_examples
        last_train_loss = total_train_loss / max(total_examples, 1)
        val_loss = evaluate_loss(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )
        logger.info(
            "epoch=%s train_loss=%.6f val_loss=%.6f",
            epoch,
            last_train_loss,
            val_loss,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise ValueError("training did not produce a best checkpoint state")

    checkpoint_path = checkpoint_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        },
        checkpoint_path,
    )
    model.load_state_dict(best_state)
    return TrainingArtifacts(
        checkpoint_path=checkpoint_path,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        last_train_loss=last_train_loss,
    )


def evaluate_model(
    *,
    model: torch.nn.Module,
    dataset: list[object],
    batch_size: int,
    device: torch.device,
    target_names: list[str],
    target_scalers: dict[str, TargetScaler] | None = None,
) -> dict[str, float]:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions, targets = collect_predictions(
        model=model,
        dataloader=dataloader,
        device=device,
    )
    metrics = compute_regression_metrics(predictions, targets, target_names)
    if target_scalers is not None:
        metrics.update(
            compute_original_scale_metrics(
                predictions,
                targets,
                target_names,
                target_scalers,
            )
        )
    return metrics


def evaluate_loss(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: WeightedSmoothL1Loss,
    device: torch.device,
) -> float:
    predictions, targets = collect_predictions(
        model=model,
        dataloader=dataloader,
        device=device,
    )
    loss = criterion(predictions, targets)
    if not torch.isfinite(loss):
        raise ValueError("validation loss became non-finite")
    return float(loss.item())


def collect_predictions(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            batch_predictions = model(batch).cpu()
            batch_targets = batch.y.float()
            if batch_targets.dim() == 1:
                batch_targets = batch_targets.unsqueeze(0)
            targets.append(batch_targets.cpu())
            predictions.append(batch_predictions)
    if not predictions or not targets:
        raise ValueError("evaluation dataset must not be empty")
    return torch.cat(predictions, dim=0), torch.cat(targets, dim=0)


def resolve_loss_weights(
    loss_weights: list[float] | None,
    *,
    target_dim: int,
) -> list[float]:
    if loss_weights is None:
        return [1.0] * target_dim
    if len(loss_weights) != target_dim:
        raise ValueError(
            f"loss_weights length mismatch: {len(loss_weights)} != {target_dim}"
        )
    return loss_weights
