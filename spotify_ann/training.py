"""Training loop for the genre-classification ANN."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .evaluation import compute_classification_metrics


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    n = 0
    logits_chunks: list[np.ndarray] = []
    targets_chunks: list[np.ndarray] = []
    with torch.set_grad_enabled(is_training):
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            if is_training:
                optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, targets)
            if is_training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * features.size(0)
            n += features.size(0)
            logits_chunks.append(logits.detach().cpu().numpy())
            targets_chunks.append(targets.detach().cpu().numpy())
    return total_loss / n, np.concatenate(logits_chunks), np.concatenate(targets_chunks)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    epochs: int,
    learning_rate: float,
) -> dict[str, list[float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "test_loss": [],
        "train_top1": [],
        "test_top1": [],
        "train_top5": [],
        "test_top5": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_logits, train_targets = _run_epoch(
            model, train_loader, device, criterion, optimizer
        )
        test_loss, test_logits, test_targets = _run_epoch(
            model, test_loader, device, criterion
        )

        train_top1, train_top5, _ = compute_classification_metrics(train_targets, train_logits)
        test_top1, test_top5, _ = compute_classification_metrics(test_targets, test_logits)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_top1"].append(train_top1)
        history["test_top1"].append(test_top1)
        history["train_top5"].append(train_top5)
        history["test_top5"].append(test_top5)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | train loss: {train_loss:.4f} "
                f"| test loss: {test_loss:.4f} "
                f"| test top-1: {test_top1:.2f}% "
                f"| test top-5: {test_top5:.2f}%"
            )

    return history
