"""Training loop for the streams ANN."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .evaluation import compute_percentage_metrics


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
    preds_chunks: list[np.ndarray] = []
    targets_chunks: list[np.ndarray] = []
    with torch.set_grad_enabled(is_training):
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            if is_training:
                optimizer.zero_grad()
            pred = model(features)
            loss = criterion(pred, targets)
            if is_training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * features.size(0)
            n += features.size(0)
            preds_chunks.append(pred.detach().cpu().numpy())
            targets_chunks.append(targets.detach().cpu().numpy())
    return total_loss / n, np.concatenate(preds_chunks), np.concatenate(targets_chunks)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    epochs: int,
    learning_rate: float,
    tolerance: float,
) -> dict[str, list[float]]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history: dict[str, list[float]] = {
        "train_mse": [],
        "test_mse": [],
        "train_mape": [],
        "test_mape": [],
        "train_accuracy": [],
        "test_accuracy": [],
    }

    for epoch in range(1, epochs + 1):
        train_mse, train_preds, train_targets = _run_epoch(
            model, train_loader, device, criterion, optimizer
        )
        test_mse, test_preds, test_targets = _run_epoch(
            model, test_loader, device, criterion
        )

        train_mape, train_acc = compute_percentage_metrics(
            train_targets, train_preds, tolerance
        )
        test_mape, test_acc = compute_percentage_metrics(
            test_targets, test_preds, tolerance
        )

        history["train_mse"].append(train_mse)
        history["test_mse"].append(test_mse)
        history["train_mape"].append(train_mape)
        history["test_mape"].append(test_mape)
        history["train_accuracy"].append(train_acc)
        history["test_accuracy"].append(test_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | train MSE: {train_mse:.3e} "
                f"| test MSE: {test_mse:.3e}"
            )

    return history
