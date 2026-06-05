"""Training loop for the genre-classification ANN."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from . import config


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_samples = 0
    correct_top1 = 0
    correct_top5 = 0
    # Per-epoch top-1/top-5 are counted inline on-device for speed; the final
    # end-of-run report uses evaluation.compute_classification_metrics (sklearn)
    # so per-class F1 / classification_report are available. Both paths must
    # agree numerically on top-1 and top-5.
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
            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            _, top5_preds = logits.topk(config.TOP_K, dim=1)
            target_in_top5 = (top5_preds == targets.unsqueeze(1)).any(dim=1)
            correct_top1 += (logits.argmax(dim=1) == targets).sum().item()
            correct_top5 += target_in_top5.sum().item()
    avg_loss = total_loss / total_samples
    top1_pct = 100.0 * correct_top1 / total_samples
    top5_pct = 100.0 * correct_top5 / total_samples
    return avg_loss, top1_pct, top5_pct


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, list[float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "test_loss": [],
        "train_top1": [],
        "test_top1": [],
        "train_top5": [],
        "test_top5": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_top1, train_top5 = _run_epoch(
            model, train_loader, device, criterion, optimizer
        )
        test_loss, test_top1, test_top5 = _run_epoch(
            model, test_loader, device, criterion
        )

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
