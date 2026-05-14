"""Test-set evaluation: forward pass + classification metrics."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    top_k_accuracy_score,
)

from . import config


def compute_classification_metrics(
    y_true: np.ndarray, logits: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """Return (top-1 accuracy %, top-k accuracy %, argmax predictions)."""
    predictions = logits.argmax(axis=1)
    top1 = accuracy_score(y_true, predictions) * 100
    num_classes = logits.shape[1]
    topk = (
        top_k_accuracy_score(
            y_true, logits, k=config.TOP_K, labels=np.arange(num_classes)
        )
        * 100
    )
    return float(top1), float(topk), predictions


def evaluate_model(
    model: nn.Module,
    X_test_t: torch.Tensor,
    y_test: np.ndarray,
    device: str,
    class_names: np.ndarray,
) -> tuple[np.ndarray, dict]:
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t.to(device)).cpu().numpy()

    top1, topk, predictions = compute_classification_metrics(y_test, logits)
    macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0,
        digits=3,
    )

    metrics = {
        "top1": top1,
        "topk": topk,
        "macro_f1": float(macro_f1),
        "report": report,
    }
    return predictions, metrics


def print_metrics(metrics: dict, n_test: int) -> None:
    print("=== Classification metrics ===")
    print(f"Top-1 accuracy : {metrics['top1']:.2f}%   ({n_test} test tracks)")
    print(f"Top-{config.TOP_K} accuracy : {metrics['topk']:.2f}%")
    print(f"Macro F1       : {metrics['macro_f1']:.4f}")
    print("\n=== Per-class report ===")
    print(metrics["report"])
