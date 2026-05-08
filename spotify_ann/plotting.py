"""Curve plots and predicted-vs-actual scatter, written to disk."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_curves(
    train_values: list[float],
    test_values: list[float],
    output_path: Path,
    *,
    ylabel: str,
    title: str,
    train_label: str,
    test_label: str,
    yscale: str = "linear",
    ylim: tuple[float, float] | None = None,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(train_values, label=train_label)
    plt.plot(test_values, label=test_label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.yscale(yscale)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_predicted_vs_actual(
    y_test: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    lims = [
        min(y_test.min(), predictions.min()),
        max(y_test.max(), predictions.max()),
    ]
    plt.plot(lims, lims, "r--", label="perfect prediction")
    plt.xlabel("Actual streams")
    plt.ylabel("Predicted streams")
    plt.title("Predicted vs actual")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
