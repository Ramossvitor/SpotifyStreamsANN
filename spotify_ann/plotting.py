"""Curve plots and confusion matrix, written to disk."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


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
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_values, label=train_label)
    ax.plot(test_values, label=test_label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: np.ndarray,
    output_path: Path,
) -> None:
    labels = np.arange(len(class_names))
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("Predicted genre")
    ax.set_ylabel("True genre")
    ax.set_title("Confusion matrix (row-normalized)")
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(class_names, rotation=90, fontsize=5)
    ax.set_yticklabels(class_names, fontsize=5)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
