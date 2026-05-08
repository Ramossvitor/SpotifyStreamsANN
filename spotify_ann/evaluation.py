"""Test-set evaluation: forward pass + raw and readable metrics."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_percentage_metrics(
    y_true: np.ndarray, predictions: np.ndarray, tolerance: float
) -> tuple[float, float]:
    """Return (MAPE %, accuracy % within ±tolerance), masking zero targets."""
    mask = y_true != 0
    rel_err = np.abs((y_true[mask] - predictions[mask]) / y_true[mask])
    within_tol = rel_err <= tolerance
    return float(rel_err.mean() * 100), float(within_tol.mean() * 100)


def evaluate_model(
    model: nn.Module,
    X_test_t: torch.Tensor,
    y_test: np.ndarray,
    device: str,
    tolerance: float,
) -> tuple[np.ndarray, dict]:
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_t.to(device)).cpu().numpy()

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    rmse = float(np.sqrt(mse))

    mape, accuracy_pct = compute_percentage_metrics(y_test, predictions, tolerance)

    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "accuracy_pct": accuracy_pct,
    }
    return predictions, metrics


def print_metrics(metrics: dict, tolerance: float, n_test: int) -> None:
    print("=== Raw-scale metrics ===")
    print(f"MSE : {metrics['mse']:.3e}")
    print(f"RMSE: {metrics['rmse']:.3e}  (avg error in streams)")
    print(f"MAE : {metrics['mae']:.3e}")
    print(f"R^2 : {metrics['r2']:.4f}")

    print("\n=== Readable metrics ===")
    print(f"MAPE                  : {metrics['mape']:.1f}%   (avg % error per track)")
    count = int(round(metrics["accuracy_pct"] * n_test / 100))
    print(
        f"Accuracy (within ±{tolerance*100:.0f}%) : {metrics['accuracy_pct']:.1f}%   "
        f"({count}/{n_test} tracks)"
    )
