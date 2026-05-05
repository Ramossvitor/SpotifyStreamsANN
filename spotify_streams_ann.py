"""Predicting Spotify Streams with a basic ANN.

Dataset: Top Spotify Songs 2023
    https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023

Goal: minimal fully-connected feed-forward network in PyTorch that predicts
the `streams` column from numeric features.

Constraints (on purpose, for learning):
- No RNN / CNN — only nn.Linear + ReLU.
- No anti-overfitting tricks: no dropout, no weight decay, no batch norm,
  no early stopping. We *want* to see the model overfit.
- Raw `streams` as the target (no log / standardize). MSE values will look
  enormous — that is intentional and instructive.

Run:
    pip install kagglehub
    python spotify_streams_ann.py

Plots are saved to ./plots/.
"""

from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 2. Load the dataset
# ---------------------------------------------------------------------------
# Prefer the local committed CSV; fall back to kagglehub if it's missing.
local_csv = Path(__file__).parent / "spotify-2023.csv"
if local_csv.exists():
    csv_path = local_csv
else:
    dataset_dir = kagglehub.dataset_download("nelgiriyewithana/top-spotify-songs-2023")
    csv_path = Path(dataset_dir) / "spotify-2023.csv"

print("Reading:", csv_path)
df = pd.read_csv(csv_path, encoding="latin-1")
print(df.head())


# ---------------------------------------------------------------------------
# 3. Quick EDA
# ---------------------------------------------------------------------------
print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nMissing values per column:\n", df.isna().sum())

df["streams"] = pd.to_numeric(df["streams"], errors="coerce")
print("\nstreams describe:\n", df["streams"].describe())


# ---------------------------------------------------------------------------
# 4. Preprocessing
# ---------------------------------------------------------------------------
tracks = df.copy()
tracks = tracks.dropna(subset=["streams"])

drop_cols = ["track_name", "artist(s)_name", "key", "mode"]
tracks = tracks.drop(columns=drop_cols)

for col in tracks.columns:
    tracks[col] = pd.to_numeric(tracks[col], errors="coerce")

tracks = tracks.dropna()
print("Cleaned shape:", tracks.shape)

features_df = tracks.drop(columns=["streams"])
y = tracks["streams"].values.astype(np.float32)
X = features_df.values.astype(np.float32)
feature_names = features_df.columns.tolist()
print("Number of features:", X.shape[1])
print("Features:", feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize for optimization stability — not regularization (intentional).
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)
print("X_train:", X_train.shape, "X_test:", X_test.shape)


# ---------------------------------------------------------------------------
# 5. Tensors & DataLoaders
# ---------------------------------------------------------------------------
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds = TensorDataset(X_test_t, y_test_t)

BATCH_SIZE = 32
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


# ---------------------------------------------------------------------------
# 6. Model
# ---------------------------------------------------------------------------
class StreamsANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


input_dim = X_train.shape[1]
model = StreamsANN(input_dim).to(device)
print(model)


# ---------------------------------------------------------------------------
# 7. Loss & optimizer
# ---------------------------------------------------------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# ---------------------------------------------------------------------------
# 8. Training loop
# ---------------------------------------------------------------------------
EPOCHS = 100
train_losses = []
test_losses = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_train_loss = 0.0
    n_train = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * xb.size(0)
        n_train += xb.size(0)
    epoch_train_loss /= n_train

    model.eval()
    epoch_test_loss = 0.0
    n_test = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            epoch_test_loss += loss.item() * xb.size(0)
            n_test += xb.size(0)
    epoch_test_loss /= n_test

    train_losses.append(epoch_train_loss)
    test_losses.append(epoch_test_loss)

    if epoch % 10 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:3d} | train MSE: {epoch_train_loss:.3e} "
            f"| test MSE: {epoch_test_loss:.3e}"
        )


# ---------------------------------------------------------------------------
# 9. Evaluate on the test set
# ---------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    preds = model(X_test_t.to(device)).cpu().numpy()

mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
rmse = np.sqrt(mse)

print(f"MSE : {mse:.3e}")
print(f"RMSE: {rmse:.3e}  (avg error in streams)")
print(f"MAE : {mae:.3e}")
print(f"R^2 : {r2:.4f}")
print(f"\nFor reference, std(streams) on test set: {y_test.std():.3e}")


# ---------------------------------------------------------------------------
# 10. Plots
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="train MSE")
plt.plot(test_losses, label="test MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE (raw streams)")
plt.yscale("log")
plt.title("Training vs test loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(PLOTS_DIR / "loss_curves.png", dpi=120, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds, alpha=0.5)
lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
plt.plot(lims, lims, "r--", label="perfect prediction")
plt.xlabel("Actual streams")
plt.ylabel("Predicted streams")
plt.title("Predicted vs actual")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(PLOTS_DIR / "predicted_vs_actual.png", dpi=120, bbox_inches="tight")
plt.close()

print(f"\nPlots saved to {PLOTS_DIR}")
