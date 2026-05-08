"""Dataset loading, preprocessing, and DataLoader construction."""

from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from . import config


def load_dataset() -> pd.DataFrame:
    if config.LOCAL_CSV.exists():
        csv_path = config.LOCAL_CSV
    else:
        dataset_dir = kagglehub.dataset_download(config.KAGGLE_DATASET)
        csv_path = Path(dataset_dir) / "spotify-2023.csv"

    print("Reading:", csv_path)
    df = pd.read_csv(csv_path, encoding="latin-1")
    print(df.head())

    df[config.TARGET_COLUMN] = pd.to_numeric(df[config.TARGET_COLUMN], errors="coerce")
    return df


def summarize_dataset(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing values per column:\n", df.isna().sum())
    print(f"\n{config.TARGET_COLUMN} describe:\n", df[config.TARGET_COLUMN].describe())


def preprocess(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tracks = df.copy()
    tracks = tracks.dropna(subset=[config.TARGET_COLUMN])
    tracks = tracks.drop(columns=config.DROP_COLUMNS)

    for col in tracks.columns:
        tracks[col] = pd.to_numeric(tracks[col], errors="coerce")

    tracks = tracks.dropna()
    print("Cleaned shape:", tracks.shape)

    features_df = tracks.drop(columns=[config.TARGET_COLUMN])
    y = tracks[config.TARGET_COLUMN].values.astype(np.float32)
    X = features_df.values.astype(np.float32)
    print("Number of features:", X.shape[1])
    print("Features:", features_df.columns.tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED
    )

    # Standardize for optimization stability — not regularization (intentional).
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    print("X_train:", X_train.shape, "X_test:", X_test.shape)

    return X_train, X_test, y_train, y_test


def make_loaders(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, X_test_t
