"""Dataset loading, preprocessing, and DataLoader construction."""

from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from . import config


def load_dataset() -> pd.DataFrame:
    if config.LOCAL_CSV.exists():
        csv_path = config.LOCAL_CSV
    else:
        dataset_dir = kagglehub.dataset_download(config.KAGGLE_DATASET)
        csv_path = Path(dataset_dir) / "dataset.csv"

    print("Reading:", csv_path)
    # `index_col=0` discards the unnamed integer column the CSV ships with.
    df = pd.read_csv(csv_path, encoding="utf-8", index_col=0)
    print(df.head())

    df = df.dropna(subset=[config.TARGET_COLUMN])
    return df


def summarize_dataset(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing values per column:\n", df.isna().sum())

    genres = df[config.TARGET_COLUMN]
    print(f"\n{config.TARGET_COLUMN}: {genres.nunique()} unique values")
    print(f"\nClass-size distribution:\n{genres.value_counts().describe()}")


def preprocess(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    tracks = df.drop(columns=config.DROP_COLUMNS)

    features_df = tracks.drop(columns=[config.TARGET_COLUMN])
    for col in features_df.columns:
        features_df[col] = pd.to_numeric(features_df[col], errors="coerce")

    mask = features_df.notna().all(axis=1)
    features_df = features_df.loc[mask]
    print("Cleaned shape:", features_df.shape)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(tracks.loc[mask, config.TARGET_COLUMN]).astype(np.int64)

    X = features_df.values.astype(np.float32)
    print("Number of features:", X.shape[1])
    print("Features:", features_df.columns.tolist())
    print("Number of classes:", len(label_encoder.classes_))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.SEED,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    print("X_train:", X_train.shape, "X_test:", X_test.shape)

    return X_train, X_test, y_train, y_test, label_encoder


def make_loaders(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, X_test_t
