"""Project-wide constants, paths, and reproducibility helpers."""

from pathlib import Path

import numpy as np
import torch

SEED = 42
BATCH_SIZE = 256
EPOCHS = 100
# Held fixed despite the larger batch so the architecture is the only moving
# variable; Adam's per-parameter scaling makes it largely batch-insensitive.
LEARNING_RATE = 1e-3
TEST_SIZE = 0.2
TOP_K = 5
HIDDEN_DIMS = [512, 512, 256, 128]

TARGET_COLUMN = "track_genre"
# Identifier / string columns that can't feed a numeric network.
# `artists` is excluded on purpose: artist identity is so tightly coupled to
# genre that including it would let the model memorize artist -> genre instead
# of learning audio-feature -> genre.
DROP_COLUMNS = ["track_id", "artists", "album_name", "track_name"]
# Integer-coded categorical columns. One-hot encoded in data.preprocess so the
# network doesn't read them as ordered magnitudes (e.g., key=11 is musically
# adjacent to key=0 but far apart on the number line).
CATEGORICAL_COLUMNS = ["key", "mode", "time_signature", "explicit"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"
LOCAL_CSV = PROJECT_ROOT / "dataset.csv"
KAGGLE_DATASET = "maharshipandya/-spotify-tracks-dataset"


def set_seeds() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
