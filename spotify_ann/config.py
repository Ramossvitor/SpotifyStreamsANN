"""Project-wide constants, paths, and reproducibility helpers."""

from pathlib import Path

import numpy as np
import torch

SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
TEST_SIZE = 0.2
TOLERANCE = 0.10

TARGET_COLUMN = "streams"
DROP_COLUMNS = ["track_name", "artist(s)_name", "key", "mode"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"
LOCAL_CSV = PROJECT_ROOT / "spotify-2023.csv"
KAGGLE_DATASET = "nelgiriyewithana/top-spotify-songs-2023"


def set_seeds() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
