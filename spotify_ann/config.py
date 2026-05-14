"""Project-wide constants, paths, and reproducibility helpers."""

from pathlib import Path

import numpy as np
import torch

SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
TEST_SIZE = 0.2
TOP_K = 5

TARGET_COLUMN = "track_genre"
# Identifier / string columns that can't feed a numeric network.
# `artists` is excluded on purpose: artist identity is so tightly coupled to
# genre that including it would let the model memorize artist -> genre instead
# of learning audio-feature -> genre.
DROP_COLUMNS = ["track_id", "artists", "album_name", "track_name"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"
LOCAL_CSV = PROJECT_ROOT / "dataset.csv"
KAGGLE_DATASET = "maharshipandya/-spotify-tracks-dataset"


def set_seeds() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
