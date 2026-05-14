<!-- Keep this file and .claude/docs/ updated when project structure, conventions, or tooling changes -->

# SpotifyGenreANN

Study project: classify a Spotify track's `track_genre` (114 genres) from numeric audio features using a basic feed-forward ANN. Built with Python, PyTorch, pandas, numpy, scikit-learn, matplotlib, and kagglehub.

## Conventions

- Pipeline logic lives in the `spotify_ann/` package (one module per concern: `config`, `data`, `model`, `training`, `evaluation`, `plotting`). `spotify_genre_ann.py` is a thin entry point that wires the modules together — keep it short.
- **Pedagogical constraints are intentional, not bugs.** The model deliberately uses only `nn.Linear` + ReLU (no RNN/CNN), and omits dropout, weight decay, batch norm, and early stopping so overfitting is observable. The target `track_genre` is label-encoded; `CrossEntropyLoss` expects raw integer class indices (not one-hot). Do not "fix" these without explicit user direction.
- `artists` is deliberately dropped from the input features (see `config.DROP_COLUMNS`): artist identity is so tightly coupled to genre that including it would let the model memorize artist→genre instead of learning audio-feature→genre. Don't add it back without discussion.
- Reproducibility: `torch.manual_seed(42)` and `np.random.seed(42)` are applied via `config.set_seeds()`. Hyperparameters and the seed live in `spotify_ann/config.py` — change them there, not inline.
- Device handling: `config.get_device()` returns `"cuda" if torch.cuda.is_available() else "cpu"` — keep the auto-detection pattern.
- Dataset is the Kaggle [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (~114k rows, 114 genres, balanced 1000/genre). It is **not** committed to the repo — `data.load_dataset()` reads `dataset.csv` from the project root if present, otherwise falls back to `kagglehub.dataset_download`. Preserve that order.
- Plots are written to `./plots/` (created on run) instead of shown interactively, so the script runs unattended.

## Commands

```bash
pip install -r requirements.txt  # Install runtime deps
python spotify_genre_ann.py      # Run the full pipeline (training + plots)
pip install black isort          # Formatters used by the auto-format hook (already installed)
```

No build, test, or lint scripts — this is a study project.

## Project Structure

- `spotify_genre_ann.py` — entry point; calls into `spotify_ann/` to run the pipeline.
- `spotify_ann/` — package containing the pipeline:
  - `config.py` — seeds, device, paths, hyperparameters.
  - `data.py` — dataset loading, label encoding, preprocessing, DataLoader construction.
  - `model.py` — `GenreANN` (Linear + ReLU stack with a `num_classes` output head).
  - `training.py` — training loop (CrossEntropy + Adam, tracks top-1/top-5 per epoch).
  - `evaluation.py` — test-set metrics (top-1, top-5, macro F1, per-class report).
  - `plotting.py` — loss / accuracy / top-5 curves and confusion-matrix heatmap, written to disk.
- `requirements.txt` — runtime dependencies.
- `plots/` — generated output (loss curves, accuracy curves, top-5 curves, confusion matrix). Not committed.
- `dataset.csv` — Kaggle Spotify Tracks Dataset. **Not committed.** Downloaded automatically via kagglehub on first run.

## Before Writing Code

ALWAYS read `.claude/docs/coding-guidelines.md` before planning or implementing any changes to code files. All code must follow these principles.

## Documentation

(See the `## Before Writing Code` section above.)
