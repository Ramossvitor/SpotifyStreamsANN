<!-- Keep this file and .claude/docs/ updated when project structure, conventions, or tooling changes -->

# SpotifyStreamsANN

Study project: predict Spotify `streams` from numeric track features using a basic feed-forward ANN. Built with Python, PyTorch, pandas, numpy, scikit-learn, matplotlib, and kagglehub.

## Conventions

- Pipeline logic lives in the `spotify_ann/` package (one module per concern: `config`, `data`, `model`, `training`, `evaluation`, `plotting`). `spotify_streams_ann.py` is a thin entry point that wires the modules together — keep it short.
- **Pedagogical constraints are intentional, not bugs.** The model deliberately uses only `nn.Linear` + ReLU (no RNN/CNN), and omits dropout, weight decay, batch norm, and early stopping so overfitting is observable. The target `streams` is left raw (no log transform) so MSE reads in raw stream counts. Do not "fix" these without explicit user direction.
- Reproducibility: `torch.manual_seed(42)` and `np.random.seed(42)` are applied via `config.set_seeds()`. Hyperparameters and the seed live in `spotify_ann/config.py` — change them there, not inline.
- Device handling: `config.get_device()` returns `"cuda" if torch.cuda.is_available() else "cpu"` — keep the auto-detection pattern.
- Dataset (`spotify-2023.csv`) is committed in the repo root for study reproducibility. `data.load_dataset()` reads this local copy first and falls back to `kagglehub.dataset_download` only if it's missing — preserve that order.
- Plots are written to `./plots/` (created on run) instead of shown interactively, so the script runs unattended.

## Commands

```bash
pip install -r requirements.txt  # Install runtime deps
python spotify_streams_ann.py    # Run the full pipeline (training + plots)
pip install black isort          # Formatters used by the auto-format hook (already installed)
```

No build, test, or lint scripts — this is a study project.

## Project Structure

- `spotify_streams_ann.py` — entry point; calls into `spotify_ann/` to run the pipeline.
- `spotify_ann/` — package containing the pipeline:
  - `config.py` — seeds, device, paths, hyperparameters.
  - `data.py` — dataset loading, preprocessing, DataLoader construction.
  - `model.py` — `StreamsANN` (Linear + ReLU stack).
  - `training.py` — training loop.
  - `evaluation.py` — test-set metrics and printing.
  - `plotting.py` — loss / MAPE / accuracy curves and predicted-vs-actual scatter, written to disk.
- `requirements.txt` — runtime dependencies.
- `spotify-2023.csv` — [Top Spotify Songs 2023](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023) dataset, committed for reproducibility.
- `plots/` — generated output (loss curves, predicted-vs-actual). Not committed.

## Before Writing Code

ALWAYS read `.claude/docs/coding-guidelines.md` before planning or implementing any changes to code files. All code must follow these principles.

## Documentation

(See the `## Before Writing Code` section above.)
