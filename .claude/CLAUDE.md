<!-- Keep this file and .claude/docs/ updated when project structure, conventions, or tooling changes -->

# SpotifyStreamsANN

Study project: predict Spotify `streams` from numeric track features using a basic feed-forward ANN. Built with Python, PyTorch, pandas, numpy, scikit-learn, matplotlib, and kagglehub.

## Conventions

- Single-script project — all data loading, preprocessing, model, training, and evaluation live in `spotify_streams_ann.py`. There is no `src/` package.
- **Pedagogical constraints are intentional, not bugs.** The script deliberately uses only `nn.Linear` + ReLU (no RNN/CNN), and omits dropout, weight decay, batch norm, and early stopping so overfitting is observable. The target `streams` is left raw (no log transform) so MSE reads in raw stream counts. Do not "fix" these without explicit user direction.
- Reproducibility: `torch.manual_seed(42)` and `np.random.seed(42)` are set near the top of the script. Preserve them when editing.
- Device handling: `device = "cuda" if torch.cuda.is_available() else "cpu"` — keep the auto-detection pattern.
- Dataset (`spotify-2023.csv`) is committed in the repo root for study reproducibility. The script reads this local copy first and falls back to `kagglehub.dataset_download` only if it's missing — preserve that order.
- Plots are written to `./plots/` (created on run) instead of shown interactively, so the script runs unattended.

## Commands

```bash
python spotify_streams_ann.py    # Run the full pipeline (training + plots)
pip install black isort          # Formatters used by the auto-format hook (already installed)
```

No build, test, or lint scripts — this is a study project.

## Project Structure

- `spotify_streams_ann.py` — main script (sections 1-10: setup, load, EDA, preprocessing, tensors, model, optimizer, training, evaluation, plots).
- `spotify-2023.csv` — [Top Spotify Songs 2023](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023) dataset, committed for reproducibility.
- `plots/` — generated output (loss curves, predicted-vs-actual). Not committed.

## Before Writing Code

ALWAYS read `.claude/docs/coding-guidelines.md` before planning or implementing any changes to code files. All code must follow these principles.

## Documentation

(See the `## Before Writing Code` section above.)
