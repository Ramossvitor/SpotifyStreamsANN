<!-- Keep this file and .claude/docs/ updated when project structure, conventions, or tooling changes -->

# SpotifyStreamsANN

Study project: predict Spotify `streams` from numeric track features using a basic feed-forward ANN. Built with Python, PyTorch, pandas, numpy, scikit-learn, matplotlib, and kagglehub (Jupyter notebook).

## Conventions

- Single-notebook project — all data loading, preprocessing, model, training, and evaluation live in `spotify_streams_ann.ipynb`. There is no `src/` package.
- **Pedagogical constraints are intentional, not bugs.** The notebook deliberately uses only `nn.Linear` + ReLU (no RNN/CNN), and omits dropout, weight decay, batch norm, and early stopping so overfitting is observable. The target `streams` is left raw (no log transform) so MSE reads in raw stream counts. Do not "fix" these without explicit user direction.
- Reproducibility: `torch.manual_seed(42)` and `np.random.seed(42)` are set near the top of the notebook. Preserve them when editing.
- Device handling: `device = "cuda" if torch.cuda.is_available() else "cpu"` — keep the auto-detection pattern.
- Dataset (`spotify-2023.csv`) is committed in the repo root for study reproducibility. Do not move it without updating the `kagglehub.dataset_download` fallback path in the notebook.

## Commands

```bash
jupyter notebook spotify_streams_ann.ipynb   # Open the notebook
pip install black isort                      # Formatters used by the auto-format hook (already installed)
```

No build, test, or lint scripts — this is a study project.

## Project Structure

- `spotify_streams_ann.ipynb` — main notebook (sections 1-10: install, imports, EDA, preprocessing, model, training, evaluation, plots).
- `spotify-2023.csv` — [Top Spotify Songs 2023](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023) dataset, committed for reproducibility.

## Before Writing Code

ALWAYS read `.claude/docs/coding-guidelines.md` before planning or implementing any changes to code files. All code must follow these principles.

## Documentation

(See the `## Before Writing Code` section above.)
