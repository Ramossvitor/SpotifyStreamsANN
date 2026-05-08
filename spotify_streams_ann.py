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
    pip install -r requirements.txt
    python spotify_streams_ann.py

Plots are saved to ./plots/.
"""

from spotify_ann import config, data, evaluation, plotting, training
from spotify_ann.model import StreamsANN


def main() -> None:
    config.set_seeds()
    device = config.get_device()
    print("Using device:", device)
    config.PLOTS_DIR.mkdir(exist_ok=True)

    df = data.load_dataset()
    data.summarize_dataset(df)
    X_train, X_test, y_train, y_test = data.preprocess(df)
    train_loader, test_loader, X_test_t = data.make_loaders(
        X_train, X_test, y_train, y_test
    )

    model = StreamsANN(input_dim=X_train.shape[1]).to(device)
    print(model)

    history = training.train_model(
        model,
        train_loader,
        test_loader,
        device,
        epochs=config.EPOCHS,
        learning_rate=config.LEARNING_RATE,
        tolerance=config.TOLERANCE,
    )

    predictions, metrics = evaluation.evaluate_model(
        model, X_test_t, y_test, device, tolerance=config.TOLERANCE
    )
    evaluation.print_metrics(metrics, config.TOLERANCE, n_test=len(y_test))

    plotting.plot_curves(
        history["train_mse"],
        history["test_mse"],
        config.PLOTS_DIR / "loss_curves.png",
        ylabel="MSE (raw streams)",
        title="Training vs test loss",
        train_label="train MSE",
        test_label="test MSE",
        yscale="log",
    )
    plotting.plot_curves(
        history["train_mape"],
        history["test_mape"],
        config.PLOTS_DIR / "mape_curves.png",
        ylabel="MAPE (%)",
        title="Training vs test MAPE",
        train_label="train MAPE",
        test_label="test MAPE",
    )
    plotting.plot_curves(
        history["train_accuracy"],
        history["test_accuracy"],
        config.PLOTS_DIR / "accuracy_curves.png",
        ylabel="Accuracy within ±tol (%)",
        title="Training vs test accuracy",
        train_label="train accuracy",
        test_label="test accuracy",
        ylim=(0, 100),
    )
    plotting.plot_predicted_vs_actual(
        y_test, predictions, config.PLOTS_DIR / "predicted_vs_actual.png"
    )
    print(f"\nPlots saved to {config.PLOTS_DIR}")


if __name__ == "__main__":
    main()
