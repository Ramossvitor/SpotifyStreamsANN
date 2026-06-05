"""Classifying Spotify track genres with a feed-forward ANN.

Dataset: Spotify Tracks Dataset
    https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

Goal: a fully-connected feed-forward network in PyTorch that predicts the
`track_genre` column (114 genres) from numeric audio features and generalizes
to unseen tracks.

Design notes:
- Architecture is nn.Linear + ReLU (no RNN / CNN).
- The optimizer is AdamW, whose decoupled weight decay regularizes the model
  to curb overfitting and improve test-set generalization.
- 114-way classification with no class weighting. The dataset is perfectly
  balanced (1000 tracks per genre), so plain CrossEntropyLoss is enough.

Run:
    pip install -r requirements.txt
    python spotify_genre_ann.py

Plots are saved to ./plots/.
"""

from spotify_ann import config, data, evaluation, plotting, training
from spotify_ann.model import GenreANN


def main() -> None:
    config.set_seeds()
    device = config.get_device()
    print("Using device:", device)
    config.PLOTS_DIR.mkdir(exist_ok=True)

    df = data.load_dataset()
    data.summarize_dataset(df)
    X_train, X_test, y_train, y_test, label_encoder = data.preprocess(df)
    train_loader, test_loader, X_test_t = data.make_loaders(
        X_train, X_test, y_train, y_test
    )

    num_classes = len(label_encoder.classes_)
    model = GenreANN(input_dim=X_train.shape[1], num_classes=num_classes).to(device)
    print(model)

    history = training.train_model(
        model,
        train_loader,
        test_loader,
        device,
        epochs=config.EPOCHS,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    predictions, metrics = evaluation.evaluate_model(
        model, X_test_t, y_test, device, class_names=label_encoder.classes_
    )
    evaluation.print_metrics(metrics, n_test=len(y_test))

    plotting.plot_curves(
        history["train_loss"],
        history["test_loss"],
        config.PLOTS_DIR / "loss_curves.png",
        ylabel="Cross-entropy loss",
        title="Training vs test loss",
        train_label="train loss",
        test_label="test loss",
    )
    plotting.plot_curves(
        history["train_top1"],
        history["test_top1"],
        config.PLOTS_DIR / "accuracy_curves.png",
        ylabel="Top-1 accuracy (%)",
        title="Training vs test top-1 accuracy",
        train_label="train top-1",
        test_label="test top-1",
        ylim=(0, 100),
    )
    plotting.plot_curves(
        history["train_top5"],
        history["test_top5"],
        config.PLOTS_DIR / "top5_curves.png",
        ylabel="Top-5 accuracy (%)",
        title="Training vs test top-5 accuracy",
        train_label="train top-5",
        test_label="test top-5",
        ylim=(0, 100),
    )
    plotting.plot_confusion_matrix(
        y_test,
        predictions,
        label_encoder.classes_,
        config.PLOTS_DIR / "confusion_matrix.png",
    )
    print(f"\nPlots saved to {config.PLOTS_DIR}")


if __name__ == "__main__":
    main()
