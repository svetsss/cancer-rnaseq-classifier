from pathlib import Path

import pandas as pd

from src.neural_evaluation import MLPSummary
from src.visualization import (
    save_class_distribution,
    save_mlp_epoch_selection,
    save_mlp_fold_metrics,
    save_mlp_loss_curves,
    save_mlp_validation_f1,
    save_readme_model_comparison,
    save_robustness_f1,
)


def test_save_class_distribution_creates_png(tmp_path: Path) -> None:
    target = pd.Series(["BRCA", "KIRC", "BRCA"], dtype="string", name="cancer_type")
    output_path = tmp_path / "class_distribution.png"

    result = save_class_distribution(target, output_path)

    assert result == output_path
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_save_mlp_training_figures_create_separate_png_files(tmp_path: Path) -> None:
    history = [
        {
            "epoch": epoch,
            "train_loss": 1.0 / epoch,
            "validation_loss": 1.2 / epoch,
            "validation_f1_macro": 0.7 + epoch / 100,
            "validation_accuracy": 0.72 + epoch / 100,
        }
        for epoch in range(1, 4)
    ]
    fold_metric = {
        "fold": 1,
        "selected_epochs": 2,
        "epochs_run": 3,
        "f1_macro": 0.95,
        "accuracy": 0.96,
        "precision_macro": 0.95,
        "recall_macro": 0.95,
    }
    summary: MLPSummary = {
        "experiment_id": "E17",
        "architecture": ["Linear(4, 2)"],
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "batch_size": 8,
        "max_epochs": 3,
        "patience": 1,
        "inner_validation_size": 0.15,
        "fold_epochs": [2, 2],
        "early_stopping_epochs_run": [3, 3],
        "fold_histories": [history, history],
        "fold_metrics": [fold_metric, {**fold_metric, "fold": 2}],
        "mean_epochs": 2.0,
        "refit_on_full_outer_train": True,
        "device": "cpu",
        "random_state": 42,
    }
    functions = (
        save_mlp_loss_curves,
        save_mlp_validation_f1,
        save_mlp_epoch_selection,
        save_mlp_fold_metrics,
    )

    for function in functions:
        output_path = tmp_path / f"{function.__name__}.png"
        result = function(summary, output_path)
        assert result == output_path
        assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_save_readme_model_comparison_creates_png(tmp_path: Path) -> None:
    metrics = pd.DataFrame(
        {
            "experiment_id": ["E01", "E08", "E10", "E16", "E17"],
            "cv_f1_macro_mean": [0.995, 1.0, 1.0, 0.996, 0.999],
            "cv_f1_macro_std": [0.006, 0.0, 0.0, 0.003, 0.002],
        }
    )
    output_path = tmp_path / "readme_model_comparison.png"

    result = save_readme_model_comparison(metrics, output_path)

    assert result == output_path
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_save_robustness_f1_creates_png(tmp_path: Path) -> None:
    summary = {
        "experiment_id": "E10-robustness",
        "evaluation_scope": "training subset only; final test set is not evaluated",
        "cv_type": "RepeatedStratifiedKFold",
        "n_splits": 2,
        "n_repeats": 2,
        "random_state": 42,
        "evaluations": 4,
        "f1_macro_mean": 0.985,
        "f1_macro_std": 0.01,
        "f1_macro_min": 0.97,
        "f1_macro_max": 1.0,
        "accuracy_mean": 0.985,
        "accuracy_std": 0.01,
        "accuracy_min": 0.97,
        "accuracy_max": 1.0,
        "fold_f1_macro": [0.97, 0.98, 0.99, 1.0],
        "fold_accuracy": [0.97, 0.98, 0.99, 1.0],
    }
    output_path = tmp_path / "robustness.png"

    result = save_robustness_f1(summary, output_path)

    assert result == output_path
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
