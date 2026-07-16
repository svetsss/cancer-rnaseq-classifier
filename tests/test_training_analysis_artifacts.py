import json
from pathlib import Path

import pandas as pd
import pytest

SUMMARY_PATH = Path("results/mlp_training_analysis.json")
FOLD_METRICS_PATH = Path("results/mlp_fold_metrics.csv")
FIGURE_PATHS = (
    Path("figures/readme_model_comparison.png"),
    Path("figures/mlp_loss_curves.png"),
    Path("figures/mlp_validation_f1.png"),
    Path("figures/mlp_epoch_selection.png"),
    Path("figures/mlp_fold_metrics.png"),
)


def test_mlp_training_analysis_artifacts_are_complete() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    fold_metrics = pd.read_csv(FOLD_METRICS_PATH)

    assert len(summary["fold_histories"]) == 5
    assert len(summary["fold_metrics"]) == 5
    assert len(fold_metrics) == 5
    assert [len(history) for history in summary["fold_histories"]] == summary[
        "early_stopping_epochs_run"
    ]
    assert fold_metrics["fold"].tolist() == [1, 2, 3, 4, 5]


def test_canonical_e17_metrics_match_corrected_fold_metrics() -> None:
    metrics = pd.read_csv("results/metrics.csv").set_index("experiment_id")
    fold_metrics = pd.read_csv(FOLD_METRICS_PATH)
    e17 = metrics.loc["E17"]

    for metric_name, fold_column in (
        ("f1_macro", "f1_macro"),
        ("accuracy", "accuracy"),
        ("precision_macro", "precision_macro"),
        ("recall_macro", "recall_macro"),
    ):
        assert e17[f"cv_{metric_name}_mean"] == pytest.approx(fold_metrics[fold_column].mean())
        assert e17[f"cv_{metric_name}_std"] == pytest.approx(fold_metrics[fold_column].std(ddof=0))


def test_mlp_training_figures_are_separate_png_files() -> None:
    for figure_path in FIGURE_PATHS:
        assert figure_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
