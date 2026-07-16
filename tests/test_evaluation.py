import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation import evaluate_classifier
from src.experiments import save_experiment_metrics
from src.sklearn_pipelines import build_dummy_classifier
from src.splitting import make_stratified_cv, split_dataset


def _binary_dataset() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.Index([f"sample_{number}" for number in range(20)], name="sample_id")
    features = pd.DataFrame(
        {"gene_0": np.arange(20, dtype=float), "gene_1": np.arange(20, 40, dtype=float)},
        index=index,
    )
    target = pd.Series(
        ["A", "B"] * 10,
        index=index,
        dtype="string",
        name="cancer_type",
    )
    return features, target


def test_cross_validation_receives_only_training_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, target = _binary_dataset()
    features_train, features_test, target_train, _ = split_dataset(features, target)
    observed_indices: set[str] = set()

    def fake_cross_validate(
        estimator: object,
        evaluated_features: pd.DataFrame,
        evaluated_target: pd.Series,
        *,
        cv: object,
        scoring: object,
    ) -> dict[str, np.ndarray]:
        observed_indices.update(map(str, evaluated_features.index))
        assert evaluated_features.index.equals(evaluated_target.index)
        return {
            "fit_time": np.array([0.1, 0.1]),
            "score_time": np.array([0.01, 0.01]),
            "test_f1_macro": np.array([0.5, 0.5]),
            "test_accuracy": np.array([0.5, 0.5]),
            "test_precision_macro": np.array([0.5, 0.5]),
            "test_recall_macro": np.array([0.5, 0.5]),
        }

    monkeypatch.setattr("src.evaluation.cross_validate", fake_cross_validate)

    evaluate_classifier(
        build_dummy_classifier(),
        features_train,
        target_train,
        make_stratified_cv(n_splits=2),
        experiment_id="test",
        model_name="DummyClassifier",
    )

    assert observed_indices == set(map(str, features_train.index))
    assert observed_indices.isdisjoint(map(str, features_test.index))


def test_macro_f1_is_calculated_from_cross_validation() -> None:
    features, target = _binary_dataset()

    result = evaluate_classifier(
        build_dummy_classifier(),
        features,
        target,
        make_stratified_cv(n_splits=2),
        experiment_id="test",
        model_name="DummyClassifier",
    )

    assert result["cv_f1_macro_mean"] == pytest.approx(1 / 3)
    assert result["cv_f1_macro_std"] == pytest.approx(0.0)


def test_saved_metrics_contain_required_columns(tmp_path: Path) -> None:
    features, target = _binary_dataset()
    result = evaluate_classifier(
        build_dummy_classifier(),
        features,
        target,
        make_stratified_cv(n_splits=2),
        experiment_id="test",
        model_name="DummyClassifier",
    )
    output_path = tmp_path / "metrics.csv"

    save_experiment_metrics([result], output_path)

    with output_path.open(encoding="utf-8", newline="") as metrics_file:
        fieldnames = csv.DictReader(metrics_file).fieldnames
    required_columns = {
        "experiment_id",
        "model",
        "feature_method",
        "n_features",
        "cv_f1_macro_mean",
        "cv_f1_macro_std",
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "cv_precision_macro_mean",
        "cv_precision_macro_std",
        "cv_recall_macro_mean",
        "cv_recall_macro_std",
        "cv_fit_time_mean",
        "cv_score_time_mean",
        "total_cv_time_seconds",
        "random_state",
        "cv_splits",
    }
    assert fieldnames is not None
    assert required_columns == set(fieldnames)
    assert not any(column.startswith("test_") for column in fieldnames)
