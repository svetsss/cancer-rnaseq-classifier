from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import ExperimentResult, evaluate_classifier
from src.pca_experiments import PCA_EXPERIMENTS
from src.sklearn_pipelines import build_logistic_pca_pipeline
from src.splitting import make_stratified_cv
from src.visualization import save_pca_comparison, save_pca_train_projection


def _result(experiment_id: str, model: str, components: int) -> ExperimentResult:
    return {
        "experiment_id": experiment_id,
        "model": model,
        "feature_method": "pca",
        "n_features": components,
        "cv_f1_macro_mean": 0.9,
        "cv_f1_macro_std": 0.01,
        "cv_accuracy_mean": 0.9,
        "cv_accuracy_std": 0.01,
        "cv_precision_macro_mean": 0.9,
        "cv_precision_macro_std": 0.01,
        "cv_recall_macro_mean": 0.9,
        "cv_recall_macro_std": 0.01,
        "cv_fit_time_mean": 0.1,
        "cv_score_time_mean": 0.01,
        "total_cv_time_seconds": 0.5,
        "random_state": 42,
        "cv_splits": 5,
    }


def _dataset() -> tuple[pd.DataFrame, pd.Series]:
    random = np.random.default_rng(42)
    features = pd.DataFrame(random.normal(size=(20, 6)))
    target = pd.Series(["A", "B"] * 10, dtype="string")
    return features, target


def _scores() -> dict[str, np.ndarray]:
    return {
        "fit_time": np.array([0.1, 0.1]),
        "score_time": np.array([0.01, 0.01]),
        "test_f1_macro": np.array([0.9, 0.9]),
        "test_accuracy": np.array([0.9, 0.9]),
        "test_precision_macro": np.array([0.9, 0.9]),
        "test_recall_macro": np.array([0.9, 0.9]),
    }


def test_pca_pipeline_step_order() -> None:
    pipeline = build_logistic_pca_pipeline(20)

    assert list(pipeline.named_steps) == ["scaler", "pca", "classifier"]


def test_pca_is_inside_estimator_passed_to_cross_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, target = _dataset()
    observed_estimator: Pipeline | None = None

    def fake_cross_validate(
        estimator: Pipeline,
        evaluated_features: pd.DataFrame,
        evaluated_target: pd.Series,
        *,
        cv: object,
        scoring: object,
    ) -> dict[str, np.ndarray]:
        nonlocal observed_estimator
        observed_estimator = estimator
        assert isinstance(estimator.named_steps["pca"], PCA)
        assert not hasattr(estimator.named_steps["pca"], "components_")
        return _scores()

    monkeypatch.setattr("src.evaluation.cross_validate", fake_cross_validate)
    evaluate_classifier(
        build_logistic_pca_pipeline(2),
        features,
        target,
        make_stratified_cv(n_splits=2),
        experiment_id="test",
        model_name="LogisticRegression",
        n_features=2,
    )

    assert isinstance(observed_estimator, Pipeline)


def test_pca_pipeline_receives_component_count() -> None:
    pipeline = build_logistic_pca_pipeline(50)

    assert pipeline.named_steps["pca"].n_components == 50


def test_pca_pipeline_uses_randomized_solver() -> None:
    pipeline = build_logistic_pca_pipeline(20)

    assert pipeline.named_steps["pca"].svd_solver == "randomized"


def test_pca_pipeline_uses_random_state() -> None:
    pipeline = build_logistic_pca_pipeline(20)

    assert pipeline.named_steps["pca"].random_state == 42


def test_pca_experiment_matrix() -> None:
    assert PCA_EXPERIMENTS == (
        ("E10", "LogisticRegression", 20),
        ("E11", "LogisticRegression", 50),
        ("E12", "LogisticRegression", 100),
        ("E13", "LinearSVC", 20),
        ("E14", "LinearSVC", 50),
        ("E15", "LinearSVC", 100),
    )


def test_pca_comparison_is_png(tmp_path: Path) -> None:
    results = [_result(*experiment) for experiment in PCA_EXPERIMENTS]
    output_path = tmp_path / "pca_comparison.png"

    result = save_pca_comparison(results, output_path)

    assert result == output_path
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_pca_projection_fits_only_supplied_training_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features_train, target_train = _dataset()
    observed_indices: pd.Index | None = None
    original_fit_transform = StandardScaler.fit_transform

    def tracking_fit_transform(
        scaler: StandardScaler,
        features: pd.DataFrame,
        target: object = None,
        **fit_params: object,
    ) -> np.ndarray:
        nonlocal observed_indices
        observed_indices = features.index
        return original_fit_transform(scaler, features, target, **fit_params)

    monkeypatch.setattr(
        "src.visualization.StandardScaler.fit_transform",
        tracking_fit_transform,
    )
    save_pca_train_projection(
        features_train,
        target_train,
        tmp_path / "pca_train_projection.png",
    )

    assert observed_indices is not None
    assert observed_indices.equals(features_train.index)
