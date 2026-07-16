import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.evaluation import ExperimentResult, evaluate_classifier
from src.experiments import save_experiment_metrics
from src.feature_selection import (
    FEATURE_METHOD,
    FEATURE_SELECTION_EXPERIMENTS,
    combine_feature_selection_results,
)
from src.sklearn_pipelines import (
    build_linear_svc_feature_selection_pipeline,
    build_logistic_feature_selection_pipeline,
)
from src.splitting import make_stratified_cv
from src.visualization import save_feature_selection_comparison


def _result(experiment_id: str, model: str, n_features: int) -> ExperimentResult:
    return {
        "experiment_id": experiment_id,
        "model": model,
        "feature_method": FEATURE_METHOD if experiment_id not in {"E00", "E01"} else "none",
        "n_features": n_features,
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


def _baseline_results() -> list[ExperimentResult]:
    return [
        _result("E00", "DummyClassifier", 20531),
        _result("E01", "LogisticRegression", 20531),
    ]


def _selection_results() -> list[ExperimentResult]:
    return [
        _result(experiment_id, model_name, k)
        for experiment_id, model_name, k in FEATURE_SELECTION_EXPERIMENTS
    ]


def _dataset() -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame(
        np.arange(120, dtype=float).reshape(20, 6),
        columns=[f"gene_{index}" for index in range(6)],
    )
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


def test_feature_selection_pipeline_step_order() -> None:
    pipeline = build_logistic_feature_selection_pipeline(50)

    assert list(pipeline.named_steps) == ["selector", "scaler", "classifier"]


def test_feature_selection_pipeline_uses_select_k_best() -> None:
    pipeline = build_logistic_feature_selection_pipeline(50)

    assert isinstance(pipeline.named_steps["selector"], SelectKBest)


def test_feature_selection_pipeline_uses_f_classif() -> None:
    pipeline = build_logistic_feature_selection_pipeline(50)

    assert pipeline.named_steps["selector"].score_func is f_classif


def test_feature_selection_pipeline_receives_k() -> None:
    pipeline = build_logistic_feature_selection_pipeline(200)

    assert pipeline.named_steps["selector"].k == 200


def test_logistic_feature_selection_configuration() -> None:
    pipeline = build_logistic_feature_selection_pipeline(100)
    classifier = pipeline.named_steps["classifier"]

    assert isinstance(classifier, LogisticRegression)
    assert classifier.max_iter == 3000
    assert classifier.random_state == 42


def test_linear_svc_feature_selection_configuration() -> None:
    pipeline = build_linear_svc_feature_selection_pipeline(100)
    classifier = pipeline.named_steps["classifier"]

    assert isinstance(classifier, LinearSVC)
    assert classifier.C == 1.0
    assert classifier.dual == "auto"
    assert classifier.max_iter == 5000
    assert classifier.random_state == 42


def test_selector_is_inside_estimator_passed_to_cross_validation(
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
        selector = estimator.named_steps["selector"]
        assert isinstance(selector, SelectKBest)
        assert not hasattr(selector, "scores_")
        return _scores()

    monkeypatch.setattr("src.evaluation.cross_validate", fake_cross_validate)

    evaluate_classifier(
        build_logistic_feature_selection_pipeline(5),
        features,
        target,
        make_stratified_cv(n_splits=2),
        experiment_id="test",
        model_name="LogisticRegression",
        n_features=5,
    )

    assert isinstance(observed_estimator, Pipeline)


def test_feature_selection_matrix_contains_e02_through_e09() -> None:
    experiment_ids = [item[0] for item in FEATURE_SELECTION_EXPERIMENTS]

    assert experiment_ids == [f"E{number:02d}" for number in range(2, 10)]


def test_feature_selection_matrix_has_one_id_per_model_and_k() -> None:
    pairs = [(model_name, k) for _, model_name, k in FEATURE_SELECTION_EXPERIMENTS]
    expected_pairs = {
        (model_name, k)
        for model_name in ("LogisticRegression", "LinearSVC")
        for k in (50, 100, 200, 500)
    }

    assert len(pairs) == len(set(pairs))
    assert set(pairs) == expected_pairs


def test_combining_results_preserves_baselines() -> None:
    baselines = _baseline_results()

    combined = combine_feature_selection_results(baselines, _selection_results())

    assert combined[:2] == baselines


def test_repeated_combining_does_not_duplicate_experiments() -> None:
    selection_results = _selection_results()
    first_combination = combine_feature_selection_results(_baseline_results(), selection_results)

    second_combination = combine_feature_selection_results(first_combination, selection_results)
    experiment_ids = [result["experiment_id"] for result in second_combination]

    assert len(experiment_ids) == 10
    assert len(experiment_ids) == len(set(experiment_ids))


def test_feature_selection_metrics_have_no_test_columns(tmp_path: Path) -> None:
    output_path = tmp_path / "metrics.csv"
    save_experiment_metrics(
        combine_feature_selection_results(_baseline_results(), _selection_results()),
        output_path,
    )

    with output_path.open(encoding="utf-8", newline="") as metrics_file:
        fieldnames = csv.DictReader(metrics_file).fieldnames

    assert fieldnames is not None
    assert not any(column.startswith("test_") for column in fieldnames)


def test_feature_selection_comparison_is_png(tmp_path: Path) -> None:
    output_path = tmp_path / "feature_selection_comparison.png"

    result = save_feature_selection_comparison(_selection_results(), output_path)

    assert result == output_path
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_evaluation_uses_fold_count_from_cross_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, target = _dataset()

    def fake_cross_validate(
        estimator: object,
        evaluated_features: pd.DataFrame,
        evaluated_target: pd.Series,
        *,
        cv: object,
        scoring: object,
    ) -> dict[str, np.ndarray]:
        return _scores()

    monkeypatch.setattr("src.evaluation.cross_validate", fake_cross_validate)
    result = evaluate_classifier(
        build_logistic_feature_selection_pipeline(5),
        features,
        target,
        make_stratified_cv(n_splits=2),
        experiment_id="test",
        model_name="LogisticRegression",
        n_features=5,
    )

    assert result["cv_splits"] == 2
