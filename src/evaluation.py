from time import perf_counter
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import CV_SPLITS, RANDOM_STATE

SCORING = {
    "f1_macro": "f1_macro",
    "accuracy": "accuracy",
    "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
    "recall_macro": "recall_macro",
}


class ExperimentResult(TypedDict):
    """One cross-validation result row saved to the experiment table."""

    experiment_id: str
    model: str
    feature_method: str
    n_features: int
    cv_f1_macro_mean: float
    cv_f1_macro_std: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    cv_precision_macro_mean: float
    cv_precision_macro_std: float
    cv_recall_macro_mean: float
    cv_recall_macro_std: float
    cv_fit_time_mean: float
    cv_score_time_mean: float
    total_cv_time_seconds: float
    random_state: int
    cv_splits: int


def evaluate_classifier(
    estimator: BaseEstimator,
    features_train: pd.DataFrame,
    target_train: pd.Series,
    cross_validator: StratifiedKFold,
    *,
    experiment_id: str,
    model_name: str,
    feature_method: str = "none",
    random_state: int = RANDOM_STATE,
    cv_splits: int = CV_SPLITS,
) -> ExperimentResult:
    """Evaluate one classifier by cross-validation on the training subset only."""
    started_at = perf_counter()
    scores = cross_validate(
        estimator,
        features_train,
        target_train,
        cv=cross_validator,
        scoring=SCORING,
    )
    total_cv_time = perf_counter() - started_at

    f1_scores = np.asarray(scores["test_f1_macro"], dtype=float)
    accuracy_scores = np.asarray(scores["test_accuracy"], dtype=float)
    precision_scores = np.asarray(scores["test_precision_macro"], dtype=float)
    recall_scores = np.asarray(scores["test_recall_macro"], dtype=float)
    fit_times = np.asarray(scores["fit_time"], dtype=float)
    score_times = np.asarray(scores["score_time"], dtype=float)

    return {
        "experiment_id": experiment_id,
        "model": model_name,
        "feature_method": feature_method,
        "n_features": features_train.shape[1],
        "cv_f1_macro_mean": float(f1_scores.mean()),
        "cv_f1_macro_std": float(f1_scores.std()),
        "cv_accuracy_mean": float(accuracy_scores.mean()),
        "cv_accuracy_std": float(accuracy_scores.std()),
        "cv_precision_macro_mean": float(precision_scores.mean()),
        "cv_precision_macro_std": float(precision_scores.std()),
        "cv_recall_macro_mean": float(recall_scores.mean()),
        "cv_recall_macro_std": float(recall_scores.std()),
        "cv_fit_time_mean": float(fit_times.mean()),
        "cv_score_time_mean": float(score_times.mean()),
        "total_cv_time_seconds": total_cv_time,
        "random_state": random_state,
        "cv_splits": cv_splits,
    }
