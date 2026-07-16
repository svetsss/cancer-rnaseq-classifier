from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from src.config import RANDOM_STATE
from src.evaluation import SCORING
from src.sklearn_pipelines import build_logistic_pca_pipeline


class RobustnessSummary(TypedDict):
    experiment_id: str
    evaluation_scope: str
    cv_type: str
    n_splits: int
    n_repeats: int
    random_state: int
    evaluations: int
    f1_macro_mean: float
    f1_macro_std: float
    f1_macro_min: float
    f1_macro_max: float
    accuracy_mean: float
    accuracy_std: float
    accuracy_min: float
    accuracy_max: float
    fold_f1_macro: list[float]
    fold_accuracy: list[float]


def evaluate_e10_repeated_cv(
    features_train: pd.DataFrame,
    target_train: pd.Series,
    *,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
) -> RobustnessSummary:
    """Estimate E10 variability over repeated train-only stratified folds."""
    cross_validator = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    scores = cross_validate(
        build_logistic_pca_pipeline(20, random_state=random_state),
        features_train,
        target_train,
        cv=cross_validator,
        scoring=SCORING,
    )
    f1_scores = np.asarray(scores["test_f1_macro"], dtype=float)
    accuracy_scores = np.asarray(scores["test_accuracy"], dtype=float)
    return {
        "experiment_id": "E10-robustness",
        "evaluation_scope": "training subset only; final test set is not evaluated",
        "cv_type": "RepeatedStratifiedKFold",
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "random_state": random_state,
        "evaluations": len(f1_scores),
        "f1_macro_mean": float(f1_scores.mean()),
        "f1_macro_std": float(f1_scores.std()),
        "f1_macro_min": float(f1_scores.min()),
        "f1_macro_max": float(f1_scores.max()),
        "accuracy_mean": float(accuracy_scores.mean()),
        "accuracy_std": float(accuracy_scores.std()),
        "accuracy_min": float(accuracy_scores.min()),
        "accuracy_max": float(accuracy_scores.max()),
        "fold_f1_macro": f1_scores.tolist(),
        "fold_accuracy": accuracy_scores.tolist(),
    }
