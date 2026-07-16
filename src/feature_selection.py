import pandas as pd

from src.config import FEATURE_COUNTS
from src.evaluation import ExperimentResult, evaluate_classifier
from src.sklearn_pipelines import (
    build_linear_svc_feature_selection_pipeline,
    build_logistic_feature_selection_pipeline,
)
from src.splitting import make_stratified_cv

FEATURE_METHOD = "select_k_best_f_classif"
FEATURE_SELECTION_EXPERIMENTS = (
    ("E02", "LogisticRegression", FEATURE_COUNTS[0]),
    ("E03", "LogisticRegression", FEATURE_COUNTS[1]),
    ("E04", "LogisticRegression", FEATURE_COUNTS[2]),
    ("E05", "LogisticRegression", FEATURE_COUNTS[3]),
    ("E06", "LinearSVC", FEATURE_COUNTS[0]),
    ("E07", "LinearSVC", FEATURE_COUNTS[1]),
    ("E08", "LinearSVC", FEATURE_COUNTS[2]),
    ("E09", "LinearSVC", FEATURE_COUNTS[3]),
)


def run_feature_selection_experiments(
    features_train: pd.DataFrame,
    target_train: pd.Series,
) -> list[ExperimentResult]:
    """Evaluate the eight approved fold-local SelectKBest configurations."""
    cross_validator = make_stratified_cv()
    results: list[ExperimentResult] = []

    for experiment_id, model_name, k in FEATURE_SELECTION_EXPERIMENTS:
        if model_name == "LogisticRegression":
            estimator = build_logistic_feature_selection_pipeline(k)
        else:
            estimator = build_linear_svc_feature_selection_pipeline(k)

        results.append(
            evaluate_classifier(
                estimator,
                features_train,
                target_train,
                cross_validator,
                experiment_id=experiment_id,
                model_name=model_name,
                feature_method=FEATURE_METHOD,
                n_features=k,
            )
        )

    return results


def combine_feature_selection_results(
    existing_results: list[ExperimentResult],
    feature_selection_results: list[ExperimentResult],
) -> list[ExperimentResult]:
    """Preserve E00-E01 and replace E02-E09 in experiment order."""
    existing_by_id = {result["experiment_id"]: result for result in existing_results}
    allowed_existing_ids = {"E00", "E01"} | {
        experiment_id for experiment_id, _, _ in FEATURE_SELECTION_EXPERIMENTS
    }
    if set(existing_by_id) - allowed_existing_ids:
        raise ValueError("Unexpected experiment_id in metrics.csv")

    baseline_results = [existing_by_id[experiment_id] for experiment_id in ("E00", "E01")]
    selection_by_id = {result["experiment_id"]: result for result in feature_selection_results}
    selection_ids = [experiment_id for experiment_id, _, _ in FEATURE_SELECTION_EXPERIMENTS]
    if set(selection_by_id) != set(selection_ids):
        raise ValueError("Feature-selection results must contain E02 through E09")

    return baseline_results + [selection_by_id[experiment_id] for experiment_id in selection_ids]
