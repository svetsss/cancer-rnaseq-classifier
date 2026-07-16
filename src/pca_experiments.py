import pandas as pd

from src.config import PCA_COMPONENTS
from src.evaluation import ExperimentResult, evaluate_classifier
from src.sklearn_pipelines import build_linear_svc_pca_pipeline, build_logistic_pca_pipeline
from src.splitting import make_stratified_cv

PCA_EXPERIMENTS = (
    ("E10", "LogisticRegression", PCA_COMPONENTS[0]),
    ("E11", "LogisticRegression", PCA_COMPONENTS[1]),
    ("E12", "LogisticRegression", PCA_COMPONENTS[2]),
    ("E13", "LinearSVC", PCA_COMPONENTS[0]),
    ("E14", "LinearSVC", PCA_COMPONENTS[1]),
    ("E15", "LinearSVC", PCA_COMPONENTS[2]),
)


def run_pca_experiments(
    features_train: pd.DataFrame,
    target_train: pd.Series,
) -> list[ExperimentResult]:
    """Evaluate the six approved fold-local PCA configurations."""
    cross_validator = make_stratified_cv()
    results: list[ExperimentResult] = []

    for experiment_id, model_name, n_components in PCA_EXPERIMENTS:
        if model_name == "LogisticRegression":
            estimator = build_logistic_pca_pipeline(n_components)
        else:
            estimator = build_linear_svc_pca_pipeline(n_components)

        results.append(
            evaluate_classifier(
                estimator,
                features_train,
                target_train,
                cross_validator,
                experiment_id=experiment_id,
                model_name=model_name,
                feature_method="pca",
                n_features=n_components,
            )
        )

    return results
