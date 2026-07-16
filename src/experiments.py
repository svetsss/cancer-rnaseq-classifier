import csv
from pathlib import Path
from typing import cast

import pandas as pd

from src.config import METRICS_PATH
from src.evaluation import ExperimentResult, evaluate_classifier
from src.sklearn_pipelines import build_dummy_classifier, build_logistic_pipeline
from src.splitting import make_stratified_cv

METRICS_COLUMNS = (
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
)


def run_baseline_experiments(
    features_train: pd.DataFrame,
    target_train: pd.Series,
) -> list[ExperimentResult]:
    """Run the two approved baselines on identical training folds."""
    cross_validator = make_stratified_cv()
    dummy_result = evaluate_classifier(
        build_dummy_classifier(),
        features_train,
        target_train,
        cross_validator,
        experiment_id="E00",
        model_name="DummyClassifier",
    )
    logistic_result = evaluate_classifier(
        build_logistic_pipeline(),
        features_train,
        target_train,
        cross_validator,
        experiment_id="E01",
        model_name="LogisticRegression",
    )
    return [dummy_result, logistic_result]


def load_experiment_metrics(
    input_path: Path = METRICS_PATH,
) -> list[ExperimentResult]:
    """Load the saved experiment table and require its current schema and unique IDs."""
    with input_path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if tuple(reader.fieldnames or ()) != METRICS_COLUMNS:
            raise ValueError("Unexpected metrics.csv columns")
        results = [_parse_experiment_result(row) for row in reader]

    experiment_ids = [result["experiment_id"] for result in results]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("Duplicate experiment_id in metrics.csv")
    return results


def save_experiment_metrics(
    results: list[ExperimentResult],
    output_path: Path = METRICS_PATH,
) -> Path:
    """Atomically replace the experiment metrics table."""
    experiment_ids = [result["experiment_id"] for result in results]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("Duplicate experiment_id in experiment results")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.csv")
    with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=METRICS_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(results)
    temporary_path.replace(output_path)
    return output_path


def _parse_experiment_result(row: dict[str, str | None]) -> ExperimentResult:
    return {
        "experiment_id": cast(str, row["experiment_id"]),
        "model": cast(str, row["model"]),
        "feature_method": cast(str, row["feature_method"]),
        "n_features": int(cast(str, row["n_features"])),
        "cv_f1_macro_mean": float(cast(str, row["cv_f1_macro_mean"])),
        "cv_f1_macro_std": float(cast(str, row["cv_f1_macro_std"])),
        "cv_accuracy_mean": float(cast(str, row["cv_accuracy_mean"])),
        "cv_accuracy_std": float(cast(str, row["cv_accuracy_std"])),
        "cv_precision_macro_mean": float(cast(str, row["cv_precision_macro_mean"])),
        "cv_precision_macro_std": float(cast(str, row["cv_precision_macro_std"])),
        "cv_recall_macro_mean": float(cast(str, row["cv_recall_macro_mean"])),
        "cv_recall_macro_std": float(cast(str, row["cv_recall_macro_std"])),
        "cv_fit_time_mean": float(cast(str, row["cv_fit_time_mean"])),
        "cv_score_time_mean": float(cast(str, row["cv_score_time_mean"])),
        "total_cv_time_seconds": float(cast(str, row["total_cv_time_seconds"])),
        "random_state": int(cast(str, row["random_state"])),
        "cv_splits": int(cast(str, row["cv_splits"])),
    }
