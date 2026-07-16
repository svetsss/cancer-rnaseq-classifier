import logging
from math import isclose

from src.config import METRICS_PATH, RUNS_DIR
from src.data_loader import download_dataset, load_dataset
from src.evaluation import ExperimentResult
from src.experiments import (
    combine_extended_results,
    load_experiment_metrics,
    save_experiment_metrics,
)
from src.neural_evaluation import evaluate_mlp_cv, save_mlp_summary
from src.pca_experiments import run_pca_experiments
from src.random_forest_experiments import run_random_forest_experiment
from src.splitting import split_dataset, summarize_split, verify_split_checksums
from src.visualization import (
    save_model_comparison,
    save_pca_comparison,
    save_pca_train_projection,
)

LOGGER = logging.getLogger(__name__)


def _best_result(results: list[ExperimentResult]) -> ExperimentResult:
    maximum_f1 = max(float(result["cv_f1_macro_mean"]) for result in results)
    leaders = [
        result
        for result in results
        if isclose(
            float(result["cv_f1_macro_mean"]),
            maximum_f1,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    return min(
        leaders,
        key=lambda result: (int(result["n_features"]), result["experiment_id"]),
    )


def main() -> None:
    """Run E10-E17 on training data without final fitting or test evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    archive_path = download_dataset()
    features, target = load_dataset(archive_path)
    features_train, features_test, target_train, target_test = split_dataset(features, target)
    split_summary = summarize_split(target_train, target_test)
    verify_split_checksums(split_summary)

    existing_results = load_experiment_metrics()
    pca_results = run_pca_experiments(features_train, target_train)
    random_forest_result = run_random_forest_experiment(features_train, target_train)
    mlp_result, mlp_summary = evaluate_mlp_cv(features_train, target_train)
    extended_results = [*pca_results, random_forest_result, mlp_result]
    combined_results = combine_extended_results(existing_results, extended_results)
    metrics_path = save_experiment_metrics(combined_results, RUNS_DIR / "extended_metrics.csv")
    mlp_path = save_mlp_summary(mlp_summary, RUNS_DIR / "mlp_summary.json")

    save_pca_comparison(pca_results, RUNS_DIR / "pca_comparison.png")
    save_pca_train_projection(
        features_train,
        target_train,
        RUNS_DIR / "pca_train_projection.png",
    )
    logistic_pca = [result for result in pca_results if result["model"] == "LogisticRegression"]
    linear_svc_pca = [result for result in pca_results if result["model"] == "LinearSVC"]
    best_logistic_pca = _best_result(logistic_pca)
    best_linear_svc_pca = _best_result(linear_svc_pca)
    combined_by_id = {result["experiment_id"]: result for result in combined_results}
    comparison_results = [
        combined_by_id["E01"],
        combined_by_id["E05"],
        combined_by_id["E08"],
        best_logistic_pca,
        best_linear_svc_pca,
        random_forest_result,
        mlp_result,
    ]
    save_model_comparison(comparison_results, RUNS_DIR / "model_comparison.png")

    LOGGER.info(
        "Reproduced %d training and %d reserved test samples; split checksums match",
        features_train.shape[0],
        features_test.shape[0],
    )
    for result in extended_results:
        LOGGER.info(
            "%s %s, features=%d: CV macro F1 %.6f, total %.2f s",
            result["experiment_id"],
            result["model"],
            result["n_features"],
            result["cv_f1_macro_mean"],
            result["total_cv_time_seconds"],
        )
    LOGGER.info("MLP fold epochs: %s", mlp_summary["fold_epochs"])
    LOGGER.info("Source frozen metrics: %s", METRICS_PATH)
    LOGGER.info("New experiment metrics: %s", metrics_path)
    LOGGER.info("New MLP summary: %s", mlp_path)
    LOGGER.info("Test samples remained closed; no final model or test metrics were produced")
    LOGGER.info("Frozen canonical files in results/ and figures/ were not modified")


if __name__ == "__main__":
    main()
