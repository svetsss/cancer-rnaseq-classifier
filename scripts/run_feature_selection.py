import logging

from src.config import METRICS_PATH
from src.data_loader import download_dataset, load_dataset
from src.experiments import load_experiment_metrics, save_experiment_metrics
from src.feature_selection import (
    combine_feature_selection_results,
    run_feature_selection_experiments,
)
from src.splitting import split_dataset, summarize_split, verify_split_checksums
from src.visualization import save_feature_selection_comparison

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Run fold-local SelectKBest experiments without evaluating the test set."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    archive_path = download_dataset()
    features, target = load_dataset(archive_path)
    features_train, features_test, target_train, target_test = split_dataset(features, target)
    split_summary = summarize_split(target_train, target_test)
    verify_split_checksums(split_summary)

    existing_results = load_experiment_metrics()
    selection_results = run_feature_selection_experiments(features_train, target_train)
    combined_results = combine_feature_selection_results(existing_results, selection_results)
    save_experiment_metrics(combined_results)
    figure_path = save_feature_selection_comparison(selection_results)

    LOGGER.info(
        "Reproduced %d training and %d reserved test samples; split checksums match",
        features_train.shape[0],
        features_test.shape[0],
    )
    for result in selection_results:
        LOGGER.info(
            "%s %s, k=%d: CV macro F1 %.6f, total %.2f s",
            result["experiment_id"],
            result["model"],
            result["n_features"],
            result["cv_f1_macro_mean"],
            result["total_cv_time_seconds"],
        )
    LOGGER.info("Experiment metrics: %s", METRICS_PATH)
    LOGGER.info("Feature-selection comparison: %s", figure_path)
    LOGGER.info("Test samples remained reserved; no test metrics were computed")


if __name__ == "__main__":
    main()
