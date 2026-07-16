import logging

from src.config import METRICS_PATH, SPLIT_SUMMARY_PATH
from src.data_loader import download_dataset, load_dataset
from src.experiments import run_baseline_experiments, save_baseline_metrics
from src.splitting import save_split_summary, split_dataset, summarize_split

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Create the holdout split and evaluate the approved training baselines."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    archive_path = download_dataset()
    features, target = load_dataset(archive_path)
    features_train, features_test, target_train, target_test = split_dataset(features, target)

    split_summary = summarize_split(target_train, target_test)
    save_split_summary(split_summary)

    results = run_baseline_experiments(features_train, target_train)
    save_baseline_metrics(results)

    LOGGER.info(
        "Reserved %d training and %d test samples; test labels were not evaluated",
        features_train.shape[0],
        features_test.shape[0],
    )
    for result in results:
        LOGGER.info(
            "%s %s: CV macro F1 %.6f, accuracy %.6f, total %.2f s",
            result["experiment_id"],
            result["model"],
            result["cv_f1_macro_mean"],
            result["cv_accuracy_mean"],
            result["total_cv_time_seconds"],
        )
    LOGGER.info("Split summary: %s", SPLIT_SUMMARY_PATH)
    LOGGER.info("Baseline metrics: %s", METRICS_PATH)


if __name__ == "__main__":
    main()
