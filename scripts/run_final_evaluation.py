import logging

from src.config import (
    FINAL_CANDIDATE_PATH,
    METRICS_PATH,
    PROJECT_ROOT,
    SPLIT_SUMMARY_PATH,
)
from src.data_loader import download_dataset, load_dataset
from src.final_evaluation import (
    build_frozen_e10_pipeline,
    ensure_no_final_artifacts,
    evaluate_frozen_pipeline,
    final_artifact_paths,
    load_and_validate_frozen_inputs,
    save_final_artifacts,
    validate_split_metadata,
)
from src.splitting import split_dataset, summarize_split

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Train and evaluate the frozen E10 pipeline exactly once."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    paths = final_artifact_paths(PROJECT_ROOT)
    ensure_no_final_artifacts(paths)
    load_and_validate_frozen_inputs(
        FINAL_CANDIDATE_PATH,
        METRICS_PATH,
        SPLIT_SUMMARY_PATH,
    )

    archive_path = download_dataset()
    features, target = load_dataset(archive_path)
    features_train, features_test, target_train, target_test = split_dataset(features, target)
    reproduced_split = summarize_split(target_train, target_test)
    validate_split_metadata(reproduced_split)

    pipeline = build_frozen_e10_pipeline()
    evaluation = evaluate_frozen_pipeline(
        pipeline,
        features_train,
        target_train,
        features_test,
        target_test,
    )
    manifest = save_final_artifacts(
        pipeline,
        evaluation,
        features_test,
        target_test,
        paths,
        PROJECT_ROOT,
    )

    metrics = evaluation["test_metrics"]
    LOGGER.info(
        "E10 final test: macro F1 %.6f, accuracy %.6f, precision %.6f, recall %.6f",
        metrics["f1_macro"],
        metrics["accuracy"],
        metrics["precision_macro"],
        metrics["recall_macro"],
    )
    LOGGER.info(
        "Training %.3f s, predict %.3f s, predict_proba %.3f s",
        evaluation["training_time_seconds"],
        evaluation["prediction_time_seconds"],
        evaluation["probability_prediction_time_seconds"],
    )
    LOGGER.info("Final evaluation marker: %s", paths["evaluation"])
    LOGGER.info(
        "Evaluation status %s; no other model was evaluated or tuned",
        manifest["evaluation_status"],
    )


if __name__ == "__main__":
    main()
