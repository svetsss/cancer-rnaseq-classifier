import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import (
    FIGURES_DIR,
    METRICS_PATH,
    MLP_EPOCH_SELECTION_PATH,
    MLP_FOLD_METRICS_FIGURE_PATH,
    MLP_FOLD_METRICS_PATH,
    MLP_LOSS_CURVES_PATH,
    MLP_TRAINING_ANALYSIS_PATH,
    MLP_VALIDATION_F1_PATH,
    MODEL_COMPARISON_PATH,
    README_MODEL_COMPARISON_PATH,
    RESULTS_DIR,
    RUNS_DIR,
)
from src.data_loader import download_dataset, load_dataset
from src.evaluation import ExperimentResult
from src.experiments import load_experiment_metrics, save_experiment_metrics
from src.neural_evaluation import evaluate_mlp_cv, save_mlp_summary
from src.splitting import split_dataset, summarize_split, verify_split_checksums
from src.visualization import (
    save_mlp_epoch_selection,
    save_mlp_fold_metrics,
    save_mlp_loss_curves,
    save_mlp_validation_f1,
    save_model_comparison,
    save_readme_model_comparison,
)

LOGGER = logging.getLogger(__name__)


def _save_fold_metrics(metrics: list[dict[str, float | int]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.csv")
    pd.DataFrame(metrics).to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)
    return output_path


def _replace_mlp_result(
    metrics: list[ExperimentResult],
    corrected_result: ExperimentResult,
) -> list[ExperimentResult]:
    updated = [
        corrected_result if metric["experiment_id"] == "E17" else metric for metric in metrics
    ]
    if sum(metric["experiment_id"] == "E17" for metric in updated) != 1:
        raise ValueError("Canonical metrics must contain exactly one E17 row")
    return updated


def main() -> None:
    """Record MLP epoch curves and fold metrics using training data only."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="write the presentation artifacts to results/ and figures/ instead of runs/",
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    archive_path = download_dataset()
    features, target = load_dataset(archive_path)
    features_train, features_test, target_train, target_test = split_dataset(features, target)
    split_summary = summarize_split(target_train, target_test)
    verify_split_checksums(split_summary)

    result, summary = evaluate_mlp_cv(features_train, target_train)
    if arguments.publish:
        summary_path = MLP_TRAINING_ANALYSIS_PATH
        fold_metrics_path = MLP_FOLD_METRICS_PATH
        figure_paths = (
            MLP_LOSS_CURVES_PATH,
            MLP_VALIDATION_F1_PATH,
            MLP_EPOCH_SELECTION_PATH,
            MLP_FOLD_METRICS_FIGURE_PATH,
            README_MODEL_COMPARISON_PATH,
        )
    else:
        summary_path = RUNS_DIR / MLP_TRAINING_ANALYSIS_PATH.name
        fold_metrics_path = RUNS_DIR / MLP_FOLD_METRICS_PATH.name
        figure_paths = tuple(
            RUNS_DIR / path.name
            for path in (
                MLP_LOSS_CURVES_PATH,
                MLP_VALIDATION_F1_PATH,
                MLP_EPOCH_SELECTION_PATH,
                MLP_FOLD_METRICS_FIGURE_PATH,
                README_MODEL_COMPARISON_PATH,
            )
        )

    save_mlp_summary(summary, summary_path)
    _save_fold_metrics(summary["fold_metrics"], fold_metrics_path)
    save_mlp_loss_curves(summary, figure_paths[0])
    save_mlp_validation_f1(summary, figure_paths[1])
    save_mlp_epoch_selection(summary, figure_paths[2])
    save_mlp_fold_metrics(summary, figure_paths[3])
    canonical_metrics = load_experiment_metrics()
    updated_metrics = _replace_mlp_result(canonical_metrics, result)
    comparison_metrics = pd.DataFrame(updated_metrics)
    if arguments.publish:
        save_experiment_metrics(updated_metrics, METRICS_PATH)
        by_id = {metric["experiment_id"]: metric for metric in updated_metrics}
        save_model_comparison(
            [
                by_id[experiment_id]
                for experiment_id in ("E01", "E05", "E08", "E10", "E13", "E16", "E17")
            ],
            MODEL_COMPARISON_PATH,
        )
    save_readme_model_comparison(comparison_metrics, figure_paths[4])

    LOGGER.info(
        "MLP training analysis used %d training rows; %d test rows remained reserved",
        len(features_train),
        len(features_test),
    )
    LOGGER.info(
        "Corrected MLP CV: macro F1 %.6f ± %.6f",
        result["cv_f1_macro_mean"],
        result["cv_f1_macro_std"],
    )
    LOGGER.info("Epoch histories: %s", summary_path)
    LOGGER.info("Fold metrics: %s", fold_metrics_path)
    LOGGER.info("Matplotlib figures: %s", ", ".join(str(path) for path in figure_paths))
    if arguments.publish:
        LOGGER.info(
            "Published canonical E17 metrics and figures under %s and %s", RESULTS_DIR, FIGURES_DIR
        )


if __name__ == "__main__":
    main()
