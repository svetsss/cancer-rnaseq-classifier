import argparse
import json
import logging
from pathlib import Path

from src.config import (
    ROBUSTNESS_FIGURE_PATH,
    ROBUSTNESS_PATH,
    RUNS_DIR,
)
from src.data_loader import download_dataset, load_dataset
from src.robustness import evaluate_e10_repeated_cv
from src.splitting import split_dataset, summarize_split, verify_split_checksums
from src.visualization import save_robustness_f1

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated train-only CV for E10 without touching the final test metrics."
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="write the reproducibility result to results/ and figures/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    archive_path = download_dataset()
    features, target = load_dataset(archive_path)
    features_train, features_test, target_train, target_test = split_dataset(features, target)
    split_summary = summarize_split(target_train, target_test)
    verify_split_checksums(split_summary)

    summary = evaluate_e10_repeated_cv(
        features_train,
        target_train,
        n_repeats=args.repeats,
    )
    default_output = ROBUSTNESS_PATH if args.publish else RUNS_DIR / ROBUSTNESS_PATH.name
    output_path = args.output or default_output
    figure_path = ROBUSTNESS_FIGURE_PATH if args.publish else RUNS_DIR / ROBUSTNESS_FIGURE_PATH.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.json")
    temporary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    save_robustness_f1(summary, figure_path)
    LOGGER.info(
        "Repeated CV macro F1 %.6f ± %.6f; output: %s",
        summary["f1_macro_mean"],
        summary["f1_macro_std"],
        output_path,
    )
    LOGGER.info("Robustness figure: %s", figure_path)
    LOGGER.info("Reserved test rows remained unused: %d", len(features_test))


if __name__ == "__main__":
    main()
