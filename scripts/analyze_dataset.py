import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from src.config import RUNS_DIR
from src.data_loader import download_dataset, load_dataset
from src.validation import DatasetSummary, summarize_dataset
from src.visualization import save_class_distribution

LOGGER = logging.getLogger(__name__)


def _save_summary(summary: DatasetSummary, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.json")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    temporary_path.replace(output_path)


def main() -> None:
    """Download, validate, and summarize the UCI RNA-Seq dataset."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    archive_path = download_dataset()
    LOGGER.info("Dataset archive: %s", archive_path)
    features, target = load_dataset(archive_path)

    with archive_path.open("rb") as archive_file:
        archive_sha256 = hashlib.file_digest(archive_file, "sha256").hexdigest()

    summary = summarize_dataset(
        features,
        target,
        archive_sha256=archive_sha256,
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    summary_path = RUNS_DIR / "dataset_summary.json"
    _save_summary(summary, summary_path)
    figure_path = save_class_distribution(target, RUNS_DIR / "class_distribution.png")

    LOGGER.info(
        "Validated %d samples, %d features, and %d classes",
        summary["sample_count"],
        summary["feature_count"],
        summary["class_count"],
    )
    LOGGER.info("Dataset summary: %s", summary_path)
    LOGGER.info("Class distribution: %s", figure_path)
    LOGGER.info("Frozen canonical files in results/ and figures/ were not modified")


if __name__ == "__main__":
    main()
