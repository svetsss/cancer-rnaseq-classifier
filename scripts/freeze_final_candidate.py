import json
import logging

from src.candidate_selection import (
    SELECTED_EXPERIMENT_ID,
    build_final_candidate,
    freeze_final_candidate,
    select_final_candidate,
)
from src.config import FINAL_CANDIDATE_PATH, METRICS_PATH, SPLIT_SUMMARY_PATH
from src.experiments import load_experiment_metrics

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Freeze E10 using only saved train CV results and split metadata."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = load_experiment_metrics(METRICS_PATH)
    selected_result = select_final_candidate(results)
    if selected_result["experiment_id"] != SELECTED_EXPERIMENT_ID:
        raise RuntimeError("Saved train CV results no longer select E10")

    loaded_split_summary = json.loads(SPLIT_SUMMARY_PATH.read_text(encoding="utf-8"))
    if not isinstance(loaded_split_summary, dict):
        raise ValueError("Unexpected split_summary.json structure")
    candidate = build_final_candidate(selected_result, loaded_split_summary)
    freeze_final_candidate(candidate, FINAL_CANDIDATE_PATH)

    LOGGER.info("Frozen E10: StandardScaler -> PCA(20, randomized) -> LogisticRegression")
    LOGGER.info("Candidate record: %s", FINAL_CANDIDATE_PATH)
    LOGGER.info("Dataset and test objects were not read or evaluated; no model was trained")


if __name__ == "__main__":
    main()
