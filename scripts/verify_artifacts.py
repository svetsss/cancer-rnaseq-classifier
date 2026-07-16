import logging

from src.artifact_verification import require_valid_final_artifacts
from src.config import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Verify committed final outputs without loading data or unpickling the model."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    statuses = require_valid_final_artifacts(PROJECT_ROOT)
    for status in statuses:
        LOGGER.info("OK %-24s %s", status["name"], status["path"])
    LOGGER.info("Verified %d frozen artifacts", len(statuses))


if __name__ == "__main__":
    main()
