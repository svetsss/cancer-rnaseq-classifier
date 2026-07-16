from pathlib import Path

from src.artifact_verification import require_valid_final_artifacts


def test_committed_final_artifacts_match_manifest() -> None:
    project_root = Path(__file__).resolve().parents[1]

    statuses = require_valid_final_artifacts(project_root)

    assert statuses
    assert all(status["exists"] and status["sha256_matches"] for status in statuses)
