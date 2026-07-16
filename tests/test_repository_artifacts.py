import json
import re
from pathlib import Path

from src.artifact_verification import require_valid_final_artifacts
from src.candidate_selection import (
    build_final_candidate,
    freeze_final_candidate,
    select_final_candidate,
)
from src.experiments import load_experiment_metrics

MARKDOWN_LINK_PATTERN = re.compile(r"!?\[[^]]*]\(([^)]+)\)")


def test_committed_final_artifacts_match_manifest() -> None:
    project_root = Path(__file__).resolve().parents[1]

    statuses = require_valid_final_artifacts(project_root)

    assert statuses
    assert all(status["exists"] and status["sha256_matches"] for status in statuses)


def test_readme_local_links_point_to_existing_files() -> None:
    project_root = Path(__file__).resolve().parents[1]
    readme = (project_root / "README.md").read_text(encoding="utf-8")
    targets = MARKDOWN_LINK_PATTERN.findall(readme)
    local_targets = [
        target.split("#", maxsplit=1)[0]
        for target in targets
        if "://" not in target and not target.startswith("#")
    ]

    missing = [target for target in local_targets if not (project_root / target).exists()]

    assert not missing
    assert "figures/project_pipeline.png" in local_targets
    assert "defense_guide" not in readme


def test_committed_historic_candidate_is_compatible_with_current_selection(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    committed_path = project_root / "results" / "final_candidate.json"
    copied_path = tmp_path / "final_candidate.json"
    copied_path.write_bytes(committed_path.read_bytes())
    metrics = load_experiment_metrics(project_root / "results" / "metrics.csv")
    split_summary = json.loads(
        (project_root / "results" / "split_summary.json").read_text(encoding="utf-8")
    )
    current_candidate = build_final_candidate(
        select_final_candidate(metrics),
        split_summary,
    )

    freeze_final_candidate(current_candidate, copied_path)

    assert copied_path.read_bytes() == committed_path.read_bytes()
