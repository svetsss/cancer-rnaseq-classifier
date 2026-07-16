import json
from pathlib import Path

import pytest

from src.artifact_verification import require_valid_final_artifacts, sha256_file


def _project(tmp_path: Path) -> Path:
    artifact_path = tmp_path / "models" / "model.bin"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"trusted model bytes")
    manifest_path = tmp_path / "results" / "final_evaluation.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "artifacts": {
                    "model": {
                        "path": "models/model.bin",
                        "sha256": sha256_file(artifact_path),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_valid_artifact_manifest_is_accepted(tmp_path: Path) -> None:
    statuses = require_valid_final_artifacts(_project(tmp_path))

    assert statuses == [
        {
            "name": "model",
            "path": "models/model.bin",
            "exists": True,
            "sha256_matches": True,
        }
    ]


def test_changed_artifact_is_rejected(tmp_path: Path) -> None:
    project_root = _project(tmp_path)
    (project_root / "models" / "model.bin").write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="model"):
        require_valid_final_artifacts(project_root)


def test_artifact_path_cannot_escape_project_root(tmp_path: Path) -> None:
    project_root = _project(tmp_path)
    manifest_path = project_root / "results" / "final_evaluation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["model"]["path"] = "../outside.bin"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes project root"):
        require_valid_final_artifacts(project_root)
