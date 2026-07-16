import hashlib
import json
from pathlib import Path
from typing import TypedDict


class ArtifactStatus(TypedDict):
    name: str
    path: str
    exists: bool
    sha256_matches: bool


def sha256_file(path: Path) -> str:
    """Calculate SHA-256 without deserializing the artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_final_artifacts(
    project_root: Path,
    manifest_path: Path | None = None,
) -> list[ArtifactStatus]:
    """Verify every frozen artifact path and hash without loading the joblib model."""
    resolved_root = project_root.resolve()
    manifest_file = manifest_path or resolved_root / "results" / "final_evaluation.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Final evaluation manifest does not contain artifacts")

    statuses: list[ArtifactStatus] = []
    for name, raw_record in artifacts.items():
        if not isinstance(raw_record, dict):
            raise ValueError(f"Unexpected artifact record: {name}")
        relative_path = raw_record.get("path")
        expected_sha256 = raw_record.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_sha256, str):
            raise ValueError(f"Incomplete artifact record: {name}")

        artifact_path = (resolved_root / relative_path).resolve()
        if not artifact_path.is_relative_to(resolved_root):
            raise ValueError(f"Artifact path escapes project root: {relative_path}")
        exists = artifact_path.is_file()
        statuses.append(
            {
                "name": str(name),
                "path": relative_path,
                "exists": exists,
                "sha256_matches": exists and sha256_file(artifact_path) == expected_sha256,
            }
        )
    return statuses


def require_valid_final_artifacts(
    project_root: Path,
    manifest_path: Path | None = None,
) -> list[ArtifactStatus]:
    """Raise when a frozen final artifact is missing or has changed."""
    statuses = verify_final_artifacts(project_root, manifest_path)
    invalid = [
        status for status in statuses if not status["exists"] or not status["sha256_matches"]
    ]
    if invalid:
        names = ", ".join(status["name"] for status in invalid)
        raise RuntimeError(f"Final artifact verification failed: {names}")
    return statuses
