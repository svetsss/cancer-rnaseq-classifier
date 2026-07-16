import hashlib
import shutil
import tarfile
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from src.config import (
    DATA_MEMBER,
    DATASET_ARCHIVE_PATH,
    DATASET_URL,
    EXPECTED_DATASET_SHA256,
    LABELS_MEMBER,
)


def download_dataset(
    destination: Path = DATASET_ARCHIVE_PATH,
    *,
    expected_sha256: str | None = EXPECTED_DATASET_SHA256,
) -> Path:
    """Download the exact UCI archive, reusing only a validated local copy."""
    if destination.exists():
        if _archive_is_valid(destination, expected_sha256):
            return destination
        destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(DATASET_URL, headers={"User-Agent": "cancer-rnaseq-classifier/0.1"})

    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(dir=destination.parent, delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with urlopen(request, timeout=120) as response:
                shutil.copyfileobj(response, temporary_file)
        if not _archive_is_valid(temporary_path, expected_sha256):
            raise ValueError(
                "Downloaded dataset archive is invalid or its SHA-256 does not match "
                f"the frozen UCI snapshot: {DATASET_URL}"
            )
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return destination


def load_dataset(archive_path: Path = DATASET_ARCHIVE_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load the feature matrix and target from the original UCI archive."""
    if not archive_path.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive_path}")

    with tarfile.open(archive_path, mode="r:gz") as archive:
        features = _read_csv_member(archive, DATA_MEMBER)
        labels = _read_csv_member(archive, LABELS_MEMBER)

    if labels.shape[1] != 1:
        raise ValueError(f"Expected one target column in {LABELS_MEMBER}, found {labels.shape[1]}")

    target = labels.iloc[:, 0].astype("string").rename("cancer_type")
    features.index.name = "sample_id"
    target.index.name = "sample_id"
    _validate_loaded_dataset(features, target)
    return features, target


def _read_csv_member(archive: tarfile.TarFile, member_name: str) -> pd.DataFrame:
    member = archive.getmember(member_name)
    extracted_file = archive.extractfile(member)
    if extracted_file is None:
        raise ValueError(f"Archive member is not a regular file: {member_name}")
    with extracted_file:
        return pd.read_csv(extracted_file, index_col=0)


def _archive_has_required_files(archive_path: Path) -> bool:
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            return all(
                archive.getmember(member_name).isfile()
                for member_name in (DATA_MEMBER, LABELS_MEMBER)
            )
    except (KeyError, OSError, tarfile.TarError):
        return False


def _archive_is_valid(archive_path: Path, expected_sha256: str | None) -> bool:
    if not _archive_has_required_files(archive_path):
        return False
    return expected_sha256 is None or _sha256_file(archive_path) == expected_sha256


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_loaded_dataset(features: pd.DataFrame, target: pd.Series) -> None:
    if features.empty or target.empty:
        raise ValueError("Feature matrix and target must not be empty")
    if not features.index.equals(target.index):
        raise ValueError("Feature and target sample identifiers or their order do not match")
    if features.index.has_duplicates:
        raise ValueError("Feature matrix contains duplicate sample identifiers")
    if features.columns.has_duplicates:
        raise ValueError("Feature matrix contains duplicate feature names")
    if target.name in features.columns:
        raise ValueError("Target column is present in the feature matrix")
    if features.select_dtypes(exclude="number").shape[1]:
        raise ValueError("Feature matrix must contain only numeric values")
    if target.isna().any() or features.isna().to_numpy().any():
        raise ValueError("Dataset contains missing values")
    if not np.isfinite(features.to_numpy(dtype=np.float64, copy=False)).all():
        raise ValueError("Feature matrix contains infinite values")
    if target.nunique() < 2:
        raise ValueError("Target must contain at least two classes")
