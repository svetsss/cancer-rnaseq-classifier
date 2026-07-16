import shutil
import tarfile
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

import pandas as pd

from src.config import DATA_MEMBER, DATASET_ARCHIVE_PATH, DATASET_URL, LABELS_MEMBER


def download_dataset(destination: Path = DATASET_ARCHIVE_PATH) -> Path:
    """Download the UCI archive, reusing a valid local copy when available."""
    if destination.exists():
        if _archive_has_required_files(destination):
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
        if not _archive_has_required_files(temporary_path):
            raise ValueError(f"Downloaded dataset archive is invalid: {DATASET_URL}")
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
