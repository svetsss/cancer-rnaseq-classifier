import shutil
import tarfile
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

import pandas as pd

from src.config import DATA_MEMBER, DATASET_ARCHIVE_PATH, DATASET_URL, LABELS_MEMBER


def download_dataset(destination: Path = DATASET_ARCHIVE_PATH) -> Path:
    """Download the immutable UCI archive unless it is already present."""
    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(DATASET_URL, headers={"User-Agent": "cancer-rnaseq-classifier/0.1"})

    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(dir=destination.parent, delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with urlopen(request, timeout=120) as response:
                shutil.copyfileobj(response, temporary_file)
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
