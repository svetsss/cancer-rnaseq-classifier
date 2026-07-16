import io
import tarfile
from pathlib import Path

import pandas as pd

from src.config import DATA_MEMBER, LABELS_MEMBER
from src.data_loader import download_dataset, load_dataset


def _add_csv(archive: tarfile.TarFile, member_name: str, csv_text: str) -> None:
    content = csv_text.encode("utf-8")
    member = tarfile.TarInfo(member_name)
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


def _write_test_archive(path: Path) -> None:
    features = pd.DataFrame(
        {"gene_0": [1.0, 2.0, 3.0], "gene_1": [4.0, 5.0, 6.0]},
        index=["sample_0", "sample_1", "sample_2"],
    )
    labels = pd.DataFrame(
        {"Class": ["BRCA", "KIRC", "BRCA"]},
        index=features.index,
    )
    with tarfile.open(path, mode="w:gz") as archive:
        _add_csv(archive, DATA_MEMBER, features.to_csv())
        _add_csv(archive, LABELS_MEMBER, labels.to_csv())


def test_load_dataset_separates_features_and_target(tmp_path: Path) -> None:
    archive_path = tmp_path / "dataset.tar.gz"
    _write_test_archive(archive_path)

    features, target = load_dataset(archive_path)

    assert features.shape == (3, 2)
    assert target.name == "cancer_type"
    assert target.tolist() == ["BRCA", "KIRC", "BRCA"]
    assert "Class" not in features.columns
    assert "cancer_type" not in features.columns
    assert features.index.equals(target.index)


def test_download_dataset_keeps_existing_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / "dataset.tar.gz"
    archive_path.write_bytes(b"existing archive")

    result = download_dataset(archive_path)

    assert result == archive_path
    assert archive_path.read_bytes() == b"existing archive"
