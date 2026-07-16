import io
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from src.config import DATA_MEMBER, LABELS_MEMBER
from src.data_loader import download_dataset, load_dataset


def _add_csv(archive: tarfile.TarFile, member_name: str, csv_text: str) -> None:
    content = csv_text.encode("utf-8")
    member = tarfile.TarInfo(member_name)
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


def _test_archive_bytes(*, include_labels: bool = True) -> bytes:
    features = pd.DataFrame(
        {"gene_0": [1.0, 2.0, 3.0], "gene_1": [4.0, 5.0, 6.0]},
        index=["sample_0", "sample_1", "sample_2"],
    )
    labels = pd.DataFrame(
        {"Class": ["BRCA", "KIRC", "BRCA"]},
        index=features.index,
    )
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        _add_csv(archive, DATA_MEMBER, features.to_csv())
        if include_labels:
            _add_csv(archive, LABELS_MEMBER, labels.to_csv())
    return buffer.getvalue()


def _write_test_archive(path: Path, *, include_labels: bool = True) -> None:
    path.write_bytes(_test_archive_bytes(include_labels=include_labels))


def _mock_download(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    monkeypatch.setattr(
        "src.data_loader.urlopen",
        lambda request, timeout: io.BytesIO(payload),
    )


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


def test_download_dataset_reuses_valid_existing_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "dataset.tar.gz"
    _write_test_archive(archive_path)
    original_content = archive_path.read_bytes()
    monkeypatch.setattr(
        "src.data_loader.urlopen",
        lambda request, timeout: pytest.fail("Unexpected download"),
    )

    result = download_dataset(archive_path)

    assert result == archive_path
    assert archive_path.read_bytes() == original_content


def test_download_dataset_replaces_corrupted_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "dataset.tar.gz"
    archive_path.write_bytes(b"not a tar archive")
    _mock_download(monkeypatch, _test_archive_bytes())

    result = download_dataset(archive_path)
    features, target = load_dataset(result)

    assert features.shape == (3, 2)
    assert target.tolist() == ["BRCA", "KIRC", "BRCA"]


def test_download_dataset_replaces_archive_without_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "dataset.tar.gz"
    _write_test_archive(archive_path, include_labels=False)
    _mock_download(monkeypatch, _test_archive_bytes())

    result = download_dataset(archive_path)
    _, target = load_dataset(result)

    assert target.tolist() == ["BRCA", "KIRC", "BRCA"]


def test_download_dataset_rejects_invalid_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "dataset.tar.gz"
    _mock_download(monkeypatch, b"invalid download")

    with pytest.raises(ValueError, match="Downloaded dataset archive is invalid"):
        download_dataset(archive_path)

    assert not archive_path.exists()
