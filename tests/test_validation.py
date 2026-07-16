import pandas as pd
import pytest

from src.validation import summarize_dataset


def _valid_dataset() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.Index(["sample_0", "sample_1", "sample_2"], name="sample_id")
    features = pd.DataFrame(
        {"gene_0": [1.0, 2.0, 3.0], "gene_1": [3.0, 2.0, 1.0]},
        index=index,
    )
    target = pd.Series(["BRCA", "KIRC", "BRCA"], index=index, dtype="string", name="cancer_type")
    return features, target


def test_summary_contains_required_dataset_characteristics() -> None:
    features, target = _valid_dataset()

    summary = summarize_dataset(
        features,
        target,
        archive_sha256="a" * 64,
        generated_at_utc="2026-07-16T12:00:00+00:00",
    )

    assert summary["sample_count"] == 3
    assert summary["feature_count"] == 2
    assert summary["class_count"] == 2
    assert summary["class_distribution"] == {"BRCA": 2, "KIRC": 1}
    assert summary["quality"]["missing_values"] == 0
    assert summary["quality"]["duplicate_feature_rows"] == 0


def test_validation_rejects_non_numeric_feature() -> None:
    features, target = _valid_dataset()
    features["gene_2"] = ["low", "medium", "high"]

    with pytest.raises(ValueError, match="non-numeric columns: gene_2"):
        summarize_dataset(
            features,
            target,
            archive_sha256="a" * 64,
            generated_at_utc="2026-07-16T12:00:00+00:00",
        )


def test_validation_rejects_misaligned_target() -> None:
    features, target = _valid_dataset()
    target.index = pd.Index(["sample_1", "sample_0", "sample_2"], name="sample_id")

    with pytest.raises(ValueError, match="identifiers do not match"):
        summarize_dataset(
            features,
            target,
            archive_sha256="a" * 64,
            generated_at_utc="2026-07-16T12:00:00+00:00",
        )
