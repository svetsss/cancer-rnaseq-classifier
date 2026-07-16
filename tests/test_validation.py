import numpy as np
import pandas as pd
import pytest

from src.validation import DatasetSummary, summarize_dataset


def _valid_dataset() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.Index(["sample_0", "sample_1", "sample_2"], name="sample_id")
    features = pd.DataFrame(
        {"gene_0": [1.0, 2.0, 3.0], "gene_1": [3.0, 2.0, 1.0]},
        index=index,
    )
    target = pd.Series(["BRCA", "KIRC", "BRCA"], index=index, dtype="string", name="cancer_type")
    return features, target


def _summarize(features: pd.DataFrame, target: pd.Series) -> DatasetSummary:
    return summarize_dataset(
        features,
        target,
        archive_sha256="a" * 64,
        generated_at_utc="2026-07-16T12:00:00+00:00",
    )


def test_summary_contains_required_dataset_characteristics() -> None:
    features, target = _valid_dataset()

    summary = _summarize(features, target)

    assert summary["sample_count"] == 3
    assert summary["feature_count"] == 2
    assert summary["class_count"] == 2
    assert summary["class_distribution"] == {"BRCA": 2, "KIRC": 1}
    assert summary["quality"]["missing_values"] == 0
    assert summary["quality"]["duplicate_feature_rows"] == 0
    assert summary["quality"]["constant_features"] == 0


def test_validation_rejects_non_numeric_feature() -> None:
    features, target = _valid_dataset()
    features["gene_2"] = ["low", "medium", "high"]

    with pytest.raises(ValueError, match="non-numeric columns: gene_2"):
        _summarize(features, target)


def test_validation_rejects_misaligned_target() -> None:
    features, target = _valid_dataset()
    target.index = pd.Index(["sample_1", "sample_0", "sample_2"], name="sample_id")

    with pytest.raises(ValueError, match="identifiers do not match"):
        _summarize(features, target)


def test_validation_rejects_missing_value() -> None:
    features, target = _valid_dataset()
    features.loc["sample_0", "gene_0"] = np.nan

    with pytest.raises(ValueError, match="contains 1 missing values"):
        _summarize(features, target)


def test_validation_rejects_infinite_value() -> None:
    features, target = _valid_dataset()
    features.loc["sample_0", "gene_0"] = np.inf

    with pytest.raises(ValueError, match="contains 1 infinite values"):
        _summarize(features, target)


def test_validation_rejects_duplicate_sample_id() -> None:
    features, target = _valid_dataset()
    duplicate_index = pd.Index(["sample_0", "sample_0", "sample_2"], name="sample_id")
    features.index = duplicate_index
    target.index = duplicate_index

    with pytest.raises(ValueError, match="duplicate sample identifiers"):
        _summarize(features, target)


def test_validation_rejects_duplicate_feature_name() -> None:
    features, target = _valid_dataset()
    features.columns = ["gene_0", "gene_0"]

    with pytest.raises(ValueError, match="duplicate feature names"):
        _summarize(features, target)


def test_validation_rejects_target_column_inside_features() -> None:
    features, target = _valid_dataset()
    features["cancer_type"] = [0.0, 1.0, 0.0]

    with pytest.raises(ValueError, match="Target column 'cancer_type'"):
        _summarize(features, target)


def test_summary_counts_duplicate_feature_rows() -> None:
    features, target = _valid_dataset()
    features.loc["sample_2"] = features.loc["sample_0"]

    summary = _summarize(features, target)

    assert summary["quality"]["duplicate_feature_rows"] == 1


def test_summary_counts_constant_features() -> None:
    features, target = _valid_dataset()
    features["gene_constant"] = 7.0

    summary = _summarize(features, target)

    assert summary["quality"]["constant_features"] == 1
