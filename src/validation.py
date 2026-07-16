from typing import TypedDict

import numpy as np
import pandas as pd

from src.config import DATASET_ID, DATASET_NAME, DATASET_URL


class QualitySummary(TypedDict):
    """Quality checks recorded for the loaded dataset."""

    missing_values: int
    infinite_values: int
    duplicate_sample_ids: int
    duplicate_feature_names: int
    duplicate_feature_rows: int


class DatasetSummary(TypedDict):
    """Serializable evidence produced by the stage 1 analysis."""

    dataset_id: int
    dataset_name: str
    source_url: str
    generated_at_utc: str
    archive_sha256: str
    sample_count: int
    feature_count: int
    class_count: int
    target_name: str
    class_distribution: dict[str, int]
    feature_dtype_counts: dict[str, int]
    quality: QualitySummary


def summarize_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    archive_sha256: str,
    generated_at_utc: str,
) -> DatasetSummary:
    """Validate the loaded dataset and return a JSON-serializable summary."""
    if features.empty:
        raise ValueError("Feature matrix is empty")
    if target.empty:
        raise ValueError("Target is empty")
    if not features.index.equals(target.index):
        raise ValueError("Feature and target sample identifiers do not match")
    if features.index.has_duplicates:
        raise ValueError("Feature matrix contains duplicate sample identifiers")
    if features.columns.has_duplicates:
        raise ValueError("Feature matrix contains duplicate feature names")
    if target.name in features.columns:
        raise ValueError(f"Target column {target.name!r} is present in the feature matrix")

    non_numeric_columns = features.select_dtypes(exclude="number").columns.tolist()
    if non_numeric_columns:
        preview = ", ".join(map(str, non_numeric_columns[:5]))
        raise ValueError(f"Feature matrix contains non-numeric columns: {preview}")

    missing_values = int(features.isna().to_numpy().sum() + target.isna().sum())
    if missing_values:
        raise ValueError(f"Dataset contains {missing_values} missing values")

    infinite_values = int(np.isinf(features.to_numpy(dtype=np.float64, copy=False)).sum())
    if infinite_values:
        raise ValueError(f"Feature matrix contains {infinite_values} infinite values")

    class_distribution = {
        str(label): int(count) for label, count in target.value_counts().sort_index().items()
    }
    if len(class_distribution) < 2:
        raise ValueError("Target must contain at least two classes")

    dtype_counts = {
        str(dtype): int(count)
        for dtype, count in features.dtypes.astype(str).value_counts().sort_index().items()
    }

    return {
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "source_url": DATASET_URL,
        "generated_at_utc": generated_at_utc,
        "archive_sha256": archive_sha256,
        "sample_count": int(features.shape[0]),
        "feature_count": int(features.shape[1]),
        "class_count": len(class_distribution),
        "target_name": str(target.name),
        "class_distribution": class_distribution,
        "feature_dtype_counts": dtype_counts,
        "quality": {
            "missing_values": missing_values,
            "infinite_values": infinite_values,
            "duplicate_sample_ids": int(features.index.duplicated().sum()),
            "duplicate_feature_names": int(features.columns.duplicated().sum()),
            "duplicate_feature_rows": int(features.duplicated().sum()),
        },
    }
