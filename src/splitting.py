import hashlib
import json
from pathlib import Path
from typing import TypedDict

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import CV_SPLITS, RANDOM_STATE, SPLIT_SUMMARY_PATH, TEST_SIZE


class SplitSummary(TypedDict):
    """Reproducible description of the reserved train and test subsets."""

    random_state: int
    test_size: float
    total_samples: int
    train_samples: int
    test_samples: int
    train_class_distribution: dict[str, int]
    test_class_distribution: dict[str, int]
    train_index_sha256: str
    test_index_sha256: str
    cv_type: str
    cv_splits: int
    cv_shuffle: bool


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the stratified holdout split used by all later experiments."""
    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    return features_train, features_test, target_train, target_test


def make_stratified_cv(
    *,
    n_splits: int = CV_SPLITS,
    random_state: int = RANDOM_STATE,
) -> StratifiedKFold:
    """Create the shuffled stratified cross-validation scheme for training data."""
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def summarize_split(
    target_train: pd.Series,
    target_test: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    cv_splits: int = CV_SPLITS,
) -> SplitSummary:
    """Describe split sizes, class balance, and sample identifier checksums."""
    return {
        "random_state": random_state,
        "test_size": test_size,
        "total_samples": len(target_train) + len(target_test),
        "train_samples": len(target_train),
        "test_samples": len(target_test),
        "train_class_distribution": _class_distribution(target_train),
        "test_class_distribution": _class_distribution(target_test),
        "train_index_sha256": _index_checksum(target_train.index),
        "test_index_sha256": _index_checksum(target_test.index),
        "cv_type": "StratifiedKFold",
        "cv_splits": cv_splits,
        "cv_shuffle": True,
    }


def save_split_summary(
    summary: SplitSummary,
    output_path: Path = SPLIT_SUMMARY_PATH,
) -> Path:
    """Atomically replace the saved split description."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.json")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    temporary_path.replace(output_path)
    return output_path


def verify_split_checksums(
    current_summary: SplitSummary,
    saved_path: Path = SPLIT_SUMMARY_PATH,
) -> None:
    """Require reproduced train and test index checksums to match the saved split."""
    saved_summary = json.loads(saved_path.read_text(encoding="utf-8"))
    for field in ("train_index_sha256", "test_index_sha256"):
        if current_summary[field] != saved_summary[field]:
            raise RuntimeError(f"Saved {field} does not match the reproduced split")


def _class_distribution(target: pd.Series) -> dict[str, int]:
    return {str(label): int(count) for label, count in target.value_counts().sort_index().items()}


def _index_checksum(index: pd.Index) -> str:
    digest = hashlib.sha256()
    for sample_id in index:
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
