import json
from pathlib import Path

import pandas as pd

from src.splitting import (
    make_stratified_cv,
    save_split_summary,
    split_dataset,
    summarize_split,
)


def _dataset() -> tuple[pd.DataFrame, pd.Series]:
    index = pd.Index([f"sample_{number}" for number in range(50)], name="sample_id")
    features = pd.DataFrame(
        {"gene_0": range(50), "gene_1": range(50, 100)},
        index=index,
    )
    target = pd.Series(
        ["A"] * 20 + ["B"] * 15 + ["C"] * 10 + ["D"] * 5,
        index=index,
        dtype="string",
        name="cancer_type",
    )
    return features, target


def test_split_preserves_class_proportions() -> None:
    features, target = _dataset()

    _, _, _, target_test = split_dataset(features, target)

    assert target_test.value_counts().to_dict() == {"A": 4, "B": 3, "C": 2, "D": 1}


def test_split_indices_do_not_overlap() -> None:
    features, target = _dataset()

    features_train, features_test, _, _ = split_dataset(features, target)

    assert set(features_train.index).isdisjoint(features_test.index)


def test_split_preserves_total_sample_count() -> None:
    features, target = _dataset()

    features_train, features_test, _, _ = split_dataset(features, target)

    assert len(features_train) + len(features_test) == len(features)


def test_split_is_reproducible() -> None:
    features, target = _dataset()

    first_split = split_dataset(features, target)
    second_split = split_dataset(features, target)

    for first_part, second_part in zip(first_split, second_split, strict=True):
        assert first_part.index.equals(second_part.index)


def test_cross_validator_configuration() -> None:
    cross_validator = make_stratified_cv()

    assert cross_validator.n_splits == 5
    assert cross_validator.shuffle is True
    assert cross_validator.random_state == 42


def test_saved_split_summary_contains_required_fields(tmp_path: Path) -> None:
    features, target = _dataset()
    _, _, target_train, target_test = split_dataset(features, target)
    summary = summarize_split(target_train, target_test)
    output_path = tmp_path / "split_summary.json"

    save_split_summary(summary, output_path)

    saved_summary = json.loads(output_path.read_text(encoding="utf-8"))
    required_fields = {
        "random_state",
        "test_size",
        "total_samples",
        "train_samples",
        "test_samples",
        "train_class_distribution",
        "test_class_distribution",
        "cv_type",
        "cv_splits",
        "cv_shuffle",
    }
    assert required_fields <= saved_summary.keys()
    assert len(saved_summary["train_index_sha256"]) == 64
    assert len(saved_summary["test_index_sha256"]) == 64
