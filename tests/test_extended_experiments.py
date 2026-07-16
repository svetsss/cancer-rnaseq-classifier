import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scripts import run_extended_experiments

from src.evaluation import ExperimentResult
from src.experiments import combine_extended_results, save_experiment_metrics
from src.neural_evaluation import MLPSummary
from src.splitting import SplitSummary, verify_split_checksums
from src.visualization import save_model_comparison


def _result(experiment_id: str) -> ExperimentResult:
    experiment_number = int(experiment_id[1:])
    return {
        "experiment_id": experiment_id,
        "model": "Model",
        "feature_method": "none",
        "n_features": 200,
        "cv_f1_macro_mean": 0.8 + experiment_number / 1000,
        "cv_f1_macro_std": 0.01,
        "cv_accuracy_mean": 0.8,
        "cv_accuracy_std": 0.01,
        "cv_precision_macro_mean": 0.8,
        "cv_precision_macro_std": 0.01,
        "cv_recall_macro_mean": 0.8,
        "cv_recall_macro_std": 0.01,
        "cv_fit_time_mean": 0.1,
        "cv_score_time_mean": 0.01,
        "total_cv_time_seconds": 0.5,
        "random_state": 42,
        "cv_splits": 5,
    }


def _preserved_results() -> list[ExperimentResult]:
    return [_result(f"E{number:02d}") for number in range(10)]


def _extended_results() -> list[ExperimentResult]:
    return [_result(f"E{number:02d}") for number in range(10, 18)]


def _mlp_summary() -> MLPSummary:
    return {
        "experiment_id": "E17",
        "architecture": ["Linear(200, 128)", "Linear(128, 64)", "Linear(64, 5)"],
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "batch_size": 32,
        "max_epochs": 100,
        "patience": 10,
        "inner_validation_size": 0.15,
        "fold_epochs": [10, 10],
        "mean_epochs": 10.0,
        "device": "cpu",
        "random_state": 42,
    }


def test_combined_metrics_contain_e00_through_e17() -> None:
    combined = combine_extended_results(_preserved_results(), _extended_results())

    assert [result["experiment_id"] for result in combined] == [
        f"E{number:02d}" for number in range(18)
    ]


def test_combined_experiment_ids_are_unique() -> None:
    combined = combine_extended_results(_preserved_results(), _extended_results())
    experiment_ids = [result["experiment_id"] for result in combined]

    assert len(experiment_ids) == len(set(experiment_ids))


def test_extended_metrics_have_no_test_columns(tmp_path: Path) -> None:
    output_path = tmp_path / "metrics.csv"
    combined = combine_extended_results(_preserved_results(), _extended_results())

    save_experiment_metrics(combined, output_path)

    with output_path.open(encoding="utf-8", newline="") as metrics_file:
        fieldnames = csv.DictReader(metrics_file).fieldnames
    assert fieldnames is not None
    assert not any(column.startswith("test_") for column in fieldnames)


def test_combining_extended_results_preserves_e00_through_e09() -> None:
    preserved = _preserved_results()

    combined = combine_extended_results(preserved, _extended_results())

    assert combined[:10] == preserved


def test_repeated_extended_combining_does_not_duplicate_rows() -> None:
    extended = _extended_results()
    first_combination = combine_extended_results(_preserved_results(), extended)

    second_combination = combine_extended_results(first_combination, extended)

    assert len(second_combination) == 18
    assert len({result["experiment_id"] for result in second_combination}) == 18


def test_split_checksum_verification_does_not_modify_summary(tmp_path: Path) -> None:
    saved_path = tmp_path / "split_summary.json"
    summary: SplitSummary = {
        "random_state": 42,
        "test_size": 0.2,
        "total_samples": 10,
        "train_samples": 8,
        "test_samples": 2,
        "train_class_distribution": {"A": 4, "B": 4},
        "test_class_distribution": {"A": 1, "B": 1},
        "train_index_sha256": "train-checksum",
        "test_index_sha256": "test-checksum",
        "cv_type": "StratifiedKFold",
        "cv_splits": 5,
        "cv_shuffle": True,
    }
    saved_path.write_text(json.dumps(summary), encoding="utf-8")
    before = saved_path.read_bytes()

    verify_split_checksums(summary, saved_path)

    assert saved_path.read_bytes() == before


def test_model_comparison_is_png(tmp_path: Path) -> None:
    experiment_ids = ("E01", "E05", "E08", "E10", "E13", "E16", "E17")
    output_path = tmp_path / "model_comparison.png"

    result = save_model_comparison(
        [_result(experiment_id) for experiment_id in experiment_ids], output_path
    )

    assert result == output_path
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_extended_evaluators_receive_only_training_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    random = np.random.default_rng(42)
    features = pd.DataFrame(random.normal(size=(20, 6)), index=[f"sample_{i}" for i in range(20)])
    target = pd.Series(["A", "B"] * 10, index=features.index, dtype="string")
    features_train = features.iloc[:16]
    features_test = features.iloc[16:]
    target_train = target.iloc[:16]
    target_test = target.iloc[16:]
    observed_indices: list[pd.Index] = []
    pca_results = [_result(f"E{number:02d}") for number in range(10, 16)]
    for result in pca_results[:3]:
        result["model"] = "LogisticRegression"
    for result in pca_results[3:]:
        result["model"] = "LinearSVC"
    random_forest_result = _result("E16")
    mlp_result = _result("E17")

    monkeypatch.setattr(run_extended_experiments, "download_dataset", lambda: Path("archive"))
    monkeypatch.setattr(run_extended_experiments, "load_dataset", lambda path: (features, target))
    monkeypatch.setattr(
        run_extended_experiments,
        "split_dataset",
        lambda loaded_features, loaded_target: (
            features_train,
            features_test,
            target_train,
            target_test,
        ),
    )
    monkeypatch.setattr(
        run_extended_experiments,
        "summarize_split",
        lambda train_target, test_target: {},
    )
    monkeypatch.setattr(run_extended_experiments, "verify_split_checksums", lambda summary: None)
    monkeypatch.setattr(
        run_extended_experiments,
        "load_experiment_metrics",
        _preserved_results,
    )

    def fake_pca(
        evaluated_features: pd.DataFrame, evaluated_target: pd.Series
    ) -> list[ExperimentResult]:
        observed_indices.append(evaluated_features.index)
        return pca_results

    def fake_random_forest(
        evaluated_features: pd.DataFrame,
        evaluated_target: pd.Series,
    ) -> ExperimentResult:
        observed_indices.append(evaluated_features.index)
        return random_forest_result

    def fake_mlp(
        evaluated_features: pd.DataFrame,
        evaluated_target: pd.Series,
    ) -> tuple[ExperimentResult, MLPSummary]:
        observed_indices.append(evaluated_features.index)
        return mlp_result, _mlp_summary()

    monkeypatch.setattr(run_extended_experiments, "run_pca_experiments", fake_pca)
    monkeypatch.setattr(
        run_extended_experiments,
        "run_random_forest_experiment",
        fake_random_forest,
    )
    monkeypatch.setattr(run_extended_experiments, "evaluate_mlp_cv", fake_mlp)
    monkeypatch.setattr(
        run_extended_experiments, "save_experiment_metrics", lambda results: Path("metrics.csv")
    )
    monkeypatch.setattr(
        run_extended_experiments, "save_mlp_summary", lambda summary: Path("mlp.json")
    )
    monkeypatch.setattr(
        run_extended_experiments, "save_pca_comparison", lambda results: Path("pca.png")
    )
    monkeypatch.setattr(
        run_extended_experiments,
        "save_pca_train_projection",
        lambda evaluated_features, evaluated_target: observed_indices.append(
            evaluated_features.index
        ),
    )
    monkeypatch.setattr(
        run_extended_experiments, "save_model_comparison", lambda results: Path("models.png")
    )

    run_extended_experiments.main()

    assert observed_indices
    assert all(index.equals(features_train.index) for index in observed_indices)
    assert all(set(index).isdisjoint(features_test.index) for index in observed_indices)
