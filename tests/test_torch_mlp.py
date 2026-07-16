from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.neural_evaluation import evaluate_mlp_cv, prepare_mlp_features
from src.torch_mlp import CancerMLP, predict_mlp, train_mlp


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    random = np.random.default_rng(42)
    features_train = random.normal(size=(24, 4)).astype(np.float32)
    target_train = np.array([0, 1] * 12, dtype=np.int64)
    features_validation = random.normal(size=(8, 4)).astype(np.float32)
    target_validation = np.array([0, 1] * 4, dtype=np.int64)
    return features_train, target_train, features_validation, target_validation


def _dataset() -> tuple[pd.DataFrame, pd.Series]:
    random = np.random.default_rng(42)
    features = pd.DataFrame(random.normal(size=(40, 6)))
    target = pd.Series(["A", "B"] * 20, dtype="string")
    return features, target


def _patch_fast_training(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_train_mlp(
        features_train: np.ndarray,
        target_train: np.ndarray,
        features_validation: np.ndarray,
        target_validation: np.ndarray,
        *,
        class_count: int,
        **parameters: object,
    ) -> tuple[CancerMLP, int]:
        return CancerMLP(features_train.shape[1], class_count), 2

    def fake_predict_mlp(model: CancerMLP, features: np.ndarray) -> np.ndarray:
        return np.zeros(len(features), dtype=np.int64)

    monkeypatch.setattr("src.neural_evaluation.train_mlp", fake_train_mlp)
    monkeypatch.setattr("src.neural_evaluation.predict_mlp", fake_predict_mlp)


def test_mlp_layer_dimensions() -> None:
    model = CancerMLP()
    linear_layers = [layer for layer in model.network if isinstance(layer, nn.Linear)]

    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
        (200, 128),
        (128, 64),
        (64, 5),
    ]


def test_mlp_output_has_five_classes() -> None:
    model = CancerMLP()

    output = model(torch.zeros((3, 200)))

    assert output.shape == (3, 5)


def test_mlp_training_loop_runs() -> None:
    arrays = _arrays()

    model, epochs = train_mlp(*arrays, class_count=2, max_epochs=2, patience=2, batch_size=8)

    assert epochs == 2
    assert predict_mlp(model, arrays[2]).shape == (8,)


def test_mlp_early_stopping_finishes_training() -> None:
    arrays = _arrays()

    _, epochs = train_mlp(
        *arrays,
        class_count=2,
        max_epochs=10,
        patience=1,
        min_delta=1e6,
        batch_size=8,
    )

    assert epochs == 2


def test_outer_validation_is_not_used_for_early_stopping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, target = _dataset()
    validation_sizes: list[int] = []

    def recording_train_mlp(
        features_train: np.ndarray,
        target_train: np.ndarray,
        features_validation: np.ndarray,
        target_validation: np.ndarray,
        *,
        class_count: int,
        **parameters: object,
    ) -> tuple[CancerMLP, int]:
        validation_sizes.append(len(features_validation))
        return CancerMLP(features_train.shape[1], class_count), 1

    monkeypatch.setattr("src.neural_evaluation.train_mlp", recording_train_mlp)
    monkeypatch.setattr(
        "src.neural_evaluation.predict_mlp",
        lambda model, evaluated_features: np.zeros(len(evaluated_features), dtype=np.int64),
    )

    evaluate_mlp_cv(features, target, n_features=4, n_splits=2, max_epochs=1)

    assert validation_sizes == [3, 3]
    assert all(size < 20 for size in validation_sizes)


def test_mlp_preprocessing_fits_only_inner_train(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    random = np.random.default_rng(42)
    features_inner_train = pd.DataFrame(random.normal(size=(12, 6)))
    target_inner_train = pd.Series(["A", "B"] * 6, dtype="string")
    features_inner_validation = pd.DataFrame(random.normal(size=(4, 6)))
    features_outer_validation = pd.DataFrame(random.normal(size=(5, 6)))
    fitted_row_counts: list[int] = []
    original_selector_fit_transform = SelectKBest.fit_transform
    original_scaler_fit_transform = StandardScaler.fit_transform

    def selector_fit_transform(
        selector: SelectKBest,
        features: pd.DataFrame,
        target: pd.Series,
        **fit_params: object,
    ) -> np.ndarray:
        fitted_row_counts.append(len(features))
        return original_selector_fit_transform(selector, features, target, **fit_params)

    def scaler_fit_transform(
        scaler: StandardScaler,
        features: np.ndarray,
        target: object = None,
        **fit_params: object,
    ) -> np.ndarray:
        fitted_row_counts.append(len(features))
        return original_scaler_fit_transform(scaler, features, target, **fit_params)

    monkeypatch.setattr("src.neural_evaluation.SelectKBest.fit_transform", selector_fit_transform)
    monkeypatch.setattr(
        "src.neural_evaluation.StandardScaler.fit_transform",
        scaler_fit_transform,
    )

    prepare_mlp_features(
        features_inner_train,
        target_inner_train,
        features_inner_validation,
        features_outer_validation,
        n_features=3,
    )

    assert fitted_row_counts == [12, 12]


def test_mlp_training_is_reproducible() -> None:
    arrays = _arrays()

    first_model, first_epochs = train_mlp(
        *arrays,
        class_count=2,
        max_epochs=2,
        patience=2,
        batch_size=8,
        seed=42,
    )
    second_model, second_epochs = train_mlp(
        *arrays,
        class_count=2,
        max_epochs=2,
        patience=2,
        batch_size=8,
        seed=42,
    )

    assert first_epochs == second_epochs
    assert all(
        torch.equal(first_model.state_dict()[name], second_model.state_dict()[name])
        for name in first_model.state_dict()
    )


def test_mlp_fold_metrics_are_aggregated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, target = _dataset()
    _patch_fast_training(monkeypatch)
    f1_values = iter([0.2, 0.4])
    accuracy_values = iter([0.3, 0.5])
    precision_values = iter([0.4, 0.6])
    recall_values = iter([0.5, 0.7])
    monkeypatch.setattr("src.neural_evaluation.f1_score", lambda *args, **kwargs: next(f1_values))
    monkeypatch.setattr(
        "src.neural_evaluation.accuracy_score",
        lambda *args, **kwargs: next(accuracy_values),
    )
    monkeypatch.setattr(
        "src.neural_evaluation.precision_score",
        lambda *args, **kwargs: next(precision_values),
    )
    monkeypatch.setattr(
        "src.neural_evaluation.recall_score",
        lambda *args, **kwargs: next(recall_values),
    )

    result, _ = evaluate_mlp_cv(features, target, n_features=4, n_splits=2, max_epochs=1)

    assert result["cv_f1_macro_mean"] == pytest.approx(0.3)
    assert result["cv_f1_macro_std"] == pytest.approx(0.1)
    assert result["cv_accuracy_mean"] == pytest.approx(0.4)
    assert result["cv_precision_macro_mean"] == pytest.approx(0.5)
    assert result["cv_recall_macro_mean"] == pytest.approx(0.6)


def test_mlp_summary_has_required_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    features, target = _dataset()
    _patch_fast_training(monkeypatch)

    _, summary = evaluate_mlp_cv(features, target, n_features=4, n_splits=2, max_epochs=1)

    assert set(summary) == {
        "experiment_id",
        "architecture",
        "optimizer",
        "learning_rate",
        "batch_size",
        "max_epochs",
        "patience",
        "inner_validation_size",
        "fold_epochs",
        "mean_epochs",
        "device",
        "random_state",
    }


def test_mlp_evaluation_does_not_save_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, target = _dataset()
    _patch_fast_training(monkeypatch)
    monkeypatch.chdir(tmp_path)

    evaluate_mlp_cv(features, target, n_features=4, n_splits=2, max_epochs=1)

    assert not list(tmp_path.rglob("*.pt"))
    assert not list(tmp_path.rglob("*.pth"))


def test_mlp_runs_on_cpu() -> None:
    arrays = _arrays()

    model, _ = train_mlp(*arrays, class_count=2, max_epochs=1, patience=1, batch_size=8)

    assert {parameter.device.type for parameter in model.parameters()} == {"cpu"}
