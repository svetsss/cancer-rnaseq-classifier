import json
from pathlib import Path
from time import perf_counter
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import CV_SPLITS, MLP_TRAINING_ANALYSIS_PATH, RANDOM_STATE
from src.evaluation import ExperimentResult
from src.feature_selection import FEATURE_METHOD
from src.splitting import make_stratified_cv
from src.torch_mlp import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_DELTA,
    PATIENCE,
    TrainingEpoch,
    predict_mlp,
    train_mlp,
    train_mlp_fixed_epochs,
)

INNER_VALIDATION_SIZE = 0.15


class MLPSummary(TypedDict):
    """Training protocol and fold epoch counts for E17."""

    experiment_id: str
    architecture: list[str]
    optimizer: str
    learning_rate: float
    batch_size: int
    max_epochs: int
    patience: int
    inner_validation_size: float
    fold_epochs: list[int]
    early_stopping_epochs_run: list[int]
    fold_histories: list[list[TrainingEpoch]]
    fold_metrics: list[dict[str, float | int]]
    mean_epochs: float
    refit_on_full_outer_train: bool
    device: str
    random_state: int


def prepare_mlp_features(
    features_inner_train: pd.DataFrame,
    target_inner_train: pd.Series,
    features_inner_validation: pd.DataFrame,
    features_outer_validation: pd.DataFrame,
    *,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit feature selection and scaling only on inner-train data."""
    selector = SelectKBest(score_func=f_classif, k=n_features)
    selected_inner_train = selector.fit_transform(features_inner_train, target_inner_train)
    selected_inner_validation = selector.transform(features_inner_validation)
    selected_outer_validation = selector.transform(features_outer_validation)

    scaler = StandardScaler()
    scaled_inner_train = scaler.fit_transform(selected_inner_train)
    scaled_inner_validation = scaler.transform(selected_inner_validation)
    scaled_outer_validation = scaler.transform(selected_outer_validation)
    return (
        np.asarray(scaled_inner_train, dtype=np.float32),
        np.asarray(scaled_inner_validation, dtype=np.float32),
        np.asarray(scaled_outer_validation, dtype=np.float32),
    )


def prepare_mlp_refit_features(
    features_outer_train: pd.DataFrame,
    target_outer_train: pd.Series,
    features_outer_validation: pd.DataFrame,
    *,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Refit selector and scaler on all outer-training rows after epoch selection."""
    selector = SelectKBest(score_func=f_classif, k=n_features)
    selected_outer_train = selector.fit_transform(features_outer_train, target_outer_train)
    selected_outer_validation = selector.transform(features_outer_validation)
    scaler = StandardScaler()
    scaled_outer_train = scaler.fit_transform(selected_outer_train)
    scaled_outer_validation = scaler.transform(selected_outer_validation)
    return (
        np.asarray(scaled_outer_train, dtype=np.float32),
        np.asarray(scaled_outer_validation, dtype=np.float32),
    )


def evaluate_mlp_cv(
    features_train: pd.DataFrame,
    target_train: pd.Series,
    *,
    n_features: int = 200,
    n_splits: int = CV_SPLITS,
    random_state: int = RANDOM_STATE,
    inner_validation_size: float = INNER_VALIDATION_SIZE,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    min_delta: float = MIN_DELTA,
) -> tuple[ExperimentResult, MLPSummary]:
    """Evaluate E17 with outer CV and inner validation for early stopping."""
    labels = sorted(str(label) for label in target_train.unique())
    label_indices = {label: index for index, label in enumerate(labels)}
    encoded_target = target_train.astype(str).map(label_indices).astype(int)
    cross_validator = make_stratified_cv(n_splits=n_splits, random_state=random_state)

    f1_scores: list[float] = []
    accuracy_scores: list[float] = []
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    fit_times: list[float] = []
    score_times: list[float] = []
    fold_epochs: list[int] = []
    early_stopping_epochs_run: list[int] = []
    fold_histories: list[list[TrainingEpoch]] = []
    fold_metrics: list[dict[str, float | int]] = []
    started_at = perf_counter()

    for fold_index, (outer_train_indices, outer_validation_indices) in enumerate(
        cross_validator.split(features_train, target_train)
    ):
        features_outer_train = features_train.iloc[outer_train_indices]
        target_outer_train = encoded_target.iloc[outer_train_indices]
        features_outer_validation = features_train.iloc[outer_validation_indices]
        target_outer_validation = encoded_target.iloc[outer_validation_indices]

        (
            features_inner_train,
            features_inner_validation,
            target_inner_train,
            target_inner_validation,
        ) = train_test_split(
            features_outer_train,
            target_outer_train,
            test_size=inner_validation_size,
            random_state=random_state,
            stratify=target_outer_train,
        )

        fit_started_at = perf_counter()
        (
            transformed_inner_train,
            transformed_inner_validation,
            transformed_outer_validation,
        ) = prepare_mlp_features(
            features_inner_train,
            target_inner_train,
            features_inner_validation,
            features_outer_validation,
            n_features=n_features,
        )
        selection_model, epochs_run = train_mlp(
            transformed_inner_train,
            target_inner_train.to_numpy(dtype=np.int64),
            transformed_inner_validation,
            target_inner_validation.to_numpy(dtype=np.int64),
            class_count=len(labels),
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            seed=random_state + fold_index,
        )
        selected_epochs = selection_model.best_epoch or epochs_run
        transformed_outer_train, transformed_outer_validation = prepare_mlp_refit_features(
            features_outer_train,
            target_outer_train,
            features_outer_validation,
            n_features=n_features,
        )
        model = train_mlp_fixed_epochs(
            transformed_outer_train,
            target_outer_train.to_numpy(dtype=np.int64),
            class_count=len(labels),
            epochs=selected_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seed=random_state + fold_index,
        )
        fit_times.append(perf_counter() - fit_started_at)
        early_stopping_epochs_run.append(epochs_run)
        fold_epochs.append(selected_epochs)
        fold_histories.append(selection_model.training_history)

        score_started_at = perf_counter()
        predicted_target = predict_mlp(model, transformed_outer_validation)
        score_times.append(perf_counter() - score_started_at)
        expected_target = target_outer_validation.to_numpy(dtype=np.int64)
        fold_f1 = float(f1_score(expected_target, predicted_target, average="macro"))
        fold_accuracy = float(accuracy_score(expected_target, predicted_target))
        fold_precision = float(
            precision_score(
                expected_target,
                predicted_target,
                average="macro",
                zero_division=0,
            )
        )
        fold_recall = float(recall_score(expected_target, predicted_target, average="macro"))
        f1_scores.append(fold_f1)
        accuracy_scores.append(fold_accuracy)
        precision_scores.append(fold_precision)
        recall_scores.append(fold_recall)
        fold_metrics.append(
            {
                "fold": fold_index + 1,
                "selected_epochs": selected_epochs,
                "epochs_run": epochs_run,
                "f1_macro": fold_f1,
                "accuracy": fold_accuracy,
                "precision_macro": fold_precision,
                "recall_macro": fold_recall,
            }
        )

    result: ExperimentResult = {
        "experiment_id": "E17",
        "model": "PyTorchMLP",
        "feature_method": FEATURE_METHOD,
        "n_features": n_features,
        "cv_f1_macro_mean": float(np.mean(f1_scores)),
        "cv_f1_macro_std": float(np.std(f1_scores)),
        "cv_accuracy_mean": float(np.mean(accuracy_scores)),
        "cv_accuracy_std": float(np.std(accuracy_scores)),
        "cv_precision_macro_mean": float(np.mean(precision_scores)),
        "cv_precision_macro_std": float(np.std(precision_scores)),
        "cv_recall_macro_mean": float(np.mean(recall_scores)),
        "cv_recall_macro_std": float(np.std(recall_scores)),
        "cv_fit_time_mean": float(np.mean(fit_times)),
        "cv_score_time_mean": float(np.mean(score_times)),
        "total_cv_time_seconds": perf_counter() - started_at,
        "random_state": random_state,
        "cv_splits": cross_validator.n_splits,
    }
    summary: MLPSummary = {
        "experiment_id": "E17",
        "architecture": [
            f"Linear({n_features}, 128)",
            "ReLU",
            "Dropout(0.3)",
            "Linear(128, 64)",
            "ReLU",
            "Dropout(0.2)",
            f"Linear(64, {len(labels)})",
        ],
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "inner_validation_size": inner_validation_size,
        "fold_epochs": fold_epochs,
        "early_stopping_epochs_run": early_stopping_epochs_run,
        "fold_histories": fold_histories,
        "fold_metrics": fold_metrics,
        "mean_epochs": float(np.mean(fold_epochs)),
        "refit_on_full_outer_train": True,
        "device": "cpu",
        "random_state": random_state,
    }
    return result, summary


def save_mlp_summary(
    summary: MLPSummary,
    output_path: Path = MLP_TRAINING_ANALYSIS_PATH,
) -> Path:
    """Atomically replace the E17 training summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.json")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    temporary_path.replace(output_path)
    return output_path
