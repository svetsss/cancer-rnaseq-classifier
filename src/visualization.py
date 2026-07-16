from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import (
    CLASS_DISTRIBUTION_PATH,
    FEATURE_SELECTION_COMPARISON_PATH,
    MODEL_COMPARISON_PATH,
    PCA_COMPARISON_PATH,
    PCA_TRAIN_PROJECTION_PATH,
    PROJECT_PIPELINE_PATH,
)
from src.evaluation import ExperimentResult

if TYPE_CHECKING:
    from src.neural_evaluation import MLPSummary
    from src.robustness import RobustnessSummary
    from src.torch_mlp import TrainingEpoch

EpochMetric = Literal[
    "train_loss",
    "validation_loss",
    "validation_f1_macro",
    "validation_accuracy",
]


def save_project_pipeline(output_path: Path = PROJECT_PIPELINE_PATH) -> Path:
    """Save a compact diagram of the complete evaluation protocol."""
    stages = (
        ("Данные", "801 образец\n20 531 признак"),
        ("Разделение", "Train: 640\nTest: 161"),
        ("Train CV", "18 конфигураций\n5 folds"),
        ("Выбор E10", "Scaler → PCA(20)\n→ LogisticRegression"),
        ("Финальная оценка", "161 / 161 верно\n95% ДИ Wilson\n0.9767-1.0000"),
    )
    positions = np.asarray([0.0, 2.0, 4.1, 6.5, 9.1])
    half_widths = np.asarray([0.75, 0.75, 0.82, 1.15, 1.1])
    figure = Figure(figsize=(13, 3.5), facecolor="#f7f9fc")
    axes = figure.subplots()
    axes.set_facecolor("#f7f9fc")

    for index, ((title, details), position) in enumerate(zip(stages, positions, strict=True)):
        selected = index == 3
        final = index == 4
        facecolor = "#dcefe8" if selected else "#eee8f6" if final else "#e8eef7"
        edgecolor = "#16816d" if selected else "#7258a5" if final else "#6f87a6"
        axes.text(
            position,
            0.5,
            f"{title}\n\n{details}",
            ha="center",
            va="center",
            fontsize=10,
            color="#172033",
            bbox={
                "boxstyle": "round,pad=0.8",
                "facecolor": facecolor,
                "edgecolor": edgecolor,
                "linewidth": 2 if selected or final else 1.2,
            },
        )
        if index < len(stages) - 1:
            axes.annotate(
                "",
                xy=(positions[index + 1] - half_widths[index + 1] - 0.08, 0.5),
                xytext=(position + half_widths[index] + 0.08, 0.5),
                arrowprops={"arrowstyle": "->", "color": "#4f5b66", "linewidth": 1.5},
            )

    axes.set_title("Схема эксперимента", fontsize=16, fontweight="bold", color="#172033")
    axes.set_xlim(-1.1, 10.5)
    axes.set_ylim(0, 1)
    axes.axis("off")
    figure.text(
        0.5,
        0.035,
        "Preprocessing обучается внутри train folds; test используется один раз.",
        ha="center",
        color="#52606d",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    return _save_figure(figure, output_path)


def save_class_distribution(target: pd.Series, output_path: Path = CLASS_DISTRIBUTION_PATH) -> Path:
    """Save a report-ready class distribution chart."""
    counts = target.value_counts().sort_index()
    figure = Figure(figsize=(8, 5))
    axes = figure.subplots()
    bars = axes.bar(counts.index.astype(str), counts.to_numpy(), color="#386cb0")
    axes.bar_label(bars, padding=3)
    axes.set_title("Распределение классов опухолей")
    axes.set_xlabel("Класс")
    axes.set_ylabel("Число образцов")
    axes.set_ylim(0, max(counts) * 1.12)
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.png")
    figure.savefig(temporary_path, dpi=180)
    temporary_path.replace(output_path)
    return output_path


def save_feature_selection_comparison(
    results: Sequence[ExperimentResult],
    output_path: Path = FEATURE_SELECTION_COMPARISON_PATH,
) -> Path:
    """Save the CV macro F1 comparison for the feature-selection experiments."""
    figure = Figure(figsize=(8, 5))
    axes = figure.subplots()

    model_styles = (
        ("LogisticRegression", "Logistic Regression", "#386cb0"),
        ("LinearSVC", "LinearSVC", "#e6550d"),
    )
    for model_name, label, color in model_styles:
        model_results = sorted(
            (result for result in results if result["model"] == model_name),
            key=lambda result: result["n_features"],
        )
        axes.errorbar(
            [result["n_features"] for result in model_results],
            [result["cv_f1_macro_mean"] for result in model_results],
            yerr=[result["cv_f1_macro_std"] for result in model_results],
            marker="o",
            linewidth=1.8,
            capsize=4,
            color=color,
            label=label,
        )

    axes.set_title("SelectKBest: сравнение моделей")
    axes.set_xlabel("Число выбранных признаков")
    axes.set_ylabel("Средний CV macro F1")
    axes.set_xticks([50, 100, 200, 500])
    axes.grid(alpha=0.3)
    axes.legend()
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.png")
    figure.savefig(temporary_path, dpi=200)
    temporary_path.replace(output_path)
    return output_path


def save_pca_comparison(
    results: Sequence[ExperimentResult],
    output_path: Path = PCA_COMPARISON_PATH,
) -> Path:
    """Save the CV macro F1 comparison for PCA experiments."""
    figure = Figure(figsize=(8, 5))
    axes = figure.subplots()
    model_styles = (
        ("LogisticRegression", "Logistic Regression", "#386cb0"),
        ("LinearSVC", "LinearSVC", "#e6550d"),
    )

    for model_name, label, color in model_styles:
        model_results = sorted(
            (result for result in results if result["model"] == model_name),
            key=lambda result: result["n_features"],
        )
        axes.errorbar(
            [result["n_features"] for result in model_results],
            [result["cv_f1_macro_mean"] for result in model_results],
            yerr=[result["cv_f1_macro_std"] for result in model_results],
            marker="o",
            linewidth=1.8,
            capsize=4,
            color=color,
            label=label,
        )

    axes.set_title("PCA: сравнение моделей")
    axes.set_xlabel("Число главных компонент")
    axes.set_ylabel("Средний CV macro F1")
    axes.set_xticks([20, 50, 100])
    axes.grid(alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return _save_figure(figure, output_path)


def save_pca_train_projection(
    features_train: pd.DataFrame,
    target_train: pd.Series,
    output_path: Path = PCA_TRAIN_PROJECTION_PATH,
) -> Path:
    """Project only the training subset onto its first two principal components."""
    scaled_features = StandardScaler().fit_transform(features_train)
    projection = PCA(n_components=2, random_state=42).fit_transform(scaled_features)
    colors = ("#386cb0", "#e6550d", "#31a354", "#756bb1", "#636363")

    figure = Figure(figsize=(8, 6))
    axes = figure.subplots()
    for index, label in enumerate(sorted(target_train.unique())):
        class_mask = target_train.to_numpy() == label
        axes.scatter(
            projection[class_mask, 0],
            projection[class_mask, 1],
            s=24,
            alpha=0.75,
            color=colors[index],
            label=str(label),
        )

    axes.set_title("PCA-проекция обучающей выборки")
    axes.set_xlabel("Первая главная компонента")
    axes.set_ylabel("Вторая главная компонента")
    axes.grid(alpha=0.2)
    axes.legend(title="Класс")
    figure.tight_layout()
    return _save_figure(figure, output_path)


def save_model_comparison(
    results: Sequence[ExperimentResult],
    output_path: Path = MODEL_COMPARISON_PATH,
) -> Path:
    """Save macro F1 means and standard deviations for selected candidates."""
    labels = {
        "E01": "E01\nLogReg\n20 531",
        "E05": "E05\nLogReg\nSKB 500",
        "E08": "E08\nLinearSVC\nSKB 200",
        "E10": "E10\nLogReg\nPCA 20",
        "E11": "E11\nLogReg\nPCA 50",
        "E12": "E12\nLogReg\nPCA 100",
        "E13": "E13\nLinearSVC\nPCA 20",
        "E14": "E14\nLinearSVC\nPCA 50",
        "E15": "E15\nLinearSVC\nPCA 100",
        "E16": "E16\nRandom Forest\nSKB 200",
        "E17": "E17\nPyTorch MLP\nSKB 200",
    }
    figure = Figure(figsize=(11, 6))
    axes = figure.subplots()
    positions = list(range(len(results)))
    bars = axes.bar(
        positions,
        [result["cv_f1_macro_mean"] for result in results],
        yerr=[result["cv_f1_macro_std"] for result in results],
        capsize=4,
        color="#386cb0",
        alpha=0.85,
    )
    axes.set_title("Сравнение кандидатов по train CV")
    axes.set_ylabel("Средний CV macro F1")
    axes.set_xticks(positions, [labels[result["experiment_id"]] for result in results])
    axes.grid(axis="y", alpha=0.3)
    axes.bar_label(bars, fmt="%.4f", padding=3)
    figure.text(
        0.5,
        0.01,
        "E17: внешний CV, внутренняя validation-часть для early stopping",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    return _save_figure(figure, output_path)


def save_readme_model_comparison(metrics: pd.DataFrame, output_path: Path) -> Path:
    """Save a focused comparison of the main candidates for the project README."""
    labels = {
        "E01": "E01  LogisticRegression",
        "E08": "E08  LinearSVC + SelectKBest",
        "E10": "E10  LogisticRegression + PCA",
        "E16": "E16  Random Forest",
        "E17": "E17  PyTorch MLP",
    }
    selected = metrics.loc[metrics["experiment_id"].isin(labels)].copy()
    selected["order"] = selected["experiment_id"].map(
        {experiment_id: index for index, experiment_id in enumerate(labels)}
    )
    selected = selected.sort_values("order", ascending=False)
    if len(selected) != len(labels):
        raise ValueError("Metrics must contain E01, E08, E10, E16 and E17")

    figure = Figure(figsize=(10, 5.8), facecolor="#f7f9fc")
    axes = figure.subplots()
    axes.set_facecolor("#ffffff")
    positions = np.arange(len(selected))
    means = selected["cv_f1_macro_mean"].to_numpy(dtype=float)
    deviations = selected["cv_f1_macro_std"].to_numpy(dtype=float)
    colors = [
        "#7258a5" if experiment_id == "E10" else "#2f67b1"
        for experiment_id in selected["experiment_id"]
    ]
    for position, mean, deviation, color in zip(positions, means, deviations, colors, strict=True):
        axes.errorbar(
            mean,
            position,
            xerr=deviation,
            fmt="o",
            markersize=9,
            capsize=5,
            linewidth=2,
            color=color,
        )
        axes.text(
            mean - deviation - 0.0004,
            position + 0.2,
            f"{mean:.4f} ± {deviation:.4f}",
            ha="right",
            va="center",
            fontsize=9,
            color="#334155",
        )

    axes.axvline(1.0, color="#9aa6b2", linewidth=1, linestyle="--")
    axes.set_yticks(
        positions,
        [labels[experiment_id] for experiment_id in selected["experiment_id"]],
    )
    axes.set_xlim(0.986, 1.0015)
    axes.set_xlabel("Средний CV macro F1")
    axes.set_title("Сравнение основных моделей", fontsize=16, fontweight="bold")
    axes.grid(axis="x", color="#d9e2ec", alpha=0.8)
    axes.spines[["top", "right", "left"]].set_visible(False)
    figure.text(
        0.5,
        0.018,
        "Точка показывает среднее по 5 folds, горизонтальный отрезок — стандартное отклонение.",
        ha="center",
        color="#52606d",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    return _save_figure(figure, output_path)


def save_mlp_loss_curves(summary: MLPSummary, output_path: Path) -> Path:
    """Compare train and validation loss in one focused figure."""
    histories, _ = _validated_mlp_data(summary)
    epochs = np.arange(1, max(len(history) for history in histories) + 1)
    figure, axes = _training_axes("Динамика функции потерь")
    _plot_epoch_band(
        axes,
        epochs,
        _epoch_metric_matrix(histories, "train_loss"),
        label="Train loss",
        color="#2563a6",
    )
    _plot_epoch_band(
        axes,
        epochs,
        _epoch_metric_matrix(histories, "validation_loss"),
        label="Validation loss",
        color="#e76f51",
    )
    axes.set_xlabel("Эпоха")
    axes.set_ylabel("Cross-entropy loss")
    axes.legend(frameon=False)
    _finish_training_figure(
        figure,
        "Линии показывают среднее по folds, заливка показывает диапазон от min до max.",
    )
    return _save_figure(figure, output_path)


def save_mlp_validation_f1(summary: MLPSummary, output_path: Path) -> Path:
    """Show validation macro F1 by epoch in one focused figure."""
    histories, _ = _validated_mlp_data(summary)
    epochs = np.arange(1, max(len(history) for history in histories) + 1)
    values = _epoch_metric_matrix(histories, "validation_f1_macro")
    figure, axes = _training_axes("Macro F1 на inner-validation")
    _plot_epoch_band(
        axes,
        epochs,
        values,
        label="Validation macro F1",
        color="#178f7a",
    )
    axes.set_xlabel("Эпоха")
    axes.set_ylabel("Macro F1")
    axes.set_ylim(max(0.0, float(np.nanmin(values)) - 0.02), 1.005)
    axes.legend(frameon=False)
    _finish_training_figure(
        figure,
        "Validation macro F1 рассчитан после каждой эпохи; заливка показывает разброс folds.",
    )
    return _save_figure(figure, output_path)


def save_mlp_epoch_selection(summary: MLPSummary, output_path: Path) -> Path:
    """Compare the selected and stopping epochs for each outer fold."""
    _, fold_metrics = _validated_mlp_data(summary)
    positions = np.arange(len(fold_metrics))
    width = 0.36
    figure, axes = _training_axes("Early stopping по внешним folds")
    selected_bars = axes.bar(
        positions - width / 2,
        [int(metric["selected_epochs"]) for metric in fold_metrics],
        width,
        label="Выбранная эпоха",
        color="#2563a6",
    )
    stopped_bars = axes.bar(
        positions + width / 2,
        [int(metric["epochs_run"]) for metric in fold_metrics],
        width,
        label="Остановка обучения",
        color="#f0b44d",
    )
    axes.bar_label(selected_bars, padding=3)
    axes.bar_label(stopped_bars, padding=3)
    axes.set_xlabel("Внешний fold")
    axes.set_ylabel("Число эпох")
    axes.set_xticks(positions, [str(int(metric["fold"])) for metric in fold_metrics])
    axes.legend(frameon=False)
    _finish_training_figure(
        figure,
        "Синяя колонка отмечает минимум validation loss, жёлтая — остановку после patience = 10.",
    )
    return _save_figure(figure, output_path)


def save_mlp_fold_metrics(summary: MLPSummary, output_path: Path) -> Path:
    """Compare macro F1 and accuracy across outer folds."""
    _, fold_metrics = _validated_mlp_data(summary)
    positions = np.arange(len(fold_metrics))
    width = 0.36
    f1_values = [float(metric["f1_macro"]) for metric in fold_metrics]
    accuracy_values = [float(metric["accuracy"]) for metric in fold_metrics]
    figure, axes = _training_axes("Метрики MLP на outer-validation")
    f1_bars = axes.bar(
        positions - width / 2,
        f1_values,
        width,
        label="Macro F1",
        color="#178f7a",
    )
    accuracy_bars = axes.bar(
        positions + width / 2,
        accuracy_values,
        width,
        label="Accuracy",
        color="#7c62a3",
    )
    axes.bar_label(f1_bars, fmt="%.3f", padding=3)
    axes.bar_label(accuracy_bars, fmt="%.3f", padding=3)
    axes.set_xlabel("Внешний fold")
    axes.set_ylabel("Значение метрики")
    axes.set_ylim(max(0.0, min([*f1_values, *accuracy_values]) - 0.02), 1.005)
    axes.set_xticks(positions, [str(int(metric["fold"])) for metric in fold_metrics])
    axes.legend(frameon=False)
    _finish_training_figure(
        figure,
        "Метрики рассчитаны на outer-validation после refit модели на полном outer-train.",
    )
    return _save_figure(figure, output_path)


def save_robustness_f1(summary: RobustnessSummary, output_path: Path) -> Path:
    """Show E10 macro F1 across every validation fold of repeated CV."""
    values = np.asarray(summary["fold_f1_macro"], dtype=float)
    if values.size == 0:
        raise ValueError("Robustness summary must contain fold_f1_macro values")

    evaluations = np.arange(1, values.size + 1)
    mean_value = float(summary["f1_macro_mean"])
    figure, axes = _training_axes("Устойчивость E10 при repeated cross-validation")
    axes.scatter(
        evaluations,
        values,
        s=34,
        color="#2f67b1",
        alpha=0.8,
        label="Macro F1 отдельного fold",
    )
    axes.axhline(
        mean_value,
        color="#7258a5",
        linewidth=2,
        linestyle="--",
        label=f"Среднее: {mean_value:.4f}",
    )
    axes.set_xlabel("Номер validation fold")
    axes.set_ylabel("Macro F1")
    axes.set_xlim(0, values.size + 1)
    axes.set_ylim(max(0.0, float(values.min()) - 0.01), 1.005)
    axes.legend(frameon=False, loc="lower right")
    _finish_training_figure(
        figure,
        "Каждая точка — один validation fold; test-выборка в расчёте не использовалась.",
    )
    return _save_figure(figure, output_path)


def _validated_mlp_data(
    summary: MLPSummary,
) -> tuple[list[list[TrainingEpoch]], list[dict[str, float | int]]]:
    histories = summary["fold_histories"]
    fold_metrics = summary["fold_metrics"]
    if not histories or not all(histories):
        raise ValueError("MLP summary must contain a non-empty history for every fold")
    if len(histories) != len(fold_metrics):
        raise ValueError("MLP histories and fold metrics must have the same length")
    return histories, fold_metrics


def _training_axes(title: str) -> tuple[Figure, Axes]:
    figure = Figure(figsize=(9, 5.4), facecolor="#f7f9fc")
    axes = figure.subplots()
    axes.set_facecolor("#ffffff")
    axes.grid(axis="y", color="#d9e2ec", alpha=0.75, linewidth=0.8)
    axes.spines[["top", "right"]].set_visible(False)
    axes.set_title(title, fontsize=16, fontweight="bold", color="#172033")
    return figure, axes


def _finish_training_figure(figure: Figure, caption: str) -> None:
    figure.text(0.5, 0.018, caption, ha="center", color="#52606d", fontsize=9)
    figure.tight_layout(rect=(0, 0.06, 1, 1))


def _epoch_metric_matrix(
    histories: list[list[TrainingEpoch]],
    metric: EpochMetric,
) -> np.ndarray:
    matrix = np.full((len(histories), max(len(history) for history in histories)), np.nan)
    for row_index, history in enumerate(histories):
        matrix[row_index, : len(history)] = [float(epoch[metric]) for epoch in history]
    return matrix


def _plot_epoch_band(
    axes: Axes,
    epochs: np.ndarray,
    values: np.ndarray,
    *,
    label: str,
    color: str,
) -> None:
    mean_values = np.nanmean(values, axis=0)
    minimum_values = np.nanmin(values, axis=0)
    maximum_values = np.nanmax(values, axis=0)
    axes.plot(epochs, mean_values, color=color, linewidth=2.2, label=label)
    axes.fill_between(epochs, minimum_values, maximum_values, color=color, alpha=0.14)


def _save_figure(figure: Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.png")
    figure.savefig(temporary_path, dpi=200)
    temporary_path.replace(output_path)
    return output_path
