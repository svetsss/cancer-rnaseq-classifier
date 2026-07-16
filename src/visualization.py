from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import (
    CLASS_DISTRIBUTION_PATH,
    FEATURE_SELECTION_COMPARISON_PATH,
    MODEL_COMPARISON_PATH,
    PCA_COMPARISON_PATH,
    PCA_TRAIN_PROJECTION_PATH,
)
from src.evaluation import ExperimentResult


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


def _save_figure(figure: Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.png")
    figure.savefig(temporary_path, dpi=200)
    temporary_path.replace(output_path)
    return output_path
