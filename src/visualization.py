from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from src.config import CLASS_DISTRIBUTION_PATH, FEATURE_SELECTION_COMPARISON_PATH
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
