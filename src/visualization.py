from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from src.config import CLASS_DISTRIBUTION_PATH


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
