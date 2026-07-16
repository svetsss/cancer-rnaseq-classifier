import numpy as np
import pandas as pd

from src.robustness import evaluate_e10_repeated_cv


def test_repeated_cv_reports_all_folds_without_test_data() -> None:
    random = np.random.default_rng(42)
    features = pd.DataFrame(random.normal(size=(60, 25)))
    target = pd.Series(["A", "B", "C"] * 20, dtype="string")
    for class_index, label in enumerate(("A", "B", "C")):
        mask = target == label
        features.loc[mask, class_index * 3 : class_index * 3 + 2] += 5.0

    summary = evaluate_e10_repeated_cv(features, target, n_splits=3, n_repeats=2)

    assert summary["evaluation_scope"].startswith("training subset only")
    assert len(summary["fold_f1_macro"]) == 6
    assert len(summary["fold_accuracy"]) == 6
    assert summary["evaluations"] == 6
    assert summary["f1_macro_min"] == min(summary["fold_f1_macro"])
    assert summary["f1_macro_max"] == max(summary["fold_f1_macro"])
    assert 0.0 <= summary["f1_macro_mean"] <= 1.0
