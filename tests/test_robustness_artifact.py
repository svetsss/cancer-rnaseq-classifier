import json
from pathlib import Path

import numpy as np
import pytest


def test_published_robustness_artifacts_are_consistent() -> None:
    summary_path = Path("results/robustness_e10.json")
    figure_path = Path("figures/e10_robustness.png")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    f1_values = np.asarray(summary["fold_f1_macro"], dtype=float)

    assert summary["evaluation_scope"].startswith("training subset only")
    assert summary["evaluations"] == summary["n_splits"] * summary["n_repeats"] == 50
    assert len(f1_values) == summary["evaluations"]
    assert summary["f1_macro_mean"] == pytest.approx(float(f1_values.mean()))
    assert summary["f1_macro_std"] == pytest.approx(float(f1_values.std()))
    assert summary["f1_macro_min"] == pytest.approx(float(f1_values.min()))
    assert summary["f1_macro_max"] == pytest.approx(float(f1_values.max()))
    assert figure_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
