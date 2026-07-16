import json
from pathlib import Path

import pandas as pd
import pytest

from src.statistical_context import wilson_interval


def test_statistical_context_matches_frozen_predictions() -> None:
    project_root = Path(__file__).resolve().parents[1]
    predictions = pd.read_csv(project_root / "results" / "final_test_predictions.csv")
    context = json.loads(
        (project_root / "results" / "statistical_context.json").read_text(encoding="utf-8")
    )

    correct = int(predictions["correct"].astype(str).str.lower().eq("true").sum())
    expected_accuracy = wilson_interval(correct, len(predictions))
    assert context["accuracy"]["lower"] == pytest.approx(expected_accuracy["lower"])

    for label, group in predictions.groupby("true_label"):
        successes = int((group["true_label"] == group["predicted_label"]).sum())
        expected = wilson_interval(successes, len(group))
        assert context["per_class_recall"][label]["lower"] == pytest.approx(expected["lower"])
