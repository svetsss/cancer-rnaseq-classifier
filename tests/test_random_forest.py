from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest

from src.random_forest_experiments import (
    RANDOM_FOREST_EXPERIMENT,
    build_random_forest_pipeline,
)


def test_random_forest_pipeline_steps() -> None:
    pipeline = build_random_forest_pipeline()

    assert list(pipeline.named_steps) == ["selector", "classifier"]


def test_random_forest_selector_uses_200_features() -> None:
    pipeline = build_random_forest_pipeline()
    selector = pipeline.named_steps["selector"]

    assert isinstance(selector, SelectKBest)
    assert selector.k == 200


def test_random_forest_classifier_type() -> None:
    pipeline = build_random_forest_pipeline()

    assert isinstance(pipeline.named_steps["classifier"], RandomForestClassifier)


def test_random_forest_tree_count() -> None:
    pipeline = build_random_forest_pipeline()

    assert pipeline.named_steps["classifier"].n_estimators == 300


def test_random_forest_random_state() -> None:
    pipeline = build_random_forest_pipeline()

    assert pipeline.named_steps["classifier"].random_state == 42


def test_random_forest_experiment_id() -> None:
    assert RANDOM_FOREST_EXPERIMENT == ("E16", "RandomForestClassifier", 200)


def test_random_forest_pipeline_has_no_scaler() -> None:
    pipeline = build_random_forest_pipeline()

    assert "scaler" not in pipeline.named_steps
