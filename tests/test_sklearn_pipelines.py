from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.sklearn_pipelines import build_dummy_classifier, build_logistic_pipeline


def test_dummy_classifier_configuration() -> None:
    classifier = build_dummy_classifier()

    assert isinstance(classifier, DummyClassifier)
    assert classifier.strategy == "most_frequent"
    assert classifier.random_state == 42


def test_logistic_baseline_is_pipeline() -> None:
    pipeline = build_logistic_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)
    assert pipeline.named_steps["classifier"].max_iter == 3000
    assert pipeline.named_steps["classifier"].random_state == 42


def test_scaler_precedes_logistic_classifier() -> None:
    pipeline = build_logistic_pipeline()

    assert list(pipeline.named_steps) == ["scaler", "classifier"]
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)


def test_logistic_pipeline_has_no_feature_selection() -> None:
    pipeline = build_logistic_pipeline()

    assert len(pipeline.steps) == 2
    assert all("select" not in step_name for step_name, _ in pipeline.steps)
