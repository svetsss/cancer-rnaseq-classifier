from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import src.app_services as app_services_module
from src.app_services import (
    PREDICTION_COLUMN,
    load_final_pipeline,
    predict_uploaded_features,
    validate_uploaded_features,
)


@pytest.fixture
def synthetic_pipeline() -> Pipeline:
    features = pd.DataFrame(
        {
            "gene_0": [0.0, 0.2, 1.0, 1.2, 2.0, 2.2],
            "gene_1": [0.1, 0.0, 1.1, 1.0, 2.1, 2.0],
            "gene_2": [0.2, 0.1, 1.2, 1.1, 2.2, 2.1],
        }
    )
    target = pd.Series(["BRCA", "BRCA", "COAD", "COAD", "KIRC", "KIRC"])
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipeline.fit(features, target)
    return pipeline


@pytest.fixture
def uploaded_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gene_0": [0.1, 2.1],
            "gene_1": [0.0, 2.0],
            "gene_2": [0.2, 2.2],
        },
        index=["sample_a", "sample_b"],
    )


def test_load_final_pipeline(tmp_path: Path, synthetic_pipeline: Pipeline) -> None:
    model_path = tmp_path / "pipeline.joblib"
    joblib.dump(synthetic_pipeline, model_path)

    loaded = load_final_pipeline(model_path)

    assert isinstance(loaded, Pipeline)
    assert list(loaded.feature_names_in_) == ["gene_0", "gene_1", "gene_2"]


def test_exact_feature_columns_are_accepted(
    synthetic_pipeline: Pipeline,
    uploaded_features: pd.DataFrame,
) -> None:
    validated = validate_uploaded_features(uploaded_features, synthetic_pipeline)

    pd.testing.assert_frame_equal(validated, uploaded_features)


def test_feature_columns_are_restored_to_model_order(
    synthetic_pipeline: Pipeline,
    uploaded_features: pd.DataFrame,
) -> None:
    reordered = uploaded_features[["gene_2", "gene_0", "gene_1"]]

    validated = validate_uploaded_features(reordered, synthetic_pipeline)

    assert list(validated.columns) == ["gene_0", "gene_1", "gene_2"]


def test_missing_feature_column_is_rejected(
    synthetic_pipeline: Pipeline,
    uploaded_features: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="отсутствуют столбцы"):
        validate_uploaded_features(uploaded_features.drop(columns="gene_1"), synthetic_pipeline)


def test_extra_feature_column_is_rejected(
    synthetic_pipeline: Pipeline,
    uploaded_features: pd.DataFrame,
) -> None:
    features = uploaded_features.assign(unexpected=1.0)

    with pytest.raises(ValueError, match="лишние столбцы"):
        validate_uploaded_features(features, synthetic_pipeline)


def test_non_numeric_value_is_rejected(
    synthetic_pipeline: Pipeline,
    uploaded_features: pd.DataFrame,
) -> None:
    features = uploaded_features.astype(object)
    features.loc["sample_a", "gene_0"] = "not-a-number"

    with pytest.raises(ValueError, match="числовыми"):
        validate_uploaded_features(features, synthetic_pipeline)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_nan_and_infinity_are_rejected(
    synthetic_pipeline: Pipeline,
    uploaded_features: pd.DataFrame,
    invalid_value: float,
) -> None:
    features = uploaded_features.copy()
    features.loc["sample_a", "gene_0"] = invalid_value

    with pytest.raises(ValueError, match="пропуски или бесконечные"):
        validate_uploaded_features(features, synthetic_pipeline)


def test_upload_row_limit_is_enforced(synthetic_pipeline: Pipeline) -> None:
    features = pd.DataFrame(
        np.zeros((101, 3)),
        columns=["gene_0", "gene_1", "gene_2"],
    )

    with pytest.raises(ValueError, match="не более 100 строк"):
        validate_uploaded_features(features, synthetic_pipeline)


def test_predict_uploaded_features_returns_classes_and_probabilities(
    synthetic_pipeline: Pipeline,
    uploaded_features: pd.DataFrame,
) -> None:
    result = predict_uploaded_features(uploaded_features, synthetic_pipeline)

    assert len(result) == len(uploaded_features)
    assert result.index.tolist() == uploaded_features.index.tolist()
    assert set(result[PREDICTION_COLUMN]) <= set(synthetic_pipeline.classes_)
    assert np.allclose(result.filter(like="prob_").sum(axis=1), 1.0)


def test_probability_columns_follow_classifier_class_order(
    synthetic_pipeline: Pipeline,
    uploaded_features: pd.DataFrame,
) -> None:
    result = predict_uploaded_features(uploaded_features, synthetic_pipeline)

    assert list(result.columns) == [
        PREDICTION_COLUMN,
        *(f"prob_{label}" for label in synthetic_pipeline.classes_),
    ]


def test_app_services_contains_no_training_calls() -> None:
    source = Path(app_services_module.__file__).read_text(encoding="utf-8")

    assert ".fit(" not in source
    assert "cross_validate" not in source
    assert "run_final_evaluation" not in source


def test_prediction_does_not_modify_saved_artifacts(
    tmp_path: Path,
    synthetic_pipeline: Pipeline,
    uploaded_features: pd.DataFrame,
) -> None:
    artifact_paths = [
        tmp_path / "results" / "metrics.csv",
        tmp_path / "results" / "final_candidate.json",
        tmp_path / "results" / "final_evaluation.json",
        tmp_path / "models" / "final_e10_pipeline.joblib",
    ]
    for path in artifact_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic {path.name}".encode())
    before = {path: path.read_bytes() for path in artifact_paths}

    predict_uploaded_features(uploaded_features, synthetic_pipeline)

    assert {path: path.read_bytes() for path in artifact_paths} == before
