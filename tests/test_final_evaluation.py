import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scripts import run_final_evaluation as final_script
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

import src.final_evaluation as final_evaluation_module
from src.candidate_selection import EXPECTED_SPLIT
from src.final_evaluation import (
    EXPECTED_CANDIDATE,
    EXPECTED_CLASS_ORDER,
    EXPECTED_PIPELINE_RECORD,
    FinalEvaluationResult,
    build_confusion_table,
    build_frozen_e10_pipeline,
    build_predictions_table,
    ensure_no_final_artifacts,
    evaluate_frozen_pipeline,
    final_artifact_paths,
    load_and_validate_frozen_inputs,
    save_confusion_matrix_figure,
    save_final_artifacts,
    sha256_file,
    validate_frozen_candidate,
    validate_frozen_pipeline,
    validate_split_metadata,
)


def _candidate() -> dict[str, object]:
    return {
        **EXPECTED_CANDIDATE,
        "pipeline": copy.deepcopy(EXPECTED_PIPELINE_RECORD),
    }


def _dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    random = np.random.default_rng(42)
    class_order = list(EXPECTED_CLASS_ORDER)
    train_labels = np.repeat(class_order, 12)
    test_labels = np.repeat(class_order, 4)
    train_features = random.normal(size=(len(train_labels), 25))
    test_features = random.normal(size=(len(test_labels), 25))
    for class_index in range(len(class_order)):
        train_mask = train_labels == class_order[class_index]
        test_mask = test_labels == class_order[class_index]
        train_features[train_mask, class_index * 4 : class_index * 4 + 4] += 4.0
        test_features[test_mask, class_index * 4 : class_index * 4 + 4] += 4.0

    features_train = pd.DataFrame(
        train_features,
        index=[f"train_{index}" for index in range(len(train_labels))],
    )
    target_train = pd.Series(train_labels, index=features_train.index, dtype="string")
    features_test = pd.DataFrame(
        test_features,
        index=[f"test_{index}" for index in range(len(test_labels))],
    )
    target_test = pd.Series(test_labels, index=features_test.index, dtype="string")
    return features_train, target_train, features_test, target_test


def _evaluate() -> tuple[
    Pipeline,
    FinalEvaluationResult,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    features_train, target_train, features_test, target_test = _dataset()
    pipeline = build_frozen_e10_pipeline()
    evaluation = evaluate_frozen_pipeline(
        pipeline,
        features_train,
        target_train,
        features_test,
        target_test,
    )
    return pipeline, evaluation, features_train, target_train, features_test, target_test


def _publish(tmp_path: Path) -> tuple[dict[str, object], dict[str, Path]]:
    pipeline, evaluation, _, _, features_test, target_test = _evaluate()
    paths = final_artifact_paths(tmp_path)
    manifest = save_final_artifacts(
        pipeline,
        evaluation,
        features_test,
        target_test,
        paths,
        tmp_path,
    )
    return manifest, paths


def test_frozen_candidate_e10_is_valid() -> None:
    validate_frozen_candidate(_candidate())


def test_other_experiment_id_is_rejected() -> None:
    candidate = _candidate()
    candidate["selected_experiment_id"] = "E08"

    with pytest.raises(RuntimeError, match="selected_experiment_id"):
        validate_frozen_candidate(candidate)


def test_changed_pipeline_parameters_are_rejected() -> None:
    candidate = _candidate()
    pipeline = copy.deepcopy(EXPECTED_PIPELINE_RECORD)
    pipeline["pca"]["n_components"] = 50
    candidate["pipeline"] = pipeline

    with pytest.raises(RuntimeError, match="pipeline configuration"):
        validate_frozen_candidate(candidate)


def test_preexisting_test_evaluated_true_is_rejected() -> None:
    candidate = _candidate()
    candidate["test_evaluated"] = True

    with pytest.raises(RuntimeError, match="test_evaluated"):
        validate_frozen_candidate(candidate)


def test_preexisting_final_model_trained_true_is_rejected() -> None:
    candidate = _candidate()
    candidate["final_model_trained"] = True

    with pytest.raises(RuntimeError, match="final_model_trained"):
        validate_frozen_candidate(candidate)


def test_metrics_sha256_is_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidate_path = tmp_path / "final_candidate.json"
    metrics_path = tmp_path / "metrics.csv"
    split_path = tmp_path / "split_summary.json"
    candidate_path.write_text(json.dumps(_candidate()), encoding="utf-8")
    metrics_path.write_text("synthetic metrics\n", encoding="utf-8")
    split_path.write_text(json.dumps(EXPECTED_SPLIT), encoding="utf-8")
    monkeypatch.setattr(
        final_evaluation_module,
        "EXPECTED_METRICS_SHA256",
        sha256_file(metrics_path),
    )

    loaded_candidate, loaded_split = load_and_validate_frozen_inputs(
        candidate_path,
        metrics_path,
        split_path,
    )

    assert loaded_candidate["selected_experiment_id"] == "E10"
    assert loaded_split["train_index_sha256"] == EXPECTED_SPLIT["train_index_sha256"]


def test_changed_metrics_sha256_is_rejected(tmp_path: Path) -> None:
    candidate_path = tmp_path / "final_candidate.json"
    metrics_path = tmp_path / "metrics.csv"
    split_path = tmp_path / "split_summary.json"
    candidate_path.write_text(json.dumps(_candidate()), encoding="utf-8")
    metrics_path.write_text("changed metrics\n", encoding="utf-8")
    split_path.write_text(json.dumps(EXPECTED_SPLIT), encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"metrics\.csv SHA-256"):
        load_and_validate_frozen_inputs(candidate_path, metrics_path, split_path)


def test_split_checksums_are_checked() -> None:
    split_summary = dict(EXPECTED_SPLIT)
    split_summary["test_index_sha256"] = "changed"

    with pytest.raises(RuntimeError, match="test_index_sha256"):
        validate_split_metadata(split_summary)


def test_pipeline_has_exact_step_order() -> None:
    pipeline = build_frozen_e10_pipeline()

    assert list(pipeline.named_steps) == ["scaler", "pca", "classifier"]


def test_pipeline_has_exact_pca_parameters() -> None:
    pca = build_frozen_e10_pipeline().named_steps["pca"]

    assert isinstance(pca, PCA)
    assert pca.n_components == 20
    assert pca.svd_solver == "randomized"
    assert pca.random_state == 42
    assert pca.whiten is False


def test_pipeline_has_exact_logistic_regression_parameters() -> None:
    classifier = build_frozen_e10_pipeline().named_steps["classifier"]

    assert isinstance(classifier, LogisticRegression)
    assert classifier.max_iter == 3000
    assert classifier.random_state == 42


def test_pipeline_is_fitted_on_all_passed_train_rows() -> None:
    pipeline, _, features_train, _, _, _ = _evaluate()
    scaler = pipeline.named_steps["scaler"]

    assert int(scaler.n_samples_seen_) == len(features_train)


def test_test_rows_are_not_passed_to_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    features_train, target_train, features_test, target_test = _dataset()
    pipeline = build_frozen_e10_pipeline()
    original_fit = pipeline.fit
    fitted_indices: list[pd.Index] = []

    def recording_fit(features: pd.DataFrame, target: pd.Series) -> Pipeline:
        fitted_indices.append(features.index)
        return original_fit(features, target)

    monkeypatch.setattr(pipeline, "fit", recording_fit)
    evaluate_frozen_pipeline(
        pipeline,
        features_train,
        target_train,
        features_test,
        target_test,
    )

    assert fitted_indices == [features_train.index]
    assert set(fitted_indices[0]).isdisjoint(features_test.index)


def test_final_test_metrics_are_calculated_correctly() -> None:
    _, evaluation, _, _, _, target_test = _evaluate()
    predictions = evaluation["predictions"]

    assert evaluation["test_metrics"] == pytest.approx(
        {
            "f1_macro": f1_score(target_test, predictions, average="macro"),
            "accuracy": accuracy_score(target_test, predictions),
            "precision_macro": precision_score(
                target_test, predictions, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(target_test, predictions, average="macro"),
        }
    )


def test_class_order_comes_from_fitted_classifier() -> None:
    pipeline, evaluation, _, _, _, _ = _evaluate()
    classifier = pipeline.named_steps["classifier"]

    assert evaluation["class_order"] == [str(label) for label in classifier.classes_]


def test_predictions_table_has_required_structure() -> None:
    _, evaluation, _, _, features_test, target_test = _evaluate()
    table = build_predictions_table(features_test, target_test, evaluation)

    assert list(table.columns) == [
        "sample_id",
        "true_label",
        "predicted_label",
        "correct",
        "prob_BRCA",
        "prob_COAD",
        "prob_KIRC",
        "prob_LUAD",
        "prob_PRAD",
    ]
    assert len(table) == len(features_test)
    assert table["sample_id"].tolist() == features_test.index.tolist()
    assert set(table["correct"]) <= {"true", "false"}


def test_prediction_probabilities_sum_to_one() -> None:
    _, evaluation, _, _, features_test, target_test = _evaluate()
    table = build_predictions_table(features_test, target_test, evaluation)
    probability_columns = [f"prob_{label}" for label in EXPECTED_CLASS_ORDER]

    assert np.allclose(table[probability_columns].sum(axis=1), 1.0)


def test_classification_report_has_class_and_aggregate_rows() -> None:
    _, evaluation, _, _, _, _ = _evaluate()
    report = evaluation["classification_report"]

    assert set(EXPECTED_CLASS_ORDER) <= set(report)
    assert {"accuracy", "macro avg", "weighted avg"} <= set(report)
    assert {"precision", "recall", "f1-score", "support"} <= set(report["BRCA"])


def test_confusion_table_has_required_structure() -> None:
    _, evaluation, _, _, _, _ = _evaluate()
    table = build_confusion_table(evaluation)

    assert list(table.columns) == ["true_label", *EXPECTED_CLASS_ORDER]
    assert table["true_label"].tolist() == list(EXPECTED_CLASS_ORDER)


def test_confusion_matrix_sum_matches_test_size() -> None:
    _, evaluation, _, _, _, target_test = _evaluate()

    assert int(evaluation["confusion_matrix"].sum()) == len(target_test)


def test_confusion_matrix_png_is_created(tmp_path: Path) -> None:
    _, evaluation, _, _, _, _ = _evaluate()
    output_path = tmp_path / "final_confusion_matrix.png"

    save_confusion_matrix_figure(evaluation, output_path)

    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_joblib_pipeline_can_be_saved_and_loaded(tmp_path: Path) -> None:
    pipeline, _, _, _, _, _ = _evaluate()
    model_path = tmp_path / "final_e10_pipeline.joblib"

    joblib.dump(pipeline, model_path)
    loaded = joblib.load(model_path)

    assert isinstance(loaded, Pipeline)
    validate_frozen_pipeline(loaded, require_fitted=True)


def test_joblib_predictions_match_before_and_after_load(tmp_path: Path) -> None:
    pipeline, evaluation, _, _, features_test, _ = _evaluate()
    model_path = tmp_path / "final_e10_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    loaded = joblib.load(model_path)

    assert np.array_equal(loaded.predict(features_test), evaluation["predictions"])


def test_final_evaluation_manifest_has_required_structure(tmp_path: Path) -> None:
    manifest, _ = _publish(tmp_path)

    assert {
        "evaluation_status",
        "candidate_freeze_commit",
        "selected_experiment_id",
        "selected_model",
        "feature_method",
        "representation_dimensions",
        "original_feature_count",
        "train_samples",
        "test_samples",
        "split",
        "cv_reference",
        "test_metrics",
        "test_minus_cv_f1_macro",
        "class_order",
        "training_time_seconds",
        "prediction_time_seconds",
        "probability_prediction_time_seconds",
        "final_model_trained",
        "test_evaluated",
        "artifacts",
        "runtime",
    } == set(manifest)


def test_manifest_marks_test_as_evaluated(tmp_path: Path) -> None:
    manifest, _ = _publish(tmp_path)

    assert manifest["test_evaluated"] is True


def test_manifest_marks_final_model_as_trained(tmp_path: Path) -> None:
    manifest, _ = _publish(tmp_path)

    assert manifest["final_model_trained"] is True


def test_manifest_contains_valid_artifact_hashes(tmp_path: Path) -> None:
    manifest, paths = _publish(tmp_path)
    artifacts = manifest["artifacts"]

    assert isinstance(artifacts, dict)
    for name, record in artifacts.items():
        assert isinstance(record, dict)
        assert record["sha256"] == sha256_file(paths[name])
        assert not str(record["path"]).startswith("/")


def test_existing_final_marker_refuses_before_dataset_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = final_artifact_paths(tmp_path)
    paths["evaluation"].parent.mkdir(parents=True)
    paths["evaluation"].write_text("{}", encoding="utf-8")
    dataset_loaded = False

    def forbidden_download() -> Path:
        nonlocal dataset_loaded
        dataset_loaded = True
        raise AssertionError("dataset must not be loaded")

    monkeypatch.setattr(final_script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(final_script, "download_dataset", forbidden_download)

    with pytest.raises(RuntimeError, match="Refusing to evaluate"):
        final_script.main()

    assert dataset_loaded is False


@pytest.mark.parametrize(
    "artifact_name",
    [
        "classification_report",
        "predictions",
        "confusion_csv",
        "confusion_png",
        "model",
    ],
)
def test_any_partial_final_artifact_refuses_evaluation(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    paths = final_artifact_paths(tmp_path)
    paths[artifact_name].parent.mkdir(parents=True, exist_ok=True)
    paths[artifact_name].write_text("partial", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Refusing to evaluate"):
        ensure_no_final_artifacts(paths)


def test_final_script_has_no_force_option() -> None:
    source = Path(final_script.__file__).read_text(encoding="utf-8")

    assert "--force" not in source
    assert "argparse" not in source


def test_artifact_save_does_not_modify_metrics_file(tmp_path: Path) -> None:
    metrics_path = tmp_path / "results" / "metrics.csv"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text("synthetic metrics\n", encoding="utf-8")
    before = metrics_path.read_bytes()

    _publish(tmp_path)

    assert metrics_path.read_bytes() == before


def test_artifact_save_does_not_modify_frozen_candidate(tmp_path: Path) -> None:
    candidate_path = tmp_path / "results" / "final_candidate.json"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text(json.dumps(_candidate()), encoding="utf-8")
    before = candidate_path.read_bytes()

    _publish(tmp_path)

    assert candidate_path.read_bytes() == before


def test_final_workflow_uses_only_e10() -> None:
    pipeline = build_frozen_e10_pipeline()
    source = Path(final_script.__file__).read_text(encoding="utf-8")

    assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)
    assert "E08" not in source
    assert "E09" not in source


def test_final_workflow_has_no_cv_or_tuning() -> None:
    source = Path(final_evaluation_module.__file__).read_text(encoding="utf-8")
    script_source = Path(final_script.__file__).read_text(encoding="utf-8")
    forbidden_terms = {
        "cross_validate",
        "GridSearchCV",
        "RandomizedSearchCV",
        "SelectKBest",
        "LinearSVC",
    }

    assert all(term not in source for term in forbidden_terms)
    assert all(term not in script_source for term in forbidden_terms)
