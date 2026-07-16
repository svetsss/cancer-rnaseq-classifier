import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
from typing import TypedDict, cast

import joblib
import numpy as np
import pandas as pd
import sklearn
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.candidate_selection import EXPECTED_SPLIT
from src.config import PROJECT_ROOT
from src.sklearn_pipelines import build_logistic_pca_pipeline

CANDIDATE_FREEZE_COMMIT = "b5ebb02b61a7961fc86380f4072e809746af4f44"
EXPECTED_METRICS_SHA256 = "3d07cbc37d69a912640d782df3f2dd66daeb788c0570114285614f59828c899b"
EXPECTED_CLASS_ORDER = ("BRCA", "COAD", "KIRC", "LUAD", "PRAD")
FINAL_ARTIFACT_ERROR = (
    "Final test evaluation artifacts already exist. Refusing to evaluate the test set again."
)
EXPECTED_CANDIDATE = {
    "selection_status": "frozen_before_test",
    "selected_experiment_id": "E10",
    "selected_model": "LogisticRegression",
    "feature_method": "pca",
    "representation_dimensions": 20,
    "test_evaluated": False,
    "final_model_trained": False,
    "selection_source_commit": "0303133aef4ab853c30e7ee142f84067b36c7c98",
}
EXPECTED_PIPELINE_RECORD = {
    "scaler": {"class": "StandardScaler"},
    "pca": {
        "class": "PCA",
        "n_components": 20,
        "svd_solver": "randomized",
        "random_state": 42,
        "whiten": False,
    },
    "classifier": {
        "class": "LogisticRegression",
        "max_iter": 3000,
        "random_state": 42,
    },
}


FinalArtifactPaths = dict[str, Path]


class FinalEvaluationResult(TypedDict):
    predictions: np.ndarray
    probabilities: np.ndarray
    class_order: list[str]
    test_metrics: dict[str, float]
    classification_report: dict[str, object]
    confusion_matrix: np.ndarray
    training_time_seconds: float
    prediction_time_seconds: float
    probability_prediction_time_seconds: float


def final_artifact_paths(project_root: Path = PROJECT_ROOT) -> FinalArtifactPaths:
    """Return the fixed output paths for the one-time E10 evaluation."""
    return {
        "evaluation": project_root / "results" / "final_evaluation.json",
        "classification_report": project_root / "results" / "final_classification_report.json",
        "predictions": project_root / "results" / "final_test_predictions.csv",
        "confusion_csv": project_root / "results" / "final_confusion_matrix.csv",
        "confusion_png": project_root / "figures" / "final_confusion_matrix.png",
        "model": project_root / "models" / "final_e10_pipeline.joblib",
    }


def ensure_no_final_artifacts(paths: FinalArtifactPaths) -> None:
    """Refuse evaluation when any complete or partial final output already exists."""
    if any(path.exists() for path in paths.values()):
        raise RuntimeError(FINAL_ARTIFACT_ERROR)


def load_and_validate_frozen_inputs(
    candidate_path: Path,
    metrics_path: Path,
    split_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate all frozen inputs before dataset access."""
    loaded_candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if not isinstance(loaded_candidate, dict):
        raise ValueError("Unexpected final_candidate.json structure")
    validate_frozen_candidate(loaded_candidate)

    if sha256_file(metrics_path) != EXPECTED_METRICS_SHA256:
        raise RuntimeError("results/metrics.csv SHA-256 does not match the frozen input")

    loaded_split = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(loaded_split, dict):
        raise ValueError("Unexpected split_summary.json structure")
    validate_split_metadata(loaded_split)
    return loaded_candidate, loaded_split


def validate_frozen_candidate(candidate: dict[str, object]) -> None:
    """Require the exact pre-test E10 decision and pipeline configuration."""
    for field, expected_value in EXPECTED_CANDIDATE.items():
        if candidate.get(field) != expected_value:
            raise RuntimeError(f"Frozen candidate has unexpected {field}")
    if candidate.get("pipeline") != EXPECTED_PIPELINE_RECORD:
        raise RuntimeError("Frozen candidate has unexpected pipeline configuration")


def validate_split_metadata(split_summary: dict[str, object]) -> None:
    """Require the saved train/test sizes and index checksums."""
    for field, expected_value in EXPECTED_SPLIT.items():
        if split_summary.get(field) != expected_value:
            raise RuntimeError(f"Split metadata has unexpected {field}")


def build_frozen_e10_pipeline() -> Pipeline:
    """Build the exact E10 pipeline selected before test evaluation."""
    pipeline = build_logistic_pca_pipeline(20, random_state=42)
    validate_frozen_pipeline(pipeline)
    return pipeline


def validate_frozen_pipeline(pipeline: Pipeline, *, require_fitted: bool = False) -> None:
    """Require the exact E10 steps, parameters, and optional fitted state."""
    if list(pipeline.named_steps) != ["scaler", "pca", "classifier"]:
        raise RuntimeError("Final pipeline has unexpected step order")

    scaler = pipeline.named_steps["scaler"]
    pca = pipeline.named_steps["pca"]
    classifier = pipeline.named_steps["classifier"]
    if not isinstance(scaler, StandardScaler):
        raise RuntimeError("Final pipeline scaler is not StandardScaler")
    if not isinstance(pca, PCA):
        raise RuntimeError("Final pipeline reducer is not PCA")
    if (
        pca.n_components != 20
        or pca.svd_solver != "randomized"
        or pca.random_state != 42
        or pca.whiten
    ):
        raise RuntimeError("Final pipeline has unexpected PCA parameters")
    if not isinstance(classifier, LogisticRegression):
        raise RuntimeError("Final pipeline classifier is not LogisticRegression")
    if classifier.max_iter != 3000 or classifier.random_state != 42:
        raise RuntimeError("Final pipeline has unexpected LogisticRegression parameters")
    if require_fitted and not hasattr(classifier, "classes_"):
        raise RuntimeError("Reloaded final pipeline is not fitted")


def evaluate_frozen_pipeline(
    pipeline: Pipeline,
    features_train: pd.DataFrame,
    target_train: pd.Series,
    features_test: pd.DataFrame,
    target_test: pd.Series,
) -> FinalEvaluationResult:
    """Fit E10 on full train and evaluate the reserved test subset once."""
    training_started_at = perf_counter()
    pipeline.fit(features_train, target_train)
    training_time = perf_counter() - training_started_at
    validate_frozen_pipeline(pipeline, require_fitted=True)

    prediction_started_at = perf_counter()
    predictions = np.asarray(pipeline.predict(features_test), dtype=str)
    prediction_time = perf_counter() - prediction_started_at

    probability_started_at = perf_counter()
    probabilities = np.asarray(pipeline.predict_proba(features_test), dtype=float)
    probability_prediction_time = perf_counter() - probability_started_at

    classifier = cast(LogisticRegression, pipeline.named_steps["classifier"])
    class_order = [str(label) for label in classifier.classes_]
    if tuple(class_order) != EXPECTED_CLASS_ORDER:
        raise RuntimeError("Final classifier has unexpected class order")

    test_metrics = {
        "f1_macro": float(f1_score(target_test, predictions, average="macro")),
        "accuracy": float(accuracy_score(target_test, predictions)),
        "precision_macro": float(
            precision_score(target_test, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(recall_score(target_test, predictions, average="macro")),
    }
    report = cast(
        dict[str, object],
        classification_report(
            target_test,
            predictions,
            labels=class_order,
            output_dict=True,
            zero_division=0,
        ),
    )
    matrix = confusion_matrix(target_test, predictions, labels=class_order)
    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "class_order": class_order,
        "test_metrics": test_metrics,
        "classification_report": report,
        "confusion_matrix": matrix,
        "training_time_seconds": training_time,
        "prediction_time_seconds": prediction_time,
        "probability_prediction_time_seconds": probability_prediction_time,
    }


def build_predictions_table(
    features_test: pd.DataFrame,
    target_test: pd.Series,
    evaluation: FinalEvaluationResult,
) -> pd.DataFrame:
    """Build the ordered final test prediction table."""
    true_labels = target_test.astype(str).to_numpy()
    predictions = evaluation["predictions"]
    table = pd.DataFrame(
        {
            "sample_id": features_test.index.astype(str),
            "true_label": true_labels,
            "predicted_label": predictions,
            "correct": np.where(true_labels == predictions, "true", "false"),
        }
    )
    for class_index, label in enumerate(evaluation["class_order"]):
        table[f"prob_{label}"] = evaluation["probabilities"][:, class_index]
    return table


def build_confusion_table(evaluation: FinalEvaluationResult) -> pd.DataFrame:
    """Build the integer final confusion matrix table."""
    table = pd.DataFrame(
        evaluation["confusion_matrix"],
        index=evaluation["class_order"],
        columns=evaluation["class_order"],
        dtype=int,
    )
    table.index.name = "true_label"
    return table.reset_index()


def save_confusion_matrix_figure(
    evaluation: FinalEvaluationResult,
    output_path: Path,
) -> Path:
    """Save the unnormalized final confusion matrix with count labels."""
    matrix = evaluation["confusion_matrix"]
    class_order = evaluation["class_order"]
    figure = Figure(figsize=(7, 6))
    axes = figure.subplots()
    image = axes.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axes)
    axes.set_title("Матрица ошибок финальной оценки")
    axes.set_xlabel("Предсказанный класс")
    axes.set_ylabel("Истинный класс")
    axes.set_xticks(range(len(class_order)), class_order)
    axes.set_yticks(range(len(class_order)), class_order)
    threshold = float(matrix.max()) / 2
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            axes.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    return output_path


def save_final_artifacts(
    pipeline: Pipeline,
    evaluation: FinalEvaluationResult,
    features_test: pd.DataFrame,
    target_test: pd.Series,
    paths: FinalArtifactPaths,
    project_root: Path,
) -> dict[str, object]:
    """Validate and publish all final artifacts, writing the manifest last."""
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths = {name: _temporary_path(path) for name, path in paths.items()}

    report_temp = temporary_paths["classification_report"]
    with report_temp.open("w", encoding="utf-8") as report_file:
        json.dump(evaluation["classification_report"], report_file, ensure_ascii=False, indent=2)
        report_file.write("\n")

    predictions_table = build_predictions_table(features_test, target_test, evaluation)
    predictions_table.to_csv(temporary_paths["predictions"], index=False, lineterminator="\n")
    confusion_table = build_confusion_table(evaluation)
    confusion_table.to_csv(temporary_paths["confusion_csv"], index=False, lineterminator="\n")
    save_confusion_matrix_figure(evaluation, temporary_paths["confusion_png"])
    joblib.dump(pipeline, temporary_paths["model"])

    _validate_temporary_artifacts(
        evaluation,
        features_test,
        target_test,
        temporary_paths,
    )
    artifact_names = (
        "classification_report",
        "predictions",
        "confusion_csv",
        "confusion_png",
        "model",
    )
    artifact_hashes = {name: sha256_file(temporary_paths[name]) for name in artifact_names}
    for name in artifact_names:
        temporary_paths[name].replace(paths[name])
        if sha256_file(paths[name]) != artifact_hashes[name]:
            raise RuntimeError(f"Final artifact hash changed during publish: {name}")

    manifest = _build_final_manifest(evaluation, paths, artifact_hashes, project_root)
    manifest_temp = temporary_paths["evaluation"]
    with manifest_temp.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
        manifest_file.write("\n")
    loaded_manifest = json.loads(manifest_temp.read_text(encoding="utf-8"))
    if loaded_manifest != manifest:
        raise RuntimeError("Final evaluation manifest validation failed")
    manifest_temp.replace(paths["evaluation"])
    return manifest


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of one frozen artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _temporary_path(path: Path) -> Path:
    return path.with_suffix(f".tmp{path.suffix}")


def _validate_temporary_artifacts(
    evaluation: FinalEvaluationResult,
    features_test: pd.DataFrame,
    target_test: pd.Series,
    temporary_paths: dict[str, Path],
) -> None:
    saved_report = json.loads(temporary_paths["classification_report"].read_text(encoding="utf-8"))
    if saved_report != evaluation["classification_report"]:
        raise RuntimeError("Classification report validation failed")

    saved_predictions = pd.read_csv(
        temporary_paths["predictions"],
        dtype={
            "sample_id": str,
            "true_label": str,
            "predicted_label": str,
            "correct": str,
        },
    )
    expected_predictions = build_predictions_table(features_test, target_test, evaluation)
    expected_columns = [
        "sample_id",
        "true_label",
        "predicted_label",
        "correct",
        *(f"prob_{label}" for label in evaluation["class_order"]),
    ]
    if list(saved_predictions.columns) != expected_columns or len(saved_predictions) != len(
        target_test
    ):
        raise RuntimeError("Final prediction table validation failed")
    for column in ("sample_id", "true_label", "predicted_label", "correct"):
        if saved_predictions[column].tolist() != expected_predictions[column].tolist():
            raise RuntimeError("Final prediction table validation failed")
    probability_columns = [f"prob_{label}" for label in evaluation["class_order"]]
    if not np.allclose(
        saved_predictions[probability_columns],
        expected_predictions[probability_columns],
    ) or not np.allclose(saved_predictions[probability_columns].sum(axis=1), 1.0):
        raise RuntimeError("Final prediction probabilities do not sum to one")

    saved_confusion = pd.read_csv(temporary_paths["confusion_csv"])
    expected_confusion = build_confusion_table(evaluation)
    if not saved_confusion.equals(expected_confusion) or int(
        saved_confusion[evaluation["class_order"]].to_numpy().sum()
    ) != len(target_test):
        raise RuntimeError("Final confusion matrix count does not match test size")
    if not temporary_paths["confusion_png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("Final confusion matrix PNG validation failed")

    reloaded_pipeline = joblib.load(temporary_paths["model"])
    if not isinstance(reloaded_pipeline, Pipeline):
        raise RuntimeError("Reloaded final model is not a Pipeline")
    validate_frozen_pipeline(reloaded_pipeline, require_fitted=True)
    reloaded_predictions = np.asarray(reloaded_pipeline.predict(features_test), dtype=str)
    if not np.array_equal(reloaded_predictions, evaluation["predictions"]):
        raise RuntimeError("Reloaded final model predictions do not match")


def _build_final_manifest(
    evaluation: FinalEvaluationResult,
    paths: FinalArtifactPaths,
    artifact_hashes: dict[str, str],
    project_root: Path,
) -> dict[str, object]:
    artifacts = {
        name: {
            "path": paths[name].relative_to(project_root).as_posix(),
            "sha256": artifact_hashes[name],
        }
        for name in artifact_hashes
    }
    return {
        "evaluation_status": "completed_once",
        "candidate_freeze_commit": CANDIDATE_FREEZE_COMMIT,
        "selected_experiment_id": "E10",
        "selected_model": "LogisticRegression",
        "feature_method": "pca",
        "representation_dimensions": 20,
        "original_feature_count": 20531,
        "train_samples": EXPECTED_SPLIT["train_samples"],
        "test_samples": EXPECTED_SPLIT["test_samples"],
        "split": {
            "random_state": EXPECTED_SPLIT["random_state"],
            "train_index_sha256": EXPECTED_SPLIT["train_index_sha256"],
            "test_index_sha256": EXPECTED_SPLIT["test_index_sha256"],
        },
        "cv_reference": {
            "cv_f1_macro_mean": 1.0,
            "cv_f1_macro_std": 0.0,
        },
        "test_metrics": evaluation["test_metrics"],
        "test_minus_cv_f1_macro": evaluation["test_metrics"]["f1_macro"] - 1.0,
        "class_order": evaluation["class_order"],
        "training_time_seconds": evaluation["training_time_seconds"],
        "prediction_time_seconds": evaluation["prediction_time_seconds"],
        "probability_prediction_time_seconds": evaluation["probability_prediction_time_seconds"],
        "final_model_trained": True,
        "test_evaluated": True,
        "artifacts": artifacts,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
