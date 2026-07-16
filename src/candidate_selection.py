import json
from math import isclose
from pathlib import Path

from src.config import FINAL_CANDIDATE_PATH
from src.evaluation import ExperimentResult
from src.experiments import EXTENDED_EXPERIMENT_IDS, PRESERVED_EXPERIMENT_IDS

SELECTED_EXPERIMENT_ID = "E10"
SELECTION_SOURCE_COMMIT = "0303133aef4ab853c30e7ee142f84067b36c7c98"
EXPECTED_EXPERIMENT_IDS = PRESERVED_EXPERIMENT_IDS + EXTENDED_EXPERIMENT_IDS
EXPECTED_SPLIT = {
    "random_state": 42,
    "test_size": 0.2,
    "train_samples": 640,
    "test_samples": 161,
    "train_index_sha256": "5b6a7488ce4e227431d25515675bcb8ad544fcf65620e0abbc09d368419e6d34",
    "test_index_sha256": "833cf2ee6a93f35eb303125d093170889a808a62014a0e5e7ffb469eaf9ce0c8",
}
TIE_BREAK_RULE = [
    "maximum cv_f1_macro_mean",
    "lower cv_f1_macro_std",
    "simpler classifier family",
    "simpler feature method",
    "smaller representation only within the same method",
]
MODEL_PRIORITY = {
    "LogisticRegression": 0,
    "LinearSVC": 1,
    "RandomForestClassifier": 2,
    "PyTorchMLP": 3,
    "DummyClassifier": 4,
}
FEATURE_METHOD_PRIORITY = {
    "none": 0,
    "pca": 1,
    "select_k_best_f_classif": 2,
}


def select_final_candidate(results: list[ExperimentResult]) -> ExperimentResult:
    """Select the frozen candidate from the complete train CV result table."""
    experiment_ids = [result["experiment_id"] for result in results]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("Duplicate experiment_id in candidate selection results")
    if set(experiment_ids) != set(EXPECTED_EXPERIMENT_IDS):
        raise ValueError("Candidate selection requires exactly E00 through E17")

    maximum_f1 = max(float(result["cv_f1_macro_mean"]) for result in results)
    leaders = [
        result
        for result in results
        if isclose(
            float(result["cv_f1_macro_mean"]),
            maximum_f1,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    return min(
        leaders,
        key=lambda result: (
            float(result["cv_f1_macro_std"]),
            MODEL_PRIORITY.get(result["model"], 99),
            FEATURE_METHOD_PRIORITY.get(result["feature_method"], 99),
            int(result["n_features"]),
            result["experiment_id"],
        ),
    )


def build_final_candidate(
    selected_result: ExperimentResult,
    split_summary: dict[str, object],
) -> dict[str, object]:
    """Build the frozen E10 record from saved CV results and split metadata."""
    if selected_result["experiment_id"] != SELECTED_EXPERIMENT_ID:
        raise RuntimeError("Selected candidate has unexpected experiment_id")
    if selected_result["model"] != "LogisticRegression":
        raise RuntimeError("Selected candidate has unexpected model")
    if selected_result["feature_method"] != "pca":
        raise RuntimeError("Selected candidate has unexpected feature_method")
    if selected_result["n_features"] != 20:
        raise RuntimeError("Selected candidate has unexpected n_features")
    if selected_result["cv_f1_macro_mean"] != 1.0:
        raise RuntimeError("Selected candidate has unexpected cv_f1_macro_mean")
    if selected_result["cv_f1_macro_std"] != 0.0:
        raise RuntimeError("Selected candidate has unexpected cv_f1_macro_std")

    for field, expected_value in EXPECTED_SPLIT.items():
        if split_summary.get(field) != expected_value:
            raise RuntimeError(f"Saved split metadata has unexpected {field}")

    return {
        "selection_status": "frozen_before_test",
        "selected_experiment_id": SELECTED_EXPERIMENT_ID,
        "selected_model": selected_result["model"],
        "feature_method": selected_result["feature_method"],
        "representation_dimensions": selected_result["n_features"],
        "representation_type": "pca_components",
        "uses_all_original_features": True,
        "original_feature_count": 20531,
        "selection_metric": "cv_f1_macro_mean",
        "selection_metric_value": selected_result["cv_f1_macro_mean"],
        "selection_metric_std": selected_result["cv_f1_macro_std"],
        "selection_source": "results/metrics.csv",
        "selection_source_commit": SELECTION_SOURCE_COMMIT,
        "tie_break_rule": TIE_BREAK_RULE,
        "pipeline": {
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
        },
        "split": dict(EXPECTED_SPLIT),
        "test_evaluated": False,
        "final_model_trained": False,
    }


def freeze_final_candidate(
    candidate: dict[str, object],
    output_path: Path = FINAL_CANDIDATE_PATH,
) -> Path:
    """Write the candidate atomically while rejecting a conflicting freeze."""
    if output_path.exists():
        existing_candidate = json.loads(output_path.read_text(encoding="utf-8"))
        if existing_candidate != candidate:
            raise RuntimeError(
                "Existing final_candidate.json differs; delete it explicitly before refreezing"
            )
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.json")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(candidate, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    temporary_path.replace(output_path)
    return output_path
