import ast
import json
from pathlib import Path

import pytest
from scripts import freeze_final_candidate as freeze_script

from src.candidate_selection import (
    EXPECTED_SPLIT,
    build_final_candidate,
    freeze_final_candidate,
    select_final_candidate,
)
from src.evaluation import ExperimentResult
from src.experiments import save_experiment_metrics


def _result(experiment_id: str) -> ExperimentResult:
    experiment_number = int(experiment_id[1:])
    is_e10 = experiment_id == "E10"
    is_leader = experiment_id in {"E08", "E09", "E10"}
    leader_dimensions = {"E08": 200, "E09": 500, "E10": 20}
    return {
        "experiment_id": experiment_id,
        "model": "LogisticRegression" if is_e10 else "LinearSVC",
        "feature_method": "pca" if is_e10 else "select_k_best_f_classif",
        "n_features": leader_dimensions.get(experiment_id, 100 + experiment_number),
        "cv_f1_macro_mean": 1.0 if is_leader else 0.9,
        "cv_f1_macro_std": 0.0 if is_leader else 0.01,
        "cv_accuracy_mean": 0.9,
        "cv_accuracy_std": 0.01,
        "cv_precision_macro_mean": 0.9,
        "cv_precision_macro_std": 0.01,
        "cv_recall_macro_mean": 0.9,
        "cv_recall_macro_std": 0.01,
        "cv_fit_time_mean": 0.1,
        "cv_score_time_mean": 0.01,
        "total_cv_time_seconds": 2.0 + experiment_number,
        "random_state": 42,
        "cv_splits": 5,
    }


def _results() -> list[ExperimentResult]:
    return [_result(f"E{number:02d}") for number in range(18)]


def _candidate() -> dict[str, object]:
    selected = select_final_candidate(_results())
    return build_final_candidate(selected, dict(EXPECTED_SPLIT))


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    metrics_path = tmp_path / "metrics.csv"
    split_path = tmp_path / "split_summary.json"
    candidate_path = tmp_path / "final_candidate.json"
    save_experiment_metrics(_results(), metrics_path)
    split_path.write_text(json.dumps(EXPECTED_SPLIT), encoding="utf-8")
    return metrics_path, split_path, candidate_path


def _run_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    metrics_path, split_path, candidate_path = _write_inputs(tmp_path)
    monkeypatch.setattr(freeze_script, "METRICS_PATH", metrics_path)
    monkeypatch.setattr(freeze_script, "SPLIT_SUMMARY_PATH", split_path)
    monkeypatch.setattr(freeze_script, "FINAL_CANDIDATE_PATH", candidate_path)
    freeze_script.main()
    return metrics_path, split_path, candidate_path


def test_e10_is_selected_from_complete_results() -> None:
    selected = select_final_candidate(_results())

    assert selected["experiment_id"] == "E10"


def test_maximum_macro_f1_has_priority() -> None:
    results = _results()
    results[11]["cv_f1_macro_mean"] = 1.1

    selected = select_final_candidate(results)

    assert selected["experiment_id"] == "E11"


def test_smaller_internal_representation_breaks_f1_tie() -> None:
    results = _results()
    results[8]["total_cv_time_seconds"] = 0.1

    selected = select_final_candidate(results)

    assert selected["experiment_id"] == "E10"


def test_lower_time_breaks_equal_f1_and_dimension_tie() -> None:
    results = _results()
    results[8]["n_features"] = 20
    results[8]["total_cv_time_seconds"] = 0.1

    selected = select_final_candidate(results)

    assert selected["experiment_id"] == "E08"


def test_candidate_configuration_matches_e10() -> None:
    candidate = _candidate()

    assert candidate["selected_experiment_id"] == "E10"
    assert candidate["selected_model"] == "LogisticRegression"
    assert candidate["feature_method"] == "pca"
    assert candidate["representation_dimensions"] == 20
    assert candidate["selection_metric_value"] == 1.0
    assert candidate["selection_metric_std"] == 0.0


def test_candidate_json_contains_required_fields(tmp_path: Path) -> None:
    output_path = freeze_final_candidate(_candidate(), tmp_path / "final_candidate.json")
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert {
        "selection_status",
        "selected_experiment_id",
        "selected_model",
        "feature_method",
        "representation_dimensions",
        "representation_type",
        "uses_all_original_features",
        "original_feature_count",
        "selection_metric",
        "selection_metric_value",
        "selection_metric_std",
        "selection_source",
        "selection_source_commit",
        "tie_break_rule",
        "pipeline",
        "split",
        "test_evaluated",
        "final_model_trained",
    } <= set(saved)


def test_candidate_marks_test_as_not_evaluated() -> None:
    assert _candidate()["test_evaluated"] is False


def test_candidate_marks_final_model_as_not_trained() -> None:
    assert _candidate()["final_model_trained"] is False


def test_candidate_pipeline_is_scaler_pca20_logistic_regression() -> None:
    pipeline = _candidate()["pipeline"]

    assert isinstance(pipeline, dict)
    assert list(pipeline) == ["scaler", "pca", "classifier"]
    assert pipeline["scaler"] == {"class": "StandardScaler"}
    assert pipeline["pca"] == {
        "class": "PCA",
        "n_components": 20,
        "svd_solver": "randomized",
        "random_state": 42,
        "whiten": False,
    }
    assert pipeline["classifier"] == {
        "class": "LogisticRegression",
        "max_iter": 3000,
        "random_state": 42,
    }


def test_split_checksums_are_copied_without_change() -> None:
    split = _candidate()["split"]

    assert isinstance(split, dict)
    assert split["train_index_sha256"] == EXPECTED_SPLIT["train_index_sha256"]
    assert split["test_index_sha256"] == EXPECTED_SPLIT["test_index_sha256"]


def test_freeze_script_does_not_import_dataset_loader() -> None:
    script_path = Path(freeze_script.__file__)
    syntax_tree = ast.parse(script_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module for node in ast.walk(syntax_tree) if isinstance(node, ast.ImportFrom)
    }
    called_names = {
        node.func.id
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "src.data_loader" not in imported_modules
    assert {"download_dataset", "load_dataset", "split_dataset"}.isdisjoint(called_names)


def test_repeated_freeze_with_same_candidate_is_allowed(tmp_path: Path) -> None:
    output_path = tmp_path / "final_candidate.json"
    candidate = _candidate()

    freeze_final_candidate(candidate, output_path)
    before = output_path.read_bytes()
    freeze_final_candidate(candidate, output_path)

    assert output_path.read_bytes() == before


def test_conflicting_existing_candidate_is_not_overwritten(tmp_path: Path) -> None:
    output_path = tmp_path / "final_candidate.json"
    conflicting = _candidate()
    conflicting["selected_experiment_id"] = "E08"
    output_path.write_text(json.dumps(conflicting), encoding="utf-8")
    before = output_path.read_bytes()

    with pytest.raises(RuntimeError, match="delete it explicitly"):
        freeze_final_candidate(_candidate(), output_path)

    assert output_path.read_bytes() == before


def test_freeze_script_does_not_modify_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics_path, split_path, candidate_path = _write_inputs(tmp_path)
    monkeypatch.setattr(freeze_script, "METRICS_PATH", metrics_path)
    monkeypatch.setattr(freeze_script, "SPLIT_SUMMARY_PATH", split_path)
    monkeypatch.setattr(freeze_script, "FINAL_CANDIDATE_PATH", candidate_path)
    before = metrics_path.read_bytes()

    freeze_script.main()

    assert metrics_path.read_bytes() == before


def test_freeze_script_creates_no_predictions_metrics_or_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, candidate_path = _run_script(tmp_path, monkeypatch)
    created_files = {path.name for path in tmp_path.iterdir()}
    saved_candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

    assert created_files == {"metrics.csv", "split_summary.json", "final_candidate.json"}
    assert "test_metrics" not in saved_candidate
    assert "predictions" not in saved_candidate
