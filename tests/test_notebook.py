import json
import re
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/cancer_rnaseq_project.ipynb")
EXPECTED_SECTIONS = (
    "1. Постановка задачи",
    "2. Обзор данных",
    "3. Экспериментальный протокол",
    "4. Сравнение экспериментов",
    "5. Выбор кандидата",
    "6. Финальная оценка",
    "7. Выводы и ограничения",
)
ALLOWED_RESULT_PATHS = {
    "results/dataset_summary.json",
    "results/split_summary.json",
    "results/metrics.csv",
    "results/final_candidate.json",
    "results/final_evaluation.json",
    "results/final_classification_report.json",
    "results/final_confusion_matrix.csv",
}


def _load_notebook() -> dict[str, object]:
    loaded = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _cell_source(cell: dict[str, object]) -> str:
    source = cell["source"]
    if isinstance(source, list):
        return "".join(str(line) for line in source)
    return str(source)


def test_notebook_is_valid_ipynb() -> None:
    notebook = _load_notebook()

    assert notebook["nbformat"] == 4
    assert isinstance(notebook["cells"], list)
    assert notebook["cells"]


def test_notebook_contains_all_required_sections() -> None:
    notebook = _load_notebook()
    cells = notebook["cells"]
    assert isinstance(cells, list)
    markdown = "\n".join(
        _cell_source(cell)
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_type") == "markdown"
    )

    assert all(section in markdown for section in EXPECTED_SECTIONS)


def test_notebook_contains_no_training_or_final_evaluation_calls() -> None:
    notebook = _load_notebook()
    cells = notebook["cells"]
    assert isinstance(cells, list)
    source = "\n".join(_cell_source(cell) for cell in cells if isinstance(cell, dict))

    for forbidden in (".fit(", "cross_validate", "run_final_evaluation", "predict_proba("):
        assert forbidden not in source


def test_notebook_reads_only_saved_artifacts() -> None:
    notebook = _load_notebook()
    cells = notebook["cells"]
    assert isinstance(cells, list)
    source = "\n".join(_cell_source(cell) for cell in cells if isinstance(cell, dict))
    result_paths = set(re.findall(r'["\'](results/[^"\']+)["\']', source))

    assert result_paths == ALLOWED_RESULT_PATHS
    assert "data/raw" not in source
    assert "joblib" not in source
    assert "urllib" not in source
    assert "requests" not in source
    assert "image/png" not in NOTEBOOK_PATH.read_text(encoding="utf-8")


def test_notebook_keeps_executed_table_outputs() -> None:
    notebook = _load_notebook()
    cells = notebook["cells"]
    assert isinstance(cells, list)
    code_cells = [
        cell for cell in cells if isinstance(cell, dict) and cell.get("cell_type") == "code"
    ]

    assert all(cell.get("execution_count") is not None for cell in code_cells)
    assert sum(bool(cell.get("outputs")) for cell in code_cells) >= 5
