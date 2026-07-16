import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/model_comparison_demo.ipynb")


def _notebook_source() -> tuple[dict[str, object], str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    assert isinstance(cells, list)
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in cells
        if isinstance(cell, dict) and isinstance(cell.get("source"), list)
    )
    return notebook, source


def test_comparison_notebook_is_valid_and_has_required_sections() -> None:
    notebook, source = _notebook_source()

    assert notebook["nbformat"] == 4
    assert all(f"## {section}." in source for section in range(1, 10))
    assert "matplotlib.pyplot" in source
    assert "mlp_fold_metrics.csv" in source
    assert "robustness_e10.json" in source
    assert "defense_pipeline.svg" not in source
    for figure_name in (
        "project_pipeline.png",
        "mlp_loss_curves.png",
        "mlp_validation_f1.png",
        "mlp_epoch_selection.png",
        "mlp_fold_metrics.png",
    ):
        assert figure_name in source


def test_comparison_notebook_does_not_train_models() -> None:
    _, source = _notebook_source()

    assert ".fit(" not in source
    assert "cross_validate" not in source
    assert "RUN_MODEL_SMOKE_TEST = False" in source
    assert "161 образца" in source


def test_comparison_notebook_contains_saved_outputs() -> None:
    notebook, _ = _notebook_source()
    cells = notebook["cells"]
    assert isinstance(cells, list)
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    outputs = [output for cell in code_cells for output in cell.get("outputs", [])]

    assert all(cell.get("execution_count") is not None for cell in code_cells)
    assert not any(output.get("output_type") == "error" for output in outputs)
    assert sum("image/png" in output.get("data", {}) for output in outputs) >= 9
