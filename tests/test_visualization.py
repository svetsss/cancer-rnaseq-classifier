from pathlib import Path

import pandas as pd

from src.visualization import save_class_distribution


def test_save_class_distribution_creates_png(tmp_path: Path) -> None:
    target = pd.Series(["BRCA", "KIRC", "BRCA"], dtype="string", name="cancer_type")
    output_path = tmp_path / "class_distribution.png"

    result = save_class_distribution(target, output_path)

    assert result == output_path
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
