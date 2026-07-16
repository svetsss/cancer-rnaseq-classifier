import pandas as pd

from src.config import METRICS_PATH, PROJECT_PIPELINE_PATH, README_MODEL_COMPARISON_PATH
from src.visualization import save_project_pipeline, save_readme_model_comparison


def main() -> None:
    """Rebuild the public overview figures from committed project artifacts."""
    metrics = pd.read_csv(METRICS_PATH)
    paths = (
        save_project_pipeline(PROJECT_PIPELINE_PATH),
        save_readme_model_comparison(metrics, README_MODEL_COMPARISON_PATH),
    )
    for path in paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
