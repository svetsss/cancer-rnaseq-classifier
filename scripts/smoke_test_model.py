import json
import logging

import pandas as pd

from src.app_services import load_final_pipeline
from src.config import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)
EXPECTED_FEATURE_COUNT = 20_531
EXPECTED_CLASSES = ("BRCA", "COAD", "KIRC", "LUAD", "PRAD")


def main() -> None:
    """Load the committed model under its runtime and validate fitted metadata."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    manifest_path = PROJECT_ROOT / "results" / "final_evaluation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_record = manifest["artifacts"]["model"]
    model_path = PROJECT_ROOT / str(model_record["path"])
    pipeline = load_final_pipeline(
        model_path,
        expected_sha256=str(model_record["sha256"]),
        expected_runtime=manifest["runtime"],
    )

    feature_count = len(pipeline.feature_names_in_)
    classes = tuple(str(label) for label in pipeline.classes_)
    if feature_count != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(f"Unexpected model feature count: {feature_count}")
    if classes != EXPECTED_CLASSES:
        raise RuntimeError(f"Unexpected model classes: {classes}")

    demo_metadata = json.loads(
        (PROJECT_ROOT / "examples" / "demo_expected.json").read_text(encoding="utf-8")
    )
    demo_features = pd.read_csv(PROJECT_ROOT / "examples" / "demo_sample.csv")
    demo_prediction = str(pipeline.predict(demo_features)[0])
    if len(demo_features) != 1:
        raise RuntimeError("Demo CSV must contain exactly one sample")
    if demo_prediction != demo_metadata["expected_class"]:
        raise RuntimeError(
            "Demo prediction changed: "
            f"expected {demo_metadata['expected_class']}, received {demo_prediction}"
        )

    LOGGER.info("Loaded final model with %d features and classes %s", feature_count, classes)
    LOGGER.info("Demo sample prediction: %s", demo_prediction)


if __name__ == "__main__":
    main()
