import hashlib
import platform
from pathlib import Path
from typing import TypedDict

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MAX_UPLOAD_ROWS = 100
PREDICTION_COLUMN = "Предсказанный моделью класс"


class DistributionShiftSummary(TypedDict):
    rows: int
    z_threshold: float
    warning_fraction_threshold: float
    maximum_absolute_z_score: float
    maximum_outlier_feature_fraction: float
    warning_rows: int


def load_final_pipeline(
    model_path: Path,
    *,
    expected_sha256: str | None = None,
    expected_runtime: dict[str, object] | None = None,
) -> Pipeline:
    """Verify the trusted model artifact and then load its fitted Pipeline."""
    if expected_sha256 is not None and _sha256_file(model_path) != expected_sha256:
        raise ValueError("SHA-256 файла модели не совпадает с финальным манифестом")
    if expected_runtime is not None:
        _validate_runtime(expected_runtime)

    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, Pipeline):
        raise ValueError("Файл модели не содержит sklearn Pipeline")
    if not hasattr(pipeline, "feature_names_in_") or not hasattr(pipeline, "classes_"):
        raise ValueError("Сохранённый Pipeline не содержит ожидаемую fitted metadata")
    return pipeline


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_runtime(expected_runtime: dict[str, object]) -> None:
    current_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    mismatches = [
        f"{name}: ожидается {expected_runtime[name]}, установлена {current_version}"
        for name, current_version in current_versions.items()
        if name in expected_runtime and str(expected_runtime[name]) != current_version
    ]
    if mismatches:
        raise RuntimeError("Окружение не совпадает с окружением модели; " + "; ".join(mismatches))


def validate_uploaded_features(
    uploaded_features: pd.DataFrame,
    pipeline: Pipeline,
    *,
    max_rows: int = MAX_UPLOAD_ROWS,
) -> pd.DataFrame:
    """Validate and order uploaded features for the saved Pipeline."""
    if uploaded_features.empty:
        raise ValueError("CSV должен содержать хотя бы одну строку данных")
    if len(uploaded_features) > max_rows:
        raise ValueError(f"CSV может содержать не более {max_rows} строк")
    if not uploaded_features.columns.is_unique:
        raise ValueError("Имена столбцов CSV должны быть уникальными")

    expected_columns = [str(name) for name in pipeline.feature_names_in_]
    uploaded_columns = [str(name) for name in uploaded_features.columns]
    missing_columns = [name for name in expected_columns if name not in uploaded_columns]
    extra_columns = [name for name in uploaded_columns if name not in expected_columns]
    if missing_columns or extra_columns:
        details = []
        if missing_columns:
            details.append(f"отсутствуют столбцы: {', '.join(missing_columns[:10])}")
        if extra_columns:
            details.append(f"лишние столбцы: {', '.join(extra_columns[:10])}")
        raise ValueError("Набор признаков не совпадает с моделью; " + "; ".join(details))

    ordered_features = uploaded_features.loc[:, expected_columns].copy()
    try:
        ordered_features = ordered_features.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Все значения признаков должны быть числовыми") from error

    if not np.isfinite(ordered_features.to_numpy(dtype=float)).all():
        raise ValueError("CSV не должен содержать пропуски или бесконечные значения")
    return ordered_features


def predict_uploaded_features(
    uploaded_features: pd.DataFrame,
    pipeline: Pipeline,
) -> pd.DataFrame:
    """Return predicted classes and ordered class probabilities."""
    features = validate_uploaded_features(uploaded_features, pipeline)
    predictions = pipeline.predict(features)
    probabilities = pipeline.predict_proba(features)
    class_order = [str(label) for label in pipeline.classes_]

    result = pd.DataFrame({PREDICTION_COLUMN: predictions}, index=features.index)
    for class_index, label in enumerate(class_order):
        result[f"prob_{label}"] = probabilities[:, class_index]
    return result


def assess_distribution_shift(
    uploaded_features: pd.DataFrame,
    pipeline: Pipeline,
    *,
    z_threshold: float = 5.0,
    warning_fraction_threshold: float = 0.01,
) -> tuple[pd.DataFrame, DistributionShiftSummary]:
    """Flag samples far outside the training scaler distribution."""
    features = validate_uploaded_features(uploaded_features, pipeline)
    scaler = pipeline.named_steps.get("scaler")
    if not isinstance(scaler, StandardScaler) or not hasattr(scaler, "mean_"):
        raise ValueError("Pipeline не содержит fitted StandardScaler для проверки сдвига")

    scale = np.asarray(scaler.scale_, dtype=float)
    safe_scale = np.where(scale == 0, 1.0, scale)
    standardized = (features.to_numpy(dtype=float) - np.asarray(scaler.mean_)) / safe_scale
    absolute_z = np.abs(standardized)
    row_fraction = (absolute_z > z_threshold).mean(axis=1)
    row_maximum = absolute_z.max(axis=1)
    warning_mask = row_fraction > warning_fraction_threshold

    details = pd.DataFrame(
        {
            "max_abs_z": row_maximum,
            "outlier_feature_fraction": row_fraction,
            "distribution_shift_warning": warning_mask,
        },
        index=features.index,
    )
    summary: DistributionShiftSummary = {
        "rows": len(features),
        "z_threshold": z_threshold,
        "warning_fraction_threshold": warning_fraction_threshold,
        "maximum_absolute_z_score": float(row_maximum.max()),
        "maximum_outlier_feature_fraction": float(row_fraction.max()),
        "warning_rows": int(warning_mask.sum()),
    }
    return details, summary
