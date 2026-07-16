from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

MAX_UPLOAD_ROWS = 100
PREDICTION_COLUMN = "Предсказанный моделью класс"


def load_final_pipeline(model_path: Path) -> Pipeline:
    """Load the saved final Pipeline used by the application."""
    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, Pipeline):
        raise ValueError("Файл модели не содержит sklearn Pipeline")
    if not hasattr(pipeline, "feature_names_in_") or not hasattr(pipeline, "classes_"):
        raise ValueError("Сохранённый Pipeline не содержит ожидаемую fitted metadata")
    return pipeline


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
