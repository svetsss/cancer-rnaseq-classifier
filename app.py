import json
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

from src.app_services import (
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_ROWS,
    assess_distribution_shift,
    load_final_pipeline,
    predict_uploaded_features,
    read_uploaded_features,
    validate_uploaded_features,
)

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODEL_PATH = PROJECT_ROOT / "models" / "final_e10_pipeline.joblib"
DISCLAIMER = "Демонстрационный учебный результат. Не предназначено для медицинской диагностики."


@st.cache_resource
def get_pipeline() -> Pipeline:
    manifest = json.loads((RESULTS_DIR / "final_evaluation.json").read_text(encoding="utf-8"))
    model_record = manifest["artifacts"]["model"]
    return load_final_pipeline(
        MODEL_PATH,
        expected_sha256=str(model_record["sha256"]),
        expected_runtime=manifest["runtime"],
    )


@st.cache_data
def read_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Ожидался JSON-объект: {path.name}")
    return loaded


@st.cache_data
def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def show_overview() -> None:
    dataset = read_json(RESULTS_DIR / "dataset_summary.json")
    split = read_json(RESULTS_DIR / "split_summary.json")

    st.header("Обзор проекта")
    st.write(
        "Классификация пяти типов опухолей по данным RNA-Seq: BRCA, COAD, KIRC, "
        "LUAD и PRAD. Проект демонстрирует воспроизводимый исследовательский протокол."
    )
    columns = st.columns(4)
    columns[0].metric("Объекты", dataset["sample_count"])
    columns[1].metric("Исходные признаки", dataset["feature_count"])
    columns[2].metric("Train", split["train_samples"])
    columns[3].metric("Test", split["test_samples"])

    class_distribution = pd.Series(dataset["class_distribution"], name="Количество")
    st.dataframe(class_distribution.to_frame(), width="stretch")
    st.image(FIGURES_DIR / "class_distribution.png", caption="Распределение классов")
    st.warning(DISCLAIMER)


def show_experiments() -> None:
    metrics = read_csv(RESULTS_DIR / "metrics.csv")

    st.header("Эксперименты")
    model_options = ["Все", *sorted(metrics["model"].unique().tolist())]
    method_options = ["Все", *sorted(metrics["feature_method"].unique().tolist())]
    model_filter = st.selectbox("Модель", model_options)
    method_filter = st.selectbox("Метод представления", method_options)

    filtered = metrics
    if model_filter != "Все":
        filtered = filtered[filtered["model"] == model_filter]
    if method_filter != "Все":
        filtered = filtered[filtered["feature_method"] == method_filter]

    shown_columns = [
        "experiment_id",
        "model",
        "feature_method",
        "n_features",
        "cv_f1_macro_mean",
        "cv_f1_macro_std",
        "total_cv_time_seconds",
    ]
    st.dataframe(filtered[shown_columns], hide_index=True, width="stretch")
    st.success(
        "E10 был выбран до открытия test set: LogisticRegression после 20 PCA-компонент. "
        "PCA использует все 20 531 исходный признак."
    )
    st.image(
        [
            FIGURES_DIR / "feature_selection_comparison.png",
            FIGURES_DIR / "pca_comparison.png",
            FIGURES_DIR / "readme_model_comparison.png",
            FIGURES_DIR / "pca_train_projection.png",
        ],
        caption=[
            "SelectKBest",
            "PCA",
            "Сравнение семейств моделей",
            "Train-проекция PCA",
        ],
    )

    robustness = read_json(RESULTS_DIR / "robustness_e10.json")
    st.subheader("Устойчивость E10")
    robustness_table = pd.DataFrame(
        {
            "Показатель": [
                "Число train-only оценок",
                "Macro F1, mean",
                "Macro F1, std",
                "Macro F1, min",
                "Macro F1, max",
            ],
            "Значение": [
                robustness["evaluations"],
                robustness["f1_macro_mean"],
                robustness["f1_macro_std"],
                robustness["f1_macro_min"],
                robustness["f1_macro_max"],
            ],
        }
    )
    st.dataframe(robustness_table, hide_index=True, width="stretch")
    st.image(
        FIGURES_DIR / "e10_robustness.png",
        caption="50 validation folds: test set не использовался",
    )


def show_final_result() -> None:
    final_evaluation = read_json(RESULTS_DIR / "final_evaluation.json")
    classification_report = read_json(RESULTS_DIR / "final_classification_report.json")
    confusion_matrix = read_csv(RESULTS_DIR / "final_confusion_matrix.csv")
    test_metrics = final_evaluation["test_metrics"]
    if not isinstance(test_metrics, dict):
        raise ValueError("Некорректная структура финальных метрик")

    st.header("Финальный результат")
    columns = st.columns(4)
    columns[0].metric("Macro F1", f"{test_metrics['f1_macro']:.6f}")
    columns[1].metric("Accuracy", f"{test_metrics['accuracy']:.6f}")
    columns[2].metric("Precision macro", f"{test_metrics['precision_macro']:.6f}")
    columns[3].metric("Recall macro", f"{test_metrics['recall_macro']:.6f}")

    st.subheader("Classification report")
    st.dataframe(pd.DataFrame(classification_report).T, width="stretch")
    st.subheader("Confusion matrix")
    st.dataframe(confusion_matrix, hide_index=True, width="stretch")
    st.image(FIGURES_DIR / "final_confusion_matrix.png")

    statistical_context = read_json(RESULTS_DIR / "statistical_context.json")
    accuracy_interval = statistical_context["accuracy"]
    if isinstance(accuracy_interval, dict):
        st.info(
            "95% Wilson interval для accuracy: "
            f"{accuracy_interval['lower']:.3%}–{accuracy_interval['upper']:.3%}. "
            "Он отражает неопределённость из-за размера test set, но не учитывает batch effects."
        )

    timings = {
        "Обучение, с": final_evaluation["training_time_seconds"],
        "Predict, с": final_evaluation["prediction_time_seconds"],
        "Predict proba, с": final_evaluation["probability_prediction_time_seconds"],
    }
    st.subheader("Время и окружение")
    st.dataframe(pd.Series(timings, name="Значение").to_frame(), width="stretch")
    st.json(final_evaluation["runtime"])
    st.warning(
        "Результат относится к одному публичному набору и одному holdout split. "
        "Независимая когорта и batch effects не исследовались. " + DISCLAIMER
    )


def show_prediction_demo() -> None:
    st.header("Техническая smoke-проверка")
    try:
        pipeline = get_pipeline()
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        st.error("Не удалось загрузить зафиксированную модель в текущем окружении.")
        st.code(str(error))
        st.info("Для модели нужен Python 3.12.x и зависимости из extra `model-runtime`.")
        return

    expected_columns = [str(name) for name in pipeline.feature_names_in_]

    st.write(
        f"Загрузите уже подготовленную матрицу в формате UCI с 1–{MAX_UPLOAD_ROWS} строками "
        f"и ровно {len(expected_columns)} числовыми признаками. Raw counts, TPM и данные с "
        "другими gene IDs использовать нельзя: проект не содержит их нормализацию и mapping. "
        f"Максимальный размер файла — {MAX_UPLOAD_BYTES // (1024 * 1024)} МБ. "
        "Файл не сохраняется."
    )
    with st.expander("Пример имён столбцов"):
        st.code(", ".join(expected_columns[:10]) + ", ...")

    uploaded_file = st.file_uploader("CSV с признаками", type="csv")
    if uploaded_file is None:
        st.info(DISCLAIMER)
        return

    try:
        uploaded_features = read_uploaded_features(
            uploaded_file,
            uploaded_size=uploaded_file.size,
        )
        validated_features = validate_uploaded_features(uploaded_features, pipeline)
        shift_details, shift_summary = assess_distribution_shift(validated_features, pipeline)
        result = predict_uploaded_features(validated_features, pipeline)
    except (TypeError, ValueError) as error:
        st.error(str(error))
        return

    if shift_summary["warning_rows"]:
        st.error(
            f"Обнаружен возможный сдвиг распределения в {shift_summary['warning_rows']} "
            "строках. Предсказания для таких данных нельзя интерпретировать как надёжные."
        )
        st.dataframe(shift_details, width="stretch")

    st.subheader("Результат")
    st.dataframe(result, width="stretch")
    st.warning("Вероятности модели не калиброваны. " + DISCLAIMER)


def main() -> None:
    st.set_page_config(page_title="RNA-Seq classifier", layout="wide")
    st.title("Классификация типов опухолей по RNA-Seq")
    st.warning(DISCLAIMER)
    section = st.sidebar.radio(
        "Раздел",
        ["Обзор проекта", "Эксперименты", "Финальный результат", "Smoke-проверка"],
    )
    if section == "Обзор проекта":
        show_overview()
    elif section == "Эксперименты":
        show_experiments()
    elif section == "Финальный результат":
        show_final_result()
    else:
        show_prediction_demo()


if __name__ == "__main__":
    main()
