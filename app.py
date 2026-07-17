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
GITHUB_URL = "https://github.com/svetsss/cancer-rnaseq-classifier"
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


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: #f6f8fb;
            overflow-x: hidden;
        }
        [data-testid="stMain"] {
            width: 100%;
            min-width: 0;
        }
        .block-container {
            width: 100%;
            max-width: 1180px;
            min-width: 0;
            box-sizing: border-box;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }
        [data-testid="stMetric"] {
            min-height: 118px;
            padding: 1rem 1.1rem;
            background: #ffffff;
            border: 1px solid #dfe5ec;
            border-top: 3px solid #b84a4a;
            border-radius: 12px;
            box-shadow: 0 4px 18px rgba(24, 38, 58, 0.05);
        }
        [data-testid="stMetricLabel"] {
            color: #5b6878;
            font-size: 0.84rem;
            font-weight: 600;
        }
        [data-testid="stMetricValue"] {
            color: #172033;
            font-weight: 650;
        }
        [data-testid="stDataFrame"], [data-testid="stFileUploader"] {
            overflow: hidden;
            background: #ffffff;
            border: 1px solid #dfe5ec;
            border-radius: 12px;
        }
        [data-testid="stAlert"] {
            border-radius: 10px;
        }
        h2, h3 {
            color: #172033;
            letter-spacing: -0.025em;
        }
        .project-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .project-header strong {
            display: block;
            color: #172033;
            font-size: 1.05rem;
        }
        .project-header span {
            color: #6b7788;
            font-size: 0.84rem;
        }
        .project-header a {
            padding: 0.5rem 0.75rem;
            color: #39475a !important;
            background: #ffffff;
            border: 1px solid #d9e0e8;
            border-radius: 8px;
            font-size: 0.86rem;
            text-decoration: none;
        }
        .project-header a:hover {
            color: #172033 !important;
            border-color: #9aa6b5;
        }
        .project-hero {
            overflow: hidden;
            margin-bottom: 0.85rem;
            padding: 2.1rem 2.3rem;
            color: #ffffff;
            background: #172033;
            border-left: 6px solid #c65a5a;
            border-radius: 16px;
            box-shadow: 0 12px 30px rgba(23, 32, 51, 0.14);
        }
        .project-hero__label {
            margin: 0 0 0.55rem;
            color: #e2a6a6;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .project-hero h1 {
            max-width: 820px;
            margin: 0;
            color: #ffffff !important;
            font-size: clamp(1.9rem, 3.6vw, 2.8rem);
            line-height: 1.05;
            letter-spacing: -0.045em;
            overflow-wrap: anywhere;
        }
        .project-hero__classes {
            margin: 0.9rem 0 1.3rem;
            color: #d7dee8;
            font-size: 1rem;
        }
        .project-hero__facts {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
        }
        .project-hero__facts span {
            padding: 0.42rem 0.72rem;
            color: #f8fafc;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 999px;
            font-size: 0.82rem;
        }
        .project-note {
            margin-bottom: 2rem;
            padding: 0.72rem 1rem;
            color: #596576;
            background: #ffffff;
            border: 1px solid #dfe5ec;
            border-radius: 10px;
            font-size: 0.88rem;
        }
        .section-intro {
            max-width: 820px;
            margin-top: -0.45rem;
            margin-bottom: 1.35rem;
            color: #667385;
            font-size: 0.98rem;
        }
        [data-testid="stPills"] {
            margin: 0.1rem 0 2.2rem;
            padding: 0.4rem;
            background: #ffffff;
            border: 1px solid #dfe5ec;
            border-radius: 11px;
        }
        .app-footer {
            margin-top: 3rem;
            padding-top: 1.1rem;
            color: #788495;
            border-top: 1px solid #dfe5ec;
            font-size: 0.82rem;
        }
        @media (max-width: 1200px) {
            .block-container {
                max-width: 100%;
                padding-right: 2rem;
                padding-left: 2rem;
            }
            .project-hero h1 {
                font-size: 2.2rem;
            }
        }
        @media (max-width: 800px) {
            .block-container {
                padding-top: 1rem;
                padding-right: 1rem;
                padding-left: 1rem;
            }
            .project-header {
                align-items: flex-start;
                flex-direction: column;
            }
            .project-hero {
                padding: 1.5rem;
            }
            [data-testid="stMetric"] {
                min-height: 100px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_hero() -> None:
    st.markdown(
        """
        <section class="project-hero">
            <p class="project-hero__label">Профиль экспрессии генов</p>
            <h1>Классификация опухолевой ткани по RNA-Seq</h1>
            <p class="project-hero__classes">BRCA · COAD · KIRC · LUAD · PRAD</p>
            <div class="project-hero__facts">
                <span>801 образец</span>
                <span>20 531 признак</span>
                <span>18 конфигураций</span>
                <span>Macro F1: 1.0000</span>
            </div>
        </section>
        <div class="project-note">
            Исследовательский учебный проект · Не предназначен для медицинской диагностики
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_navigation() -> str:
    st.markdown(
        f"""
        <div class="project-header">
            <div>
                <strong>RNA-Seq classifier</strong>
                <span>Старинская Светлана · ФИИТ · 2 курс</span>
            </div>
            <a href="{GITHUB_URL}" target="_blank">
                Исходный код на GitHub
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    show_hero()
    options = ["Обзор проекта", "Эксперименты", "Финальный результат", "Smoke-проверка"]
    section = st.pills(
        "Раздел",
        options,
        default=options[0],
        selection_mode="single",
        label_visibility="collapsed",
    )
    return str(section) if section is not None else options[0]


def show_section_intro(text: str) -> None:
    st.markdown(f'<p class="section-intro">{text}</p>', unsafe_allow_html=True)


def show_overview() -> None:
    dataset = read_json(RESULTS_DIR / "dataset_summary.json")
    split = read_json(RESULTS_DIR / "split_summary.json")

    st.header("Обзор проекта")
    show_section_intro(
        "Классификация пяти типов опухолей по данным RNA-Seq: BRCA, COAD, KIRC, "
        "LUAD и PRAD. Размеры выборок и распределение классов зафиксированы в артефактах."
    )
    columns = st.columns(2, gap="large")
    columns[0].metric("Объекты", str(dataset["sample_count"]))
    columns[1].metric("Исходные признаки", str(dataset["feature_count"]))
    columns[0].metric("Train", str(split["train_samples"]))
    columns[1].metric("Test", str(split["test_samples"]))

    class_distribution = pd.Series(dataset["class_distribution"], name="Количество")
    table_column, chart_column = st.columns([0.8, 1.4], gap="large")
    with table_column:
        st.subheader("Классы")
        st.dataframe(class_distribution.to_frame(), width="stretch")
    with chart_column:
        st.subheader("Распределение")
        st.image(
            FIGURES_DIR / "class_distribution.png",
            caption="Количество образцов каждого класса",
            use_container_width=True,
        )


def show_experiments() -> None:
    metrics = read_csv(RESULTS_DIR / "metrics.csv")

    st.header("Эксперименты")
    show_section_intro(
        "Сравнение моделей выполнялось только на train-части. Все преобразования обучались "
        "заново внутри каждого fold."
    )
    model_options = ["Все", *sorted(metrics["model"].unique().tolist())]
    method_options = ["Все", *sorted(metrics["feature_method"].unique().tolist())]
    model_column, method_column = st.columns(2, gap="large")
    with model_column:
        model_filter = st.selectbox("Модель", model_options)
    with method_column:
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
    st.subheader("Сравнение методов")
    left_charts, right_charts = st.columns(2, gap="large")
    with left_charts:
        st.image(
            FIGURES_DIR / "feature_selection_comparison.png",
            caption="SelectKBest",
            use_container_width=True,
        )
        st.image(
            FIGURES_DIR / "readme_model_comparison.png",
            caption="Сравнение семейств моделей",
            use_container_width=True,
        )
    with right_charts:
        st.image(
            FIGURES_DIR / "pca_comparison.png",
            caption="PCA",
            use_container_width=True,
        )
        st.image(
            FIGURES_DIR / "pca_train_projection.png",
            caption="Train-проекция PCA",
            use_container_width=True,
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
    robustness_values, robustness_chart = st.columns([0.8, 1.4], gap="large")
    with robustness_values:
        st.dataframe(robustness_table, hide_index=True, width="stretch")
    with robustness_chart:
        st.image(
            FIGURES_DIR / "e10_robustness.png",
            caption="50 validation folds: test set не использовался",
            use_container_width=True,
        )


def show_final_result() -> None:
    final_evaluation = read_json(RESULTS_DIR / "final_evaluation.json")
    classification_report = read_json(RESULTS_DIR / "final_classification_report.json")
    confusion_matrix = read_csv(RESULTS_DIR / "final_confusion_matrix.csv")
    test_metrics = final_evaluation["test_metrics"]
    if not isinstance(test_metrics, dict):
        raise ValueError("Некорректная структура финальных метрик")

    st.header("Финальный результат")
    show_section_intro(
        "Однократная оценка выбранного Pipeline на зафиксированной test-выборке из 161 образца."
    )
    columns = st.columns(2, gap="large")
    columns[0].metric("Macro F1", f"{test_metrics['f1_macro']:.6f}")
    columns[1].metric("Accuracy", f"{test_metrics['accuracy']:.6f}")
    columns[0].metric("Precision macro", f"{test_metrics['precision_macro']:.6f}")
    columns[1].metric("Recall macro", f"{test_metrics['recall_macro']:.6f}")

    report_column, matrix_column = st.columns([1.15, 1], gap="large")
    with report_column:
        st.subheader("Classification report")
        st.dataframe(pd.DataFrame(classification_report).T, width="stretch")
    with matrix_column:
        st.subheader("Confusion matrix")
        st.dataframe(confusion_matrix, hide_index=True, width="stretch")
        st.image(FIGURES_DIR / "final_confusion_matrix.png", use_container_width=True)

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
    with st.expander("Время выполнения и окружение"):
        st.dataframe(pd.Series(timings, name="Значение").to_frame(), width="stretch")
        st.json(final_evaluation["runtime"])
    st.warning(
        "Результат относится к одному публичному набору и одному holdout split. "
        "Независимая когорта и batch effects не исследовались. " + DISCLAIMER
    )


def show_prediction_demo() -> None:
    st.header("Техническая smoke-проверка")
    show_section_intro(
        "Проверка загрузки опубликованного sklearn Pipeline и прогноза для подготовленной "
        "матрицы признаков."
    )
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
        st.caption(DISCLAIMER)
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


def show_footer() -> None:
    st.markdown(
        """
        <div class="app-footer">
            Данные: UCI Gene Expression Cancer RNA-Seq · Код и артефакты опубликованы на GitHub
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="RNA-Seq classifier",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()
    section = show_navigation()
    if section == "Обзор проекта":
        show_overview()
    elif section == "Эксперименты":
        show_experiments()
    elif section == "Финальный результат":
        show_final_result()
    else:
        show_prediction_demo()
    show_footer()


if __name__ == "__main__":
    main()
