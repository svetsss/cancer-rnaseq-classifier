import json
from pathlib import Path
from typing import Any

import altair as alt
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
ACCENT_COLOR = "#b84a4a"
NAVY_COLOR = "#172033"
MODEL_COLORS = ["#172033", "#b84a4a", "#4d718c", "#8b6f47", "#6f7b8a"]


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


def style_chart(chart: Any, *, height: int) -> Any:
    return (
        chart.properties(height=height)
        .configure_view(stroke=None)
        .configure_axis(
            gridColor="#e7ebf0",
            labelColor="#4f5c6d",
            titleColor="#4f5c6d",
            domainColor="#cfd6df",
            tickColor="#cfd6df",
        )
        .configure_legend(labelColor="#4f5c6d", titleColor="#4f5c6d")
    )


def build_class_distribution_chart(class_distribution: pd.Series) -> Any:
    chart_data = class_distribution.rename_axis("Класс").reset_index(name="Образцы")
    chart_data["Доля"] = chart_data["Образцы"] / chart_data["Образцы"].sum()
    base = alt.Chart(chart_data).encode(
        y=alt.Y("Класс:N", sort="-x", title=None),
        x=alt.X("Образцы:Q", title="Число образцов"),
        tooltip=[
            alt.Tooltip("Класс:N"),
            alt.Tooltip("Образцы:Q", format=",d"),
            alt.Tooltip("Доля:Q", format=".1%"),
        ],
    )
    bars = base.mark_bar(color=ACCENT_COLOR, cornerRadiusEnd=5, height=24)
    labels = base.mark_text(align="left", baseline="middle", dx=7, color=NAVY_COLOR).encode(
        text=alt.Text("Образцы:Q", format=",d")
    )
    return style_chart(bars + labels, height=260)


def build_experiment_chart(metrics: pd.DataFrame, *, x_field: str) -> Any:
    chart_data = metrics[metrics["model"] != "DummyClassifier"].copy()
    chart_data["f1_low"] = chart_data["cv_f1_macro_mean"] - chart_data["cv_f1_macro_std"]
    chart_data["f1_high"] = chart_data["cv_f1_macro_mean"] + chart_data["cv_f1_macro_std"]
    x_title = "Число компонент / признаков" if x_field == "n_features" else "Время CV, с"
    x_scale = (
        alt.Scale(type="log", domain=[15, 30000])
        if x_field == "n_features"
        else alt.Scale(zero=False)
    )
    y_scale = alt.Scale(domain=[0.975, 1.001], clamp=True)
    common = {
        "x": alt.X(f"{x_field}:Q", title=x_title, scale=x_scale),
        "color": alt.Color(
            "model:N",
            title="Модель",
            scale=alt.Scale(range=MODEL_COLORS),
            legend=alt.Legend(orient="bottom", columns=2),
        ),
        "tooltip": [
            alt.Tooltip("experiment_id:N", title="Эксперимент"),
            alt.Tooltip("model:N", title="Модель"),
            alt.Tooltip("feature_method:N", title="Представление"),
            alt.Tooltip("n_features:Q", title="Компоненты / признаки", format=",d"),
            alt.Tooltip("cv_f1_macro_mean:Q", title="Macro F1", format=".4f"),
            alt.Tooltip("cv_f1_macro_std:Q", title="Std", format=".4f"),
            alt.Tooltip("total_cv_time_seconds:Q", title="Время CV, с", format=".2f"),
        ],
    }
    points = (
        alt.Chart(chart_data)
        .mark_circle(size=115, opacity=0.82)
        .encode(
            y=alt.Y(
                "cv_f1_macro_mean:Q",
                title="CV Macro F1",
                scale=y_scale,
            ),
            shape=alt.Shape(
                "feature_method:N",
                title="Представление",
                legend=alt.Legend(orient="bottom", columns=2),
            ),
            **common,
        )
    )
    error_bars = (
        alt.Chart(chart_data)
        .mark_rule(opacity=0.42)
        .encode(
            y=alt.Y("f1_low:Q", scale=y_scale, title="CV Macro F1"),
            y2=alt.Y2("f1_high:Q"),
            detail="experiment_id:N",
            **common,
        )
    )
    selected = (
        alt.Chart(chart_data[chart_data["experiment_id"] == "E10"])
        .mark_circle(size=300, fillOpacity=0, stroke=ACCENT_COLOR, strokeWidth=3)
        .encode(
            y=alt.Y("cv_f1_macro_mean:Q", scale=y_scale),
            x=alt.X(f"{x_field}:Q", scale=x_scale),
            tooltip=[alt.Tooltip("experiment_id:N", title="Выбранная конфигурация")],
        )
    )
    return style_chart(error_bars + points + selected, height=330)


def build_robustness_chart(robustness: dict[str, object]) -> Any:
    fold_values = robustness["fold_f1_macro"]
    if not isinstance(fold_values, list):
        raise ValueError("Некорректные fold-метрики robustness")
    chart_data = pd.DataFrame(
        {
            "Проверка": range(1, len(fold_values) + 1),
            "Macro F1": [float(value) for value in fold_values],
        }
    )
    mean = float(str(robustness["f1_macro_mean"]))
    std = float(str(robustness["f1_macro_std"]))
    y_scale = alt.Scale(domain=[0.985, 1.001])
    band_data = pd.DataFrame({"lower": [mean - std], "upper": [mean + std]})
    band = (
        alt.Chart(band_data)
        .mark_rect(color=ACCENT_COLOR, opacity=0.1)
        .encode(
            y=alt.Y("lower:Q", scale=y_scale, title="Macro F1"),
            y2="upper:Q",
        )
    )
    mean_line = (
        alt.Chart(pd.DataFrame({"mean": [mean]}))
        .mark_rule(
            color=ACCENT_COLOR,
            strokeDash=[6, 4],
        )
        .encode(y=alt.Y("mean:Q", scale=y_scale, title="Macro F1"))
    )
    folds = (
        alt.Chart(chart_data)
        .mark_line(
            color=NAVY_COLOR,
            point=alt.OverlayMarkDef(color=ACCENT_COLOR, size=48),
        )
        .encode(
            x=alt.X("Проверка:Q", title="Validation fold"),
            y=alt.Y(
                "Macro F1:Q",
                scale=y_scale,
                title="Macro F1",
            ),
            tooltip=[
                alt.Tooltip("Проверка:Q", format="d"),
                alt.Tooltip("Macro F1:Q", format=".4f"),
            ],
        )
    )
    return style_chart(band + mean_line + folds, height=310)


def build_mlp_loss_chart(training_analysis: dict[str, object], *, fold_number: int) -> Any:
    fold_histories = training_analysis["fold_histories"]
    selected_epochs = training_analysis["fold_epochs"]
    if not isinstance(fold_histories, list) or not isinstance(selected_epochs, list):
        raise ValueError("Некорректная история обучения MLP")
    history = fold_histories[fold_number - 1]
    if not isinstance(history, list):
        raise ValueError("Некорректная fold-история MLP")
    chart_data = pd.DataFrame(history)[["epoch", "train_loss", "validation_loss"]].melt(
        id_vars="epoch",
        var_name="Выборка",
        value_name="Loss",
    )
    chart_data["Выборка"] = chart_data["Выборка"].replace(
        {"train_loss": "Train", "validation_loss": "Validation"}
    )
    lines = (
        alt.Chart(chart_data)
        .mark_line(point=True)
        .encode(
            x=alt.X("epoch:Q", title="Эпоха"),
            y=alt.Y("Loss:Q", title="Cross-entropy loss", scale=alt.Scale(type="log")),
            color=alt.Color(
                "Выборка:N",
                title=None,
                scale=alt.Scale(domain=["Train", "Validation"], range=[NAVY_COLOR, ACCENT_COLOR]),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("epoch:Q", title="Эпоха", format="d"),
                alt.Tooltip("Выборка:N"),
                alt.Tooltip("Loss:Q", format=".6f"),
            ],
        )
    )
    selected_epoch = int(selected_epochs[fold_number - 1])
    marker = (
        alt.Chart(pd.DataFrame({"Эпоха": [selected_epoch]}))
        .mark_rule(color=ACCENT_COLOR, strokeDash=[6, 4])
        .encode(x=alt.X("Эпоха:Q"))
    )
    return style_chart(lines + marker, height=310)


def build_confusion_matrix_chart(confusion_matrix: pd.DataFrame) -> Any:
    chart_data = confusion_matrix.melt(
        id_vars="true_label",
        var_name="Предсказанный класс",
        value_name="Объекты",
    ).rename(columns={"true_label": "Истинный класс"})
    base = alt.Chart(chart_data).encode(
        x=alt.X("Предсказанный класс:N", title="Предсказанный класс"),
        y=alt.Y("Истинный класс:N", title="Истинный класс"),
        tooltip=[
            alt.Tooltip("Истинный класс:N"),
            alt.Tooltip("Предсказанный класс:N"),
            alt.Tooltip("Объекты:Q", format="d"),
        ],
    )
    heatmap = base.mark_rect().encode(
        color=alt.Color(
            "Объекты:Q",
            title="Объекты",
            scale=alt.Scale(range=["#f2f5f8", NAVY_COLOR]),
        )
    )
    labels = base.mark_text(fontSize=16, fontWeight=600).encode(
        text=alt.Text("Объекты:Q", format="d"),
        color=alt.condition(alt.datum["Объекты"] > 0, alt.value("white"), alt.value("#697586")),
    )
    return style_chart(heatmap + labels, height=370)


def build_confidence_chart(predictions: pd.DataFrame) -> Any:
    probability_columns = [column for column in predictions if column.startswith("prob_")]
    chart_data = predictions[["sample_id", "true_label", *probability_columns]].copy()
    chart_data["Уверенность"] = chart_data[probability_columns].max(axis=1)
    points = (
        alt.Chart(chart_data)
        .mark_circle(
            color=NAVY_COLOR,
            opacity=0.28,
            size=48,
        )
        .encode(
            x=alt.X("true_label:N", title="Истинный класс"),
            y=alt.Y(
                "Уверенность:Q",
                title="Максимальная вероятность",
                scale=alt.Scale(domain=[0.5, 1.005]),
            ),
            tooltip=[
                alt.Tooltip("sample_id:N", title="Образец"),
                alt.Tooltip("true_label:N", title="Класс"),
                alt.Tooltip("Уверенность:Q", format=".4f"),
            ],
        )
    )
    boxplot = (
        alt.Chart(chart_data)
        .mark_boxplot(
            color=ACCENT_COLOR,
            extent="min-max",
            size=34,
        )
        .encode(
            x=alt.X("true_label:N", axis=None),
            y=alt.Y(
                "Уверенность:Q",
                scale=alt.Scale(domain=[0.5, 1.005]),
                axis=None,
            ),
        )
    )
    return style_chart(points + boxplot, height=330)


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
        st.altair_chart(
            build_class_distribution_chart(class_distribution),
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
    st.subheader("Качество конфигураций")
    show_section_intro(
        "Точки показывают средний Macro F1, вертикальные линии — стандартное отклонение. "
        "Ось числа признаков логарифмическая; выбранная конфигурация E10 обведена красным."
    )
    quality_column, time_column = st.columns(2, gap="large")
    with quality_column:
        st.caption("Macro F1 и размер представления")
        st.altair_chart(
            build_experiment_chart(filtered, x_field="n_features"),
            use_container_width=True,
        )
    with time_column:
        st.caption("Macro F1 и полное время cross-validation")
        st.altair_chart(
            build_experiment_chart(filtered, x_field="total_cv_time_seconds"),
            use_container_width=True,
        )

    with st.expander("PCA-проекция train-выборки"):
        st.image(
            FIGURES_DIR / "pca_train_projection.png",
            caption="Проекция обучающей выборки на первые две PCA-компоненты",
            use_container_width=True,
        )

    with st.expander("PyTorch MLP: обучение по эпохам"):
        mlp_training = read_json(RESULTS_DIR / "mlp_training_analysis.json")
        fold_number = st.select_slider("Fold", options=[1, 2, 3, 4, 5], value=1)
        st.altair_chart(
            build_mlp_loss_chart(mlp_training, fold_number=fold_number),
            use_container_width=True,
        )
        selected_epochs = mlp_training["fold_epochs"]
        if isinstance(selected_epochs, list):
            st.caption(
                f"Для fold {fold_number} сохранено состояние после эпохи "
                f"{selected_epochs[fold_number - 1]}; дальнейшие эпохи относятся к patience "
                "механизма early stopping."
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
    robustness_values, robustness_chart = st.columns([0.72, 1.7], gap="large")
    with robustness_values:
        st.dataframe(robustness_table, hide_index=True, width="stretch")
    with robustness_chart:
        st.altair_chart(
            build_robustness_chart(robustness),
            use_container_width=True,
        )


def show_final_result() -> None:
    final_evaluation = read_json(RESULTS_DIR / "final_evaluation.json")
    classification_report = read_json(RESULTS_DIR / "final_classification_report.json")
    confusion_matrix = read_csv(RESULTS_DIR / "final_confusion_matrix.csv")
    predictions = read_csv(RESULTS_DIR / "final_test_predictions.csv")
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

    report_column, matrix_column = st.columns([1, 1.3], gap="large")
    with report_column:
        st.subheader("Classification report")
        st.dataframe(pd.DataFrame(classification_report).T, width="stretch")
    with matrix_column:
        st.subheader("Confusion matrix")
        st.altair_chart(
            build_confusion_matrix_chart(confusion_matrix),
            use_container_width=True,
        )

    st.subheader("Уверенность на test-выборке")
    show_section_intro(
        "Распределение максимальной вероятности для 161 объекта. Точки позволяют увидеть "
        "отдельные менее уверенные, но правильно классифицированные образцы."
    )
    st.altair_chart(build_confidence_chart(predictions), use_container_width=True)

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
