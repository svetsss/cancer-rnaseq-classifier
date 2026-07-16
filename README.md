# Классификация типов опухолей по данным RNA-Seq

**Автор:** Старинская Светлана Романовна<br>
**Курс:** 2 курс<br>
**Группа:** ФИИТ

Проект классифицирует образцы RNA-Seq по профилю экспрессии генов. Для каждого образца
модель определяет один из пяти классов опухолевой ткани: BRCA, COAD, KIRC, LUAD или PRAD.

## Функциональность

- автоматическая загрузка и проверка целостности датасета UCI;
- анализ распределения классов и качества данных;
- воспроизводимое разделение данных на train и test;
- сравнение классических моделей и нейронной сети;
- отбор признаков с помощью `SelectKBest` и снижение размерности с помощью PCA;
- cross-validation с преобразованиями внутри каждого fold;
- сохранение метрик, графиков, предсказаний и обученного Pipeline;
- Jupyter notebooks для анализа и демонстрации проекта;
- Streamlit-интерфейс для просмотра результатов и проверки модели.

## Как работает проект

Последовательность работы:

1. матрица RNA-Seq проходит проверку структуры и целостности;
2. выполняется стратифицированное разделение 80/20;
3. модели сравниваются только на train-части с помощью 5-fold cross-validation;
4. выбранный Pipeline обучается на полном train и оценивается на test;
5. метрики, predictions, модель и графики сохраняются как артефакты проекта.

## Датасет

Используется набор данных
[Gene Expression Cancer RNA-Seq](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq)
из UCI Machine Learning Repository.

Характеристики датасета:

| Параметр | Значение |
|---|---:|
| Образцы | 801 |
| Признаки экспрессии | 20 531 |
| Классы | 5 |
| Пропущенные значения | 0 |
| Константные признаки | 267 |

Классы:

- `BRCA` — инвазивная карцинома молочной железы;
- `COAD` — аденокарцинома толстой кишки;
- `KIRC` — светлоклеточный рак почки;
- `LUAD` — аденокарцинома лёгкого;
- `PRAD` — аденокарцинома предстательной железы.

![Распределение классов](figures/class_distribution.png)

BRCA — самый крупный класс: 300 образцов. Самый малочисленный класс COAD содержит 78
образцов, поэтому основной метрикой выбрана macro F1: она одинаково учитывает каждый класс.

## Модели

В экспериментальной части сравниваются:

- `DummyClassifier`;
- `LogisticRegression`;
- `LinearSVC`;
- `RandomForestClassifier`;
- многослойный перцептрон на PyTorch.

Основная метрика — `macro F1`, поскольку классы представлены в датасете неравномерно.
Модели оцениваются с помощью стратифицированной 5-fold cross-validation на train-части.
Стандартизация, PCA и отбор признаков входят в Pipeline и обучаются отдельно внутри
каждого fold.

Лучшая конфигурация классической модели:

```text
StandardScaler
PCA(n_components=20, random_state=42)
LogisticRegression(max_iter=3000, random_state=42)
```

Основные результаты train cross-validation:

| Эксперимент | Модель | Представление | CV macro F1 |
|---|---|---|---:|
| E01 | LogisticRegression | 20 531 признаков | 0.9954 ± 0.0065 |
| E08 | LinearSVC | SelectKBest, 200 | 1.0000 ± 0.0000 |
| E10 | LogisticRegression | PCA, 20 | 1.0000 ± 0.0000 |
| E16 | Random Forest | SelectKBest, 200 | 0.9961 ± 0.0032 |
| E17 | PyTorch MLP | SelectKBest, 200 | 0.9987 ± 0.0026 |

![Сравнение основных моделей](figures/readme_model_comparison.png)

Точка показывает средний macro F1 по пяти folds, горизонтальный отрезок — стандартное
отклонение. E08 и E10 получили максимальный результат. Для финального Pipeline выбран E10
с представлением из 20 PCA-компонент.

### Проверка устойчивости E10

Дополнительно E10 проверен с помощью `RepeatedStratifiedKFold`: 10 повторов по 5 folds,
всего 50 оценок на train-части. Зафиксированный test set в этом анализе не использовался.

| Показатель | Значение |
|---|---:|
| Число оценок | 50 |
| Macro F1, mean ± std | 0.9980 ± 0.0045 |
| Минимальный Macro F1 | 0.9873 |
| Максимальный Macro F1 | 1.0000 |

![Устойчивость E10 при repeated cross-validation](figures/e10_robustness.png)

Каждая точка соответствует одному validation fold, пунктир показывает среднее значение.
Высокий результат сохраняется при разных разбиениях train-выборки, однако такая проверка
не заменяет оценку на независимой когорте.

### Как меняется представление данных

Перед LogisticRegression признаки стандартизируются, затем PCA преобразует 20 531 исходный
признак в 20 компонент. Ниже показана двумерная проекция train-выборки только для визуального
анализа расположения классов.

![PCA-проекция обучающей выборки](figures/pca_train_projection.png)

Цвет обозначает класс опухолевой ткани. На проекции особенно хорошо отделяется KIRC;
остальные классы частично пересекаются в двух измерениях. Финальная модель использует 20
компонент, поэтому эта двумерная картинка является только иллюстрацией структуры данных.

## Анализ обучения MLP

Для MLP сохраняются `train loss`, `validation loss`, `validation macro F1` и accuracy после
каждой эпохи. Early stopping выполняется отдельно в каждом внешнем fold.

| Fold | Выбранная эпоха | Остановка | Macro F1 | Accuracy |
|---:|---:|---:|---:|---:|
| 1 | 19 | 29 | 0.9935 | 0.9922 |
| 2 | 16 | 26 | 1.0000 | 1.0000 |
| 3 | 4 | 14 | 1.0000 | 1.0000 |
| 4 | 4 | 14 | 1.0000 | 1.0000 |
| 5 | 11 | 21 | 1.0000 | 1.0000 |

Средний CV macro F1 обновлённого MLP-протокола: `0.9987 ± 0.0026`.

### Динамика функции потерь

![Train loss и validation loss](figures/mlp_loss_curves.png)

Синяя линия показывает средний train loss, оранжевая — средний validation loss. Заливка
показывает диапазон значений между folds. Обе кривые быстро снижаются в первые эпохи.

### Macro F1 по эпохам

![Validation macro F1 по эпохам](figures/mlp_validation_f1.png)

Зелёная линия показывает средний validation macro F1, заливка — минимальное и максимальное
значение среди folds. Уже на первых эпохах среднее превышает 0.98, затем достигает 1.0.

### Выбор эпохи и early stopping

![Выбор эпохи и early stopping](figures/mlp_epoch_selection.png)

Синие колонки обозначают эпоху с минимальным validation loss. Жёлтые колонки показывают
момент остановки после `patience = 10`. Разное число эпох по folds подтверждает пользу
отдельного early stopping для каждого разбиения.

### Метрики по folds

![Macro F1 и accuracy по folds](figures/mlp_fold_metrics.png)

Зелёные колонки — macro F1, фиолетовые — accuracy на outer-validation. В первом fold
получены 0.9935 и 0.9922, в остальных четырёх folds обе метрики равны 1.0.

## Результаты

Финальная модель обучена на 640 train-образцах и оценена на 161 test-образце.

| Метрика | Значение |
|---|---:|
| Macro F1 | 1.000000 |
| Accuracy | 1.000000 |
| Precision macro | 1.000000 |
| Recall macro | 1.000000 |

95% интервал Wilson для accuracy: `0.9767–1.0000`.

![Confusion matrix](figures/final_confusion_matrix.png)

Все значения находятся на главной диагонали confusion matrix: 161 test-образец отнесён к
своему классу. Числа в строках соответствуют фактическому количеству объектов каждого
класса в test-выборке.

Полные результаты экспериментов находятся в [results](results), а описание протокола — в
[журнале экспериментов](docs/experiment_log.md).

## Jupyter notebook

- [cancer_rnaseq_project.ipynb](notebooks/cancer_rnaseq_project.ipynb) — полный разбор
  датасета, моделей и результатов.

## Структура проекта

```text
.
├── src/                 # загрузка данных, модели и оценка
├── scripts/             # запуск этапов эксперимента
├── notebooks/           # исследовательский и демонстрационный notebooks
├── results/             # сохранённые метрики и предсказания
├── figures/             # графики с результатами экспериментов
├── models/              # обученный sklearn Pipeline
├── docs/                # методика и описание экспериментов
├── tests/               # автоматические тесты
├── app.py               # Streamlit-приложение
└── pyproject.toml        # зависимости и настройки проекта
```

## Технологии

- Python 3.11–3.12;
- NumPy и pandas;
- scikit-learn;
- PyTorch;
- Matplotlib;
- JupyterLab;
- Streamlit;
- pytest, Ruff и mypy.

## Установка

Клонирование репозитория:

```bash
git clone https://github.com/svetsss/cancer-rnaseq-classifier.git
cd cancer-rnaseq-classifier
```

Создание виртуального окружения:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Для Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Установка проекта с инструментами анализа и тестирования:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,neural,notebook]"
```

## Запуск анализа

Основные этапы запускаются последовательно:

```bash
python -m scripts.analyze_dataset
python -m scripts.run_baselines
python -m scripts.run_feature_selection
python -m scripts.run_extended_experiments
python -m scripts.run_mlp_training_analysis --publish
python -m scripts.run_robustness_analysis --repeats 10 --publish
```

Новые результаты сохраняются в каталоге `runs/`, а опубликованные результаты в `results/`
остаются зафиксированными.

Запуск JupyterLab:

```bash
jupyter lab
```

## Streamlit-приложение

Для приложения с сохранённой моделью используется Python 3.12.13 и зафиксированное
окружение модели:

```bash
python -m pip install -e ".[app,model-runtime]"
streamlit run app.py
```

Приложение содержит:

- обзор датасета;
- таблицу экспериментов с фильтрами;
- итоговые метрики и confusion matrix;
- техническую проверку обученного Pipeline на подготовленной матрице признаков.

## Проверка проекта

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy src
python -m scripts.verify_artifacts
python -m pytest -q
```

## Источник данных

Fiorini, S. (2016). *Gene Expression Cancer RNA-Seq*. UCI Machine Learning Repository.
[https://doi.org/10.24432/C5R88H](https://doi.org/10.24432/C5R88H)
