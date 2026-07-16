# Участие в разработке

## Окружение

Проект требует Python 3.11–3.12. Форматирование и статические проверки выполняются
Ruff и mypy, тесты — pytest.

Для воспроизведения опубликованного анализа используйте ограничения прямых зависимостей:

```bash
python -m pip install -c requirements/analysis.txt -e ".[dev,neural,notebook]"
```

Сохранённая модель проверяется отдельно на Python 3.12.x с extra `model-runtime`.

## Требования к коду

- Добавляйте type hints для публичных функций.
- Используйте предметные имена модулей, функций и переменных.
- Оставляйте комментарии только для неочевидных решений и ограничений.
- Не добавляйте классы и дополнительные abstraction layers без текущей необходимости.
- Используйте `pathlib.Path` для путей.
- Размещайте зависящий от данных preprocessing внутри `sklearn.pipeline.Pipeline`.
- Разделяйте данные до обучения preprocessing и не используйте test set для выбора модели,
  признаков или гиперпараметров.

## Проверка изменений

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy src scripts/smoke_test_model.py
python -m scripts.verify_artifacts
python -m scripts.smoke_test_model
python -m pytest -q
```
