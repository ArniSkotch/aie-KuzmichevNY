# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

    - создаст виртуальное окружение .venv;

    - установит зависимости из pyproject.toml;

    - установит сам проект eda-cli в окружение.

Запуск CLI
Краткий обзор
bash

uv run eda-cli overview data/example.csv

Параметры:

    --sep – разделитель (по умолчанию ,);

    --encoding – кодировка (по умолчанию utf-8).

Полный EDA-отчёт
bash

uv run eda-cli report data/example.csv --out-dir reports

В результате в каталоге reports/ появятся:

    - report.md – основной отчёт в Markdown;

    - summary.csv – таблица по колонкам;

    - missing.csv – пропуски по колонкам;

    - correlation.csv – корреляционная матрица (если есть числовые признаки);

    - top_categories/*.csv – top-k категорий по строковым признакам;

    - hist_*.png – гистограммы числовых колонок;

    - missing_matrix.png – визуализация пропусков;

    - correlation_heatmap.png – тепловая карта корреляций.

Новые параметры команды report:
text

--max-hist-columns – максимальное количество числовых колонок для гистограмм (по умолчанию 6).

--top-k-categories – количество топ-значений для категориальных признаков (по умолчанию 5).

--min-missing-share – порог доли пропусков для определения проблемных столбцов (по умолчанию 0.3).

Новые эвристики качества данных:
text

1)	Обнаружение константных столбцов - столбцы, где все значения одинаковы.

2)	Обнаружение категориальных признаков с высокой кардинальностью - более 80% уникальных значений.

HTTP API Сервис (для HW04)

Проект включает FastAPI сервис для оценки качества данных через HTTP. Для запуска сервиса:
bash

uv run uvicorn eda_cli.api:app --reload --port 8000

Доступные эндпоинты:

    - GET /health - Проверка работоспособности сервиса

    - POST /quality - Оценка качества по агрегированным признакам (принимает JSON)

    - POST /quality-from-csv - Оценка качества по CSV-файлу

    - POST /quality-flags-from-csv - Полный набор флагов качества для CSV-файла (включая эвристики из HW03)

Пример использования нового эндпоинта /quality-flags-from-csv:

```
bash
curl -X POST -F "file=@data/example.csv" http://localhost:8000/quality-flags-from-csv
```

Ответ включает все флаги качества, в том числе новые эвристики из HW03:

    - has_constant_columns - есть ли константные столбцы

    - constant_columns_count - количество константных столбцов

    - constant_columns_names - имена константных столбцов

    - has_high_cardinality_categoricals - есть ли категориальные признаки с высокой кардинальностью

    - high_cardinality_columns_count - количество таких столбцов

    - high_cardinality_columns_names - имена таких столбцов

    - quality_score - общая оценка качества (0.0-1.0)

    - и другие стандартные флаги

Документация API

После запуска сервиса документация доступна по адресам:

    Swagger UI: http://localhost:8000/docs

    OpenAPI схема: http://localhost:8000/openapi.json

Тесты

```
bash
uv run pytest -q
```

## 1. Копирование проекта из HW03

Скопируйте содержимое вашего проекта из `homeworks/HW03/eda-cli` в `homeworks/HW04/eda-cli`:

```
bash
cp -r homeworks/HW03/eda-cli homeworks/HW04/
```

## 2. Обновление файлов

Замените следующие файлы в проекте homeworks/HW04/eda-cli:

    - pyproject.toml - добавлены зависимости FastAPI

    - src/eda_cli/api.py - добавлен новый эндпоинт /quality-flags-from-csv

    - README.md - добавлено описание HTTP API

## 3. Установка зависимостей

```
bash
cd homeworks/HW04/eda-cli
uv sync
```

## 4. Проверка работы
Проверка CLI (ядро из HW03 должно работать):

```
bash
uv run eda-cli overview data/example.csv
uv run eda-cli report data/example.csv --out-dir reports_example
```

Запуск HTTP сервиса:

```
bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

Тестирование нового эндпоинта:

    Через curl:
        ```
        bash
        curl -X POST -F "file=@data/example.csv" http://localhost:8000/quality-flags-from-csv
        ```
    Через Swagger UI: откройте в браузере http://localhost:8000/docs

## 5. Запуск тестов

```
bash
uv run pytest -q
```

Новый эндпоинт /quality-flags-from-csv будет возвращать полный набор флагов качества, включая эвристики из HW03 (константные столбцы и категориальные признаки с высокой кардинальностью).

## Проверка реализации

Новый эндпоинт `/quality-flags-from-csv`:

1. **Принимает CSV-файл** через multipart/form-data (аналогично `/quality-from-csv`)
2. **Использует EDA-ядро** из HW03:
   - `summarize_dataset()` - для получения сводки по датасету
   - `missing_table()` - для таблицы пропусков
   - `compute_quality_flags()` - для вычисления флагов качества (включая новые эвристики)
3. **Возвращает полный словарь флагов**:
   - Все стандартные флаги (`too_few_rows`, `too_many_columns`, `too_many_missing`, и т.д.)
   - Новые эвристики из HW03 (`has_constant_columns`, `has_high_cardinality_categoricals`, и т.д.)
   - `quality_score` - общая оценка качества
   - Метаинформация (`latency_ms`, `filename`, `n_rows`, `n_cols`)

Это полностью соответствует требованиям задания 2.3.2 (вариант A): эндпоинт не дублирует существующие, использует доработки из HW03, и возвращает полный набор флагов качества.
