from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_compute_quality_flags_new_heuristics():
    """
    Тест новых эвристик качества данных!!!!!!!!!!
    """

    # Тест 1. Датафрейм с константным столбцом
    df_const = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "constant_col": [1, 1, 1, 1, 1],  # константный столбец
        "normal_col": [10, 20, 30, 40, 50]
    })

    summary_const = summarize_dataset(df_const)
    missing_df_const = missing_table(df_const)
    flags_const = compute_quality_flags(summary_const, missing_df_const)

    assert flags_const["has_constant_columns"] == True
    assert flags_const["constant_columns_count"] == 1
    assert "constant_col" in flags_const["constant_columns_names"]

    # Тест 2: Датафрейм с высокой кардинальностью
    df_high_card = pd.DataFrame({
        "id": list(range(100)),
        "high_card_col": [f"value_{i}" for i in range(100)],  # 100 уникальных значений из 100 строк
        "normal_col": [1, 2, 3] * 33 + [1]  # невысокая кардинальность
    })

    summary_high_card = summarize_dataset(df_high_card)
    missing_df_high_card = missing_table(df_high_card)
    flags_high_card = compute_quality_flags(summary_high_card, missing_df_high_card)

    assert flags_high_card["has_high_cardinality_categoricals"] == True
    assert flags_high_card["high_cardinality_columns_count"] == 1
    assert "high_card_col" in flags_high_card["high_cardinality_columns_names"]

    # Тест 3. Датафрейм без проблем
    # используем много строк, чтобы тест проходил
    df_good = pd.DataFrame({
        "id": list(range(20)),  # 20 строк
        "col1": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A",
                 "B", "C", "A", "B", "C", "A", "B", "C", "A", "B"],  # 3 уникальных из 20 = 15% < 80%
        "col2": list(range(20, 40))  # числовой столбец
    })

    summary_good = summarize_dataset(df_good)
    missing_df_good = missing_table(df_good)
    flags_good = compute_quality_flags(summary_good, missing_df_good)

    assert flags_good["has_constant_columns"] == False
    assert flags_good["has_high_cardinality_categoricals"] == False
    assert flags_good["quality_score"] > 0.7  # Должен быть высокий score

    # Дополнительный тест. Проверяем, что числовые столбцы не считаются категориальными
    df_numeric = pd.DataFrame({
        "numeric1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "numeric2": [10.0, 20.0, 30.0, 40.0, 50.0]
    })

    summary_numeric = summarize_dataset(df_numeric)
    missing_df_numeric = missing_table(df_numeric)
    flags_numeric = compute_quality_flags(summary_numeric, missing_df_numeric)

    assert flags_numeric["has_high_cardinality_categoricals"] == False