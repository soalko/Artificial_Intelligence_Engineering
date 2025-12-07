from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    compute_zero_flags,
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


# НОВЫЕ ТЕСТЫ ДЛЯ ЭВРИСТИК КАЧЕСТВА

def test_constant_columns_flag():
    """Тест на обнаружение константных колонок"""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "constant_col": [42, 42, 42, 42],  # все значения одинаковые
        "varying_col": [1, 2, 3, 4],
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    assert flags["has_constant_columns"] == True
    assert flags["n_constant_columns"] == 1
    assert "constant_col" in flags["constant_columns"]
    assert "varying_col" not in flags["constant_columns"]


def test_high_cardinality_categoricals():
    """Тест на обнаружение признаков с высокой кардинальностью"""
    # Создаем колонку с высокой кардинальностью (почти все значения уникальны)
    # Преобразуем числа в строки, чтобы колонка была категориальной
    df = pd.DataFrame({
        "user_id": [f"user_{i}" for i in range(100)],  # 100 уникальных строковых значений
        "category": ["A", "B"] * 50,  # 2 уникальных значения из 100
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)

    # Устанавливаем порог 0.3 (30%)
    flags = compute_quality_flags(summary, missing_df, high_cardinality_threshold=0.3)

    # user_id должен быть обнаружен как колонка с высокой кардинальностью
    assert flags["has_high_cardinality_categoricals"] == True
    assert flags["n_high_cardinality_columns"] == 1

    high_card_cols = flags["high_cardinality_columns"]
    assert len(high_card_cols) == 1
    assert high_card_cols[0]["name"] == "user_id"
    # 100 уникальных из 100 ненулевых = 100%
    assert abs(high_card_cols[0]["cardinality_ratio"] - 1.0) < 0.01


def test_id_duplicates():
    """Тест на обнаружение дубликатов в ID-колонках"""
    df = pd.DataFrame({
        "user_id": [1, 2, 3, 1, 2],  # дубликаты
        "customer_id": [100, 101, 102, 103, 104],  # уникальные
        "name": ["A", "B", "C", "D", "E"],
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    assert flags["has_suspicious_id_duplicates"] == True
    assert flags["n_id_duplicate_columns"] >= 1

    # Проверяем, что user_id обнаружен как проблемный
    id_cols = flags["id_duplicate_columns"]
    user_id_found = any(col["name"] == "user_id" for col in id_cols)
    assert user_id_found == True


def test_zero_values():
    """Тест на обнаружение колонок с большим количеством нулей"""
    df = pd.DataFrame({
        "col_with_zeros": [0, 0, 0, 0, 1, 2, 3],  # 4/7 = 57% нулей
        "col_many_zeros": [0, 0, 0, 0, 0, 0, 1],  # 6/7 = 86% нулей
        "col_no_zeros": [1, 2, 3, 4, 5, 6, 7],
    })

    summary = summarize_dataset(df)

    # Тест с порогом 80%
    zero_flags_80 = compute_zero_flags(df, summary, zero_share_threshold=0.8)
    assert zero_flags_80["has_many_zero_values"] == True
    assert zero_flags_80["n_high_zero_columns"] == 1  # только col_many_zeros

    # Тест с порогом 50%
    zero_flags_50 = compute_zero_flags(df, summary, zero_share_threshold=0.5)
    assert zero_flags_50["has_many_zero_values"] == True
    assert zero_flags_50["n_high_zero_columns"] == 2  # оба столбца с нулями


def test_quality_score_calculation():
    """Тест на расчет quality_score с новыми эвристиками"""
    df = pd.DataFrame({
        "id": [1, 2, 3, 1, 2],  # дубликаты ID
        "constant": [5, 5, 5, 5, 5],  # константная колонка
        "high_card": [f"val_{i}" for i in range(5)],  # высокая кардинальность
        "normal": [1, 2, 3, 4, 5],
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    # Проверяем, что все флаги установлены правильно
    assert flags["has_constant_columns"] == True
    assert flags["has_suspicious_id_duplicates"] == True

    # Quality score должен быть между 0 и 1
    assert 0.0 <= flags["quality_score"] <= 1.0

    # С таким плохим датасетом score должен быть низким
    assert flags["quality_score"] < 0.9