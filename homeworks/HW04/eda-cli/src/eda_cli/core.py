from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import numpy as np
from pandas.api import types as ptypes


@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    missing: int
    missing_share: float
    unique: int
    example_values: List[Any]
    is_numeric: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    columns: List[ColumnSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
        }


def summarize_dataset(
        df: pd.DataFrame,
        example_values_per_column: int = 3,
) -> DatasetSummary:
    """
    Полный обзор датасета по колонкам:
    - количество строк/столбцов;
    - типы;
    - пропуски;
    - количество уникальных;
    - несколько примерных значений;
    - базовые числовые статистики (для numeric).
    """
    n_rows, n_cols = df.shape
    columns: List[ColumnSummary] = []

    for name in df.columns:
        s = df[name]
        dtype_str = str(s.dtype)

        non_null = int(s.notna().sum())
        missing = n_rows - non_null
        missing_share = float(missing / n_rows) if n_rows > 0 else 0.0
        unique = int(s.nunique(dropna=True))

        # Примерные значения выводим как строки
        examples = (
            s.dropna().astype(str).unique()[:example_values_per_column].tolist()
            if non_null > 0
            else []
        )

        is_numeric = bool(ptypes.is_numeric_dtype(s))
        min_val: Optional[float] = None
        max_val: Optional[float] = None
        mean_val: Optional[float] = None
        std_val: Optional[float] = None

        if is_numeric and non_null > 0:
            min_val = float(s.min())
            max_val = float(s.max())
            mean_val = float(s.mean())
            std_val = float(s.std())

        columns.append(
            ColumnSummary(
                name=name,
                dtype=dtype_str,
                non_null=non_null,
                missing=missing,
                missing_share=missing_share,
                unique=unique,
                example_values=examples,
                is_numeric=is_numeric,
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
            )
        )

    return DatasetSummary(n_rows=n_rows, n_cols=n_cols, columns=columns)


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Таблица пропусков по колонкам: count/share.
    """
    if df.empty:
        return pd.DataFrame(columns=["missing_count", "missing_share"])

    total = df.isna().sum()
    share = total / len(df)
    result = (
        pd.DataFrame(
            {
                "missing_count": total,
                "missing_share": share,
            }
        )
        .sort_values("missing_share", ascending=False)
    )
    return result


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Корреляция Пирсона для числовых колонок.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)


def top_categories(
        df: pd.DataFrame,
        max_columns: int = 5,
        top_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Для категориальных/строковых колонок считает top-k значений.
    Возвращает словарь: колонка -> DataFrame со столбцами value/count/share.
    """
    result: Dict[str, pd.DataFrame] = {}
    candidate_cols: List[str] = []

    for name in df.columns:
        s = df[name]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            candidate_cols.append(name)

    for name in candidate_cols[:max_columns]:
        s = df[name]
        vc = s.value_counts(dropna=True).head(top_k)
        if vc.empty:
            continue
        share = vc / vc.sum()
        table = pd.DataFrame(
            {
                "value": vc.index.astype(str),
                "count": vc.values,
                "share": share.values,
            }
        )
        result[name] = table

    return result


def compute_quality_flags(
        summary: DatasetSummary,
        missing_df: pd.DataFrame,
        high_cardinality_threshold: float = 0.5,
        zero_share_threshold: float = 0.8,
        constant_column_threshold: float = 0.95
) -> Dict[str, Any]:
    """
    Простейшие эвристики «качества» данных:
    - слишком много пропусков;
    - подозрительно мало строк;
    и т.п.
    """
    flags: Dict[str, Any] = {}
    flags["too_few_rows"] = summary.n_rows < 100
    flags["too_many_columns"] = summary.n_cols > 100

    max_missing_share = float(missing_df["missing_share"].max()) if not missing_df.empty else 0.0
    flags["max_missing_share"] = max_missing_share
    flags["too_many_missing"] = max_missing_share > 0.5

    # НОВЫЕ ЭВРИСТИКИ

    # 1. Константные колонки
    flags["has_constant_columns"] = False
    constant_columns = []
    for col in summary.columns:
        if col.non_null > 0 and col.unique == 1:
            flags["has_constant_columns"] = True
            constant_columns.append(col.name)
    flags["constant_columns"] = constant_columns
    flags["n_constant_columns"] = len(constant_columns)

    # 2. Категориальные признаки с высокой кардинальностью
    flags["has_high_cardinality_categoricals"] = False
    high_cardinality_cols = []
    for col in summary.columns:
        # Проверяем, что это не числовая колонка
        if not col.is_numeric and col.unique > 0:
            cardinality_ratio = col.unique / col.non_null if col.non_null > 0 else 0
            if cardinality_ratio > high_cardinality_threshold:
                flags["has_high_cardinality_categoricals"] = True
                high_cardinality_cols.append({
                    "name": col.name,
                    "unique": col.unique,
                    "cardinality_ratio": cardinality_ratio
                })
    flags["high_cardinality_columns"] = high_cardinality_cols
    flags["n_high_cardinality_columns"] = len(high_cardinality_cols)

    # 3. Проверка уникальности ID-подобных колонок
    flags["has_suspicious_id_duplicates"] = False
    id_duplicate_cols = []
    for col in summary.columns:
        col_name_lower = col.name.lower()
        # Ищем колонки, похожие на идентификаторы
        if any(id_keyword in col_name_lower for id_keyword in ['id', 'user', 'client', 'customer']):
            if col.unique < col.non_null and col.non_null > 0:
                flags["has_suspicious_id_duplicates"] = True
                id_duplicate_cols.append({
                    "name": col.name,
                    "unique": col.unique,
                    "non_null": col.non_null,
                    "duplicate_ratio": 1 - (col.unique / col.non_null)
                })
    flags["id_duplicate_columns"] = id_duplicate_cols
    flags["n_id_duplicate_columns"] = len(id_duplicate_cols)

    # 4. Колонки с большим количеством нулей (для числовых)
    flags["has_many_zero_values"] = False
    high_zero_cols = []
    # Для этой эвристики нужен исходный DataFrame, поэтому она будет реализована отдельно
    # в функции, вызывающей compute_quality_flags
    flags["high_zero_columns"] = high_zero_cols
    flags["n_high_zero_columns"] = 0

    # Простейший «скор» качества
    score = 1.0
    score -= max_missing_share  # чем больше пропусков, тем хуже
    if summary.n_rows < 100:
        score -= 0.2
    if summary.n_cols > 100:
        score -= 0.1

    # Штрафы за новые проблемы
    if flags["has_constant_columns"]:
        score -= 0.1 * flags["n_constant_columns"]
    if flags["has_high_cardinality_categoricals"]:
        score -= 0.05 * flags["n_high_cardinality_columns"]
    if flags["has_suspicious_id_duplicates"]:
        score -= 0.15 * flags["n_id_duplicate_columns"]

    score = max(0.0, min(1.0, score))
    flags["quality_score"] = score

    return flags


def compute_zero_flags(df: pd.DataFrame, summary: DatasetSummary, zero_share_threshold: float = 0.8) -> Dict[str, Any]:
    """
    Вычисляет флаги для колонок с большим количеством нулевых значений.
    """
    zero_flags = {
        "has_many_zero_values": False,
        "high_zero_columns": [],
        "n_high_zero_columns": 0
    }

    for col in summary.columns:
        if col.is_numeric and col.non_null > 0:
            # Получаем серию из DataFrame
            s = df[col.name]
            zero_count = int((s == 0).sum())
            zero_share = zero_count / col.non_null

            if zero_share > zero_share_threshold:
                zero_flags["has_many_zero_values"] = True
                zero_flags["high_zero_columns"].append({
                    "name": col.name,
                    "zero_count": zero_count,
                    "zero_share": zero_share,
                    "non_null": col.non_null
                })

    zero_flags["n_high_zero_columns"] = len(zero_flags["high_zero_columns"])
    return zero_flags


def flatten_summary_for_print(summary: DatasetSummary) -> pd.DataFrame:
    """
    Превращает DatasetSummary в табличку для более удобного вывода.
    """
    rows: List[Dict[str, Any]] = []
    for col in summary.columns:
        rows.append(
            {
                "name": col.name,
                "dtype": col.dtype,
                "non_null": col.non_null,
                "missing": col.missing,
                "missing_share": col.missing_share,
                "unique": col.unique,
                "is_numeric": col.is_numeric,
                "min": col.min,
                "max": col.max,
                "mean": col.mean,
                "std": col.std,
            }
        )
    return pd.DataFrame(rows)

