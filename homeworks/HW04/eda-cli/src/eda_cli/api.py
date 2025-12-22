from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from .core import (
    summarize_dataset,
    missing_table,
    compute_quality_flags,
    compute_zero_flags,
    correlation_matrix,
    top_categories,
    flatten_summary_for_print,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = FastAPI(
    title="EDA API",
    description="HTTP-сервис для разведочного анализа данных",
    version="1.0.0",
)


def _read_csv_from_upload(
        file: UploadFile,
        sep: str = ",",
        encoding: str = "utf-8",
) -> pd.DataFrame:
    """Читает CSV из UploadFile в DataFrame."""
    try:
        content = file.file.read()
        if not content:
            raise ValueError("Файл пуст")
        return pd.read_csv(io.BytesIO(content), sep=sep, encoding=encoding)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения CSV: {str(e)}")
    finally:
        file.file.close()


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Проверка работоспособности сервиса."""
    return {"status": "ok", "service": "eda-api", "version": "1.0.0"}


@app.post("/quality")
async def quality_from_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Анализ качества данных из JSON.
    JSON должен быть словарём, который можно преобразовать в DataFrame.
    """
    import time
    start_time = time.time()

    try:
        df = pd.DataFrame(data)
        summary = summarize_dataset(df)
        missing_df = missing_table(df)
        flags = compute_quality_flags(summary, missing_df)
        zero_flags = compute_zero_flags(df, summary)
        flags.update(zero_flags)

        quality_score = flags.get("quality_score", 0.0)

        ok_for_model = (
                quality_score >= 0.7 and
                not flags.get("too_few_rows", False) and
                not flags.get("too_many_missing", False) and
                not flags.get("has_constant_columns", False)
        )

        latency_ms = round((time.time() - start_time) * 1000, 2)

        response = {
            "ok_for_model": ok_for_model,
            "quality_score": quality_score,
            "latency_ms": latency_ms,
            "flags": {
                "too_few_rows": flags.get("too_few_rows", False),
                "too_many_columns": flags.get("too_many_columns", False),
                "too_many_missing": flags.get("too_many_missing", False),
                "max_missing_share": flags.get("max_missing_share", 0.0),
                "has_constant_columns": flags.get("has_constant_columns", False),
                "has_high_cardinality_categoricals": flags.get("has_high_cardinality_categoricals", False),
                "has_suspicious_id_duplicates": flags.get("has_suspicious_id_duplicates", False),
                "has_many_zero_values": flags.get("has_many_zero_values", False),
            }
        }

        return response
    except Exception as e:
        # В случае ошибки также возвращаем структурированный ответ
        latency_ms = round((time.time() - start_time) * 1000, 2)
        raise HTTPException(
            status_code=400,
            detail={
                "ok_for_model": False,
                "quality_score": 0.0,
                "latency_ms": latency_ms,
                "error": str(e),
                "flags": {}
            }
        )


@app.post("/quality-from-csv")
async def quality_from_csv(
        file: UploadFile = File(...),
        sep: str = Form(","),
        encoding: str = Form("utf-8"),
) -> Dict[str, Any]:
    """
    Анализ качества данных из CSV-файла.
    Возвращает базовые флаги качества.
    """
    try:
        df = _read_csv_from_upload(file, sep=sep, encoding=encoding)
        summary = summarize_dataset(df)
        missing_df = missing_table(df)
        flags = compute_quality_flags(summary, missing_df)

        return {"flags": flags}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/quality-flags-from-csv")
async def quality_flags_from_csv(
        file: UploadFile = File(...),
        sep: str = Form(","),
        encoding: str = Form("utf-8"),
        high_cardinality_threshold: float = Form(0.5),
        zero_share_threshold: float = Form(0.8),
        min_missing_share: float = Form(0.1),
) -> Dict[str, Any]:
    """
    Полный анализ качества данных из CSV-файла.
    Возвращает все флаги качества, включая новые из HW03.

    Это новый эндпоинт, который использует доработки из HW03:
    - has_constant_columns
    - has_high_cardinality_categoricals
    - has_suspicious_id_duplicates
    - has_many_zero_values
    """
    try:
        # 1. Читаем CSV
        df = _read_csv_from_upload(file, sep=sep, encoding=encoding)

        # 2. Собираем базовую информацию
        summary = summarize_dataset(df)
        missing_df = missing_table(df)

        # 3. Вычисляем флаги качества с учетом новых эвристик
        quality_flags = compute_quality_flags(
            summary,
            missing_df,
            high_cardinality_threshold=high_cardinality_threshold
        )

        # 4. Добавляем информацию о нулях
        zero_flags = compute_zero_flags(df, summary, zero_share_threshold=zero_share_threshold)
        quality_flags.update(zero_flags)

        # 5. Выделяем проблемные колонки с пропусками
        problematic_missing_cols = []
        if not missing_df.empty:
            problematic_missing_cols = missing_df[missing_df["missing_share"] > min_missing_share].index.tolist()

        quality_flags["problematic_missing_columns"] = problematic_missing_cols
        quality_flags["n_problematic_missing_columns"] = len(problematic_missing_cols)
        quality_flags["min_missing_share_threshold"] = min_missing_share

        # 6. Структурируем ответ
        response = {
            "dataset_info": {
                "n_rows": summary.n_rows,
                "n_cols": summary.n_cols,
                "columns": [col.name for col in summary.columns],
                "numeric_columns": [col.name for col in summary.columns if col.is_numeric],
                "categorical_columns": [col.name for col in summary.columns if not col.is_numeric],
            },
            "quality_score": quality_flags["quality_score"],
            "flags": {
                # Базовые флаги
                "too_few_rows": quality_flags["too_few_rows"],
                "too_many_columns": quality_flags["too_many_columns"],
                "too_many_missing": quality_flags["too_many_missing"],
                "max_missing_share": quality_flags["max_missing_share"],

                # Новые флаги из HW03
                "has_constant_columns": quality_flags["has_constant_columns"],
                "has_high_cardinality_categoricals": quality_flags["has_high_cardinality_categoricals"],
                "has_suspicious_id_duplicates": quality_flags["has_suspicious_id_duplicates"],
                "has_many_zero_values": quality_flags["has_many_zero_values"],
            },
            "detailed_issues": {
                "constant_columns": quality_flags.get("constant_columns", []),
                "high_cardinality_columns": quality_flags.get("high_cardinality_columns", []),
                "id_duplicate_columns": quality_flags.get("id_duplicate_columns", []),
                "high_zero_columns": quality_flags.get("high_zero_columns", []),
                "problematic_missing_columns": problematic_missing_cols,
            },
            "counts": {
                "n_constant_columns": quality_flags.get("n_constant_columns", 0),
                "n_high_cardinality_columns": quality_flags.get("n_high_cardinality_columns", 0),
                "n_id_duplicate_columns": quality_flags.get("n_id_duplicate_columns", 0),
                "n_high_zero_columns": quality_flags.get("n_high_zero_columns", 0),
                "n_problematic_missing_columns": quality_flags.get("n_problematic_missing_columns", 0),
            },
            "thresholds_used": {
                "high_cardinality_threshold": high_cardinality_threshold,
                "zero_share_threshold": zero_share_threshold,
                "min_missing_share": min_missing_share,
            }
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка анализа качества данных: {str(e)}")


@app.post("/summary-from-csv")
async def summary_from_csv(
        file: UploadFile = File(...),
        sep: str = Form(","),
        encoding: str = Form("utf-8"),
        example_values_per_column: int = Form(3),
) -> Dict[str, Any]:
    """
    Полная сводка по датасету из CSV-файла.
    Возвращает детальную информацию по всем колонкам.
    """
    try:
        df = _read_csv_from_upload(file, sep=sep, encoding=encoding)
        summary = summarize_dataset(df, example_values_per_column=example_values_per_column)

        # Преобразуем сводку в удобный формат для JSON
        columns_info = []
        for col in summary.columns:
            col_info = {
                "name": col.name,
                "dtype": col.dtype,
                "non_null": col.non_null,
                "missing": col.missing,
                "missing_share": col.missing_share,
                "unique": col.unique,
                "is_numeric": col.is_numeric,
                "example_values": col.example_values,
            }

            # Добавляем числовые статистики если есть
            if col.is_numeric:
                if col.min is not None:
                    col_info["min"] = col.min
                if col.max is not None:
                    col_info["max"] = col.max
                if col.mean is not None:
                    col_info["mean"] = col.mean
                if col.std is not None:
                    col_info["std"] = col.std

            columns_info.append(col_info)

        response = {
            "dataset_info": {
                "n_rows": summary.n_rows,
                "n_cols": summary.n_cols,
            },
            "columns": columns_info,
            "settings": {
                "example_values_per_column": example_values_per_column,
            }
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка создания сводки: {str(e)}")


@app.post("/eda-report")
async def eda_report(
        file: UploadFile = File(...),
        sep: str = Form(","),
        encoding: str = Form("utf-8"),
        top_k_categories: int = Form(5),
        max_hist_columns: int = Form(6),
        output_dir: str = Form("reports"),
) -> Dict[str, Any]:
    """
    Генерация полного EDA отчёта.
    Создаёт файлы отчёта на сервере и возвращает информацию о них.
    """
    try:
        df = _read_csv_from_upload(file, sep=sep, encoding=encoding)
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        # 1. Базовый анализ
        summary = summarize_dataset(df)
        missing_df = missing_table(df)
        corr_df = correlation_matrix(df)
        top_cats = top_categories(df, top_k=top_k_categories)

        # 2. Анализ качества
        quality_flags = compute_quality_flags(summary, missing_df)
        zero_flags = compute_zero_flags(df, summary)
        quality_flags.update(zero_flags)

        # 3. Сохраняем табличные данные
        summary_df = flatten_summary_for_print(summary)
        summary_df.to_csv(out_root / "summary.csv", index=False)

        if not missing_df.empty:
            missing_df.to_csv(out_root / "missing.csv", index=True)

        if not corr_df.empty:
            corr_df.to_csv(out_root / "correlation.csv", index=True)

        if top_cats:
            save_top_categories_tables(top_cats, out_root / "top_categories")

        # 4. Генерируем графики
        plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
        plot_missing_matrix(df, out_root / "missing_matrix.png")
        plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

        # 5. Создаем Markdown отчет
        md_path = out_root / "report.md"
        with md_path.open("w", encoding="utf-8") as f:
            f.write(f"# EDA Report (Generated via API)\n\n")
            f.write(f"## Dataset Info\n\n")
            f.write(f"- Rows: **{summary.n_rows}**\n")
            f.write(f"- Columns: **{summary.n_cols}**\n\n")

            f.write(f"## Quality Flags\n\n")
            f.write(f"- Quality Score: **{quality_flags['quality_score']:.2f}**\n")
            for key, value in quality_flags.items():
                if isinstance(value, bool) and key not in ['too_few_rows', 'too_many_columns']:
                    f.write(f"- {key}: **{value}**\n")

        # 6. Формируем ответ
        files_created = {
            "summary_csv": str(out_root / "summary.csv"),
            "missing_csv": str(out_root / "missing.csv") if not missing_df.empty else None,
            "correlation_csv": str(out_root / "correlation.csv") if not corr_df.empty else None,
            "top_categories_dir": str(out_root / "top_categories") if top_cats else None,
            "missing_matrix_png": str(out_root / "missing_matrix.png"),
            "correlation_heatmap_png": str(out_root / "correlation_heatmap.png"),
            "report_md": str(out_root / "report.md"),
        }

        # Убираем None значения
        files_created = {k: v for k, v in files_created.items() if v is not None}

        response = {
            "status": "success",
            "message": f"EDA report generated in {output_dir}",
            "dataset_info": {
                "n_rows": summary.n_rows,
                "n_cols": summary.n_cols,
            },
            "quality_score": quality_flags["quality_score"],
            "files_created": files_created,
            "settings_used": {
                "top_k_categories": top_k_categories,
                "max_hist_columns": max_hist_columns,
                "output_dir": output_dir,
            }
        }

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка генерации отчёта: {str(e)}"
        )

