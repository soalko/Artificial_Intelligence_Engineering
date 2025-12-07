from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    compute_zero_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
        path: Path,
        sep: str = ",",
        encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
        path: str = typer.Argument(..., help="Путь к CSV-файлу."),
        sep: str = typer.Option(",", help="Разделитель в CSV."),
        encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
        path: str = typer.Argument(..., help="Путь к CSV-файлу."),
        out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
        sep: str = typer.Option(",", help="Разделитель в CSV."),
        encoding: str = typer.Option("utf-8", help="Кодировка файла."),
        max_hist_columns: int = typer.Option(6, help="Максимум числовых колонок для гистограмм."),
        # НОВЫЕ ПАРАМЕТРЫ
        top_k_categories: int = typer.Option(5, help="Сколько top-значений выводить для категориальных признаков."),
        title: str = typer.Option("EDA-отчёт", help="Заголовок отчёта в Markdown."),
        min_missing_share: float = typer.Option(0.1, help="Порог доли пропусков для выделения проблемных колонок."),
        high_cardinality_threshold: float = typer.Option(0.5,
                                                         help="Порог доли уникальных значений для категориальных признаков."),
        zero_share_threshold: float = typer.Option(0.8, help="Порог доли нулей в числовых колонках."),
) -> None:
    """
    Сгенерировать полный EDA-отчёт:
    - текстовый overview и summary по колонкам (CSV/Markdown);
    - статистика пропусков;
    - корреляционная матрица;
    - top-k категорий по категориальным признакам;
    - картинки: гистограммы, матрица пропусков, heatmap корреляции.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # 1. Обзор
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df, top_k=top_k_categories)

    # 2. Качество в целом
    quality_flags = compute_quality_flags(
        summary,
        missing_df,
        high_cardinality_threshold=high_cardinality_threshold
    )

    # Добавляем информацию о нулях
    zero_flags = compute_zero_flags(df, summary, zero_share_threshold=zero_share_threshold)
    quality_flags.update(zero_flags)

    # 3. Выделяем проблемные колонки с пропусками
    problematic_missing_cols = []
    if not missing_df.empty:
        problematic_missing_cols = missing_df[missing_df["missing_share"] > min_missing_share].index.tolist()
    quality_flags["problematic_missing_columns"] = problematic_missing_cols
    quality_flags["n_problematic_missing_columns"] = len(problematic_missing_cols)
    quality_flags["min_missing_share_threshold"] = min_missing_share

    # 4. Сохраняем табличные артефакты
    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    # 5. Markdown-отчёт
    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"## Настройки отчёта\n\n")
        f.write(f"- Исходный файл: `{Path(path).name}`\n")
        f.write(f"- top_k_categories: `{top_k_categories}`\n")
        f.write(f"- min_missing_share: `{min_missing_share:.2%}`\n")
        f.write(f"- high_cardinality_threshold: `{high_cardinality_threshold:.0%}`\n")
        f.write(f"- zero_share_threshold: `{zero_share_threshold:.0%}`\n")
        f.write(f"- max_hist_columns: `{max_hist_columns}`\n\n")

        f.write(f"## Общая информация\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Оценка качества: **{quality_flags['quality_score']:.2f}**\n")
        f.write(f"- Макс. доля пропусков по колонке: **{quality_flags['max_missing_share']:.2%}**\n")
        f.write(f"- Слишком мало строк: **{quality_flags['too_few_rows']}**\n")
        f.write(f"- Слишком много колонок: **{quality_flags['too_many_columns']}**\n")
        f.write(f"- Слишком много пропусков: **{quality_flags['too_many_missing']}**\n\n")

        # Новые эвристики
        f.write("### Новые эвристики качества\n\n")
        f.write(
            f"- Константные колонки: **{quality_flags['has_constant_columns']}** (кол-во: {quality_flags['n_constant_columns']})\n")
        if quality_flags['has_constant_columns']:
            f.write(f"  - Константные колонки: {', '.join(quality_flags['constant_columns'])}\n")

        f.write(
            f"- Высокая кардинальность категориальных признаков: **{quality_flags['has_high_cardinality_categoricals']}** (кол-во: {quality_flags['n_high_cardinality_columns']})\n")
        if quality_flags['has_high_cardinality_categoricals']:
            for col in quality_flags['high_cardinality_columns'][:3]:  # показываем только первые 3
                f.write(f"  - `{col['name']}`: уникальных {col['unique']} ({col['cardinality_ratio']:.1%})\n")

        f.write(
            f"- Дубликаты в ID-колонках: **{quality_flags['has_suspicious_id_duplicates']}** (кол-во: {quality_flags['n_id_duplicate_columns']})\n")
        if quality_flags['has_suspicious_id_duplicates']:
            for col in quality_flags['id_duplicate_columns'][:3]:
                f.write(
                    f"  - `{col['name']}`: уникальных {col['unique']} из {col['non_null']} ({col['duplicate_ratio']:.1%} дубликатов)\n")

        f.write(
            f"- Много нулей в числовых колонках: **{quality_flags['has_many_zero_values']}** (кол-во: {quality_flags['n_high_zero_columns']})\n")
        if quality_flags['has_many_zero_values']:
            for col in quality_flags['high_zero_columns'][:3]:
                f.write(f"  - `{col['name']}`: {col['zero_count']} нулей ({col['zero_share']:.1%})\n")

        f.write(
            f"- Проблемные колонки с пропусками (> {min_missing_share:.0%}): **{quality_flags['n_problematic_missing_columns']}**\n")
        if quality_flags['n_problematic_missing_columns'] > 0:
            f.write(f"  - Колонки: {', '.join(quality_flags['problematic_missing_columns'][:5])}")
            if len(quality_flags['problematic_missing_columns']) > 5:
                f.write(f" ... и ещё {len(quality_flags['problematic_missing_columns']) - 5}")
            f.write("\n")
        f.write("\n")

        f.write("## Колонки\n\n")
        f.write("См. файл `summary.csv`.\n\n")

        f.write("## Пропуски\n\n")
        if missing_df.empty:
            f.write("Пропусков нет или датасет пуст.\n\n")
        else:
            f.write("См. файлы `missing.csv` и `missing_matrix.png`.\n\n")

        f.write("## Корреляция числовых признаков\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок для корреляции.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

        f.write("## Категориальные признаки\n\n")
        if not top_cats:
            f.write("Категориальные/строковые признаки не найдены.\n\n")
        else:
            f.write(f"Топ-{top_k_categories} значений для категориальных признаков:\n\n")
            f.write("См. файлы в папке `top_categories/`.\n\n")

        f.write("## Гистограммы числовых колонок\n\n")
        f.write(f"Построены гистограммы для первых {max_hist_columns} числовых колонок.\n")
        f.write("См. файлы `hist_*.png`.\n")

    # 6. Картинки
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"- Основной markdown: {md_path}")
    typer.echo(f"- Заголовок отчёта: {title}")
    typer.echo(f"- Настройки: top_k={top_k_categories}, min_missing={min_missing_share:.0%}")
    typer.echo("- Табличные файлы: summary.csv, missing.csv, correlation.csv, top_categories/*.csv")
    typer.echo("- Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png")


if __name__ == "__main__":
    app()