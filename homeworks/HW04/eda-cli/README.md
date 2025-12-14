# EDA CLI Tool

Инструмент командной строки для автоматического разведочного анализа данных (EDA) CSV-файлов.

## Установка

```bash
cd eda-cli
pip install -e .
```

## Краткий обзор датасета: размеры, типы данных, основная статистика по колонкам.
```bash
uv run eda-cli overview data/example.csv
```

## Полный EDA-отчёт с графиками, таблицами и анализом качества данных.
```bash
uv run eda-cli report data/example.csv --out-dir reports
```
### Основные опции команды report:
- --out-dir: каталог для отчёта (по умолчанию "reports")
- --sep: разделитель в CSV (по умолчанию ",")
- --encoding: кодировка файла (по умолчанию "utf-8")
- --max-hist-columns: сколько числовых колонок включать в гистограммы (по умолчанию 6)

### Новые опции:
- --top-k-categories: сколько top-значений выводить для категориальных признаков (по умолчанию 5)
- --title: заголовок отчёта в Markdown (по умолчанию "EDA-отчёт")
- --min-missing-share: порог доли пропусков для выделения проблемных колонок (по умолчанию 0.1)
- --high-cardinality-threshold: порог доли уникальных значений для категориальных признаков (по умолчанию 0.5)
- --zero-share-threshold: порог доли нулей в числовых колонках (по умолчанию 0.8)

## Пример вывода с новыми опциями:
```bash
eda-cli report data/example.csv \
  --out-dir my_report \
  --title "Анализ пользователей" \
  --top-k-categories 10 \
  --min-missing-share 0.05 \
  --max-hist-columns 10 \
  --zero-share-threshold 0.9
```

# Запуск сервера
```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

## GET /health
```bash
curl http://localhost:8000/health
```
## POST /quality-from-csv
```bash
curl -X POST http://localhost:8000/quality-from-csv \
  -F "file=@data/example.csv" \
  -F "sep=," \
  -F "encoding=utf-8"
```
## POST /quality-flags-from-csv 
### Возвращает:
- Все флаги качества 
- Детальную информацию о проблемах

```bash
curl -X POST http://localhost:8000/quality-flags-from-csv \
  -F "file=@data/example.csv" \
  -F "sep=," \
  -F "encoding=utf-8" \
  -F "high_cardinality_threshold=0.5" \
  -F "zero_share_threshold=0.8" \
  -F "min_missing_share=0.1"
```

### Новые флаги качества:
- has_constant_columns - есть ли колонки с одинаковыми значениями
- has_high_cardinality_categoricals - категориальные признаки с высокой долей уникальных значений
- has_suspicious_id_duplicates - дубликаты в ID-подобных колонках
- has_many_zero_values - числовые колонки с большим количеством нулей




## POST /summary-from-csv
### Полная сводка по датасету.

```bash
curl -X POST http://localhost:8000/summary-from-csv \
  -F "file=@data/example.csv" \
  -F "sep=," \
  -F "encoding=utf-8" \
  -F "example_values_per_column=3"
```

## POST /eda-report
### Генерация полного EDA отчёта с файлами на сервере.

```bash
curl -X POST http://localhost:8000/eda-report \
  -F "file=@data/example.csv" \
  -F "sep=," \
  -F "encoding=utf-8" \
  -F "top_k_categories=5" \
  -F "max_hist_columns=6" \
  -F "output_dir=api_reports"
```