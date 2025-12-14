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
