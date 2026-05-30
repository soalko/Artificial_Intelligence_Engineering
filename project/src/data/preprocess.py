import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def preprocess(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Чистка данных с учётом реальных названий колонок."""
    date_col = config["data"]["date_col"]
    store_col = config["data"]["store_col"]
    item_col = config["data"]["item_col"]
    target = config["data"]["target_col"]

    # Приводим дату к datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Сортируем по дате в разрезе магазин+товар
    df = df.sort_values([store_col, item_col, date_col])

    # Заполняем пропуски (если есть) – для числовых колонок
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df.groupby([store_col, item_col])[col].fillna(method="ffill")

    # Удаляем дубликаты
    df = df.drop_duplicates()

    # Убираем аномалии (отрицательные продажи)
    df = df[df[target] >= 0]

    return df


def main():
    config = load_config()
    raw_path = PROJECT_ROOT / config["data"]["raw_path"]
    if not raw_path.exists():
        logger.error(f"Сырые данные не найдены: {raw_path}")
        return
    df = pd.read_csv(raw_path)
    df_clean = preprocess(df, config)

    processed_path = PROJECT_ROOT / config["data"]["processed_path"]
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(processed_path, index=False)
    logger.info(f"Обработанные данные сохранены в {processed_path}")


if __name__ == "__main__":
    main()