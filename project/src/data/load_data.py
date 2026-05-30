import os
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
TARGET_FILE = RAW_DATA_DIR / "demand_forecasting.csv"

def download_from_kaggle():
    try:
        import kagglehub
        path = kagglehub.dataset_download("hosan707/demand-forecasting-dataset")
        # kagglehub возвращает папку, в которой лежит CSV
        csv_file = list(Path(path).glob("*.csv"))[0]
        df = pd.read_csv(csv_file)
        df.to_csv(TARGET_FILE, index=False)
        logger.info(f"Датасет сохранён в {TARGET_FILE}")
        return True
    except Exception as e:
        logger.error(f"Ошибка загрузки через kagglehub: {e}")
        return False

def manual_instructions():
    """Инструкция для ручного скачивания."""
    print("\nНе удалось загрузить автоматически. Сделайте вручную:")
    print("1. Перейдите на https://www.kaggle.com/datasets/hosan707/demand-forecasting-dataset")
    print("2. Скачайте demand_forecasting.csv")
    print(f"3. Поместите файл в {RAW_DATA_DIR}")
    print("4. Переименуйте при необходимости в demand_forecasting.csv\n")

if __name__ == "__main__":
    if not TARGET_FILE.exists():
        if not download_from_kaggle():
            manual_instructions()
    else:
        logger.info("Файл уже существует.")