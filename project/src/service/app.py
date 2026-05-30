import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
import sys
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Добавляем путь к корню проекта (на случай, если модуль не виден)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Импортируем функции для построения признаков
try:
    from src.features.build_features import (
        add_date_features, add_lag_features, add_rolling_features, add_price_ratio, encode_categorical
    )

    logger.info("Функции из build_features успешно импортированы")
except ImportError as e:
    logger.error(f"Ошибка импорта из build_features: {e}")
    # Можно выбросить исключение или использовать заглушку
    raise

# Пути к модели и данным
MODEL_PATH = PROJECT_ROOT / "artifacts" / "xgboost_model.pkl"
HISTORY_PATH = PROJECT_ROOT / "data/processed" / "demand_features.csv"

# Загрузка модели
if not MODEL_PATH.exists():
    logger.error(f"Модель не найдена по пути {MODEL_PATH}. Запустите python -m src.models.train")
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
logger.info("Модель загружена")

# Загрузка истории для построения признаков
if not HISTORY_PATH.exists():
    logger.error(f"Файл с признаками не найден: {HISTORY_PATH}. Запустите src.features.build_features")
    raise FileNotFoundError(f"Features not found at {HISTORY_PATH}")
history_df = pd.read_csv(HISTORY_PATH, parse_dates=["Date"])
logger.info(f"История загружена, форма: {history_df.shape}")

# FastAPI приложение
app = FastAPI(title="Demand Forecast Service", description="Прогноз продаж товара в магазине")


class PredictRequest(BaseModel):
    Store_ID: int
    Product_ID: int
    Date: str  # YYYY-MM-DD


def build_features_for_prediction(store_id: int, product_id: int, date_str: str) -> pd.DataFrame:
    """Строит признаки для одной точки прогноза на основе исторических данных."""
    date_obj = pd.to_datetime(date_str)
    # Фильтруем историю по магазину и товару
    hist = history_df[(history_df["Store ID"] == store_id) & (history_df["Product ID"] == product_id)].copy()
    if hist.empty:
        raise ValueError(f"Нет данных для Store ID={store_id}, Product ID={product_id}")

    # Берём последнюю известную строку (с актуальными значениями price, inventory и т.д.)
    last_row = hist.iloc[-1:].copy()
    # Меняем дату на запрашиваемую
    last_row["Date"] = date_obj

    # Добавляем признаки даты (пересчитываем для новой даты)
    last_row = add_date_features(last_row, "Date", ["year", "month", "day", "dayofweek", "weekend", "quarter"])

    # Объединяем историю с новой строкой и вычисляем лаги/скользящие
    combined = pd.concat([hist, last_row], ignore_index=True).sort_values("Date")
    combined = add_lag_features(combined, ["Store ID", "Product ID"], "Units Sold", [1, 2, 3, 7, 14, 28])
    combined = add_rolling_features(combined, ["Store ID", "Product ID"], "Units Sold", [7, 14, 30])
    combined = add_price_ratio(combined, ["Store ID", "Product ID"])

    # Кодируем категориальные признаки (если они ещё не закодированы)
    # Для простоты берём последнюю строку и удаляем нечисловые колонки
    result = combined.iloc[-1:].drop(columns=["Date", "Units Sold"], errors="ignore")
    # Оставляем только числовые колонки (на случай, если encode_categorical не отработал)
    result = result.select_dtypes(include=["number"])
    return result


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        features = build_features_for_prediction(req.Store_ID, req.Product_ID, req.Date)
        pred = model.predict(features)[0]
        return {"predicted_units_sold": round(float(pred), 2)}
    except Exception as e:
        logger.exception("Ошибка при предсказании")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    logger.info("Запуск сервера на http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)