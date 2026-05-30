import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def add_date_features(df: pd.DataFrame, date_col: str, features: list) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["weekend"] = (df[date_col].dt.dayofweek >= 5).astype(int)
    df["quarter"] = df[date_col].dt.quarter

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def add_lag_features(df: pd.DataFrame, group_cols: list, target: str, lags: list) -> pd.DataFrame:
    df = df.sort_values(["Date"])
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby(group_cols)[target].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, group_cols: list, target: str, windows: list) -> pd.DataFrame:
    for window in windows:
        df[f"rolling_mean_{window}"] = df.groupby(group_cols)[target].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}"] = df.groupby(group_cols)[target].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )
    return df


def add_price_ratio(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    df["price_ratio_7d"] = df["Price"] / (df.groupby(group_cols)["Price"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    ))
    return df


def encode_categorical(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """Кодирует все перечисленные категориальные колонки с помощью LabelEncoder."""
    df = df.copy()
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            # Заполняем NaN временно, если они есть
            df[col] = df[col].fillna("missing")
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def main():
    config = load_config()
    processed_path = PROJECT_ROOT / config["data"]["processed_path"]
    df = pd.read_csv(processed_path, parse_dates=[config["data"]["date_col"]])

    date_col = config["data"]["date_col"]
    target = config["data"]["target_col"]
    store_col = config["data"]["store_col"]
    item_col = config["data"]["item_col"]
    group_cols = [store_col, item_col]

    # 1. Добавляем признаки даты
    df = add_date_features(df, date_col, config["features"]["date_features"])

    # 2. Лаги целевой переменной
    df = add_lag_features(df, group_cols, target, config["features"]["lags"])

    # 3. Скользящие статистики
    df = add_rolling_features(df, group_cols, target, config["features"]["rolling_windows"])

    # 4. Отношение цены
    df = add_price_ratio(df, group_cols)

    # 5. Кодирование всех категориальных признаков
    categorical_cols = [
        store_col,
        item_col,
        config["data"].get("category_col", "Category"),
        config["data"].get("region_col", "Region"),
        config["data"].get("weather_col", "Weather Condition"),
        config["data"].get("promotion_col", "Promotion"),
        config["data"].get("seasonality_col", "Seasonality"),  # добавили
        config["data"].get("epidemic_col", "Epidemic")  # добавили
    ]
    # Удаляем None и те, что реально есть в колонках
    categorical_cols = [c for c in categorical_cols if c and c in df.columns]
    df = encode_categorical(df, categorical_cols)

    # 6. Убеждаемся, что все оставшиеся колонки — числовые
    # Удаляем оригинальные текстовые колонки (они уже закодированы)
    # Оставляем только числовые колонки (int, float)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Принудительно добавляем дату (она может быть datetime, её удалим позже)
    # Саму дату и целевой столбец пока оставляем, но в финальном X они будут удалены
    df = df[numeric_cols + ["Date", target]]  # сохраняем только числовые плюс служебные

    # Удаляем строки с NA (из-за лагов)
    df = df.dropna().reset_index(drop=True)

    # Сохраняем
    feature_path = PROJECT_ROOT / "data/processed/demand_features.csv"
    df.to_csv(feature_path, index=False)
    logger.info(f"Признаки сохранены в {feature_path}")
    logger.info(f"Итоговые колонки: {df.columns.tolist()}")


if __name__ == "__main__":
    main()