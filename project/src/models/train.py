import pandas as pd
import numpy as np
import yaml
import pickle
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_feature_data():
    data_path = PROJECT_ROOT / "data/processed/demand_features.csv"
    df = pd.read_csv(data_path, parse_dates=["Date"])
    target = load_config()["data"]["target_col"]
    drop_cols = ["Date", target, "Demand", "Units Ordered", "Inventory Level", "Units Sold.1"]
    X = df.drop(columns=drop_cols, errors="ignore")
    print("Колонки признаков:", X.columns.tolist())
    y = df[target]
    return X, y


def main():
    config = load_config()
    X, y = load_feature_data()

    tscv = TimeSeriesSplit(n_splits=config["training"]["num_folds"])
    mae_scores = []
    model_params = config["model"]["params"]

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(**model_params, objective="reg:squarederror")
        # Убрали early_stopping_rounds и eval_set, чтобы избежать ошибки
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        mae_scores.append(mae)
        logger.info(f"Fold {fold + 1}: MAE = {mae:.4f}")

        rmse = np.sqrt(mean_squared_error(y_val, preds))
        logger.info(f"Fold {fold + 1}: RMSE = {rmse:.4f}")

        # print("Пример реальных значений:", y_val[:5].values)
        # print("Пример предсказаний:     ", preds[:5])

    logger.info(f"Средний MAE по CV: {np.mean(mae_scores):.4f} (+- {np.std(mae_scores):.4f})")

    # Финальная модель на всех данных
    final_model = xgb.XGBRegressor(**model_params, objective="reg:squarederror")
    final_model.fit(X, y)

    model_path = ARTIFACTS_DIR / "xgboost_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    import json
    metrics = {
        "cv_mae_mean": float(np.mean(mae_scores)),
        "cv_mae_std": float(np.std(mae_scores)),
        "model_params": model_params
    }
    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Модель сохранена в {model_path}")


if __name__ == "__main__":
    main()