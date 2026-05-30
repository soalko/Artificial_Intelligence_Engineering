import pytest
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent

def test_raw_data_exists():
    raw_path = PROJECT_ROOT / "data" / "raw" / "demand_forecasting.csv"
    assert raw_path.exists(), "Сырые данные не найдены. Запустите src.data.load_data"

def test_processed_data_exists():
    proc_path = PROJECT_ROOT / "data" / "processed" / "demand_forecast_processed.csv"
    assert proc_path.exists(), "Обработанные данные не найдены. Запустите src.data.preprocess"

def test_features_created():
    feat_path = PROJECT_ROOT / "data" / "processed" / "demand_features.csv"
    assert feat_path.exists(), "Признаки не созданы. Запустите src.features.build_features"

def test_model_exists():
    model_path = PROJECT_ROOT / "artifacts" / "xgboost_model.pkl"
    assert model_path.exists(), "Модель не обучена. Запустите src.models.train"