import pytest
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

@pytest.fixture
def model():
    model_path = PROJECT_ROOT / "artifacts" / "xgboost_model.pkl"
    assert model_path.exists(), "Модель не найдена. Сначала выполните: python -m src.models.train"
    with open(model_path, "rb") as f:
        return pickle.load(f)

def test_model_returns_positive_prediction(model):
    # Получаем ожидаемые имена признаков из самой модели
    feature_names = model.get_booster().feature_names
    # Создаём датафрейм с одной строкой, заполненной нулями (или любыми числами)
    dummy_input = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
    pred = model.predict(dummy_input)[0]
    # Проверяем, что предсказание неотрицательно (продажи не могут быть отрицательными)
    assert pred >= 0