import requests
import json
import pandas as pd
from pathlib import Path

BASE_URL = "http://localhost:8000"


def test_health():
    """Тест эндпоинта /health"""
    response = requests.get(f"{BASE_URL}/health")
    print("GET /health:")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")
    return response.status_code == 200


def test_quality_from_csv():
    """Тест эндпоинта /quality-from-csv"""
    # Создаем тестовый CSV файл
    test_data = {
        "age": [25, 30, 35, None, 40],
        "salary": [50000, 60000, 70000, 80000, 90000],
        "department": ["IT", "HR", "IT", "Finance", None]
    }
    df = pd.DataFrame(test_data)

    # Сохраняем во временный файл
    temp_path = Path("test_data.csv")
    df.to_csv(temp_path, index=False)

    try:
        # Отправляем запрос
        with open(temp_path, "rb") as f:
            files = {"file": ("test_data.csv", f, "text/csv")}
            data = {"sep": ",", "encoding": "utf-8"}
            response = requests.post(
                f"{BASE_URL}/quality-from-csv",
                files=files,
                data=data
            )

        print("POST /quality-from-csv:")
        print(f"Status: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
        print()

        return response.status_code == 200
    finally:
        # Удаляем временный файл
        if temp_path.exists():
            temp_path.unlink()


def test_quality_flags_from_csv():
    """Тест нового эндпоинта /quality-flags-from-csv с нашими улучшениями"""
    # Создаем более сложный тестовый датасет для проверки новых эвристик
    test_data = {
        "user_id": [1, 2, 3, 1, 2],  # Дубликаты ID
        "constant_col": [42, 42, 42, 42, 42],  # Константная колонка
        "high_card_col": [f"user_{i}" for i in range(5)],  # Высокая кардинальность
        "zeros_col": [0, 0, 0, 1, 2],  # Много нулей
        "normal_col": [10, 20, 30, 40, 50],
        "missing_col": [1, None, 3, None, 5]  # Пропуски
    }
    df = pd.DataFrame(test_data)

    # Сохраняем во временный файл
    temp_path = Path("test_quality_data.csv")
    df.to_csv(temp_path, index=False)

    try:
        # Отправляем запрос
        with open(temp_path, "rb") as f:
            files = {"file": ("test_quality_data.csv", f, "text/csv")}
            data = {
                "sep": ",",
                "encoding": "utf-8",
                "high_cardinality_threshold": 0.5,
                "zero_share_threshold": 0.5,  # Низкий порог для теста
                "min_missing_share": 0.1
            }
            response = requests.post(
                f"{BASE_URL}/quality-flags-from-csv",
                files=files,
                data=data
            )

        print("POST /quality-flags-from-csv (наш новый эндпоинт):")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\nОсновные флаги качества:")
            print(json.dumps(result.get("flags", {}), indent=2))

            print("\nДетальные проблемы:")
            for issue_type, issues in result.get("detailed_issues", {}).items():
                if issues:
                    print(f"  {issue_type}: {len(issues)} проблем")
                    for issue in issues[:2]:  # Показываем первые 2
                        print(f"    - {issue}")

            print("\nСчетчики:")
            print(json.dumps(result.get("counts", {}), indent=2))

            print(f"\nQuality Score: {result.get('quality_score', 'N/A')}")

        print()
        return response.status_code == 200
    finally:
        # Удаляем временный файл
        if temp_path.exists():
            temp_path.unlink()


def test_summary_from_csv():
    """Тест эндпоинта /summary-from-csv"""
    # Используем существующий example.csv
    test_path = Path("data/example.csv")

    if not test_path.exists():
        print(f"Файл {test_path} не найден, пропускаем тест")
        return False

    try:
        # Отправляем запрос
        with open(test_path, "rb") as f:
            files = {"file": ("example.csv", f, "text/csv")}
            data = {"sep": ",", "encoding": "utf-8", "example_values_per_column": 2}
            response = requests.post(
                f"{BASE_URL}/summary-from-csv",
                files=files,
                data=data
            )

        print("POST /summary-from-csv:")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Dataset: {result['dataset_info']['n_rows']} rows, {result['dataset_info']['n_cols']} columns")
            print(f"First 2 columns:")
            for col in result['columns'][:2]:
                print(f"  {col['name']} ({col['dtype']}): {col['non_null']} non-null, {col['unique']} unique")

        print()
        return response.status_code == 200
    except Exception as e:
        print(f"Ошибка: {e}")
        return False


def main():
    """Запуск всех тестов"""
    print("=" * 60)
    print("Тестирование EDA API")
    print("=" * 60)
    print()

    tests = [
        ("Health Check", test_health),
        ("Quality from CSV", test_quality_from_csv),
        ("Quality Flags from CSV (новый)", test_quality_flags_from_csv),
        ("Summary from CSV", test_summary_from_csv),
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"Запуск теста: {test_name}")
        print("-" * 40)
        try:
            success = test_func()
            results[test_name] = "PASS" if success else "FAIL"
        except Exception as e:
            print(f"Ошибка при выполнении теста: {e}")
            results[test_name] = "ERROR"
        print()

    print("=" * 60)
    print("Результаты тестирования:")
    print("=" * 60)
    for test_name, result in results.items():
        print(f"{test_name:30} {result}")


if __name__ == "__main__":
    main()