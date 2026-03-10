"""
Упрощенный модуль сохранения данных
"""

import pandas as pd
from pathlib import Path

def save_final_data(df):
    """
    Сохраняет финальные данные
    """
    print("=" * 50)
    print("СОХРАНЕНИЕ ДАННЫХ")
    print("=" * 50)

    # Проверяем наличие целевой переменной
    if 'defect_label' not in df.columns:
        print("❌ В данных нет колонки defect_label")
        return False

    # Базовая статистика
    print(f"Всего записей: {len(df)}")
    print(f"Колонок: {len(df.columns)}")
    print(f"Колонки: {list(df.columns)}")

    defects = df['defect_label'].sum()
    print(f"Модулей с дефектами: {defects} ({defects/len(df)*100:.1f}%)")
    print(f"Модулей без дефектов: {len(df) - defects} ({(len(df)-defects)/len(df)*100:.1f}%)")

    # Сохраняем
    Path("data/final").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/final/final_defect_data.csv", index=False)

    print(f"✅ Данные сохранены в data/final/final_defect_data.csv")
    return True
