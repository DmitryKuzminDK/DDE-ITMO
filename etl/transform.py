import pandas as pd
import numpy as np
from pathlib import Path

def clean_data(df):
    print("=" * 50)
    print("ОЧИСТКА ДАННЫХ")
    print("=" * 50)

    original_rows = len(df)

    # 1. Очищаем названия колонок
    df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]

    # 2. Удаляем дубликаты
    before = len(df)
    df = df.drop_duplicates()
    print(f"Удалено дубликатов: {before - len(df)}")

    # 3. Обрабатываем пропуски
    missing_before = df.isnull().sum().sum()

    # Для числовых колонок - заполняем медианой
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != 'defect_label' and df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Удаляем строки с пропущенной целевой переменной
    if 'defect_label' in df.columns:
        df = df.dropna(subset=['defect_label'])

    missing_after = df.isnull().sum().sum()
    print(f"Обработано пропусков: {missing_before - missing_after}")

    # 4. Проверяем числовые значения
    if 'defect_label' in df.columns:
        # Оставляем только 0 и 1
        df = df[df['defect_label'].isin([0, 1])]

    # 5. Убираем отрицательные значения в метриках
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != 'defect_label':
            df = df[df[col] >= 0]

    print(f"Строк до очистки: {original_rows}")
    print(f"Строк после очистки: {len(df)}")
    print(f"Удалено: {original_rows - len(df)} ({((original_rows - len(df))/original_rows*100):.1f}%)")

    # Сохраняем результат
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/clean_defect_data.csv", index=False)
    print(f" Данные сохранены в data/processed/clean_defect_data.csv")

    return df
