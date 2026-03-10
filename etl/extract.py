"""
Упрощенный модуль загрузки данных с Google Drive
"""

import os
import pandas as pd
import gdown
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def extract_from_gdrive(file_id=None):
    """
    Загружает файл с Google Drive
    """
    print("=" * 50)
    print("ЗАГРУЗКА С GOOGLE DRIVE")
    print("=" * 50)

    if file_id is None:
        file_id = os.getenv('GDRIVE_FILE_ID')

    if not file_id:
        print("❌ Не указан ID файла на Google Drive")
        return None

    # Создаем папку для данных
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    # Формируем ссылку для скачивания
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "data/raw/raw_defect_data.csv"

    try:
        print(f"Скачивание файла с Google Drive...")
        gdown.download(url, output, quiet=False)

        # Загружаем данные
        df = pd.read_csv(output)
        print(f"✅ Загружено {len(df)} строк, {len(df.columns)} колонок")
        print(f"Колонки: {list(df.columns)}")
        return df

    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return None

def extract_from_local(file_path):
    """
    Загружает локальный файл
    """
    print("=" * 50)
    print("ЗАГРУЗКА ЛОКАЛЬНОГО ФАЙЛА")
    print("=" * 50)

    path = Path(file_path)
    if not path.exists():
        print(f"❌ Файл не найден: {file_path}")
        return None

    df = pd.read_csv(path)

    # Сохраняем копию
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/raw/raw_defect_data.csv", index=False)

    print(f"✅ Загружено {len(df)} строк, {len(df.columns)} колонок")
    return df

def extract_from_env():
    """
    Загружает согласно настройкам в .env
    """
    method = os.getenv('DATA_SOURCE_METHOD', 'gdrive')

    if method == 'gdrive':
        return extract_from_gdrive()
    else:
        return extract_from_local(os.getenv('LOCAL_DATA_PATH'))
