"""
Главный модуль ETL
"""

import sys
import argparse
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.extract import extract_from_env, extract_from_local, extract_from_gdrive
from etl.transform import clean_data
from etl.load import save_final_data

def main():
    parser = argparse.ArgumentParser(description='Простой ETL для данных о дефектах ПО')

    parser.add_argument('--local', help='Загрузить из локального файла')
    parser.add_argument('--gdrive-id', help='ID файла на Google Drive')
    parser.add_argument('--only-extract', action='store_true', help='Только загрузка')
    parser.add_argument('--only-transform', action='store_true', help='Только очистка')
    parser.add_argument('--only-load', action='store_true', help='Только сохранение')

    args = parser.parse_args()

    print("""
    ╔══════════════════════════════════════╗
    ║  Простой ETL для данных о дефектах   ║
    ╚══════════════════════════════════════╝
    """)

    # Режимы работы
    if args.only_extract:
        if args.local:
            extract_from_local(args.local)
        elif args.gdrive_id:
            extract_from_gdrive(args.gdrive_id)
        else:
            extract_from_env()

    elif args.only_transform:
        if Path("data/raw/raw_defect_data.csv").exists():
            df = pd.read_csv("data/raw/raw_defect_data.csv")
            clean_data(df)
        else:
            print("❌ Сначала выполните extract")

    elif args.only_load:
        if Path("data/processed/clean_defect_data.csv").exists():
            df = pd.read_csv("data/processed/clean_defect_data.csv")
            save_final_data(df)
        else:
            print("❌ Сначала выполните transform")

    else:
        # Полный цикл
        print("\n1️⃣ ЭТАП: ЗАГРУЗКА")
        if args.local:
            df = extract_from_local(args.local)
        elif args.gdrive_id:
            df = extract_from_gdrive(args.gdrive_id)
        else:
            df = extract_from_env()

        if df is None:
            print("❌ Ошибка загрузки")
            return

        print("\n2️⃣ ЭТАП: ОЧИСТКА")
        df_clean = clean_data(df)

        print("\n3️⃣ ЭТАП: СОХРАНЕНИЕ")
        save_final_data(df_clean)

        print("\n✅ ГОТОВО! Данные сохранены в data/final/final_defect_data.csv")

if __name__ == "__main__":
    import pandas as pd
    main()
