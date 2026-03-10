## Простой ETL для данных о дефектах ПО

Простой ETL-конвейер для загрузки, очистки и анализа данных о дефектах в программном обеспечении.

## Что делает этот проект?

1. **Загружает** данные с Google Drive или локального файла
2. **Очищает** данные (дубликаты, пропуски, ошибки)
3. **Сохраняет** результат для анализа
4. **Показывает** простые графики в Jupyter Notebook

## Быстрый старт

```bash
# 1. Установка
pip install -r requirements.txt

# 2. Настройка .env файла
echo "GDRIVE_FILE_ID=1O3GpKitaVD4jjsqNiwahw8qU2ARUxDMJ" > .env

# 3. Запуск ETL
python etl/main.py

# 4. Анализ
jupyter notebook notebooks/EDA.ipynb
