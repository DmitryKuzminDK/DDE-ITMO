"""
Модуль извлечения данных (Extract) для ETL-конвейера.
Поддерживает загрузку с Kaggle и из локальных CSV-файлов.
"""

import os
import sys
import logging
import pandas as pd
import zipfile
from pathlib import Path
from dotenv import load_dotenv
import subprocess
import shutil

# Загружаем переменные окружения из .env файла
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataExtractor:
    """
    Класс для извлечения данных из различных источников.
    Поддерживает Kaggle и локальные файлы.
    """
    
    def __init__(self):
        """Инициализация экстрактора с проверкой доступных источников."""
        self.data = None
        self.source_type = None
        self.source_path = None
        
        # Создаем необходимые директории
        self.raw_data_dir = Path("data/raw")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Проверяем доступность Kaggle API
        self.kaggle_available = self._check_kaggle()
        
    def _check_kaggle(self):
        """
        Проверяет наличие и правильность настройки Kaggle API.
        
        Returns:
            bool: True если Kaggle API доступен, иначе False
        """
        try:
            # Проверяем наличие kaggle.json
            kaggle_json_path = Path.home() / '.kaggle' / 'kaggle.json'
            
            if not kaggle_json_path.exists():
                logger.warning("Kaggle API не настроен. Файл kaggle.json не найден.")
                logger.info("Для настройки Kaggle API:")
                logger.info("1. Скачайте kaggle.json с https://www.kaggle.com/settings")
                logger.info("2. Поместите файл в ~/.kaggle/kaggle.json")
                logger.info("3. Установите права: chmod 600 ~/.kaggle/kaggle.json")
                return False
            
            # Проверяем права доступа (важно для Linux/Mac)
            if os.name != 'nt':  # Не Windows
                import stat
                mode = os.stat(kaggle_json_path).st_mode
                if mode & (stat.S_IRWXG | stat.S_IRWXO):
                    logger.warning("Файл kaggle.json имеет неправильные права доступа")
                    logger.info("Выполните: chmod 600 ~/.kaggle/kaggle.json")
            
            # Пробуем выполнить тестовую команду
            result = subprocess.run(
                ['kaggle', 'datasets', 'list', '--limit', '1'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info("Kaggle API успешно настроен и доступен")
                return True
            else:
                logger.error(f"Ошибка при проверке Kaggle API: {result.stderr}")
                return False
                
        except FileNotFoundError:
            logger.error("Kaggle CLI не установлен. Установите: pip install kaggle")
            return False
        except Exception as e:
            logger.error(f"Неожиданная ошибка при проверке Kaggle: {e}")
            return False
    
    def extract_from_kaggle(self, dataset_name=None, filename=None):
        """
        Загружает датасет с Kaggle.
        
        Args:
            dataset_name (str): Идентификатор датасета (username/dataset-name)
            filename (str): Имя конкретного файла в датасете (опционально)
        
        Returns:
            pd.DataFrame: Загруженные данные или None в случае ошибки
        """
        try:
            # Получаем параметры из .env, если не указаны явно
            if dataset_name is None:
                dataset_name = os.getenv('KAGGLE_DATASET')
            
            if not dataset_name:
                logger.error("Не указан идентификатор датасета Kaggle")
                return None
            
            logger.info(f"=" * 70)
            logger.info(f"ЭТАП 1: ИЗВЛЕЧЕНИЕ ДАННЫХ С KAGGLE")
            logger.info(f"=" * 70)
            logger.info(f"Датасет: {dataset_name}")
            
            # Создаем временную директорию для загрузки
            temp_dir = self.raw_data_dir / "temp_download"
            temp_dir.mkdir(exist_ok=True)
            
            # Формируем команду для загрузки
            cmd = [
                'kaggle', 'datasets', 'download',
                '-d', dataset_name,
                '-p', str(temp_dir),
                '--unzip'
            ]
            
            logger.info(f"Загрузка датасета...")
            
            # Выполняем загрузку
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                logger.error(f"Ошибка при загрузке: {result.stderr}")
                return None
            
            logger.info(f"Датасет успешно загружен в {temp_dir}")
            
            # Ищем CSV файлы в распакованной директории
            csv_files = list(temp_dir.glob("*.csv"))
            
            if not csv_files:
                logger.error("CSV файлы не найдены в загруженном датасете")
                return None
            
            # Если указан конкретный файл, ищем его
            if filename:
                target_file = temp_dir / filename
                if target_file.exists():
                    csv_files = [target_file]
                else:
                    logger.warning(f"Файл {filename} не найден. Будут использованы все CSV файлы.")
            
            # Загружаем данные
            if len(csv_files) == 1:
                # Один файл - загружаем как DataFrame
                file_path = csv_files[0]
                logger.info(f"Загрузка файла: {file_path.name}")
                self.data = pd.read_csv(file_path)
                self.source_path = file_path
                
                # Сохраняем копию в raw директорию
                raw_file_path = self.raw_data_dir / "raw_defect_data.csv"
                shutil.copy2(file_path, raw_file_path)
                logger.info(f"Копия сохранена в: {raw_file_path}")
                
            else:
                # Несколько файлов - объединяем
                logger.info(f"Найдено {len(csv_files)} CSV файлов. Выполняется объединение...")
                data_frames = []
                for file_path in csv_files:
                    logger.info(f"Загрузка: {file_path.name}")
                    df = pd.read_csv(file_path)
                    df['source_file'] = file_path.name
                    data_frames.append(df)
                
                self.data = pd.concat(data_frames, ignore_index=True)
            
            # Очищаем временную директорию
            shutil.rmtree(temp_dir)
            
            logger.info(f"✅ Данные успешно загружены!")
            logger.info(f"   - Форма: {self.data.shape}")
            logger.info(f"   - Колонки: {', '.join(self.data.columns[:5])}...")
            logger.info(f"   - Всего колонок: {len(self.data.columns)}")
            
            self.source_type = 'kaggle'
            return self.data
            
        except subprocess.TimeoutExpired:
            logger.error("Превышено время ожидания при загрузке с Kaggle")
            return None
        except Exception as e:
            logger.error(f"Ошибка при загрузке с Kaggle: {e}")
            return None
    
    def extract_from_local(self, file_path):
        """
        Загружает данные из локального CSV файла.
        
        Args:
            file_path (str): Путь к локальному CSV файлу
        
        Returns:
            pd.DataFrame: Загруженные данные или None в случае ошибки
        """
        try:
            logger.info(f"=" * 70)
            logger.info(f"ЭТАП 1: ИЗВЛЕЧЕНИЕ ДАННЫХ ИЗ ЛОКАЛЬНОГО ФАЙЛА")
            logger.info(f"=" * 70)
            
            # Проверяем существование файла
            path = Path(file_path)
            if not path.exists():
                logger.error(f"Файл не найден: {file_path}")
                return None
            
            logger.info(f"Загрузка файла: {path}")
            
            # Определяем формат файла по расширению
            if path.suffix.lower() == '.csv':
                self.data = pd.read_csv(path)
            elif path.suffix.lower() in ['.xlsx', '.xls']:
                self.data = pd.read_excel(path)
            elif path.suffix.lower() == '.parquet':
                self.data = pd.read_parquet(path)
            else:
                logger.error(f"Неподдерживаемый формат файла: {path.suffix}")
                return None
            
            # Сохраняем копию в raw директорию
            raw_file_path = self.raw_data_dir / "raw_defect_data.csv"
            self.data.to_csv(raw_file_path, index=False)
            
            logger.info(f"✅ Данные успешно загружены!")
            logger.info(f"   - Форма: {self.data.shape}")
            logger.info(f"   - Колонки: {', '.join(self.data.columns[:5])}...")
            logger.info(f"   - Всего колонок: {len(self.data.columns)}")
            logger.info(f"   - Копия сохранена в: {raw_file_path}")
            
            self.source_type = 'local'
            self.source_path = path
            return self.data
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке локального файла: {e}")
            return None
    
    def extract_from_env(self):
        """
        Загружает данные согласно настройкам в .env файле.
        
        Returns:
            pd.DataFrame: Загруженные данные или None в случае ошибки
        """
        method = os.getenv('DATA_SOURCE_METHOD', 'kaggle')
        
        if method.lower() == 'kaggle':
            dataset = os.getenv('KAGGLE_DATASET')
            filename = os.getenv('KAGGLE_FILENAME')
            
            if not self.kaggle_available:
                logger.error("Kaggle API недоступен. Проверьте настройки.")
                return None
            
            return self.extract_from_kaggle(dataset, filename)
            
        elif method.lower() == 'local':
            local_path = os.getenv('LOCAL_DATA_PATH')
            if not local_path:
                logger.error("LOCAL_DATA_PATH не указан в .env файле")
                return None
            
            return self.extract_from_local(local_path)
        else:
            logger.error(f"Неизвестный метод загрузки: {method}")
            return None
    
    def get_data_info(self):
        """
        Возвращает информацию о загруженных данных.
        
        Returns:
            dict: Словарь с информацией о данных
        """
        if self.data is None:
            return {"error": "Данные не загружены"}
        
        info = {
            "source_type": self.source_type,
            "source_path": str(self.source_path) if self.source_path else None,
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "memory_usage": f"{self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        
        # Добавляем статистику для числовых колонок
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            info["numeric_stats"] = self.data[numeric_cols].describe().to_dict()
        
        return info
    
    def preview_data(self, n_rows=5):
        """
        Показывает превью загруженных данных.
        
        Args:
            n_rows (int): Количество строк для отображения
        """
        if self.data is None:
            logger.warning("Нет данных для предпросмотра")
            return
        
        print("\n" + "="*70)
        print("ПРЕДПРОСМОТР ДАННЫХ")
        print("="*70)
        print(f"\nПервые {n_rows} строк:")
        print(self.data.head(n_rows))
        
        print(f"\nПоследние {n_rows} строк:")
        print(self.data.tail(n_rows))
        
        print("\nИнформация о данных:")
        print(self.data.info())
        
        print("\nСтатистика числовых колонок:")
        print(self.data.describe())


# Функции для обратной совместимости и простого использования
def extract_from_kaggle(dataset_name=None, filename=None):
    """
    Упрощенная функция для загрузки с Kaggle.
    """
    extractor = DataExtractor()
    return extractor.extract_from_kaggle(dataset_name, filename)


def extract_from_local(file_path):
    """
    Упрощенная функция для загрузки из локального файла.
    """
    extractor = DataExtractor()
    return extractor.extract_from_local(file_path)


def extract_from_env():
    """
    Упрощенная функция для загрузки согласно .env.
    """
    extractor = DataExtractor()
    return extractor.extract_from_env()


# Тестирование модуля при прямом запуске
if __name__ == "__main__":
    print("="*70)
    print("ТЕСТИРОВАНИЕ МОДУЛЯ ИЗВЛЕЧЕНИЯ ДАННЫХ")
    print("="*70)
    
    # Создаем экстрактор
    extractor = DataExtractor()
    
    # Проверяем доступность Kaggle
    print(f"\nKaggle API доступен: {extractor.kaggle_available}")
    
    # Пробуем загрузить из .env настроек
    print("\nПопытка загрузки из .env настроек...")
    data = extractor.extract_from_env()
    
    if data is not None:
        print("\n✅ Тест успешен! Данные загружены.")
        extractor.preview_data(3)
        
        # Показываем информацию о данных
        print("\n" + "="*70)
        print("ИНФОРМАЦИЯ О ДАННЫХ")
        print("="*70)
        info = extractor.get_data_info()
        for key, value in info.items():
            if key not in ['dtypes', 'missing_values', 'numeric_stats']:
                print(f"{key}: {value}")
    else:
        print("\n❌ Тест не удался. Проверьте настройки в .env файле.")
