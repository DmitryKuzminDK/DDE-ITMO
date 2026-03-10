"""
Модуль загрузки данных (Load) для ETL-конвейера.
Сохраняет обработанные данные в финальном формате и выполняет финальную валидацию.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Класс для загрузки обработанных данных в финальное хранилище.
    Поддерживает различные форматы и выполняет финальную валидацию.
    """
    
    def __init__(self):
        """Инициализация загрузчика с настройками путей."""
        self.final_dir = Path("data/final")
        self.final_dir.mkdir(parents=True, exist_ok=True)
        
        self.processed_dir = Path("data/processed")
        self.archive_dir = Path("data/archive")
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        self.load_stats = {}
        self.load_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def load_from_processed(self, filename="clean_defect_data.parquet"):
        """
        Загружает очищенные данные из processed директории.
        
        Args:
            filename (str): Имя файла для загрузки
        
        Returns:
            pd.DataFrame: Загруженные данные или None
        """
        try:
            file_path = self.processed_dir / filename
            
            if not file_path.exists():
                logger.error(f"Файл не найден: {file_path}")
                
                if filename.endswith('.parquet'):
                    alt_path = self.processed_dir / filename.replace('.parquet', '.csv')
                else:
                    alt_path = self.processed_dir / filename.replace('.csv', '.parquet')
                
                if alt_path.exists():
                    logger.info(f"Найден альтернативный файл: {alt_path}")
                    file_path = alt_path
                else:
                    return None
            
            if file_path.suffix.lower() == '.csv':
                data = pd.read_csv(file_path)
                logger.info(f"Загружен CSV файл: {file_path}")
            elif file_path.suffix.lower() == '.parquet':
                data = pd.read_parquet(file_path)
                logger.info(f"Загружен Parquet файл: {file_path}")
            else:
                logger.error(f"Неподдерживаемый формат файла: {file_path.suffix}")
                return None
            
            logger.info(f"  - Форма: {data.shape}")
            logger.info(f"  - Колонок: {len(data.columns)}")
            
            return data
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке из processed: {e}")
            return None
    
    def _validate_final_data(self, data):
        """
        Выполняет финальную валидацию данных перед сохранением.
        
        Args:
            data (pd.DataFrame): Данные для валидации
        
        Returns:
            tuple: (is_valid, validation_report)
        """
        logger.info("Финальная валидация данных...")
        
        validation_report = {
            'timestamp': self.load_timestamp,
            'checks': {},
            'warnings': [],
            'errors': []
        }
        
        if data.empty:
            validation_report['errors'].append("Датасет пуст")
            return False, validation_report
        else:
            validation_report['checks']['not_empty'] = True
        
        required_columns = ['loc', 'cyclo_complexity', 'defect_label']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            validation_report['errors'].append(f"Отсутствуют обязательные колонки: {missing_columns}")
        else:
            validation_report['checks']['required_columns'] = True
        
        expected_dtypes = {
            'defect_label': ['int64', 'float64', 'Int64'],
            'loc': ['int64', 'float64', 'Int64'],
            'cyclo_complexity': ['int64', 'float64', 'Int64']
        }
        
        for col, expected_types in expected_dtypes.items():
            if col in data.columns:
                if str(data[col].dtype) not in expected_types:
                    validation_report['warnings'].append(
                        f"Колонка {col} имеет тип {data[col].dtype}, ожидался один из {expected_types}"
                    )
        
        missing_total = data.isnull().sum().sum()
        if missing_total > 0:
            missing_by_col = data.isnull().sum()[data.isnull().sum() > 0].to_dict()
            validation_report['warnings'].append(f"Обнаружены пропущенные значения: {missing_total}")
            validation_report['checks']['missing_by_column'] = missing_by_col
        else:
            validation_report['checks']['no_missing'] = True
        
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            validation_report['warnings'].append(f"Обнаружено {duplicates} дубликатов")
        else:
            validation_report['checks']['no_duplicates'] = True
        
        if 'defect_label' in data.columns:
            class_dist = data['defect_label'].value_counts().to_dict()
            validation_report['class_distribution'] = class_dist
            
            if len(class_dist) == 2:
                minority_ratio = min(class_dist.values()) / sum(class_dist.values())
                if minority_ratio < 0.1:
                    validation_report['warnings'].append(
                        f"Сильный дисбаланс классов: доля минорного класса {minority_ratio:.1%}"
                    )
        
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            stats = data[numeric_cols].describe().to_dict()
            validation_report['numeric_stats'] = stats
            
            for col in numeric_cols:
                if col != 'defect_label':
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
                    if outliers > 0:
                        outlier_ratio = outliers / len(data)
                        if outlier_ratio > 0.05:
                            validation_report['warnings'].append(
                                f"Колонка {col}: {outliers} выбросов ({outlier_ratio:.1%})"
                            )
        
        is_valid = len(validation_report['errors']) == 0
        
        if is_valid:
            logger.info("✅ Финальная валидация пройдена успешно")
            if validation_report['warnings']:
                logger.info(f"   (с предупреждениями: {len(validation_report['warnings'])})")
        else:
            logger.error(f"❌ Ошибки валидации: {len(validation_report['errors'])}")
        
        return is_valid, validation_report
    
    def save_final_data(self, data, format='csv', archive_previous=True):
        """
        Сохраняет финальные данные с архивацией предыдущей версии.
        
        Args:
            data (pd.DataFrame): Данные для сохранения
            format (str): Формат сохранения ('csv' или 'parquet')
            archive_previous (bool): Архивировать ли предыдущую версию
        
        Returns:
            tuple: (success, file_path)
        """
        try:
            logger.info("=" * 70)
            logger.info("ЭТАП 3: ЗАГРУЗКА ФИНАЛЬНЫХ ДАННЫХ (LOAD)")
            logger.info("=" * 70)
            
            is_valid, validation_report = self._validate_final_data(data)
            
            if not is_valid:
                logger.error("Финальная валидация не пройдена. Сохранение отменено.")
                self.load_stats['validation'] = validation_report
                return False, None
            
            if archive_previous:
                self._archive_previous_version()
            
            if format.lower() == 'csv':
                filename = f"final_defect_data_{self.load_timestamp}.csv"
                final_path = self.final_dir / "final_defect_data.csv"
                archive_path = self.archive_dir / filename
            elif format.lower() == 'parquet':
                filename = f"final_defect_data_{self.load_timestamp}.parquet"
                final_path = self.final_dir / "final_defect_data.parquet"
                archive_path = self.archive_dir / filename
            else:
                logger.error(f"Неподдерживаемый формат: {format}")
                return False, None
            
            if format.lower() == 'csv':
                data.to_csv(final_path, index=False)
                data.to_csv(archive_path, index=False)
            else:
                data.to_parquet(final_path, index=False)
                data.to_parquet(archive_path, index=False)
            
            file_size_mb = final_path.stat().st_size / 1024**2
            
            self.load_stats = {
                'timestamp': self.load_timestamp,
                'format': format,
                'rows': len(data),
                'columns': len(data.columns),
                'file_size_mb': file_size_mb,
                'file_path': str(final_path),
                'archive_path': str(archive_path),
                'validation': validation_report
            }
            
            logger.info(f"\n✅ Данные успешно сохранены!")
            logger.info(f"  - Формат: {format.upper()}")
            logger.info(f"  - Строк: {len(data)}")
            logger.info(f"  - Колонок: {len(data.columns)}")
            logger.info(f"  - Размер файла: {file_size_mb:.2f} MB")
            logger.info(f"  - Путь: {final_path}")
            logger.info(f"  - Архив: {archive_path}")
            
            self._save_metadata(data)
            
            return True, final_path
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении финальных данных: {e}")
            import traceback
            traceback.print_exc()
            return False, None
    
    def _archive_previous_version(self):
        """Архивирует предыдущую версию финальных данных."""
        try:
            for ext in ['*.csv', '*.parquet']:
                for file_path in self.final_dir.glob(ext):
                    if file_path.name.startswith('final_defect_data'):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        archive_name = f"final_defect_data_previous_{timestamp}{file_path.suffix}"
                        archive_path = self.archive_dir / archive_name
                        
                        import shutil
                        shutil.copy2(file_path, archive_path)
                        logger.info(f"  - Предыдущая версия архивирована: {archive_path}")
                        
        except Exception as e:
            logger.warning(f"Не удалось архивировать предыдущую версию: {e}")
    
    def _save_metadata(self, data):
        """
        Сохраняет метаданные о загруженных данных.
        
        Args:
            data (pd.DataFrame): Загруженные данные
        """
        try:
            metadata = {
                'load_timestamp': self.load_timestamp,
                'dataset_info': {
                    'rows': len(data),
                    'columns': len(data.columns),
                    'column_names': list(data.columns),
                    'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()}
                },
                'summary_statistics': {},
                'load_stats': self.load_stats
            }
            
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                stats = data[numeric_cols].describe()
                metadata['summary_statistics'] = {
                    col: {
                        'mean': float(stats[col]['mean']) if pd.notna(stats[col]['mean']) else None,
                        'std': float(stats[col]['std']) if pd.notna(stats[col]['std']) else None,
                        'min': float(stats[col]['min']) if pd.notna(stats[col]['min']) else None,
                        '25%': float(stats[col]['25%']) if pd.notna(stats[col]['25%']) else None,
                        '50%': float(stats[col]['50%']) if pd.notna(stats[col]['50%']) else None,
                        '75%': float(stats[col]['75%']) if pd.notna(stats[col]['75%']) else None,
                        'max': float(stats[col]['max']) if pd.notna(stats[col]['max']) else None
                    }
                    for col in numeric_cols
                }
            
            metadata_path = self.final_dir / f"metadata_{self.load_timestamp}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"  - Метаданные сохранены: {metadata_path}")
            
        except Exception as e:
            logger.warning(f"Не удалось сохранить метаданные: {e}")
    
    def get_load_stats(self):
        """Возвращает статистику последней загрузки."""
        return self.load_stats
    
    def generate_data_dictionary(self, data=None):
        """
        Генерирует словарь данных (data dictionary) для документации.
        
        Args:
            data (pd.DataFrame, optional): Данные для анализа
        
        Returns:
            dict: Словарь данных
        """
        if data is None:
            data = self.load_from_processed()
            if data is None:
                logger.error("Нет данных для генерации словаря")
                return {}
        
        data_dict = {
            'generated_at': self.load_timestamp,
            'dataset_summary': {
                'rows': len(data),
                'columns': len(data.columns),
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2
            },
            'columns': {}
        }
        
        for col in data.columns:
            col_info = {
                'dtype': str(data[col].dtype),
                'null_count': int(data[col].isnull().sum()),
                'null_percentage': float((data[col].isnull().sum() / len(data)) * 100),
                'unique_values': int(data[col].nunique()) if data[col].nunique() < 100 else '>100'
            }
            
            if data[col].nunique() < 10:
                col_info['unique_values_list'] = data[col].unique().tolist()
            else:
                col_info['sample_values'] = data[col].dropna().sample(min(5, len(data))).tolist()
            
            if pd.api.types.is_numeric_dtype(data[col]):
                col_info['statistics'] = {
                    'min': float(data[col].min()) if pd.notna(data[col].min()) else None,
                    'max': float(data[col].max()) if pd.notna(data[col].max()) else None,
                    'mean': float(data[col].mean()) if pd.notna(data[col].mean()) else None,
                    'median': float(data[col].median()) if pd.notna(data[col].median()) else None,
                    'std': float(data[col].std()) if pd.notna(data[col].std()) else None
                }
            
            data_dict['columns'][col] = col_info
        
        dict_path = self.final_dir / f"data_dictionary_{self.load_timestamp}.json"
        with open(dict_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Словарь данных сохранен: {dict_path}")
        
        return data_dict


def load_data(data, format='csv', archive_previous=True):
    """Упрощенная функция для загрузки данных."""
    loader = DataLoader()
    return loader.save_final_data(data, format, archive_previous)


def load_from_processed(filename="clean_defect_data.parquet"):
    """Упрощенная функция для загрузки из processed."""
    loader = DataLoader()
    return loader.load_from_processed(filename)


if __name__ == "__main__":
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ МОДУЛЯ ЗАГРУЗКИ ДАННЫХ")
    print("=" * 70)
    
    np.random.seed(42)
    n_samples = 1000
    
    test_data = pd.DataFrame({
        'loc': np.random.randint(10, 5000, n_samples),
        'cyclo_complexity': np.random.randint(1, 100, n_samples),
        'halstead_volume': np.random.uniform(100, 50000, n_samples),
        'halstead_difficulty': np.random.uniform(1, 500, n_samples),
        'branch_count': np.random.randint(0, 200, n_samples),
        'defect_label': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    })
    
    test_data.loc[0:5, 'loc'] = -100
    test_data.loc[10:15, 'cyclo_complexity'] = np.nan
    
    print(f"Тестовые данные: {test_data.shape}")
    print(f"Распределение классов:\n{test_data['defect_label'].value_counts()}")
    
    print("\n" + "=" * 70)
    print("ЗАПУСК ЗАГРУЗКИ ДАННЫХ")
    print("=" * 70)
    
    loader = DataLoader()
    
    print("\n--- Тест 1: Сохранение в CSV ---")
    success, path = loader.save_final_data(test_data, format='csv')
    
    if success:
        print(f"✓ CSV сохранен успешно")
        
        print("\n--- Тест 2: Загрузка из processed ---")
        loaded_data = loader.load_from_processed("clean_defect_data.parquet")
        
        if loaded_data is not None:
            print(f"✓ Данные загружены из processed")
            print(f"  Форма: {loaded_data.shape}")
    
    print("\n--- Тест 3: Генерация словаря данных ---")
    data_dict = loader.generate_data_dictionary(test_data)
    print(f"✓ Словарь данных создан")
    print(f"  Колонок в словаре: {len(data_dict.get('columns', {}))}")
    
    print("\n--- Тест 4: Статистика загрузки ---")
    stats = loader.get_load_stats()
    if stats:
        print("Статистика последней загрузки:")
        for key, value in stats.items():
            if key != 'validation':
                print(f"  {key}: {value}")
