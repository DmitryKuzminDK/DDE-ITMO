"""
Модуль трансформации данных (Transform) для ETL-конвейера.
Выполняет очистку, валидацию и подготовку метрик кода.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataTransformer:
    """
    Класс для трансформации и очистки данных о метриках ПО.
    Выполняет валидацию, нормализацию и подготовку данных.
    """
    
    def __init__(self):
        """Инициализация трансформера с настройками валидации."""
        self.original_data = None
        self.transformed_data = None
        self.validation_report = {}
        
        # Создаем директории для сохранения результатов
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Определяем ожидаемые колонки и их типы для датасета дефектов ПО
        self.expected_columns = {
            'loc': 'int64',
            'cyclo_complexity': 'int64',
            'halstead_volume': 'float64',
            'halstead_difficulty': 'float64',
            'halstead_effort': 'float64',
            'num_operators': 'int64',
            'num_operands': 'int64',
            'branch_count': 'int64',
            'call_pairs': 'int64',
            'condition_count': 'int64',
            'multiple_condition_count': 'int64',
            'defect_label': 'int64'
        }
        
        # Определяем допустимые диапазоны для метрик
        self.validation_ranges = {
            'loc': (1, 10000),
            'cyclo_complexity': (1, 1000),
            'halstead_volume': (0, 100000),
            'halstead_difficulty': (0, 1000),
            'halstead_effort': (0, 10000000),
            'branch_count': (0, 1000),
            'condition_count': (0, 1000),
            'defect_label': (0, 1)
        }
    
    def _clean_column_names(self, df):
        """
        Очищает и нормализует названия колонок.
        
        Args:
            df (pd.DataFrame): Исходный DataFrame
        
        Returns:
            pd.DataFrame: DataFrame с очищенными названиями колонок
        """
        logger.info("Очистка названий колонок...")
        
        original_columns = df.columns.tolist()
        cleaned_columns = []
        
        for col in original_columns:
            # Приводим к нижнему регистру
            col_clean = col.lower().strip()
            
            # Заменяем пробелы и спецсимволы на подчеркивания
            col_clean = col_clean.replace(' ', '_')
            col_clean = col_clean.replace('-', '_')
            col_clean = col_clean.replace('/', '_')
            col_clean = col_clean.replace('\\', '_')
            col_clean = col_clean.replace('.', '_')
            
            # Удаляем лишние подчеркивания
            while '__' in col_clean:
                col_clean = col_clean.replace('__', '_')
            
            # Удаляем подчеркивания в начале и конце
            col_clean = col_clean.strip('_')
            
            cleaned_columns.append(col_clean)
        
        df.columns = cleaned_columns
        
        # Логируем изменения
        changes = [(orig, new) for orig, new in zip(original_columns, cleaned_columns) if orig != new]
        if changes:
            logger.info(f"  - Переименовано колонок: {len(changes)}")
            for orig, new in changes[:5]:
                logger.info(f"    * '{orig}' -> '{new}'")
            if len(changes) > 5:
                logger.info(f"    ... и еще {len(changes) - 5}")
        
        return df
    
    def _validate_numeric_columns(self, df):
        """
        Проверяет и корректирует числовые колонки.
        
        Args:
            df (pd.DataFrame): DataFrame для проверки
        
        Returns:
            pd.DataFrame: DataFrame с проверенными числовыми значениями
        """
        logger.info("Валидация числовых колонок...")
        
        validation_stats = {
            'total_rows': len(df),
            'invalid_values': {},
            'corrected_values': {},
            'removed_rows': 0
        }
        
        # Проходим по каждой колонке с ожидаемым типом
        for col, expected_type in self.expected_columns.items():
            if col not in df.columns:
                logger.warning(f"  - Колонка '{col}' отсутствует в данных")
                continue
            
            # Проверяем тип данных
            if expected_type == 'int64':
                # Пробуем конвертировать в числа
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Проверяем на целые числа
                if df[col].dtype == 'float64':
                    df[col] = df[col].round().astype('Int64')
            
            elif expected_type == 'float64':
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Проверяем допустимые диапазоны
            if col in self.validation_ranges:
                min_val, max_val = self.validation_ranges[col]
                
                # Находим некорректные значения
                invalid_mask = (df[col] < min_val) | (df[col] > max_val)
                invalid_count = invalid_mask.sum()
                
                if invalid_count > 0:
                    validation_stats['invalid_values'][col] = invalid_count
                    
                    # Для бинарных меток просто удаляем некорректные строки
                    if col == 'defect_label':
                        df = df[~invalid_mask]
                        validation_stats['removed_rows'] += invalid_count
                        logger.info(f"  - Удалено {invalid_count} строк с некорректной меткой дефекта")
                    
                    # Для остальных метрик заменяем на NaN
                    else:
                        df.loc[invalid_mask, col] = np.nan
                        logger.info(f"  - В колонке '{col}' обнаружено {invalid_count} значений вне диапазона [{min_val}, {max_val}]")
        
        self.validation_report['numeric_validation'] = validation_stats
        return df
    
    def _handle_missing_values(self, df):
        """
        Обрабатывает пропущенные значения.
        
        Args:
            df (pd.DataFrame): DataFrame с пропусками
        
        Returns:
            pd.DataFrame: DataFrame с обработанными пропусками
        """
        logger.info("Обработка пропущенных значений...")
        
        missing_stats = {
            'total_missing': df.isnull().sum().sum(),
            'missing_by_column': df.isnull().sum().to_dict(),
            'rows_with_missing': df.isnull().any(axis=1).sum(),
            'removed_rows': 0
        }
        
        if missing_stats['total_missing'] == 0:
            logger.info("  - Пропущенные значения отсутствуют")
            return df
        
        logger.info(f"  - Всего пропусков: {missing_stats['total_missing']}")
        logger.info(f"  - Строк с пропусками: {missing_stats['rows_with_missing']}")
        
        # Показываем распределение пропусков по колонкам
        missing_cols = {k: v for k, v in missing_stats['missing_by_column'].items() if v > 0}
        for col, count in missing_cols.items():
            percentage = (count / len(df)) * 100
            logger.info(f"    * {col}: {count} ({percentage:.1f}%)")
        
        rows_before = len(df)
        
        # Удаляем строки, где пропущена целевая переменная
        if 'defect_label' in df.columns:
            defect_missing = df['defect_label'].isnull().sum()
            if defect_missing > 0:
                df = df.dropna(subset=['defect_label'])
                logger.info(f"  - Удалено {defect_missing} строк с пропущенной меткой дефекта")
        
        # Для числовых метрик: заполняем медианой
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col != 'defect_label' and df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"  - В колонке '{col}' пропуски заполнены медианой ({median_val:.2f})")
        
        missing_stats['removed_rows'] = rows_before - len(df)
        missing_stats['rows_after'] = len(df)
        
        self.validation_report['missing_values'] = missing_stats
        
        logger.info(f"  - После обработки: {len(df)} строк")
        return df
    
    def _remove_duplicates(self, df):
        """
        Удаляет дубликаты строк.
        
        Args:
            df (pd.DataFrame): DataFrame для проверки
        
        Returns:
            pd.DataFrame: DataFrame без дубликатов
        """
        logger.info("Поиск и удаление дубликатов...")
        
        duplicates_stats = {
            'total_duplicates': 0,
            'removed_duplicates': 0
        }
        
        # Находим полные дубликаты
        full_duplicates = df.duplicated(keep='first')
        duplicates_count = full_duplicates.sum()
        
        if duplicates_count > 0:
            duplicates_stats['total_duplicates'] = duplicates_count
            
            # Удаляем дубликаты
            df = df.drop_duplicates(keep='first')
            duplicates_stats['removed_duplicates'] = duplicates_count
            
            logger.info(f"  - Удалено {duplicates_count} полных дубликатов")
            
            # Проверяем частичные дубликаты
            if 'defect_label' in df.columns:
                feature_cols = [c for c in df.columns if c != 'defect_label']
                conflicts = df.groupby(feature_cols)['defect_label'].nunique()
                conflict_count = (conflicts > 1).sum()
                
                if conflict_count > 0:
                    logger.warning(f"  - Найдено {conflict_count} групп с противоречивыми метками дефекта")
                    duplicates_stats['conflict_groups'] = conflict_count
        else:
            logger.info("  - Дубликаты не найдены")
        
        self.validation_report['duplicates'] = duplicates_stats
        return df
    
    def _normalize_data(self, df):
        """
        Нормализует данные для анализа.
        
        Args:
            df (pd.DataFrame): DataFrame для нормализации
        
        Returns:
            pd.DataFrame: Нормализованный DataFrame
        """
        logger.info("Нормализация данных...")
        
        df_norm = df.copy()
        
        numeric_cols = df_norm.select_dtypes(include=['number']).columns
        feature_cols = [c for c in numeric_cols if c != 'defect_label']
        
        if feature_cols:
            for col in feature_cols:
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                
                if max_val > min_val:
                    df_norm[f'{col}_normalized'] = (df_norm[col] - min_val) / (max_val - min_val)
                    logger.info(f"  - Добавлена нормализованная колонка: {col}_normalized")
            
            for col in ['loc', 'halstead_volume', 'halstead_effort']:
                if col in df_norm.columns:
                    df_norm[f'{col}_log'] = np.log1p(df_norm[col])
                    logger.info(f"  - Добавлена логарифмическая колонка: {col}_log")
        
        self.validation_report['normalization'] = {
            'features_normalized': len(feature_cols),
            'added_columns': [f'{c}_normalized' for c in feature_cols] + 
                            [f'{c}_log' for c in ['loc', 'halstead_volume', 'halstead_effort'] 
                             if c in df_norm.columns]
        }
        
        return df_norm
    
    def transform(self, data, save_intermediate=True):
        """
        Выполняет полный цикл трансформации данных.
        
        Args:
            data (pd.DataFrame): Исходные данные
            save_intermediate (bool): Сохранять ли промежуточные результаты
        
        Returns:
            pd.DataFrame: Трансформированные данные
        """
        try:
            logger.info("=" * 70)
            logger.info("ЭТАП 2: ТРАНСФОРМАЦИЯ И ОЧИСТКА ДАННЫХ (TRANSFORM)")
            logger.info("=" * 70)
            
            self.original_data = data.copy()
            
            initial_stats = {
                'rows': len(data),
                'columns': len(data.columns),
                'memory_mb': data.memory_usage(deep=True).sum() / 1024**2,
                'missing_values': data.isnull().sum().sum()
            }
            
            logger.info(f"\nИсходные данные:")
            logger.info(f"  - Строк: {initial_stats['rows']}")
            logger.info(f"  - Колонок: {initial_stats['columns']}")
            logger.info(f"  - Память: {initial_stats['memory_mb']:.2f} MB")
            
            df = self._clean_column_names(data)
            df = self._validate_numeric_columns(df)
            df = self._handle_missing_values(df)
            df = self._remove_duplicates(df)
            df = self._normalize_data(df)
            
            self.transformed_data = df
            
            final_stats = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'missing_values': df.isnull().sum().sum()
            }
            
            self.validation_report['summary'] = {
                'initial': initial_stats,
                'final': final_stats,
                'rows_removed': initial_stats['rows'] - final_stats['rows'],
                'rows_removed_percentage': ((initial_stats['rows'] - final_stats['rows']) / initial_stats['rows'] * 100) if initial_stats['rows'] > 0 else 0,
                'columns_added': final_stats['columns'] - initial_stats['columns']
            }
            
            logger.info(f"\n{'='*70}")
            logger.info("ИТОГИ ТРАНСФОРМАЦИИ:")
            logger.info(f"{'='*70}")
            logger.info(f"  • Исходных записей: {initial_stats['rows']}")
            logger.info(f"  • Финальных записей: {final_stats['rows']}")
            logger.info(f"  • Удалено: {self.validation_report['summary']['rows_removed']} "
                       f"({self.validation_report['summary']['rows_removed_percentage']:.2f}%)")
            logger.info(f"  • Исходных колонок: {initial_stats['columns']}")
            logger.info(f"  • Финальных колонок: {final_stats['columns']}")
            logger.info(f"  • Добавлено колонок: {final_stats['columns'] - initial_stats['columns']}")
            
            if save_intermediate:
                self._save_processed_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка при трансформации данных: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_processed_data(self, df):
        """
        Сохраняет обработанные данные в CSV и Parquet форматах.
        
        Args:
            df (pd.DataFrame): Обработанные данные
        """
        try:
            logger.info(f"\nСохранение обработанных данных...")
            
            csv_path = self.processed_dir / "clean_defect_data.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"  ✓ CSV сохранен: {csv_path} ({csv_path.stat().st_size / 1024**2:.2f} MB)")
            
            parquet_path = self.processed_dir / "clean_defect_data.parquet"
            df.to_parquet(parquet_path, index=False)
            logger.info(f"  ✓ Parquet сохранен: {parquet_path} ({parquet_path.stat().st_size / 1024**2:.2f} MB)")
            
            report_path = self.processed_dir / "transformation_report.txt"
            with open(report_path, 'w') as f:
                f.write("ОТЧЕТ О ТРАНСФОРМАЦИИ ДАННЫХ\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("Сводка:\n")
                for key, value in self.validation_report.get('summary', {}).items():
                    f.write(f"  {key}: {value}\n")
                
                f.write("\nДетали валидации:\n")
                for key, value in self.validation_report.items():
                    if key != 'summary':
                        f.write(f"\n{key}:\n")
                        f.write(f"  {value}\n")
            
            logger.info(f"  ✓ Отчет сохранен: {report_path}")
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении обработанных данных: {e}")
    
    def get_transformation_report(self):
        """
        Возвращает отчет о трансформации.
        
        Returns:
            dict: Отчет с деталями трансформации
        """
        return self.validation_report
    
    def prepare_for_ml(self, target_col='defect_label'):
        """
        Подготавливает данные для машинного обучения.
        
        Args:
            target_col (str): Название целевой колонки
        
        Returns:
            tuple: (X, y) - признаки и целевая переменная
        """
        if self.transformed_data is None:
            logger.error("Данные не трансформированы. Сначала выполните transform().")
            return None, None
        
        df = self.transformed_data.copy()
        
        if target_col in df.columns:
            y = df[target_col]
            
            feature_cols = df.select_dtypes(include=['number']).columns
            feature_cols = [c for c in feature_cols if c != target_col and not c.endswith('_normalized')]
            
            X = df[feature_cols]
            
            logger.info(f"Данные подготовлены для ML:")
            logger.info(f"  - Признаков: {X.shape[1]}")
            logger.info(f"  - Образцов: {X.shape[0]}")
            logger.info(f"  - Целевая переменная: {target_col}")
            logger.info(f"  - Распределение классов: {y.value_counts().to_dict()}")
            
            return X, y
        else:
            logger.error(f"Целевая колонка '{target_col}' не найдена")
            return None, None


def transform_data(data, save_intermediate=True):
    """
    Упрощенная функция для трансформации данных.
    """
    transformer = DataTransformer()
    return transformer.transform(data, save_intermediate)


def quick_clean(data):
    """
    Быстрая очистка данных без детального отчета.
    """
    transformer = DataTransformer()
    df = transformer._clean_column_names(data)
    df = transformer._validate_numeric_columns(df)
    df = transformer._handle_missing_values(df)
    df = transformer._remove_duplicates(df)
    return df


if __name__ == "__main__":
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ МОДУЛЯ ТРАНСФОРМАЦИИ ДАННЫХ")
    print("=" * 70)
    
    test_data = pd.DataFrame({
        'LOC': [100, 250, 150, 500, -10, 3000, 200],
        'Cyclo Complexity': [5, 10, 8, 25, 3, 45, 12],
        'Halstead Volume': [1000, 2500, 1500, 5000, -500, 30000, 2000],
        'defect label': [0, 1, 0, 1, 2, 1, 0],
        'branch count': [10, 20, 15, 35, 8, 50, 18]
    })
    
    print("Исходные данные:")
    print(test_data)
    
    print("\n" + "=" * 70)
    print("ЗАПУСК ТРАНСФОРМАЦИИ")
    print("=" * 70)
    
    transformer = DataTransformer()
    transformed = transformer.transform(test_data, save_intermediate=True)
    
    if transformed is not None:
        print("\n✅ Трансформация успешно выполнена!")
        print("\nТрансформированные данные:")
        print(transformed.head())
        
        print("\nОтчет о трансформации:")
        report = transformer.get_transformation_report()
        for key, value in report.items():
            if key != 'summary':
                print(f"\n{key}:")
                print(f"  {value}")
        
        print("\nПодготовка данных для ML:")
        X, y = transformer.prepare_for_ml()
