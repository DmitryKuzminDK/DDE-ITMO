"""
Главный модуль ETL-конвейера для анализа дефектов ПО.
Предоставляет CLI интерфейс для управления процессом.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.extract import DataExtractor, extract_from_env, extract_from_local, extract_from_kaggle
from etl.transform import DataTransformer, transform_data, quick_clean
from etl.load import DataLoader, load_data, load_from_processed

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'logs/etl_{datetime.now().strftime("%Y%m%d")}.log')
    ]
)
logger = logging.getLogger(__name__)

Path("logs").mkdir(exist_ok=True)


class ETLPipeline:
    """
    Главный класс ETL-конвейера, объединяющий все этапы.
    """
    
    def __init__(self, verbose=False):
        """
        Инициализация конвейера.
        
        Args:
            verbose (bool): Режим подробного вывода
        """
        self.verbose = verbose
        self.extractor = DataExtractor()
        self.transformer = DataTransformer()
        self.loader = DataLoader()
        
        self.pipeline_stats = {
            'start_time': None,
            'end_time': None,
            'stages': {},
            'errors': []
        }
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
    
    def run_extract(self, source=None, **kwargs):
        """
        Запускает этап извлечения данных.
        
        Args:
            source (str): Источник данных ('kaggle', 'local', 'env')
            **kwargs: Дополнительные параметры
        
        Returns:
            pd.DataFrame: Извлеченные данные или None
        """
        stage_start = datetime.now()
        logger.info("=" * 70)
        logger.info("ЗАПУСК ЭТАПА ИЗВЛЕЧЕНИЯ ДАННЫХ (EXTRACT)")
        logger.info("=" * 70)
        
        try:
            if source == 'kaggle' or (source is None and kwargs.get('kaggle_dataset')):
                dataset = kwargs.get('kaggle_dataset', os.getenv('KAGGLE_DATASET'))
                filename = kwargs.get('kaggle_filename', os.getenv('KAGGLE_FILENAME'))
                data = self.extractor.extract_from_kaggle(dataset, filename)
                
            elif source == 'local' or kwargs.get('local_file'):
                file_path = kwargs.get('local_file', os.getenv('LOCAL_DATA_PATH'))
                data = self.extractor.extract_from_local(file_path)
                
            elif source == 'env' or (source is None and not kwargs):
                data = self.extractor.extract_from_env()
                
            else:
                logger.error(f"Неизвестный источник данных: {source}")
                return None
            
            stage_time = (datetime.now() - stage_start).total_seconds()
            self.pipeline_stats['stages']['extract'] = {
                'status': 'success' if data is not None else 'failed',
                'duration': stage_time,
                'data_shape': data.shape if data is not None else None
            }
            
            if data is not None and self.verbose:
                self.extractor.preview_data(3)
            
            return data
            
        except Exception as e:
            logger.error(f"Ошибка на этапе извлечения: {e}")
            self.pipeline_stats['errors'].append({
                'stage': 'extract',
                'error': str(e),
                'time': datetime.now().isoformat()
            })
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None
    
    def run_transform(self, data=None, save_intermediate=True):
        """
        Запускает этап трансформации данных.
        
        Args:
            data (pd.DataFrame): Данные для трансформации
            save_intermediate (bool): Сохранять промежуточные результаты
        
        Returns:
            pd.DataFrame: Трансформированные данные или None
        """
        stage_start = datetime.now()
        logger.info("=" * 70)
        logger.info("ЗАПУСК ЭТАПА ТРАНСФОРМАЦИИ ДАННЫХ (TRANSFORM)")
        logger.info("=" * 70)
        
        try:
            if data is None:
                raw_path = Path("data/raw/raw_defect_data.csv")
                if raw_path.exists():
                    logger.info("Загрузка данных из raw директории...")
                    data = pd.read_csv(raw_path)
                else:
                    logger.error("Нет данных для трансформации")
                    return None
            
            transformed = self.transformer.transform(data, save_intermediate)
            
            stage_time = (datetime.now() - stage_start).total_seconds()
            self.pipeline_stats['stages']['transform'] = {
                'status': 'success' if transformed is not None else 'failed',
                'duration': stage_time,
                'data_shape': transformed.shape if transformed is not None else None,
                'report': self.transformer.get_transformation_report()
            }
            
            return transformed
            
        except Exception as e:
            logger.error(f"Ошибка на этапе трансформации: {e}")
            self.pipeline_stats['errors'].append({
                'stage': 'transform',
                'error': str(e),
                'time': datetime.now().isoformat()
            })
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None
    
    def run_load(self, data=None, format='csv', archive_previous=True):
        """
        Запускает этап загрузки данных.
        
        Args:
            data (pd.DataFrame): Данные для загрузки
            format (str): Формат сохранения
            archive_previous (bool): Архивировать предыдущую версию
        
        Returns:
            tuple: (success, file_path)
        """
        stage_start = datetime.now()
        logger.info("=" * 70)
        logger.info("ЗАПУСК ЭТАПА ЗАГРУЗКИ ДАННЫХ (LOAD)")
        logger.info("=" * 70)
        
        try:
            if data is None:
                logger.info("Загрузка данных из processed директории...")
                data = self.loader.load_from_processed()
                if data is None:
                    logger.error("Нет данных для загрузки")
                    return False, None
            
            success, path = self.loader.save_final_data(data, format, archive_previous)
            
            stage_time = (datetime.now() - stage_start).total_seconds()
            self.pipeline_stats['stages']['load'] = {
                'status': 'success' if success else 'failed',
                'duration': stage_time,
                'file_path': str(path) if path else None,
                'format': format,
                'stats': self.loader.get_load_stats()
            }
            
            return success, path
            
        except Exception as e:
            logger.error(f"Ошибка на этапе загрузки: {e}")
            self.pipeline_stats['errors'].append({
                'stage': 'load',
                'error': str(e),
                'time': datetime.now().isoformat()
            })
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False, None
    
    def run_full_pipeline(self, source=None, **kwargs):
        """
        Запускает полный ETL-конвейер.
        
        Args:
            source (str): Источник данных
            **kwargs: Дополнительные параметры
        
        Returns:
            bool: Успешность выполнения
        """
        self.pipeline_stats['start_time'] = datetime.now().isoformat()
        logger.info("=" * 70)
        logger.info("ЗАПУСК ПОЛНОГО ETL-КОНВЕЙЕРА")
        logger.info("=" * 70)
        
        data = self.run_extract(source, **kwargs)
        if data is None:
            logger.error("Прерывание конвейера: ошибка на этапе Extract")
            self._print_summary()
            return False
        
        transformed = self.run_transform(data)
        if transformed is None:
            logger.error("Прерывание конвейера: ошибка на этапе Transform")
            self._print_summary()
            return False
        
        format = kwargs.get('format', 'csv')
        success, path = self.run_load(transformed, format)
        
        self.pipeline_stats['end_time'] = datetime.now().isoformat()
        
        self._print_summary()
        
        return success
    
    def _print_summary(self):
        """Выводит сводку по выполнению конвейера."""
        logger.info("=" * 70)
        logger.info("ИТОГИ ВЫПОЛНЕНИЯ ETL-КОНВЕЙЕРА")
        logger.info("=" * 70)
        
        if self.pipeline_stats['start_time']:
            logger.info(f"Начало: {self.pipeline_stats['start_time']}")
        if self.pipeline_stats['end_time']:
            logger.info(f"Конец: {self.pipeline_stats['end_time']}")
        
        logger.info("\nСтатистика по этапам:")
        for stage, stats in self.pipeline_stats['stages'].items():
            status_icon = "✅" if stats.get('status') == 'success' else "❌"
            logger.info(f"  {status_icon} {stage.upper()}: {stats.get('duration', 0):.2f} сек")
            
            if 'data_shape' in stats and stats['data_shape']:
                logger.info(f"      Форма данных: {stats['data_shape']}")
        
        if self.pipeline_stats['errors']:
            logger.warning(f"\nОшибки ({len(self.pipeline_stats['errors'])}):")
            for error in self.pipeline_stats['errors']:
                logger.warning(f"  - {error['stage']}: {error['error']}")
        else:
            logger.info("\n✅ Конвейер выполнен без ошибок")
    
    def get_pipeline_stats(self):
        """Возвращает статистику выполнения."""
        return self.pipeline_stats


def create_parser():
    """
    Создает парсер аргументов командной строки.
    
    Returns:
        argparse.ArgumentParser: Настроенный парсер
    """
    parser = argparse.ArgumentParser(
        description='ETL-конвейер для анализа дефектов в программном обеспечении',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s                          # Полный ETL из .env настроек
  %(prog)s --local data/my_data.csv # Загрузка из локального файла
  %(prog)s --kaggle-dataset username/dataset # Загрузка с Kaggle
  %(prog)s --extract-only            # Только извлечение
  %(prog)s --transform-only          # Только трансформация
  %(prog)s --load-only                # Только загрузка
  %(prog)s --verbose                  # Подробный вывод
        """
    )
    
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        '--local',
        metavar='FILE',
        help='Загрузить данные из локального CSV файла'
    )
    source_group.add_argument(
        '--kaggle-dataset',
        metavar='DATASET',
        help='Загрузить датасет с Kaggle (формат: username/dataset-name)'
    )
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--extract-only',
        action='store_true',
        help='Выполнить только извлечение данных'
    )
    mode_group.add_argument(
        '--transform-only',
        action='store_true',
        help='Выполнить только трансформацию (требует наличия raw данных)'
    )
    mode_group.add_argument(
        '--load-only',
        action='store_true',
        help='Выполнить только загрузку (требует наличия processed данных)'
    )
    
    parser.add_argument(
        '--format',
        choices=['csv', 'parquet'],
        default='csv',
        help='Формат сохранения финальных данных (по умолчанию: csv)'
    )
    parser.add_argument(
        '--output',
        metavar='DIR',
        help='Директория для сохранения результатов (переопределяет .env)'
    )
    parser.add_argument(
        '--no-archive',
        action='store_true',
        help='Не архивировать предыдущие версии'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Подробный вывод с отладочной информацией'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='ETL Pipeline for Defect Prediction v1.0.0'
    )
    
    return parser


def main():
    """Главная функция запуска ETL-конвейера."""
    
    parser = create_parser()
    args = parser.parse_args()
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║     ETL Pipeline for Software Defect Prediction v1.0      ║
    ║         Анализ и подготовка данных о дефектах ПО          ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    pipeline = ETLPipeline(verbose=args.verbose)
    
    if args.extract_only:
        if args.local:
            pipeline.run_extract('local', local_file=args.local)
        elif args.kaggle_dataset:
            pipeline.run_extract('kaggle', kaggle_dataset=args.kaggle_dataset)
        else:
            pipeline.run_extract('env')
            
    elif args.transform_only:
        pipeline.run_transform()
        
    elif args.load_only:
        pipeline.run_load(format=args.format, archive_previous=not args.no_archive)
        
    else:
        source = None
        kwargs = {'format': args.format, 'archive_previous': not args.no_archive}
        
        if args.local:
            source = 'local'
            kwargs['local_file'] = args.local
        elif args.kaggle_dataset:
            source = 'kaggle'
            kwargs['kaggle_dataset'] = args.kaggle_dataset
        
        success = pipeline.run_full_pipeline(source, **kwargs)
        
        if success:
            logger.info("\n✨ ETL-конвейер успешно завершен!")
            
            logger.info("\nРезультаты:")
            logger.info(f"  📁 Сырые данные: data/raw/raw_defect_data.csv")
            logger.info(f"  📁 Очищенные данные: data/processed/")
            logger.info(f"  📁 Финальные данные: data/final/")
            logger.info(f"  📊 Отчеты: data/final/metadata_*.json")
            
            logger.info("\n🚀 Для анализа данных запустите:")
            logger.info("   jupyter lab notebooks/EDA.ipynb")
            
            sys.exit(0)
        else:
            logger.error("\n❌ ETL-конвейер завершен с ошибками")
            sys.exit(1)


if __name__ == "__main__":
    main()
