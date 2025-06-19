"""
Загрузчик данных из PostgreSQL
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
from datetime import datetime
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, text
import pickle
from pathlib import Path
import hashlib
from tqdm import tqdm
from contextlib import contextmanager
import os

from utils.logger import get_logger

class CryptoDataLoader:
    """Загрузчик исторических данных криптовалютных фьючерсов"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger("DataLoader")
        self.cache_dir = Path(config.get('performance', {}).get('cache_dir', 'cache'))
        self.cache_dir.mkdir(exist_ok=True)
        self.engine = self._create_engine()
        
    def _create_engine(self):
        """Создание SQLAlchemy engine с поддержкой переменных окружения"""
        db_config = self.config['database'].copy()
        
        # Обработка переменных окружения
        for key, value in db_config.items():
            if isinstance(value, str) and value.startswith('${'):
                # Извлечение имени переменной и значения по умолчанию
                var_name = value.split(':')[0][2:]
                default_value = value.split(':')[1].rstrip('}')
                db_config[key] = os.getenv(var_name, default_value)
        
        # Преобразование порта в int
        db_config['port'] = int(db_config['port'])
        
        # Создание connection string
        connection_string = (
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        
        self.logger.info(f"Подключение к БД {db_config['database']} на {db_config['host']}:{db_config['port']}...")
        
        # Создание engine с дополнительными параметрами для стабильности
        engine = create_engine(
            connection_string,
            pool_size=db_config.get('pool_size', 10),
            max_overflow=db_config.get('max_overflow', 20),
            pool_pre_ping=True,  # Проверка соединения перед использованием
            echo=False  # Отключаем SQL логирование
        )
        
        # Тест подключения
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            self.logger.info("✅ Подключение к БД успешно установлено")
        except Exception as e:
            self.logger.error(f"❌ Ошибка подключения к БД: {e}")
            raise
            
        return engine
    
    def _get_cache_key(self, symbols: List[str], start_date: str, end_date: str) -> str:
        """Генерация ключа кэша"""
        key_string = f"{','.join(sorted(symbols))}_{start_date}_{end_date}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Загрузка данных из кэша"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists() and self.config.get('performance', {}).get('cache_features', True):
            self.logger.info(f"Загрузка данных из кэша: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки кэша: {e}")
                return None
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_key: str):
        """Сохранение данных в кэш"""
        if self.config.get('performance', {}).get('cache_features', True):
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            self.logger.info(f"Сохранение данных в кэш: {cache_file}")
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                self.logger.warning(f"Ошибка сохранения кэша: {e}")
    
    def load_data(self, 
                  symbols: Optional[List[str]] = None,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """Загрузка данных из БД с валидацией"""
        symbols = symbols or self.config['data']['symbols']
        start_date = start_date or self.config['data']['start_date']
        end_date = end_date or self.config['data']['end_date']
        
        # Валидация параметров
        self._validate_symbols(symbols)
        self._validate_dates(start_date, end_date)
        
        cache_key = self._get_cache_key(symbols, start_date, end_date)
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        self.logger.start_stage("data_loading", symbols_count=len(symbols))
        
        try:
            # SQL запрос для pandas (используем %s синтаксис)
            query = """
            SELECT 
                id,
                symbol,
                timestamp,
                datetime,
                open,
                high,
                low,
                close,
                volume,
                turnover
            FROM raw_market_data
            WHERE 
                symbol = ANY(%(symbols)s)
                AND datetime >= %(start_date)s
                AND datetime <= %(end_date)s
                AND market_type = 'futures'
                AND interval_minutes = 15
            ORDER BY symbol, datetime
            """
            
            chunk_size = 100000
            chunks = []
            
            # Используем прямое подключение psycopg2 для загрузки данных
            db_config = self.config['database'].copy()
            # Обработка переменных окружения
            for key, value in db_config.items():
                if isinstance(value, str) and value.startswith('${'):
                    var_name = value.split(':')[0][2:]
                    default_value = value.split(':')[1].rstrip('}')
                    db_config[key] = os.getenv(var_name, default_value)
            db_config['port'] = int(db_config['port'])
            
            # Прямое подключение через psycopg2
            conn = psycopg2.connect(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password']
            )
            
            try:
                # Подсчет общего количества записей
                count_query = """
                SELECT COUNT(*) 
                FROM raw_market_data 
                WHERE 
                    symbol = ANY(%s)
                    AND datetime >= %s
                    AND datetime <= %s
                    AND market_type = 'futures'
                    AND interval_minutes = 15
                """
                
                cursor = conn.cursor()
                cursor.execute(count_query, (symbols, start_date, end_date))
                total_records = cursor.fetchone()[0]
                cursor.close()
                
                self.logger.info(f"Загрузка {total_records:,} записей...")
                
                # Загрузка данных через pandas
                with tqdm(total=total_records, desc="Загрузка данных") as pbar:
                    for chunk in pd.read_sql(
                        query,
                        conn,
                        params={
                            "symbols": symbols,
                            "start_date": start_date,
                            "end_date": end_date
                        },
                        chunksize=chunk_size
                    ):
                        chunks.append(chunk)
                        pbar.update(len(chunk))
            finally:
                conn.close()
            
            df = pd.concat(chunks, ignore_index=True)
            
            # Обработка типов данных
            df['datetime'] = pd.to_datetime(df['datetime'])
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
            df[numeric_columns] = df[numeric_columns].astype(np.float32)
            
            self._log_data_statistics(df)
            self._save_to_cache(df, cache_key)
            
            self.logger.end_stage("data_loading", records=len(df))
            
            return df
            
        except Exception as e:
            self.logger.log_error(e, "load_data")
            raise
    
    def _log_data_statistics(self, df: pd.DataFrame):
        """Логирование статистики по загруженным данным"""
        self.logger.info("📊 Статистика загруженных данных:")
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            date_range = (
                symbol_data['datetime'].min().strftime('%Y-%m-%d'),
                symbol_data['datetime'].max().strftime('%Y-%m-%d')
            )
            
            self.logger.log_data_info(
                symbol=symbol,
                records=len(symbol_data),
                date_range=date_range
            )
    
    def load_symbol_data(self, symbol: str, 
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """Загрузка данных для одного символа"""
        return self.load_data([symbol], start_date, end_date)
    
    def get_available_symbols(self) -> List[str]:
        """Получение списка доступных символов в БД"""
        query = """
        SELECT DISTINCT symbol 
        FROM raw_market_data 
        WHERE market_type = 'futures'
        ORDER BY symbol
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            symbols = [row[0] for row in result]
        
        self.logger.info(f"Найдено {len(symbols)} доступных символов")
        return symbols
    
    def get_date_range(self, symbol: Optional[str] = None) -> Tuple[datetime, datetime]:
        """Получение диапазона дат для символа или всех данных"""
        if symbol:
            query = """
            SELECT MIN(datetime), MAX(datetime)
            FROM raw_market_data
            WHERE symbol = :symbol AND market_type = 'futures'
            """
            params = {"symbol": symbol}
        else:
            query = """
            SELECT MIN(datetime), MAX(datetime)
            FROM raw_market_data
            WHERE market_type = 'futures'
            """
            params = {}
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params).fetchone()
            
        return result[0], result[1]
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Проверка качества данных"""
        self.logger.start_stage("data_validation")
        
        quality_report = {}
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('datetime')
            
            report = {
                'total_records': len(symbol_data),
                'missing_values': symbol_data.isnull().sum().to_dict(),
                'duplicates': symbol_data.duplicated(subset=['datetime']).sum(),
                'gaps': 0,
                'anomalies': {}
            }
            
            # Проверка пропусков во времени
            expected_freq = pd.Timedelta(minutes=15)
            time_diff = symbol_data['datetime'].diff()
            gaps = time_diff[time_diff > expected_freq * 1.5]
            report['gaps'] = len(gaps)
            
            if len(gaps) > 0:
                self.logger.warning(
                    f"Обнаружено {len(gaps)} пропусков в данных {symbol}"
                )
            
            # Проверка нулевого объема
            zero_volume = (symbol_data['volume'] == 0).sum()
            if zero_volume > 0:
                report['anomalies']['zero_volume'] = zero_volume
            
            # Проверка отсутствия движения цены
            no_movement = (
                (symbol_data['open'] == symbol_data['high']) & 
                (symbol_data['high'] == symbol_data['low']) & 
                (symbol_data['low'] == symbol_data['close'])
            ).sum()
            if no_movement > 0:
                report['anomalies']['no_price_movement'] = no_movement
            
            # Проверка экстремальных изменений цены
            price_change = symbol_data['close'].pct_change()
            extreme_changes = (price_change.abs() > 0.2).sum()
            if extreme_changes > 0:
                report['anomalies']['extreme_price_changes'] = extreme_changes
            
            quality_report[symbol] = report
        
        self.logger.end_stage("data_validation", issues_found=sum(
            len(r['anomalies']) for r in quality_report.values()
        ))
        
        return quality_report
    
    def resample_data(self, df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
        """Изменение временного интервала данных"""
        self.logger.info(f"Ресемплирование данных к интервалу: {target_interval}")
        
        resampled_dfs = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data.set_index('datetime', inplace=True)
            
            agg_rules = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'turnover': 'sum'
            }
            
            resampled = symbol_data.resample(target_interval).agg(agg_rules)
            resampled['symbol'] = symbol
            resampled.reset_index(inplace=True)
            
            resampled_dfs.append(resampled)
        
        return pd.concat(resampled_dfs, ignore_index=True)
    
    def _validate_symbols(self, symbols: List[str]):
        """Валидация символов"""
        if not symbols:
            raise ValueError("Список символов не может быть пустым")
        
        for symbol in symbols:
            if not isinstance(symbol, str) or not symbol.replace('USDT', '').replace('1000', '').isalnum():
                raise ValueError(f"Некорректный символ: {symbol}")
    
    def _validate_dates(self, start_date: str, end_date: str):
        """Валидация дат"""
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            if start >= end:
                raise ValueError("Начальная дата должна быть раньше конечной")
        except Exception as e:
            raise ValueError(f"Некорректный формат даты: {e}")
    
    def get_data_completeness(self, symbols: List[str], start_date: str, end_date: str) -> Dict:
        """Проверка полноты данных"""
        query = """
        SELECT 
            symbol,
            COUNT(*) as actual_records,
            MIN(datetime) as first_record,
            MAX(datetime) as last_record
        FROM raw_market_data 
        WHERE symbol = ANY(:symbols)
        AND datetime BETWEEN :start_date AND :end_date
        AND market_type = 'futures'
        GROUP BY symbol
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {
                'symbols': symbols,
                'start_date': start_date,
                'end_date': end_date
            })
            
            completeness = {}
            for row in result:
                symbol, actual, first, last = row
                
                # Вычисляем ожидаемое количество записей (15-минутные интервалы)
                time_diff = (last - first).total_seconds()
                expected_records = int(time_diff / (15 * 60)) + 1
                
                completeness[symbol] = {
                    'actual_records': actual,
                    'expected_records': expected_records,
                    'completeness_pct': (actual / expected_records) * 100 if expected_records > 0 else 0,
                    'first_record': first,
                    'last_record': last
                }
        
        return completeness