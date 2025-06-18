"""
Загрузчик данных из PostgreSQL
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
from datetime import datetime
import psycopg2
from sqlalchemy import create_engine
import pickle
from pathlib import Path
import hashlib
from tqdm import tqdm

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
        """Создание SQLAlchemy engine"""
        db_config = self.config['database']
        connection_string = (
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        
        self.logger.info("Подключение к базе данных...")
        return create_engine(connection_string, pool_size=10, max_overflow=20)
    
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
        """Загрузка данных из БД"""
        symbols = symbols or self.config['data']['symbols']
        start_date = start_date or self.config['data']['start_date']
        end_date = end_date or self.config['data']['end_date']
        
        cache_key = self._get_cache_key(symbols, start_date, end_date)
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        self.logger.start_stage("data_loading", symbols_count=len(symbols))
        
        try:
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
            
            with self.engine.connect() as conn:
                # Подсчет общего количества записей
                count_query = """
                SELECT COUNT(*) 
                FROM raw_market_data 
                WHERE 
                    symbol = ANY(%(symbols)s)
                    AND datetime >= %(start_date)s
                    AND datetime <= %(end_date)s
                    AND market_type = 'futures'
                    AND interval_minutes = 15
                """
                
                total_records = conn.execute(
                    count_query,
                    {"symbols": symbols, "start_date": start_date, "end_date": end_date}
                ).scalar()
                
                self.logger.info(f"Загрузка {total_records:,} записей...")
                
                # Загрузка данных по частям
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
            result = conn.execute(query)
            symbols = [row[0] for row in result]
        
        self.logger.info(f"Найдено {len(symbols)} доступных символов")
        return symbols
    
    def get_date_range(self, symbol: Optional[str] = None) -> Tuple[datetime, datetime]:
        """Получение диапазона дат для символа или всех данных"""
        if symbol:
            query = """
            SELECT MIN(datetime), MAX(datetime)
            FROM raw_market_data
            WHERE symbol = %(symbol)s AND market_type = 'futures'
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
            result = conn.execute(query, params).fetchone()
            
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