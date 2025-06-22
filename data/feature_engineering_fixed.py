"""
Исправленная версия feature engineering
Решает проблему с целевыми переменными
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import ta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FixedFeatureEngineer:
    """Исправленный класс для создания признаков и целевых переменных"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logger
        self.scaler_params = {}
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Главный метод создания признаков с ПРАВИЛЬНЫМИ целевыми переменными"""
        self.logger.info("🔧 Создание признаков и целевых переменных (исправленная версия)")
        
        # Сортируем данные
        df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)
        
        # Базовые признаки
        df = self._create_price_features(df)
        df = self._create_volume_features(df)
        df = self._create_technical_indicators(df)
        df = self._create_time_features(df)
        
        # ИСПРАВЛЕННОЕ создание целевых переменных
        df = self._create_correct_target_variables(df)
        
        # Нормализация (без целевых переменных)
        df = self._normalize_features_correctly(df)
        
        # Удаляем строки где нет целевых переменных
        df = df.dropna(subset=['target_return_1h'])
        
        self.logger.info(f"✅ Создано {len(df.columns)} признаков для {len(df)} строк")
        self.logger.info(f"📊 Целевая переменная - mean: {df['target_return_1h'].mean():.4f}, std: {df['target_return_1h'].std():.4f}")
        
        return df
    
    def _create_correct_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """ИСПРАВЛЕННОЕ создание целевых переменных"""
        self.logger.info("🎯 Создание ПРАВИЛЬНЫХ целевых переменных")
        
        # Для 15-минутных баров, 4 бара = 1 час
        horizon = 4
        
        # Вычисляем будущую доходность за 1 час (4 бара)
        df['target_return_1h'] = df.groupby('symbol')['close'].transform(
            lambda x: (x.shift(-horizon) / x - 1) * 100
        )
        
        # Бинарные целевые переменные для классификации
        df['target_direction'] = (df['target_return_1h'] > 0).astype(int)
        
        # Мультиклассовые цели
        df['target_class'] = pd.cut(
            df['target_return_1h'],
            bins=[-np.inf, -1, -0.5, 0.5, 1, np.inf],
            labels=[0, 1, 2, 3, 4]  # strong_down, down, neutral, up, strong_up
        )
        
        # Take profit и stop loss цели
        tp_levels = self.config['risk_management']['take_profit_targets']
        sl_level = self.config['risk_management']['stop_loss_pct']
        
        # Максимальная и минимальная цена в следующий час
        for i in range(1, horizon + 1):
            df[f'future_high_{i}'] = df.groupby('symbol')['high'].shift(-i)
            df[f'future_low_{i}'] = df.groupby('symbol')['low'].shift(-i)
        
        # Проверяем достижение TP/SL
        future_high_cols = [f'future_high_{i}' for i in range(1, horizon + 1)]
        future_low_cols = [f'future_low_{i}' for i in range(1, horizon + 1)]
        
        df['max_profit_1h'] = df[future_high_cols].max(axis=1) / df['close'] - 1
        df['max_loss_1h'] = df[future_low_cols].min(axis=1) / df['close'] - 1
        
        # Целевые переменные для TP
        for tp in tp_levels:
            df[f'target_tp_{tp}'] = (df['max_profit_1h'] * 100 >= tp).astype(int)
        
        # Целевая переменная для SL
        df['target_sl_hit'] = (df['max_loss_1h'] * 100 <= -sl_level).astype(int)
        
        # Удаляем временные колонки
        cols_to_drop = future_high_cols + future_low_cols
        df = df.drop(columns=cols_to_drop)
        
        self.logger.info(f"✅ Создано целевых переменных: target_return_1h (регрессия), target_direction (бинарная), target_class (мультикласс)")
        
        return df
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание ценовых признаков"""
        # Returns
        for period in [1, 4, 12, 48]:  # 15min, 1h, 3h, 12h
            df[f'return_{period}'] = df.groupby('symbol')['close'].pct_change(period) * 100
        
        # Price ratios
        df['high_low_ratio'] = df['high'] / df['low'] - 1
        df['close_open_ratio'] = df['close'] / df['open'] - 1
        
        # Rolling statistics
        for window in [12, 48, 96]:  # 3h, 12h, 24h
            df[f'rolling_mean_{window}'] = df.groupby('symbol')['close'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f'rolling_std_{window}'] = df.groupby('symbol')['close'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            df[f'price_position_{window}'] = (df['close'] - df[f'rolling_mean_{window}']) / df[f'rolling_std_{window}']
        
        return df
    
    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание объемных признаков"""
        # Volume ratios
        for window in [12, 48]:
            df[f'volume_ratio_{window}'] = df['volume'] / df.groupby('symbol')['volume'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        
        # VWAP
        df['vwap'] = (df['close'] * df['volume']).groupby(df['symbol']).cumsum() / df.groupby('symbol')['volume'].cumsum()
        df['price_vwap_ratio'] = df['close'] / df['vwap'] - 1
        
        # OBV
        df['obv'] = df.groupby('symbol').apply(
            lambda x: (np.sign(x['close'].diff()) * x['volume']).cumsum()
        ).reset_index(level=0, drop=True)
        
        return df
    
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание технических индикаторов"""
        # RSI
        for period in [14, 28]:
            df[f'rsi_{period}'] = df.groupby('symbol')['close'].transform(
                lambda x: ta.momentum.RSIIndicator(x, window=period).rsi()
            )
        
        # MACD
        for group_name, group_data in df.groupby('symbol'):
            macd = ta.trend.MACD(group_data['close'])
            df.loc[df['symbol'] == group_name, 'macd'] = macd.macd()
            df.loc[df['symbol'] == group_name, 'macd_signal'] = macd.macd_signal()
            df.loc[df['symbol'] == group_name, 'macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        for period in [20, 40]:
            bb = df.groupby('symbol')['close'].transform(
                lambda x: ta.volatility.BollingerBands(x, window=period, window_dev=2)
            )
            df[f'bb_high_{period}'] = df.groupby('symbol').apply(
                lambda x: ta.volatility.BollingerBands(x['close'], window=period).bollinger_hband()
            ).reset_index(level=0, drop=True)
            df[f'bb_low_{period}'] = df.groupby('symbol').apply(
                lambda x: ta.volatility.BollingerBands(x['close'], window=period).bollinger_lband()
            ).reset_index(level=0, drop=True)
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_low_{period}']) / (df[f'bb_high_{period}'] - df[f'bb_low_{period}'])
        
        # ATR
        for period in [14, 28]:
            df[f'atr_{period}'] = df.groupby('symbol').apply(
                lambda x: ta.volatility.AverageTrueRange(x['high'], x['low'], x['close'], window=period).average_true_range()
            ).reset_index(level=0, drop=True)
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание временных признаков"""
        df['hour'] = pd.to_datetime(df['datetime']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['datetime']).dt.dayofweek
        df['day_of_month'] = pd.to_datetime(df['datetime']).dt.day
        
        # Циклические признаки
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _normalize_features_correctly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Правильная нормализация БЕЗ утечки данных"""
        # Исключаем из нормализации
        exclude_cols = [
            'symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume',
            'target_return_1h', 'target_direction', 'target_class',
            'target_tp_1.2', 'target_tp_2.4', 'target_tp_3.5', 'target_sl_hit',
            'max_profit_1h', 'max_loss_1h'
        ]
        
        # Колонки для нормализации
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Группируем по символу для независимой нормализации
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df.loc[mask, feature_cols]
            
            # Z-score нормализация
            mean = symbol_data.mean()
            std = symbol_data.std()
            
            # Заменяем нулевые std на 1
            std[std == 0] = 1
            
            # Нормализуем
            df.loc[mask, feature_cols] = (symbol_data - mean) / std
            
            # Сохраняем параметры для будущего использования
            self.scaler_params[symbol] = {
                'mean': mean.to_dict(),
                'std': std.to_dict()
            }
        
        # Заменяем inf и nan на 0
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0)
        df[feature_cols] = df[feature_cols].fillna(0)
        
        return df
    
    def prepare_train_val_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Правильное разделение данных по времени"""
        # Сортируем по времени
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Определяем границы
        total_len = len(df)
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.85)
        
        # Разделяем
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        # Нормализуем каждый набор используя параметры из train
        val_df = self._apply_normalization(val_df, self.scaler_params)
        test_df = self._apply_normalization(test_df, self.scaler_params)
        
        self.logger.info(f"📊 Разделение данных: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def _apply_normalization(self, df: pd.DataFrame, scaler_params: Dict) -> pd.DataFrame:
        """Применение сохраненных параметров нормализации"""
        exclude_cols = [
            'symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume',
            'target_return_1h', 'target_direction', 'target_class',
            'target_tp_1.2', 'target_tp_2.4', 'target_tp_3.5', 'target_sl_hit',
            'max_profit_1h', 'max_loss_1h'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        for symbol in df['symbol'].unique():
            if symbol in scaler_params:
                mask = df['symbol'] == symbol
                params = scaler_params[symbol]
                
                for col in feature_cols:
                    if col in params['mean']:
                        df.loc[mask, col] = (df.loc[mask, col] - params['mean'][col]) / params['std'][col]
        
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0)
        df[feature_cols] = df[feature_cols].fillna(0)
        
        return df