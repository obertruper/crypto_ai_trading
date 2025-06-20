"""
Инженерия признаков для криптовалютных данных
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger

class FeatureEngineer:
    """Создание признаков для модели прогнозирования"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger("FeatureEngineer")
        self.feature_config = config['features']
        self.scalers = {}
    
    @staticmethod
    def safe_divide(numerator: pd.Series, denominator: pd.Series, fill_value=0.0) -> pd.Series:
        """Безопасное деление с обработкой нулей и малых значений"""
        # Минимальное значение для знаменателя
        min_denominator = 1e-10
        
        # Создаем безопасный знаменатель
        safe_denominator = denominator.copy()
        
        # Заменяем нули и очень маленькие значения
        mask_small = (safe_denominator.abs() < min_denominator)
        safe_denominator[mask_small] = np.sign(safe_denominator[mask_small]) * min_denominator
        safe_denominator[safe_denominator == 0] = min_denominator  # Для точных нулей
        
        # Выполняем деление
        result = numerator / safe_denominator
        
        # Обрабатываем inf и nan
        # Если fill_value - это Series, используем другой подход
        if isinstance(fill_value, pd.Series):
            # Находим позиции с inf и заменяем их соответствующими значениями из fill_value
            inf_mask = np.isinf(result)
            result.loc[inf_mask] = fill_value.loc[inf_mask]
        else:
            # Если fill_value - скаляр, используем стандартный replace
            result = result.replace([np.inf, -np.inf], fill_value)
        
        result = result.fillna(fill_value)
        
        return result
        
    def create_features(self, df: pd.DataFrame, train_end_date: Optional[str] = None) -> pd.DataFrame:
        """Создание всех признаков для датасета с walk-forward валидацией"""
        self.logger.start_stage("feature_engineering", 
                               symbols=df['symbol'].nunique())
        
        # Валидация данных
        self._validate_data(df)
        
        featured_dfs = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('datetime')
            
            symbol_data = self._create_basic_features(symbol_data)
            symbol_data = self._create_technical_indicators(symbol_data)
            symbol_data = self._create_microstructure_features(symbol_data)
            symbol_data = self._create_temporal_features(symbol_data)
            symbol_data = self._create_target_variables(symbol_data)
            
            featured_dfs.append(symbol_data)
        
        result_df = pd.concat(featured_dfs, ignore_index=True)
        result_df = self._create_cross_asset_features(result_df)
        
        # Обработка NaN значений
        result_df = self._handle_missing_values(result_df)
        
        # Walk-forward нормализация если указана дата
        if train_end_date:
            result_df = self._normalize_walk_forward(result_df, train_end_date)
        else:
            result_df = self._normalize_features(result_df)
        
        self._log_feature_statistics(result_df)
        
        self.logger.end_stage("feature_engineering", 
                            total_features=len(result_df.columns))
        
        return result_df
    
    def _validate_data(self, df: pd.DataFrame):
        """Валидация целостности данных"""
        # Проверка на отсутствующие значения
        if df.isnull().any().any():
            self.logger.warning("Обнаружены пропущенные значения в данных")
            
        # Проверка на аномальные цены
        price_changes = df.groupby('symbol')['close'].pct_change()
        extreme_moves = abs(price_changes) > 0.15  # >15% за 15 минут
        
        if extreme_moves.sum() > 0:
            self.logger.warning(f"Обнаружено {extreme_moves.sum()} экстремальных движений цены")
            
        # Проверка временных гэпов (только значительные разрывы > 2 часов)
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            time_diff = symbol_data['datetime'].diff()
            expected_diff = pd.Timedelta('15 minutes')
            # Считаем большими только разрывы больше 2 часов (8 интервалов)
            large_gaps = time_diff > expected_diff * 8
            
            if large_gaps.sum() > 0:
                self.logger.warning(f"Символ {symbol}: обнаружено {large_gaps.sum()} значительных временных разрывов (> 2 часов)")
    
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Базовые признаки из OHLCV данных без look-ahead bias"""
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Доходности за разные периоды
        for period in [5, 10, 20]:
            df[f'returns_{period}'] = np.log(
                df['close'] / df['close'].shift(period)
            )
        
        # Ценовые соотношения
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Позиция закрытия в диапазоне
        df['close_position'] = (
            (df['close'] - df['low']) / 
            (df['high'] - df['low'] + 1e-10)
        )
        
        # Объемные соотношения с использованием только исторических данных
        df['volume_ratio'] = self.safe_divide(
            df['volume'], 
            df['volume'].rolling(20, min_periods=20).mean(),
            fill_value=1.0
        )
        df['turnover_ratio'] = self.safe_divide(
            df['turnover'], 
            df['turnover'].rolling(20, min_periods=20).mean(),
            fill_value=1.0
        )
        
        # VWAP
        df['vwap'] = self.safe_divide(df['turnover'], df['volume'], fill_value=df['close'])
        df['close_vwap_ratio'] = self.safe_divide(df['close'], df['vwap'], fill_value=1.0)
        
        return df
    
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Технические индикаторы"""
        tech_config = self.feature_config['technical']
        
        # SMA
        sma_config = next((c for c in tech_config if c['name'] == 'sma'), None)
        if sma_config:
            for period in sma_config['periods']:
                df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], period)
                df[f'close_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
        
        # EMA
        ema_config = next((c for c in tech_config if c['name'] == 'ema'), None)
        if ema_config:
            for period in ema_config['periods']:
                df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], period)
                df[f'close_ema_{period}_ratio'] = df['close'] / df[f'ema_{period}']
        
        # RSI
        rsi_config = next((c for c in tech_config if c['name'] == 'rsi'), None)
        if rsi_config:
            df['rsi'] = ta.momentum.RSIIndicator(
                df['close'], 
                window=rsi_config['period']
            ).rsi()
            
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # MACD
        macd_config = next((c for c in tech_config if c['name'] == 'macd'), None)
        if macd_config:
            macd = ta.trend.MACD(
                df['close'],
                window_slow=macd_config['slow'],
                window_fast=macd_config['fast'],
                window_sign=macd_config['signal']
            )
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb_config = next((c for c in tech_config if c['name'] == 'bollinger_bands'), None)
        if bb_config:
            bb = ta.volatility.BollingerBands(
                df['close'],
                window=bb_config['period'],
                window_dev=bb_config['std_dev']
            )
            df['bb_high'] = bb.bollinger_hband()
            df['bb_low'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_width'] = df['bb_high'] - df['bb_low']
            df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_width'] + 1e-10)
        
        # ATR
        atr_config = next((c for c in tech_config if c['name'] == 'atr'), None)
        if atr_config:
            df['atr'] = ta.volatility.AverageTrueRange(
                df['high'], 
                df['low'], 
                df['close'],
                window=atr_config['period']
            ).average_true_range()
            
            df['atr_pct'] = df['atr'] / df['close']
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(
            df['high'], 
            df['low'], 
            df['close'],
            window=14,
            smooth_window=3
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ADX
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        # Parabolic SAR
        psar = ta.trend.PSARIndicator(df['high'], df['low'], df['close'])
        df['psar'] = psar.psar()
        # Вместо отдельных psar_up и psar_down, создаем индикатор направления
        df['psar_trend'] = (df['close'] > df['psar']).astype(float)
        df['psar_distance'] = (df['close'] - df['psar']) / df['close']
        
        return df
    
    def _create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Признаки микроструктуры рынка"""
        # Спред high-low
        df['hl_spread'] = self.safe_divide(df['high'] - df['low'], df['close'], fill_value=0.0)
        df['hl_spread_ma'] = df['hl_spread'].rolling(20).mean()
        
        # Направление цены и объем
        df['price_direction'] = np.sign(df['close'] - df['open'])
        df['directed_volume'] = df['volume'] * df['price_direction']
        df['volume_imbalance'] = df['directed_volume'].rolling(10).sum() / \
                                 df['volume'].rolling(10).sum()
        
        # Ценовое воздействие
        df['price_impact'] = df['returns'].abs() / (np.log(df['volume'] + 1) + 1e-10)
        df['toxicity'] = 1 / (1 + df['price_impact'])
        
        # Амихуд неликвидность
        df['amihud_illiquidity'] = df['returns'].abs() / (df['turnover'] + 1e-10)
        df['amihud_ma'] = df['amihud_illiquidity'].rolling(20).mean()
        
        # Кайл лямбда
        df['kyle_lambda'] = df['returns'].rolling(10).std() / \
                           (df['volume'].rolling(10).std() + 1e-10)
        
        # Реализованная волатильность
        df['realized_vol'] = df['returns'].rolling(20).std() * np.sqrt(96)  # 96 = 24*4 (15мин интервалы)
        
        # Соотношение объема к волатильности
        df['volume_volatility_ratio'] = df['volume'] / (df['realized_vol'] + 1e-10)
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Временные признаки"""
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        
        # Циклическое кодирование времени
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        df['day'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Торговые сессии
        df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['european_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
        df['american_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
        
        # Пересечение сессий
        df['session_overlap'] = (
            (df['asian_session'] + df['european_session'] + df['american_session']) > 1
        ).astype(int)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка пропущенных значений"""
        self.logger.info("Обработка пропущенных значений...")
        
        # Сохраняем информационные колонки
        info_cols = ['id', 'symbol', 'timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        
        # Группируем по символам для правильной обработки
        processed_dfs = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            
            # Для каждой колонки применяем соответствующий метод заполнения
            for col in symbol_data.columns:
                if col in info_cols:
                    continue
                    
                if symbol_data[col].isna().any():
                    # Для технических индикаторов используем forward fill
                    if any(indicator in col for indicator in ['sma', 'ema', 'rsi', 'macd', 'bb_', 'adx']):
                        symbol_data[col] = symbol_data[col].ffill()
                    # Для остальных используем 0
                    else:
                        symbol_data[col] = symbol_data[col].fillna(0)
            
            # Удаляем первые строки где могут быть NaN из-за расчета индикаторов
            # Находим максимальный период среди всех индикаторов
            max_period = 50  # SMA50 требует минимум 50 периодов
            symbol_data = symbol_data.iloc[max_period:].copy()
            
            processed_dfs.append(symbol_data)
        
        result_df = pd.concat(processed_dfs, ignore_index=True)
        
        # Финальная проверка
        nan_count = result_df.isna().sum().sum()
        if nan_count > 0:
            self.logger.warning(f"Остались {nan_count} NaN значений после обработки")
            # Принудительно заполняем оставшиеся NaN
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns
            result_df[numeric_cols] = result_df[numeric_cols].fillna(0)
        
        # Проверка на бесконечные значения
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(result_df[numeric_cols]).sum().sum()
        if inf_count > 0:
            self.logger.warning(f"Обнаружены {inf_count} бесконечных значений, заменяем на конечные")
            result_df[numeric_cols] = result_df[numeric_cols].replace([np.inf, -np.inf], [1e10, -1e10])
        
        self.logger.info(f"Обработка завершена. Итоговый размер: {len(result_df)} записей")
        return result_df
    
    def _create_cross_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Кросс-активные признаки"""
        self.logger.info("Создание кросс-активных признаков...")
        
        # BTC как базовый актив
        btc_data = df[df['symbol'] == 'BTCUSDT'][['datetime', 'close', 'returns']].copy()
        if len(btc_data) > 0:
            btc_data.rename(columns={
                'close': 'btc_close',
                'returns': 'btc_returns'
            }, inplace=True)
            
            df = df.merge(btc_data, on='datetime', how='left')
            
            # Корреляция с BTC
            for symbol in df['symbol'].unique():
                if symbol != 'BTCUSDT':
                    mask = df['symbol'] == symbol
                    df.loc[mask, 'btc_correlation'] = (
                        df.loc[mask, 'returns']
                        .rolling(window=96)
                        .corr(df.loc[mask, 'btc_returns'])
                    )
            
            df.loc[df['symbol'] == 'BTCUSDT', 'btc_correlation'] = 1.0
            
            # Относительная сила к BTC
            df['relative_strength_btc'] = df['close'] / df['btc_close']
            df['rs_btc_ma'] = df.groupby('symbol')['relative_strength_btc'].transform(
                lambda x: x.rolling(20).mean()
            )
        else:
            # Заполняем нулями если нет данных BTC
            df['btc_close'] = 0
            df['btc_returns'] = 0
            df['btc_correlation'] = 0
            df['relative_strength_btc'] = 0
            df['rs_btc_ma'] = 0
        
        # Определяем сектора
        defi_tokens = ['AAVEUSDT', 'UNIUSDT', 'CAKEUSDT', 'DYDXUSDT']
        layer1_tokens = ['ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'DOTUSDT', 'NEARUSDT']
        meme_tokens = ['DOGEUSDT', 'FARTCOINUSDT', 'MELANIAUSDT', 'TRUMPUSDT', 
                      'POPCATUSDT', 'PNUTUSDT', 'ZEREBROUSDT', 'WIFUSDT']
        
        df['sector'] = 'other'
        df.loc[df['symbol'].isin(defi_tokens), 'sector'] = 'defi'
        df.loc[df['symbol'].isin(layer1_tokens), 'sector'] = 'layer1'
        df.loc[df['symbol'].isin(meme_tokens), 'sector'] = 'meme'
        df.loc[df['symbol'] == 'BTCUSDT', 'sector'] = 'btc'
        
        # Секторные доходности
        df['sector_returns'] = df.groupby(['datetime', 'sector'])['returns'].transform('mean')
        
        # Относительная доходность к сектору
        df['relative_to_sector'] = df['returns'] - df['sector_returns']
        
        # Ранк доходности
        df['returns_rank'] = df.groupby('datetime')['returns'].rank(pct=True)
        
        # 24-часовой моментум
        df['momentum_24h'] = df.groupby('symbol')['returns'].transform(
            lambda x: x.rolling(96).sum()  # 96 = 24*4 (15мин интервалы)
        )
        df['is_momentum_leader'] = (
            df.groupby('datetime')['momentum_24h']
            .rank(ascending=False) <= 5
        ).astype(int)
        
        return df
    
    def _create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание целевых переменных для обучения"""
        risk_config = self.config['risk_management']
        
        # Будущие цены
        for horizon in range(1, 5):
            df[f'future_close_{horizon}'] = df.groupby('symbol')['close'].shift(-horizon)
            df[f'future_return_{horizon}'] = (
                df[f'future_close_{horizon}'] / df['close'] - 1
            ) * 100
        
        # Цели по take profit
        for tp_level in risk_config['take_profit_targets']:
            df[f'target_tp_{tp_level}'] = 0
            
            for horizon in range(1, 5):
                future_return_col = f'future_return_{horizon}'
                if future_return_col in df.columns:
                    df[f'target_tp_{tp_level}'] = np.maximum(
                        df[f'target_tp_{tp_level}'],
                        (df[future_return_col] >= tp_level).astype(int)
                    )
        
        # Stop loss
        sl_level = risk_config['stop_loss_pct']
        df['target_sl_hit'] = 0
        
        for horizon in range(1, 5):
            future_return_col = f'future_return_{horizon}'
            if future_return_col in df.columns:
                df['target_sl_hit'] = np.maximum(
                    df['target_sl_hit'],
                    (df[future_return_col] <= -sl_level).astype(int)
                )
        
        # Оптимальное действие
        df['optimal_action'] = 0
        
        for i, tp_level in enumerate(risk_config['take_profit_targets'], 1):
            condition = (df[f'target_tp_{tp_level}'] == 1) & (df['target_sl_hit'] == 0)
            df.loc[condition, 'optimal_action'] = i
        
        # Минимальная и максимальная будущая доходность
        future_return_cols = [f'future_return_{i}' for i in range(1, 5) 
                             if f'future_return_{i}' in df.columns]
        
        if future_return_cols:
            df['future_min_return'] = df[future_return_cols].min(axis=1)
            df['future_max_return'] = df[future_return_cols].max(axis=1)
        
        return df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ИСПРАВЛЕННАЯ нормализация признаков с обработкой inf/nan"""
        self.logger.info("Нормализация признаков...")
        
        # Столбцы для исключения из нормализации
        exclude_cols = [
            'id', 'symbol', 'timestamp', 'datetime', 'sector',
            'open', 'high', 'low', 'close', 'volume', 'turnover'
        ]
        
        # Целевые переменные
        target_cols = [col for col in df.columns if col.startswith(('target_', 'future_', 'optimal_'))]
        exclude_cols.extend(target_cols)
        
        # Временные колонки
        time_cols = ['hour', 'minute', 'dayofweek', 'day', 'month', 'is_weekend',
                    'asian_session', 'european_session', 'american_session', 'session_overlap']
        exclude_cols.extend(time_cols)
        
        # Определяем признаки для нормализации
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Нормализация по символам
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            
            if symbol not in self.scalers:
                self.scalers[symbol] = RobustScaler()
            
            # Только валидные данные (без NaN и inf)
            valid_mask = mask & df[feature_cols].notna().all(axis=1)
            
            if valid_mask.sum() > 0:
                data_to_scale = df.loc[valid_mask, feature_cols].copy()
                
                # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Тщательная очистка inf и экстремальных значений
                for col in feature_cols:
                    # 1. Заменяем inf на NaN
                    data_to_scale[col] = data_to_scale[col].replace([np.inf, -np.inf], np.nan)
                    
                    # 2. Заполняем NaN медианой
                    if data_to_scale[col].isna().any():
                        median_val = data_to_scale[col].median()
                        if pd.isna(median_val):  # Если все значения NaN
                            median_val = 0.0
                        data_to_scale[col] = data_to_scale[col].fillna(median_val)
                    
                    # 3. Более агрессивный клиппинг экстремальных значений
                    q05 = data_to_scale[col].quantile(0.05)  # Изменено с 0.01 на 0.05
                    q95 = data_to_scale[col].quantile(0.95)  # Изменено с 0.99 на 0.95
                    
                    # Проверяем валидность квантилей
                    if pd.isna(q05) or pd.isna(q95) or q05 == q95:
                        # Если квантили некорректны, используем симметричное клиппинг
                        std_val = data_to_scale[col].std()
                        mean_val = data_to_scale[col].mean()
                        if pd.notna(std_val) and std_val > 0:
                            q05 = mean_val - 3 * std_val
                            q95 = mean_val + 3 * std_val
                        else:
                            q05, q95 = -1, 1  # Дефолтные значения
                    
                    data_to_scale[col] = data_to_scale[col].clip(lower=q05, upper=q95)
                    
                    # 4. Финальная проверка на inf (на всякий случай)
                    if np.isinf(data_to_scale[col]).any():
                        self.logger.warning(f"Обнаружены inf в {col} после обработки, заменяем на 0")
                        data_to_scale[col] = data_to_scale[col].replace([np.inf, -np.inf], 0)
                
                # 5. Проверяем что данные готовы для скейлинга
                if data_to_scale.shape[0] > 0 and not data_to_scale.isna().any().any():
                    try:
                        df.loc[valid_mask, feature_cols] = self.scalers[symbol].fit_transform(data_to_scale)
                    except Exception as e:
                        self.logger.error(f"Ошибка скейлинга для {symbol}: {e}")
                        # Если скейлинг не удался, оставляем исходные данные
                        df.loc[valid_mask, feature_cols] = data_to_scale.fillna(0)
                else:
                    self.logger.warning(f"Нет валидных данных для скейлинга {symbol}")
        
        return df
    
    def _normalize_walk_forward(self, df: pd.DataFrame, train_end_date: str) -> pd.DataFrame:
        """Walk-forward нормализация без data leakage"""
        self.logger.info(f"Walk-forward нормализация до {train_end_date}...")
        
        # Столбцы для исключения из нормализации
        exclude_cols = [
            'id', 'symbol', 'timestamp', 'datetime', 'sector',
            'open', 'high', 'low', 'close', 'volume', 'turnover'
        ]
        
        # Целевые переменные
        target_cols = [col for col in df.columns if col.startswith(('target_', 'future_', 'optimal_'))]
        exclude_cols.extend(target_cols)
        
        # Временные колонки
        time_cols = ['hour', 'minute', 'dayofweek', 'day', 'month', 'is_weekend',
                    'asian_session', 'european_session', 'american_session', 'session_overlap']
        exclude_cols.extend(time_cols)
        
        # Определяем признаки для нормализации
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Маска для обучающих данных
        train_mask = df['datetime'] <= pd.to_datetime(train_end_date)
        
        # Нормализация по символам
        for symbol in df['symbol'].unique():
            symbol_mask = df['symbol'] == symbol
            train_symbol_mask = symbol_mask & train_mask
            
            if train_symbol_mask.sum() > 0:
                if symbol not in self.scalers:
                    self.scalers[symbol] = StandardScaler()
                
                # Обучаем scaler только на train данных
                train_data = df.loc[train_symbol_mask, feature_cols].dropna()
                if len(train_data) > 0:
                    self.scalers[symbol].fit(train_data)
                    
                    # Применяем ко всем данным символа
                    valid_mask = symbol_mask & df[feature_cols].notna().all(axis=1)
                    if valid_mask.sum() > 0:
                        df.loc[valid_mask, feature_cols] = self.scalers[symbol].transform(
                            df.loc[valid_mask, feature_cols]
                        )
        
        return df
    
    def _log_feature_statistics(self, df: pd.DataFrame):
        """Логирование статистики по признакам"""
        feature_counts = {
            'basic': len([col for col in df.columns if col in [
                'returns', 'high_low_ratio', 'close_open_ratio', 'volume_ratio'
            ]]),
            'technical': len([col for col in df.columns if any(
                ind in col for ind in ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr']
            )]),
            'microstructure': len([col for col in df.columns if any(
                ms in col for ms in ['spread', 'imbalance', 'toxicity', 'illiquidity']
            )]),
            'temporal': len([col for col in df.columns if any(
                t in col for t in ['hour', 'day', 'month', 'session']
            )]),
            'cross_asset': len([col for col in df.columns if any(
                ca in col for ca in ['btc_', 'sector', 'rank', 'momentum']
            )])
        }
        
        self.logger.info(f"📊 Создано признаков по категориям: {feature_counts}")
        
        # Проверка пропущенных значений
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            self.logger.warning(
                f"⚠️ Обнаружены пропущенные значения в {missing_counts[missing_counts > 0].shape[0]} признаках"
            )
    
    def get_feature_names(self, include_targets: bool = False) -> List[str]:
        """Получение списка названий признаков"""
        # TODO: Реализовать правильное хранение названий признаков
        return []
    
    def save_scalers(self, path: str):
        """Сохранение скейлеров для использования в продакшене"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.scalers, f)
        
        self.logger.info(f"Скейлеры сохранены в {path}")
    
    def load_scalers(self, path: str):
        """Загрузка сохраненных скейлеров"""
        import pickle
        with open(path, 'rb') as f:
            self.scalers = pickle.load(f)
        
        self.logger.info(f"Скейлеры загружены из {path}")