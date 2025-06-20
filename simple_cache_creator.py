#!/usr/bin/env python3
"""
Простое создание кеша размером 2.9 ГБ для демонстрации
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def create_cache_29gb():
    print("🚀 Создание кеша размером 2.9 ГБ...")
    
    # Настройки для получения ~2.9 ГБ
    n_samples = 800000  # 800K записей
    n_features = 150    # 150 признаков
    
    print(f"📊 Генерация {n_samples:,} записей с {n_features} признаками...")
    
    # Базовые временные данные
    np.random.seed(42)
    timestamps = pd.date_range(start='2020-01-01', periods=n_samples, freq='15min')
    
    # Создаем структуру данных
    data = {}
    
    # Временные метки
    data['timestamp'] = timestamps
    data['symbol'] = np.random.choice(['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'], n_samples)
    
    # Генерация OHLCV данных
    print("💰 Генерация OHLCV данных...")
    base_prices = np.random.uniform(1000, 100000, n_samples)
    data['open'] = base_prices * (1 + np.random.normal(0, 0.01, n_samples))
    data['high'] = base_prices * (1 + np.abs(np.random.normal(0, 0.02, n_samples)))
    data['low'] = base_prices * (1 - np.abs(np.random.normal(0, 0.02, n_samples)))
    data['close'] = base_prices
    data['volume'] = np.random.lognormal(15, 2, n_samples)
    
    # Генерация множества технических индикаторов
    print("📈 Создание технических индикаторов...")
    
    # Moving averages
    for period in [5, 10, 20, 50, 100, 200]:
        data[f'sma_{period}'] = np.random.uniform(0.9, 1.1, n_samples) * base_prices
        data[f'ema_{period}'] = np.random.uniform(0.9, 1.1, n_samples) * base_prices
    
    # Oscillators
    data['rsi'] = np.random.uniform(20, 80, n_samples)
    data['stoch_k'] = np.random.uniform(0, 100, n_samples)
    data['stoch_d'] = np.random.uniform(0, 100, n_samples)
    data['williams_r'] = np.random.uniform(-100, 0, n_samples)
    data['cci'] = np.random.normal(0, 50, n_samples)
    data['mfi'] = np.random.uniform(0, 100, n_samples)
    
    # MACD
    data['macd'] = np.random.normal(0, 100, n_samples)
    data['macd_signal'] = np.random.normal(0, 80, n_samples)
    data['macd_histogram'] = data['macd'] - data['macd_signal']
    
    # Bollinger Bands
    data['bb_upper'] = base_prices * 1.05
    data['bb_lower'] = base_prices * 0.95
    data['bb_middle'] = base_prices
    data['bb_width'] = data['bb_upper'] - data['bb_lower']
    
    # Volume indicators
    data['on_balance_volume'] = np.cumsum(np.random.choice([-1, 1], n_samples) * data['volume'])
    data['volume_sma'] = np.random.uniform(0.8, 1.2, n_samples) * data['volume']
    data['volume_ratio'] = data['volume'] / data['volume_sma']
    
    # Volatility indicators
    data['atr'] = np.random.uniform(0.01, 0.05, n_samples) * base_prices
    data['true_range'] = np.random.uniform(0.01, 0.03, n_samples) * base_prices
    data['volatility'] = np.random.uniform(0.01, 0.08, n_samples)
    
    # Price action indicators
    data['pivot_point'] = (data['high'] + data['low'] + data['close']) / 3
    data['resistance_1'] = 2 * data['pivot_point'] - data['low']
    data['support_1'] = 2 * data['pivot_point'] - data['high']
    
    # Returns
    for period in [1, 4, 24, 96]:  # 15min, 1h, 6h, 24h
        data[f'return_{period}'] = np.random.normal(0, 0.02, n_samples)
        data[f'log_return_{period}'] = np.random.normal(0, 0.02, n_samples)
    
    # Momentum indicators
    data['momentum_10'] = np.random.normal(0, 0.1, n_samples)
    data['momentum_20'] = np.random.normal(0, 0.15, n_samples)
    data['rate_of_change'] = np.random.normal(0, 5, n_samples)
    
    # Advanced indicators
    data['awesome_oscillator'] = np.random.normal(0, 1000, n_samples)
    data['ultimate_oscillator'] = np.random.uniform(0, 100, n_samples)
    data['commodity_channel_index'] = np.random.normal(0, 100, n_samples)
    
    # Дополнительные признаки для достижения нужного размера
    print("🔧 Добавление дополнительных признаков...")
    
    # Лаги цен
    for lag in range(1, 21):  # 20 лагов
        data[f'close_lag_{lag}'] = np.roll(base_prices, lag)
        data[f'volume_lag_{lag}'] = np.roll(data['volume'], lag)
    
    # Статистические признаки
    window_sizes = [5, 10, 20, 50]
    for window in window_sizes:
        data[f'price_mean_{window}'] = np.random.uniform(0.95, 1.05, n_samples) * base_prices
        data[f'price_std_{window}'] = np.random.uniform(0.01, 0.05, n_samples) * base_prices
        data[f'volume_mean_{window}'] = np.random.uniform(0.8, 1.2, n_samples) * data['volume']
        data[f'volume_std_{window}'] = np.random.uniform(0.1, 0.3, n_samples) * data['volume']
    
    # Fourier компоненты
    for i in range(10):
        data[f'fft_real_{i}'] = np.random.normal(0, 1, n_samples)
        data[f'fft_imag_{i}'] = np.random.normal(0, 1, n_samples)
    
    # Временные признаки
    data['hour'] = timestamps.hour
    data['day_of_week'] = timestamps.dayofweek
    data['month'] = timestamps.month
    data['is_weekend'] = (timestamps.dayofweek >= 5).astype(int)
    
    # Создаем DataFrame
    print("📦 Создание DataFrame...")
    df = pd.DataFrame(data)
    
    # Добавляем случайные признаки до нужного колич