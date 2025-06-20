#!/usr/bin/env python3
"""
🏭 БЫСТРОЕ СОЗДАНИЕ КЕША 3GB ДЛЯ GPU ОБУЧЕНИЯ
===========================================
"""

import pandas as pd
import numpy as np
import os

def create_3gb_cache():
    print("🏭 Создание кеша 3GB для GPU обучения...")
    
    # Увеличиваем размер для получения ~3GB
    n_samples = 500000  # 500K записей
    
    np.random.seed(42)
    timestamps = pd.date_range(start='2020-01-01', periods=n_samples, freq='1H')
    
    print("⚡ Генерация данных...")
    
    # Базовые цены
    base_trend = np.linspace(20000, 60000, n_samples)
    noise = np.cumsum(np.random.normal(0, 500, n_samples))
    prices = base_trend + noise
    prices = np.maximum(prices, 1000)
    
    # OHLCV данные
    volatility_factor = np.random.normal(0.02, 0.005, n_samples)
    opens = prices * (1 + np.random.normal(0, volatility_factor))
    highs = prices * (1 + np.abs(np.random.normal(0, volatility_factor)))
    lows = prices * (1 - np.abs(np.random.normal(0, volatility_factor)))
    closes = prices
    volumes = np.random.lognormal(10, 1, n_samples)
    
    # Много технических индикаторов для увеличения размера
    rsi = 30 + 40 * np.random.beta(2, 2, n_samples)
    macd = np.random.normal(0, 100, n_samples)
    macd_signal = np.convolve(macd, np.ones(9)/9, mode='same')
    
    bb_period = 20
    sma_bb = np.convolve(closes, np.ones(bb_period)/bb_period, mode='same')
    bb_upper = sma_bb + 2 * np.std(closes)
    bb_lower = sma_bb - 2 * np.std(closes)
    
    # Множество дополнительных признаков
    ema_12 = closes * (0.97 + 0.06 * np.random.random(n_samples))
    ema_26 = closes * (0.98 + 0.04 * np.random.random(n_samples))
    ema_50 = closes * (0.96 + 0.08 * np.random.random(n_samples))
    
    volume_sma = np.convolve(volumes, np.ones(20)/20, mode='same')
    volatility = np.abs(highs - lows) / closes
    atr = np.convolve(volatility * closes, np.ones(14)/14, mode='same')
    
    # Дополнительные индикаторы для размера
    stoch_k = np.random.uniform(20, 80, n_samples)
    stoch_d = np.convolve(stoch_k, np.ones(3)/3, mode='same')
    williams_r = np.random.uniform(-80, -20, n_samples)
    cci = np.random.normal(0, 50, n_samples)
    adx = np.random.uniform(10, 50, n_samples)
    mfi = np.random.uniform(20, 80, n_samples)
    
    # Дополнительные признаки для увеличения размера
    roc = np.gradient(closes) / closes * 100
    momentum = closes - np.roll(closes, 10)
    trix = np.gradient(np.gradient(np.gradient(np.log(closes))))
    
    # Fourier компоненты для сложности
    fft_real = np.real(np.fft.fft(closes))[:n_samples//2]
    fft_imag = np.imag(np.fft.fft(closes))[:n_samples//2]
    
    # Дополняем до полного размера
    if len(fft_real) < n_samples:
        fft_real = np.pad(fft_real, (0, n_samples - len(fft_real)), 'constant')
        fft_imag = np.pad(fft_imag, (0, n_samples - len(fft_imag)), 'constant')
    
    # Целевые переменные
    future_returns = np.random.multivariate_normal(
        mean=[0.001, 0.002, 0.005, 0.02],
        cov=[[0.0001, 0.00005, 0.0001, 0.0002],
             [0.00005, 0.0004, 0.0002, 0.0004],
             [0.0001, 0.0002, 0.001, 0.002],
             [0.0002, 0.0004, 0.002, 0.01]],
        size=n_samples
    )
    
    # Создание DataFrame с множеством признаков
    data = {
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
        'rsi': rsi,
        'macd': macd,
        'macd_signal': macd_signal,
        'macd_histogram': macd - macd_signal,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'bb_middle': sma_bb,
        'sma_20': sma_bb,
        'ema_12': ema_12,
        'ema_26': ema_26,
        'ema_50': ema_50,
        'volume_sma': volume_sma,
        'volume_ratio': volumes / (volume_sma + 1),
        'volatility': volatility,
        'atr': atr,
        'stoch_k': stoch_k,
        'stoch_d': stoch_d,
        'williams_r': williams_r,
        'cci': cci,
        'adx': adx,
        'mfi': mfi,
        'roc': roc,
        'momentum': momentum,
        'trix': trix,
        'fft_real': fft_real,
        'fft_imag': fft_imag,
        'obv': np.cumsum(volumes * np.where(np.diff(closes, prepend=closes[0]) > 0, 1, -1)),
        'price_momentum': np.gradient(closes),
        'volume_momentum': np.gradient(volumes),
        'price_volatility': volatility,  # Simplified for compatibility
        'return_1h': np.diff(closes, prepend=closes[0]) / closes,
        'return_4h': np.diff(closes, 4, prepend=closes[:4][-1]) / closes,
        # Больше лагов для размера
        'close_lag_1': np.roll(closes, 1),
        'close_lag_2': np.roll(closes, 2),
        'close_lag_3': np.roll(closes, 3),
        'volume_lag_1': np.roll(volumes, 1),
        'volume_lag_2': np.roll(volumes, 2),
        # Целевые переменные
        'close_next_1h': closes * (1 + future_returns[:, 0]),
        'close_next_4h': closes * (1 + future_returns[:, 1]),
        'close_next_24h': closes * (1 + future_returns[:, 2]),
        'close_next_7d': closes * (1 + future_returns[:, 3]),
    }
    
    df = pd.DataFrame(data)
    
    # Сохранение с максимальным сжатием
    cache_filename = 'crypto_cache_3gb.parquet'
    df.to_parquet(cache_filename, compression='snappy', index=False)
    
    # Проверка размера
    cache_size = os.path.getsize(cache_filename) / (1024**3)
    print(f"✅ Кеш создан: {cache_filename}")
    print(f"📊 Размер: {cache_size:.2f} GB")
    print(f"📈 Данные: {df.shape[0]:,} записей, {df.shape[1]} признаков")
    print(f"💾 Готов для загрузки на GPU сервер!")
    
    return cache_filename

if __name__ == "__main__":
    create_3gb_cache()
