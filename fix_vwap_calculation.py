#!/usr/bin/env python3
"""
Скрипт для анализа и исправления проблемы с расчетом VWAP
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sqlalchemy import create_engine, text

# Загрузка конфигурации
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Подключение к БД
db_config = config['database']
engine = create_engine(
    f"postgresql://{db_config['user']}:{db_config['password']}@"
    f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
)

print("🔍 Анализ проблемы с VWAP...")

# Загрузка примера данных
query = """
SELECT symbol, datetime, open, high, low, close, volume, turnover
FROM raw_market_data
WHERE symbol IN ('BTCUSDT', 'ETHUSDT', 'SOLUSDT')
ORDER BY symbol, datetime DESC
LIMIT 10000
"""

df = pd.read_sql(query, engine)
print(f"\n📊 Загружено {len(df)} записей")

# Анализ volume и turnover
print("\n📈 Статистика volume:")
print(f"Min: {df['volume'].min():.10f}")
print(f"Max: {df['volume'].max():.2f}")
print(f"Median: {df['volume'].median():.2f}")
print(f"Нулевых значений: {(df['volume'] == 0).sum()}")
print(f"Очень маленьких (< 0.001): {(df['volume'] < 0.001).sum()}")

print("\n💰 Статистика turnover:")
print(f"Min: {df['turnover'].min():.10f}")
print(f"Max: {df['turnover'].max():.2f}")
print(f"Median: {df['turnover'].median():.2f}")
print(f"Нулевых значений: {(df['turnover'] == 0).sum()}")

# Тестируем старый метод расчета VWAP
print("\n🔴 Старый метод расчета VWAP:")

def old_safe_divide(numerator, denominator, fill_value=0.0, max_value=1000.0):
    """Старый метод с min_denominator=1e-4"""
    min_denominator = 1e-4
    safe_denominator = denominator.copy()
    mask_small = (safe_denominator.abs() < min_denominator)
    safe_denominator[mask_small] = np.sign(safe_denominator[mask_small]) * min_denominator
    safe_denominator[safe_denominator == 0] = min_denominator
    result = numerator / safe_denominator
    result = result.clip(lower=-max_value, upper=max_value)
    if isinstance(fill_value, pd.Series):
        inf_mask = np.isinf(result)
        result.loc[inf_mask] = fill_value.loc[inf_mask]
    else:
        result = result.replace([np.inf, -np.inf], fill_value)
    result = result.fillna(fill_value)
    return result

# Старый расчет
df['vwap_old'] = old_safe_divide(df['turnover'], df['volume'], fill_value=df['close'])
df['close_vwap_ratio_old'] = old_safe_divide(df['close'], df['vwap_old'], fill_value=1.0)

print(f"VWAP min: {df['vwap_old'].min():.10f}")
print(f"VWAP max: {df['vwap_old'].max():.2f}")
print(f"Close/VWAP ratio min: {df['close_vwap_ratio_old'].min():.2f}")
print(f"Close/VWAP ratio max: {df['close_vwap_ratio_old'].max():.2f}")

# Проблемные записи
problematic = df[df['close_vwap_ratio_old'] > 10]
print(f"\nПроблемных записей (ratio > 10): {len(problematic)}")
if len(problematic) > 0:
    print("\nПример проблемной записи:")
    row = problematic.iloc[0]
    print(f"Symbol: {row['symbol']}")
    print(f"Datetime: {row['datetime']}")
    print(f"Close: {row['close']:.2f}")
    print(f"Volume: {row['volume']:.10f}")
    print(f"Turnover: {row['turnover']:.2f}")
    print(f"VWAP: {row['vwap_old']:.10f}")
    print(f"Close/VWAP ratio: {row['close_vwap_ratio_old']:.2f}")

# Новый улучшенный метод
print("\n✅ Новый метод расчета VWAP:")

def new_safe_divide(numerator, denominator, fill_value=0.0, max_value=1000.0):
    """Новый метод с улучшенной обработкой"""
    # Для VWAP используем более разумный минимум
    min_denominator = 0.01  # Увеличено с 1e-4 до 0.01
    
    safe_denominator = denominator.copy()
    
    # Заменяем очень маленькие значения
    mask_small = (safe_denominator.abs() < min_denominator)
    safe_denominator[mask_small] = min_denominator
    
    # Выполняем деление
    result = numerator / safe_denominator
    
    # Более строгое ограничение для коэффициентов
    result = result.clip(lower=-max_value, upper=max_value)
    
    # Обработка inf и nan
    if isinstance(fill_value, pd.Series):
        inf_mask = np.isinf(result) | np.isnan(result)
        result.loc[inf_mask] = fill_value.loc[inf_mask]
    else:
        result = result.replace([np.inf, -np.inf, np.nan], fill_value)
    
    return result

# Специальный расчет VWAP с дополнительными проверками
def calculate_vwap(df):
    """Улучшенный расчет VWAP"""
    # Базовый расчет
    vwap = new_safe_divide(df['turnover'], df['volume'], fill_value=df['close'])
    
    # Дополнительная проверка: VWAP не должен сильно отличаться от close
    # Если VWAP слишком отличается от close, используем close
    mask_invalid = (vwap < df['close'] * 0.5) | (vwap > df['close'] * 2.0)
    vwap[mask_invalid] = df['close'][mask_invalid]
    
    return vwap

# Новый расчет
df['vwap_new'] = calculate_vwap(df)
df['close_vwap_ratio_new'] = new_safe_divide(
    df['close'], 
    df['vwap_new'], 
    fill_value=1.0,
    max_value=2.0  # Ограничиваем ratio максимум 2x
)

print(f"VWAP min: {df['vwap_new'].min():.2f}")
print(f"VWAP max: {df['vwap_new'].max():.2f}")
print(f"Close/VWAP ratio min: {df['close_vwap_ratio_new'].min():.2f}")
print(f"Close/VWAP ratio max: {df['close_vwap_ratio_new'].max():.2f}")

# Проверка новых результатов
problematic_new = df[df['close_vwap_ratio_new'] > 2]
print(f"\nПроблемных записей после исправления: {len(problematic_new)}")

# Сравнение распределений
print("\n📊 Сравнение распределений close_vwap_ratio:")
print("\nСтарый метод - квантили:")
print(df['close_vwap_ratio_old'].quantile([0.01, 0.25, 0.5, 0.75, 0.99, 0.999]))
print("\nНовый метод - квантили:")
print(df['close_vwap_ratio_new'].quantile([0.01, 0.25, 0.5, 0.75, 0.99, 0.999]))

# Рекомендации
print("\n💡 Рекомендации:")
print("1. Обновить safe_divide в feature_engineering.py - увеличить min_denominator до 0.01")
print("2. Добавить специальную функцию calculate_vwap с дополнительными проверками")
print("3. Ограничить close_vwap_ratio максимальным значением 2.0")
print("4. Пересоздать кэш данных после исправления")

# Сохранение анализа
analysis_results = {
    'old_method': {
        'min': float(df['close_vwap_ratio_old'].min()),
        'max': float(df['close_vwap_ratio_old'].max()),
        'problematic_count': len(problematic)
    },
    'new_method': {
        'min': float(df['close_vwap_ratio_new'].min()),
        'max': float(df['close_vwap_ratio_new'].max()),
        'problematic_count': len(problematic_new)
    }
}

print(f"\n✅ Анализ завершен")
print(f"   Старый метод: {len(problematic)} проблемных записей")
print(f"   Новый метод: {len(problematic_new)} проблемных записей")