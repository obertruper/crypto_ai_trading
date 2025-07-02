#!/usr/bin/env python3
"""
Проверка улучшенных индикаторов после исправлений
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml

def check_indicators():
    """Проверка корректности улучшенных индикаторов"""
    
    print("🔍 Проверка улучшенных индикаторов после исправлений\n")
    
    # Загружаем небольшую выборку данных
    cache_dir = Path("data/processed")
    train_file = cache_dir / "train_data.parquet"
    
    if not train_file.exists():
        print("❌ Файл train_data.parquet не найден")
        return
    
    # Читаем первые 10000 записей для быстрой проверки
    df = pd.read_parquet(train_file, engine='pyarrow').head(10000)
    print(f"📊 Загружено {len(df)} записей для проверки\n")
    
    # Список индикаторов для проверки
    indicators_to_check = {
        'close_vwap_ratio': {
            'expected_range': (0.7, 1.3),
            'description': 'Расширенные границы для криптовалют (±30%)'
        },
        'vwap_extreme_deviation': {
            'expected_range': (0, 1),
            'description': 'Бинарный индикатор экстремальных отклонений'
        },
        'bb_position': {
            'expected_range': (0, 1),
            'description': 'Позиция в Bollinger Bands (с клиппингом)'
        },
        'bb_breakout_upper': {
            'expected_range': (0, 1),
            'description': 'Пробой верхней границы BB'
        },
        'bb_breakout_lower': {
            'expected_range': (0, 1),
            'description': 'Пробой нижней границы BB'
        },
        'price_impact': {
            'expected_range': (0, 10),
            'description': 'Price impact с dollar volume'
        },
        'kyle_lambda': {
            'expected_range': (0, 10),
            'description': 'Kyle Lambda (правильная формула)'
        },
        'realized_vol_daily': {
            'expected_range': (0, 5),
            'description': 'Дневная волатильность'
        },
        'volume_zscore': {
            'expected_range': (-50, 50),
            'description': 'Z-score объема (расширенный диапазон)'
        },
        'toxicity': {
            'expected_range': (0, 1),
            'description': 'Toxicity = 1/(1+price_impact)'
        }
    }
    
    print("📈 СТАТИСТИКА ИНДИКАТОРОВ:\n")
    
    for indicator, info in indicators_to_check.items():
        if indicator in df.columns:
            stats = df[indicator].describe()
            min_val, max_val = info['expected_range']
            
            # Проверка диапазона
            in_range = (stats['min'] >= min_val - 0.01) and (stats['max'] <= max_val + 0.01)
            status = "✅" if in_range else "⚠️"
            
            print(f"{status} {indicator}:")
            print(f"   {info['description']}")
            print(f"   Диапазон: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"   Среднее: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            print(f"   Ожидается: {info['expected_range']}")
            
            # Специальные проверки
            if indicator == 'toxicity':
                mean_expected = stats['mean'] > 0.95
                print(f"   {'✅' if mean_expected else '❌'} Mean > 0.95: {mean_expected}")
            
            if indicator == 'bb_breakout_upper':
                breakout_pct = stats['mean'] * 100
                print(f"   Процент пробоев вверх: {breakout_pct:.1f}%")
            
            if indicator == 'volume_zscore':
                extreme_count = ((df[indicator] > 10) | (df[indicator] < -10)).sum()
                extreme_pct = extreme_count / len(df) * 100
                print(f"   Экстремальные значения (|z| > 10): {extreme_pct:.2f}%")
            
            print()
        else:
            print(f"❌ {indicator}: НЕ НАЙДЕН В ДАННЫХ\n")
    
    # Проверка корреляций
    print("📊 КОРРЕЛЯЦИИ МЕЖДУ ИНДИКАТОРАМИ:\n")
    
    correlation_pairs = [
        ('price_impact', 'toxicity'),
        ('bb_position', 'bb_breakout_upper'),
        ('close_vwap_ratio', 'vwap_extreme_deviation'),
        ('volume_zscore', 'price_impact')
    ]
    
    for ind1, ind2 in correlation_pairs:
        if ind1 in df.columns and ind2 in df.columns:
            corr = df[ind1].corr(df[ind2])
            print(f"Корреляция {ind1} <-> {ind2}: {corr:.3f}")
    
    print("\n🎯 РЕКОМЕНДАЦИИ:")
    print("1. Если индикаторы отсутствуют - нужно пересоздать датасет")
    print("2. Если диапазоны некорректны - проверить формулы в feature_engineering.py")
    print("3. Для применения изменений запустите: python prepare_trading_data.py --force-recreate")

if __name__ == "__main__":
    check_indicators()