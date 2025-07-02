#!/usr/bin/env python3
"""
Скрипт для проверки совместимости кэша между prepare_trading_data.py и main.py
"""

import pandas as pd
import pickle
from pathlib import Path
import yaml

def check_cache_compatibility():
    """Проверка что кэш из prepare_trading_data.py совместим с main.py"""
    
    print("🔍 Проверка совместимости кэша данных...\n")
    
    # Проверяем наличие файлов
    cache_dir = Path("data/processed")
    required_files = [
        "train_data.parquet",
        "val_data.parquet", 
        "test_data.parquet"
    ]
    
    print("📁 Проверка наличия файлов:")
    all_exist = True
    for file_name in required_files:
        file_path = cache_dir / file_name
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ {file_name}: {size_mb:.2f} MB")
        else:
            print(f"   ❌ {file_name}: НЕ НАЙДЕН")
            all_exist = False
    
    if not all_exist:
        print("\n❌ Не все файлы кэша найдены!")
        return False
    
    # Загружаем данные
    print("\n📊 Загрузка и анализ данных:")
    train_data = pd.read_parquet(cache_dir / "train_data.parquet")
    val_data = pd.read_parquet(cache_dir / "val_data.parquet")
    test_data = pd.read_parquet(cache_dir / "test_data.parquet")
    
    print(f"\n📈 Размеры датасетов:")
    print(f"   - Train: {len(train_data):,} записей")
    print(f"   - Val: {len(val_data):,} записей")
    print(f"   - Test: {len(test_data):,} записей")
    
    # Проверяем структуру данных
    print(f"\n🔧 Структура данных:")
    print(f"   - Всего колонок: {len(train_data.columns)}")
    
    # Импортируем функции из data.constants (если существует)
    try:
        from data.constants import (
            get_feature_columns, get_target_columns, 
            validate_data_structure, TRADING_TARGET_VARIABLES
        )
        
        # Проверяем через validate_data_structure (как в main.py)
        try:
            data_info = validate_data_structure(train_data)
            feature_cols = data_info['feature_cols']
            target_cols = data_info['target_cols']
            
            print(f"   - Признаков: {len(feature_cols)}")
            print(f"   - Целевых переменных: {len(target_cols)}")
            print(f"   - Служебных колонок: {len(train_data.columns) - len(feature_cols) - len(target_cols)}")
            
            print(f"\n✅ Структура данных совместима с main.py!")
            
        except Exception as e:
            print(f"\n❌ Ошибка валидации структуры: {e}")
            return False
            
    except ImportError:
        # Если constants.py не существует, делаем базовую проверку
        print("\n⚠️ Модуль data.constants не найден, выполняем базовую проверку...")
        
        # Ищем целевые переменные
        target_patterns = ['tp', 'sl', 'reached', 'hit', 'future_return', 'best_direction', 
                          'optimal_entry', 'expected_value', 'target_return']
        
        target_cols = []
        feature_cols = []
        
        for col in train_data.columns:
            if any(pattern in col for pattern in target_patterns):
                target_cols.append(col)
            elif col not in ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'turnover']:
                feature_cols.append(col)
        
        print(f"   - Найдено признаков: ~{len(feature_cols)}")
        print(f"   - Найдено целевых переменных: ~{len(target_cols)}")
    
    # Проверяем целевые переменные из конфига
    print("\n🎯 Проверка целевых переменных из config.yaml:")
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config_targets = config['model'].get('target_variables', [])
    
    if config_targets:
        print(f"   Ожидается {len(config_targets)} целевых переменных")
        
        missing_targets = []
        for target in config_targets:
            if target not in train_data.columns:
                missing_targets.append(target)
        
        if missing_targets:
            print(f"\n❌ Отсутствуют целевые переменные: {missing_targets}")
        else:
            print(f"   ✅ Все {len(config_targets)} целевых переменных найдены!")
    
    # Проверяем close_vwap_ratio (исправленный)
    print("\n🔍 Проверка close_vwap_ratio (должен быть исправлен):")
    if 'close_vwap_ratio' in train_data.columns:
        stats = train_data['close_vwap_ratio'].describe()
        print(f"   Min: {stats['min']:.4f}")
        print(f"   Max: {stats['max']:.4f}")
        print(f"   Mean: {stats['mean']:.4f}")
        
        if stats['max'] > 10:
            print("   ❌ ПРОБЛЕМА: close_vwap_ratio все еще имеет экстремальные значения!")
            print("   Необходимо пересоздать кэш после исправления feature_engineering.py")
            return False
        else:
            print("   ✅ close_vwap_ratio в норме!")
    
    # Итоговая проверка
    print("\n" + "="*60)
    print("✅ КЭШ СОВМЕСТИМ И ГОТОВ К ИСПОЛЬЗОВАНИЮ!")
    print("="*60)
    print("\n🚀 Можно запускать обучение:")
    print("   python main.py --mode train")
    
    return True

if __name__ == "__main__":
    check_cache_compatibility()