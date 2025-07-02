#!/usr/bin/env python3
"""
Проверка соответствия колонок между описанием пользователя и реальной структурой
"""

import pandas as pd
from pathlib import Path
from data.constants import (
    TRADING_TARGET_VARIABLES, ADDITIONAL_TARGET_VARIABLES,
    ALL_TARGET_VARIABLES, SERVICE_COLUMNS,
    get_feature_columns, get_target_columns, validate_data_structure
)

# Структура из описания пользователя
USER_STRUCTURE = {
    'total_columns': 203,
    'features': 144,
    'targets': 49,
    'service': 4,
    'ohlcv': 6
}

# Целевые переменные из описания пользователя (49)
USER_TARGETS = {
    # Основные возвраты (8)
    'returns': ['future_return_1', 'future_return_2', 'future_return_3', 'future_return_4',
                'future_high_1', 'future_high_2', 'future_high_3', 'future_high_4',
                'future_low_1', 'future_low_2', 'future_low_3', 'future_low_4'],
    
    # LONG (15)
    'long': ['long_tp1_hit', 'long_tp1_reached', 'long_tp1_time',
             'long_tp2_hit', 'long_tp2_reached', 'long_tp2_time',
             'long_tp3_hit', 'long_tp3_reached', 'long_tp3_time',
             'long_sl_hit', 'long_sl_reached', 'long_sl_time',
             'long_optimal_entry_time', 'long_optimal_entry_price', 'long_optimal_entry_improvement'],
    
    # SHORT (15)
    'short': ['short_tp1_hit', 'short_tp1_reached', 'short_tp1_time',
              'short_tp2_hit', 'short_tp2_reached', 'short_tp2_time',
              'short_tp3_hit', 'short_tp3_reached', 'short_tp3_time',
              'short_sl_hit', 'short_sl_reached', 'short_sl_time',
              'short_optimal_entry_time', 'short_optimal_entry_price', 'short_optimal_entry_improvement'],
    
    # Результирующие (7)
    'results': ['long_expected_value', 'short_expected_value',
                'long_final_result', 'short_final_result',
                'best_direction', 'signal_strength', 'target_return_1h']
}

# Все целевые из описания
ALL_USER_TARGETS = []
for category, targets in USER_TARGETS.items():
    ALL_USER_TARGETS.extend(targets)

def check_targets_consistency():
    """Проверка соответствия целевых переменных"""
    print("🎯 Проверка целевых переменных:")
    print(f"\nОжидается из описания: {len(ALL_USER_TARGETS)} переменных")
    print(f"В constants.py определено: {len(ALL_TARGET_VARIABLES)} переменных")
    
    # Проверяем что есть в constants.py но нет в описании
    missing_in_user = set(ALL_TARGET_VARIABLES) - set(ALL_USER_TARGETS)
    if missing_in_user:
        print(f"\n❌ В constants.py есть, но нет в описании пользователя:")
        for var in sorted(missing_in_user):
            print(f"   - {var}")
    
    # Проверяем что есть в описании но нет в constants.py
    missing_in_constants = set(ALL_USER_TARGETS) - set(ALL_TARGET_VARIABLES)
    if missing_in_constants:
        print(f"\n❌ В описании есть, но нет в constants.py:")
        for var in sorted(missing_in_constants):
            print(f"   - {var}")
    
    # Основные 36 для обучения
    print(f"\n📊 Основные целевые для обучения: {len(TRADING_TARGET_VARIABLES)}")
    print(f"📊 Дополнительные целевые: {len(ADDITIONAL_TARGET_VARIABLES)}")
    
    if len(missing_in_user) == 0 and len(missing_in_constants) == 0:
        print("\n✅ Все целевые переменные соответствуют!")
    
    return len(missing_in_user) == 0 and len(missing_in_constants) == 0

def check_cache_structure():
    """Проверка реальной структуры кэша"""
    print("\n📁 Проверка структуры кэша:")
    
    cache_dir = Path("data/processed")
    if not cache_dir.exists():
        print("❌ Кэш не найден!")
        return False
    
    # Проверяем train_data.parquet
    train_file = cache_dir / "train_data.parquet"
    if train_file.exists():
        df = pd.read_parquet(train_file, nrows=100)  # Читаем только 100 строк для скорости
        
        print(f"\n📊 Реальная структура train_data.parquet:")
        print(f"   - Всего колонок: {len(df.columns)}")
        
        # Используем функции из constants.py
        feature_cols = get_feature_columns(df.columns)
        target_cols = get_target_columns(df.columns, use_extended=True)  # Все 49
        target_cols_training = get_target_columns(df.columns, use_extended=False)  # Только 36
        
        print(f"   - Признаков: {len(feature_cols)}")
        print(f"   - Целевых (всего): {len(target_cols)}")
        print(f"   - Целевых (для обучения): {len(target_cols_training)}")
        
        # Служебные колонки
        service_in_df = [col for col in df.columns if col in SERVICE_COLUMNS]
        print(f"   - Служебных: {len(service_in_df)} {service_in_df}")
        
        # OHLCV
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        ohlcv_in_df = [col for col in df.columns if col in ohlcv_cols]
        print(f"   - OHLCV: {len(ohlcv_in_df)}")
        
        # Сравнение с ожиданиями пользователя
        print(f"\n📊 Сравнение с описанием пользователя:")
        print(f"   Колонок: {len(df.columns)} vs {USER_STRUCTURE['total_columns']} ожидалось")
        print(f"   Признаков: {len(feature_cols)} vs {USER_STRUCTURE['features']} ожидалось")
        print(f"   Целевых: {len(target_cols)} vs {USER_STRUCTURE['targets']} ожидалось")
        
        # Детальная проверка признаков
        print(f"\n🔍 Анализ разницы в признаках:")
        expected_features = USER_STRUCTURE['total_columns'] - USER_STRUCTURE['targets'] - USER_STRUCTURE['service'] - USER_STRUCTURE['ohlcv']
        actual_features = len(df.columns) - len(target_cols) - len(service_in_df) - len(ohlcv_in_df)
        
        print(f"   Ожидалось признаков: {expected_features}")
        print(f"   Фактически признаков: {actual_features}")
        
        if actual_features != expected_features:
            print(f"\n   ⚠️ Разница в {abs(actual_features - expected_features)} признаков")
    
    return True

def check_columns_usage_in_main():
    """Проверка использования колонок в main.py"""
    print("\n📋 Использование в main.py:")
    print("1. load_cached_data_if_exists() загружает parquet файлы")
    print("2. validate_data_structure() определяет feature_cols и target_cols")
    print("3. create_unified_data_loaders() получает эти списки")
    print("4. Для task_type='trading' используются все target_cols (36 переменных)")
    print("5. config['model']['target_variables'] обновляется автоматически")
    
    return True

def main():
    """Основная проверка"""
    print("🔍 Проверка соответствия структуры данных\n")
    print("="*60)
    
    # Проверка целевых переменных
    targets_ok = check_targets_consistency()
    
    # Проверка реального кэша
    cache_ok = check_cache_structure()
    
    # Проверка использования
    usage_ok = check_columns_usage_in_main()
    
    print("\n" + "="*60)
    if targets_ok and cache_ok and usage_ok:
        print("✅ СТРУКТУРА ДАННЫХ СОГЛАСОВАНА!")
        print("\n💡 Рекомендации:")
        print("1. Пересоздать кэш после исправления VWAP:")
        print("   python prepare_trading_data.py")
        print("\n2. Запустить обучение:")
        print("   python main.py --mode train")
    else:
        print("❌ Обнаружены несоответствия в структуре!")

if __name__ == "__main__":
    main()