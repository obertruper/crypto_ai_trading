"""
Константы для определения структуры данных
"""

# Служебные колонки (не признаки и не целевые)
SERVICE_COLUMNS = ['id', 'symbol', 'datetime', 'timestamp', 'sector']

# Основные 36 целевых переменных для торговой модели
TRADING_TARGET_VARIABLES = [
    # Базовые возвраты (4)
    'future_return_1', 'future_return_2', 'future_return_3', 'future_return_4',
    
    # Long позиции (15)
    'long_tp1_hit', 'long_tp1_reached', 'long_tp1_time',
    'long_tp2_hit', 'long_tp2_reached', 'long_tp2_time',
    'long_tp3_hit', 'long_tp3_reached', 'long_tp3_time',
    'long_sl_hit', 'long_sl_reached', 'long_sl_time',
    'long_optimal_entry_time', 'long_optimal_entry_price', 'long_optimal_entry_improvement',
    
    # Short позиции (15)
    'short_tp1_hit', 'short_tp1_reached', 'short_tp1_time',
    'short_tp2_hit', 'short_tp2_reached', 'short_tp2_time',
    'short_tp3_hit', 'short_tp3_reached', 'short_tp3_time',
    'short_sl_hit', 'short_sl_reached', 'short_sl_time',
    'short_optimal_entry_time', 'short_optimal_entry_price', 'short_optimal_entry_improvement',
    
    # Направление и целевая (2)
    'best_direction', 'target_return_1h'
]

# Дополнительные целевые переменные (13) - для анализа, но не для обучения
ADDITIONAL_TARGET_VARIABLES = [
    # Будущие экстремумы (8)
    'future_high_1', 'future_high_2', 'future_high_3', 'future_high_4',
    'future_low_1', 'future_low_2', 'future_low_3', 'future_low_4',
    
    # Ожидаемые значения и результаты (4)
    'long_expected_value', 'short_expected_value',
    'long_final_result', 'short_final_result',
    
    # Сила сигнала (1)
    'signal_strength'
]

# Все целевые переменные (49)
ALL_TARGET_VARIABLES = TRADING_TARGET_VARIABLES + ADDITIONAL_TARGET_VARIABLES

# Префиксы для автоматического определения целевых переменных
TARGET_PREFIXES = (
    'target_', 'future_return_', 'long_tp', 'short_tp', 
    'long_sl', 'short_sl', 'long_optimal', 'short_optimal',
    'best_direction'
)

# Расширенные префиксы (для поиска всех 49)
EXTENDED_TARGET_PREFIXES = (
    'target_', 'future_', 'long_tp', 'short_tp', 'long_sl', 'short_sl',
    'long_optimal', 'short_optimal', 'long_expected', 'short_expected',
    'best_direction', 'signal_strength', 'long_final', 'short_final'
)

def get_feature_columns(df_columns):
    """Получить список признаков из колонок DataFrame"""
    return [col for col in df_columns 
            if col not in SERVICE_COLUMNS 
            and col not in ALL_TARGET_VARIABLES]

def get_target_columns(df_columns, use_extended=False):
    """Получить список целевых переменных из колонок DataFrame"""
    if use_extended:
        # Возвращает все 49 переменных
        return [col for col in df_columns if col in ALL_TARGET_VARIABLES]
    else:
        # Возвращает только основные 36 для обучения
        return [col for col in df_columns if col in TRADING_TARGET_VARIABLES]

def validate_data_structure(df):
    """Проверить структуру данных"""
    missing_targets = [col for col in TRADING_TARGET_VARIABLES if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Отсутствуют обязательные целевые переменные: {missing_targets}")
    
    feature_cols = get_feature_columns(df.columns)
    target_cols = get_target_columns(df.columns)
    
    return {
        'n_features': len(feature_cols),
        'n_targets': len(target_cols),
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'missing_targets': missing_targets
    }