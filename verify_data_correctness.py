#!/usr/bin/env python3
"""
Полная проверка корректности сформированных данных с валидацией индикаторов
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Импорт нашего валидатора индикаторов
from utils.indicator_validator import IndicatorValidator

# Цветной вывод
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")

def print_success(text):
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKBLUE}ℹ️  {text}{Colors.ENDC}")

def check_file_existence():
    """Проверка наличия файлов"""
    print_header("1. ПРОВЕРКА НАЛИЧИЯ ФАЙЛОВ")
    
    cache_dir = Path("data/processed")
    required_files = ["train_data.parquet", "val_data.parquet", "test_data.parquet"]
    
    all_exist = True
    file_sizes = {}
    
    for file_name in required_files:
        file_path = cache_dir / file_name
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            file_sizes[file_name] = size_mb
            print_success(f"{file_name}: {size_mb:.2f} MB")
        else:
            print_error(f"{file_name}: НЕ НАЙДЕН")
            all_exist = False
    
    return all_exist, file_sizes

def check_data_structure(df, name):
    """Проверка структуры данных"""
    print(f"\n{Colors.BOLD}Проверка {name}:{Colors.ENDC}")
    
    # Базовая информация
    print_info(f"Размер: {len(df):,} записей, {len(df.columns)} колонок")
    print_info(f"Память: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Проверка обязательных колонок
    required_cols = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'turnover']
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        print_error(f"Отсутствуют обязательные колонки: {missing_required}")
    else:
        print_success("Все обязательные колонки присутствуют")
    
    # Проверка целевых переменных
    from data.constants import TRADING_TARGET_VARIABLES, ALL_TARGET_VARIABLES
    
    missing_targets = [col for col in TRADING_TARGET_VARIABLES if col not in df.columns]
    if missing_targets:
        print_error(f"Отсутствуют целевые переменные: {len(missing_targets)} из {len(TRADING_TARGET_VARIABLES)}")
    else:
        print_success(f"Все {len(TRADING_TARGET_VARIABLES)} основных целевых переменных присутствуют")
    
    # Подсчет типов колонок
    feature_cols = [col for col in df.columns 
                   if col not in ALL_TARGET_VARIABLES 
                   and col not in ['id', 'symbol', 'datetime', 'timestamp']]
    
    print_info(f"Признаков: {len(feature_cols)}")
    print_info(f"Целевых переменных: {len([col for col in df.columns if col in ALL_TARGET_VARIABLES])}")
    
    return len(missing_required) == 0 and len(missing_targets) == 0

def check_data_quality(df, name):
    """Проверка качества данных"""
    print(f"\n{Colors.BOLD}Качество данных {name}:{Colors.ENDC}")
    
    issues = []
    
    # 1. Проверка NaN
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print_warning(f"Найдено {len(nan_cols)} колонок с NaN значениями")
        worst_cols = nan_cols.nlargest(5)
        for col, count in worst_cols.items():
            pct = count / len(df) * 100
            print(f"   - {col}: {count:,} NaN ({pct:.1f}%)")
        issues.append("nan_values")
    else:
        print_success("Нет пропущенных значений")
    
    # 2. Проверка inf
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count
    
    if inf_counts:
        print_warning(f"Найдено {len(inf_counts)} колонок с inf значениями")
        for col, count in list(inf_counts.items())[:5]:
            print(f"   - {col}: {count:,} inf")
        issues.append("inf_values")
    else:
        print_success("Нет бесконечных значений")
    
    # 3. Проверка дубликатов
    if 'datetime' in df.columns and 'symbol' in df.columns:
        duplicates = df.duplicated(subset=['datetime', 'symbol']).sum()
        if duplicates > 0:
            print_warning(f"Найдено {duplicates:,} дубликатов по datetime+symbol")
            issues.append("duplicates")
        else:
            print_success("Нет дубликатов")
    
    return issues


def check_target_distribution(df, name):
    """Проверка распределения целевых переменных"""
    print(f"\n{Colors.BOLD}Распределение целевых переменных {name}:{Colors.ENDC}")
    
    # Проверка бинарных целевых
    binary_targets = []
    for col in df.columns:
        if any(pattern in col for pattern in ['_hit', '_reached']):
            binary_targets.append(col)
    
    issues = []
    for target in binary_targets[:10]:  # Проверяем первые 10
        if target in df.columns:
            unique_vals = df[target].unique()
            if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                print_error(f"{target}: некорректные значения {unique_vals}")
                issues.append(f"{target}_invalid")
            else:
                pos_rate = df[target].mean() * 100
                print_info(f"{target}: {pos_rate:.1f}% положительных")
    
    # Проверка best_direction
    if 'best_direction' in df.columns:
        direction_counts = df['best_direction'].value_counts()
        print("\nРаспределение best_direction:")
        for direction, count in direction_counts.items():
            pct = count / len(df) * 100
            print(f"   {direction}: {count:,} ({pct:.1f}%)")
    
    return issues

def check_temporal_consistency(df, name):
    """Проверка временной последовательности"""
    print(f"\n{Colors.BOLD}Временная последовательность {name}:{Colors.ENDC}")
    
    if 'datetime' not in df.columns:
        print_error("Колонка datetime отсутствует")
        return ["no_datetime"]
    
    issues = []
    
    # Проверка сортировки
    if not df['datetime'].is_monotonic_increasing:
        print_warning("Данные не отсортированы по времени")
        issues.append("not_sorted")
    else:
        print_success("Данные отсортированы по времени")
    
    # Проверка диапазона дат
    date_min = df['datetime'].min()
    date_max = df['datetime'].max()
    print_info(f"Период: {date_min} - {date_max}")
    
    # Проверка гэпов по символам
    if 'symbol' in df.columns:
        symbols = df['symbol'].unique()
        print(f"\nПроверка временных гэпов для {len(symbols)} символов:")
        
        large_gaps = 0
        for symbol in symbols[:5]:  # Проверяем первые 5 символов
            symbol_data = df[df['symbol'] == symbol].sort_values('datetime')
            time_diff = symbol_data['datetime'].diff()
            expected_diff = pd.Timedelta('15 minutes')
            gaps = time_diff[time_diff > expected_diff * 2]  # Гэпы больше 30 минут
            if len(gaps) > 0:
                large_gaps += len(gaps)
        
        if large_gaps > 0:
            print_warning(f"Обнаружено {large_gaps} временных гэпов > 30 минут")
            issues.append("time_gaps")
        else:
            print_success("Нет значительных временных гэпов")
    
    return issues

def check_critical_indicators(df, name):
    """Проверка критических индикаторов с использованием профессионального валидатора"""
    print(f"\n{Colors.BOLD}Проверка индикаторов {name} (Enhanced):{Colors.ENDC}")
    
    # Используем наш продвинутый валидатор
    validator = IndicatorValidator()
    
    try:
        validation_results = validator.validate_dataframe(df, strict=False)
        
        # Конвертируем результаты в наш формат
        issues = []
        
        # Ошибки - критические проблемы
        for error in validation_results['errors']:
            print_error(error)
            if "extreme" in error.lower():
                issues.append("extreme_values")
            elif "нормализ" in error.lower():
                issues.append("normalization_error")
            else:
                issues.append("indicator_error")
        
        # Предупреждения
        for warning in validation_results['warnings']:
            print_warning(warning)
            if "vwap" in warning.lower():
                issues.append("vwap_warning")
            else:
                issues.append("indicator_warning")
        
        # Информационные сообщения (первые 3)
        for info in validation_results['info'][:3]:
            print_success(info)
        
        if len(validation_results['info']) > 3:
            print_info(f"... и еще {len(validation_results['info']) - 3} корректных индикаторов")
        
        # Статистика по проверенным индикаторам
        total_checked = len(validation_results['statistics'])
        total_errors = len(validation_results['errors'])
        total_warnings = len(validation_results['warnings'])
        
        print(f"\n{Colors.BOLD}📊 Статистика валидации:{Colors.ENDC}")
        print_info(f"Проверено индикаторов: {total_checked}")
        if total_errors > 0:
            print_error(f"Критические ошибки: {total_errors}")
        if total_warnings > 0:
            print_warning(f"Предупреждения: {total_warnings}")
        if total_errors == 0 and total_warnings == 0:
            print_success("Все индикаторы прошли валидацию!")
            
        return issues
        
    except Exception as e:
        print_error(f"Ошибка при валидации индикаторов: {e}")
        # Fallback к базовой проверке
        return check_basic_indicators(df)

def check_basic_indicators(df):
    """Базовая проверка индикаторов (fallback)"""
    issues = []
    
    # Основные проверки
    critical_ranges = {
        'rsi': (0, 100),
        'stoch_k': (0, 100), 
        'stoch_d': (0, 100),
        'adx': (0, 100),
        'toxicity': (0, 1),
        'bb_position': (0, 1)
    }
    
    for indicator, (min_val, max_val) in critical_ranges.items():
        if indicator in df.columns:
            stats = df[indicator].describe()
            
            # Проверка на нормализацию (подозрительные значения)
            if abs(stats['mean']) < 1.0 and stats['std'] < 2.0 and indicator in ['rsi', 'stoch_k', 'stoch_d', 'adx']:
                print_error(f"❌ {indicator}: ПОДОЗРЕНИЕ НА НОРМАЛИЗАЦИЮ! Mean={stats['mean']:.3f}, Std={stats['std']:.3f}")
                issues.append(f"{indicator}_normalized")
            elif stats['min'] < min_val or stats['max'] > max_val:
                print_warning(f"⚠️ {indicator}: [{stats['min']:.3f}, {stats['max']:.3f}] выходит за [{min_val}, {max_val}]")
                issues.append(f"{indicator}_range")
            else:
                print_success(f"✅ {indicator}: корректный диапазон [{stats['min']:.3f}, {stats['max']:.3f}]")
    
    return issues

def check_technical_indicators_enhanced(df, name):
    """ИСПРАВЛЕННАЯ проверка технических индикаторов"""
    print(f"\n{Colors.BOLD}Расширенная проверка технических индикаторов {name}:{Colors.ENDC}")
    
    issues = []
    indicators_checked = 0
    indicators_passed = 0
    
    # ИСПРАВЛЕНО: Правильные диапазоны для технических индикаторов
    technical_ranges = {
        'rsi': (0, 100, "RSI должен быть от 0 до 100"),
        'stoch_k': (0, 100, "Stochastic %K должен быть от 0 до 100"),
        'stoch_d': (0, 100, "Stochastic %D должен быть от 0 до 100"),
        'adx': (0, 100, "ADX должен быть от 0 до 100"),
        'adx_pos': (0, 100, "ADX+ должен быть от 0 до 100"),
        'adx_neg': (0, 100, "ADX- должен быть от 0 до 100"),
        'bb_position': (0, 1, "Bollinger Bands Position должен быть от 0 до 1"),
        'close_position': (0, 1, "Close Position должен быть от 0 до 1"),
        'psar_trend': (0, 1, "PSAR Trend должен быть 0 или 1"),
        'rsi_oversold': (0, 1, "RSI Oversold должен быть 0 или 1"),
        'rsi_overbought': (0, 1, "RSI Overbought должен быть 0 или 1"),
    }
    
    # Специальные проверки
    special_checks = {
        'close_vwap_ratio': (0.5, 1.5, "Close/VWAP ratio должен быть от 0.5 до 1.5"),
        'close_open_ratio': (0.8, 1.25, "Close/Open ratio должен быть от 0.8 до 1.25"),
        'high_low_ratio': (1.0, 2.0, "High/Low ratio должен быть от 1.0 до 2.0"),
        'toxicity': (0.5, 1.0, "Toxicity должен быть от 0.5 до 1.0"),
    }
    
    # Объединяем все проверки
    all_checks = {**technical_ranges, **special_checks}
    
    for indicator, (min_val, max_val, description) in all_checks.items():
        if indicator in df.columns:
            indicators_checked += 1
            stats = df[indicator].describe()
            
            # Проверка диапазона
            if stats['min'] < min_val or stats['max'] > max_val:
                print_error(f"{indicator}: выход за диапазон [{min_val}, {max_val}] -> [{stats['min']:.4f}, {stats['max']:.4f}]")
                issues.append(f"{indicator}_range_error")
            else:
                print_success(f"{indicator}: корректный диапазон [{stats['min']:.4f}, {stats['max']:.4f}]")
                indicators_passed += 1
            
            # ИСПРАВЛЕНО: Специальные проверки для конкретных индикаторов
            if indicator == 'toxicity':
                # Toxicity не должен быть константой около 1.0
                if stats['std'] < 0.01 and stats['mean'] > 0.99:
                    print_error(f"TOXICITY: подозрение на ошибку в формуле! Mean={stats['mean']:.6f}, Std={stats['std']:.6f}")
                    issues.append("toxicity_formula_error")
                elif 0.5 <= stats['mean'] <= 1.0 and stats['std'] > 0.01:
                    print_success(f"TOXICITY: корректные значения Mean={stats['mean']:.4f}, Std={stats['std']:.4f}")
            
            elif indicator == 'rsi':
                # RSI должен иметь разумное распределение
                if 20 <= stats['mean'] <= 80 and stats['std'] > 5:
                    print_success(f"RSI: здоровое распределение Mean={stats['mean']:.2f}, Std={stats['std']:.2f}")
                else:
                    print_warning(f"RSI: необычное распределение Mean={stats['mean']:.2f}, Std={stats['std']:.2f}")
            
            elif indicator in ['stoch_k', 'stoch_d']:
                # Stochastic должен иметь хорошую вариацию
                if stats['std'] > 15:
                    print_success(f"{indicator}: хорошая вариация Std={stats['std']:.2f}")
                else:
                    print_warning(f"{indicator}: низкая вариация Std={stats['std']:.2f}")
    
    # ИСПРАВЛЕНО: Проверка проблемных признаков с большими значениями
    problematic_features = ['bb_width', 'daily_range']
    for feature in problematic_features:
        if feature in df.columns:
            indicators_checked += 1
            stats = df[feature].describe()
            
            # bb_width и daily_range должны быть как процент от цены (обычно < 0.5)
            if stats['max'] > 1.0:  # Больше 100%
                print_error(f"{feature}: экстремально большие значения! Max={stats['max']:.2e}")
                print_info(f"   Рекомендация: пересчитать как процент от цены")
                issues.append(f"{feature}_extreme_values")
            elif stats['max'] > 0.5:  # Больше 50%
                print_warning(f"{feature}: большие значения Max={stats['max']:.4f}, возможно нужно ограничить")
                issues.append(f"{feature}_large_values")
            else:
                print_success(f"{feature}: разумные значения Max={stats['max']:.4f}")
                indicators_passed += 1
    
    # ИСПРАВЛЕНО: Проверка подозрительной нормализации
    normalization_suspects = [
        'price_direction', 'volume_zscore', 'momentum_1h', 'trend_1h_strength',
        'future_return_3', 'future_return_4', 'target_return_1h'
    ]
    allowed_zscore = {
        'price_direction', 'volume_zscore', 'momentum_1h',
        'trend_1h_strength', 'future_return_3', 'future_return_4',
        'target_return_1h'
    }
    
    print(f"\n{Colors.BOLD}Проверка подозрительной нормализации:{Colors.ENDC}")
    normalization_issues = 0
    
    for col in normalization_suspects:
        if col in df.columns:
            stats = df[col].describe()
            
            # Признаки неправильной нормализации:
            # 1. Mean очень близко к 0
            # 2. Std очень близко к 1
            # 3. Но это не должно быть технический индикатор
            is_normalized = (abs(stats['mean']) < 0.1 and 0.8 < stats['std'] < 1.2)
            
            if is_normalized and col not in allowed_zscore:
                print_error(f"{col}: ПОДОЗРЕНИЕ НА НОРМАЛИЗАЦИЮ! Mean={stats['mean']:.3f}, Std={stats['std']:.3f}")
                issues.append(f"{col}_normalized")
                normalization_issues += 1
            elif col in allowed_zscore and is_normalized:
                print_success(f"{col}: корректная Z-score нормализация Mean={stats['mean']:.3f}, Std={stats['std']:.3f}")
            else:
                print_success(f"{col}: нормальное распределение Mean={stats['mean']:.3f}, Std={stats['std']:.3f}")
    
    # Итоговая статистика
    print(f"\n{Colors.BOLD}📊 Итоговая статистика валидации индикаторов:{Colors.ENDC}")
    print_info(f"Проверено индикаторов: {indicators_checked}")
    print_info(f"Прошли проверку: {indicators_passed}")
    
    if len(issues) == 0:
        print_success("🎉 ВСЕ ИНДИКАТОРЫ КОРРЕКТНЫ!")
    else:
        critical_issues = len([i for i in issues if 'error' in i])
        warning_issues = len(issues) - critical_issues
        
        if critical_issues > 0:
            print_error(f"Критические ошибки: {critical_issues}")
        if warning_issues > 0:
            print_warning(f"Предупреждения: {warning_issues}")
        if normalization_issues > 0:
            print_error(f"Проблемы нормализации: {normalization_issues}")
    
    return issues

def check_advanced_issues(df, name):
    """Расширенная проверка проблем данных"""
    print(f"\n{Colors.BOLD}Расширенная диагностика {name}:{Colors.ENDC}")
    
    issues = []
    
    # 1. Проверка корреляций (признак переобучения)
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:20]  # Первые 20 для скорости
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        # Исключаем диагональ
        corr_matrix = corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool))
        high_corr = (corr_matrix > 0.95).sum().sum()
        if high_corr > 0:
            print_warning(f"Найдено {high_corr} пар признаков с корреляцией > 0.95")
            issues.append("high_correlation")
        else:
            print_success("Корреляции в норме")
    
    # 2. Проверка распределений (выбросы)
    extreme_features = []
    for col in numeric_cols:
        if col not in ['timestamp', 'volume', 'turnover']:
            q99 = df[col].quantile(0.99)
            q01 = df[col].quantile(0.01)
            if abs(q99) > 100 or abs(q01) > 100:
                extreme_features.append(f"{col} (Q99={q99:.2e}, Q01={q01:.2e})")
    
    if extreme_features:
        print_warning(f"Признаки с экстремальными значениями:")
        for feature in extreme_features[:5]:
            print(f"   - {feature}")
        issues.append("extreme_distributions")
    else:
        print_success("Распределения в разумных пределах")
    
    # 3. Проверка временной стабильности
    if 'datetime' in df.columns and len(df) > 1000:
        # Разделим на части и проверим стабильность статистик
        mid_point = len(df) // 2
        first_half = df.iloc[:mid_point]
        second_half = df.iloc[mid_point:]
        
        unstable_features = []
        for col in numeric_cols[:10]:  # Проверяем первые 10
            if col in ['timestamp', 'datetime']:
                continue
            
            mean1 = first_half[col].mean()
            mean2 = second_half[col].mean()
            
            if abs(mean1) > 1e-6 and abs(mean2) > 1e-6:  # Избегаем деления на ноль
                ratio = abs(mean1 / mean2) if mean2 != 0 else float('inf')
                if ratio > 2 or ratio < 0.5:
                    unstable_features.append(f"{col} (ratio={ratio:.2f})")
        
        if unstable_features:
            print_warning(f"Нестабильные во времени признаки:")
            for feature in unstable_features[:3]:
                print(f"   - {feature}")
            issues.append("temporal_instability")
        else:
            print_success("Временная стабильность в норме")
    
    return issues

def check_gpu_readiness(df, name):
    """Проверка готовности данных для GPU обучения"""
    print(f"\n{Colors.BOLD}Проверка GPU готовности {name}:{Colors.ENDC}")
    
    issues = []
    
    # 1. Проверка доступности CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_success(f"GPU доступно: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print_error("CUDA недоступно!")
        issues.append("no_cuda")
    
    # 2. Проверка размера данных
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    if memory_mb > 1000:  # > 1 GB
        print_warning(f"Большой размер данных: {memory_mb:.1f} MB")
        print_info("   Рекомендуется batch_size <= 64")
        issues.append("large_dataset")
    else:
        print_success(f"Размер данных: {memory_mb:.1f} MB (оптимален для GPU)")
    
    # 3. Проверка типов данных
    float64_cols = df.select_dtypes(include=['float64']).columns
    if len(float64_cols) > 0:
        print_warning(f"Найдено {len(float64_cols)} колонок с float64")
        print_info("   Рекомендуется преобразовать в float32 для GPU")
        issues.append("float64_types")
    else:
        print_success("Типы данных оптимизированы для GPU")
    
    # 4. Проверка на готовность к батчам
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    batch_ready = True
    for col in numeric_cols:
        if df[col].isna().any():
            print_error(f"Найдены NaN в {col} - недопустимо для GPU")
            batch_ready = False
            break
        if np.isinf(df[col]).any():
            print_error(f"Найдены Inf в {col} - недопустимо для GPU")
            batch_ready = False
            break
    
    if batch_ready:
        print_success("Данные готовы для GPU батчей")
    else:
        print_error("Данные НЕ готовы для GPU обучения")
        issues.append("not_batch_ready")
    
    return issues

def generate_detailed_report(all_issues, file_sizes):
    """ИСПРАВЛЕННАЯ генерация детального отчета с рекомендациями"""
    print_header("📊 ДЕТАЛЬНЫЙ ОТЧЕТ ДИАГНОСТИКИ")
    
    total_issues = sum(len(issues) for issues in all_issues.values())
    
    # Анализ типов проблем
    critical_issues = []
    normalization_issues = []
    formula_errors = []
    extreme_value_issues = []
    gpu_issues = []
    warnings = []
    
    for dataset, issues in all_issues.items():
        for issue in issues:
            if 'error' in issue or issue in ['not_batch_ready', 'no_cuda']:
                critical_issues.append(f"{dataset}: {issue}")
            elif 'normalized' in issue or 'normalization' in issue:
                normalization_issues.append(f"{dataset}: {issue}")
            elif 'formula' in issue or 'toxicity' in issue:
                formula_errors.append(f"{dataset}: {issue}")
            elif 'extreme' in issue or 'range_error' in issue:
                extreme_value_issues.append(f"{dataset}: {issue}")
            elif issue in ['large_dataset', 'float64_types']:
                gpu_issues.append(f"{dataset}: {issue}")
            else:
                warnings.append(f"{dataset}: {issue}")
    
    # КРИТИЧЕСКАЯ ДИАГНОСТИКА
    if total_issues == 0:
        print_success("🎉 ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!")
        print_success("✅ Данные полностью готовы к обучению модели")
        print_info("🚀 Можно запускать: python main.py --mode train")
    else:
        print_warning(f"⚠️ Обнаружено {total_issues} потенциальных проблем")
        
        # Детальный вывод по категориям
        if critical_issues:
            print(f"\n🔴 КРИТИЧЕСКИЕ ОШИБКИ ({len(critical_issues)}):")
            for issue in critical_issues:
                print_error(f"   {issue}")
        
        if formula_errors:
            print(f"\n🔥 ОШИБКИ В ФОРМУЛАХ ({len(formula_errors)}):")
            for issue in formula_errors:
                print_error(f"   {issue}")
            print_error("   ⚠️ НЕОБХОДИМО ИСПРАВИТЬ feature_engineering.py!")
        
        if normalization_issues:
            print(f"\n🟠 ПРОБЛЕМЫ НОРМАЛИЗАЦИИ ({len(normalization_issues)}):")
            for issue in normalization_issues:
                print_error(f"   {issue}")
            print_error("   ⚠️ Технические индикаторы не должны нормализоваться!")
        
        if extreme_value_issues:
            print(f"\n🟡 ЭКСТРЕМАЛЬНЫЕ ЗНАЧЕНИЯ ({len(extreme_value_issues)}):")
            for issue in extreme_value_issues:
                print_warning(f"   {issue}")
        
        if gpu_issues:
            print(f"\n🔵 GPU ОПТИМИЗАЦИЯ ({len(gpu_issues)}):")
            for issue in gpu_issues:
                print_warning(f"   {issue}")
        
        if warnings:
            print(f"\n⚪ ПРЕДУПРЕЖДЕНИЯ ({len(warnings)}):")
            for warning in warnings:
                print_warning(f"   {warning}")
    
    # ДЕТАЛЬНЫЕ РЕКОМЕНДАЦИИ ПО ИСПРАВЛЕНИЮ
    print(f"\n📋 ПОШАГОВЫЕ РЕКОМЕНДАЦИИ:")
    
    if formula_errors:
        print_error("   🔥 ПРИОРИТЕТ 1: Исправить формулы в feature_engineering.py")
        print_error("      - Toxicity: исправить формула должна давать диапазон 0.5-1.0")
        print_error("      - bb_width: считать как процент от цены, а не абсолютные значения") 
        print_error("      - daily_range: аналогично как процент от цены")
        print_error("      - После исправления: python main.py --mode data")
        
    elif normalization_issues:
        print_error("   🟠 ПРИОРИТЕТ 1: Исправить нормализацию")
        print_error("      - Добавить технические индикаторы в exclude_cols")
        print_error("      - RSI, Stochastic, ADX не должны нормализоваться")
        print_error("      - После исправления: python main.py --mode data")
        
    elif critical_issues:
        print_error("   🔴 ПРИОРИТЕТ 1: Устранить критические ошибки")
        print_error("      - Проверить диапазоны значений")
        print_error("      - Исправить NaN/Inf значения")
        print_error("      - После исправления повторить проверку")
        
    elif extreme_value_issues or gpu_issues:
        print_warning("   🟡 ПРИОРИТЕТ 2: Оптимизация")
        if extreme_value_issues:
            print_info("      - Добавить клиппинг экстремальных значений")
            print_info("      - Проверить формулы расчета больших признаков")
        if gpu_issues:
            print_info("      - Преобразовать float64 → float32")
            print_info("      - Уменьшить batch_size в config.yaml")
            
    elif warnings:
        print_info("   🔵 ПРИОРИТЕТ 3: Мелкие улучшения")
        print_info("      - Проанализировать предупреждения")
        print_info("      - При необходимости внести корректировки")
        
    else:
        print_success("   ✅ Всё готово! Можно запускать обучение")
    
    # СЛЕДУЮЩИЕ ШАГИ
    print(f"\n🚀 СЛЕДУЮЩИЕ ШАГИ:")
    if formula_errors or normalization_issues or critical_issues:
        print_info("   1. 🔧 Исправить feature_engineering.py согласно рекомендациям")
        print_info("   2. 🔄 python main.py --mode data  # Пересоздать кэш")
        print_info("   3. 🔍 python verify_data_correctness.py  # Повторная проверка")
        print_info("   4. 🚀 python main.py --mode train  # Запуск обучения")
    elif extreme_value_issues or gpu_issues:
        print_info("   1. 🎛️ Оптимизировать согласно рекомендациям")
        print_info("   2. 🔍 Повторить проверку (опционально)")
        print_info("   3. 🚀 python main.py --mode train")
    else:
        print_success("   🚀 python main.py --mode train")
    
    # СТАТИСТИКА ПО ФАЙЛАМ
    print(f"\n📁 ИНФОРМАЦИЯ О ФАЙЛАХ:")
    total_size = sum(file_sizes.values())
    for filename, size_mb in file_sizes.items():
        print_info(f"   {filename}: {size_mb:.2f} MB")
    print_info(f"   Общий размер: {total_size:.2f} MB")
    
    return total_issues


def main():
    """Основная функция расширенной проверки данных"""
    print_header("🔍 ПОЛНАЯ ДИАГНОСТИКА КОРРЕКТНОСТИ ДАННЫХ")
    print(f"⏰ Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🧠 Используется Enhanced Indicator Validator")
    print(f"🚀 GPU проверки включены")
    
    # 1. Проверка файлов
    files_exist, file_sizes = check_file_existence()
    if not files_exist:
        print_error("\n❌ Не все файлы найдены! Запустите: python main.py --mode data")
        return
    
    # Проверяем размеры файлов
    total_size = sum(file_sizes.values())
    print_info(f"📊 Общий размер данных: {total_size:.2f} MB")
    
    # 2. Загрузка и анализ данных
    print_header("📋 ЗАГРУЗКА И ДЕТАЛЬНЫЙ АНАЛИЗ ДАННЫХ")
    
    all_issues = {}
    
    for file_name in ['train_data.parquet', 'val_data.parquet', 'test_data.parquet']:
        print(f"\n{Colors.BOLD}{'='*70}{Colors.ENDC}")
        print(f"{Colors.BOLD}🔬 АНАЛИЗ {file_name.upper()}{Colors.ENDC}")
        print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}")
        
        file_path = Path('data/processed') / file_name
        try:
            df = pd.read_parquet(file_path)
            print_success(f"Загружено: {len(df):,} записей, {len(df.columns)} колонок")
        except Exception as e:
            print_error(f"Ошибка загрузки {file_name}: {e}")
            continue
        
        dataset_name = file_name.split('_')[0].upper()
        issues = []
        
        # Основные проверки
        if not check_data_structure(df, dataset_name):
            issues.append("structure_error")
        
        issues.extend(check_data_quality(df, dataset_name))
        issues.extend(check_critical_indicators(df, dataset_name))
        issues.extend(check_target_distribution(df, dataset_name))
        issues.extend(check_temporal_consistency(df, dataset_name))
        
        # 🆕 РАСШИРЕННЫЕ ПРОВЕРКИ
        issues.extend(check_advanced_issues(df, dataset_name))
        issues.extend(check_gpu_readiness(df, dataset_name))
        
        all_issues[dataset_name] = issues
        
        # Краткая сводка по датасету
        if issues:
            print_warning(f"⚠️ {dataset_name}: {len(issues)} проблем обнаружено")
        else:
            print_success(f"✅ {dataset_name}: Все проверки пройдены!")
    
    # 3. Финальный отчет с детальными рекомендациями
    total_issues = generate_detailed_report(all_issues, file_sizes)
    
    # 4. Сохранение детального отчета
    try:
        # Создаем детальный отчет через валидатор
        validator = IndicatorValidator()
        sample_df = pd.read_parquet(Path('data/processed/train_data.parquet'))
        detailed_report = validator.create_validation_report(sample_df)
        
        report_path = Path('logs/data_validation_report.txt')
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(detailed_report)
        
        print_info(f"\n📄 Детальный отчет сохранен: {report_path}")
        
    except Exception as e:
        print_warning(f"Не удалось создать детальный отчет: {e}")
    
    print(f"\n⏰ Проверка завершена: {datetime.now().strftime('%H:%M:%S')}")

def create_validation_report(df, filename):
    """Создание детального отчета валидации в файл"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("📊 ОТЧЕТ ВАЛИДАЦИИ ТЕХНИЧЕСКИХ ИНДИКАТОРОВ")
    report_lines.append("=" * 80)
    report_lines.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Размер данных: {len(df):,} записей, {len(df.columns)} колонок")
    report_lines.append("")
    
    # Общая статистика
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    report_lines.append("📈 ОБЩАЯ СТАТИСТИКА:")
    report_lines.append(f"   - Числовых колонок: {len(numeric_cols)}")
    report_lines.append(f"   - NaN значений: {df.isna().sum().sum()}")
    report_lines.append(f"   - Inf значений: {sum(np.isinf(df[col]).sum() for col in numeric_cols)}")
    report_lines.append("")
    
    # Детальная статистика по индикаторам
    indicators = ['rsi', 'stoch_k', 'stoch_d', 'adx', 'adx_pos', 'adx_neg', 
                 'toxicity', 'bb_position', 'close_position', 'psar_trend',
                 'rsi_oversold', 'rsi_overbought', 'close_vwap_ratio', 
                 'close_open_ratio', 'high_low_ratio', 'bb_width', 'daily_range']
    
    report_lines.append("📊 ДЕТАЛЬНАЯ СТАТИСТИКА ПО ИНДИКАТОРАМ:")
    for indicator in indicators:
        if indicator in df.columns:
            stats = df[indicator].describe()
            report_lines.append(f"   {indicator}:")
            report_lines.append(f"      Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
            report_lines.append(f"      Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            report_lines.append(f"      Записей: {len(df[indicator]):,}")
    
    # Сохранение отчета
    report_path = Path('logs') / filename
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print_info(f"📄 Детальный отчет сохранен: {report_path}")

if __name__ == "__main__":
    main()