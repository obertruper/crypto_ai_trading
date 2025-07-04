#!/usr/bin/env python3
"""
Компактная проверка корректности данных - только самое важное
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")

def print_success(text):
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKBLUE}ℹ️  {text}{Colors.ENDC}")

def check_critical_indicators(df, name):
    """Проверка только критических индикаторов"""
    print(f"\n{Colors.BOLD}🔍 Критические индикаторы {name}:{Colors.ENDC}")
    
    issues = []
    
    # 1. TOXICITY - самый важный индикатор
    if 'toxicity' in df.columns:
        stats = df['toxicity'].describe()
        mean = stats['mean']
        std = stats['std']
        
        if mean > 0.99 and std < 0.02:
            print_error(f"toxicity: mean={mean:.6f}, std={std:.6f}")
            print_error("   🚨 КРИТИЧНО: toxicity всегда ≈1.0 (бесполезный индикатор!)")
            issues.append("toxicity_broken")
        elif 0.4 <= mean <= 0.7 and std > 0.15:
            print_success(f"toxicity: mean={mean:.4f}, std={std:.4f} ✓ (корректное распределение)")
        else:
            print_warning(f"toxicity: mean={mean:.4f}, std={std:.4f}")
            if std < 0.1:
                print_warning("   ⚠️ Низкая вариативность toxicity")
    
    # 2. PRICE_IMPACT
    if 'price_impact' in df.columns:
        stats = df['price_impact'].describe()
        mean = stats['mean']
        
        if mean < 0.0001:
            print_error(f"price_impact: mean={mean:.6f} (слишком мал!)")
            issues.append("price_impact_too_small")
        elif 0.001 <= mean <= 0.1:
            print_success(f"price_impact: mean={mean:.4f} ✓")
        else:
            print_warning(f"price_impact: mean={mean:.4f} (необычно высокий)")
    
    # 3. RSI - проверка на нормализацию
    if 'rsi' in df.columns:
        stats = df['rsi'].describe()
        mean = stats['mean']
        std = stats['std']
        
        # Признаки нормализации
        if abs(mean) < 1.0 and 0.8 < std < 1.2:
            print_error(f"rsi: НОРМАЛИЗОВАН! mean={mean:.3f}, std={std:.3f}")
            issues.append("rsi_normalized")
        elif 40 <= mean <= 60 and std > 10:
            print_success(f"rsi: mean={mean:.1f}, std={std:.1f} ✓")
        else:
            print_warning(f"rsi: mean={mean:.1f}, std={std:.1f}")
    
    # 4. Stochastic
    for indicator in ['stoch_k', 'stoch_d']:
        if indicator in df.columns:
            stats = df[indicator].describe()
            mean = stats['mean']
            std = stats['std']
            
            if abs(mean) < 1.0 and 0.8 < std < 1.2:
                print_error(f"{indicator}: НОРМАЛИЗОВАН! mean={mean:.3f}, std={std:.3f}")
                issues.append(f"{indicator}_normalized")
            elif 30 <= mean <= 70 and std > 15:
                print_success(f"{indicator}: mean={mean:.1f}, std={std:.1f} ✓")
    
    return issues

def check_target_distribution(df, name):
    """Проверка целевых переменных"""
    print(f"\n{Colors.BOLD}🎯 Целевые переменные {name}:{Colors.ENDC}")
    
    # TP/SL достижения
    for target in ['long_tp1_reached', 'short_tp1_reached']:
        if target in df.columns:
            pct = df[target].mean() * 100
            if 30 <= pct <= 70:
                print_info(f"{target}: {pct:.1f}% ✓")
            else:
                print_warning(f"{target}: {pct:.1f}%")
    
    # Направление
    if 'best_direction' in df.columns:
        dist = df['best_direction'].value_counts(normalize=True) * 100
        print_info(f"best_direction: LONG={dist.get('LONG', 0):.1f}%, SHORT={dist.get('SHORT', 0):.1f}%, NEUTRAL={dist.get('NEUTRAL', 0):.1f}%")

def check_data_quality(df, name):
    """Базовая проверка качества"""
    issues = []
    
    # NaN проверка
    nan_cols = df.isna().sum()
    nan_cols = nan_cols[nan_cols > 0]
    if len(nan_cols) > 0:
        worst_col = nan_cols.idxmax()
        worst_pct = nan_cols.max() / len(df) * 100
        print_warning(f"NaN в {len(nan_cols)} колонках (худшая: {worst_col} = {worst_pct:.1f}%)")
        if worst_pct > 5:
            issues.append("high_nan")
    
    return issues

def check_ml_readiness(df, name):
    """Проверка готовности данных для ML"""
    print(f"\n{Colors.BOLD}🤖 Проверки для ML {name}:{Colors.ENDC}")
    
    issues = []
    
    # Исключаем служебные и целевые колонки
    exclude_cols = ['id', 'symbol', 'datetime', 'timestamp', 'sector']
    target_prefixes = ('target_', 'future_', 'long_', 'short_', 'best_')
    
    # ИСПРАВЛЕНО: Исключаем исходные данные (цены, объемы) из проверки экстремальных значений
    # Эти колонки могут иметь большие значения и это нормально
    raw_data_cols = [
        'open', 'high', 'low', 'close', 'volume', 'turnover',
        'vwap', 'btc_close', 'dollar_volume', 'directed_volume',
        # Скользящие средние и другие ценовые индикаторы
        'sma_10', 'sma_20', 'sma_50', 'ema_10', 'ema_20', 'ema_50',
        'bb_high', 'bb_low', 'bb_middle', 'psar',
        # EMA для MACD (могут быть большие для дорогих активов)
        'ema_12', 'ema_26',
        # Локальные экстремумы
        'local_high_20', 'local_high_50', 'local_high_100',
        'local_low_20', 'local_low_50', 'local_low_100',
        'daily_high', 'daily_low',
        # Ликвидационные цены
        'long_liquidation_price', 'short_liquidation_price',
        # ATR в абсолютных ценах (не в процентах)
        'atr',
        # Тренды в абсолютных значениях
        'trend_1h', 'trend_4h'
    ]
    
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols
                   and not col.startswith(target_prefixes)
                   and col not in raw_data_cols]
    
    # 1. Проверка экстремальных значений (исключая цены и объемы)
    extreme_cols = []
    for col in feature_cols:
        if df[col].dtype in ['float32', 'float64', 'int32', 'int64']:
            max_val = df[col].abs().max()
            if max_val > 1000:
                extreme_cols.append((col, max_val))
    
    if extreme_cols:
        print_error(f"Экстремальные значения (>1000) в {len(extreme_cols)} признаках!")
        for col, val in sorted(extreme_cols, key=lambda x: x[1], reverse=True)[:5]:
            print_error(f"   - {col}: {val:.2e}")
        issues.append("extreme_values")
    else:
        print_success("Нет экстремальных значений в признаках (исключая цены/объемы)")
    
    # 2. Проверка распределения признаков
    zero_variance_cols = []
    for col in feature_cols[:50]:  # Проверяем первые 50
        std = df[col].std()
        if std < 1e-6:
            zero_variance_cols.append(col)
    
    if zero_variance_cols:
        print_warning(f"Нулевая дисперсия в {len(zero_variance_cols)} колонках")
        issues.append("zero_variance")
    else:
        print_success("Все признаки имеют достаточную дисперсию")
    
    # 3. Проверка корреляций между признаками
    if len(feature_cols) > 10:
        corr_matrix = df[feature_cols[:20]].corr().abs()
        high_corr_pairs = []
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                if corr_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append((corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        if high_corr_pairs:
            print_warning(f"Высокая корреляция (>0.95) между {len(high_corr_pairs)} парами признаков")
        else:
            print_success("Нет сильно коррелированных признаков")
    
    # 4. Проверка баланса целевых переменных
    for direction in ['long', 'short']:
        tp1_col = f'{direction}_tp1_reached'
        if tp1_col in df.columns:
            positive_pct = df[tp1_col].mean() * 100
            if positive_pct < 5 or positive_pct > 95:
                print_error(f"{tp1_col}: {positive_pct:.1f}% - сильный дисбаланс!")
                issues.append("target_imbalance")
    
    # 5. Проверка типов данных
    non_numeric = []
    for col in feature_cols:
        if df[col].dtype == 'object':
            non_numeric.append(col)
    
    if non_numeric:
        print_error(f"Не числовые типы в {len(non_numeric)} колонках: {non_numeric[:5]}")
        issues.append("non_numeric_features")
    else:
        print_success("Все признаки имеют числовой тип")
    
    # 6. Проверка inf значений
    inf_cols = []
    for col in feature_cols:
        if np.isinf(df[col]).any():
            inf_cols.append(col)
    
    if inf_cols:
        print_error(f"Бесконечные значения в {len(inf_cols)} колонках: {inf_cols[:5]}")
        issues.append("inf_values")
    else:
        print_success("Нет бесконечных значений")
    
    return issues

def main():
    """Компактная проверка данных"""
    print_header("📊 ПРОВЕРКА КОРРЕКТНОСТИ ДАННЫХ")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    files = ['train_data.parquet', 'val_data.parquet', 'test_data.parquet']
    all_issues = []
    
    for file in files:
        path = Path(f'data/processed/{file}')
        if not path.exists():
            print_error(f"\n❌ {file} НЕ НАЙДЕН!")
            all_issues.append("file_missing")
            continue
            
        print(f"\n{Colors.OKBLUE}{Colors.BOLD}{'='*50}{Colors.ENDC}")
        print(f"{Colors.OKBLUE}{Colors.BOLD}📁 {file.upper()}{Colors.ENDC}")
        print(f"{Colors.OKBLUE}{Colors.BOLD}{'='*50}{Colors.ENDC}")
        
        df = pd.read_parquet(path)
        print_info(f"Размер: {len(df):,} записей, {len(df.columns)} колонок")
        
        # Проверки
        dataset_name = file.split('_')[0].upper()
        issues = []
        
        issues.extend(check_critical_indicators(df, dataset_name))
        issues.extend(check_data_quality(df, dataset_name))
        check_target_distribution(df, dataset_name)
        issues.extend(check_ml_readiness(df, dataset_name))
        
        # Итог по файлу
        if not issues:
            print_success(f"\n✅ {dataset_name} - данные корректны!")
        else:
            print_error(f"\n❌ {dataset_name} - обнаружено {len(issues)} проблем!")
            all_issues.extend(issues)
    
    # ФИНАЛЬНЫЙ ИТОГ
    print_header("📋 ИТОГОВЫЙ РЕЗУЛЬТАТ")
    
    if not all_issues:
        print_success("✅ ВСЕ ДАННЫЕ ГОТОВЫ К ОБУЧЕНИЮ!")
        print_success("\n🚀 Запускайте: python main.py --mode train")
    else:
        unique_issues = set(all_issues)
        print_error(f"❌ ОБНАРУЖЕНО ПРОБЛЕМ: {len(unique_issues)}")
        
        # Критические проблемы
        if 'toxicity_broken' in unique_issues:
            print_error("\n🔥 КРИТИЧНО: Индикатор toxicity не работает!")
            print_warning("   Формула в feature_engineering.py уже исправлена")
            print_warning("   Нужно пересоздать данные")
        
        if any('normalized' in issue for issue in unique_issues):
            print_error("\n🔥 КРИТИЧНО: Обнаружена нежелательная нормализация!")
            print_warning("   Технические индикаторы не должны нормализоваться")
        
        if 'extreme_values' in unique_issues:
            print_error("\n🔥 КРИТИЧНО: Экстремальные значения в признаках!")
            print_warning("   Требуется нормализация данных")
        
        if 'target_imbalance' in unique_issues:
            print_warning("\n⚠️ ВНИМАНИЕ: Дисбаланс в целевых переменных!")
            print_info("   Может потребоваться балансировка классов")
        
        # Инструкции
        print(f"\n{Colors.WARNING}{Colors.BOLD}🔧 НЕОБХОДИМЫЕ ДЕЙСТВИЯ:{Colors.ENDC}")
        print_info("1. Очистить кэш:")
        print(f"   {Colors.OKBLUE}rm -rf cache/features/*{Colors.ENDC}")
        print_info("2. Пересоздать данные:")
        print(f"   {Colors.OKBLUE}python prepare_trading_data.py --force-recreate{Colors.ENDC}")
        print_info("3. Проверить результат:")
        print(f"   {Colors.OKBLUE}python verify_data_correctness.py{Colors.ENDC}")
    
    # Автоматически сохраняем отчет без запроса
    create_detailed_report(files)

def create_detailed_report(files):
    """Создание детального отчета для логов с полным описанием всех признаков"""
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("📊 ДЕТАЛЬНЫЙ ОТЧЕТ ВАЛИДАЦИИ ДАННЫХ")
    report_lines.append("="*80)
    report_lines.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Для сбора полной статистики
    all_features = set()
    
    for file in files:
        path = Path(f'data/processed/{file}')
        if path.exists():
            df = pd.read_parquet(path)
            report_lines.append(f"\n{'='*80}")
            report_lines.append(f"📁 {file.upper()}")
            report_lines.append(f"{'='*80}")
            report_lines.append(f"Размер: {len(df):,} записей, {len(df.columns)} колонок")
            report_lines.append(f"Период: {df['datetime'].min()} - {df['datetime'].max()}")
            report_lines.append(f"Символов: {df['symbol'].nunique()}")
            
            # 1. МИКРОСТРУКТУРНЫЕ ПРИЗНАКИ
            report_lines.append("\n📈 МИКРОСТРУКТУРНЫЕ ПРИЗНАКИ:")
            microstructure_features = ['toxicity', 'price_impact', 'price_impact_log', 'amihud_illiquidity', 
                                     'kyle_lambda', 'realized_vol', 'hl_spread', 'volume_imbalance']
            for feature in microstructure_features:
                if feature in df.columns:
                    stats = df[feature].describe()
                    report_lines.append(f"\n  {feature}:")
                    report_lines.append(f"    Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")
                    report_lines.append(f"    Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
                    report_lines.append(f"    25%: {stats['25%']:.6f}, 50%: {stats['50%']:.6f}, 75%: {stats['75%']:.6f}")
                    
                    # Специальные проверки
                    if feature == 'toxicity':
                        if stats['mean'] > 0.99:
                            report_lines.append("    ⚠️ ПРОБЛЕМА: toxicity слишком высокий!")
                        else:
                            report_lines.append("    ✅ Корректное распределение")
            
            # 2. ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ
            report_lines.append("\n📊 ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ:")
            technical_indicators = ['rsi', 'stoch_k', 'stoch_d', 'adx', 'adx_pos', 'adx_neg',
                                  'macd', 'macd_signal', 'macd_diff', 'bb_width', 'bb_position',
                                  'atr', 'atr_pct', 'psar_trend', 'close_position']
            for indicator in technical_indicators:
                if indicator in df.columns:
                    stats = df[indicator].describe()
                    report_lines.append(f"\n  {indicator}:")
                    report_lines.append(f"    Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
                    report_lines.append(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
                    
                    # Проверка на нормализацию
                    if indicator in ['rsi', 'stoch_k', 'stoch_d', 'adx'] and abs(stats['mean']) < 1.0 and 0.8 < stats['std'] < 1.2:
                        report_lines.append("    ⚠️ ПРОБЛЕМА: Подозрение на нормализацию!")
            
            # 3. RALLY DETECTION ПРИЗНАКИ
            report_lines.append("\n🚀 RALLY DETECTION ПРИЗНАКИ:")
            rally_features = ['volume_cumsum_4h', 'volume_cumsum_24h', 'volume_spike', 'spring_pattern',
                            'momentum_1h', 'momentum_4h', 'momentum_24h', 'momentum_acceleration']
            for feature in rally_features:
                if feature in df.columns:
                    stats = df[feature].describe()
                    report_lines.append(f"\n  {feature}:")
                    report_lines.append(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
                    if 'spike' in feature or 'pattern' in feature:
                        positive_pct = (df[feature] > 0).mean() * 100
                        report_lines.append(f"    Положительных: {positive_pct:.1f}%")
            
            # 4. SIGNAL QUALITY ПРИЗНАКИ
            report_lines.append("\n📡 SIGNAL QUALITY ПРИЗНАКИ:")
            signal_features = ['indicators_consensus_long', 'indicators_consensus_short',
                             'trend_1h_strength', 'trend_4h_strength', 'liquidity_score']
            for feature in signal_features:
                if feature in df.columns:
                    stats = df[feature].describe()
                    report_lines.append(f"\n  {feature}:")
                    report_lines.append(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            
            # 5. FUTURES SPECIFIC ПРИЗНАКИ
            report_lines.append("\n💰 FUTURES SPECIFIC ПРИЗНАКИ:")
            futures_features = ['long_liquidation_distance_pct', 'short_liquidation_distance_pct',
                              'optimal_leverage', 'safe_leverage', 'var_95']
            for feature in futures_features:
                if feature in df.columns:
                    stats = df[feature].describe()
                    report_lines.append(f"\n  {feature}:")
                    report_lines.append(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            
            # 6. ЦЕЛЕВЫЕ ПЕРЕМЕННЫЕ
            report_lines.append("\n🎯 ЦЕЛЕВЫЕ ПЕРЕМЕННЫЕ:")
            
            # Бинарные целевые
            for direction in ['long', 'short']:
                for level in ['tp1', 'tp2', 'tp3', 'sl']:
                    target = f'{direction}_{level}_reached'
                    if target in df.columns:
                        pct = df[target].mean() * 100
                        report_lines.append(f"  {target}: {pct:.2f}%")
            
            # Expected values
            for ev in ['long_expected_value', 'short_expected_value']:
                if ev in df.columns:
                    stats = df[ev].describe()
                    report_lines.append(f"\n  {ev}:")
                    report_lines.append(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
                    report_lines.append(f"    Положительных: {(df[ev] > 0).mean() * 100:.1f}%")
            
            # Best direction
            if 'best_direction' in df.columns:
                report_lines.append(f"\n  best_direction распределение:")
                dist = df['best_direction'].value_counts()
                for direction, count in dist.items():
                    pct = count / len(df) * 100
                    report_lines.append(f"    {direction}: {count:,} ({pct:.1f}%)")
            
            # Signal strength
            if 'signal_strength' in df.columns:
                stats = df['signal_strength'].describe()
                report_lines.append(f"\n  signal_strength:")
                report_lines.append(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
                report_lines.append(f"    Max: {stats['max']:.4f}")
            
            # 7. ПРОВЕРКА NAN ЗНАЧЕНИЙ
            report_lines.append("\n⚠️ ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ:")
            nan_cols = df.isna().sum()
            nan_cols = nan_cols[nan_cols > 0].sort_values(ascending=False)
            if len(nan_cols) > 0:
                for col, count in nan_cols.head(10).items():
                    pct = count / len(df) * 100
                    report_lines.append(f"  {col}: {count:,} ({pct:.2f}%)")
            else:
                report_lines.append("  ✅ Нет пропущенных значений")
            
            # 8. ИТОГОВАЯ СТАТИСТИКА
            report_lines.append(f"\n📊 ИТОГОВАЯ СТАТИСТИКА {file.split('_')[0].upper()}:")
            
            # Подсчет признаков по категориям
            feature_categories = {
                'Базовые': ['returns', 'volume_ratio', 'high_low_ratio', 'close_open_ratio'],
                'Технические': [col for col in df.columns if any(ind in col for ind in ['sma', 'ema', 'rsi', 'macd', 'bb_', 'stoch'])],
                'Микроструктура': [col for col in df.columns if any(ms in col for ms in ['toxicity', 'impact', 'illiquidity', 'spread'])],
                'Rally detection': [col for col in df.columns if any(rd in col for rd in ['volume_cumsum', 'momentum', 'spring', 'divergence'])],
                'Signal quality': [col for col in df.columns if any(sq in col for sq in ['consensus', 'trend_strength', 'liquidity_score'])],
                'Futures': [col for col in df.columns if any(f in col for f in ['liquidation', 'leverage', 'var_', 'funding'])],
                'Временные': [col for col in df.columns if any(t in col for t in ['hour', 'day', 'month', 'session'])],
                'Cross-asset': [col for col in df.columns if any(ca in col for ca in ['btc_', 'sector', 'relative_'])],
                'Целевые': [col for col in df.columns if any(tgt in col for tgt in ['target_', 'future_', '_reached', 'expected_value', 'best_direction'])]
            }
            
            for category, features in feature_categories.items():
                count = len([f for f in features if f in df.columns])
                if count > 0:
                    report_lines.append(f"  {category}: {count} признаков")
            
            report_lines.append(f"  ВСЕГО: {len(df.columns)} колонок")
    
    report_lines.append("\n" + "="*80)
    report_lines.append("✅ ОТЧЕТ ЗАВЕРШЕН")
    report_lines.append("="*80)
    
    # Сохранение
    report_path = Path('logs/data_validation_report.txt')
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print_info(f"\n📄 Детальный отчет сохранен: {report_path}")

if __name__ == "__main__":
    main()