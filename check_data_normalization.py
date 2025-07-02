#!/usr/bin/env python3
"""
Проверка и нормализация кэшированных данных
"""

import pandas as pd
import numpy as np
from pathlib import Path
from utils.logger import get_logger

def check_data_stats():
    """Проверка статистики кэшированных данных"""
    logger = get_logger("DataCheck")
    
    logger.info("🔍 Проверка кэшированных данных...")
    
    # Пути к файлам
    data_dir = Path("data/processed")
    files = {
        "train": data_dir / "train_data.parquet",
        "val": data_dir / "val_data.parquet",
        "test": data_dir / "test_data.parquet"
    }
    
    for name, file_path in files.items():
        if not file_path.exists():
            logger.error(f"❌ Файл {file_path} не найден!")
            continue
            
        logger.info(f"\n📊 Анализ {name} данных:")
        df = pd.read_parquet(file_path)
        
        # Общая информация
        logger.info(f"  Размер: {df.shape}")
        logger.info(f"  Память: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Проверка на NaN
        nan_counts = df.isna().sum()
        if nan_counts.any():
            logger.warning(f"  ⚠️ Найдены NaN значения:")
            for col in nan_counts[nan_counts > 0].index:
                logger.warning(f"    - {col}: {nan_counts[col]} NaN")
        
        # Проверка на бесконечные значения
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_counts = np.isinf(df[numeric_cols]).sum()
        if inf_counts.any():
            logger.warning(f"  ⚠️ Найдены inf значения:")
            for col in inf_counts[inf_counts > 0].index:
                logger.warning(f"    - {col}: {inf_counts[col]} inf")
        
        # Статистика по числовым колонкам
        logger.info("\n  📈 Статистика по признакам:")
        
        # Отдельно для признаков и целевых переменных
        feature_cols = [col for col in numeric_cols if not any(
            x in col for x in ['future_return', 'long_', 'short_', 'best_direction', 'target_return']
        )]
        target_cols = [col for col in numeric_cols if any(
            x in col for x in ['future_return', 'long_', 'short_', 'best_direction', 'target_return']
        )]
        
        # Статистика признаков
        if feature_cols:
            feature_stats = df[feature_cols].describe()
            logger.info(f"\n  Признаки ({len(feature_cols)} колонок):")
            logger.info(f"    Min значения: {feature_stats.loc['min'].min():.4f}")
            logger.info(f"    Max значения: {feature_stats.loc['max'].max():.4f}")
            logger.info(f"    Среднее: {feature_stats.loc['mean'].mean():.4f}")
            logger.info(f"    Std: {feature_stats.loc['std'].mean():.4f}")
            
            # Проверка на очень большие значения
            large_values = (df[feature_cols].abs() > 1000).any()
            if large_values.any():
                logger.warning(f"\n  ⚠️ Колонки с очень большими значениями (>1000):")
                for col in large_values[large_values].index:
                    max_val = df[col].abs().max()
                    logger.warning(f"    - {col}: max={max_val:.2f}")
        
        # Статистика целевых переменных
        if target_cols:
            target_stats = df[target_cols].describe()
            logger.info(f"\n  Целевые переменные ({len(target_cols)} колонок):")
            logger.info(f"    Min значения: {target_stats.loc['min'].min():.4f}")
            logger.info(f"    Max значения: {target_stats.loc['max'].max():.4f}")
            logger.info(f"    Среднее: {target_stats.loc['mean'].mean():.4f}")
            logger.info(f"    Std: {target_stats.loc['std'].mean():.4f}")
            
        logger.info("=" * 80)

def normalize_cached_data():
    """Нормализация кэшированных данных если необходимо"""
    logger = get_logger("DataNormalize")
    
    logger.info("🔧 Проверка необходимости нормализации...")
    
    # Проверяем только train данные для решения
    train_file = Path("data/processed/train_data.parquet")
    if not train_file.exists():
        logger.error("❌ Train файл не найден!")
        return
        
    df = pd.read_parquet(train_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Проверяем максимальные значения
    max_values = df[numeric_cols].abs().max()
    needs_normalization = (max_values > 1000).any()
    
    if needs_normalization:
        logger.warning("⚠️ Обнаружены очень большие значения, рекомендуется пересоздать датасет!")
        logger.info("Запустите: python prepare_trading_data.py --force-recreate")
    else:
        logger.info("✅ Данные в нормальных пределах, нормализация не требуется")
        
    # Проверка на NaN/Inf
    has_nan = df[numeric_cols].isna().any().any()
    has_inf = np.isinf(df[numeric_cols]).any().any()
    
    if has_nan or has_inf:
        logger.warning("⚠️ Обнаружены NaN или Inf значения!")
        logger.info("Рекомендуется пересоздать датасет или очистить данные")

if __name__ == "__main__":
    # Проверка статистики
    check_data_stats()
    
    # Проверка необходимости нормализации
    print()
    normalize_cached_data()