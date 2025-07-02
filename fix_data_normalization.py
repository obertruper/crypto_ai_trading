#!/usr/bin/env python3
"""
Исправление и нормализация кэшированных данных
"""

import pandas as pd
import numpy as np
from pathlib import Path
from utils.logger import get_logger
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

def fix_cached_data():
    """Исправляет проблемы с кэшированными данными"""
    logger = get_logger("DataFix")
    
    logger.info("🔧 Исправление кэшированных данных...")
    
    # Пути к файлам
    data_dir = Path("data/processed")
    files = {
        "train": data_dir / "train_data.parquet",
        "val": data_dir / "val_data.parquet", 
        "test": data_dir / "test_data.parquet"
    }
    
    # Колонки, которые не нужно нормализовать
    skip_columns = ['symbol', 'datetime', 'id']
    
    # Колонки с проблемными значениями
    problematic_columns = [
        'close_vwap_ratio', 'amihud_illiquidity', 'amihud_ma',
        'volume_volatility_ratio', 'liquidity_score'
    ]
    
    for name, file_path in files.items():
        if not file_path.exists():
            logger.error(f"❌ Файл {file_path} не найден!")
            continue
            
        logger.info(f"\n📊 Обработка {name} данных...")
        df = pd.read_parquet(file_path)
        original_shape = df.shape
        
        # 1. Замена inf на NaN
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        logger.info("  Замена inf значений...")
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # 2. Обработка проблемных колонок
        for col in problematic_columns:
            if col in df.columns:
                logger.info(f"  Обработка {col}...")
                # Заменяем экстремальные значения на медиану
                median_val = df[col].median()
                std_val = df[col].std()
                upper_bound = median_val + 10 * std_val
                lower_bound = median_val - 10 * std_val
                
                # Клиппинг экстремальных значений
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
                # Заполнение NaN медианой
                df[col] = df[col].fillna(median_val)
        
        # 3. Обработка остальных числовых колонок
        for col in numeric_cols:
            if col in skip_columns or col in problematic_columns:
                continue
                
            # Проверка на большие значения
            if df[col].abs().max() > 10000:
                logger.info(f"  Нормализация {col} (max={df[col].abs().max():.2f})...")
                
                # Для цен и объемов используем log-трансформацию
                if any(x in col for x in ['price', 'volume', 'turnover', 'open', 'high', 'low', 'close']):
                    # Log transform для положительных значений
                    mask = df[col] > 0
                    df.loc[mask, col] = np.log1p(df.loc[mask, col])
                else:
                    # Для остальных - клиппинг на основе персентилей
                    p1 = df[col].quantile(0.001)
                    p99 = df[col].quantile(0.999)
                    df[col] = df[col].clip(lower=p1, upper=p99)
        
        # 4. Финальная проверка и заполнение NaN
        logger.info("  Финальная обработка NaN...")
        for col in numeric_cols:
            if col in skip_columns:
                continue
            
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                # Заполняем нулями для технических индикаторов
                if any(x in col for x in ['rsi', 'macd', 'bb_', 'ema', 'sma']):
                    df[col] = df[col].fillna(0)
                else:
                    # Для остальных используем forward fill затем backward fill
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 5. Нормализация признаков (кроме целевых)
        logger.info("  Применение RobustScaler к признакам...")
        
        # Определяем признаки и целевые переменные
        feature_cols = [col for col in numeric_cols if not any(
            x in col for x in ['future_return', 'long_', 'short_', 'best_direction', 'target_return']
        ) and col not in skip_columns]
        
        if feature_cols:
            scaler = RobustScaler()
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
        
        # 6. Проверка результата
        logger.info("  Проверка результата...")
        max_val = df[numeric_cols].abs().max().max()
        nan_count = df[numeric_cols].isna().sum().sum()
        
        logger.info(f"  ✅ Обработано: {original_shape} -> {df.shape}")
        logger.info(f"  ✅ Max значение: {max_val:.4f}")
        logger.info(f"  ✅ NaN значений: {nan_count}")
        
        # Сохранение с backup
        backup_path = file_path.with_suffix('.parquet.backup')
        logger.info(f"  💾 Создание backup: {backup_path}")
        
        # Сохраняем оригинал как backup если его еще нет
        if not backup_path.exists():
            import shutil
            shutil.copy(file_path, backup_path)
        
        # Сохраняем исправленные данные
        logger.info(f"  💾 Сохранение исправленных данных...")
        df.to_parquet(file_path, index=False)
        
        logger.info(f"  ✅ {name} данные успешно обработаны!")
    
    logger.info("\n✅ Все данные обработаны!")
    logger.info("Теперь можно запускать обучение: python main.py --mode train")

if __name__ == "__main__":
    fix_cached_data()