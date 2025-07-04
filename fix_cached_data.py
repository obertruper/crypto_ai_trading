#!/usr/bin/env python3
"""
Скрипт для исправления и нормализации кэшированных данных
Применяет нормализацию к существующим parquet файлам
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler
import pickle
from tqdm import tqdm
import argparse

from utils.logger import get_logger
from data.constants import get_feature_columns


def identify_problematic_columns(df: pd.DataFrame) -> dict:
    """Определение колонок с экстремальными значениями"""
    logger = get_logger("DataAnalysis")
    
    problematic_cols = {
        'extreme_values': [],  # > 1e9
        'volume_based': [],
        'price_based': [],
        'ratio_based': []
    }
    
    # Пропускаем служебные колонки
    service_cols = ['id', 'symbol', 'datetime', 'timestamp', 'sector']
    feature_cols = [col for col in df.columns if col not in service_cols]
    
    for col in feature_cols:
        if col not in df.columns:
            continue
            
        # Проверка на экстремальные значения
        max_val = df[col].max()
        if pd.notna(max_val) and abs(max_val) > 1e9:
            problematic_cols['extreme_values'].append((col, max_val))
        
        # Классификация колонок
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in ['volume', 'turnover', 'obv', 'liquidity', 'cmf', 'mfi']):
            problematic_cols['volume_based'].append(col)
        elif any(pattern in col_lower for pattern in ['price', 'vwap', 'high', 'low', 'open', 'close']):
            problematic_cols['price_based'].append(col)
        elif any(pattern in col_lower for pattern in ['ratio', 'rsi', 'stoch', 'bb_', 'pct', 'toxicity']):
            problematic_cols['ratio_based'].append(col)
    
    logger.info(f"🔍 Найдено проблемных колонок:")
    logger.info(f"   - С экстремальными значениями (>1e9): {len(problematic_cols['extreme_values'])}")
    logger.info(f"   - Объемные индикаторы: {len(problematic_cols['volume_based'])}")
    logger.info(f"   - Ценовые индикаторы: {len(problematic_cols['price_based'])}")
    logger.info(f"   - Ratio индикаторы: {len(problematic_cols['ratio_based'])}")
    
    return problematic_cols


def normalize_data(df: pd.DataFrame, scaler_path: str = None, fit_scaler: bool = True) -> pd.DataFrame:
    """Нормализация данных с сохранением структуры"""
    logger = get_logger("DataNormalization")
    
    logger.info("🔧 Начало нормализации данных...")
    
    # Создаем копию для работы
    df_norm = df.copy()
    
    # Получаем информацию о колонках
    cols_info = identify_problematic_columns(df)
    
    # Служебные колонки которые не трогаем
    service_cols = ['id', 'symbol', 'datetime', 'timestamp', 'sector']
    target_prefixes = ['future_return_', 'long_', 'short_', 'best_direction']
    
    # Определяем feature колонки для нормализации
    feature_cols = [col for col in df.columns 
                   if col not in service_cols 
                   and not any(col.startswith(prefix) for prefix in target_prefixes)]
    
    logger.info(f"📊 Колонок для нормализации: {len(feature_cols)}")
    
    # 1. Log-трансформация для объемных колонок
    for col in cols_info['volume_based']:
        if col in feature_cols:
            logger.info(f"   📈 Log-трансформация: {col}")
            # Защита от отрицательных значений
            df_norm[col] = np.log1p(np.clip(df_norm[col], 0, None))
    
    # 2. Клиппинг экстремальных значений (кроме ratio)
    for col in feature_cols:
        if col not in cols_info['ratio_based']:
            q99 = df_norm[col].quantile(0.99)
            q01 = df_norm[col].quantile(0.01)
            
            # Проверка на экстремальные значения
            if abs(q99) > 1e6 or abs(q01) < -1e6:
                logger.info(f"   ✂️ Клиппинг {col}: [{q01:.2f}, {q99:.2f}]")
                df_norm[col] = np.clip(df_norm[col], q01, q99)
    
    # 3. RobustScaler для всех feature колонок
    if feature_cols:
        logger.info("🎯 Применение RobustScaler...")
        
        scaler = RobustScaler(quantile_range=(5, 95))
        
        if fit_scaler:
            # Обучаем scaler
            scaler_data = df_norm[feature_cols].values
            scaler.fit(scaler_data)
            
            # Сохраняем scaler
            if scaler_path:
                Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                logger.info(f"💾 Scaler сохранен: {scaler_path}")
        else:
            # Загружаем существующий scaler
            if scaler_path and Path(scaler_path).exists():
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                logger.info(f"📥 Scaler загружен: {scaler_path}")
            else:
                logger.warning("⚠️ Scaler не найден, создаем новый")
                scaler.fit(df_norm[feature_cols].values)
        
        # Применяем scaler
        df_norm[feature_cols] = scaler.transform(df_norm[feature_cols].values)
        
        # Финальный клиппинг после нормализации
        logger.info("🎯 Финальный клиппинг [-10, 10]...")
        df_norm[feature_cols] = np.clip(df_norm[feature_cols], -10, 10)
    
    # 4. Проверка результатов
    logger.info("✅ Проверка результатов нормализации:")
    
    # Проверяем экстремальные значения после нормализации
    extreme_after = 0
    for col in feature_cols:
        max_val = df_norm[col].max()
        if pd.notna(max_val) and abs(max_val) > 100:
            extreme_after += 1
            logger.warning(f"   ⚠️ {col}: max = {max_val:.2f}")
    
    logger.info(f"   - Колонок с экстремальными значениями: {extreme_after}")
    
    # Статистика
    for col in feature_cols[:5]:  # Первые 5 для примера
        logger.info(f"   - {col}: mean={df_norm[col].mean():.3f}, std={df_norm[col].std():.3f}")
    
    return df_norm


def main():
    parser = argparse.ArgumentParser(description='Исправление и нормализация кэшированных данных')
    parser.add_argument('--backup', action='store_true', help='Создать резервные копии')
    parser.add_argument('--scaler-path', default='models_saved/data_scaler.pkl', 
                       help='Путь для сохранения/загрузки scaler')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Только анализ без изменения файлов')
    
    args = parser.parse_args()
    
    logger = get_logger("FixCachedData")
    
    logger.info("="*80)
    logger.info("🔧 Исправление кэшированных данных")
    logger.info("="*80)
    
    # Пути к файлам
    data_dir = Path("data/processed")
    files = {
        'train': data_dir / "train_data.parquet",
        'val': data_dir / "val_data.parquet",
        'test': data_dir / "test_data.parquet"
    }
    
    # Проверка существования файлов
    for split, file_path in files.items():
        if not file_path.exists():
            logger.error(f"❌ Файл не найден: {file_path}")
            logger.info("💡 Сначала запустите: python prepare_trading_data.py")
            return
    
    # Создание резервных копий
    if args.backup and not args.dry_run:
        backup_dir = data_dir / "backup"
        backup_dir.mkdir(exist_ok=True)
        
        logger.info("💾 Создание резервных копий...")
        for split, file_path in files.items():
            backup_path = backup_dir / f"{split}_data_original.parquet"
            if not backup_path.exists():
                import shutil
                shutil.copy(file_path, backup_path)
                logger.info(f"   ✅ {backup_path.name}")
    
    # Обработка файлов
    for split, file_path in files.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"📁 Обработка {split} данных: {file_path.name}")
        logger.info(f"{'='*60}")
        
        # Загрузка данных
        logger.info("📥 Загрузка данных...")
        df = pd.read_parquet(file_path)
        logger.info(f"   Размер: {len(df):,} записей, {len(df.columns)} колонок")
        
        # Анализ проблем
        problems = identify_problematic_columns(df)
        
        if args.dry_run:
            logger.info("\n🔍 Режим анализа (dry-run)")
            
            # Показываем топ-10 проблемных колонок
            if problems['extreme_values']:
                logger.info("\n📊 Топ-10 колонок с экстремальными значениями:")
                sorted_problems = sorted(problems['extreme_values'], 
                                       key=lambda x: abs(x[1]), reverse=True)
                for col, val in sorted_problems[:10]:
                    logger.info(f"   - {col}: {val:.2e}")
            continue
        
        # Нормализация
        fit_scaler = (split == 'train')  # Обучаем scaler только на train
        df_normalized = normalize_data(df, 
                                     scaler_path=args.scaler_path,
                                     fit_scaler=fit_scaler)
        
        # Сохранение
        logger.info(f"💾 Сохранение нормализованных данных...")
        df_normalized.to_parquet(file_path, index=False)
        logger.info(f"   ✅ Сохранено: {file_path}")
        
        # Освобождаем память
        del df, df_normalized
    
    logger.info("\n" + "="*80)
    logger.info("✅ Обработка завершена!")
    logger.info("🚀 Теперь можно запускать обучение: python main.py --mode train")
    logger.info("="*80)


if __name__ == "__main__":
    main()