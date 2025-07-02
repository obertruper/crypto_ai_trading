#!/usr/bin/env python3
"""
Проверка использования GPU во время обучения
"""

import torch
import yaml
from pathlib import Path
import traceback

from utils.logger import get_logger
from models.patchtst_unified import create_unified_model
from main import load_cached_data_if_exists, create_unified_data_loaders

def check_gpu_setup():
    """Проверка настройки GPU"""
    logger = get_logger("GPUCheck")
    
    logger.info("🔍 Проверка GPU...")
    logger.info(f"CUDA доступна: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"Количество GPU: {torch.cuda.device_count()}")
        logger.info(f"Текущий GPU: {torch.cuda.current_device()}")
        logger.info(f"GPU имя: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Проверка выделенной памяти
        logger.info(f"Выделено памяти: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        logger.info(f"Зарезервировано памяти: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        logger.error("❌ CUDA не доступна!")
        return False
    
    return True

def check_model_on_gpu():
    """Проверка что модель на GPU"""
    logger = get_logger("GPUCheck")
    
    try:
        # Загрузка конфигурации
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Создание модели
        logger.info("🏗️ Создание модели...")
        config['model']['name'] = 'UnifiedPatchTST'
        config['model']['output_size'] = 36
        model = create_unified_model(config)
        
        # Проверка устройства модели
        device = next(model.parameters()).device
        logger.info(f"📍 Модель на устройстве: {device}")
        
        if device.type != 'cuda':
            logger.warning("⚠️ Модель НЕ на GPU! Перемещаем...")
            model = model.cuda()
            device = next(model.parameters()).device
            logger.info(f"✅ Модель перемещена на: {device}")
        
        # Проверка всех параметров
        all_on_gpu = all(p.is_cuda for p in model.parameters())
        logger.info(f"Все параметры на GPU: {all_on_gpu}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка при проверке модели: {e}")
        traceback.print_exc()
        return False

def check_data_loading():
    """Проверка загрузки данных на GPU"""
    logger = get_logger("GPUCheck")
    
    try:
        # Загрузка конфигурации
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Загрузка данных
        logger.info("📥 Загрузка данных...")
        train_data, val_data, test_data, feature_cols, target_cols = load_cached_data_if_exists(logger)
        
        if train_data is None:
            logger.error("❌ Данные не найдены!")
            return False
        
        # Создание DataLoader
        logger.info("🔄 Создание DataLoader...")
        train_loader, val_loader, test_loader, config_updated = create_unified_data_loaders(
            train_data, val_data, test_data, feature_cols, target_cols, config, logger
        )
        
        # Проверка первого батча
        logger.info("🔍 Проверка первого батча...")
        batch = next(iter(train_loader))
        X, y, info = batch
        
        logger.info(f"📊 Размерности батча:")
        logger.info(f"   X: {X.shape}, устройство: {X.device}")
        logger.info(f"   y: {y.shape}, устройство: {y.device}")
        
        # Проверка перемещения на GPU
        logger.info("🚀 Перемещение на GPU...")
        X_gpu = X.cuda()
        y_gpu = y.cuda()
        
        logger.info(f"   X на GPU: {X_gpu.device}")
        logger.info(f"   y на GPU: {y_gpu.device}")
        
        # Проверка памяти после загрузки батча
        logger.info(f"💾 Память после загрузки батча:")
        logger.info(f"   Выделено: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        logger.info(f"   Зарезервировано: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка при проверке данных: {e}")
        traceback.print_exc()
        return False

def run_mini_training():
    """Запуск мини-обучения для проверки GPU"""
    logger = get_logger("GPUCheck")
    
    try:
        from training.trainer import Trainer
        
        # Загрузка конфигурации
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Создание модели
        config['model']['name'] = 'UnifiedPatchTST'
        config['model']['output_size'] = 36
        config['model']['epochs'] = 1  # Только 1 эпоха для теста
        model = create_unified_model(config)
        
        # Загрузка данных
        train_data, val_data, test_data, feature_cols, target_cols = load_cached_data_if_exists(logger)
        train_loader, val_loader, test_loader, config_updated = create_unified_data_loaders(
            train_data, val_data, test_data, feature_cols, target_cols, config, logger
        )
        
        # Создание трейнера
        trainer = Trainer(model, config_updated)
        
        logger.info(f"🎯 Устройство трейнера: {trainer.device}")
        logger.info(f"🔥 Модель на устройстве: {next(trainer.model.parameters()).device}")
        
        # Обучение на 10 батчах
        logger.info("🚀 Запуск тестового обучения...")
        
        model.train()
        for i, (X, y, info) in enumerate(train_loader):
            if i >= 10:
                break
            
            # Проверка устройств
            logger.info(f"\n📍 Батч {i}:")
            logger.info(f"   X устройство до: {X.device}")
            logger.info(f"   y устройство до: {y.device}")
            
            # Перемещение на GPU
            X = X.to(trainer.device)
            y = y.to(trainer.device)
            
            logger.info(f"   X устройство после: {X.device}")
            logger.info(f"   y устройство после: {y.device}")
            
            # Forward pass
            outputs = model(X)
            logger.info(f"   Outputs устройство: {outputs.device}")
            
            # Память GPU
            logger.info(f"   GPU память: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        logger.info("\n✅ Тест GPU завершен успешно!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка при тестовом обучении: {e}")
        traceback.print_exc()
        return False

def main():
    """Основная функция проверки"""
    logger = get_logger("GPUCheck")
    
    logger.info("="*80)
    logger.info("🔍 ПРОВЕРКА ИСПОЛЬЗОВАНИЯ GPU")
    logger.info("="*80)
    
    # 1. Проверка доступности GPU
    if not check_gpu_setup():
        return
    
    # 2. Проверка модели на GPU
    logger.info("\n" + "="*50)
    if not check_model_on_gpu():
        return
    
    # 3. Проверка загрузки данных
    logger.info("\n" + "="*50)
    if not check_data_loading():
        return
    
    # 4. Запуск мини-обучения
    logger.info("\n" + "="*50)
    if not run_mini_training():
        return
    
    logger.info("\n" + "="*80)
    logger.info("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!")
    logger.info("="*80)

if __name__ == "__main__":
    main()