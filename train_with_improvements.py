"""
Обучение с улучшенной моделью и масштабированием целевой переменной
"""

import os
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Импорты из проекта
from models.patchtst_improved import ImprovedPatchTST
from data.target_scaler import TargetScaler, scale_targets_in_dataset
from data.data_loader import CryptoDataLoader
from data.dataset import create_data_loaders
from data.feature_engineering import FeatureEngineer
from training.trainer import Trainer
from utils.logger import get_logger
from utils.nan_diagnostics import NaNDiagnostics, stabilize_model_initialization, add_gradient_hooks
# from utils.config import load_config  # Этого модуля нет, используем yaml напрямую


def prepare_data_with_scaling(config, logger):
    """Подготовка данных с масштабированием целевой переменной"""
    
    # 1. Загрузка данных
    logger.info("📥 Загрузка данных из БД...")
    data_loader = CryptoDataLoader(config)
    raw_data = data_loader.load_data(
        symbols=config['data']['symbols'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    
    if raw_data.empty:
        raise ValueError("Не удалось загрузить данные!")
    
    logger.info(f"Загружено {len(raw_data):,} записей")
    
    # 2. Проверка качества данных
    logger.info("🔍 Проверка качества данных...")
    quality_report = data_loader.validate_data_quality(raw_data)
    
    # 3. Создание признаков
    logger.info("🛠️ Создание признаков...")
    feature_engineer = FeatureEngineer(config)
    
    # Используем метод без data leakage
    train_data, val_data, test_data = feature_engineer.create_features_with_train_split(
        raw_data, 
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio']
    )
    
    # Получаем список признаков
    exclude_cols = ['id', 'symbol', 'datetime', 'timestamp', 'sector']
    feature_cols = [col for col in train_data.columns 
                    if col not in exclude_cols 
                    and not col.startswith(('target_', 'future_', 'optimal_'))]
    
    logger.info(f"Создано {len(feature_cols)} признаков")
    
    # 4. Масштабирование целевой переменной
    logger.info("🎯 Масштабирование целевой переменной...")
    target_col = config['model']['target_variable']
    
    # Проверяем наличие целевой переменной
    if target_col not in train_data.columns:
        logger.error(f"Целевая переменная {target_col} не найдена!")
        logger.info(f"Доступные целевые: {[col for col in train_data.columns if 'future' in col or 'target' in col]}")
        raise ValueError(f"Целевая переменная {target_col} не найдена в данных!")
    
    # Масштабируем целевую переменную
    train_scaled, val_scaled, test_scaled, target_scaler = scale_targets_in_dataset(
        train_data, val_data, test_data, target_col,
        scaler_path='models_saved/target_scaler.pkl'
    )
    
    # Обновляем конфиг для использования масштабированной целевой
    scaled_target_col = f"{target_col}_scaled"
    
    return train_scaled, val_scaled, test_scaled, feature_cols, scaled_target_col, target_scaler


def create_improved_model(config, n_features):
    """Создание улучшенной модели PatchTST"""
    
    model_params = {
        'c_in': n_features,
        'c_out': config['model']['output_size'],
        'context_window': config['model']['context_window'],
        'target_window': config['model']['target_window'],
        'patch_len': config['model']['patch_len'],
        'stride': config['model']['stride'],
        'd_model': config['model']['d_model'],
        'n_heads': config['model']['n_heads'],
        'd_ff': config['model']['d_ff'],
        'n_layers': config['model']['e_layers'],
        'dropout': config['model']['dropout'],
        'activation': config['model']['activation']
    }
    
    model = ImprovedPatchTST(**model_params)
    
    # Стабильная инициализация весов
    model = stabilize_model_initialization(model, method='xavier')
    
    return model


def load_config(config_path):
    """Загрузка конфигурации из YAML файла"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Основная функция обучения"""
    
    # Настройка логирования
    logger = get_logger("ImprovedTraining")
    
    # Загрузка конфигурации
    config = load_config('config/config.yaml')
    
    logger.info("="*80)
    logger.info("🚀 Запуск улучшенного обучения Crypto AI Trading")
    logger.info("="*80)
    
    try:
        # 1. Подготовка данных
        logger.info("📊 Этап 1: Подготовка данных")
        train_data, val_data, test_data, feature_cols, scaled_target_col, target_scaler = prepare_data_with_scaling(config, logger)
        
        # Временно обновляем конфиг
        original_target = config['model']['target_variable']
        config['model']['target_variable'] = scaled_target_col
        
        # 2. Создание DataLoaders
        logger.info("🏗️ Этап 2: Создание DataLoaders")
        train_loader, val_loader, test_loader = create_data_loaders(
            train_data, val_data, test_data, config, feature_cols
        )
        
        logger.info(f"Батчей в эпохе: Train={len(train_loader)}, Val={len(val_loader)}")
        
        # 3. Создание модели
        logger.info("🤖 Этап 3: Создание улучшенной модели")
        model = create_improved_model(config, len(feature_cols))
        logger.info(f"Параметров модели: {sum(p.numel() for p in model.parameters()):,}")
        
        # Добавляем хуки для отслеживания градиентов
        add_gradient_hooks(model, logger)
        
        # Создаем диагностику NaN
        nan_diagnostics = NaNDiagnostics(logger)
        
        # 4. Создание тренера
        logger.info("🎯 Этап 4: Настройка обучения")
        trainer = Trainer(model, config)
        
        # Делаем оптимизатор безопасным от NaN
        trainer.optimizer = nan_diagnostics.create_nan_safe_optimizer(trainer.optimizer)
        
        # 5. Обучение
        logger.info("🚀 Этап 5: Запуск обучения")
        logger.info(f"Learning rate: {config['model']['learning_rate']}")
        logger.info(f"Batch size: {config['model']['batch_size']}")
        logger.info(f"Epochs: {config['model']['epochs']}")
        
        # Обучение модели
        history = trainer.train(train_loader, val_loader)
        
        # Находим лучшую модель
        import glob
        model_files = glob.glob("models_saved/best_model_*.pth")
        if model_files:
            best_model_path = max(model_files, key=lambda x: os.path.getmtime(x))
            logger.info(f"✅ Обучение завершено! Лучшая модель: {best_model_path}")
        else:
            logger.error("❌ Модель не была сохранена!")
            return
        
        # 6. Финальная оценка
        logger.info("📈 Этап 6: Финальная оценка на тестовых данных")
        
        # Загружаем лучшую модель
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Оценка на тесте
        test_metrics = trainer.validate(test_loader)
        
        logger.info("📊 Результаты на тестовых данных:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Сохраняем информацию о масштабировании
        metadata = {
            'original_target': original_target,
            'scaled_target': scaled_target_col,
            'target_scaler_path': 'models_saved/target_scaler.pkl',
            'feature_cols': feature_cols,
            'model_type': 'ImprovedPatchTST',
            'training_completed': datetime.now().isoformat()
        }
        
        import joblib
        metadata_path = Path(best_model_path).parent / 'training_metadata.pkl'
        joblib.dump(metadata, metadata_path)
        logger.info(f"Метаданные сохранены в {metadata_path}")
        
        # Восстанавливаем оригинальный конфиг
        config['model']['target_variable'] = original_target
        
        logger.info("="*80)
        logger.info("✨ Обучение успешно завершено!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Ошибка при обучении: {str(e)}")
        raise


if __name__ == "__main__":
    main()