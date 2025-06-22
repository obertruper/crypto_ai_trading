"""
Скрипт для тестирования улучшений модели
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

# Импорты из проекта
from models.patchtst_improved import ImprovedPatchTST
from data.target_scaler import TargetScaler, scale_targets_in_dataset
from data.data_loader import CryptoDataLoader
from data.dataset import create_data_loaders
from utils.logger import get_logger


def test_improved_model():
    """Тестирование улучшенной модели"""
    logger = get_logger("TestImproved")
    
    # Загружаем конфигурацию
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("🚀 Тестирование улучшений модели")
    
    # 1. Загрузка данных
    logger.info("📥 Загрузка данных...")
    data_loader = CryptoDataLoader(config)
    train_data, val_data, test_data, feature_cols = data_loader.get_train_val_test_data()
    
    logger.info(f"Размеры данных: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # 2. Масштабирование целевой переменной
    logger.info("🎯 Масштабирование целевой переменной...")
    target_col = config['model']['target_variable']
    
    train_scaled, val_scaled, test_scaled, target_scaler = scale_targets_in_dataset(
        train_data, val_data, test_data, target_col,
        scaler_path='models_saved/target_scaler.pkl'
    )
    
    # Обновляем конфиг для использования масштабированной целевой
    config['model']['target_variable'] = f"{target_col}_scaled"
    
    # 3. Создание DataLoaders
    logger.info("🏗️ Создание DataLoaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_scaled, val_scaled, test_scaled, config, feature_cols
    )
    
    # 4. Создание улучшенной модели
    logger.info("🤖 Создание улучшенной модели PatchTST...")
    
    model_params = {
        'c_in': len(feature_cols),
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
    logger.info(f"Параметров модели: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. Тест forward pass
    logger.info("🔍 Тестирование forward pass...")
    
    # Получаем один батч
    for batch in train_loader:
        inputs, targets, info = batch
        break
    
    logger.info(f"Input shape: {inputs.shape}")
    logger.info(f"Target shape: {targets.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        
    logger.info(f"Output shape: {outputs.shape}")
    logger.info(f"Output statistics: mean={outputs.mean():.4f}, std={outputs.std():.4f}")
    
    # 6. Тест loss
    criterion = torch.nn.MSELoss()
    loss = criterion(outputs, targets)
    logger.info(f"Initial loss: {loss.item():.4f}")
    
    # 7. Сравнение с baseline
    baseline_pred = torch.zeros_like(targets)  # Предсказание нулей (среднее после нормализации)
    baseline_loss = criterion(baseline_pred, targets)
    logger.info(f"Baseline loss (нули): {baseline_loss.item():.4f}")
    
    # Предсказание среднего
    mean_pred = targets.mean() * torch.ones_like(targets)
    mean_loss = criterion(mean_pred, targets)
    logger.info(f"Baseline loss (среднее): {mean_loss.item():.4f}")
    
    logger.info("✅ Тестирование завершено успешно!")
    
    return model, train_loader, val_loader, target_scaler


if __name__ == "__main__":
    test_improved_model()