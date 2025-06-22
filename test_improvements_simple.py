"""
Упрощенный тест улучшений модели
"""

import torch
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

# Импорты из проекта
from models.patchtst_improved import ImprovedPatchTST
from data.target_scaler import TargetScaler
from utils.logger import get_logger


def test_model_improvements():
    """Тестирование улучшений модели"""
    logger = get_logger("TestImproved")
    
    # Загружаем конфигурацию
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("🚀 Тестирование улучшений модели")
    
    # 1. Создаем синтетические данные для теста
    logger.info("📊 Создание синтетических данных...")
    
    batch_size = 32
    context_window = config['model']['context_window']
    n_features = 86  # Типичное количество признаков
    target_window = config['model']['target_window']
    
    # Создаем входные данные
    inputs = torch.randn(batch_size, context_window, n_features)
    targets = torch.randn(batch_size, target_window, 1)
    
    logger.info(f"Input shape: {inputs.shape}")
    logger.info(f"Target shape: {targets.shape}")
    
    # 2. Создание улучшенной модели
    logger.info("🤖 Создание улучшенной модели PatchTST...")
    
    model_params = {
        'c_in': n_features,
        'c_out': 1,
        'context_window': context_window,
        'target_window': target_window,
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
    
    # 3. Тест forward pass
    logger.info("🔍 Тестирование forward pass...")
    
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        
    logger.info(f"Output shape: {outputs.shape}")
    logger.info(f"Output statistics: mean={outputs.mean():.4f}, std={outputs.std():.4f}")
    
    # 4. Тест loss
    criterion = torch.nn.MSELoss()
    loss = criterion(outputs, targets)
    logger.info(f"Initial loss: {loss.item():.4f}")
    
    # 5. Тест масштабирования целевой переменной
    logger.info("\n🎯 Тестирование масштабирования целевой переменной...")
    
    # Создаем тестовые целевые значения похожие на future_return_4
    test_targets = np.random.normal(0.006, 1.2, size=10000)  # mean=0.006, std=1.2
    
    scaler = TargetScaler(method='robust', clip_quantiles=(0.01, 0.99))
    scaler.fit(test_targets)
    
    scaled_targets = scaler.transform(test_targets)
    logger.info(f"После масштабирования - mean: {np.mean(scaled_targets):.4f}, std: {np.std(scaled_targets):.4f}")
    
    # 6. Сравнение архитектур
    logger.info("\n📊 Сравнение с оригинальной PatchTST...")
    
    try:
        from models.patchtst import PatchTST
        original_model = PatchTST(**model_params)
        original_params = sum(p.numel() for p in original_model.parameters())
        improved_params = sum(p.numel() for p in model.parameters())
        
        logger.info(f"Оригинальная модель: {original_params:,} параметров")
        logger.info(f"Улучшенная модель: {improved_params:,} параметров")
        logger.info(f"Увеличение параметров: {(improved_params - original_params):,} ({(improved_params/original_params - 1)*100:.1f}%)")
        
        # Тест оригинальной модели
        with torch.no_grad():
            original_outputs = original_model(inputs)
            original_loss = criterion(original_outputs, targets)
        
        logger.info(f"\nОригинальная модель loss: {original_loss.item():.4f}")
        logger.info(f"Улучшенная модель loss: {loss.item():.4f}")
        
    except Exception as e:
        logger.warning(f"Не удалось сравнить с оригинальной моделью: {e}")
    
    # 7. Тест градиентов
    logger.info("\n🔧 Тестирование градиентов...")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Forward
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward
    loss.backward()
    
    # Проверка градиентов
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_norms.append(grad_norm)
            if grad_norm > 10:
                logger.warning(f"Большой градиент в {name}: {grad_norm:.4f}")
    
    logger.info(f"Средняя норма градиентов: {np.mean(grad_norms):.4f}")
    logger.info(f"Максимальная норма градиентов: {np.max(grad_norms):.4f}")
    
    logger.info("\n✅ Тестирование завершено успешно!")
    
    return model


if __name__ == "__main__":
    test_model_improvements()