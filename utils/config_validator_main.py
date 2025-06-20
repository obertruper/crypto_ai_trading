"""
Валидация основной конфигурации при запуске
"""

import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import sys

def validate_config(config_path: str = "config/config.yaml") -> Tuple[bool, List[str]]:
    """
    Валидация конфигурационного файла
    
    Returns:
        (is_valid, errors): кортеж с флагом валидности и списком ошибок
    """
    errors = []
    
    # Проверка существования файла
    if not Path(config_path).exists():
        return False, [f"Файл конфигурации не найден: {config_path}"]
    
    # Загрузка конфигурации
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return False, [f"Ошибка чтения конфигурации: {e}"]
    
    # Проверка основных секций
    required_sections = ['data', 'model', 'database', 'risk_management', 'performance']
    for section in required_sections:
        if section not in config:
            errors.append(f"Отсутствует обязательная секция: {section}")
    
    # Валидация секции data
    if 'data' in config:
        data = config['data']
        if 'symbols' not in data:
            errors.append("data.symbols не определен")
        elif isinstance(data['symbols'], list) and len(data['symbols']) == 0:
            errors.append("data.symbols пустой список")
        
        # Проверка интервала
        if 'interval_minutes' in data:
            if not isinstance(data['interval_minutes'], int) or data['interval_minutes'] <= 0:
                errors.append(f"data.interval_minutes должен быть положительным числом, получено: {data['interval_minutes']}")
    
    # Валидация секции model
    if 'model' in config:
        model = config['model']
        
        # Проверка размеров патчей
        context_window = model.get('context_window', 96)
        patch_len = model.get('patch_len', 16)
        stride = model.get('stride', 8)
        
        if context_window < patch_len:
            errors.append(f"model.context_window ({context_window}) должен быть >= model.patch_len ({patch_len})")
        
        if stride <= 0 or stride > patch_len:
            errors.append(f"model.stride ({stride}) должен быть в диапазоне (0, {patch_len}]")
        
        # Проверка размеров модели
        d_model = model.get('d_model', 256)
        n_heads = model.get('n_heads', 16)
        
        if d_model % n_heads != 0:
            errors.append(f"model.d_model ({d_model}) должен быть кратен model.n_heads ({n_heads})")
        
        # Проверка параметров обучения
        if 'batch_size' in model and model['batch_size'] <= 0:
            errors.append(f"model.batch_size должен быть положительным, получено: {model['batch_size']}")
        
        if 'learning_rate' in model and (model['learning_rate'] <= 0 or model['learning_rate'] > 1):
            errors.append(f"model.learning_rate должен быть в диапазоне (0, 1], получено: {model['learning_rate']}")
    
    # Валидация секции database
    if 'database' in config:
        db = config['database']
        required_db_fields = ['host', 'port', 'database', 'user', 'password']
        for field in required_db_fields:
            if field not in db:
                errors.append(f"database.{field} не определен")
    
    # Валидация секции risk_management
    if 'risk_management' in config:
        risk = config['risk_management']
        
        # Проверка stop_loss и take_profit
        stop_loss = risk.get('stop_loss_pct', 2.0)
        take_profits = risk.get('take_profit_targets', [])
        
        if stop_loss <= 0:
            errors.append(f"risk_management.stop_loss_pct должен быть положительным, получено: {stop_loss}")
        
        if not take_profits or len(take_profits) == 0:
            errors.append("risk_management.take_profit_targets не определен или пустой")
        else:
            for i, tp in enumerate(take_profits):
                if tp <= 0:
                    errors.append(f"risk_management.take_profit_targets[{i}] должен быть положительным, получено: {tp}")
        
        # Проверка partial_close_sizes
        partial_sizes = risk.get('partial_close_sizes', [])
        if partial_sizes and sum(partial_sizes) != 100:
            errors.append(f"Сумма risk_management.partial_close_sizes должна быть 100, получено: {sum(partial_sizes)}")
    
    # Валидация секции performance
    if 'performance' in config:
        perf = config['performance']
        
        # Проверка device
        device = perf.get('device', 'cpu')
        if device not in ['cpu', 'cuda', 'mps']:
            errors.append(f"performance.device должен быть 'cpu', 'cuda' или 'mps', получено: {device}")
    
    return len(errors) == 0, errors


def validate_and_exit_on_error(config_path: str = "config/config.yaml"):
    """Валидация с выходом при ошибке"""
    is_valid, errors = validate_config(config_path)
    
    if not is_valid:
        print("❌ Ошибки валидации конфигурации:")
        for error in errors:
            print(f"  • {error}")
        sys.exit(1)
    else:
        print("✅ Конфигурация валидна")


if __name__ == "__main__":
    # Тест валидации
    is_valid, errors = validate_config()
    
    if is_valid:
        print("✅ Конфигурация валидна")
    else:
        print("❌ Найдены ошибки:")
        for error in errors:
            print(f"  • {error}")