#!/usr/bin/env python3
"""
Быстрый запуск демо обучения
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path

print("🚀 Запуск демо обучения (5 эпох)")
print("=" * 60)

# Проверяем наличие кэша
cache_file = Path("cache/features_cache.pkl")
if not cache_file.exists():
    print("❌ Кэш данных не найден!")
    print("Сначала запустите: python run_full_pipeline.py --mode data")
    sys.exit(1)

print(f"✅ Найден кэш данных: {cache_file.stat().st_size / (1024*1024):.1f} MB")

# Загружаем конфигурацию
config_path = Path("config/config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Сохраняем оригинальное количество эпох
original_epochs = config['model']['epochs']

# Устанавливаем 5 эпох для демо
config['model']['epochs'] = 5

# Сохраняем временную конфигурацию
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print(f"📊 Символов в датасете: {len(config['data']['symbols'])}")
print(f"🔢 Эпох: {config['model']['epochs']}")
print("=" * 60)

try:
    # Запускаем обучение
    result = subprocess.run(
        ["python", "run_full_pipeline.py", "--mode", "train"],
        env={**os.environ, 'USE_CACHE_ONLY': '1'}
    )
    
    if result.returncode == 0:
        print("\n✅ Демо обучение завершено успешно!")
    else:
        print("\n❌ Ошибка при обучении")
        
finally:
    # Восстанавливаем оригинальную конфигурацию
    config['model']['epochs'] = original_epochs
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print("✅ Конфигурация восстановлена")