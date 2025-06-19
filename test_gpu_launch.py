#!/usr/bin/env python3
"""
Тестовый скрипт для проверки запуска GPU обучения
"""

import subprocess
import os

print("🧪 Тест запуска GPU обучения")

# Установка переменных окружения
env = os.environ.copy()
env['USE_CACHE_ONLY'] = '1'
env['GPU_TRAINING_MODE'] = '1'
env['GPU_TRAINING_EPOCHS'] = '5'

print("📊 Параметры:")
print(f"   USE_CACHE_ONLY: {env['USE_CACHE_ONLY']}")
print(f"   GPU_TRAINING_MODE: {env['GPU_TRAINING_MODE']}")
print(f"   GPU_TRAINING_EPOCHS: {env['GPU_TRAINING_EPOCHS']}")

print("\n🚀 Запуск скрипта...")

result = subprocess.run(
    ["bash", "scripts/run_on_vast.sh"],
    env=env,
    capture_output=True,
    text=True
)

print("\n📋 Вывод:")
print(result.stdout)

if result.stderr:
    print("\n❌ Ошибки:")
    print(result.stderr)

print(f"\n✅ Код завершения: {result.returncode}")