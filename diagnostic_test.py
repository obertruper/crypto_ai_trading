#!/usr/bin/env python3

import sys
from pathlib import Path

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Тест импортов основных модулей"""
    print("🧪 ТЕСТ ИМПОРТОВ МОДУЛЕЙ")
    print("=" * 40)
    
    imports_to_test = [
        ("yaml", "import yaml"),
        ("torch", "import torch"),
        ("pandas", "import pandas as pd"),
        ("numpy", "import numpy as np"),
        ("data.data_loader", "from data.data_loader import CryptoDataLoader"),
        ("data.feature_engineering", "from data.feature_engineering import FeatureEngineer"),
        ("models.patchtst", "from models.patchtst import PatchTSTForPrediction"),
        ("training.trainer", "from training.trainer import Trainer"),
        ("utils.logger", "from utils.logger import get_logger"),
    ]
    
    success_count = 0
    
    for name, import_str in imports_to_test:
        try:
            exec(import_str)
            print(f"✅ {name}")
            success_count += 1
        except Exception as e:
            print(f"❌ {name}: {e}")
    
    print(f"\n📊 Результат: {success_count}/{len(imports_to_test)} импортов успешно")
    
    return success_count == len(imports_to_test)

def test_config():
    """Тест загрузки конфигурации"""
    print("\n⚙️ ТЕСТ КОНФИГУРАЦИИ")
    print("=" * 40)
    
    try:
        import yaml
        config_path = Path("config/config.yaml")
        
        if not config_path.exists():
            print(f"❌ Файл конфигурации не найден: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['data', 'model', 'database', 'features']
        missing_sections = []
        
        for section in required_sections:
            if section in config:
                print(f"✅ Секция '{section}' найдена")
            else:
                print(f"❌ Секция '{section}' отсутствует")
                missing_sections.append(section)
        
        if not missing_sections:
            print("✅ Все обязательные секции присутствуют")
            return True
        else:
            print(f"❌ Отсутствуют секции: {missing_sections}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка загрузки конфигурации: {e}")
        return False

def test_database():
    """Тест подключения к БД"""
    print("\n🗄️ ТЕСТ БАЗЫ ДАННЫХ")
    print("=" * 40)
    
    try:
        import yaml
        import os
        
        # Загружаем конфигурацию
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Проверяем переменные окружения
        db_config = config['database']
        print("Проверка переменных окружения:")
        
        env_vars = {
            'DB_HOST': os.getenv('DB_HOST', 'localhost'),
            'DB_PORT': os.getenv('DB_PORT', '5555'),  
            'DB_NAME': os.getenv('DB_NAME', 'crypto_trading'),
            'DB_USER': os.getenv('DB_USER', 'ruslan'),
            'DB_PASSWORD': os.getenv('DB_PASSWORD', 'ruslan')
        }
        
        for var, value in env_vars.items():
            if value:
                print(f"✅ {var}: {value}")
            else:
                print(f"❌ {var}: не установлена")
        
        # Пробуем подключиться
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=env_vars['DB_HOST'],
                port=int(env_vars['DB_PORT']),
                database=env_vars['DB_NAME'],
                user=env_vars['DB_USER'],
                password=env_vars['DB_PASSWORD']
            )
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            conn.close()
            print("✅ Подключение к БД успешно")
            return True
        except Exception as e:
            print(f"❌ Ошибка подключения к БД: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка теста БД: {e}")
        return False

def main():
    print("🔍 ДИАГНОСТИКА СИСТЕМЫ")
    print("=" * 60)
    
    test1 = test_imports()
    test2 = test_config()
    test3 = test_database()
    
    print("\n" + "=" * 60)
    print("📊 ОБЩИЙ РЕЗУЛЬТАТ:")
    
    if test1 and test2 and test3:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Система готова к работе.")
        return True
    else:
        print("⚠️ Некоторые тесты провалились. Требуется исправление.")
        print("\n💡 Рекомендации:")
        if not test1:
            print("  - Установите недостающие зависимости: pip install -r requirements.txt")
        if not test2:
            print("  - Проверьте файл config/config.yaml")
        if not test3:
            print("  - Проверьте настройки PostgreSQL и переменные окружения")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
