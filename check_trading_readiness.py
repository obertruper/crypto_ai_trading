#!/usr/bin/env python3
"""
Проверка готовности системы к обучению торговой модели
"""

import yaml
import torch
import psycopg2
from pathlib import Path
import importlib
import sys

def check_item(description, check_func):
    """Выполнить проверку и вывести результат"""
    try:
        result, details = check_func()
        if result:
            print(f"✅ {description}: {details}")
            return True
        else:
            print(f"❌ {description}: {details}")
            return False
    except Exception as e:
        print(f"❌ {description}: ОШИБКА - {str(e)}")
        return False


def check_gpu():
    """Проверка GPU"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return True, f"{device_name} ({memory:.1f} GB)"
    return False, "GPU не найден"


def check_database():
    """Проверка подключения к БД"""
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        db_config = config['database']
        conn = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['database'],
            user=db_config['user'],
            password=db_config['password']
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM raw_market_data")
        count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return True, f"{count:,} записей в БД"
    except:
        return False, "Не удалось подключиться"


def check_modules():
    """Проверка ключевых модулей"""
    modules = {
        'feature_engineering': 'data.feature_engineering',
        'patchtst': 'models.patchtst',
        'signals_v2': 'trading.signals_v2',
        'trading_losses': 'models.trading_losses'
    }
    
    missing = []
    for name, module_path in modules.items():
        try:
            importlib.import_module(module_path)
        except:
            missing.append(name)
    
    if missing:
        return False, f"Отсутствуют модули: {', '.join(missing)}"
    return True, "Все модули на месте"


def check_config():
    """Проверка конфигурации"""
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        checks = []
        
        # Проверка task_type
        if config['model'].get('task_type') == 'trading':
            checks.append("task_type=trading ✓")
        else:
            return False, "task_type должен быть 'trading'"
        
        # Проверка loss
        if 'trading' in config['loss'].get('name', ''):
            checks.append("loss=trading_multi_task ✓")
        
        # Проверка риск-менеджмента
        if config['risk_management']['stop_loss_pct'] == 2.0:
            checks.append("SL=2% ✓")
        
        tp_targets = config['risk_management']['take_profit_targets']
        if tp_targets == [1.5, 2.5, 4.0]:
            checks.append("TP=[1.5,2.5,4.0]% ✓")
        
        return True, ", ".join(checks)
    except:
        return False, "Ошибка чтения config.yaml"


def check_data_files():
    """Проверка наличия подготовленных данных"""
    data_dir = Path("data/processed")
    files = ['train_data.parquet', 'val_data.parquet', 'test_data.parquet']
    
    existing = []
    for f in files:
        if (data_dir / f).exists():
            existing.append(f)
    
    if len(existing) == len(files):
        return True, "Все файлы данных готовы"
    elif existing:
        return False, f"Найдены только: {', '.join(existing)}"
    else:
        return False, "Данные не подготовлены. Запустите prepare_trading_data.py"


def main():
    print("\n" + "="*60)
    print("🔍 ПРОВЕРКА ГОТОВНОСТИ К ОБУЧЕНИЮ ТОРГОВОЙ МОДЕЛИ")
    print("="*60 + "\n")
    
    checks = [
        ("GPU доступность", check_gpu),
        ("База данных PostgreSQL", check_database),
        ("Ключевые модули", check_modules),
        ("Конфигурация", check_config),
        ("Подготовленные данные", check_data_files),
    ]
    
    passed = 0
    for description, check_func in checks:
        if check_item(description, check_func):
            passed += 1
    
    print("\n" + "-"*60)
    print(f"Результат: {passed}/{len(checks)} проверок пройдено")
    
    if passed == len(checks):
        print("\n✅ СИСТЕМА ПОЛНОСТЬЮ ГОТОВА К ОБУЧЕНИЮ!")
        print("\n🚀 Запустите обучение:")
        print("   python main.py --mode train")
        print("\n📊 Или используйте интерактивный режим:")
        print("   python run_interactive.py")
    else:
        print("\n⚠️ Необходимо исправить проблемы перед запуском")
        
        if "данные не подготовлены" in str([c[1]() for c in checks]).lower():
            print("\n💡 Сначала подготовьте данные:")
            print("   python prepare_trading_data.py")


if __name__ == "__main__":
    main()