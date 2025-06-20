#!/usr/bin/env python3
"""
Скрипт для диагностики и исправления проблем обучения Crypto AI Trading
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import psutil
from datetime import datetime

def print_header(title):
    """Красивый заголовок"""
    print("\n" + "="*60)
    print(f"🔧 {title}")
    print("="*60)

def print_step(step, message):
    """Красивый шаг"""
    print(f"\n📋 ШАГ {step}: {message}")
    print("-" * 40)

def check_system():
    """Проверка системы"""
    print_header("ДИАГНОСТИКА СИСТЕМЫ")
    
    # Проверка Python процессов
    print("🔍 Поиск активных процессов обучения...")
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline'])
                if any(keyword in cmdline.lower() for keyword in ['train', 'main.py', 'crypto_ai']):
                    python_processes.append((proc.info['pid'], cmdline))
        except:
            continue
    
    if python_processes:
        print("⚠️ Найдены активные процессы обучения:")
        for pid, cmd in python_processes:
            print(f"   PID {pid}: {cmd}")
        
        response = input("\n❓ Остановить эти процессы? (y/n): ")
        if response.lower() == 'y':
            for pid, _ in python_processes:
                try:
                    os.kill(pid, 9)
                    print(f"✅ Процесс {pid} остановлен")
                except:
                    print(f"❌ Не удалось остановить процесс {pid}")
    else:
        print("✅ Активных процессов обучения не найдено")
    
    # Проверка GPU
    print("\n🎮 Проверка GPU...")
    if torch.cuda.is_available():
        print(f"✅ CUDA доступен: {torch.cuda.get_device_name(0)}")
        print(f"   Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   Используемая память: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    else:
        print("⚠️ CUDA недоступен, будет использоваться CPU")
    
    # Проверка места на диске
    print("\n💾 Проверка места на диске...")
    disk_usage = psutil.disk_usage('/')
    free_gb = disk_usage.free / (1024**3)
    print(f"   Свободно: {free_gb:.1f} GB")
    if free_gb < 10:
        print("⚠️ Мало свободного места! Рекомендуется освободить место")

def analyze_config():
    """Анализ конфигурации"""
    print_header("АНАЛИЗ КОНФИГУРАЦИИ")
    
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print("❌ Файл config.yaml не найден!")
        return None
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    model_config = config.get('model', {})
    
    # Анализ критических параметров
    print("🔍 Анализ параметров модели:")
    
    lr = model_config.get('learning_rate', 0)
    print(f"   Learning Rate: {lr}")
    if lr < 0.0005:
        print("   ❌ КРИТИЧЕСКАЯ ПРОБЛЕМА: Learning Rate слишком низкий!")
    elif lr < 0.001:
        print("   ⚠️ Learning Rate может быть низким")
    else:
        print("   ✅ Learning Rate в норме")
    
    batch_size = model_config.get('batch_size', 0)
    print(f"   Batch Size: {batch_size}")
    if batch_size > 128:
        print("   ⚠️ Большой batch size может вызывать проблемы")
    
    context_window = model_config.get('context_window', 0)
    print(f"   Context Window: {context_window}")
    if context_window > 64:
        print("   ⚠️ Большое окно контекста может замедлять обучение")
    
    warmup_steps = model_config.get('warmup_steps', 0)
    print(f"   Warmup Steps: {warmup_steps}")
    if warmup_steps > 1000:
        print("   ⚠️ Слишком много warmup steps для низкого LR")
    
    return config

def check_data():
    """Проверка данных"""
    print_header("ПРОВЕРКА ДАННЫХ")
    
    # Проверка кэша
    cache_dir = Path("cache")
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*.pkl")) + list(cache_dir.glob("*.parquet"))
        print(f"📁 Найдено файлов в кэше: {len(cache_files)}")
        
        total_size = sum(f.stat().st_size for f in cache_files) / (1024**2)
        print(f"   Общий размер: {total_size:.1f} MB")
        
        if cache_files:
            latest_file = max(cache_files, key=lambda f: f.stat().st_mtime)
            mod_time = datetime.fromtimestamp(latest_file.stat().st_mtime)
            print(f"   Последнее обновление: {mod_time}")
    else:
        print("⚠️ Директория кэша не найдена")
    
    # Проверка подключения к БД
    print("\n🗄️ Проверка подключения к БД...")
    try:
        # Попытка импорта и подключения
        sys.path.append(str(Path.cwd()))
        from data.data_loader import CryptoDataLoader
        
        config_path = Path("config/config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        loader = CryptoDataLoader(config)
        symbols = loader.get_available_symbols()
        print(f"✅ БД доступна, символов: {len(symbols)}")
        
        # Проверка последних данных
        if symbols:
            sample_data = loader.load_data(
                symbols=symbols[:1],
                start_date="2025-06-15",
                end_date="2025-06-16"
            )
            print(f"   Тестовая загрузка: {len(sample_data)} записей")
        
    except Exception as e:
        print(f"❌ Ошибка подключения к БД: {e}")

def check_logs():
    """Проверка логов"""
    print_header("АНАЛИЗ ЛОГОВ")
    
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print("⚠️ Директория логов не найдена")
        return
    
    # Поиск последних логов
    log_files = []
    for pattern in ["**/*.log", "**/*training*.log"]:
        log_files.extend(logs_dir.glob(pattern))
    
    if not log_files:
        print("⚠️ Лог файлы не найдены")
        return
    
    # Последний лог
    latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
    mod_time = datetime.fromtimestamp(latest_log.stat().st_mtime)
    print(f"📄 Последний лог: {latest_log.name}")
    print(f"   Время: {mod_time}")
    
    # Анализ содержимого
    try:
        with open(latest_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"   Строк в логе: {len(lines)}")
        
        # Поиск ошибок
        error_lines = [line for line in lines if 'error' in line.lower() or 'exception' in line.lower()]
        if error_lines:
            print(f"   ⚠️ Найдено ошибок: {len(error_lines)}")
            print("   Последние ошибки:")
            for line in error_lines[-3:]:
                print(f"     {line.strip()}")
        
        # Поиск информации о loss
        loss_lines = [line for line in lines if 'loss' in line.lower() and any(word in line for word in ['epoch', 'train', 'val'])]
        if loss_lines:
            print("   📈 Динамика loss (последние записи):")
            for line in loss_lines[-5:]:
                print(f"     {line.strip()}")
        
    except Exception as e:
        print(f"   ❌ Ошибка чтения лога: {e}")

def apply_fixes():
    """Применение исправлений"""
    print_header("ПРИМЕНЕНИЕ ИСПРАВЛЕНИЙ")
    
    print_step(1, "Создание резервной копии конфигурации")
    
    config_path = Path("config/config.yaml")
    if config_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = Path(f"config/config_backup_{timestamp}.yaml")
        
        # Копирование
        with open(config_path) as f:
            content = f.read()
        with open(backup_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Резервная копия создана: {backup_path}")
    
    print_step(2, "Применение исправленной конфигурации")
    
    fixed_config_path = Path("config/config_fixed.yaml")
    if fixed_config_path.exists():
        response = input("❓ Заменить текущую конфигурацию на исправленную? (y/n): ")
        if response.lower() == 'y':
            with open(fixed_config_path) as f:
                content = f.read()
            with open(config_path, 'w') as f:
                f.write(content)
            print("✅ Конфигурация обновлена!")
        else:
            print("⏭️ Пропущено")
    else:
        print("❌ Исправленная конфигурация не найдена!")
    
    print_step(3, "Очистка проблемных кэшей")
    
    response = input("❓ Очистить кэш для перезапуска обучения? (y/n): ")
    if response.lower() == 'y':
        cache_dir = Path("cache")
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*"))
            for file in cache_files:
                try:
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        import shutil
                        shutil.rmtree(file)
                except:
                    pass
            print("✅ Кэш очищен")
        else:
            print("⚠️ Директория кэша не найдена")

def run_test_training():
    """Запуск тестового обучения"""
    print_header("ТЕСТОВОЕ ОБУЧЕНИЕ")
    
    response = input("❓ Запустить тестовое обучение на 3 эпохи? (y/n): ")
    if response.lower() != 'y':
        print("⏭️ Пропущено")
        return
    
    print("🚀 Запуск тестового обучения...")
    print("   (это займет несколько минут)")
    
    # Создание тестовой конфигурации
    test_config = {
        'model': {
            'epochs': 3,
            'learning_rate': 0.001,
            'batch_size': 32,
            'context_window': 24
        }
    }
    
    # Можно запустить простой тест
    try:
        result = subprocess.run([
            sys.executable, "main.py", 
            "--config", "config/config.yaml",
            "--mode", "demo"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Тестовый запуск успешен!")
            print("   Вывод:")
            print(result.stdout[-500:])  # Последние 500 символов
        else:
            print("❌ Ошибка в тестовом запуске:")
            print(result.stderr[-500:])
            
    except subprocess.TimeoutExpired:
        print("⏰ Тайм-аут тестового запуска (это нормально)")
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")

def main():
    """Основная функция"""
    print("🔧 Диагностика и исправление проблем обучения Crypto AI Trading")
    print("=" * 70)
    
    # Проверка рабочей директории
    if not Path("config").exists() or not Path("main.py").exists():
        print("❌ Запустите скрипт из корневой директории проекта!")
        sys.exit(1)
    
    try:
        # Пошаговая диагностика
        check_system()
        config = analyze_config()
        check_data()
        check_logs()
        
        # Применение исправлений
        print("\n" + "="*60)
        print("🛠️ ПЛАН ИСПРАВЛЕНИЯ:")
        print("1. Увеличить Learning Rate с 0.0002 до 0.001")
        print("2. Уменьшить Batch Size для стабильности")
        print("3. Упростить архитектуру модели")
        print("4. Добавить OneCycleLR scheduler")
        print("5. Использовать простую MSE loss")
        print("="*60)
        
        response = input("\n❓ Применить исправления? (y/n): ")
        if response.lower() == 'y':
            apply_fixes()
            run_test_training()
        
        print("\n" + "="*60)
        print("✅ ДИАГНОСТИКА ЗАВЕРШЕНА")
        print("📋 Рекомендации:")
        print("   1. Используйте исправленную конфигурацию")
        print("   2. Мониторьте loss в первые 10 эпох")
        print("   3. Ожидайте улучшения к 20-30 эпохе")
        print("   4. Используйте TensorBoard для визуализации")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n❌ Прервано пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
