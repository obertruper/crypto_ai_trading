#!/usr/bin/env python3
"""
Универсальный скрипт обучения модели Crypto AI Trading
Запуск: python train_model.py
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Запускает полный цикл обучения модели"""
    
    # Проверяем, что мы в правильной директории
    if not os.path.exists('config/config.yaml'):
        print("❌ Ошибка: запустите скрипт из корневой директории проекта!")
        print("cd /Users/ruslan/PycharmProjects/LLM\\ TRANSFORM/crypto_ai_trading")
        sys.exit(1)
    
    print("="*80)
    print("🚀 CRYPTO AI TRADING - ОБУЧЕНИЕ МОДЕЛИ")
    print("="*80)
    print()
    print("📋 Этапы обучения:")
    print("1️⃣ Загрузка данных из PostgreSQL")
    print("2️⃣ Создание признаков с защитой от data leakage")
    print("3️⃣ Обучение модели PatchTST")
    print("4️⃣ Валидация и бэктестинг")
    print("5️⃣ Сохранение лучшей модели")
    print()
    print("⏱️  Примерное время: 30-60 минут")
    print("="*80)
    print()
    
    # Запускаем основной скрипт
    cmd = [sys.executable, "main.py", "--mode", "full", "--config", "config/config.yaml"]
    
    try:
        print("🔄 Начинаем обучение...\n")
        result = subprocess.run(cmd, check=True)
        
        print("\n" + "="*80)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("="*80)
        print()
        print("📊 Результаты:")
        print("  • Модель сохранена в: models_saved/best_model.pth")
        print("  • Логи обучения: experiments/logs/")
        print("  • Метрики: experiments/logs/*_metrics.csv")
        print()
        print("🚀 Что дальше?")
        print("  1. Проверьте результаты: python run_interactive.py")
        print("  2. Запустите бэктест: python main.py --mode backtest")
        print("  3. Настройте live торговлю в config/config.yaml")
        print()
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Ошибка при обучении!")
        print(f"Код ошибки: {e.returncode}")
        print("\n💡 Возможные решения:")
        print("  1. Проверьте подключение к PostgreSQL (порт 5555)")
        print("  2. Убедитесь, что есть данные в БД")
        print("  3. Проверьте логи в experiments/logs/")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⚠️ Обучение прервано пользователем")
        print("💡 Вы можете продолжить обучение позже, запустив скрипт снова")
        sys.exit(0)

if __name__ == "__main__":
    main()