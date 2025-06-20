#!/usr/bin/env python3
"""
Тестовый скрипт для проверки исправления дублированных методов
"""

import sys
import ast
import inspect

def test_duplicate_methods():
    """Проверяем, что в файле нет дублированных методов"""
    
    # Читаем файл
    with open('run_interactive.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Ищем все определения launch_gpu_training
    lines = content.split('\n')
    launch_gpu_training_lines = []
    
    for i, line in enumerate(lines, 1):
        if 'def launch_gpu_training(' in line:
            launch_gpu_training_lines.append(i)
    
    print(f"🔍 Найдено определений launch_gpu_training(): {len(launch_gpu_training_lines)}")
    
    if len(launch_gpu_training_lines) > 1:
        print(f"❌ ОШИБКА: Найдено {len(launch_gpu_training_lines)} дублированных методов на строках: {launch_gpu_training_lines}")
        return False
    elif len(launch_gpu_training_lines) == 1:
        print(f"✅ Отлично! Найден 1 метод launch_gpu_training() на строке {launch_gpu_training_lines[0]}")
        return True
    else:
        print("❌ ОШИБКА: Метод launch_gpu_training() не найден!")
        return False

def test_syntax():
    """Проверяем синтаксис файла"""
    try:
        with open('run_interactive.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Пытаемся разобрать синтаксис
        ast.parse(content)
        print("✅ Синтаксис корректен")
        return True
    except SyntaxError as e:
        print(f"❌ Синтаксическая ошибка: {e}")
        print(f"   Строка {e.lineno}: {e.text}")
        return False
    except Exception as e:
        print(f"❌ Ошибка при проверке синтаксиса: {e}")
        return False

def main():
    print("🚀 Проверка исправления дублированных методов")
    print("=" * 50)
    
    # Проверяем дублированные методы
    duplicates_ok = test_duplicate_methods()
    print()
    
    # Проверяем синтаксис
    syntax_ok = test_syntax()
    print()
    
    # Итоговый результат
    if duplicates_ok and syntax_ok:
        print("🎉 ВСЕ ПРОВЕРКИ ПРОШЛИ УСПЕШНО!")
        print("✅ Дублированные методы удалены")
        print("✅ Синтаксис корректен")
        print("✅ Файл готов к использованию")
    else:
        print("⚠️  ТРЕБУЕТСЯ ДОПОЛНИТЕЛЬНАЯ РАБОТА")
        if not duplicates_ok:
            print("❌ Все еще есть дублированные методы")
        if not syntax_ok:
            print("❌ Есть синтаксические ошибки")

if __name__ == "__main__":
    main()
