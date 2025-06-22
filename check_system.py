#!/usr/bin/env python3
"""
Проверка готовности системы к работе
"""

import os
import sys
import psycopg2
import yaml
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

def check_postgres():
    """Проверка подключения к PostgreSQL"""
    try:
        config = yaml.safe_load(open('config/config.yaml'))
        db_config = config['database']
        
        conn = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['database'],
            user=db_config['user'],
            password=db_config['password']
        )
        
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM raw_market_data")
        count = cur.fetchone()[0]
        
        cur.close()
        conn.close()
        
        return True, f"✅ PostgreSQL работает ({count:,} записей)"
    except Exception as e:
        return False, f"❌ PostgreSQL не доступен: {str(e)}"

def check_files():
    """Проверка основных файлов"""
    required_files = [
        'main.py',
        'train_model.py',
        'run_interactive.py',
        'config/config.yaml',
        'models/patchtst.py',
        'data/feature_engineering.py'
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        return False, f"❌ Отсутствуют файлы: {', '.join(missing)}"
    return True, "✅ Все основные файлы на месте"

def check_dependencies():
    """Проверка Python зависимостей"""
    try:
        import torch
        import pandas
        import numpy
        import sklearn
        import ta
        return True, "✅ Основные зависимости установлены"
    except ImportError as e:
        return False, f"❌ Отсутствуют зависимости: {str(e)}"

def main():
    """Главная функция проверки"""
    console.print("\n[bold cyan]🔍 ПРОВЕРКА СИСТЕМЫ CRYPTO AI TRADING[/bold cyan]\n")
    
    # Создаем таблицу результатов
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Компонент", style="cyan", width=30)
    table.add_column("Статус", width=50)
    
    # Проверки
    checks = [
        ("PostgreSQL", check_postgres()),
        ("Файлы проекта", check_files()),
        ("Python зависимости", check_dependencies()),
    ]
    
    all_passed = True
    for name, (status, message) in checks:
        table.add_row(name, message)
        if not status:
            all_passed = False
    
    console.print(table)
    
    if all_passed:
        console.print("\n[bold green]✅ СИСТЕМА ГОТОВА К РАБОТЕ![/bold green]")
        console.print("\n[yellow]Запустите обучение:[/yellow]")
        console.print("[bold]python train_model.py[/bold]")
    else:
        console.print("\n[bold red]❌ ОБНАРУЖЕНЫ ПРОБЛЕМЫ[/bold red]")
        console.print("\n[yellow]Исправьте проблемы и запустите проверку снова[/yellow]")
        sys.exit(1)

if __name__ == "__main__":
    main()