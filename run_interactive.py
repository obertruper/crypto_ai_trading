#!/usr/bin/env python3
"""
Интерактивное меню для управления Crypto AI Trading System
"""

import sys
import os
import subprocess
import yaml
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Настройка логирования
import logging
log_dir = Path("logs/interactive")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"menu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
    ]
)
menu_logger = logging.getLogger('interactive_menu')

# Импорты для интерактивного меню
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from rich.markdown import Markdown

console = Console()

class CryptoTradingMenu:
    """Интерактивное меню для управления системой"""
    
    def __init__(self):
        self.console = Console()
        self.config_path = Path("config/config.yaml")
        self.config = self.load_config()
        self.logger = menu_logger
        self.logger.info(f"Запуск интерактивного меню. Лог файл: {log_file}")
        
    def load_config(self) -> Dict:
        """Загрузка конфигурации"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def save_config(self):
        """Сохранение конфигурации"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def display_main_menu(self):
        """Отображение главного меню"""
        self.console.clear()
        
        # Заголовок
        self.console.print(Panel.fit(
            "[bold cyan]🚀 Crypto AI Trading System[/bold cyan]\n"
            "[dim]Интерактивное управление[/dim]",
            border_style="blue"
        ))
        
        # Опции меню
        menu_options = Table(show_header=False, box=None)
        menu_options.add_column("Option", style="cyan", width=3)
        menu_options.add_column("Description", style="white")
        
        menu_options.add_row("1", "📊 Управление данными")
        menu_options.add_row("2", "🧠 Обучение моделей")
        menu_options.add_row("3", "📈 Мониторинг и логи")
        menu_options.add_row("4", "⚙️  Настройки системы")
        menu_options.add_row("5", "🔧 Утилиты и инструменты")
        menu_options.add_row("6", "📋 Информация о системе")
        menu_options.add_row("7", "🚀 Быстрый запуск")
        menu_options.add_row("0", "❌ Выход")
        
        self.console.print(menu_options)
        
    def data_management_menu(self):
        """Меню управления данными"""
        while True:
            self.console.clear()
            self.console.print(Panel("📊 Управление данными", style="cyan"))
            
            options = Table(show_header=False, box=None)
            options.add_column("Option", style="cyan", width=3)
            options.add_column("Description", style="white")
            
            options.add_row("1", "Проверить подключение к БД")
            options.add_row("2", "Загрузить данные из БД")
            options.add_row("3", "Создать/обновить признаки")
            options.add_row("4", "Статистика по данным")
            options.add_row("5", "Очистить кэш")
            options.add_row("0", "Назад")
            
            self.console.print(options)
            
            choice = Prompt.ask("\n[bold cyan]Выберите опцию[/bold cyan]")
            
            if choice == "1":
                self.check_database_connection()
            elif choice == "2":
                self.load_data_from_db()
            elif choice == "3":
                self.create_features()
            elif choice == "4":
                self.show_data_statistics()
            elif choice == "5":
                self.clear_cache()
            elif choice == "0":
                break
    
    def training_menu(self):
        """Меню обучения моделей"""
        while True:
            self.console.clear()
            self.console.print(Panel("🧠 Обучение моделей", style="cyan"))
            
            options = Table(show_header=False, box=None)
            options.add_column("Option", style="cyan", width=3)
            options.add_column("Description", style="white")
            
            options.add_row("1", "Быстрое обучение (demo)")
            options.add_row("2", "Полное обучение")
            options.add_row("3", "Продолжить обучение")
            options.add_row("4", "Настройки обучения")
            options.add_row("5", "Обучение на GPU (Vast.ai)")
            options.add_row("0", "Назад")
            
            self.console.print(options)
            
            choice = Prompt.ask("\n[bold cyan]Выберите опцию[/bold cyan]")
            
            if choice == "1":
                self.run_demo_training()
            elif choice == "2":
                self.run_full_training()
            elif choice == "3":
                self.resume_training()
            elif choice == "4":
                self.configure_training()
            elif choice == "5":
                self.run_gpu_training()
            elif choice == "0":
                break
    
    def monitoring_menu(self):
        """Меню мониторинга"""
        while True:
            self.console.clear()
            self.console.print(Panel("📈 Мониторинг и логи", style="cyan"))
            
            options = Table(show_header=False, box=None)
            options.add_column("Option", style="cyan", width=3)
            options.add_column("Description", style="white")
            
            options.add_row("1", "Мониторинг обучения в реальном времени")
            options.add_row("2", "Просмотр логов")
            options.add_row("3", "TensorBoard")
            options.add_row("4", "Результаты последнего обучения")
            options.add_row("5", "Очистить логи")
            options.add_row("0", "Назад")
            
            self.console.print(options)
            
            choice = Prompt.ask("\n[bold cyan]Выберите опцию[/bold cyan]")
            
            if choice == "1":
                self.monitor_training()
            elif choice == "2":
                self.view_logs()
            elif choice == "3":
                self.launch_tensorboard()
            elif choice == "4":
                self.show_training_results()
            elif choice == "5":
                self.clear_logs()
            elif choice == "0":
                break
    
    def settings_menu(self):
        """Меню настроек"""
        while True:
            self.console.clear()
            self.console.print(Panel("⚙️ Настройки системы", style="cyan"))
            
            options = Table(show_header=False, box=None)
            options.add_column("Option", style="cyan", width=3)
            options.add_column("Description", style="white")
            
            options.add_row("1", "Настройки модели")
            options.add_row("2", "Настройки данных")
            options.add_row("3", "Настройки риск-менеджмента")
            options.add_row("4", "Параметры бэктестинга")
            options.add_row("5", "Сохранить конфигурацию")
            options.add_row("6", "Загрузить конфигурацию")
            options.add_row("0", "Назад")
            
            self.console.print(options)
            
            choice = Prompt.ask("\n[bold cyan]Выберите опцию[/bold cyan]")
            
            if choice == "1":
                self.configure_model()
            elif choice == "2":
                self.configure_data()
            elif choice == "3":
                self.configure_risk_management()
            elif choice == "4":
                self.configure_backtesting()
            elif choice == "5":
                self.save_config()
                self.console.print("[green]✅ Конфигурация сохранена[/green]")
                time.sleep(1)
            elif choice == "6":
                self.config = self.load_config()
                self.console.print("[green]✅ Конфигурация загружена[/green]")
                time.sleep(1)
            elif choice == "0":
                break
    
    def utilities_menu(self):
        """Меню утилит"""
        while True:
            self.console.clear()
            self.console.print(Panel("🔧 Утилиты и инструменты", style="cyan"))
            
            options = Table(show_header=False, box=None)
            options.add_column("Option", style="cyan", width=3)
            options.add_column("Description", style="white")
            
            options.add_row("1", "Проверить доступные символы")
            options.add_row("2", "Валидация данных")
            options.add_row("3", "Бэктестинг стратегии")
            options.add_row("4", "Генерация отчетов")
            options.add_row("5", "Экспорт модели")
            options.add_row("0", "Назад")
            
            self.console.print(options)
            
            choice = Prompt.ask("\n[bold cyan]Выберите опцию[/bold cyan]")
            
            if choice == "1":
                self.check_available_symbols()
            elif choice == "2":
                self.validate_data()
            elif choice == "3":
                self.run_backtesting()
            elif choice == "4":
                self.generate_reports()
            elif choice == "5":
                self.export_model()
            elif choice == "0":
                break
    
    # Реализация функций
    def check_database_connection(self):
        """Проверка подключения к БД"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Проверка подключения к БД...", total=None)
            
            try:
                result = subprocess.run(
                    ["python", "-c", """
import psycopg2
conn = psycopg2.connect(
    host='localhost',
    port=5555,
    database='crypto_trading',
    user='ruslan',
    password='ruslan'
)
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM raw_market_data')
count = cursor.fetchone()[0]
print(f'✅ Подключение успешно! Записей в БД: {count:,}')
conn.close()
                    """],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    self.console.print(f"\n[green]{result.stdout.strip()}[/green]")
                else:
                    self.console.print(f"\n[red]❌ Ошибка подключения: {result.stderr}[/red]")
            except Exception as e:
                self.console.print(f"\n[red]❌ Ошибка: {e}[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def load_data_from_db(self):
        """Загрузка данных из БД"""
        symbols = self.config['data']['symbols']
        self.console.print(f"\n[cyan]Будут загружены данные для {len(symbols)} символов[/cyan]")
        
        if Confirm.ask("Продолжить?"):
            result = subprocess.run(
                ["python", "run_full_pipeline.py", "--mode", "data"],
                text=True
            )
            if result.returncode == 0:
                self.console.print("[green]✅ Данные успешно загружены[/green]")
            else:
                self.console.print("[red]❌ Ошибка при загрузке данных[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def create_features(self):
        """Создание признаков"""
        self.console.print("\n[cyan]Создание технических индикаторов и признаков...[/cyan]")
        
        if Confirm.ask("Пересоздать кэш признаков?"):
            # Удаляем старый кэш
            cache_file = Path("cache/features_cache.pkl")
            if cache_file.exists():
                cache_file.unlink()
                self.console.print("[yellow]Старый кэш удален[/yellow]")
            
            # Запускаем создание признаков
            result = subprocess.run(
                ["python", "run_full_pipeline.py", "--mode", "data"],
                text=True
            )
            if result.returncode == 0:
                self.console.print("[green]✅ Признаки успешно созданы[/green]")
            else:
                self.console.print("[red]❌ Ошибка при создании признаков[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def show_data_statistics(self):
        """Показать статистику по данным"""
        try:
            result = subprocess.run(
                ["python", "-c", """
import pickle
import pandas as pd

try:
    with open('cache/features_cache.pkl', 'rb') as f:
        features = pickle.load(f)
    
    print(f'\\n📊 Статистика данных:')
    print(f'Всего записей: {len(features):,}')
    print(f'Период: {features.datetime.min()} - {features.datetime.max()}')
    print(f'Количество признаков: {len(features.columns)}')
    
    print(f'\\nРаспределение по символам:')
    for symbol, count in features.groupby('symbol').size().items():
        print(f'  {symbol}: {count:,} записей')
    
    print(f'\\nЦелевые переменные:')
    target_cols = [col for col in features.columns if col.startswith('future_return_')]
    for col in target_cols:
        print(f'  {col}: mean={features[col].mean():.4f}, std={features[col].std():.4f}')
except:
    print('❌ Кэш данных не найден. Сначала создайте признаки.')
                """],
                capture_output=True,
                text=True
            )
            self.console.print(result.stdout)
        except Exception as e:
            self.console.print(f"[red]❌ Ошибка: {e}[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def clear_cache(self):
        """Очистка кэша"""
        cache_dir = Path("cache")
        if cache_dir.exists():
            files = list(cache_dir.glob("*.pkl"))
            if files:
                self.console.print(f"\n[yellow]Найдено {len(files)} файлов кэша[/yellow]")
                if Confirm.ask("Удалить все файлы кэша?"):
                    for file in files:
                        file.unlink()
                    self.console.print("[green]✅ Кэш очищен[/green]")
            else:
                self.console.print("[yellow]Кэш пуст[/yellow]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def run_demo_training(self):
        """Быстрое демо обучение"""
        self.console.print("\n[cyan]Запуск демо обучения (5 эпох)...[/cyan]")
        
        # Временно меняем конфигурацию
        original_epochs = self.config['model']['epochs']
        self.config['model']['epochs'] = 5
        self.save_config()
        
        try:
            # Запускаем обучение в фоне
            process = subprocess.Popen(
                ["python", "run_full_pipeline.py", "--mode", "train"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Показываем вывод в реальном времени
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line.strip())
            
            process.wait()
            
            if process.returncode == 0:
                self.console.print("\n[green]✅ Демо обучение завершено[/green]")
            else:
                self.console.print("\n[red]❌ Ошибка при обучении[/red]")
        finally:
            # Восстанавливаем конфигурацию
            self.config['model']['epochs'] = original_epochs
            self.save_config()
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def run_full_training(self):
        """Полное обучение"""
        self.console.print(f"\n[cyan]Запуск полного обучения ({self.config['model']['epochs']} эпох)...[/cyan]")
        
        if Confirm.ask("Это может занять много времени. Продолжить?"):
            # Запускаем обучение в фоне
            subprocess.Popen(
                ["python", "run_full_pipeline.py", "--mode", "train"],
                stdout=open("logs/training.log", "w"),
                stderr=subprocess.STDOUT
            )
            
            self.console.print("[green]✅ Обучение запущено в фоновом режиме[/green]")
            self.console.print("[yellow]Используйте 'Мониторинг обучения' для отслеживания прогресса[/yellow]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def resume_training(self):
        """Продолжить обучение с последней сохраненной модели"""
        self.console.print("\n[cyan]Продолжение обучения с последней контрольной точки...[/cyan]")
        
        # Проверяем наличие сохраненных моделей
        models_dir = Path("models_saved")
        if not models_dir.exists() or not list(models_dir.glob("*.pth")):
            self.console.print("[yellow]⚠️ Сохраненные модели не найдены[/yellow]")
            Prompt.ask("\nНажмите Enter для продолжения")
            return
        
        # Находим последнюю модель
        latest_model = max(models_dir.glob("*.pth"), key=lambda x: x.stat().st_mtime)
        self.console.print(f"[green]Найдена модель: {latest_model.name}[/green]")
        
        if Confirm.ask("\nПродолжить обучение с этой модели?"):
            # TODO: Реализовать продолжение обучения
            self.console.print("[yellow]Функция в разработке[/yellow]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def configure_training(self):
        """Настройка параметров обучения"""
        self.console.print("\n[cyan]⚙️ Настройки обучения:[/cyan]")
        
        # Показываем текущие настройки
        table = Table(title="Текущие параметры обучения")
        table.add_column("Параметр", style="cyan")
        table.add_column("Значение", style="yellow")
        
        table.add_row("Эпохи", str(self.config['model']['epochs']))
        table.add_row("Batch size", str(self.config['model']['batch_size']))
        table.add_row("Learning rate", str(self.config['model']['learning_rate']))
        table.add_row("Early stopping", str(self.config['model']['early_stopping_patience']))
        table.add_row("Устройство", "GPU" if self.config['performance']['device'] == 'cuda' else "CPU")
        
        self.console.print(table)
        
        if Confirm.ask("\nИзменить параметры?"):
            self.config['model']['epochs'] = IntPrompt.ask("Количество эпох", default=self.config['model']['epochs'])
            self.config['model']['batch_size'] = IntPrompt.ask("Batch size", default=self.config['model']['batch_size'])
            self.config['model']['learning_rate'] = FloatPrompt.ask("Learning rate", default=self.config['model']['learning_rate'])
            self.config['model']['early_stopping_patience'] = IntPrompt.ask("Early stopping patience", default=self.config['model']['early_stopping_patience'])
            
            device_choice = Prompt.ask("Устройство (cpu/cuda)", default=self.config['performance']['device'])
            self.config['performance']['device'] = device_choice
            
            self.save_config()
            self.console.print("[green]✅ Параметры обучения обновлены[/green]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    
    def launch_gpu_training(self):
        """Запуск обучения на GPU"""
        self.console.print("\n[cyan]🚀 Запуск обучения на GPU сервере[/cyan]")
        
        # Выбор режима
        self.console.print("\n[cyan]Выберите режим обучения:[/cyan]")
        self.console.print("1. Демо (5 эпох) - ~15-20 минут")
        self.console.print("2. Стандартное (50 эпох) - ~2-3 часа")
        self.console.print("3. Полное (100 эпох) - ~5-6 часов")
        self.console.print("4. Пользовательское")
        
        choice = Prompt.ask("Выбор", default="1")
        
        epochs = {
            "1": 5,
            "2": 50,
            "3": 100
        }.get(choice)
        
        if choice == "4":
            epochs = IntPrompt.ask("Количество эпох", default=20)
        
        if epochs:
            # Временно обновляем конфигурацию
            original_epochs = self.config['model']['epochs']
            self.config['model']['epochs'] = epochs
            self.save_config()
            
            self.console.print(f"\n[yellow]Запуск обучения на {epochs} эпох...[/yellow]")
            
            script_path = "scripts/run_on_vast.sh"
            if os.path.exists(script_path):
                try:
                    subprocess.run([script_path])
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]Прервано пользователем[/yellow]")
            else:
                self.console.print("[red]❌ Скрипт запуска не найден[/red]")
            
            # Восстанавливаем конфигурацию
            self.config['model']['epochs'] = original_epochs
            self.save_config()
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def monitor_gpu(self):
        """Мониторинг GPU сервера"""
        self.console.print("\n[cyan]📊 Команды для мониторинга GPU:[/cyan]")
        self.console.print("\nПосле подключения к серверу используйте:")
        self.console.print("   • nvidia-smi -l 1        # Мониторинг GPU в реальном времени")
        self.console.print("   • htop                   # Мониторинг CPU и памяти")
        self.console.print("   • tmux attach -t training # Подключение к сессии обучения")
        self.console.print("   • tail -f logs/training_gpu.log # Просмотр логов")
        self.console.print("\n[yellow]Для TensorBoard откройте http://localhost:6006[/yellow]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def show_gpu_instructions(self):
        """Показать инструкции по GPU"""
        self.console.print("\n[cyan]📋 Инструкция по работе с GPU сервером[/cyan]")
        
        instructions = """
1. **Первоначальная настройка:**
   - Создайте SSH ключ: ~/.ssh/vast_ai_key
   - Установите права: chmod 600 ~/.ssh/vast_ai_key
   
2. **Рабочий процесс:**
   - Синхронизируйте проект (опция 1)
   - Подключитесь к серверу (опция 2)
   - Запустите обучение (опция 3)
   
3. **Мониторинг:**
   - TensorBoard: http://localhost:6006
   - GPU статус: nvidia-smi -l 1
   - Логи: tail -f logs/training_gpu.log
   
4. **Управление tmux:**
   - Подключиться: tmux attach -t training
   - Отключиться: Ctrl+B, затем D
   - Список сессий: tmux ls
        """
        
        self.console.print(Markdown(instructions))
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def configure_gpu_server(self):
        """Настройка GPU сервера"""
        self.console.print("\n[cyan]⚙️ Настройка GPU сервера[/cyan]")
        
        remote_config = self.config.setdefault('remote_server', {})
        
        self.console.print("\nТекущие настройки:")
        self.console.print(f"Host: {remote_config.get('primary', {}).get('host', 'Не задан')}")
        self.console.print(f"Port: {remote_config.get('primary', {}).get('port', 'Не задан')}")
        
        if Confirm.ask("\nИзменить настройки?"):
            host = Prompt.ask("IP адрес сервера", default="114.32.64.6")
            port = IntPrompt.ask("SSH порт", default=40134)
            
            remote_config['enabled'] = True
            remote_config.setdefault('primary', {})['host'] = host
            remote_config.setdefault('primary', {})['port'] = port
            remote_config['user'] = 'root'
            remote_config['key_path'] = '~/.ssh/vast_ai_key'
            
            self.save_config()
            self.console.print("[green]✅ Настройки сохранены[/green]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def monitor_training(self):
        """Мониторинг обучения"""
        self.console.print("\n[cyan]Запуск мониторинга обучения...[/cyan]")
        
        try:
            subprocess.run(["python", "monitor_training.py"])
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Мониторинг остановлен[/yellow]")
        except Exception as e:
            self.console.print(f"[red]❌ Ошибка: {e}[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def view_logs(self):
        """Просмотр логов"""
        log_dir = Path("experiments/logs")
        if log_dir.exists():
            log_files = sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
            
            if log_files:
                self.console.print("\n[cyan]Доступные логи:[/cyan]")
                for i, file in enumerate(log_files[:10], 1):
                    size = file.stat().st_size / 1024  # KB
                    mtime = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    self.console.print(f"{i}. {file.name} ({size:.1f} KB, {mtime})")
                
                choice = IntPrompt.ask("\nВыберите файл (0 для отмены)", default=0)
                
                if 0 < choice <= len(log_files[:10]):
                    log_file = log_files[choice - 1]
                    
                    # Показываем последние строки
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        last_lines = lines[-50:] if len(lines) > 50 else lines
                        
                        self.console.print(f"\n[cyan]Последние строки из {log_file.name}:[/cyan]")
                        for line in last_lines:
                            print(line.rstrip())
            else:
                self.console.print("[yellow]Логи не найдены[/yellow]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def launch_tensorboard(self):
        """Запуск TensorBoard"""
        tb_dir = Path("experiments/runs")
        if tb_dir.exists():
            self.console.print("\n[cyan]Запуск TensorBoard...[/cyan]")
            self.console.print("[yellow]Откройте http://localhost:6006 в браузере[/yellow]")
            self.console.print("[yellow]Нажмите Ctrl+C для остановки[/yellow]\n")
            
            try:
                subprocess.run(["tensorboard", "--logdir", str(tb_dir)])
            except KeyboardInterrupt:
                self.console.print("\n[yellow]TensorBoard остановлен[/yellow]")
        else:
            self.console.print("[yellow]Директория TensorBoard не найдена[/yellow]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def configure_model(self):
        """Настройка параметров модели"""
        self.console.print("\n[cyan]Настройки модели PatchTST:[/cyan]")
        
        model_config = self.config['model']
        
        # Показываем текущие настройки
        table = Table(title="Текущие параметры")
        table.add_column("Параметр", style="cyan")
        table.add_column("Значение", style="yellow")
        
        table.add_row("d_model", str(model_config['d_model']))
        table.add_row("n_heads", str(model_config['n_heads']))
        table.add_row("e_layers", str(model_config['e_layers']))
        table.add_row("d_ff", str(model_config['d_ff']))
        table.add_row("dropout", str(model_config['dropout']))
        table.add_row("batch_size", str(model_config['batch_size']))
        table.add_row("learning_rate", str(model_config['learning_rate']))
        table.add_row("epochs", str(model_config['epochs']))
        
        self.console.print(table)
        
        if Confirm.ask("\nИзменить параметры?"):
            model_config['d_model'] = IntPrompt.ask("d_model", default=model_config['d_model'])
            model_config['n_heads'] = IntPrompt.ask("n_heads", default=model_config['n_heads'])
            model_config['e_layers'] = IntPrompt.ask("e_layers", default=model_config['e_layers'])
            model_config['d_ff'] = IntPrompt.ask("d_ff", default=model_config['d_ff'])
            model_config['dropout'] = FloatPrompt.ask("dropout", default=model_config['dropout'])
            model_config['batch_size'] = IntPrompt.ask("batch_size", default=model_config['batch_size'])
            model_config['learning_rate'] = FloatPrompt.ask("learning_rate", default=model_config['learning_rate'])
            model_config['epochs'] = IntPrompt.ask("epochs", default=model_config['epochs'])
            
            self.save_config()
            self.console.print("[green]✅ Параметры модели обновлены[/green]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def configure_data(self):
        """Настройка параметров данных"""
        self.console.print("\n[cyan]Настройки данных:[/cyan]")
        
        data_config = self.config['data']
        
        # Показываем текущие символы
        self.console.print(f"\nТекущие символы ({len(data_config['symbols'])}):")
        for symbol in data_config['symbols'][:10]:
            self.console.print(f"  • {symbol}")
        if len(data_config['symbols']) > 10:
            self.console.print(f"  ... и еще {len(data_config['symbols']) - 10}")
        
        if Confirm.ask("\nИзменить список символов?"):
            self.console.print("\nВыберите опцию:")
            self.console.print("1. Топ-10 монет")
            self.console.print("2. Топ-20 монет")
            self.console.print("3. Все доступные символы")
            self.console.print("4. Ввести вручную")
            
            choice = Prompt.ask("Выбор")
            
            if choice == "1":
                data_config['symbols'] = [
                    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
                    "ADAUSDT", "DOTUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT"
                ]
            elif choice == "2":
                data_config['symbols'] = [
                    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
                    "ADAUSDT", "DOTUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT",
                    "MATICUSDT", "UNIUSDT", "ATOMUSDT", "XLMUSDT", "ETCUSDT",
                    "NEARUSDT", "ALGOUSDT", "FILUSDT", "ICPUSDT", "TRXUSDT"
                ]
            elif choice == "3":
                data_config['symbols'] = ["ALL"]
            elif choice == "4":
                symbols_str = Prompt.ask("Введите символы через запятую")
                data_config['symbols'] = [s.strip() for s in symbols_str.split(",")]
            
            self.save_config()
            self.console.print("[green]✅ Список символов обновлен[/green]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def check_available_symbols(self):
        """Проверка доступных символов"""
        self.console.print("\n[cyan]Проверка доступных символов в БД...[/cyan]")
        
        try:
            result = subprocess.run(
                ["python", "-c", """
import psycopg2
import pandas as pd

conn = psycopg2.connect(
    host='localhost', port=5555, database='crypto_trading',
    user='ruslan', password='ruslan'
)

query = '''
SELECT DISTINCT symbol, COUNT(*) as records, 
       MIN(datetime) as first_date, MAX(datetime) as last_date
FROM raw_market_data
GROUP BY symbol
ORDER BY symbol
'''

df = pd.read_sql(query, conn)
conn.close()

print(f'\\nНайдено {len(df)} символов:\\n')
for _, row in df.iterrows():
    print(f"{row['symbol']:15} {row['records']:8,} записей  ({row['first_date'].strftime('%Y-%m-%d')} - {row['last_date'].strftime('%Y-%m-%d')})")
                """],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.console.print(result.stdout)
            else:
                self.console.print(f"[red]❌ Ошибка: {result.stderr}[/red]")
        except Exception as e:
            self.console.print(f"[red]❌ Ошибка: {e}[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def validate_data(self):
        """Валидация данных"""
        self.console.print("\n[cyan]Валидация данных...[/cyan]")
        
        cache_file = Path("cache/features_cache.pkl")
        if not cache_file.exists():
            self.console.print("[yellow]⚠️ Кэш данных не найден[/yellow]")
            Prompt.ask("\nНажмите Enter для продолжения")
            return
        
        try:
            import pickle
            with open(cache_file, 'rb') as f:
                features = pickle.load(f)
            
            self.console.print(f"\n[green]✅ Данные загружены успешно[/green]")
            self.console.print(f"Размер: {features.shape}")
            self.console.print(f"Период: {features.datetime.min()} - {features.datetime.max()}")
            
            # Проверка на NaN
            nan_count = features.isna().sum().sum()
            if nan_count > 0:
                self.console.print(f"\n[yellow]⚠️ Найдено {nan_count} NaN значений[/yellow]")
            else:
                self.console.print(f"\n[green]✅ NaN значений нет[/green]")
            
            # Проверка целевых переменных
            target_cols = [col for col in features.columns if col.startswith('future_return_')]
            self.console.print(f"\n[cyan]Целевые переменные ({len(target_cols)}):[/cyan]")
            for col in target_cols:
                mean_val = features[col].mean()
                std_val = features[col].std()
                self.console.print(f"  {col}: mean={mean_val:.4f}, std={std_val:.4f}")
        
        except Exception as e:
            self.console.print(f"[red]❌ Ошибка при валидации: {e}[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def run_backtesting(self):
        """Запуск бэктестинга"""
        self.console.print("\n[cyan]Бэктестинг стратегии[/cyan]")
        
        models_dir = Path("models_saved")
        if not models_dir.exists() or not list(models_dir.glob("*.pth")):
            self.console.print("[yellow]⚠️ Сохраненные модели не найдены[/yellow]")
            self.console.print("Сначала обучите модель")
            Prompt.ask("\nНажмите Enter для продолжения")
            return
        
        # Выбор модели
        models = sorted(models_dir.glob("*.pth"), key=lambda x: x.stat().st_mtime, reverse=True)
        self.console.print("\n[cyan]Доступные модели:[/cyan]")
        for i, model in enumerate(models[:5], 1):
            self.console.print(f"{i}. {model.name}")
        
        choice = IntPrompt.ask("\nВыберите модель (0 для отмены)", default=1)
        
        if 0 < choice <= len(models[:5]):
            selected_model = models[choice - 1]
            self.console.print(f"\n[yellow]Запуск бэктестинга с моделью {selected_model.name}...[/yellow]")
            
            if Confirm.ask("Запустить бэктестинг?"):
                subprocess.run([
                    "python", "run_full_pipeline.py", 
                    "--mode", "backtest",
                    "--model", str(selected_model)
                ])
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def generate_reports(self):
        """Генерация отчетов"""
        self.console.print("\n[cyan]Генерация отчетов[/cyan]")
        
        self.console.print("\n[cyan]Выберите тип отчета:[/cyan]")
        self.console.print("1. Отчет по обучению")
        self.console.print("2. Отчет по бэктестингу")
        self.console.print("3. Анализ признаков")
        self.console.print("4. Полный отчет")
        
        choice = Prompt.ask("Выбор", default="1")
        
        if choice == "1":
            # Отчет по обучению
            log_dir = Path("experiments/logs")
            if log_dir.exists():
                self.console.print("\n[yellow]Генерация отчета по обучению...[/yellow]")
                # TODO: Реализовать генерацию отчета
                self.console.print("[yellow]Функция в разработке[/yellow]")
        
        elif choice == "2":
            # Отчет по бэктестингу
            self.console.print("\n[yellow]Генерация отчета по бэктестингу...[/yellow]")
            # TODO: Реализовать
            self.console.print("[yellow]Функция в разработке[/yellow]")
        
        elif choice == "3":
            # Анализ признаков
            self.console.print("\n[yellow]Анализ важности признаков...[/yellow]")
            # TODO: Реализовать
            self.console.print("[yellow]Функция в разработке[/yellow]")
        
        elif choice == "4":
            # Полный отчет
            self.console.print("\n[yellow]Генерация полного отчета...[/yellow]")
            # TODO: Реализовать
            self.console.print("[yellow]Функция в разработке[/yellow]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def export_model(self):
        """Экспорт модели"""
        self.console.print("\n[cyan]Экспорт модели[/cyan]")
        
        models_dir = Path("models_saved")
        if not models_dir.exists() or not list(models_dir.glob("*.pth")):
            self.console.print("[yellow]⚠️ Сохраненные модели не найдены[/yellow]")
            Prompt.ask("\nНажмите Enter для продолжения")
            return
        
        # Выбор модели
        models = sorted(models_dir.glob("*.pth"), key=lambda x: x.stat().st_mtime, reverse=True)
        self.console.print("\n[cyan]Доступные модели:[/cyan]")
        for i, model in enumerate(models[:5], 1):
            size = model.stat().st_size / (1024 * 1024)  # MB
            self.console.print(f"{i}. {model.name} ({size:.1f} MB)")
        
        choice = IntPrompt.ask("\nВыберите модель для экспорта (0 для отмены)", default=1)
        
        if 0 < choice <= len(models[:5]):
            selected_model = models[choice - 1]
            
            self.console.print("\n[cyan]Формат экспорта:[/cyan]")
            self.console.print("1. ONNX (универсальный)")
            self.console.print("2. TorchScript")
            self.console.print("3. Копировать файл")
            
            format_choice = Prompt.ask("Выбор", default="3")
            
            if format_choice == "3":
                # Простое копирование
                export_dir = Path("exports")
                export_dir.mkdir(exist_ok=True)
                
                export_path = export_dir / f"model_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                
                import shutil
                shutil.copy2(selected_model, export_path)
                
                self.console.print(f"\n[green]✅ Модель экспортирована: {export_path}[/green]")
            else:
                self.console.print("[yellow]Функция в разработке[/yellow]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def configure_risk_management(self):
        """Настройка риск-менеджмента"""
        self.console.print("\n[cyan]⚙️ Настройки риск-менеджмента:[/cyan]")
        
        risk_config = self.config['risk_management']
        
        # Показываем текущие настройки
        table = Table(title="Текущие параметры риск-менеджмента")
        table.add_column("Параметр", style="cyan")
        table.add_column("Значение", style="yellow")
        
        table.add_row("Stop Loss %", f"{risk_config['stop_loss_pct']}%")
        table.add_row("Take Profit уровни", str(risk_config['take_profit_targets']))
        table.add_row("Частичные закрытия", str(risk_config['partial_close_sizes']))
        table.add_row("Макс риск на сделку", f"{risk_config['position_sizing']['max_risk_per_trade']}%")
        table.add_row("Метод размера позиции", risk_config['position_sizing']['method'])
        
        self.console.print(table)
        
        if Confirm.ask("\nИзменить параметры?"):
            risk_config['stop_loss_pct'] = FloatPrompt.ask("Stop Loss %", default=risk_config['stop_loss_pct'])
            
            # Take Profit уровни
            tp_str = Prompt.ask("Take Profit уровни (через запятую)", default=",".join(map(str, risk_config['take_profit_targets'])))
            risk_config['take_profit_targets'] = [float(x.strip()) for x in tp_str.split(",")]
            
            # Частичные закрытия
            pc_str = Prompt.ask("Частичные закрытия % (через запятую)", default=",".join(map(str, risk_config['partial_close_sizes'])))
            risk_config['partial_close_sizes'] = [int(x.strip()) for x in pc_str.split(",")]
            
            risk_config['position_sizing']['max_risk_per_trade'] = FloatPrompt.ask(
                "Макс риск на сделку %", 
                default=risk_config['position_sizing']['max_risk_per_trade']
            )
            
            self.save_config()
            self.console.print("[green]✅ Параметры риск-менеджмента обновлены[/green]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def configure_backtesting(self):
        """Настройка параметров бэктестинга"""
        self.console.print("\n[cyan]⚙️ Настройки бэктестинга:[/cyan]")
        
        backtest_config = self.config['backtesting']
        
        # Показываем текущие настройки
        table = Table(title="Текущие параметры бэктестинга")
        table.add_column("Параметр", style="cyan")
        table.add_column("Значение", style="yellow")
        
        table.add_row("Начальный капитал", f"${backtest_config['initial_capital']:,}")
        table.add_row("Комиссия", f"{backtest_config['commission']*100:.1f}%")
        table.add_row("Проскальзывание", f"{backtest_config['slippage']*100:.2f}%")
        
        self.console.print(table)
        
        if Confirm.ask("\nИзменить параметры?"):
            backtest_config['initial_capital'] = IntPrompt.ask(
                "Начальный капитал $", 
                default=backtest_config['initial_capital']
            )
            
            commission_pct = FloatPrompt.ask(
                "Комиссия %", 
                default=backtest_config['commission']*100
            )
            backtest_config['commission'] = commission_pct / 100
            
            slippage_pct = FloatPrompt.ask(
                "Проскальзывание %", 
                default=backtest_config['slippage']*100
            )
            backtest_config['slippage'] = slippage_pct / 100
            
            self.save_config()
            self.console.print("[green]✅ Параметры бэктестинга обновлены[/green]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def show_training_results(self):
        """Показать результаты последнего обучения"""
        self.console.print("\n[cyan]📊 Результаты последнего обучения[/cyan]")
        
        # Находим последнюю директорию с обучением
        exp_dir = Path("experiments/runs")
        if not exp_dir.exists():
            self.console.print("[yellow]⚠️ Результаты обучения не найдены[/yellow]")
            Prompt.ask("\nНажмите Enter для продолжения")
            return
        
        training_dirs = sorted(exp_dir.glob("training_*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not training_dirs:
            self.console.print("[yellow]⚠️ Результаты обучения не найдены[/yellow]")
            Prompt.ask("\nНажмите Enter для продолжения")
            return
        
        latest_dir = training_dirs[0]
        self.console.print(f"\n[green]Последнее обучение: {latest_dir.name}[/green]")
        
        # Читаем финальный отчет если есть
        report_file = latest_dir / "final_report.txt"
        if report_file.exists():
            with open(report_file, 'r') as f:
                self.console.print("\n[cyan]Финальный отчет:[/cyan]")
                self.console.print(f.read())
        
        # Проверяем метрики
        metrics_files = list(latest_dir.glob("*_metrics.csv"))
        if metrics_files:
            self.console.print("\n[cyan]Файлы метрик:[/cyan]")
            for file in metrics_files:
                self.console.print(f"  • {file.name}")
        
        # Проверяем графики
        plots_dir = latest_dir / "plots"
        if plots_dir.exists():
            plots = list(plots_dir.glob("*.png"))
            if plots:
                self.console.print(f"\n[cyan]Найдено {len(plots)} графиков[/cyan]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def run_gpu_training(self):
        """Обучение на GPU сервере"""
        while True:
            self.console.clear()
            self.console.print(Panel("🚀 Обучение на GPU сервере (Vast.ai)", style="cyan"))
            
            # Информация о сервере
            info_table = Table(show_header=False, box=None)
            info_table.add_column("Parameter", style="cyan")
            info_table.add_column("Value", style="white")
            
            info_table.add_row("📊 Информация о сервере:", "")
            info_table.add_row("   • GPU:", "2x RTX 5090 (216.2 TFLOPS)")
            info_table.add_row("   • VRAM:", "32 GB")
            info_table.add_row("   • RAM:", "129 GB")
            info_table.add_row("   • Ускорение:", "15-30x по сравнению с CPU")
            
            self.console.print(info_table)
            
            # Проверяем состояние сервера
            with self.console.status("[cyan]Проверка подключения к серверу...[/cyan]"):
                server_status = self._check_server_status()
            
            if server_status['connected']:
                self.console.print("[green]✅ Сервер доступен[/green]")
                if server_status['project_exists']:
                    self.console.print("[green]✅ Проект синхронизирован[/green]")
                else:
                    self.console.print("[yellow]⚠️  Проект не найден на сервере[/yellow]")
            else:
                self.console.print("[red]❌ Сервер недоступен[/red]")
            
            # Меню действий
            self.console.print("\n[bold cyan]Выберите действие:[/bold cyan]")
            action_table = Table(show_header=False, box=None)
            action_table.add_column("Option", style="cyan", width=5)
            action_table.add_column("Description", style="white")
            
            action_table.add_row("1", "📤 Синхронизировать проект")
            action_table.add_row("2", "🚀 Запустить обучение")
            action_table.add_row("3", "📊 Мониторинг с браузером")
            action_table.add_row("4", "📋 Проверить логи")
            action_table.add_row("5", "🔧 Настройки сервера")
            action_table.add_row("6", "🔌 Настроить туннель БД")
            action_table.add_row("0", "Назад")
            
            self.console.print(action_table)
            
            choice = Prompt.ask("\n[bold cyan]Выберите опцию[/bold cyan]")
            
            if choice == "1":
                self.logger.info("GPU меню: выбрана синхронизация")
                self.sync_to_gpu_server()
            elif choice == "2":
                self.logger.info("GPU меню: выбран запуск обучения")
                if not server_status['project_exists']:
                    self.console.print("\n[yellow]⚠️  Сначала нужно синхронизировать проект![/yellow]")
                    Prompt.ask("\nНажмите Enter для продолжения")
                else:
                    self.launch_gpu_training()
            elif choice == "3":
                self.logger.info("GPU меню: выбран мониторинг")
                self.monitor_with_browser()
            elif choice == "4":
                self.logger.info("GPU меню: выбраны логи")
                self.check_gpu_logs()
            elif choice == "5":
                self.logger.info("GPU меню: выбраны настройки")
                self.configure_gpu_server()
            elif choice == "6":
                self.logger.info("GPU меню: выбран туннель БД")
                self.setup_db_tunnel()
            elif choice == "0":
                self.logger.info("GPU меню: выход")
                break
    
    def _check_server_status(self):
        """Проверка состояния сервера"""
        try:
            # Получаем SSH алиас из конфига или переменной окружения
            ssh_alias = os.environ.get('VAST_SSH_ALIAS', 'vast-current')
            
            # Проверяем доступность сервера
            result = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", ssh_alias, 
                 "test -d /root/crypto_ai_trading && echo 'PROJECT_EXISTS' || echo 'NO_PROJECT'"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                project_exists = "PROJECT_EXISTS" in result.stdout
                return {'connected': True, 'project_exists': project_exists}
            else:
                return {'connected': False, 'project_exists': False}
        except:
            return {'connected': False, 'project_exists': False}
    
    def sync_to_gpu_server(self):
        """Синхронизация проекта с GPU сервером"""
        self.console.print("\n[cyan]📤 Синхронизация проекта с GPU сервером...[/cyan]")
        
        script_path = "scripts/sync_to_vast.sh"
        if Path(script_path).exists():
            subprocess.run(["bash", script_path])
            self.console.print("\n[green]✅ Синхронизация завершена[/green]")
        else:
            self.console.print(f"[red]❌ Скрипт {script_path} не найден[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def launch_gpu_training(self):
        """Запуск обучения на GPU сервере"""
        self.console.print("\n[cyan]🚀 Запуск обучения на GPU сервере[/cyan]")
        
        # Проверяем наличие кэша локально
        cache_file = Path("cache/features_cache.pkl")
        if not cache_file.exists():
            self.console.print("[red]❌ Файл кэша не найден локально![/red]")
            self.console.print("[yellow]Сначала создайте кэш через 'Управление данными' -> 'Создать/обновить признаки'[/yellow]")
            Prompt.ask("\nНажмите Enter для продолжения")
            return
        
        # Получаем SSH алиас
        ssh_alias = os.environ.get('VAST_SSH_ALIAS', 'vast-current')
        
        # Проверяем наличие кэша на сервере
        with self.console.status("[cyan]Проверка данных на сервере...[/cyan]"):
            result = subprocess.run(
                ["ssh", ssh_alias, 
                 "test -f /root/crypto_ai_trading/cache/features_cache.pkl && echo 'EXISTS' || echo 'NOT_EXISTS'"],
                capture_output=True,
                text=True
            )
            
        if "NOT_EXISTS" in result.stdout:
            self.console.print("[yellow]📤 Кэш не найден на сервере. Копирование...[/yellow]")
            self.console.print(f"[dim]Размер файла: {cache_file.stat().st_size / (1024*1024):.1f} MB[/dim]")
            
            # Создаем директорию и копируем файл
            subprocess.run(["ssh", ssh_alias, "mkdir -p /root/crypto_ai_trading/cache"])
            
            with Progress() as progress:
                task = progress.add_task("[cyan]Копирование кэша...", total=100)
                result = subprocess.run(
                    ["scp", str(cache_file), f"{ssh_alias}:/root/crypto_ai_trading/cache/"],
                    capture_output=True,
                    text=True
                )
                progress.update(task, completed=100)
            
            if result.returncode == 0:
                self.console.print("[green]✅ Кэш скопирован на сервер[/green]")
            else:
                self.console.print("[red]❌ Ошибка копирования кэша[/red]")
                Prompt.ask("\nНажмите Enter для продолжения")
                return
        else:
            self.console.print("[green]✅ Кэш найден на сервере[/green]")
        
        # Выбор режима обучения
        self.console.print("\n[bold cyan]Выберите режим обучения:[/bold cyan]")
        self.console.print("1. Демо (5 эпох) - ~15-20 минут")
        self.console.print("2. Стандартное (50 эпох) - ~2-3 часа")
        self.console.print("3. Полное (100 эпох) - ~5-6 часов")
        self.console.print("4. Пользовательское")
        
        choice = Prompt.ask("Выбор", default="1")
        
        if choice == "1":
            epochs = 5
            mode_choice = "1"
        elif choice == "2":
            epochs = 50
            mode_choice = "3"
        elif choice == "3":
            epochs = 100
            mode_choice = "2"
        elif choice == "4":
            epochs = IntPrompt.ask("Количество эпох", default=10)
            mode_choice = "3"
        else:
            return
        
        self.console.print(f"\n[yellow]Запуск обучения на {epochs} эпох...[/yellow]")
        
        # Передаем выбор режима в скрипт через переменную окружения
        env = os.environ.copy()
        env['GPU_TRAINING_MODE'] = mode_choice
        env['GPU_TRAINING_EPOCHS'] = str(epochs)
        env['USE_CACHE_ONLY'] = '1'  # Флаг для использования кэша
        
        # Запуск скрипта на сервере
        script_path = "scripts/run_on_vast.sh"
        if Path(script_path).exists():
            try:
                # Запускаем скрипт и ждем его завершения
                result = subprocess.run(
                    ["bash", script_path], 
                    env=env,
                    capture_output=True,
                    text=True
                )
                
                # Показываем вывод скрипта
                if result.stdout:
                    print(result.stdout)
                
                if result.returncode == 0:
                    self.console.print("\n[green]✅ Скрипт завершен успешно![/green]")
                    
                    # Предлагаем запустить мониторинг
                    if Confirm.ask("\nЗапустить мониторинг в браузере?"):
                        self.monitor_with_browser()
                else:
                    self.console.print(f"\n[red]❌ Ошибка запуска: код {result.returncode}[/red]")
                    if result.stderr:
                        self.console.print(f"[red]{result.stderr}[/red]")
            except Exception as e:
                self.console.print(f"[red]❌ Ошибка: {e}[/red]")
        else:
            self.console.print(f"[red]❌ Скрипт {script_path} не найден[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def monitor_with_browser(self):
        """Мониторинг с автоматическим запуском браузера"""
        self.logger.info("Запуск monitor_with_browser")
        try:
            self.console.print("\n[cyan]📊 Запуск мониторинга обучения...[/cyan]")
            
            # Проверяем, запущен ли TensorBoard на сервере
            self.console.print("[yellow]Проверка TensorBoard на сервере...[/yellow]")
            
            # Получаем SSH алиас
            ssh_alias = os.environ.get('VAST_SSH_ALIAS', 'vast-current')
            
            result = subprocess.run(
                ["ssh", ssh_alias, "pgrep -f tensorboard"],
                capture_output=True
            )
            
            if result.returncode != 0:
                self.console.print("[yellow]TensorBoard не запущен. Запускаем...[/yellow]")
                
                # Сначала проверяем, где есть логи
                check_dirs = subprocess.run(
                    ["ssh", ssh_alias, 
                     "cd /root/crypto_ai_trading && find . -name 'events.out.tfevents*' | head -5"],
                    capture_output=True,
                    text=True
                )
                
                if check_dirs.stdout.strip():
                    self.console.print(f"[dim]Найдены логи: {check_dirs.stdout.strip().split()[0]}[/dim]")
                
                # Запускаем TensorBoard на порту 6007 (6006 занят Caddy)
                subprocess.run(
                    ["ssh", ssh_alias, 
                     "cd /root/crypto_ai_trading && pkill -f tensorboard; nohup tensorboard --logdir ./experiments/runs --bind_all --port 6007 > logs/tensorboard.log 2>&1 &"],
                    capture_output=True
                )
                time.sleep(3)
                self.console.print("[green]✅ TensorBoard запущен на порту 6007[/green]")
            
            # Запускаем SSH с пробросом портов
            self.console.print("[yellow]Открываем туннель к серверу...[/yellow]")
            
            import threading
            import webbrowser
            import time as time_module
            
            def open_browser():
                time_module.sleep(3)  # Даем время на установку туннеля
                self.console.print("\n[green]Открываем TensorBoard в браузере...[/green]")
                self.console.print("[yellow]TensorBoard доступен на: http://localhost:6007[/yellow]")
                webbrowser.open('http://localhost:6007')
            
            # Запускаем браузер в отдельном потоке
            browser_thread = threading.Thread(target=open_browser)
            browser_thread.start()
            
            # Запускаем SSH туннель с автоматическим выбором
            env = os.environ.copy()
            env['VAST_CONNECTION_MODE'] = '1'  # Прямое подключение
            
            script_path = "scripts/connect_vast.sh"
            if Path(script_path).exists():
                try:
                    subprocess.run(["bash", script_path], env=env)
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]Мониторинг остановлен[/yellow]")
                except Exception as e:
                    self.console.print(f"[red]❌ Ошибка: {e}[/red]")
            else:
                self.console.print(f"[red]❌ Скрипт {script_path} не найден[/red]")
                
        except Exception as e:
            self.logger.error(f"Ошибка в monitor_with_browser: {e}", exc_info=True)
            self.console.print(f"[red]❌ Ошибка: {e}[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def check_gpu_logs(self):
        """Проверить логи на GPU сервере"""
        self.console.print("\n[cyan]📋 Получение логов с сервера...[/cyan]")
        
        # Получаем SSH алиас
        ssh_alias = os.environ.get('VAST_SSH_ALIAS', 'vast-current')
        
        result = subprocess.run(
            ["ssh", ssh_alias, 
             "tail -n 50 /root/crypto_ai_trading/logs/training_gpu.log 2>/dev/null || echo 'Логи не найдены'"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            self.console.print("\n[yellow]Последние строки лога:[/yellow]")
            self.console.print(result.stdout)
        else:
            self.console.print("[red]❌ Не удалось получить логи[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def configure_gpu_server(self):
        """Настройка GPU сервера"""
        self.console.print("\n[cyan]⚙️ Настройка GPU сервера[/cyan]")
        
        remote_config = self.config.setdefault('remote_server', {})
        
        self.console.print("\nТекущие настройки:")
        self.console.print(f"Host: {remote_config.get('primary', {}).get('host', 'Не задан')}")
        self.console.print(f"Port: {remote_config.get('primary', {}).get('port', 'Не задан')}")
        
        if Confirm.ask("\nИзменить настройки?"):
            host = Prompt.ask("IP адрес сервера", default="114.32.64.6")
            port = IntPrompt.ask("SSH порт", default=40134)
            
            remote_config['enabled'] = True
            remote_config.setdefault('primary', {})['host'] = host
            remote_config.setdefault('primary', {})['port'] = port
            remote_config['user'] = 'root'
            remote_config['key_path'] = '~/.ssh/vast_ai_key'
            
            self.save_config()
            self.console.print("[green]✅ Настройки сохранены[/green]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def setup_db_tunnel(self):
        """Настройка туннеля к локальной БД"""
        self.logger.info("Запуск setup_db_tunnel")
        try:
            self.console.print("\n[cyan]🔌 Настройка туннеля к локальной БД[/cyan]")
            
            # Проверяем, запущена ли локальная БД
            self.console.print("[yellow]Проверка локальной PostgreSQL...[/yellow]")
            
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', 5555))
            sock.close()
            
            if result != 0:
                self.console.print("[red]❌ PostgreSQL не запущен на порту 5555[/red]")
                self.console.print("[yellow]Запустите БД командой:[/yellow]")
                self.console.print("   docker-compose up -d postgres")
                Prompt.ask("\nНажмите Enter для продолжения")
                return
            
            self.console.print("[green]✅ Локальная БД доступна[/green]")
            
            # Проверяем существующие туннели
            result = subprocess.run(
                ["pgrep", "-f", "ssh.*-R.*5555"],
                capture_output=True
            )
            
            if result.returncode == 0:
                self.console.print("[yellow]⚠️  Туннель уже существует[/yellow]")
                if Confirm.ask("Пересоздать туннель?"):
                    subprocess.run(["pkill", "-f", "ssh.*-R.*5555"])
                    time.sleep(1)
                else:
                    Prompt.ask("\nНажмите Enter для продолжения")
                    return
            
            # Запускаем скрипт настройки туннеля
            script_path = "scripts/setup_db_tunnel.sh"
            if Path(script_path).exists():
                self.console.print("[yellow]🚇 Создание SSH туннеля...[/yellow]")
                result = subprocess.run(
                    ["bash", script_path],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    self.console.print("[green]✅ Туннель успешно создан![/green]")
                    self.console.print("\n[cyan]Теперь GPU сервер может использовать БД[/cyan]")
                    self.console.print("При запуске обучения БД будет доступна автоматически")
                else:
                    self.console.print("[red]❌ Ошибка создания туннеля[/red]")
                    if result.stderr:
                        self.console.print(f"[red]{result.stderr}[/red]")
            else:
                self.console.print(f"[red]❌ Скрипт {script_path} не найден[/red]")
                
        except Exception as e:
            self.logger.error(f"Ошибка в setup_db_tunnel: {e}", exc_info=True)
            self.console.print(f"[red]❌ Ошибка: {e}[/red]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def clear_logs(self):
        """Очистка логов"""
        self.console.print("\n[cyan]🗑️ Очистка логов[/cyan]")
        
        log_dirs = [
            Path("logs"),
            Path("experiments/logs"),
            Path("experiments/runs")
        ]
        
        total_size = 0
        file_count = 0
        
        for log_dir in log_dirs:
            if log_dir.exists():
                for file in log_dir.rglob("*"):
                    if file.is_file():
                        total_size += file.stat().st_size
                        file_count += 1
        
        if file_count > 0:
            size_mb = total_size / (1024 * 1024)
            self.console.print(f"\nНайдено {file_count} файлов ({size_mb:.1f} MB)")
            
            if Confirm.ask("Удалить все логи?"):
                for log_dir in log_dirs:
                    if log_dir.exists():
                        import shutil
                        shutil.rmtree(log_dir)
                        log_dir.mkdir(exist_ok=True)
                
                self.console.print("[green]✅ Логи очищены[/green]")
        else:
            self.console.print("[yellow]Логи не найдены[/yellow]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def show_system_info(self):
        """Показать информацию о системе"""
        self.console.print(Panel("📋 Информация о системе", style="cyan"))
        
        # Версия Python
        import sys
        self.console.print(f"\n[cyan]Python:[/cyan] {sys.version.split()[0]}")
        
        # Установленные пакеты
        try:
            import torch
            self.console.print(f"[cyan]PyTorch:[/cyan] {torch.__version__}")
        except:
            pass
        
        try:
            import pandas as pd
            self.console.print(f"[cyan]Pandas:[/cyan] {pd.__version__}")
        except:
            pass
        
        # Статус GPU
        try:
            import torch
            if torch.cuda.is_available():
                self.console.print(f"\n[green]✅ GPU доступен:[/green] {torch.cuda.get_device_name(0)}")
            else:
                self.console.print("\n[yellow]⚠️ GPU не доступен, используется CPU[/yellow]")
        except:
            pass
        
        # Структура проекта
        self.console.print("\n[cyan]Структура проекта:[/cyan]")
        dirs = ["data", "models", "training", "trading", "utils", "config", "cache", "experiments"]
        for dir_name in dirs:
            dir_path = Path(dir_name)
            if dir_path.exists():
                file_count = len(list(dir_path.glob("*.py")))
                self.console.print(f"  📁 {dir_name}/ ({file_count} файлов)")
        
        # Последние модели
        models_dir = Path("models_saved")
        if models_dir.exists():
            models = sorted(models_dir.glob("*.pth"), key=lambda x: x.stat().st_mtime, reverse=True)
            if models:
                self.console.print(f"\n[cyan]Последние модели:[/cyan]")
                for model in models[:3]:
                    size = model.stat().st_size / (1024 * 1024)  # MB
                    mtime = datetime.fromtimestamp(model.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    self.console.print(f"  • {model.name} ({size:.1f} MB, {mtime})")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def quick_start(self):
        """Быстрый запуск"""
        self.console.print("\n[cyan]🚀 Быстрый запуск системы[/cyan]")
        
        steps = [
            ("Проверка БД", lambda: subprocess.run(["python", "-c", "from data.data_loader import CryptoDataLoader; print('OK')"], capture_output=True).returncode == 0),
            ("Проверка данных", lambda: Path("cache/features_cache.pkl").exists()),
            ("Проверка моделей", lambda: Path("models").exists()),
        ]
        
        all_ok = True
        for step_name, check_func in steps:
            try:
                if check_func():
                    self.console.print(f"✅ {step_name}")
                else:
                    self.console.print(f"❌ {step_name}")
                    all_ok = False
            except:
                self.console.print(f"❌ {step_name}")
                all_ok = False
        
        if all_ok:
            self.console.print("\n[green]Система готова к работе![/green]")
            
            if Confirm.ask("\nЗапустить полный пайплайн?"):
                self.console.print("\n[cyan]Выберите режим:[/cyan]")
                self.console.print("1. Только подготовка данных")
                self.console.print("2. Подготовка данных + обучение")
                self.console.print("3. Полный пайплайн (данные + обучение + бэктест)")
                
                mode_choice = Prompt.ask("Выбор", default="2")
                
                if mode_choice == "1":
                    subprocess.run(["python", "run_full_pipeline.py", "--mode", "data"])
                elif mode_choice == "2":
                    subprocess.run(["python", "run_full_pipeline.py", "--mode", "demo"])
                elif mode_choice == "3":
                    subprocess.run(["python", "run_full_pipeline.py", "--mode", "full"])
        else:
            self.console.print("\n[yellow]⚠️ Требуется настройка системы[/yellow]")
        
        Prompt.ask("\nНажмите Enter для продолжения")
    
    def run(self):
        """Главный цикл меню"""
        while True:
            self.display_main_menu()
            choice = Prompt.ask("\n[bold cyan]Выберите опцию[/bold cyan]", default="0")
            
            if choice == "1":
                self.data_management_menu()
            elif choice == "2":
                self.training_menu()
            elif choice == "3":
                self.monitoring_menu()
            elif choice == "4":
                self.settings_menu()
            elif choice == "5":
                self.utilities_menu()
            elif choice == "6":
                self.show_system_info()
            elif choice == "7":
                self.quick_start()
            elif choice == "0":
                if Confirm.ask("\n[yellow]Вы уверены, что хотите выйти?[/yellow]"):
                    self.console.print("\n[cyan]До свидания! 👋[/cyan]")
                    break
            else:
                self.console.print("[red]Неверный выбор. Попробуйте снова.[/red]")
                time.sleep(1)


def main():
    """Точка входа"""
    menu_logger.info("="*60)
    menu_logger.info("Запуск Crypto AI Trading System")
    menu_logger.info(f"Файл логов: {log_file}")
    menu_logger.info("="*60)
    
    try:
        menu = CryptoTradingMenu()
        menu.run()
    except KeyboardInterrupt:
        menu_logger.info("Программа прервана пользователем")
        console.print("\n\n[yellow]Программа прервана пользователем[/yellow]")
    except Exception as e:
        menu_logger.error(f"Критическая ошибка: {e}", exc_info=True)
        console.print(f"\n[red]Критическая ошибка: {e}[/red]")
        import traceback
        traceback.print_exc()
    finally:
        menu_logger.info("Завершение работы")


if __name__ == "__main__":
    main()