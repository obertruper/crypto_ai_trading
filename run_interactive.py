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
    try:
        menu = CryptoTradingMenu()
        menu.run()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Программа прервана пользователем[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Критическая ошибка: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()