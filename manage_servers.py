#!/usr/bin/env python3
"""
Утилита управления профилями GPU серверов
Позволяет легко переключаться между разными серверами для обучения
"""

import yaml
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.panel import Panel

console = Console()

class ServerProfileManager:
    def __init__(self):
        self.profiles_file = Path("config/server_profiles.yaml")
        self.main_config_file = Path("config/config.yaml")
        self.profiles_data = self.load_profiles()
        
    def load_profiles(self):
        """Загрузка профилей серверов"""
        if not self.profiles_file.exists():
            console.print(f"[red]❌ Файл профилей не найден: {self.profiles_file}[/red]")
            return None
        
        with open(self.profiles_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def save_profiles(self):
        """Сохранение профилей"""
        with open(self.profiles_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.profiles_data, f, default_flow_style=False, allow_unicode=True)
    
    def update_main_config(self, profile_name):
        """Обновление основного конфига с данными выбранного профиля"""
        if profile_name not in self.profiles_data['profiles']:
            console.print(f"[red]❌ Профиль '{profile_name}' не найден[/red]")
            return False
        
        profile = self.profiles_data['profiles'][profile_name]
        
        # Загружаем основной конфиг
        with open(self.main_config_file, 'r', encoding='utf-8') as f:
            main_config = yaml.safe_load(f)
        
        # Обновляем настройки remote_server
        if 'remote_server' not in main_config:
            main_config['remote_server'] = {}
        
        main_config['remote_server']['direct_connection'] = {
            'host': profile['connection']['host'],
            'port': profile['connection']['port'],
            'user': profile['connection']['user'],
            'key_path': profile['connection']['key_path']
        }
        
        main_config['remote_server']['enabled'] = True
        main_config['remote_server']['remote_path'] = profile['paths']['remote_project']
        main_config['remote_server']['python_path'] = profile['paths']['python']
        
        if 'ports' in profile:
            main_config['remote_server']['ports'] = profile['ports']
        
        # Сохраняем обновленный конфиг
        with open(self.main_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(main_config, f, default_flow_style=False, allow_unicode=True)
        
        # Обновляем активный профиль
        self.profiles_data['active_profile'] = profile_name
        self.save_profiles()
        
        return True
    
    def list_profiles(self):
        """Показать все доступные профили"""
        console.print(Panel("🖥️ Доступные профили серверов", style="cyan"))
        
        active = self.profiles_data.get('active_profile', 'none')
        
        table = Table()
        table.add_column("Статус", style="green", width=6)
        table.add_column("Имя", style="cyan")
        table.add_column("Описание", style="white")
        table.add_column("GPU", style="yellow")
        table.add_column("Стоимость", style="magenta")
        table.add_column("Хост", style="dim")
        
        for profile_id, profile in self.profiles_data['profiles'].items():
            status = "🟢 ✓" if profile_id == active else ""
            
            table.add_row(
                status,
                profile['name'],
                profile['description'],
                profile['specs']['gpu'],
                profile['specs']['cost_per_hour'],
                f"{profile['connection']['host']}:{profile['connection']['port']}"
            )
        
        console.print(table)
        console.print(f"\\n[green]Активный профиль: {active}[/green]")
    
    def switch_profile(self, profile_name=None):
        """Переключиться на другой профиль"""
        if not profile_name:
            # Показываем список для выбора
            self.list_profiles()
            console.print("\\n[cyan]Доступные профили:[/cyan]")
            
            for i, (profile_id, profile) in enumerate(self.profiles_data['profiles'].items(), 1):
                console.print(f"{i}. {profile_id} - {profile['name']}")
            
            choice = Prompt.ask("\\nВыберите номер профиля")
            
            try:
                profile_names = list(self.profiles_data['profiles'].keys())
                profile_name = profile_names[int(choice) - 1]
            except (ValueError, IndexError):
                console.print("[red]❌ Неверный выбор[/red]")
                return False
        
        if profile_name not in self.profiles_data['profiles']:
            console.print(f"[red]❌ Профиль '{profile_name}' не найден[/red]")
            return False
        
        profile = self.profiles_data['profiles'][profile_name]
        
        # Показываем информацию о профиле
        console.print(f"\\n[cyan]Переключение на профиль:[/cyan] {profile['name']}")
        console.print(f"[dim]Описание:[/dim] {profile['description']}")
        console.print(f"[dim]Хост:[/dim] {profile['connection']['host']}:{profile['connection']['port']}")
        console.print(f"[dim]GPU:[/dim] {profile['specs']['gpu']}")
        console.print(f"[dim]Стоимость:[/dim] {profile['specs']['cost_per_hour']}")
        
        if Confirm.ask("\\nПодтвердить переключение?"):
            if self.update_main_config(profile_name):
                console.print(f"\\n[green]✅ Профиль переключен на: {profile['name']}[/green]")
                console.print("[yellow]Настройки обновлены в config/config.yaml[/yellow]")
                return True
            else:
                console.print("[red]❌ Ошибка при обновлении конфигурации[/red]")
                return False
        
        return False
    
    def add_profile(self):
        """Добавить новый профиль сервера"""
        console.print(Panel("➕ Добавление нового профиля сервера", style="green"))
        
        # Базовая информация
        profile_id = Prompt.ask("ID профиля (например: vast_ai_new)")
        name = Prompt.ask("Название сервера")
        description = Prompt.ask("Описание сервера")
        
        # Подключение
        console.print("\\n[cyan]Настройки подключения:[/cyan]")
        host = Prompt.ask("IP адрес или хост")
        port = int(Prompt.ask("SSH порт", default="22"))
        user = Prompt.ask("Пользователь", default="root")
        key_path = Prompt.ask("Путь к SSH ключу", default="~/.ssh/id_rsa")
        
        # Спецификации
        console.print("\\n[cyan]Характеристики сервера:[/cyan]")
        gpu = Prompt.ask("GPU (например: 2x RTX 4090)")
        vram = Prompt.ask("VRAM (например: 48 GB)")
        ram = Prompt.ask("RAM (например: 90 GB)")
        cost = Prompt.ask("Стоимость в час (например: $1.20)")
        
        # Пути
        console.print("\\n[cyan]Пути на сервере:[/cyan]")
        remote_project = Prompt.ask("Путь к проекту", default="/root/crypto_ai_trading")
        python_path = Prompt.ask("Путь к Python", default="/opt/conda/bin/python")
        
        # Создаем профиль
        new_profile = {
            'name': name,
            'description': description,
            'connection': {
                'host': host,
                'port': port,
                'user': user,
                'key_path': key_path
            },
            'specs': {
                'gpu': gpu,
                'vram': vram,
                'ram': ram,
                'cost_per_hour': cost
            },
            'ports': {
                'ssh': port,
                'tensorboard': 6006,
                'jupyter': 8888
            },
            'paths': {
                'remote_project': remote_project,
                'python': python_path
            }
        }
        
        # Добавляем профиль
        self.profiles_data['profiles'][profile_id] = new_profile
        self.save_profiles()
        
        console.print(f"\\n[green]✅ Профиль '{profile_id}' добавлен![/green]")
        
        if Confirm.ask("Переключиться на новый профиль?"):
            self.switch_profile(profile_id)
    
    def show_current_profile(self):
        """Показать информацию о текущем активном профиле"""
        active = self.profiles_data.get('active_profile')
        
        if not active or active not in self.profiles_data['profiles']:
            console.print("[yellow]⚠️ Активный профиль не установлен[/yellow]")
            return
        
        profile = self.profiles_data['profiles'][active]
        
        console.print(Panel(f"🟢 Текущий активный профиль: {profile['name']}", style="green"))
        
        info_table = Table(show_header=False, box=None)
        info_table.add_column("Параметр", style="cyan")
        info_table.add_column("Значение", style="white")
        
        info_table.add_row("Название:", profile['name'])
        info_table.add_row("Описание:", profile['description'])
        info_table.add_row("Хост:", f"{profile['connection']['host']}:{profile['connection']['port']}")
        info_table.add_row("Пользователь:", profile['connection']['user'])
        info_table.add_row("SSH ключ:", profile['connection']['key_path'])
        info_table.add_row("GPU:", profile['specs']['gpu'])
        info_table.add_row("VRAM:", profile['specs']['vram'])
        info_table.add_row("RAM:", profile['specs']['ram'])
        info_table.add_row("Стоимость:", profile['specs']['cost_per_hour'])
        info_table.add_row("Удаленный путь:", profile['paths']['remote_project'])
        
        console.print(info_table)
    
    def test_connection(self, profile_name=None):
        """Тестировать подключение к серверу"""
        if not profile_name:
            profile_name = self.profiles_data.get('active_profile')
        
        if not profile_name or profile_name not in self.profiles_data['profiles']:
            console.print("[red]❌ Профиль не найден[/red]")
            return False
        
        profile = self.profiles_data['profiles'][profile_name]
        
        console.print(f"\\n[cyan]🔍 Тестирование подключения к: {profile['name']}[/cyan]")
        
        # Команда для тестирования SSH
        import subprocess
        ssh_cmd = [
            "ssh",
            "-i", os.path.expanduser(profile['connection']['key_path']),
            "-p", str(profile['connection']['port']),
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            f"{profile['connection']['user']}@{profile['connection']['host']}",
            "echo 'Connection test successful' && uname -a"
        ]
        
        try:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                console.print("[green]✅ Подключение успешно![/green]")
                console.print(f"[dim]Ответ сервера: {result.stdout.strip()}[/dim]")
                return True
            else:
                console.print("[red]❌ Ошибка подключения[/red]")
                console.print(f"[red]{result.stderr}[/red]")
                return False
                
        except subprocess.TimeoutExpired:
            console.print("[red]❌ Таймаут подключения[/red]")
            return False
        except Exception as e:
            console.print(f"[red]❌ Ошибка: {e}[/red]")
            return False


def main():
    """Главное меню управления профилями"""
    manager = ServerProfileManager()
    
    if not manager.profiles_data:
        console.print("[red]❌ Не удалось загрузить профили серверов[/red]")
        return
    
    while True:
        console.clear()
        console.print(Panel.fit(
            "[bold cyan]🖥️ Управление профилями GPU серверов[/bold cyan]\\n"
            "[dim]Быстрое переключение между серверами для обучения[/dim]",
            border_style="blue"
        ))
        
        # Показываем текущий профиль
        manager.show_current_profile()
        
        # Меню
        console.print("\\n[bold cyan]Доступные действия:[/bold cyan]")
        menu_table = Table(show_header=False, box=None)
        menu_table.add_column("Опция", style="cyan", width=3)
        menu_table.add_column("Описание", style="white")
        
        menu_table.add_row("1", "📋 Показать все профили")
        menu_table.add_row("2", "🔄 Переключить профиль")
        menu_table.add_row("3", "➕ Добавить новый профиль")
        menu_table.add_row("4", "🔍 Тестировать подключение")
        menu_table.add_row("5", "📊 Информация о текущем профиле")
        menu_table.add_row("0", "❌ Выход")
        
        console.print(menu_table)
        
        choice = Prompt.ask("\\n[bold cyan]Выберите действие[/bold cyan]")
        
        if choice == "1":
            manager.list_profiles()
            Prompt.ask("\\nНажмите Enter для продолжения")
        elif choice == "2":
            manager.switch_profile()
            Prompt.ask("\\nНажмите Enter для продолжения")
        elif choice == "3":
            manager.add_profile()
            Prompt.ask("\\nНажмите Enter для продолжения")
        elif choice == "4":
            manager.test_connection()
            Prompt.ask("\\nНажмите Enter для продолжения")
        elif choice == "5":
            manager.show_current_profile()
            Prompt.ask("\\nНажмите Enter для продолжения")
        elif choice == "0":
            console.print("\\n[cyan]До свидания! 👋[/cyan]")
            break
        else:
            console.print("[red]Неверный выбор. Попробуйте снова.[/red]")


if __name__ == "__main__":
    main()
