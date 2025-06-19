"""
Логгер для интерактивного меню
"""

import logging
from pathlib import Path
from datetime import datetime
import sys

def setup_interactive_logger():
    """Настройка логгера для интерактивного меню"""
    
    # Создаем директорию для логов
    log_dir = Path("logs/interactive")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Имя файла лога с датой
    log_file = log_dir / f"interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Настройка логгера
    logger = logging.getLogger('interactive_menu')
    logger.setLevel(logging.DEBUG)
    
    # Форматирование
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Файловый обработчик
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Консольный обработчик для ошибок
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Логируем начало сессии
    logger.info("=" * 60)
    logger.info("Запуск интерактивного меню Crypto AI Trading System")
    logger.info(f"Лог файл: {log_file}")
    logger.info("=" * 60)
    
    return logger

def log_menu_action(logger, action: str, details: dict = None):
    """Логирование действия в меню"""
    msg = f"Действие: {action}"
    if details:
        msg += f" | Детали: {details}"
    logger.info(msg)

def log_menu_error(logger, error: Exception, context: str = ""):
    """Логирование ошибки в меню"""
    msg = f"Ошибка в {context}: {type(error).__name__}: {str(error)}"
    logger.error(msg, exc_info=True)