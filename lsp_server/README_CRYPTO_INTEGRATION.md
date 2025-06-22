# LSP Server для проекта Crypto AI Trading

## 📋 Описание

LSP (Language Server Protocol) сервер для улучшенной работы с кодом проекта crypto_ai_trading. Предоставляет интеллектуальные возможности для анализа кода, автодополнения и интеграции с AI-ассистентами.

## 🚀 Быстрый старт

### 1. Активация виртуального окружения
```bash
cd /Users/ruslan/PycharmProjects/LLM\ TRANSFORM/crypto_ai_trading/lsp_server
source venv/bin/activate
```

### 2. Установка зависимостей (если требуется)
```bash
pip install -r requirements.txt
```

### 3. Запуск LSP сервера
```bash
./run_lsp.sh
```

## 🔧 Основные компоненты

### enhanced_lsp_server.py
Основной LSP сервер с поддержкой:
- Автодополнение кода с учетом контекста проекта
- Навигация по определениям (Go to Definition)
- Поиск использований (Find References)
- Документация при наведении (Hover)
- Диагностика кода в реальном времени

### mcp_lsp_bridge.py
Мост для интеграции с MCP (Model Context Protocol):
- Автоматическое отслеживание изменений в проекте
- Предоставление контекста для AI-ассистентов
- Кеширование и оптимизация для больших кодовых баз

### claude_integration.py
Специальная интеграция для Claude:
- Оптимизированный формат контекста
- Интеллектуальный выбор релевантных файлов
- Поддержка thinking протокола

## 📁 Структура проекта crypto_ai_trading

```
crypto_ai_trading/
├── config/          # Конфигурация системы
├── data/            # Загрузка и обработка данных
├── models/          # ML модели (PatchTST)
├── trading/         # Торговые стратегии
├── training/        # Обучение моделей
├── utils/           # Утилиты
├── lsp_server/      # LSP сервер (эта директория)
└── main.py          # Точка входа
```

## 🎯 Интеграция с crypto_ai_trading

### Ключевые файлы для анализа:
1. **models/patchtst.py** - основная архитектура модели
2. **training/trainer.py** - процесс обучения
3. **data/feature_engineering.py** - инженерия признаков
4. **trading/signals.py** - торговые сигналы
5. **config/config.yaml** - конфигурация проекта

### Автодополнение специфичное для проекта:
- Названия технических индикаторов
- Параметры модели PatchTST
- Торговые стратегии
- Методы работы с данными Bybit

## 🛠️ Конфигурация

### config.yaml
```yaml
server:
  port: 2087
  host: "127.0.0.1"

context:
  max_tokens: 1000000
  max_related_files: 5
  
features:
  completion:
    enabled: true
    include_snippets: true
  diagnostics:
    enabled: true
    linters: ["pylint", "flake8", "mypy"]
```

## 📊 Мониторинг

### Логи сервера
```bash
tail -f enhanced-lsp.log
```

### Статистика использования
```bash
python test_lsp_logs.py
```

## 🔍 Тестирование

### Проверка интеграции
```bash
python test_crypto_integration.py
```

### Полный тест функциональности
```bash
python test_lsp.py
```

## 💡 Полезные команды

### Перезапуск сервера
```bash
pkill -f enhanced_lsp_server.py && ./run_lsp.sh
```

### Очистка кеша
```bash
rm -rf .lsp_data/cache/*
```

### Переиндексация проекта
```bash
python -c "from enhanced_lsp_server import EnhancedPythonLSP; lsp = EnhancedPythonLSP(); lsp.index_workspace('/Users/ruslan/PycharmProjects/LLM TRANSFORM/crypto_ai_trading')"
```

## 🐛 Решение проблем

### Сервер не запускается
1. Проверьте, что порт 2087 свободен: `lsof -i :2087`
2. Убедитесь, что виртуальное окружение активировано
3. Проверьте логи: `cat enhanced-lsp.log`

### Медленная работа
1. Увеличьте количество воркеров в config.yaml
2. Исключите большие директории (logs/, data/raw/)
3. Используйте кеширование

### Ошибки импорта
1. Добавьте пути в PYTHONPATH
2. Проверьте виртуальное окружение
3. Переиндексируйте проект

## 📚 Дополнительные ресурсы

- [Документация проекта crypto_ai_trading](../README.md)
- [LSP Specification](https://microsoft.github.io/language-server-protocol/)
- [pygls Documentation](https://pygls.readthedocs.io/)