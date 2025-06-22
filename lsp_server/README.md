# Enhanced Python LSP Server для Claude Code

Продвинутый LSP сервер для Python, оптимизированный для работы с Claude Code и другими LLM-моделями.

## 🚀 Основные возможности

- **Интеллектуальная индексация проекта** - полный анализ AST всех Python файлов
- **Расширенное управление контекстом** - отслеживание связанных файлов и символов
- **Оптимизация для LLM** - генерация контекста специально для AI-моделей
- **Кеширование и производительность** - быстрая работа даже с большими проектами
- **Интеграция с документацией** - автоматический парсинг docstrings и внешней документации
- **Умное автодополнение** - с учетом контекста и типов
- **Граф зависимостей** - отслеживание импортов и связей между модулями

## 📋 Требования

- Python 3.8+
- pip или poetry для управления зависимостями
- 100+ MB свободного места для кеша

## 🛠 Установка

### 1. Установка зависимостей

```bash
cd /Users/ruslan/crypto_ai_trading/lsp_server

# Создайте виртуальное окружение
python3 -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate

# Установите зависимости
pip install -r requirements.txt
```

### 2. Настройка конфигурации

Отредактируйте `config.yaml` под ваши нужды:

```yaml
server:
  port: 2087  # Порт для LSP сервера
  
context:
  max_tokens: 8000  # Размер контекста для LLM
  
features:
  diagnostics:
    linters: ["pylint", "flake8"]  # Выберите нужные линтеры
```

### 3. Запуск сервера

```bash
python enhanced_lsp_server.py
```

Или для работы через stdio (рекомендуется):

```bash
python enhanced_lsp_server.py --stdio
```

## 🔧 Интеграция с редакторами

### Claude Code / VS Code

1. Установите расширение для работы с кастомными LSP серверами (если требуется)

2. Добавьте в настройки проекта `.vscode/settings.json`:

```json
{
  "python.languageServer": "None",
  "python.linting.enabled": false,
  
  "[python]": {
    "editor.defaultFormatter": null
  },
  
  "lsp": {
    "enhanced-python-lsp": {
      "command": [
        "/Users/ruslan/crypto_ai_trading/lsp_server/venv/bin/python",
        "/Users/ruslan/crypto_ai_trading/lsp_server/enhanced_lsp_server.py",
        "--stdio"
      ],
      "filetypes": ["python"],
      "rootPatterns": ["pyproject.toml", "setup.py", ".git"],
      "settings": {
        "enhanced_lsp": {
          "context": {
            "max_tokens": 8000
          }
        }
      }
    }
  }
}
```

### Neovim

Добавьте в вашу конфигурацию:

```lua
local lspconfig = require('lspconfig')
local configs = require('lspconfig.configs')

-- Определяем новый LSP сервер
configs.enhanced_python_lsp = {
  default_config = {
    cmd = {
      '/Users/ruslan/crypto_ai_trading/lsp_server/venv/bin/python',
      '/Users/ruslan/crypto_ai_trading/lsp_server/enhanced_lsp_server.py',
      '--stdio'
    },
    filetypes = {'python'},
    root_dir = lspconfig.util.root_pattern('pyproject.toml', 'setup.py', '.git'),
    settings = {
      enhanced_lsp = {
        context = {
          max_tokens = 8000
        }
      }
    }
  }
}

-- Активируем сервер
lspconfig.enhanced_python_lsp.setup{}
```

## 🎯 Использование

### Основные команды

- **Автодополнение** - начните печатать и нажмите `Ctrl+Space`
- **Информация о символе** - наведите курсор на символ
- **Перейти к определению** - `F12` или `Ctrl+Click`
- **Найти все ссылки** - `Shift+F12`
- **Символы в файле** - `Ctrl+Shift+O`
- **Символы в проекте** - `Ctrl+T`

### Специальные команды для LLM

LSP сервер предоставляет специальные команды для работы с контекстом:

```python
# Получить текущий контекст LLM
# Выполните команду: enhanced_lsp.get_context

# Переиндексировать проект
# Выполните команду: enhanced_lsp.reindex_workspace
```

## 🧠 Оптимизация для Claude Code

### 1. Контекстное окно

Сервер автоматически отслеживает:
- Текущий файл и позицию курсора
- Импортированные модули
- Связанные файлы (до 5)
- Последние изменения (до 10)
- Символы в текущей области видимости

### 2. Умная индексация

- Парсинг AST для точного анализа кода
- Извлечение сигнатур функций и типов
- Построение графа зависимостей
- Кеширование результатов анализа

### 3. Интеграция с документацией

- Автоматический парсинг docstrings (NumPy, Google, Sphinx стили)
- Поддержка внешней документации
- Генерация примеров использования

## 📊 Мониторинг и отладка

### Просмотр логов

```bash
tail -f enhanced-lsp.log
```

### Проверка статистики

В редакторе выполните команду `enhanced_lsp.get_context` для получения информации о:
- Количестве проиндексированных файлов
- Общем количестве символов
- Попаданиях в кеш
- Текущем контексте

### Уровни логирования

В `config.yaml`:

```yaml
server:
  log_level: "DEBUG"  # DEBUG, INFO, WARNING, ERROR
```

## 🐛 Решение проблем

### Сервер не запускается

1. Проверьте, что все зависимости установлены:
   ```bash
   pip install -r requirements.txt
   ```

2. Убедитесь, что используется правильная версия Python:
   ```bash
   python --version  # Должно быть 3.8+
   ```

3. Проверьте логи на наличие ошибок:
   ```bash
   cat enhanced-lsp.log
   ```

### Медленная индексация

1. Уменьшите `max_workers` в конфигурации
2. Исключите большие директории в `exclude_dirs`
3. Увеличьте `debounce_interval`

### Проблемы с памятью

1. Уменьшите `max_tokens` для контекста
2. Очистите кеш: `rm -rf .lsp_cache`
3. Ограничьте количество индексируемых файлов

## 🤝 Вклад в проект

Мы приветствуем улучшения! Особенно интересны:

- Оптимизация производительности
- Новые анализаторы кода
- Улучшение генерации контекста для LLM
- Поддержка новых редакторов
- Расширение документации

## 📄 Лицензия

MIT License - свободно используйте и модифицируйте под свои нужды.

## 🔗 Полезные ссылки

- [LSP Specification](https://microsoft.github.io/language-server-protocol/)
- [pygls Documentation](https://pygls.readthedocs.io/)
- [Python AST](https://docs.python.org/3/library/ast.html)
- [Claude API Documentation](https://docs.anthropic.com/)

---

**Примечание**: Этот LSP сервер специально оптимизирован для работы с Claude Code и другими LLM-инструментами. Он предоставляет расширенный контекст и метаданные, которые помогают AI лучше понимать структуру вашего проекта.