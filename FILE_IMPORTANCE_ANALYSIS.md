# Анализ уровней важности файлов проекта crypto_ai_trading

## 🚨 КРИТИЧЕСКИЕ ФАЙЛЫ (изменения могут сломать систему)

### Модели (models/)
- **patchtst.py** - Основная архитектура PatchTST, ядро системы
- **patchtst_improved.py** - Улучшенная версия модели с защитой от переобучения
- **losses.py** - Функции потерь для обучения
- **ensemble.py** - Ансамблирование моделей

### Данные (data/)
- **data_loader.py** - Загрузка данных из PostgreSQL, критически важен
- **dataset.py** - Классы Dataset для PyTorch, основа для обучения
- **feature_engineering.py** - Создание 100+ технических индикаторов
- **preprocessor.py** - Нормализация и подготовка данных
- **target_scaler.py** - Масштабирование целевых переменных

### Конфигурация (config/)
- **config.yaml** - Главный конфигурационный файл, все параметры системы

### Обучение (training/)
- **trainer.py** - Основной класс для обучения моделей
- **optimizer.py** - Оптимизаторы и планировщики learning rate

## ⚡ ВАЖНЫЕ ФАЙЛЫ (основная бизнес-логика)

### Торговля (trading/)
- **signals.py** - Генерация торговых сигналов на основе предсказаний
- **risk_manager.py** - Управление рисками, стоп-лоссы, тейк-профиты
- **position_sizer.py** - Расчет размера позиций (Kelly, Volatility-based и др.)
- **backtester.py** - Бэктестинг стратегий с учетом комиссий

### Главные файлы
- **main.py** - Основная точка входа, координирует весь процесс
- **train_model.py** - Альтернативная точка входа для обучения
- **train_with_improvements.py** - Версия с дополнительными улучшениями

### Валидация (training/)
- **validator.py** - Валидация моделей, метрики качества

## 📊 СТАНДАРТНЫЕ ФАЙЛЫ (утилиты, визуализация)

### Утилиты (utils/)
- **logger.py** - Логирование с поддержкой структурированного вывода
- **interactive_logger.py** - Интерактивный логгер для визуализации прогресса
- **metrics.py** - Расчет метрик (Sharpe, Sortino, Calmar и др.)
- **visualization.py** - Построение графиков и визуализация результатов
- **config_validator.py** - Валидация конфигурационных файлов
- **nan_diagnostics.py** - Диагностика NaN значений в данных

### Мониторинг
- **monitor_training.py** - Мониторинг процесса обучения в реальном времени
- **check_system.py** - Проверка системных требований и настроек

### Интерактивные инструменты
- **run_interactive.py** - Интерактивное меню для управления системой
- **test_improvements.py** - Тестирование улучшений модели

### Notebooks (notebooks/)
- **01_data_exploration.ipynb** - Исследование данных
- **02_feature_analysis.ipynb** - Анализ признаков
- **03_model_evaluation.ipynb** - Оценка моделей

## 📌 НИЗКИЙ ПРИОРИТЕТ (тесты, примеры, вспомогательные)

### Скрипты (scripts/)
- Shell скрипты для работы с Vast.ai
- Вспомогательные скрипты для настройки окружения
- Скрипты мониторинга и синхронизации

### LSP Server (lsp_server/)
- Интеграция с Language Server Protocol
- Тестовые файлы LSP
- Вспомогательные инструменты для IDE

### Кэш (cache/)
- Временные файлы кэширования признаков
- Сериализованные промежуточные результаты

### Логи (logs/, experiments/logs/)
- Файлы логов выполнения
- Структурированные JSON логи
- Отчеты о тренировках

### Документация
- README файлы
- Markdown документация
- Инструкции по настройке

## 🔒 Правила работы с файлами разной важности

### Для КРИТИЧЕСКИХ файлов:
1. **НИКОГДА** не вносить изменения без полного понимания последствий
2. Всегда создавать резервные копии перед изменением
3. Тестировать на небольшом датасете после любых изменений
4. Документировать все изменения в комментариях

### Для ВАЖНЫХ файлов:
1. Вносить изменения осторожно
2. Проверять совместимость с остальной системой
3. Запускать тесты после изменений
4. Следить за обратной совместимостью

### Для СТАНДАРТНЫХ файлов:
1. Можно модифицировать для улучшения функциональности
2. Следовать существующим паттернам кода
3. Добавлять документацию для новых функций

### Для файлов НИЗКОГО ПРИОРИТЕТА:
1. Можно свободно экспериментировать
2. Использовать для тестирования новых идей
3. Удалять устаревшие или неиспользуемые файлы

## 📋 Рекомендации по разработке

1. **Все изменения архитектуры** → только в `models/patchtst.py` или `models/patchtst_improved.py`
2. **Новые признаки** → добавлять в `data/feature_engineering.py`
3. **Новые стратегии** → реализовывать в `trading/signals.py`
4. **Новые метрики** → добавлять в `utils/metrics.py`
5. **Конфигурация** → все параметры через `config/config.yaml`

## ⚠️ Особые замечания

1. **PostgreSQL** критически важна - порт 5555, пользователь ruslan
2. **Модели сохраняются** в `models_saved/` - не удалять без необходимости
3. **Кэширование признаков** ускоряет работу - папка `cache/`
4. **Логи** содержат важную информацию о тренировках - сохранять для анализа
5. **GPU поддержка** - проверять через `check_system.py` перед обучением