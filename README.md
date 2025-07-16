# 🚀 Crypto AI Trading System v3.0

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-blue.svg)](https://www.postgresql.org/)

Профессиональная система алгоритмической торговли криптовалютами на основе PatchTST архитектуры с 32 целевыми переменными для точного прогнозирования.

📚 **[Подробная документация о признаках и целевых переменных](docs/FEATURES_AND_TARGETS_LOGIC.md)**

## 🎯 Быстрый старт

### 1. Подготовка данных (первый раз)
```bash
python prepare_trading_data.py
```

### 2. Обучение модели
```bash
python main.py --mode train
```

### 3. Полный цикл (данные + обучение)
```bash
python main.py --mode full
```

## 📋 Основные команды

### 1. Полное обучение модели
```bash
python main.py --mode full
```

### 2. Интерактивное меню
```bash
python main.py --mode interactive
# или
python run_interactive.py
```

### 3. Демо режим (проверка БД)
```bash
python main.py --mode demo
```

### 4. Использование улучшенной модели
```bash
python main.py --mode full --use-improved-model
```

### 5. Валидация конфигурации
```bash
python main.py --validate-only
```

## ✨ Ключевые особенности

### 🎯 36 Целевых переменных
- **Базовые**: future_return_1-4 (доходность через 1-4 свечи)
- **LONG позиции**: tp1/tp2/tp3_reached, sl_reached, optimal_entry_time/price
- **SHORT позиции**: аналогичные метрики для коротких позиций
- **Направление**: best_direction (LONG/SHORT/NEUTRAL)
- **Сила сигнала**: expected_value, signal_strength

### 🧠 Защита от переобучения
- **Early Stopping** с patience=15 эпох
- **Learning Rate**: 0.0001 (оптимально для трансформеров)
- **Min Delta**: 0.00005 (чувствительный порог)
- **Dropout**: 0.2 для регуляризации
- **Gradient Clipping**: 0.5 для стабильности

### 🏗️ Архитектура
- **PatchTST** - SOTA трансформер для временных рядов
- **Context Window**: 168 (42 часа истории)
- **159 признаков**: технические индикаторы + микроструктура
- **Multi-Task Learning**: одновременное предсказание всех целей

### 💼 Риск-менеджмент
- **Частичные закрытия**: 40%, 40%, 20% на уровнях TP
- **Динамические уровни**: адаптация к волатильности
- **6 стратегий позиций**: Kelly, Volatility-based, Risk Parity и др.

## 📊 Структура проекта

```
crypto_ai_trading/
├── config/               # Конфигурация
│   └── config.yaml      # Основной конфиг
├── data/                # Работа с данными
│   ├── data_loader.py   # Загрузка из БД
│   ├── dataset.py       # PyTorch datasets
│   └── feature_engineering.py  # Создание признаков
├── models/              # Модели ML
│   ├── patchtst.py     # PatchTST архитектура (включает улучшения)
│   └── ensemble.py     # Ансамбли
├── trading/            # Торговая логика
│   ├── signals.py      # Генерация сигналов
│   ├── risk_manager.py # Управление рисками
│   └── backtester.py   # Бэктестинг
├── training/           # Обучение
│   ├── trainer.py      # Основной трейнер
│   └── optimizer.py    # Оптимизаторы
├── utils/              # Утилиты
│   ├── logger.py       # Логирование
│   └── metrics.py      # Метрики
├── main.py            # Главный скрипт (единая точка входа)
└── run_interactive.py # Интерактивное меню
```

## 🔧 Конфигурация

Основные параметры в `config/config.yaml`:

```yaml
model:
  batch_size: 256       # Размер батча
  context_window: 168   # Окно контекста (42 часа)
  d_model: 256         # Размерность модели
  dropout: 0.2         # Dropout rate
  learning_rate: 0.0001 # Скорость обучения
  epochs: 100          # Количество эпох
  early_stopping_patience: 15  # Терпение для early stopping
  min_delta: 0.00005   # Минимальное улучшение
  
data:
  symbols: ['BTCUSDT', 'ETHUSDT', ...]  # Торговые пары
  train_ratio: 0.6     # Доля train
  val_ratio: 0.2       # Доля validation
  test_ratio: 0.2      # Доля test
```

## 📈 Результаты

После обучения:
- Модель сохраняется в `models_saved/best_model.pth`
- Логи в `experiments/logs/`
- Метрики в `experiments/logs/*_metrics.csv`

## 🚨 Решение проблем

### PostgreSQL не подключается
```bash
# Проверить статус
pg_ctl -D /usr/local/var/postgres status

# Перезапустить
brew services restart postgresql
```

### Недостаточно памяти
- Уменьшить `batch_size` в конфиге
- Уменьшить количество символов

### Модель переобучается
- Увеличить `dropout`
- Уменьшить `d_model`
- Добавить больше данных

## 📝 Лицензия

MIT License - см. файл [LICENSE](LICENSE)

## 📊 Мониторинг обучения

```bash
# TensorBoard для визуализации
tensorboard --logdir logs/

# Или встроенный мониторинг
python monitor_training.py
```

## 🚀 Производительность

- **RTX 5090**: ~30-60 минут на полное обучение
- **CPU**: ~2-4 часа (не рекомендуется)
- **Размер модели**: ~50MB
- **RAM**: минимум 16GB
- **VRAM**: минимум 8GB

---

**Crypto AI Trading System v3.0** - профессиональная система с 36 целевыми переменными для точного прогнозирования криптовалютных рынков.