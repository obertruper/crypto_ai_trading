# 🚀 Crypto AI Trading System v2.0

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13+-blue.svg)](https://www.postgresql.org/)

Универсальная система алгоритмической торговли криптовалютами с защитой от переобучения.

## 🎯 Быстрый старт

```bash
cd "/Users/ruslan/PycharmProjects/LLM TRANSFORM/crypto_ai_trading"
python train_model.py
```

Всё! Модель начнёт обучение автоматически (~30-60 минут).

## 📋 Основные команды

### 1. Обучение модели
```bash
python train_model.py
```

### 2. Интерактивное меню
```bash
python run_interactive.py
```

### 3. Полный пайплайн
```bash
python main.py --mode full
```

### 4. Только демо (проверка БД)
```bash
python main.py --mode demo
```

## ✨ Ключевые возможности

### 🧠 Защита от переобучения
- **Data Leakage Prevention** - правильная нормализация без заглядывания в будущее
- **Stochastic Depth** - случайное отключение слоев
- **Weight Noise** - добавление шума к весам
- **Label Smoothing** - сглаживание целевых меток
- **Mixup Augmentation** - смешивание примеров
- **Early Stopping** - остановка при ухудшении валидации
- **Gradient Clipping** - ограничение градиентов

### 🏗️ Архитектура
- **PatchTST** - современный Transformer для временных рядов
- **Multi-Task Learning** - предсказание цены и вероятностей
- **Ensemble Methods** - объединение нескольких моделей
- **100+ индикаторов** - технический анализ

### 💼 Риск-менеджмент
- **Kelly Criterion** - оптимальный размер позиций
- **Dynamic SL/TP** - адаптация к волатильности
- **Partial Closing** - частичные закрытия позиций
- **Portfolio Correlation** - учет корреляций

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
│   ├── patchtst.py     # PatchTST архитектура
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
├── main.py            # Главный скрипт
├── train_model.py     # Быстрый запуск обучения
└── run_interactive.py # Интерактивное меню
```

## 🔧 Конфигурация

Основные параметры в `config/config.yaml`:

```yaml
model:
  batch_size: 32        # Размер батча
  context_window: 96    # Окно контекста
  d_model: 64          # Размерность модели
  dropout: 0.4         # Dropout rate
  learning_rate: 0.0001 # Скорость обучения
  epochs: 100          # Количество эпох
  
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

## 👨‍💻 Автор

Crypto AI Trading System v2.0 - универсальная система без переобучения.