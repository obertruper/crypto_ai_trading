# 🚀 Crypto AI Trading - Руководство для Claude

## 📋 О проекте
Система алгоритмической торговли криптовалютными фьючерсами с использованием PatchTST (Patch Time Series Transformer) и продвинутым риск-менеджментом.

## 🎯 Основные задачи
1. **Анализ и подготовка данных** - работа с PostgreSQL, технические индикаторы
2. **Обучение моделей** - PatchTST архитектура, ансамблирование
3. **Бэктестинг** - реалистичная симуляция торговли
4. **Оптимизация стратегий** - Kelly Criterion, динамические SL/TP

## 💻 Основные команды

### Быстрый старт:
```bash
# Демо режим (быстрый тест)
python demo_runner.py

# Интерактивное меню
python run_interactive.py

# Полный пайплайн
python main.py --mode full
```

### Работа с данными:
```bash
# Проверка данных в БД
python -c "from data.data_loader import CryptoDataLoader; loader = CryptoDataLoader(); print(loader.get_data_stats())"

# Загрузка новых данных
python main.py --mode data
```

### Обучение:
```bash
# Обучение с параметрами по умолчанию
python main.py --mode train

# Мониторинг в реальном времени
python monitor_training.py
```

### Бэктестинг:
```bash
# Тест на исторических данных
python main.py --mode backtest

# Анализ результатов
python utils/generate_report.py
```

## 📁 Структура проекта

### Основные модули:
- `config/` - конфигурация системы
- `data/` - загрузка и обработка данных
- `models/` - PatchTST и ансамбли
- `trading/` - торговые стратегии
- `training/` - обучение моделей
- `utils/` - утилиты и визуализация

### Ключевые файлы:
- `main.py` - главная точка входа
- `demo_runner.py` - демонстрационный запуск
- `config/config.yaml` - настройки системы
- `models/patchtst.py` - основная архитектура

## 🔧 Конфигурация

### База данных:
- **Хост**: localhost
- **Порт**: 5555
- **База**: crypto_trading
- **Пользователь**: ruslan

### Параметры модели:
- **Архитектура**: PatchTST
- **Patch length**: 8
- **Hidden size**: 128
- **Heads**: 8
- **Layers**: 4

### Риск-менеджмент:
- **Stop Loss**: -1.1%
- **Take Profit**: +5.8%
- **Частичные закрытия**: 40% на +1.5%, 40% на +2.5%, 20% на +4.0%

## 📊 Метрики и визуализация

### Основные метрики:
- Sharpe Ratio
- Max Drawdown
- Win Rate
- Profit Factor
- Calmar Ratio

### Визуализация:
- TensorBoard: `tensorboard --logdir logs/`
- Matplotlib графики в `results/plots/`
- Интерактивные дашборды в notebooks

## 🚀 Рабочий процесс

1. **Проверка окружения**:
   ```bash
   python setup.py
   ```

2. **Загрузка данных**:
   ```bash
   python main.py --mode data
   ```

3. **Обучение модели**:
   ```bash
   python main.py --mode train
   ```

4. **Оценка результатов**:
   ```bash
   python main.py --mode backtest
   ```

5. **Генерация отчета**:
   ```bash
   python utils/generate_report.py
   ```

## ⚠️ Важные моменты
- Всегда используем фьючерсные данные (не спот)
- Исключаем тестовые символы (TESTUSDT)
- Все изменения архитектуры - только в `models/patchtst.py`
- Логи и комментарии на русском языке
- Не создаем дубликаты файлов

## 🛠️ Разработка

### Добавление новых признаков:
Редактировать `data/feature_engineering.py`

### Новые стратегии:
Добавлять в `trading/signals.py`

### Оптимизация модели:
Изменять `models/patchtst.py`

### Метрики и визуализация:
Расширять `utils/metrics.py` и `utils/visualization.py`

## 📈 Текущий статус
- ✅ База данных настроена
- ✅ Данные загружены (3.9М записей)
- ✅ PatchTST архитектура реализована
- ✅ Бэктестинг работает
- ⚠️ Требуется дообучение на полном датасете