# 🧪 Руководство по тестированию Crypto AI Trading System

## 📋 Предварительные требования

Перед тестированием убедитесь, что:

1. **PostgreSQL запущен** на порту 5555
2. **Python окружение активировано**: `source venv/bin/activate`
3. **Все зависимости установлены**: `pip install -r requirements.txt`

## 🚀 Быстрый старт - Интерактивное меню

```bash
python run_interactive.py
```

Это откроет главное меню для управления системой. Рекомендуемый порядок действий:

1. **Проверка системы** (опция 6)
2. **Управление данными** (опция 1)
3. **Демо обучение** (опция 2 → 1)
4. **Мониторинг** (опция 3)

## 📊 Пошаговое тестирование

### 1. Проверка подключения к БД

```bash
# Через интерактивное меню:
# 1 (Управление данными) → 1 (Проверить подключение к БД)

# Или напрямую:
python -c "
from data.data_loader import CryptoDataLoader
loader = CryptoDataLoader({'database': {
    'host': 'localhost',
    'port': 5555,
    'database': 'crypto_trading',
    'user': 'ruslan',
    'password': 'ruslan'
}})
print('✅ БД подключена успешно!')
"
```

### 2. Загрузка и подготовка данных

```bash
# Через меню: 1 → 2 (Загрузить данные из БД)
# Или командная строка:
python run_full_pipeline.py --mode data
```

**Ожидаемый результат:**
- Загрузка ~1 млн записей
- Создание 100+ технических индикаторов
- Сохранение в `cache/features_cache.pkl`
- Время выполнения: 1-3 минуты

### 3. Демо обучение (5 эпох)

```bash
# Через меню: 2 → 1 (Быстрое обучение)
# Или изменить epochs в config.yaml на 5 и запустить:
python run_full_pipeline.py --mode train
```

**Мониторинг в реальном времени:**
```bash
# В отдельном терминале:
python monitor_training.py
```

**Ожидаемый результат:**
- 5 эпох обучения
- Сохранение модели в `models_saved/`
- Логи в `experiments/logs/`
- Время: 10-15 минут на CPU

### 4. Проверка результатов

```bash
# Через меню: 3 → 4 (Результаты последнего обучения)

# Или просмотр логов:
tail -50 experiments/logs/Trainer_*.log

# Проверка сохраненной модели:
ls -la models_saved/
```

## 🔍 Детальное тестирование компонентов

### Тест 1: Валидация данных

```python
# test_data_validation.py
import pickle
import pandas as pd

# Загрузка данных
with open('cache/features_cache.pkl', 'rb') as f:
    features = pickle.load(f)

print(f"Размер данных: {features.shape}")
print(f"Проверка NaN: {features.isna().sum().sum()}")
print(f"Диапазон дат: {features.datetime.min()} - {features.datetime.max()}")

# Проверка целевых переменных
target_cols = [col for col in features.columns if col.startswith('future_return_')]
print(f"\nЦелевые переменные: {target_cols}")
print(features[target_cols].describe())
```

### Тест 2: Проверка модели

```python
# test_model.py
import torch
from models.patchtst import PatchTSTForPrediction

# Создание модели
model = PatchTSTForPrediction(
    c_in=86,
    c_out=8,
    context_window=96,
    target_window=4,
    patch_len=16,
    stride=8,
    n_layers=2,
    d_model=128,
    n_heads=8,
    d_ff=256,
    dropout=0.2
)

# Тест forward pass
x = torch.randn(2, 96, 86)
with torch.no_grad():
    out = model(x)
    
print(f"Вход: {x.shape}")
print(f"Выход: {out.shape}")
print(f"Параметров в модели: {sum(p.numel() for p in model.parameters()):,}")
```

### Тест 3: Создание признаков

```python
# test_features.py
from data.feature_engineering import FeatureEngineer
import pandas as pd

# Создание тестовых данных
test_data = pd.DataFrame({
    'symbol': ['BTCUSDT'] * 200,
    'datetime': pd.date_range('2024-01-01', periods=200, freq='15min'),
    'open': 40000 + np.random.randn(200) * 100,
    'high': 40100 + np.random.randn(200) * 100,
    'low': 39900 + np.random.randn(200) * 100,
    'close': 40050 + np.random.randn(200) * 100,
    'volume': 1000000 + np.random.randn(200) * 10000
})

# Создание признаков
fe = FeatureEngineer({})
features = fe.create_features(test_data)

print(f"Создано признаков: {len(features.columns)}")
print(f"Примеры признаков: {features.columns[:10].tolist()}")
```

## 📈 Тестирование производительности

### На CPU (локально)

```bash
# Измерение времени обучения 1 эпохи
time python -c "
from run_full_pipeline import run_training_pipeline
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['model']['epochs'] = 1

# Загрузка данных из кэша
import pickle
with open('cache/features_cache.pkl', 'rb') as f:
    features = pickle.load(f)

# Запуск обучения
run_training_pipeline(config, features, None)
"
```

**Ожидаемое время:**
- 1 эпоха: 5-10 минут на CPU
- 100 эпох: 8-16 часов на CPU

### Проверка GPU (если доступен)

```python
import torch
print(f"CUDA доступна: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

## 🐛 Устранение типичных проблем

### Проблема 1: Ошибка подключения к БД

```bash
# Проверка PostgreSQL
pg_isready -h localhost -p 5555

# Если не работает, проверьте процесс:
ps aux | grep postgres
```

### Проблема 2: NaN в данных

```bash
# Пересоздание признаков
rm cache/features_cache.pkl
python run_full_pipeline.py --mode data
```

### Проблема 3: Нехватка памяти

```yaml
# Уменьшите batch_size в config/config.yaml:
model:
  batch_size: 16  # вместо 32
```

### Проблема 4: Медленное обучение

```yaml
# Упростите модель в config/config.yaml:
model:
  d_model: 64    # вместо 128
  n_heads: 4     # вместо 8
  e_layers: 1    # вместо 2
```

## 📊 Проверка результатов

### Метрики обучения

```bash
# Просмотр CSV с метриками
cat experiments/runs/training_*/metrics.csv | head -20

# График обучения (если установлен matplotlib)
python -c "
import pandas as pd
import matplotlib.pyplot as plt

metrics = pd.read_csv('experiments/runs/training_*/metrics.csv')
plt.plot(metrics['epoch'], metrics['train_loss'], label='Train')
plt.plot(metrics['epoch'], metrics['val_loss'], label='Val')
plt.legend()
plt.title('Training Progress')
plt.show()
"
```

### TensorBoard

```bash
# Запуск TensorBoard
tensorboard --logdir experiments/runs/

# Откройте в браузере: http://localhost:6006
```

## 🚀 Развертывание на GPU сервере

После успешного локального тестирования:

1. **Настройте сервер в config.yaml:**
```yaml
remote_server:
  enabled: true
  host: "YOUR_SERVER_IP"
  port: 22
  user: "root"
```

2. **Создайте скрипты для деплоя** (будут добавлены позже)

3. **Синхронизируйте код и запустите обучение**

## ✅ Чек-лист перед продакшеном

- [ ] БД работает и содержит данные
- [ ] Демо обучение проходит без ошибок
- [ ] Модель сохраняется корректно
- [ ] Логи пишутся правильно
- [ ] Мониторинг работает
- [ ] NaN значений нет в данных
- [ ] Производительность приемлемая

## 📞 Поддержка

При возникновении проблем:
1. Проверьте логи в `experiments/logs/`
2. Используйте интерактивное меню для диагностики
3. Создайте issue на GitHub с описанием проблемы