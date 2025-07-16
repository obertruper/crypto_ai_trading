# 🎯 Direction Model - Руководство по использованию

## 📊 Проблема и решение

### Проблема:
- Текущая модель с 20 выходами показала Win Rate 45.6% (хуже случайного)
- Модель научилась предсказывать околонулевые значения
- MSE Loss не оптимизирует прибыльность

### Решение:
- **DirectionPredictor** - специализированная модель для предсказания направления
- **Enhanced Features** - 50+ новых признаков (market regime, microstructure, sentiment)
- **Profit-aware Loss** - оптимизация прибыли, а не MSE

## 🚀 Быстрый старт

```bash
# Автоматический запуск всех этапов
./quick_start_direction.sh
```

Или пошагово:

### 1. Подготовка enhanced датасета
```bash
python prepare_enhanced_dataset.py --symbols BTCUSDT ETHUSDT SOLUSDT --start-date 2024-01-01
```

### 2. Обучение Direction модели
```bash
python train_direction_model.py --config configs/direction_only.yaml --epochs 50
```

### 3. Оценка результатов
```bash
python evaluate_direction_model.py --checkpoint models_saved/best_direction_model_*.pth
```

### 4. Бэктест стратегии
```bash
python backtest_direction_strategy.py --checkpoint models_saved/best_direction_model_*.pth
```

## 📈 Архитектура DirectionPredictor

```
Входы (171+ признаков)
    ↓
Multi-Scale Patches (1h, 4h, 16h паттерны)
    ↓
Temporal Consistency Module
    ↓
Direction-Specific Encoder (4 слоя)
    ↓
Attention Pooling
    ↓
Выходы: UP/DOWN/FLAT для 4 таймфреймов
```

### Ключевые особенности:
1. **Multi-scale patches** - захват паттернов разных таймфреймов
2. **Temporal consistency** - согласованность между 15m, 1h, 4h предсказаниями
3. **DirectionalTradingLoss** - учет комиссий и потенциальной прибыли
4. **Attention pooling** - фокус на важных моментах

## 🧪 Enhanced Features

### Market Regime (7 признаков)
- `regime_trend` - trending vs ranging рынок
- `regime_volatility` - уровень волатильности
- `wyckoff_phase` - фазы накопления/распределения
- `trend_strength` - сила текущего тренда

### Microstructure (10 признаков)
- `order_flow_imbalance` - дисбаланс покупок/продаж
- `tick_volume` - объем по направлению тиков
- `aggressive_buying/selling` - агрессивные сделки
- `price_impact` - влияние объема на цену

### Cross-Asset (8 признаков)
- `btc_correlation_*` - корреляция с Bitcoin
- `btc_lead_indicator` - BTC как опережающий индикатор
- `sector_strength` - сила сектора (DeFi, L1, etc)
- `beta_*` - рыночная бета

### Sentiment Proxy (5 признаков)
- `fear_greed_index` - приближение индекса страха/жадности
- `panic_selling` - детекция паники
- `euphoria_buying` - детекция эйфории
- `accumulation/distribution` - фазы накопления

## 📊 Метрики для оценки

### Критические метрики:
1. **Directional Accuracy** > 55% - минимум для прибыльности
2. **Win Rate** > 50% с учетом комиссий
3. **Profit Factor** > 1.2
4. **Sharpe Ratio** > 1.0

### Оценка готовности:
```
✅ ГОТОВА К ТОРГОВЛЕ:
- Directional Accuracy > 58%
- Win Rate > 52%
- Profit Factor > 1.5

⚠️ ТРЕБУЕТ ДОРАБОТКИ:
- Directional Accuracy 52-58%
- Win Rate 48-52%
- Profit Factor 1.0-1.5

❌ НЕ ГОТОВА:
- Directional Accuracy < 52%
- Win Rate < 48%
- Profit Factor < 1.0
```

## 🛠️ Настройка параметров

### configs/direction_only.yaml
```yaml
model:
  learning_rate: 0.0001    # Увеличить если медленная сходимость
  dropout: 0.3             # Увеличить если переобучение
  batch_size: 256          # Уменьшить для лучшей генерализации

loss:
  large_move_threshold: 0.02   # Порог крупных движений (2%)
  large_move_weight: 5.0       # Вес для крупных движений
  wrong_direction_penalty: 2.0 # Штраф за неправильное направление

# Минимальная уверенность для торговли
min_confidence: 0.65
```

## 📈 Мониторинг обучения

### TensorBoard
```bash
tensorboard --logdir logs/direction_training_*/tensorboard/
```

### Встроенный мониторинг
```bash
# В отдельном терминале
python monitor_training.py
```

### Что смотреть:
1. **Directional Accuracy** на валидации - должна расти
2. **Loss** - должен снижаться, но не слишком быстро
3. **Win Rate** - ключевая метрика для прибыльности

## 🔧 Troubleshooting

### Низкая accuracy (<50%)
1. Увеличьте количество эпох до 100
2. Добавьте больше символов в обучение
3. Проверьте качество данных

### Переобучение (train >> val)
1. Увеличьте dropout до 0.4-0.5
2. Уменьшите learning rate в 2 раза
3. Добавьте weight_decay в optimizer

### Медленная сходимость
1. Увеличьте learning rate до 0.0005
2. Используйте OneCycleLR scheduler
3. Проверьте нормализацию данных

## 🚀 Production использование

### После успешного обучения:
```python
# Загрузка модели
from models.direction_predictor import DirectionPredictor
import torch

checkpoint = torch.load('models_saved/best_direction_model.pth')
model = DirectionPredictor(checkpoint['config']['model'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Предсказание
with torch.no_grad():
    predictions = model(features)
    confidence_dict = model.predict_with_confidence(features)
    
# Торговое решение
if confidence_dict['direction_4h'][1] > 0.65:  # confidence > 65%
    direction = confidence_dict['direction_4h'][0]  # 0=UP, 1=DOWN, 2=FLAT
    if direction == 0:
        print("ОТКРЫТЬ LONG")
    elif direction == 1:
        print("ОТКРЫТЬ SHORT")
```

## 📝 Дальнейшие улучшения

1. **Ensemble моделей**
   - Обучить 5 моделей с разной инициализацией
   - Голосование по большинству

2. **Адаптивные стратегии**
   - Разные параметры для разных market regimes
   - Динамическое изменение confidence threshold

3. **Дополнительные данные**
   - Новости через API
   - On-chain метрики
   - Funding rates

## ⚠️ Важные замечания

1. **ВСЕГДА** тестируйте на демо счете минимум неделю
2. **НЕ ИСПОЛЬЗУЙТЕ** модель с accuracy < 55%
3. **Мониторьте** производительность в реальном времени
4. **Обновляйте** модель каждые 2-4 недели

---

📧 Вопросы: создайте issue в репозитории
📊 Результаты: делитесь в discussions