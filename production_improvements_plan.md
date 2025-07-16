# 🎯 План улучшений для production режима

## ✅ Что уже сделано:
1. Исправлен импорт `UnifiedPatchTSTForTrading` в `training/fine_tuner.py`
2. Добавлена поддержка `DirectionalMultiTaskLoss` в `models/losses.py`
3. Добавлен метод `validate_epoch` в `OptimizedTrainer`
4. Добавлена секция `risk_management` в `config_production.yaml`
5. Валидация конфигурации успешно пройдена

## 📊 Текущие проблемы модели:
- **Direction Accuracy**: 36.3% (целевая 42-45%)
- **Дисбаланс предсказаний**: FLAT всего 15.3% (в данных 25.4%)
- **Win Rate**: 46.3% (целевая 50%+)
- **Переобучение**: val loss 5.26 vs train loss 2.56

## 🚀 Рекомендуемые улучшения:

### 1. Изменить веса классов для лучшего баланса:
В `config_production.yaml` изменить:
```yaml
class_weights: [1.0, 1.0, 2.5]  # Увеличить вес FLAT класса
```

### 2. Увеличить пороги направлений:
В `data/feature_engineering.py`:
```python
direction_thresholds = {
    '15m': 0.003,  # 0.3% вместо 0.15%
    '1h': 0.005,   # 0.5% вместо 0.3%
    '4h': 0.01,    # 1.0% вместо 0.7%
    '12h': 0.015   # 1.5% вместо 1.0%
}
```

### 3. Использовать staged training:
Уже настроено в конфигурации - сначала учим только direction, потом добавляем остальное.

### 4. Fine-tuning с существующей моделью:
```bash
python main.py --mode production --checkpoint models_saved/best_model_20250711_122219.pth
```

### 5. Увеличить confidence threshold для сигналов:
В `trading/signals.py`:
- `min_confidence = 0.75` (вместо 0.6)
- `min_signal_strength = 0.8` (вместо 0.7)

## 📈 Ожидаемые результаты:
- Direction Accuracy: 42-45%
- Win Rate: 50-52%
- Правильное распределение FLAT: 20-25%
- Меньше ложных сигналов
- Лучшее понимание когда НЕ торговать

## 🔧 Команда для запуска:
```bash
# С fine-tuning от существующей модели
python main.py --mode production --checkpoint models_saved/best_model_20250711_122219.pth

# Или полное переобучение
python main.py --mode production
```