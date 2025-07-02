# 📋 Инструкция по работе с кэшем данных

## 🔴 ВАЖНО: Кэш был пересоздан с исправленным VWAP!

### ✅ Что было исправлено:
1. **feature_engineering.py** - исправлен расчет VWAP:
   - Увеличен `min_denominator` с 1e-4 до 0.01
   - Добавлен метод `calculate_vwap` с проверками
   - Ограничен `close_vwap_ratio` максимумом 2.0

2. **Старый кэш удален** - содержал экстремальные значения до 1.14e18

## 🚀 Как создать новый кэш:

### Вариант 1: Через prepare_trading_data.py (РЕКОМЕНДУЕТСЯ)
```bash
python prepare_trading_data.py
```

Это создаст файлы в `data/processed/`:
- `train_data.parquet` 
- `val_data.parquet`
- `test_data.parquet`

### Вариант 2: Через main.py
```bash
python main.py --mode data
```

### Вариант 3: Автоматически при первом обучении
```bash
python main.py --mode train
```
Если кэш не найден, он создастся автоматически.

## 📊 Проверка кэша:

### Проверить совместимость:
```bash
python check_cache_compatibility.py
```

### Проверить значения VWAP:
```bash
python test_vwap_fix.py
```

## ⚠️ Важные моменты:

1. **prepare_trading_data.py** использует тот же `FeatureEngineer` что и `main.py`
2. Все 36 целевых переменных из `config.yaml` сохраняются в кэш
3. Структура данных полностью совместима между скриптами
4. После создания кэша, `main.py --mode train` автоматически его подхватит

## 🎯 Ожидаемые результаты после исправления:

- `close_vwap_ratio`: от 0.5 до 2.0 (вместо 1.14e18)
- Стабильное обучение без inf/NaN
- Лучшее использование GPU
- Отсутствие warnings о больших значениях

## 📝 Команды для полного цикла:

```bash
# 1. Создать кэш (10-15 минут)
python prepare_trading_data.py

# 2. Проверить кэш
python check_cache_compatibility.py

# 3. Запустить обучение
python main.py --mode train
```

---

*Обновлено: 2025-07-01 после исправления VWAP*