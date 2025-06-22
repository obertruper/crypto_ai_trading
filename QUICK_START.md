# 🚀 БЫСТРЫЙ СТАРТ - Crypto AI Trading System

## 📋 Шаг 1: Проверка системы

```bash
cd "/Users/ruslan/PycharmProjects/LLM TRANSFORM/crypto_ai_trading"
python check_system.py
```

## 🎯 Шаг 2: Запуск обучения

```bash
python train_model.py
```

## ⏱️ Время выполнения

- Проверка системы: 5 секунд
- Обучение модели: 30-60 минут

## 📊 Результаты

После обучения вы получите:
- ✅ Обученную модель в `models_saved/best_model.pth`
- 📈 Метрики обучения в `experiments/logs/`
- 📊 Результаты бэктеста

## 🔧 Настройки

Для изменения параметров отредактируйте `config/config.yaml`:
- `batch_size`: размер батча (уменьшите при нехватке памяти)
- `epochs`: количество эпох обучения
- `symbols`: список торговых пар

## ❓ Проблемы?

1. **PostgreSQL не работает:**
   ```bash
   brew services restart postgresql
   ```

2. **Недостаточно памяти:**
   - Уменьшите `batch_size` до 16 или 8
   - Уменьшите количество символов

3. **Ошибки импорта:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎉 Готово!

Теперь у вас есть рабочая модель для торговли криптовалютами!