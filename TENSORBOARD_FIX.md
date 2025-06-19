# 🔧 Решение проблемы с TensorBoard

## Проблема
TensorBoard запрашивает логин и пароль при открытии http://localhost:6006

## Причина
На Vast.ai сервере порт 6006 занят веб-сервером Caddy, который требует аутентификацию.

## Решение

### Вариант 1: Использовать порт 6007 (рекомендуется)

1. **Запустите TensorBoard вручную на порту 6007:**
```bash
ssh vast-current
cd /root/crypto_ai_trading
pkill -f tensorboard  # Остановить старый процесс
tensorboard --logdir ./logs --bind_all --port 6007
```

2. **В новом терминале создайте туннель:**
```bash
ssh -L 6007:localhost:6007 vast-current
```

3. **Откройте в браузере:**
```
http://localhost:6007
```

### Вариант 2: Использовать встроенный мониторинг

В интерактивном меню уже обновлен код для использования порта 6007:
```bash
python run_interactive.py
# Выберите: "🚀 Обучение на GPU" -> "📊 Мониторинг с браузером"
```

### Вариант 3: Мониторинг без TensorBoard

1. **Просмотр логов в реальном времени:**
```bash
ssh vast-current "tail -f /root/crypto_ai_trading/logs/training_gpu.log"
```

2. **Мониторинг GPU:**
```bash
ssh vast-current "watch -n 1 nvidia-smi"
```

## Проверка логов обучения

Если TensorBoard не показывает данные, проверьте наличие логов:

```bash
# На сервере
ssh vast-current
cd /root/crypto_ai_trading
find . -name "*.tfevents*" -o -name "*.pb"
```

Если логов нет, значит обучение еще не начато или логи сохраняются в другое место.

## Альтернативные инструменты мониторинга

1. **Weights & Biases** (если настроен в проекте)
2. **MLflow** (если настроен)
3. **Встроенные графики** в `experiments/plots/`

## Примечание

После следующего запуска обучения TensorBoard будет автоматически использовать порт 6007.