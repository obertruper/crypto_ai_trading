# 🚀 Настройка Vast.ai для GPU обучения

## 📋 Информация о сервере

- **GPU**: 2x RTX 5090 (216.2 TFLOPS)
- **VRAM**: 32 GB
- **RAM**: 129 GB
- **CPU**: Xeon® Platinum 8352V (36 cores)
- **CUDA**: 12.8

## 🔐 Настройка SSH ключа

1. **Создайте файл для приватного ключа:**
```bash
nano ~/.ssh/vast_ai_key
```

2. **Вставьте ваш приватный ключ** (начинается с `-----BEGIN RSA PRIVATE KEY-----`)

3. **Установите правильные права:**
```bash
chmod 600 ~/.ssh/vast_ai_key
```

## 🚀 Быстрый старт

### 1. Синхронизация проекта
```bash
./scripts/sync_to_vast.sh
```

### 2. Подключение к серверу
```bash
./scripts/connect_vast.sh
```

### 3. Запуск обучения
```bash
./scripts/run_on_vast.sh
```

## 📡 Способы подключения

### Прямое подключение:
```bash
ssh -p 40134 root@114.32.64.6 -i ~/.ssh/vast_ai_key
```

### Через прокси:
```bash
ssh -p 33915 root@ssh3.vast.ai -i ~/.ssh/vast_ai_key
```

### С пробросом портов:
```bash
ssh -p 40134 root@114.32.64.6 -i ~/.ssh/vast_ai_key \
  -L 8080:localhost:8080 \
  -L 6006:localhost:6006 \
  -L 8888:localhost:8888
```

## 📊 Мониторинг

После подключения с пробросом портов:

- **TensorBoard**: http://localhost:6006
- **Web UI**: http://localhost:8080
- **Jupyter**: http://localhost:8888

### Команды на сервере:
```bash
# Мониторинг GPU
nvidia-smi -l 1

# Просмотр логов
tail -f /root/crypto_ai_trading/logs/training_gpu.log

# Подключение к tmux сессии
tmux attach -t training

# Отключение от tmux: Ctrl+B, затем D
```

## ⚡ Оптимизация для GPU

Система автоматически:
- Использует обе GPU (CUDA_VISIBLE_DEVICES=0,1)
- Включает mixed precision training
- Увеличивает batch size для GPU
- Использует pin_memory для DataLoader

## 📈 Ожидаемая производительность

- **CPU (локально)**: ~3 сек/батч
- **2x RTX 5090**: ~0.1-0.2 сек/батч
- **Ускорение**: 15-30x

### Время обучения:
- **5 эпох**: ~15-20 минут
- **100 эпох**: ~5-6 часов

## 🛠️ Устранение проблем

### CUDA Out of Memory:
```yaml
# Уменьшите batch_size в config.yaml
model:
  batch_size: 16  # вместо 32
```

### Проблемы с подключением:
1. Проверьте, что сервер запущен на Vast.ai
2. Убедитесь, что SSH ключ правильный
3. Попробуйте альтернативное подключение через прокси

### Зависимости не установлены:
```bash
ssh -p 40134 root@114.32.64.6 -i ~/.ssh/vast_ai_key
cd /root/crypto_ai_trading
pip install -r requirements.txt
```

## 💰 Управление расходами

- **Стоимость**: ~$0.50-1.00 в час (проверяйте на Vast.ai)
- **Рекомендация**: Запускайте полное обучение ночью
- **Автосохранение**: Модель сохраняется каждые 10 эпох

## 📝 Чек-лист перед запуском

- [ ] SSH ключ создан и имеет права 600
- [ ] Проект синхронизирован на сервер
- [ ] Данные подготовлены локально (cache/features_cache.pkl)
- [ ] Конфигурация проверена
- [ ] Tmux установлен на сервере