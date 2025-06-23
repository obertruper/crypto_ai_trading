# 🖥️ Информация о сервере Vast.ai

## 📡 Текущий сервер (обновлено 23.06.2025)

- **Хост**: ssh1.vast.ai
- **Порт**: 30421
- **Пользователь**: root
- **SSH ключ**: ~/.ssh/id_rsa

## 🚀 Быстрые команды

### Подключение к серверу:
```bash
ssh -p 30421 root@ssh1.vast.ai -L 8080:localhost:8080
```

### Подключение с пробросом всех портов:
```bash
./scripts/connect_vast.sh
```

### Синхронизация проекта:
```bash
./scripts/sync_to_vast.sh
```

### Мониторинг обучения:
```bash
./scripts/monitor_vast_training.sh
```

### Туннель для БД:
```bash
./setup_remote_db_tunnel.sh
```

## 📊 Проброс портов

- **8080** - Web UI
- **6006** - TensorBoard (Caddy)
- **6007** - TensorBoard (наш)
- **8888** - Jupyter Notebook
- **5555** - PostgreSQL (обратный туннель)

## 🔧 Полезные команды на сервере

```bash
# Проверка GPU
nvidia-smi

# Запуск обучения
cd /root/crypto_ai_trading
python main.py --mode full

# Мониторинг логов
tail -f logs/training_*/training.log

# TensorBoard
tensorboard --logdir logs/ --port 6007 --bind_all
```

## ⚠️ Важно

1. Всегда используйте SSH ключ `~/.ssh/id_rsa`
2. Проект находится в `/root/crypto_ai_trading`
3. Данные кэшируются в папке `cache/`
4. Модели сохраняются в `models_saved/`

## 🔄 Переключение на старый сервер

Если нужно использовать старый сервер:
- **Хост**: 109.198.107.223
- **Порт**: 48937

В скриптах выбирайте опцию 2 при запросе.