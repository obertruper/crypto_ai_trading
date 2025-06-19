# Настройка и смена серверов Vast.ai

## 🚀 Быстрая настройка нового сервера

### 1. Обновите SSH конфиг

Отредактируйте файл `~/.ssh/config` и измените секцию `vast-current`:

```ssh-config
# Текущий активный сервер Vast.ai
# Просто измените эту секцию при смене сервера
Host vast-current
    HostName YOUR_NEW_HOST       # Например: 114.32.64.6
    Port YOUR_NEW_PORT           # Например: 40134
    User root
    IdentityFile ~/.ssh/id_rsa
    ServerAliveInterval 30
    ServerAliveCountMax 3
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel QUIET
    Compression yes
    TCPKeepAlive yes
    # Автоматический проброс портов
    LocalForward 6006 localhost:6006  # TensorBoard
    LocalForward 8888 localhost:8888  # Jupyter
    LocalForward 8080 localhost:8080  # Web UI
```

### 2. Альтернативный способ - использование переменных окружения

Вы можете установить переменные окружения вместо изменения конфига:

```bash
# Использовать другой SSH алиас
export VAST_SSH_ALIAS=vast-server1

# Или прямое подключение
export VAST_HOST=114.32.64.6
export VAST_PORT=40134
export VAST_USER=root
export VAST_KEY_PATH=~/.ssh/id_rsa
```

### 3. Проверка подключения

```bash
# Быстрая проверка
ssh vast-current "echo 'Connected!' && nvidia-smi --query-gpu=name --format=csv,noheader"

# Или через скрипт
./scripts/connect_vast_simple.sh
```

## 📝 Примеры SSH алиасов для разных серверов

Добавьте в `~/.ssh/config`:

```ssh-config
# Сервер 1 - RTX 5090
Host vast-gpu1
    HostName 114.32.64.6
    Port 40134
    User root
    IdentityFile ~/.ssh/id_rsa
    # ... остальные настройки ...

# Сервер 2 - через прокси
Host vast-gpu2
    HostName ssh3.vast.ai
    Port 33915
    User root
    IdentityFile ~/.ssh/id_rsa
    # ... остальные настройки ...

# Сервер 3 - другой регион
Host vast-gpu3
    HostName 79.116.73.220
    Port 27681
    User root
    IdentityFile ~/.ssh/id_rsa
    # ... остальные настройки ...
```

## 🔧 Установка инструментов на новом сервере

```bash
# Автоматическая установка необходимых пакетов
ssh vast-current << 'EOF'
apt-get update -qq
apt-get install -y tmux screen htop ncdu
echo "✅ Инструменты установлены"
EOF
```

## 🚀 Запуск обучения

```bash
# Интерактивный режим
python run_interactive.py

# Или напрямую через скрипт
./scripts/run_on_vast_flexible.sh

# С переменными окружения
VAST_SSH_ALIAS=vast-gpu3 ./scripts/run_on_vast_flexible.sh
```

## 📊 Мониторинг

```bash
# GPU мониторинг
ssh vast-current "watch -n 1 nvidia-smi"

# Логи обучения
ssh vast-current "tail -f /root/crypto_ai_trading/logs/training_gpu.log"

# TensorBoard (порты пробрасываются автоматически)
ssh vast-current
# Затем откройте http://localhost:6006
```

## ⚡ Полезные алиасы для .bashrc/.zshrc

```bash
# Быстрое подключение к Vast
alias vast='ssh vast-current'
alias vast-gpu='ssh vast-current "nvidia-smi"'
alias vast-logs='ssh vast-current "tail -f /root/crypto_ai_trading/logs/training_gpu.log"'
alias vast-tmux='ssh -t vast-current "tmux attach -t training || tmux new -s training"'

# Смена серверов
alias vast-use-gpu1='export VAST_SSH_ALIAS=vast-gpu1'
alias vast-use-gpu2='export VAST_SSH_ALIAS=vast-gpu2'
alias vast-use-gpu3='export VAST_SSH_ALIAS=vast-gpu3'
```

## 🔄 Синхронизация проекта

```bash
# Синхронизация файлов (исключая кэш и логи)
rsync -avz --progress \
    --exclude='cache/' \
    --exclude='*.pyc' \
    --exclude='__pycache__/' \
    --exclude='.git/' \
    --exclude='logs/' \
    --exclude='models_saved/' \
    /Users/ruslan/PycharmProjects/LLM\ TRANSFORM/crypto_ai_trading/ \
    vast-current:/root/crypto_ai_trading/
```

## 📌 Важные моменты

1. **SSH ключ**: Убедитесь, что ваш публичный ключ добавлен в Vast.ai аккаунт
2. **Порты**: При смене сервера проверьте доступные порты
3. **GPU**: Разные серверы могут иметь разные GPU конфигурации
4. **Регион**: Выбирайте серверы ближе к вашему местоположению для лучшей скорости

## 🆘 Решение проблем

### Permission denied (publickey)
```bash
# Проверьте, что используется правильный ключ
ssh -v vast-current 2>&1 | grep "Offering"

# Убедитесь, что ключ добавлен в Vast.ai
cat ~/.ssh/id_rsa.pub
```

### Timeout при подключении
```bash
# Проверьте правильность хоста и порта
nc -zv YOUR_HOST YOUR_PORT

# Попробуйте через прокси
ssh -J root@ssh3.vast.ai:33915 root@YOUR_HOST
```

### tmux/screen не найден
```bash
# Установите инструменты
ssh vast-current "apt-get update && apt-get install -y tmux screen"
```