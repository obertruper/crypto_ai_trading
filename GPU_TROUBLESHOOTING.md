# 🔧 Решение проблем с GPU сервером

## 📋 Содержание
1. [Ошибки SSH подключения](#ssh-errors)
2. [Проблемы с БД](#database-issues)
3. [Ошибки TensorBoard](#tensorboard-issues)
4. [Проблемы с обучением](#training-issues)

## <a name="ssh-errors"></a>🔐 Ошибки SSH подключения

### Ошибка: "Could not resolve hostname ssh"
**Причина**: Неправильная команда SSH в скрипте  
**Решение**: Обновлен скрипт `connect_vast.sh`

### Ошибка: "Permission denied (publickey)"
**Причина**: SSH ключ не настроен или неверный  
**Решение**:
```bash
# 1. Проверьте ключ
cat ~/.ssh/vast_ai_key

# 2. Скопируйте правильный ключ
nano ~/.ssh/vast_ai_key
# Вставьте ваш приватный ключ
chmod 600 ~/.ssh/vast_ai_key

# 3. Или используйте SSH алиас
nano ~/.ssh/config
# Добавьте:
Host vast-current
    HostName 184.98.25.179
    Port 41575
    User root
    IdentityFile ~/.ssh/id_rsa
```

## <a name="database-issues"></a>💾 Проблемы с БД

### Ошибка: "connection to server at localhost:5555 failed"
**Причина**: GPU сервер не может подключиться к локальной БД  
**Решение**:

#### Вариант 1: Использовать SSH туннель (рекомендуется)
```bash
# В интерактивном меню выберите:
# "🚀 Обучение на GPU" -> "🔌 Настроить туннель БД"

# Или вручную:
./scripts/setup_db_tunnel.sh
```

#### Вариант 2: Использовать кэш вместо БД
```bash
# Создайте кэш локально:
python run_interactive.py
# Выберите: "📊 Управление данными" -> "🔧 Создать/обновить признаки"

# При запуске обучения система автоматически переключится на кэш
```

#### Вариант 3: Запустить БД на GPU сервере
```bash
ssh vast-current
cd /root/crypto_ai_trading
docker-compose up -d postgres
```

## <a name="tensorboard-issues"></a>📊 Ошибки TensorBoard

### TensorBoard не открывается
**Решение**:
```bash
# 1. Проверьте, что туннель работает
ssh vast-current -L 6006:localhost:6006

# 2. Проверьте TensorBoard на сервере
ssh vast-current "ps aux | grep tensorboard"

# 3. Запустите вручную
ssh vast-current
cd /root/crypto_ai_trading
tensorboard --logdir experiments/runs --host 0.0.0.0 --port 6006
```

## <a name="training-issues"></a>🧠 Проблемы с обучением

### Ошибка импорта модулей
**Решение**:
```bash
# Синхронизируйте проект
./scripts/sync_to_vast.sh

# Установите зависимости на сервере
ssh vast-current
cd /root/crypto_ai_trading
pip install -r requirements.txt
```

### Out of Memory (OOM)
**Решение**:
```bash
# Уменьшите batch size в config.yaml
nano config/config.yaml
# Измените:
# batch_size: 16  # вместо 32
```

## 🚀 Быстрые команды

### Полная переустановка на GPU
```bash
# 1. Очистка
ssh vast-current "rm -rf /root/crypto_ai_trading"

# 2. Синхронизация
./scripts/sync_to_vast.sh

# 3. Установка зависимостей
ssh vast-current "cd /root/crypto_ai_trading && pip install -r requirements.txt"

# 4. Запуск с кэшем
python run_interactive.py
# "🚀 Обучение на GPU" -> "🚀 Запустить обучение"
```

### Мониторинг в реальном времени
```bash
# GPU использование
ssh vast-current "watch -n 1 nvidia-smi"

# Логи обучения
ssh vast-current "tail -f /root/crypto_ai_trading/logs/training_gpu.log"

# TensorBoard
ssh -L 6006:localhost:6006 vast-current
# Откройте http://localhost:6006
```

## 📝 Проверочный список

Перед запуском обучения проверьте:

- [ ] SSH подключение работает: `ssh vast-current "echo OK"`
- [ ] Проект синхронизирован: есть опция в меню
- [ ] БД доступна ИЛИ кэш создан
- [ ] GPU доступен: `ssh vast-current "nvidia-smi"`
- [ ] Достаточно места: `ssh vast-current "df -h"`

## 🆘 Контакты поддержки

Если проблема не решается:
1. Проверьте логи: `logs/interactive/menu_*.log`
2. Создайте issue на GitHub с описанием ошибки
3. Приложите логи и скриншоты