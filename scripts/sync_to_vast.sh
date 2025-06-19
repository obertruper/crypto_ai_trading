#!/bin/bash
# Синхронизация проекта с Vast.ai сервером

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔄 Синхронизация проекта с Vast.ai${NC}"

# Параметры из конфига (можно переопределить)
HOST="114.32.64.6"
PORT="40134"
REMOTE_PATH="/root/crypto_ai_trading"
KEY_PATH="$HOME/.ssh/vast_ai_key"

# Проверка ключа
if [ ! -f "$KEY_PATH" ]; then
    echo -e "${YELLOW}⚠️  SSH ключ не найден: $KEY_PATH${NC}"
    echo -e "Попробуем использовать id_rsa..."
    KEY_PATH="$HOME/.ssh/id_rsa"
    if [ ! -f "$KEY_PATH" ]; then
        echo -e "${RED}❌ SSH ключ не найден${NC}"
        exit 1
    fi
fi

# Исключения для rsync
EXCLUDES=(
    "--exclude=.git/"
    "--exclude=__pycache__/"
    "--exclude=*.pyc"
    "--exclude=.DS_Store"
    "--exclude=cache/"
    "--exclude=logs/"
    "--exclude=experiments/runs/"
    "--exclude=models_saved/*.pth"
    "--exclude=.venv/"
    "--exclude=venv/"
)

echo -e "${YELLOW}📤 Загрузка файлов...${NC}"

# Создание директории на сервере
ssh -p $PORT -i $KEY_PATH root@$HOST "mkdir -p $REMOTE_PATH"

# Синхронизация
rsync -avzP \
    -e "ssh -p $PORT -i $KEY_PATH" \
    "${EXCLUDES[@]}" \
    ./ root@$HOST:$REMOTE_PATH/

echo -e "${GREEN}✅ Синхронизация завершена!${NC}"

# Опционально: установка зависимостей
echo -e "\n${YELLOW}Установить зависимости на сервере? (y/n)${NC}"
read -n 1 install_deps
echo

if [ "$install_deps" = "y" ]; then
    echo -e "${YELLOW}📦 Установка зависимостей...${NC}"
    ssh -p $PORT -i $KEY_PATH root@$HOST "cd $REMOTE_PATH && pip install -r requirements.txt"
    echo -e "${GREEN}✅ Зависимости установлены!${NC}"
fi