#!/bin/bash
# Проверка готовности системы для GPU обучения

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "🔍 Проверка готовности системы для GPU обучения"
echo "================================================"

# 1. Проверка SSH ключа
echo -e "\n1️⃣ Проверка SSH ключа..."
if [ -f ~/.ssh/id_rsa ]; then
    echo -e "${GREEN}✅ SSH ключ найден${NC}"
else
    echo -e "${RED}❌ SSH ключ не найден${NC}"
    echo "   Создайте файл: ~/.ssh/id_rsa"
    exit 1
fi

# 2. Проверка подключения к серверу
echo -e "\n2️⃣ Проверка подключения к серверу..."
if ssh -o ConnectTimeout=5 -p 30421 -i ~/.ssh/id_rsa root@ssh1.vast.ai "echo 'OK'" &>/dev/null; then
    echo -e "${GREEN}✅ Подключение к серверу успешно${NC}"
else
    echo -e "${RED}❌ Не удается подключиться к серверу${NC}"
    exit 1
fi

# 3. Проверка кэша локально
echo -e "\n3️⃣ Проверка кэша данных..."
if [ -f cache/features_cache.pkl ]; then
    SIZE=$(du -h cache/features_cache.pkl | cut -f1)
    echo -e "${GREEN}✅ Кэш найден (размер: $SIZE)${NC}"
else
    echo -e "${RED}❌ Кэш не найден${NC}"
    echo "   Создайте кэш через интерактивное меню"
    exit 1
fi

# 4. Проверка проекта на сервере
echo -e "\n4️⃣ Проверка проекта на сервере..."
if ssh -p 48937 -i ~/.ssh/vast_ai_key root@109.198.107.223 "test -d /root/crypto_ai_trading" &>/dev/null; then
    echo -e "${GREEN}✅ Проект найден на сервере${NC}"
else
    echo -e "${YELLOW}⚠️  Проект не найден на сервере${NC}"
    echo "   Запустите синхронизацию через меню"
fi

# 5. Проверка GPU на сервере
echo -e "\n5️⃣ Проверка GPU..."
GPU_INFO=$(ssh -p 48937 -i ~/.ssh/vast_ai_key root@109.198.107.223 "nvidia-smi --query-gpu=name --format=csv,noheader" 2>/dev/null)
if [ -n "$GPU_INFO" ]; then
    echo -e "${GREEN}✅ GPU доступны:${NC}"
    echo "$GPU_INFO" | sed 's/^/   /'
else
    echo -e "${RED}❌ GPU не обнаружены${NC}"
fi

echo -e "\n${GREEN}✅ Проверка завершена!${NC}"