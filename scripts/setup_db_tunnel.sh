#!/bin/bash
# Настройка SSH туннеля для доступа к локальной БД с удаленного сервера

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🔧 Настройка туннеля к локальной PostgreSQL${NC}"

# Получаем SSH алиас из переменной окружения или используем дефолтный
SSH_ALIAS=${VAST_SSH_ALIAS:-vast-current}

echo -e "${YELLOW}📋 Настройки:${NC}"
echo "   • SSH алиас: $SSH_ALIAS"
echo "   • Локальная БД: localhost:5555"
echo "   • Удаленный порт: 5555"

# Проверяем доступность локальной БД
if ! nc -z localhost 5555 2>/dev/null; then
    echo -e "${RED}❌ PostgreSQL не запущен на порту 5555${NC}"
    echo -e "${YELLOW}Запустите БД командой:${NC}"
    echo "   docker-compose up -d postgres"
    exit 1
fi

echo -e "${GREEN}✅ Локальная БД доступна${NC}"

# Создаем обратный туннель
echo -e "${YELLOW}🚇 Создание обратного SSH туннеля...${NC}"
echo "   Удаленный сервер сможет подключаться к localhost:5555"

# Запускаем SSH с обратным туннелем
# -R remote_port:local_host:local_port
ssh -f -N -R 5555:localhost:5555 $SSH_ALIAS

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Туннель установлен!${NC}"
    echo ""
    echo -e "${YELLOW}📌 Теперь на удаленном сервере можно использовать:${NC}"
    echo "   • Хост БД: localhost"
    echo "   • Порт БД: 5555"
    echo ""
    echo -e "${YELLOW}Для остановки туннеля:${NC}"
    echo "   pkill -f 'ssh.*-R.*5555'"
else
    echo -e "${RED}❌ Ошибка создания туннеля${NC}"
    exit 1
fi