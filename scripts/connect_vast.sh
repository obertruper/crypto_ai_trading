#!/bin/bash
# Подключение к Vast.ai серверу с пробросом портов

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Подключение к Vast.ai GPU серверу${NC}"

# Выбор способа подключения
if [ -n "$VAST_CONNECTION_MODE" ]; then
    # Автоматический выбор из переменной окружения
    choice=$VAST_CONNECTION_MODE
else
    # Интерактивный выбор
    echo -e "\n${YELLOW}Выберите способ подключения:${NC}"
    echo "1) Прямое подключение (114.32.64.6:40134)"
    echo "2) Через прокси (ssh3.vast.ai:33915)"
    echo -n "Выбор (1/2): "
    read choice
fi

if [ "$choice" = "1" ]; then
    HOST="114.32.64.6"
    PORT="40134"
else
    HOST="ssh3.vast.ai"
    PORT="33915"
fi

# Проверка SSH ключа
KEY_PATH="$HOME/.ssh/vast_ai_key"
if [ ! -f "$KEY_PATH" ]; then
    echo -e "${YELLOW}⚠️  SSH ключ не найден в $KEY_PATH${NC}"
    echo "Создайте файл и вставьте ваш приватный ключ:"
    echo "nano $KEY_PATH"
    echo "chmod 600 $KEY_PATH"
    exit 1
fi

# SSH команда с пробросом портов
SSH_CMD="ssh -p $PORT root@$HOST"
SSH_CMD="$SSH_CMD -i $KEY_PATH"
SSH_CMD="$SSH_CMD -L 8080:localhost:8080"  # Web UI
SSH_CMD="$SSH_CMD -L 6006:localhost:6006"  # TensorBoard
SSH_CMD="$SSH_CMD -L 8888:localhost:8888"  # Jupyter

echo -e "${GREEN}📡 Подключение к $HOST:$PORT${NC}"
echo -e "${GREEN}🌐 Проброшенные порты:${NC}"
echo "   • Web UI:     http://localhost:8080"
echo "   • TensorBoard: http://localhost:6006"
echo "   • Jupyter:     http://localhost:8888"
echo ""

# Выбор режима подключения
echo -e "${YELLOW}Выберите режим подключения:${NC}"
echo "1) Интерактивная SSH сессия с пробросом портов"
echo "2) Только проброс портов (фоновый режим)"
echo -n "Выбор (1/2): "
read mode

if [ "$mode" = "2" ]; then
    # Устанавливаем туннели в фоне
    echo -e "${YELLOW}Устанавливаем туннели в фоновом режиме...${NC}"
    # Запускаем SSH команду в фоне с правильными параметрами
    $SSH_CMD -f -N
    echo -e "${GREEN}✅ Туннели установлены!${NC}"
    echo ""
    echo "Теперь вы можете использовать:"
    echo "  • TensorBoard: http://localhost:6006"
    echo "  • Web UI: http://localhost:8080"
    echo "  • Jupyter: http://localhost:8888"
    echo ""
    echo -e "${YELLOW}Для остановки туннелей используйте:${NC}"
    echo "  pkill -f 'ssh.*-L.*6006'"
else
    # Интерактивное подключение
    echo -e "${GREEN}Подключение к серверу...${NC}"
    echo -e "${YELLOW}Для отключения используйте: exit${NC}"
    echo ""
    exec $SSH_CMD
fi