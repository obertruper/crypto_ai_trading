#!/bin/bash
# Обновление конфигурации сервера Vast.ai

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔄 Обновление конфигурации сервера${NC}"
echo "====================================="

# Новые параметры сервера
NEW_HOST="ssh1.vast.ai"
NEW_PORT="30421"
OLD_HOST="109.198.107.223"
OLD_PORT="48937"

echo -e "${YELLOW}Новый сервер:${NC} $NEW_HOST:$NEW_PORT"
echo -e "${YELLOW}Старый сервер:${NC} $OLD_HOST:$OLD_PORT"
echo ""

# Экспорт переменной для использования в других скриптах
export VAST_CONNECTION_MODE="1"  # 1 = новый сервер

echo -e "${GREEN}✅ Конфигурация обновлена!${NC}"
echo ""
echo "Теперь вы можете использовать:"
echo "  • ./scripts/connect_vast.sh - для подключения"
echo "  • ./scripts/sync_to_vast.sh - для синхронизации"
echo "  • ./scripts/monitor_vast_training.sh - для мониторинга"
echo ""
echo -e "${YELLOW}Совет:${NC} Добавьте в ~/.bashrc или ~/.zshrc:"
echo "export VAST_CONNECTION_MODE=1"