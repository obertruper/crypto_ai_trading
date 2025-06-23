#!/bin/bash
# Установка необходимых инструментов на Vast.ai сервере

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🔧 Установка инструментов на Vast.ai сервере${NC}"

# Параметры подключения
HOST="ssh1.vast.ai"
PORT="30421"
KEY_PATH="$HOME/.ssh/id_rsa"

# Проверка SSH ключа
if [ ! -f "$KEY_PATH" ]; then
    echo -e "${RED}❌ SSH ключ не найден: $KEY_PATH${NC}"
    exit 1
fi

# SSH команда
SSH_CMD="ssh -p $PORT -i $KEY_PATH root@$HOST"

echo -e "\n${YELLOW}📦 Подключение к серверу и установка пакетов...${NC}"

$SSH_CMD << 'REMOTE_COMMANDS'
# Обновление списка пакетов
echo "📋 Обновление списка пакетов..."
apt-get update -qq

# Установка tmux
echo "📦 Установка tmux..."
apt-get install -y tmux

# Установка screen как запасной вариант
echo "📦 Установка screen..."
apt-get install -y screen

# Установка htop для мониторинга
echo "📦 Установка htop..."
apt-get install -y htop

# Установка ncdu для анализа диска
echo "📦 Установка ncdu..."
apt-get install -y ncdu

# Установка git если нет
echo "📦 Проверка git..."
if ! command -v git &> /dev/null; then
    apt-get install -y git
fi

# Установка nvtop для мониторинга GPU
echo "📦 Установка nvtop..."
apt-get install -y nvtop || echo "nvtop недоступен"

# Создание полезных алиасов
echo "🔧 Настройка алиасов..."
cat >> ~/.bashrc << 'EOF'

# Полезные алиасы для работы с GPU
alias gpu='nvidia-smi'
alias gpuw='watch -n 1 nvidia-smi'
alias tl='tmux ls'
alias ta='tmux attach -t'
alias tn='tmux new -s'
alias logs='tail -f ~/crypto_ai_trading/logs/training_gpu.log'
alias cdcrypto='cd ~/crypto_ai_trading'

EOF

echo "✅ Установка завершена!"
echo ""
echo "🎯 Установленные инструменты:"
echo "   • tmux - для управления сессиями"
echo "   • screen - альтернатива tmux"
echo "   • htop - мониторинг процессов"
echo "   • ncdu - анализ диска"
echo "   • nvtop - мониторинг GPU (если доступен)"
echo ""
echo "💡 Полезные команды:"
echo "   • tmux new -s training - создать новую сессию"
echo "   • tmux attach -t training - подключиться к сессии"
echo "   • Ctrl+B, D - отключиться от tmux сессии"
echo "   • gpu - показать статус GPU"
echo "   • gpuw - мониторинг GPU в реальном времени"
echo "   • logs - просмотр логов обучения"

REMOTE_COMMANDS

echo -e "\n${GREEN}✅ Установка инструментов завершена!${NC}"
echo -e "${YELLOW}Теперь можно запускать обучение через run_interactive.py${NC}"