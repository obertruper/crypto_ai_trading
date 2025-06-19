#!/bin/bash
# Запуск обучения на Vast.ai сервере

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 Запуск обучения на Vast.ai GPU${NC}"

# Параметры
HOST="114.32.64.6"
PORT="40134"
REMOTE_PATH="/root/crypto_ai_trading"
# Используем правильный SSH ключ
KEY_PATH="$HOME/.ssh/vast_ai_key"

# Проверка существования ключа
if [ ! -f "$KEY_PATH" ]; then
    echo -e "${RED}❌ SSH ключ не найден: $KEY_PATH${NC}"
    echo -e "${YELLOW}Проверьте наличие файла или используйте:${NC}"
    echo "  cp ~/.ssh/id_rsa ~/.ssh/vast_ai_key"
    exit 1
fi

# SSH команда
SSH_CMD="ssh -p $PORT -i $KEY_PATH root@$HOST"

# Проверяем, передан ли режим через переменную окружения
if [ -n "$GPU_TRAINING_MODE" ]; then
    # Режим передан из интерактивного меню
    choice=$GPU_TRAINING_MODE
    EPOCHS=${GPU_TRAINING_EPOCHS:-5}
    
    case $choice in
        1)
            MODE="demo"
            ;;
        2)
            MODE="train"
            ;;
        3)
            MODE="train"
            ;;
        *)
            MODE="demo"
            ;;
    esac
else
    # Запуск напрямую из консоли - показываем меню
    echo -e "\n${YELLOW}Выберите режим запуска:${NC}"
    echo "1) Демо обучение (5 эпох)"
    echo "2) Полное обучение (100 эпох)"
    echo "3) Пользовательские настройки"
    echo -n "Выбор (1-3): "
    read choice

    case $choice in
        1)
            MODE="demo"
            EPOCHS=5
            ;;
        2)
            MODE="train"
            EPOCHS=100
            ;;
        3)
            echo -n "Количество эпох: "
            read EPOCHS
            MODE="train"
            ;;
        *)
            echo -e "${RED}❌ Неверный выбор${NC}"
            exit 1
            ;;
    esac
fi

echo -e "\n${YELLOW}📊 Параметры обучения:${NC}"
echo "   • GPU: 2x RTX 5090"
echo "   • Эпох: $EPOCHS"
echo "   • Batch size: 32"

# Создание скрипта запуска на сервере
echo -e "\n${YELLOW}🔧 Подготовка запуска...${NC}"

$SSH_CMD "cat > $REMOTE_PATH/run_training.sh << 'SCRIPT'
#!/bin/bash
cd $REMOTE_PATH

# Переданные переменные
export USE_CACHE_ONLY='${USE_CACHE_ONLY:-0}'
export MODE='${MODE}'
export EPOCHS=${EPOCHS}

# Активация окружения если есть
if [ -d 'venv' ]; then
    source venv/bin/activate
fi

# Установка переменных окружения
export CUDA_VISIBLE_DEVICES=0,1  # Используем обе GPU
export PYTHONUNBUFFERED=1

# Изменение количества эпох в конфиге
python -c \"
import yaml
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['model']['epochs'] = \$EPOCHS
config['performance']['device'] = 'cuda'
with open('config/config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
\"

# Создаем директорию для логов
mkdir -p logs

# Запуск обучения
echo '🚀 Запуск обучения на GPU...'
echo 'Режим: '\$MODE
echo 'Эпохи: '\$EPOCHS
echo 'Использовать кэш: '\$USE_CACHE_ONLY

# Проверяем флаг использования кэша
if [ \"\$USE_CACHE_ONLY\" = \"1\" ]; then
    echo '📊 Используется кэш вместо БД'
    python3 run_full_pipeline.py --mode \$MODE 2>&1 | tee logs/training_gpu.log
else
    echo '📊 Используется подключение к БД'
    python3 run_full_pipeline.py --mode \$MODE 2>&1 | tee logs/training_gpu.log
fi
SCRIPT
chmod +x $REMOTE_PATH/run_training.sh
ENDSSH

# Запуск в tmux для возможности отключения
echo -e "\n${YELLOW}🖥️  Запуск в tmux сессии...${NC}"

$SSH_CMD "
chmod +x $REMOTE_PATH/run_training.sh
tmux new-session -d -s training 'cd $REMOTE_PATH && bash run_training.sh'
"

echo -e "\n${GREEN}✅ Обучение запущено!${NC}"
echo -e "\n${YELLOW}📌 Полезные команды:${NC}"
echo "   • Подключиться к сессии: tmux attach -t training"
echo "   • Отключиться от сессии: Ctrl+B, затем D"
echo "   • Мониторинг GPU: nvidia-smi -l 1"
echo "   • Просмотр логов: tail -f logs/training_gpu.log"
echo ""
echo -e "${BLUE}💡 Для мониторинга откройте новый терминал и запустите:${NC}"
echo "   ./scripts/connect_vast.sh"
echo "   Затем откройте http://localhost:6006 для TensorBoard"

# Явный выход из скрипта
exit 0