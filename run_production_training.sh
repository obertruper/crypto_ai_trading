#!/bin/bash

# Скрипт для запуска production обучения модели с новыми настройками

echo "🚀 Запуск PRODUCTION обучения модели для трейдинга"
echo "=================================================="

# Активируем виртуальное окружение если оно есть
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Виртуальное окружение активировано"
fi

# Проверяем наличие GPU
echo ""
echo "🖥️ Проверка GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "⚠️ GPU не обнаружен"

echo ""
echo "📊 Новые настройки для обучения:"
echo "- Пороги FLAT уменьшены в 2.5 раза (0.1%-1%)"
echo "- Веса классов: LONG=2.5, SHORT=2.5, FLAT=0.3"
echo "- Direction loss weight: 25.0"
echo "- Focal loss gamma: 5.0"
echo "- WeightedRandomSampler включен"
echo ""

# Запуск подготовки данных
echo "📥 Этап 1: Подготовка данных с новыми порогами..."
python main.py --mode data --prepare-data

if [ $? -ne 0 ]; then
    echo "❌ Ошибка при подготовке данных"
    exit 1
fi

echo ""
echo "✅ Данные подготовлены"
echo ""

# Запуск обучения с production конфигом
echo "🎯 Этап 2: Обучение модели с production конфигурацией..."
python main.py --mode production

if [ $? -ne 0 ]; then
    echo "❌ Ошибка при обучении модели"
    exit 1
fi

echo ""
echo "✅ Обучение завершено!"
echo ""

# Проверка результатов
if [ -f "models_saved/best_model.pth" ]; then
    echo "📊 Модель сохранена в: models_saved/best_model.pth"
    ls -lh models_saved/best_model.pth
else
    echo "⚠️ Модель не найдена"
fi

echo ""
echo "📈 Логи обучения доступны в: logs/"
ls -lt logs/ | head -5

echo ""
echo "🎉 Production обучение завершено!"
echo ""
echo "Следующие шаги:"
echo "1. Проверьте логи обучения для анализа метрик"
echo "2. Запустите: python evaluate_model_production.py"
echo "3. Для бэктеста: python main.py --mode backtest --model-path models_saved/best_model.pth"