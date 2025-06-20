#!/usr/bin/env python3
"""
Полноценный запуск Crypto AI Trading System
"""

import sys
import os
import argparse
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Импорты проекта
from utils.logger import get_logger, setup_logging
from data.data_loader import CryptoDataLoader
from data.feature_engineering import FeatureEngineer
from data.dataset import TradingDataset
from models.patchtst import PatchTSTForPrediction
from models.ensemble import BaseEnsemble
from training.trainer import Trainer
from training.validator import ModelValidator
from trading.risk_manager import RiskManager
from trading.signals import SignalGenerator
from trading.backtester import Backtester
from utils.visualization import TradingVisualizer
from torch.utils.data import DataLoader

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Загрузка конфигурации"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def run_data_pipeline(config: dict, logger):
    """Этап 1: Загрузка и подготовка данных"""
    logger.info("="*80)
    logger.info("📊 ЭТАП 1: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
    logger.info("="*80)
    
    # Проверяем флаг использования кэша
    use_cache_only = os.environ.get('USE_CACHE_ONLY', '0')
    logger.info(f"🔍 USE_CACHE_ONLY = '{use_cache_only}'")
    
    if use_cache_only == '1':
        logger.info("📦 Используется режим работы с кэшем (без БД)")
        cache_path = Path("cache/features_cache.pkl")
        
        if not cache_path.exists():
            logger.error(f"❌ Файл кэша не найден: {cache_path}")
            logger.error(f"Текущая директория: {os.getcwd()}")
            logger.error(f"Содержимое cache/: {list(Path('cache').glob('*')) if Path('cache').exists() else 'Директория не существует'}")
            raise FileNotFoundError(f"Файл кэша не найден: {cache_path}")
        
        logger.info(f"✅ Загрузка данных из кэша: {cache_path}")
        logger.info(f"📏 Размер файла: {cache_path.stat().st_size / (1024*1024):.1f} MB")
        
        import pickle
        with open(cache_path, 'rb') as f:
            features_df = pickle.load(f)
        
        logger.info(f"✅ Загружено {len(features_df):,} записей из кэша")
        logger.info(f"📊 Форма данных: {features_df.shape}")
        logger.info(f"📅 Период: {features_df['datetime'].min()} - {features_df['datetime'].max()}")
        logger.info(f"🏷️ Символы: {sorted(features_df['symbol'].unique())}")
        
        return features_df
    
    # Обычный режим работы с БД
    # Инициализация загрузчика данных
    data_loader = CryptoDataLoader(config)
    
    # Получение доступных символов
    available_symbols = data_loader.get_available_symbols()
    logger.info(f"Найдено {len(available_symbols)} символов в БД")
    
    # Используем символы из конфига или все доступные
    symbols = config['data']['symbols']
    if symbols == 'all' or symbols == 'ALL' or (isinstance(symbols, list) and 'ALL' in symbols):
        symbols = available_symbols
    else:
        # Проверяем, что символы есть в БД
        symbols = [s for s in symbols if s in available_symbols]
    
    logger.info(f"Будет загружено данных для {len(symbols)} символов: {symbols[:5]}...")
    
    # Определение периода данных - используем все доступные данные
    end_date = datetime.now()
    start_date = datetime(2022, 6, 8)  # Начало данных в БД
    
    # Загрузка сырых данных
    raw_data = data_loader.load_data(
        symbols=symbols,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    logger.info(f"✅ Загружено {len(raw_data):,} записей")
    
    # Создание технических индикаторов
    logger.info("\n🔧 Создание технических индикаторов...")
    feature_engineer = FeatureEngineer(config)
    
    # Обработка по символам для корректного расчета индикаторов
    processed_data = []
    for symbol in symbols:
        symbol_data = raw_data[raw_data['symbol'] == symbol].copy()
        if len(symbol_data) < 100:
            logger.warning(f"Недостаточно данных для {symbol} ({len(symbol_data)} записей), пропускаем")
            continue
            
        features = feature_engineer.create_features(symbol_data)
        processed_data.append(features)
        logger.info(f"  {symbol}: {len(features)} записей, {features.shape[1]} признаков")
    
    # Объединение всех данных
    all_features = pd.concat(processed_data, ignore_index=True)
    logger.info(f"\n✅ Создано признаков: {all_features.shape}")
    
    # Сохранение промежуточных результатов
    cache_path = Path("cache/features_cache.pkl")
    cache_path.parent.mkdir(exist_ok=True)
    all_features.to_pickle(cache_path)
    logger.info(f"💾 Признаки сохранены в {cache_path}")
    
    return all_features

def run_training_pipeline(config: dict, features_data, logger):
    """Этап 2: Обучение моделей"""
    logger.info("\n" + "="*80)
    logger.info("🧠 ЭТАП 2: ОБУЧЕНИЕ МОДЕЛЕЙ")
    logger.info("="*80)
    
    # Разделение данных
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    
    n_samples = len(features_data)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    # Сортировка по времени
    features_data = features_data.sort_values(['symbol', 'datetime'])
    
    train_data = features_data.iloc[:train_size]
    val_data = features_data.iloc[train_size:train_size+val_size]
    test_data = features_data.iloc[train_size+val_size:]
    
    logger.info(f"Разделение данных:")
    logger.info(f"  Train: {len(train_data):,} записей ({train_ratio*100:.0f}%)")
    logger.info(f"  Val: {len(val_data):,} записей ({val_ratio*100:.0f}%)")
    logger.info(f"  Test: {len(test_data):,} записей ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    # Создание датасетов
    train_dataset = TradingDataset(
        data=train_data,
        context_window=config['model']['context_window'],
        prediction_window=config['model']['pred_len']
    )
    val_dataset = TradingDataset(
        data=val_data,
        context_window=config['model']['context_window'],
        prediction_window=config['model']['pred_len']
    )
    test_dataset = TradingDataset(
        data=test_data,
        context_window=config['model']['context_window'],
        prediction_window=config['model']['pred_len']
    )
    
    # Инициализация модели
    logger.info("\n🏗️ Инициализация модели PatchTST...")
    # Получаем количество признаков и целевых переменных из данных
    n_features = len(train_dataset.feature_cols)
    n_targets = len(train_dataset.target_cols)
    
    logger.info(f"Входные признаки: {n_features}, Целевые переменные: {n_targets}")
    
    model = PatchTSTForPrediction(
        c_in=n_features,
        c_out=n_targets,
        context_window=config['model']['context_window'],
        target_window=config['model']['pred_len'],
        patch_len=config['model']['patch_len'],
        stride=config['model']['stride'],
        n_layers=config['model']['e_layers'],
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout']
    )
    
    # Создание трейнера
    trainer = Trainer(model, config)
    
    # Создание DataLoader'ов
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['model']['batch_size'],
        shuffle=True,
        num_workers=config['performance'].get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['model']['batch_size'],
        shuffle=False,
        num_workers=config['performance'].get('num_workers', 4),
        pin_memory=True
    )
    
    # Обучение
    logger.info("\n🚀 Начало обучения...")
    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Сохранение лучшей модели
    best_model_path = trainer.checkpoint_dir / "best_model.pth"
    
    logger.info(f"\n✅ Обучение завершено. Лучшая модель: {best_model_path}")
    
    # Валидация на тестовом наборе
    logger.info("\n📊 Валидация на тестовом наборе...")
    validator = ModelValidator(config)
    test_metrics = validator.validate(model, test_dataset)
    
    logger.info("Метрики на тестовом наборе:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return model, test_data

def run_backtesting_pipeline(config: dict, model, test_data, logger):
    """Этап 3: Бэктестинг торговой стратегии"""
    logger.info("\n" + "="*80)
    logger.info("💰 ЭТАП 3: БЭКТЕСТИНГ ТОРГОВОЙ СТРАТЕГИИ")
    logger.info("="*80)
    
    # Инициализация компонентов торговли
    risk_manager = RiskManager(config)
    signal_generator = SignalGenerator(config)
    backtester = Backtester(config)
    
    # Генерация предсказаний модели
    logger.info("🔮 Генерация предсказаний модели...")
    predictions = model.predict(test_data)
    
    # Генерация торговых сигналов
    logger.info("📡 Генерация торговых сигналов...")
    signals = signal_generator.generate_signals(predictions, test_data)
    logger.info(f"  Сгенерировано {len(signals)} сигналов")
    
    # Запуск бэктестинга
    logger.info("\n🏃 Запуск бэктестинга...")
    backtest_results = backtester.run_backtest(
        market_data=test_data,
        features=test_data,
        model_predictions=predictions
    )
    
    # Отображение результатов
    logger.info("\n📈 РЕЗУЛЬТАТЫ БЭКТЕСТИНГА:")
    logger.info(f"  Начальный капитал: ${backtest_results['initial_capital']:,.2f}")
    logger.info(f"  Финальный капитал: ${backtest_results['final_capital']:,.2f}")
    logger.info(f"  Общая доходность: {backtest_results['total_return_pct']:.2f}%")
    logger.info(f"  Коэффициент Шарпа: {backtest_results['sharpe_ratio']:.2f}")
    logger.info(f"  Максимальная просадка: {backtest_results['max_drawdown_pct']:.2f}%")
    logger.info(f"  Win Rate: {backtest_results['win_rate_pct']:.2f}%")
    logger.info(f"  Всего сделок: {backtest_results['total_trades']}")
    
    return backtest_results

def run_visualization_pipeline(config: dict, backtest_results, logger):
    """Этап 4: Визуализация результатов"""
    logger.info("\n" + "="*80)
    logger.info("📊 ЭТАП 4: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    logger.info("="*80)
    
    visualizer = TradingVisualizer(config)
    
    # Создание графиков
    logger.info("📈 Создание графиков...")
    
    # График эквити
    equity_plot = visualizer.plot_equity_curve(backtest_results['equity_curve'])
    logger.info(f"  ✅ График эквити сохранен: {equity_plot}")
    
    # График просадок
    drawdown_plot = visualizer.plot_drawdown(backtest_results['equity_curve'])
    logger.info(f"  ✅ График просадок сохранен: {drawdown_plot}")
    
    # Статистика по символам
    if 'performance_by_symbol' in backtest_results:
        symbol_plot = visualizer.plot_symbol_performance(backtest_results['performance_by_symbol'])
        logger.info(f"  ✅ График по символам сохранен: {symbol_plot}")
    
    # Генерация отчета
    report_path = Path("results/backtest_report.txt")
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(backtester.generate_report(backtest_results))
    
    logger.info(f"\n📄 Полный отчет сохранен: {report_path}")

def main():
    """Основная функция запуска"""
    parser = argparse.ArgumentParser(description='Crypto AI Trading System')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Путь к файлу конфигурации')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'data', 'train', 'backtest', 'demo'],
                        help='Режим запуска системы')
    parser.add_argument('--skip-cache', action='store_true',
                        help='Пропустить использование кэша')
    
    args = parser.parse_args()
    
    # Валидация конфигурации перед загрузкой
    from utils.config_validator_main import validate_and_exit_on_error
    validate_and_exit_on_error(args.config)
    
    # Загрузка конфигурации
    config = load_config(args.config)
    
    # Настройка логирования
    setup_logging(config)
    logger = get_logger("CryptoAI")
    
    logger.info("="*80)
    logger.info("🚀 ЗАПУСК CRYPTO AI TRADING SYSTEM")
    logger.info(f"📋 Режим: {args.mode}")
    logger.info(f"⚙️ Конфигурация: {args.config}")
    logger.info(f"🖥️ Устройство: {config['performance']['device']}")
    logger.info(f"📦 USE_CACHE_ONLY: {os.environ.get('USE_CACHE_ONLY', 'не установлено')}")
    logger.info("="*80)
    
    try:
        if args.mode == 'demo':
            # Демо режим - только проверка подключений
            logger.info("🎯 Демонстрационный режим")
            data_loader = CryptoDataLoader(config)
            symbols = data_loader.get_available_symbols()
            logger.info(f"✅ Система готова к работе. Найдено {len(symbols)} символов")
            
        elif args.mode == 'data':
            # Только подготовка данных
            features = run_data_pipeline(config, logger)
            logger.info(f"\n✅ Данные подготовлены: {features.shape}")
            
        elif args.mode == 'train':
            # Обучение на подготовленных данных
            cache_path = Path("cache/features_cache.pkl")
            if cache_path.exists() and not args.skip_cache:
                logger.info(f"📂 Загрузка данных из кэша: {cache_path}")
                features = pd.read_pickle(cache_path)
            else:
                features = run_data_pipeline(config, logger)
            
            model, test_data = run_training_pipeline(config, features, logger)
            logger.info("\n✅ Модель обучена и сохранена")
            
        elif args.mode == 'backtest':
            # Бэктестинг с обученной моделью
            # Здесь нужно загрузить обученную модель
            logger.info("🔄 Режим бэктестинга (в разработке)")
            
        else:  # full
            # Полный цикл
            features = run_data_pipeline(config, logger)
            model, test_data = run_training_pipeline(config, features, logger)
            backtest_results = run_backtesting_pipeline(config, model, test_data, logger)
            run_visualization_pipeline(config, backtest_results, logger)
            
            logger.info("\n" + "="*80)
            logger.info("✅ ПОЛНЫЙ ЦИКЛ ЗАВЕРШЕН УСПЕШНО!")
            logger.info("="*80)
            
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    main()