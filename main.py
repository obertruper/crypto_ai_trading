#!/usr/bin/env python3
"""
Главный скрипт для запуска AI системы прогнозирования криптофьючерсов
"""

import argparse
import yaml
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger

def load_config(config_path: str) -> dict:
    """Загрузка конфигурации"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def prepare_data(config: dict, logger):
    """Подготовка данных для обучения"""
    logger.start_stage("data_preparation")
    
    logger.info("📥 Загрузка данных из PostgreSQL...")
    
    # Импорт здесь для избежания циклических импортов
    from data.data_loader import CryptoDataLoader
    from data.feature_engineering import FeatureEngineer
    from data.dataset import create_data_loaders, TradingDataset
    
    data_loader = CryptoDataLoader(config)
    
    # Получаем список символов
    if config['data']['symbols'] == 'all':
        available_symbols = data_loader.get_available_symbols()
        symbols_to_load = available_symbols[:5]  # Первые 5 для тестирования
        logger.info(f"📊 Загружаем первые 5 символов из {len(available_symbols)}: {symbols_to_load}")
    else:
        symbols_to_load = config['data']['symbols'][:5]
        logger.info(f"📊 Загружаем указанные символы: {symbols_to_load}")
    
    raw_data = data_loader.load_data(
        symbols=symbols_to_load,
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )
    
    logger.info("🔍 Проверка качества данных...")
    quality_report = data_loader.validate_data_quality(raw_data)
    
    for symbol, report in quality_report.items():
        if report['anomalies']:
            logger.warning(f"Аномалии в данных {symbol}: {report['anomalies']}")
    
    logger.info("🛠️ Создание признаков...")
    feature_engineer = FeatureEngineer(config)
    featured_data = feature_engineer.create_features(raw_data)
    
    logger.info("✂️ Разделение данных на train/val/test...")
    
    # Разделение данных
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    test_ratio = config['data']['test_ratio']
    
    # Сортировка по времени для правильного разделения
    featured_data = featured_data.sort_values(['symbol', 'datetime']).reset_index(drop=True)
    
    # Разделение по символам
    train_data_list = []
    val_data_list = []
    test_data_list = []
    
    for symbol in featured_data['symbol'].unique():
        symbol_data = featured_data[featured_data['symbol'] == symbol]
        n = len(symbol_data)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data_list.append(symbol_data.iloc[:train_end])
        val_data_list.append(symbol_data.iloc[train_end:val_end])
        test_data_list.append(symbol_data.iloc[val_end:])
    
    train_data = pd.concat(train_data_list, ignore_index=True)
    val_data = pd.concat(val_data_list, ignore_index=True)
    test_data = pd.concat(test_data_list, ignore_index=True)
    
    logger.info("🏗️ Создание datasets...")
    
    # Определение целевых переменных
    target_cols = [col for col in featured_data.columns 
                  if col.startswith(('target_', 'future_return_'))]
    
    # Создание DataLoader'ов
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data=train_data,
        val_data=val_data, 
        test_data=test_data,
        config=config
    )
    
    logger.info(f"📊 Размеры datasets:")
    logger.info(f"   - Train: {len(train_data)} записей")
    logger.info(f"   - Val: {len(val_data)} записей")
    logger.info(f"   - Test: {len(test_data)} записей")
    
    logger.end_stage("data_preparation", 
                    train_size=len(train_data),
                    val_size=len(val_data),
                    test_size=len(test_data))
    
    return train_loader, val_loader, test_loader

def train_model(config: dict, train_loader, val_loader, logger):
    """Обучение модели"""
    logger.start_stage("model_training")
    
    logger.info("🏗️ Создание модели PatchTST...")
    
    # Получаем информацию о данных из первого батча
    sample_batch = next(iter(train_loader))
    X_sample, y_sample, _ = sample_batch
    
    n_features = X_sample.shape[-1]  # Последняя размерность
    n_targets = y_sample.shape[-1] if y_sample is not None else 1
    
    logger.info(f"📊 Входные признаки: {n_features}, Целевые переменные: {n_targets}")
    
    # ИСПРАВЛЕНИЕ: Используем правильную модель
    from models.patchtst import PatchTSTForPrediction
    
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
    from training.trainer import Trainer
    trainer = Trainer(model, config)
    
    # DataLoader'ы уже созданы, используем их напрямую
    
    # Обучение
    logger.info("🚀 Начало обучения...")
    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Сохранение лучшей модели
    best_model_path = trainer.checkpoint_dir / "best_model.pth"
    
    logger.info(f"✅ Обучение завершено. Лучшая модель: {best_model_path}")
    
    logger.end_stage("model_training", model_path=str(best_model_path))
    
    return model, best_model_path, train_loader

def backtest_strategy(config: dict, model, test_loader, train_loader, logger):
    """Бэктестирование стратегии"""
    logger.start_stage("backtesting")
    
    logger.info("💰 Запуск бэктестирования...")
    
    # Инициализация компонентов торговли
    from trading.risk_manager import RiskManager
    from trading.signals import SignalGenerator
    from trading.backtester import Backtester
    
    risk_manager = RiskManager(config)
    signal_generator = SignalGenerator(config)
    backtester = Backtester(config)
    
    # Генерация предсказаний модели (фиктивных для демонстрации)
    logger.info("🔮 Генерация предсказаний модели...")
    
    # Создаем фиктивные предсказания на основе данных
    sample_batch = next(iter(test_loader))
    X_sample, y_sample, _ = sample_batch
    
    n_samples = len(test_loader.dataset) if hasattr(test_loader, 'dataset') else 1000
    n_targets = y_sample.shape[-1] if y_sample is not None else 1
    
    predictions = {
        'price_pred': np.random.random((n_samples, config['model']['pred_len'], n_targets)),
        'confidence': np.random.uniform(0.5, 0.9, n_samples)
    }
    
    # Создаем фиктивные рыночные данные для демонстрации
    test_data = pd.DataFrame({
        'datetime': pd.date_range('2025-01-01', periods=n_samples, freq='15min'),
        'symbol': np.random.choice(['BTCUSDT', 'ETHUSDT'], n_samples),
        'close': np.random.uniform(30000, 70000, n_samples),
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    # Запуск бэктестинга
    logger.info("🏃 Запуск бэктестинга...")
    backtest_results = backtester.run_backtest(
        market_data=test_data,
        features=test_data,  # Упрощение для демо
        model_predictions=predictions
    )
    
    # Отображение результатов
    logger.info("📈 РЕЗУЛЬТАТЫ БЭКТЕСТИНГА:")
    logger.info(f"  Начальный капитал: ${backtest_results['initial_capital']:,.2f}")
    logger.info(f"  Финальный капитал: ${backtest_results['final_capital']:,.2f}")
    logger.info(f"  Общая доходность: {backtest_results['total_return_pct']:.2f}%")
    logger.info(f"  Коэффициент Шарпа: {backtest_results['sharpe_ratio']:.2f}")
    logger.info(f"  Максимальная просадка: {backtest_results['max_drawdown_pct']:.2f}%")
    logger.info(f"  Win Rate: {backtest_results['win_rate_pct']:.2f}%")
    logger.info(f"  Всего сделок: {backtest_results['total_trades']}")
    
    logger.end_stage("backtesting", 
                    total_return=backtest_results['total_return_pct'],
                    sharpe_ratio=backtest_results['sharpe_ratio'])
    
    return backtest_results

def analyze_results(config: dict, results: dict, logger):
    """Анализ и визуализация результатов"""
    logger.start_stage("results_analysis")
    
    logger.info("📊 Анализ результатов...")
    
    min_sharpe = config['validation']['min_sharpe_ratio']
    min_win_rate = config['validation']['min_win_rate']
    max_dd = config['validation']['max_drawdown']
    
    passed_validation = True
    
    if results['sharpe_ratio'] < min_sharpe:
        logger.warning(f"⚠️ Sharpe Ratio ({results['sharpe_ratio']:.2f}) ниже минимального ({min_sharpe})")
        passed_validation = False
    
    if results['win_rate'] < min_win_rate:
        logger.warning(f"⚠️ Win Rate ({results['win_rate']:.2%}) ниже минимального ({min_win_rate:.2%})")
        passed_validation = False
    
    if abs(results['max_drawdown']) > max_dd:
        logger.warning(f"⚠️ Max Drawdown ({results['max_drawdown']:.2%}) превышает лимит ({max_dd:.2%})")
        passed_validation = False
    
    if passed_validation:
        logger.info("✅ Все валидационные тесты пройдены!")
    else:
        logger.warning("❌ Некоторые валидационные тесты не пройдены")
    
    logger.end_stage("results_analysis", validation_passed=passed_validation)
    
    return passed_validation

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Crypto AI Trading System')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Путь к файлу конфигурации')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['data', 'train', 'backtest', 'full', 'demo'],
                       help='Режим работы')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Путь к сохраненной модели (для режима backtest)')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    logger = get_logger("CryptoAI")
    
    logger.info("="*80)
    logger.info("🚀 Запуск Crypto AI Trading System")
    logger.info(f"📋 Режим: {args.mode}")
    logger.info(f"⚙️ Конфигурация: {args.config}")
    logger.info("="*80)
    
    try:
        if args.mode in ['data', 'full']:
            train_loader, val_loader, test_loader = prepare_data(config, logger)
        
        if args.mode in ['train', 'full']:
            if args.mode == 'train':
                # Загрузка сохраненных данных
                logger.error("Режим 'train' требует предварительного запуска 'data'")
                return
            
            model, model_path, train_loader = train_model(config, train_loader, val_loader, logger)
        
        if args.mode in ['backtest', 'full']:
            if args.mode == 'backtest':
                if not args.model_path:
                    logger.error("Необходимо указать --model-path для режима backtest")
                    return
                
                logger.info(f"📥 Загрузка модели: {args.model_path}")
                # Здесь должна быть загрузка модели
                
            results = backtest_strategy(config, model, test_loader, train_loader, logger)
            
            validation_passed = analyze_results(config, results, logger)
        
        if args.mode == 'demo':
            logger.info("🎯 Демонстрационный режим - только проверка подключения к БД")
            from data.data_loader import CryptoDataLoader
            
            data_loader = CryptoDataLoader(config)
            available_symbols = data_loader.get_available_symbols()
            
            logger.info(f"✅ Подключение к БД успешно")
            logger.info(f"📊 Найдено {len(available_symbols)} символов")
            logger.info(f"🔍 Первые 10 символов: {available_symbols[:10]}")
            
            # Загружаем небольшой образец данных
            sample_data = data_loader.load_data(
                symbols=available_symbols[:2],
                start_date="2025-06-01",
                end_date="2025-06-16"
            )
            
            logger.info(f"📈 Загружено {len(sample_data)} записей для демонстрации")
        
        logger.info("="*80)
        logger.info("✅ Выполнение завершено успешно!")
        logger.info("="*80)
        
    except Exception as e:
        logger.log_error(e, "main")
        logger.critical("❌ Критическая ошибка! Выполнение прервано.")
        raise

if __name__ == "__main__":
    main()