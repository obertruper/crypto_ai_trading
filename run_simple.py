"""
Простой запуск системы криптотрейдинга
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg2
import yaml
from pathlib import Path

# Импорты из проекта
from db_connection import get_db_config, load_data_simple
from data.feature_engineering import FeatureEngineer
from models.patchtst import PatchTST
from trading.risk_manager import RiskManager
from trading.backtester import Backtester
from utils.logger import get_logger

def main():
    """Основная функция запуска"""
    
    logger = get_logger("CryptoAI")
    logger.info("="*80)
    logger.info("🚀 Запуск Crypto AI Trading System")
    logger.info("="*80)
    
    # 1. Загрузка конфигурации
    config_path = Path("config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Загрузка данных
    logger.info("\n📊 Загрузка данных из БД...")
    
    # Берем топ-5 символов для демо
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']
    df = load_data_simple(symbols=symbols, days=90)
    
    logger.info(f"Загружено {len(df):,} записей для {len(symbols)} символов")
    
    # 3. Создание признаков
    logger.info("\n🔧 Создание технических индикаторов...")
    feature_engineer = FeatureEngineer(config)
    
    # Обработка по символам
    features_list = []
    for symbol in symbols:
        symbol_data = df[df['symbol'] == symbol].copy()
        if len(symbol_data) > 100:  # Минимум данных для индикаторов
            features = feature_engineer.create_features(symbol_data)
            features_list.append(features)
            logger.info(f"  {symbol}: {len(features)} записей, {features.shape[1]} признаков")
    
    # Объединяем все признаки
    all_features = pd.concat(features_list, ignore_index=True)
    logger.info(f"\n✅ Создано признаков: {all_features.shape}")
    
    # 4. Разделение на train/test
    train_end = df['datetime'].max() - timedelta(days=30)
    train_data = all_features[all_features['datetime'] <= train_end]
    test_data = all_features[all_features['datetime'] > train_end]
    
    logger.info(f"\n📈 Разделение данных:")
    logger.info(f"  Train: {len(train_data):,} записей до {train_end}")
    logger.info(f"  Test: {len(test_data):,} записей после {train_end}")
    
    # 5. Демо бэктестинга (без обучения модели)
    logger.info("\n💰 Запуск демо бэктестинга...")
    
    risk_manager = RiskManager(config)
    backtester = Backtester(config)
    
    # Простая стратегия на основе RSI
    def simple_rsi_signals(data):
        """Генерация сигналов на основе RSI"""
        signals = []
        
        for _, row in data.iterrows():
            if 'rsi' in row and not pd.isna(row['rsi']):
                # Покупка при RSI < 30
                if row['rsi'] < 30:
                    signals.append({
                        'datetime': row['datetime'],
                        'symbol': row['symbol'],
                        'side': 'long',
                        'price': row['close'],
                        'confidence': 0.7
                    })
                # Продажа при RSI > 70
                elif row['rsi'] > 70:
                    signals.append({
                        'datetime': row['datetime'],
                        'symbol': row['symbol'],
                        'side': 'short',
                        'price': row['close'],
                        'confidence': 0.7
                    })
        
        return signals
    
    # Генерация сигналов
    test_signals = simple_rsi_signals(test_data)
    logger.info(f"  Сгенерировано {len(test_signals)} торговых сигналов")
    
    # 6. Анализ результатов
    if test_signals:
        logger.info("\n📊 Примеры сигналов:")
        for signal in test_signals[:5]:
            logger.info(f"  {signal['datetime']} | {signal['symbol']} | "
                       f"{signal['side']} @ {signal['price']:.2f}")
    
    # 7. Статистика по данным
    logger.info("\n📈 Статистика по символам:")
    for symbol in symbols:
        symbol_data = test_data[test_data['symbol'] == symbol]
        if len(symbol_data) > 0:
            returns = symbol_data['close'].pct_change().dropna()
            logger.info(f"  {symbol}:")
            logger.info(f"    - Записей: {len(symbol_data)}")
            logger.info(f"    - Средняя цена: ${symbol_data['close'].mean():.2f}")
            logger.info(f"    - Волатильность: {returns.std() * 100:.2f}%")
            logger.info(f"    - Доходность за период: {((symbol_data['close'].iloc[-1] / symbol_data['close'].iloc[0]) - 1) * 100:.2f}%")
    
    logger.info("\n✅ Демо завершено успешно!")
    logger.info("="*80)
    
    return all_features

if __name__ == "__main__":
    data = main()