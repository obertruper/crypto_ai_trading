#!/usr/bin/env python3
"""
Демонстрационный запуск системы без ML зависимостей
"""

import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger

def load_config(config_path: str) -> dict:
    """Загрузка конфигурации"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def demo_database_connection(config: dict):
    """Демонстрация подключения к базе данных"""
    logger = get_logger("DatabaseDemo")
    
    logger.info("🔌 Демонстрация подключения к PostgreSQL...")
    
    try:
        import psycopg2
        from sqlalchemy import create_engine
        
        db_config = config['database']
        connection_string = (
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        
        engine = create_engine(connection_string)
        
        # Тест подключения
        with engine.connect() as conn:
            result = conn.execute("SELECT 1 as test")
            test_value = result.scalar()
            
        if test_value == 1:
            logger.info("✅ Подключение к базе данных успешно!")
            
            # Получение статистики
            with engine.connect() as conn:
                symbols_query = """
                SELECT COUNT(DISTINCT symbol) as symbol_count 
                FROM raw_market_data 
                WHERE market_type = 'futures'
                """
                symbol_count = conn.execute(symbols_query).scalar()
                
                records_query = """
                SELECT COUNT(*) as total_records 
                FROM raw_market_data 
                WHERE market_type = 'futures'
                """
                total_records = conn.execute(records_query).scalar()
                
                date_range_query = """
                SELECT MIN(datetime) as min_date, MAX(datetime) as max_date 
                FROM raw_market_data 
                WHERE market_type = 'futures'
                """
                date_result = conn.execute(date_range_query).fetchone()
                
            logger.info(f"📊 Статистика базы данных:")
            logger.info(f"   - Символов: {symbol_count}")
            logger.info(f"   - Записей: {total_records:,}")
            logger.info(f"   - Период: {date_result[0]} - {date_result[1]}")
            
            return True
            
    except ImportError:
        logger.error("❌ Отсутствуют зависимости для PostgreSQL (psycopg2, sqlalchemy)")
        return False
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к базе данных: {e}")
        return False

def demo_feature_engineering():
    """Демонстрация создания признаков"""
    logger = get_logger("FeatureDemo")
    
    logger.info("🛠️ Демонстрация инженерии признаков...")
    
    # Создание синтетических данных
    dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
    np.random.seed(42)
    
    price_data = {
        'datetime': dates,
        'symbol': ['BTCUSDT'] * len(dates),
        'open': 45000 + np.cumsum(np.random.randn(len(dates)) * 10),
        'high': None,
        'low': None,
        'close': None,
        'volume': np.random.exponential(1000, len(dates)),
        'turnover': None
    }
    
    # Генерация OHLC данных
    for i in range(len(dates)):
        base_price = price_data['open'][i]
        daily_volatility = 0.02
        
        high = base_price * (1 + np.random.random() * daily_volatility)
        low = base_price * (1 - np.random.random() * daily_volatility)
        close = low + (high - low) * np.random.random()
        
        price_data['high'] = price_data.get('high', []) or []
        price_data['low'] = price_data.get('low', []) or []
        price_data['close'] = price_data.get('close', []) or []
        price_data['turnover'] = price_data.get('turnover', []) or []
        
        if i == 0:
            price_data['high'] = [high]
            price_data['low'] = [low]
            price_data['close'] = [close]
            price_data['turnover'] = [price_data['volume'][i] * close]
        else:
            price_data['high'].append(high)
            price_data['low'].append(low)
            price_data['close'].append(close)
            price_data['turnover'].append(price_data['volume'][i] * close)
    
    df = pd.DataFrame(price_data)
    
    # Простые технические индикаторы
    logger.info("📈 Создание технических индикаторов...")
    
    # Moving averages
    df['sma_20'] = df['close'].rolling(20).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    
    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility
    df['volatility'] = df['returns'].rolling(20).std()
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Price ratios
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    
    # Temporal features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    logger.info(f"✅ Создано {len(df.columns)} признаков")
    logger.info(f"📊 Размер датасета: {len(df)} записей")
    
    # Статистика признаков
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    logger.info(f"🔢 Числовых признаков: {len(numeric_cols)}")
    
    return df

def demo_risk_management():
    """Демонстрация системы управления рисками"""
    logger = get_logger("RiskDemo")
    
    logger.info("⚖️ Демонстрация управления рисками...")
    
    # Симуляция портфеля
    initial_capital = 100000
    current_capital = initial_capital
    
    # Параметры риска
    risk_per_trade = 0.01  # 1% риска на сделку
    max_positions = 5
    
    # Симуляция сделок
    trades = []
    
    for i in range(10):
        # Случайные параметры сделки
        symbol = np.random.choice(['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
        entry_price = np.random.uniform(30000, 70000)
        confidence = np.random.uniform(0.6, 0.95)
        atr_value = entry_price * 0.02
        
        # Расчет размера позиции
        risk_amount = current_capital * risk_per_trade * confidence
        stop_loss_distance = atr_value * 2
        position_size = risk_amount / stop_loss_distance
        
        # Случайный результат сделки
        win_probability = confidence
        is_win = np.random.random() < win_probability
        
        if is_win:
            pnl = risk_amount * np.random.uniform(2, 5)  # 2-5x риска
            result = "WIN"
        else:
            pnl = -risk_amount
            result = "LOSS"
        
        current_capital += pnl
        
        trade = {
            'id': i + 1,
            'symbol': symbol,
            'entry_price': entry_price,
            'position_size': position_size,
            'risk_amount': risk_amount,
            'confidence': confidence,
            'pnl': pnl,
            'result': result,
            'capital_after': current_capital
        }
        
        trades.append(trade)
        
        logger.info(f"Сделка #{i+1}: {symbol} {result} PnL=${pnl:.2f} Capital=${current_capital:.2f}")
    
    # Статистика
    trades_df = pd.DataFrame(trades)
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] <= 0]
    
    total_return = (current_capital - initial_capital) / initial_capital
    win_rate = len(winning_trades) / len(trades_df)
    profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 else float('inf')
    
    logger.info(f"📈 Результаты симуляции:")
    logger.info(f"   - Общая доходность: {total_return:.2%}")
    logger.info(f"   - Win Rate: {win_rate:.2%}")
    logger.info(f"   - Profit Factor: {profit_factor:.2f}")
    logger.info(f"   - Всего сделок: {len(trades_df)}")
    
    return trades_df

def demo_signal_generation():
    """Демонстрация генерации торговых сигналов"""
    logger = get_logger("SignalDemo")
    
    logger.info("🎯 Демонстрация генерации сигналов...")
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']
    signals = []
    
    for symbol in symbols:
        # Случайные рыночные условия
        current_price = np.random.uniform(0.5, 70000)
        rsi = np.random.uniform(20, 80)
        volume_ratio = np.random.uniform(0.5, 3.0)
        volatility = np.random.uniform(0.01, 0.08)
        
        # Вероятности модели (фиктивные)
        tp_probs = np.random.random(4)  # 4 уровня TP
        signal_strength = np.max(tp_probs)
        
        # Определение направления
        if signal_strength > 0.7 and rsi < 70 and volume_ratio > 1.2:
            side = 'long' if np.random.random() > 0.5 else 'short'
            confidence = signal_strength * 0.8  # Снижение на рыночные условия
            
            # Расчет уровней
            atr_value = current_price * 0.02
            stop_loss = current_price - atr_value * 2 if side == 'long' else current_price + atr_value * 2
            
            take_profits = []
            tp_levels = [1.2, 2.4, 3.5, 5.8]  # %
            for tp_pct in tp_levels:
                if side == 'long':
                    tp_price = current_price * (1 + tp_pct / 100)
                else:
                    tp_price = current_price * (1 - tp_pct / 100)
                take_profits.append(tp_price)
            
            # Обоснование
            reasons = []
            reasons.append(f"Модель предсказывает {side} с силой {signal_strength:.2%}")
            if rsi < 30:
                reasons.append("RSI показывает перепроданность")
            elif rsi > 70:
                reasons.append("RSI показывает перекупленность")
            if volume_ratio > 1.5:
                reasons.append(f"Высокий объем ({volume_ratio:.1f}x)")
            
            signal = {
                'symbol': symbol,
                'side': side,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profits': take_profits,
                'confidence': confidence,
                'signal_strength': signal_strength,
                'reasoning': "; ".join(reasons)
            }
            
            signals.append(signal)
            
            logger.info(f"🎯 Сигнал: {symbol} {side.upper()} @ {current_price:.4f} (conf: {confidence:.2%})")
    
    logger.info(f"✅ Сгенерировано {len(signals)} качественных сигналов из {len(symbols)} символов")
    
    return signals

def main():
    """Главная функция демонстрации"""
    logger = get_logger("Demo")
    
    logger.info("=" * 80)
    logger.info("🚀 Демонстрация Crypto AI Trading System")
    logger.info("=" * 80)
    
    # Загрузка конфигурации
    config = load_config("config/config.yaml")
    
    try:
        # 1. Тест подключения к БД
        db_success = demo_database_connection(config)
        
        # 2. Демонстрация инженерии признаков
        logger.info("\n" + "=" * 50)
        featured_data = demo_feature_engineering()
        
        # 3. Демонстрация управления рисками
        logger.info("\n" + "=" * 50)
        trades_data = demo_risk_management()
        
        # 4. Демонстрация генерации сигналов
        logger.info("\n" + "=" * 50)
        signals_data = demo_signal_generation()
        
        # Сводка
        logger.info("\n" + "=" * 80)
        logger.info("📋 Сводка демонстрации:")
        logger.info(f"   ✅ Подключение к БД: {'Успешно' if db_success else 'Ошибка'}")
        logger.info(f"   ✅ Признаков создано: {len(featured_data.columns) if 'featured_data' in locals() else 0}")
        logger.info(f"   ✅ Сделок симулировано: {len(trades_data) if 'trades_data' in locals() else 0}")
        logger.info(f"   ✅ Сигналов сгенерировано: {len(signals_data) if 'signals_data' in locals() else 0}")
        
        logger.info("\n🎉 Демонстрация завершена успешно!")
        logger.info("💡 Для полного функционала установите ML зависимости: pip install -r requirements.txt")
        
    except Exception as e:
        logger.log_error(e, "demo")
        logger.critical("❌ Критическая ошибка в демонстрации!")
        raise

if __name__ == "__main__":
    main()