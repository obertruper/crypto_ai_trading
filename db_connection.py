"""
Простое подключение к PostgreSQL
"""

import psycopg2
import pandas as pd
import yaml
import os
from pathlib import Path

def get_db_config():
    """Получение конфигурации БД из config.yaml"""
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config['database']
    
    # Простая обработка переменных окружения
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5555')),
        'database': os.getenv('DB_NAME', 'crypto_trading'),
        'user': os.getenv('DB_USER', 'ruslan'),
        'password': os.getenv('DB_PASSWORD', 'ruslan')
    }

def test_connection():
    """Тест подключения к БД"""
    db_config = get_db_config()
    
    try:
        # Прямое подключение через psycopg2
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        print("✅ Подключение к БД успешно!")
        print(f"   База данных: {db_config['database']}")
        print(f"   Хост: {db_config['host']}:{db_config['port']}")
        
        # Проверка таблиц
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = cursor.fetchall()
        
        print("\n📋 Таблицы в БД:")
        for table in tables:
            print(f"   - {table[0]}")
        
        # Статистика по raw_market_data
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT symbol) as symbols,
                MIN(datetime) as min_date,
                MAX(datetime) as max_date
            FROM raw_market_data
        """)
        stats = cursor.fetchone()
        
        print("\n📊 Статистика raw_market_data:")
        print(f"   Всего записей: {stats[0]:,}")
        print(f"   Уникальных символов: {stats[1]}")
        print(f"   Период данных: {stats[2]} - {stats[3]}")
        
        # Примеры символов
        cursor.execute("""
            SELECT DISTINCT symbol 
            FROM raw_market_data 
            ORDER BY symbol 
            LIMIT 10
        """)
        symbols = cursor.fetchall()
        
        print("\n🪙 Примеры символов:")
        for sym in symbols:
            print(f"   - {sym[0]}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return False

def load_data_simple(symbols=None, days=30):
    """Простая загрузка данных"""
    db_config = get_db_config()
    conn = psycopg2.connect(**db_config)
    
    if symbols is None:
        # Берем первые 5 символов
        query = "SELECT DISTINCT symbol FROM raw_market_data ORDER BY symbol LIMIT 5"
        symbols_df = pd.read_sql(query, conn)
        symbols = symbols_df['symbol'].tolist()
    
    # Загружаем данные
    query = f"""
    SELECT * 
    FROM raw_market_data
    WHERE symbol = ANY(ARRAY{symbols})
    AND datetime >= NOW() - INTERVAL '{days} days'
    ORDER BY symbol, datetime
    """
    
    print(f"📥 Загрузка данных для {symbols} за последние {days} дней...")
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"✅ Загружено {len(df):,} записей")
    return df

if __name__ == "__main__":
    test_connection()