#!/usr/bin/env python3
"""
Скрипт для проверки подключения к базе данных
"""
import psycopg2
from psycopg2 import sql
import yaml
import os
from sqlalchemy import create_engine, text
import pandas as pd

def load_config():
    """Загрузка конфигурации"""
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_psycopg2_connection(config):
    """Тест прямого подключения через psycopg2"""
    print("\n1. Тестирование подключения через psycopg2:")
    try:
        conn = psycopg2.connect(
            host=config['database']['host'],
            port=config['database']['port'],
            database=config['database']['database'],
            user=config['database']['user'],
            password=config['database']['password']
        )
        print("✅ Подключение установлено!")
        
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"   PostgreSQL версия: {version[0]}")
        
        # Проверка таблиц
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        print(f"\n   Найдено таблиц: {len(tables)}")
        for table in tables:
            print(f"   - {table[0]}")
            
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return False

def test_sqlalchemy_connection(config):
    """Тест подключения через SQLAlchemy"""
    print("\n2. Тестирование подключения через SQLAlchemy:")
    try:
        # Формирование connection string
        db_url = f"postgresql://{config['database']['user']}:{config['database']['password']}@{config['database']['host']}:{config['database']['port']}/{config['database']['database']}"
        
        engine = create_engine(db_url, pool_size=5, max_overflow=10)
        
        # Тест подключения
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ Подключение через SQLAlchemy установлено!")
            
        # Проверка данных
        print("\n   Проверка данных в таблицах:")
        for table_name in ['raw_market_data', 'processed_market_data', 'model_metadata']:
            try:
                df = pd.read_sql(f"SELECT COUNT(*) as count FROM {table_name}", engine)
                count = df['count'].iloc[0]
                print(f"   - {table_name}: {count} записей")
            except Exception as e:
                print(f"   - {table_name}: таблица не найдена или ошибка")
                
        return True
        
    except Exception as e:
        print(f"❌ Ошибка подключения через SQLAlchemy: {e}")
        return False

def check_data_sample(config):
    """Проверка примера данных"""
    print("\n3. Проверка образца данных:")
    try:
        db_url = f"postgresql://{config['database']['user']}:{config['database']['password']}@{config['database']['host']}:{config['database']['port']}/{config['database']['database']}"
        engine = create_engine(db_url)
        
        # Проверка raw_market_data
        query = """
        SELECT symbol, COUNT(*) as count, 
               MIN(timestamp) as min_date, 
               MAX(timestamp) as max_date
        FROM raw_market_data
        GROUP BY symbol
        ORDER BY count DESC
        LIMIT 10;
        """
        
        df = pd.read_sql(query, engine)
        if not df.empty:
            print("\n   Топ 10 символов по количеству данных:")
            print(df.to_string(index=False))
        else:
            print("   ⚠️  Таблица raw_market_data пуста")
            
    except Exception as e:
        print(f"   ❌ Ошибка при чтении данных: {e}")

def main():
    """Основная функция"""
    print("=" * 60)
    print("Проверка подключения к базе данных crypto_trading")
    print("=" * 60)
    
    # Загрузка конфигурации
    config = load_config()
    print(f"\nПараметры подключения:")
    print(f"  Host: {config['database']['host']}")
    print(f"  Port: {config['database']['port']}")
    print(f"  Database: {config['database']['database']}")
    print(f"  User: {config['database']['user']}")
    
    # Тестирование подключений
    psycopg2_ok = test_psycopg2_connection(config)
    sqlalchemy_ok = test_sqlalchemy_connection(config)
    
    if psycopg2_ok and sqlalchemy_ok:
        check_data_sample(config)
        print("\n✅ Все проверки пройдены успешно!")
        print("\nРекомендации:")
        print("1. База данных готова к использованию")
        print("2. Если таблицы пусты, запустите загрузку данных:")
        print("   python main.py --mode data")
    else:
        print("\n❌ Обнаружены проблемы с подключением")
        print("\nРекомендации по устранению:")
        print("1. Проверьте, что PostgreSQL запущен на порту 5555")
        print("2. Проверьте правильность пароля в config/config.yaml")
        print("3. Убедитесь, что пользователь 'ruslan' имеет доступ к БД")

if __name__ == "__main__":
    main()