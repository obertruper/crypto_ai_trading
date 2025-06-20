#!/usr/bin/env python3
"""
🚀 АВТОМАТИЧЕСКАЯ УСТАНОВКА И GPU ОБУЧЕНИЕ
=========================================
Версия с автоматической установкой зависимостей
"""

import subprocess
import sys
import os

def install_package(package):
    """Установка пакета если его нет"""
    try:
        __import__(package.split('[')[0].split('=')[0])
        print(f"✅ {package} уже установлен")
        return True
    except ImportError:
        print(f"📦 Устанавливаем {package}...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} успешно установлен")
            return True
        except:
            print(f"❌ Ошибка установки {package}")
            return False

def check_and_install_dependencies():
    """Проверка и установка всех зависимостей"""
    print("🔍 ПРОВЕРКА И УСТАНОВКА ЗАВИСИМОСТЕЙ")
    print("=" * 40)
    
    packages = [
        'numpy',
        'pandas', 
        'scikit-learn',
        'tensorflow==2.15.0',
        'psutil',
        'tqdm'
    ]
    
    all_installed = True
    for package in packages:
        if not install_package(package):
            all_installed = False
    
    return all_installed

def check_gpu():
    """Проверка GPU"""
    try:
        import tensorflow as tf
        print("\n🎯 ПРОВЕРКА GPU")
        print("=" * 20)
        print(f"TensorFlow версия: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        print(f"Найдено GPU: {len(gpus)}")
        
        for i, gpu in enumerate(gpus):
            print(f"GPU {i}: {gpu.name}")
            tf.config.experimental.set_memory_growth(gpu, True)
        
        return len(gpus) > 0
    except Exception as e:
        print(f"❌ Ошибка проверки GPU: {e}")
        return False

def create_simple_cache():
    """Создание простого кеша для тестирования"""
    try:
        import pandas as pd
        import numpy as np
        
        print("\n🏭 СОЗДАНИЕ ТЕСТОВОГО КЕША")
        print("=" * 30)
        
        n_samples = 50000  # Меньше для быстрого тестирования
        np.random.seed(42)
        
        # Простые данные
        timestamps = pd.date_range(start='2020-01-01', periods=n_samples, freq='1H')
        prices = 50000 + np.cumsum(np.random.normal(0, 100, n_samples))
        
        data = {
            'timestamp': timestamps,
            'open': prices * (0.998 + 0.004 * np.random.random(n_samples)),
            'high': prices * (1.001 + 0.002 * np.random.random(n_samples)),
            'low': prices * (0.999 - 0.002 * np.random.random(n_samples)),
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_samples),
            'rsi': 30 + 40 * np.random.random(n_samples),
            'macd': np.random.normal(0, 100, n_samples),
            'bb_upper': prices * 1.02,
            'bb_lower': prices * 0.98,
            'volatility': np.random.uniform(0.01, 0.05, n_samples),
            'close_next_1h': prices * (1 + np.random.normal(0, 0.01, n_samples)),
            'close_next_4h': prices * (1 + np.random.normal(0, 0.02, n_samples)),
            'close_next_24h': prices * (1 + np.random.normal(0, 0.05, n_samples)),
            'close_next_7d': prices * (1 + np.random.normal(0, 0.1, n_samples)),
        }
        
        df = pd.DataFrame(data)
        
        # Сохранение
        cache_file = 'simple_cache.parquet'
        df.to_parquet(cache_file)
        
        cache_size = os.path.getsize(cache_file) / (1024**2)
        print(f"✅ Кеш создан: {cache_size:.1f} MB")
        print(f"📊 Данные: {df.shape[0]:,} записей, {df.shape[1]} признаков")
        
        return df
    
    except Exception as e:
        print(f"❌ Ошибка создания кеша: {e}")
        return None

def simple_gpu_training(df):
    """Простое GPU обучение для тестирования"""
    try:
        import tensorflow as tf
        import numpy as np
        from sklearn.preprocessing import RobustScaler
        from sklearn.model_selection import train_test_split
        
        print("\n🚀 ПРОСТОЕ GPU ОБУЧЕНИЕ")
        print("=" * 25)
        
        # Подготовка данных
        target_cols = ['close_next_1h', 'close_next_4h', 'close_next_24h', 'close_next_7d']
        feature_cols = [col for col in df.columns if col not in target_cols + ['timestamp']]
        
        # Нормализация
        scaler = RobustScaler()
        X_data = scaler.fit_transform(df[feature_cols].values)
        y_data = df[target_cols].values
        
        # Создание простых последовательностей
        seq_length = 24
        X, y = [], []
        
        for i in range(len(X_data) - seq_length):
            X.append(X_data[i:i+seq_length])
            y.append(y_data[i+seq_length])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        print(f"📊 Данные: X={X.shape}, y={y.shape}")
        
        # Разделение
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Простая модель
        with tf.device('/GPU:0'):
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(seq_length, len(feature_cols))),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(32, return_sequences=False),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(len(target_cols))
            ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print(f"🧠 Модель: {model.count_params():,} параметров")
        
        # Обучение
        print("🔥 Начинаем обучение на GPU...")
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=5,  # Быстрый тест
            batch_size=32,
            verbose=1
        )
        
        # Оценка
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"📊 Test Loss: {test_loss:.6f}")
        print(f"📊 Test MAE: {test_mae:.6f}")
        
        print("🎉 ТЕСТОВОЕ GPU ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Главная функция"""
    print("🚀 АВТОМАТИЧЕСКАЯ НАСТРОЙКА GPU СЕРВЕРА")
    print("=" * 50)
    
    # 1. Установка зависимостей
    if not check_and_install_dependencies():
        print("❌ Не удалось установить все зависимости")
        return
    
    # 2. Проверка GPU
    if not check_gpu():
        print("❌ GPU недоступен")
        return
    
    # 3. Создание кеша
    df = create_simple_cache()
    if df is None:
        print("❌ Не удалось создать кеш")
        return
    
    # 4. Тестовое обучение
    if simple_gpu_training(df):
        print("\n🎯 ВСЕ РАБОТАЕТ! СЕРВЕР ГОТОВ К ПОЛНОМУ ОБУЧЕНИЮ!")
    else:
        print("\n❌ Ошибка в тестовом обучении")

if __name__ == "__main__":
    main()
