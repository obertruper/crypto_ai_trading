import pandas as pd
import numpy as np

def safe_divide_fixed(numerator: pd.Series, denominator: pd.Series, fill_value=0.0) -> pd.Series:
    """Исправленная версия safe_divide"""
    # Минимальное значение для знаменателя
    min_denominator = 1e-10
    
    # Создаем безопасный знаменатель
    safe_denominator = denominator.copy()
    
    # Заменяем нули и очень маленькие значения
    mask_small = (safe_denominator.abs() < min_denominator)
    safe_denominator[mask_small] = np.sign(safe_denominator[mask_small]) * min_denominator
    safe_denominator[safe_denominator == 0] = min_denominator  # Для точных нулей
    
    # Выполняем деление
    result = numerator / safe_denominator
    
    # Обрабатываем inf и nan
    # Если fill_value - это Series, используем другой подход
    if isinstance(fill_value, pd.Series):
        # Находим позиции с inf и заменяем их соответствующими значениями из fill_value
        inf_mask = np.isinf(result)
        result.loc[inf_mask] = fill_value.loc[inf_mask]
    else:
        # Если fill_value - скаляр, используем стандартный replace
        result = result.replace([np.inf, -np.inf], fill_value)
    
    result = result.fillna(fill_value)
    
    return result

# Тест
print("🧪 Тестирование исправленного метода safe_divide...")

# Создаем тестовые данные
turnover = pd.Series([100, 200, 0, 500, 1000])  # numerator
volume = pd.Series([10, 0, 20, 0.00001, 100])   # denominator (с проблемными значениями)
close_prices = pd.Series([50, 45, 60, 55, 70])  # fill_value как Series

print(f"Turnover: {turnover.tolist()}")
print(f"Volume: {volume.tolist()}")
print(f"Close prices: {close_prices.tolist()}")

try:
    # Тестируем с Series как fill_value (это вызывало ошибку)
    result = safe_divide_fixed(turnover, volume, fill_value=close_prices)
    print(f"✅ Результат с Series fill_value: {result.tolist()}")
    
    # Тестируем со скалярным fill_value
    result_scalar = safe_divide_fixed(turnover, volume, fill_value=1.0)
    print(f"✅ Результат со скалярным fill_value: {result_scalar.tolist()}")
    
    print("🎉 Тест пройден успешно! Ошибка исправлена.")
    
except Exception as e:
    print(f"❌ Ошибка в тесте: {e}")
    import traceback
    traceback.print_exc()
