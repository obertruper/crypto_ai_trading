# Решение проблемы синхронизации прогресс-баров при параллельной обработке

## Проблема
При параллельной обработке нескольких символов криптовалют, прогресс-бары tqdm отображались несинхронно, перекрывая друг друга и создавая визуальный хаос в терминале.

## Решение

### 1. Добавлена блокировка для синхронизации вывода
```python
from multiprocessing import RLock
from tqdm import tqdm

# Устанавливаем глобальную блокировку для tqdm
tqdm.set_lock(RLock())

# Параллельная обработка символов
with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes, 
                                            initializer=tqdm.set_lock, 
                                            initargs=(tqdm.get_lock(),)) as executor:
```

### 2. Добавлен флаг для отключения вложенных прогресс-баров
```python
# В prepare_trading_data.py
future = executor.submit(
    process_func, 
    symbol, 
    raw_data[raw_data['symbol'] == symbol].copy(),
    position=idx * 2,
    disable_progress=True  # Отключаем прогресс-бары внутри процессов
)
```

### 3. Обновлен FeatureEngineer для поддержки флага
```python
class FeatureEngineer:
    def __init__(self, config: Dict):
        # ...
        self.process_position = None
        self.disable_progress = False  # Новый флаг
```

### 4. Логика отключения прогресс-баров в _create_target_variables
```python
# Определяем, нужно ли показывать прогресс-бар
disable_progress = self.disable_progress or self.process_position is not None

pbar = tqdm(df.index, 
           desc=f"Анализ LONG позиций ({df['symbol'].iloc[0]})", 
           unit="строка",
           disable=disable_progress,  # Отключаем при параллельной обработке
           position=position,
           leave=False)
```

### 5. Улучшен главный прогресс-бар
```python
with tqdm(total=len(symbols), desc="🚀 Обработка символов", unit="символ") as pbar:
    for future in concurrent.futures.as_completed(future_to_symbol):
        symbol = future_to_symbol[future]
        try:
            result = future.result()
            if not result.empty:
                featured_dfs.append(result)
                total_records = sum(len(df) for df in featured_dfs)
                pbar.set_postfix({
                    'Текущий': symbol, 
                    'Готово': f"{len(featured_dfs)}/{len(symbols)}",
                    'Записей': f"{total_records:,}"
                })
```

## Результат
- Прогресс-бары больше не перекрываются
- Показывается только главный прогресс-бар с информацией о текущем символе
- Детальные прогресс-бары (LONG/SHORT) отключены при параллельной обработке
- Сохранена возможность показа детальных прогресс-баров при последовательной обработке

## Дополнительные улучшения
1. Добавлена статистика в главный прогресс-бар (количество обработанных записей)
2. Добавлены статусы для пустых результатов и ошибок
3. Оптимизирован вывод информации для лучшей читаемости