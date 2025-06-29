"""
PyTorch Dataset классы для загрузки и обработки временных рядов
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger

class TimeSeriesDataset(Dataset):
    """Dataset для временных рядов криптовалют"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 context_window: int = 168,
                 prediction_window: int = 4,
                 feature_cols: List[str] = None,
                 target_cols: List[str] = None,
                 stride: int = 1):
        """
        Args:
            data: DataFrame с данными
            context_window: размер входного окна
            prediction_window: размер окна предсказания
            feature_cols: список признаков
            target_cols: список целевых переменных
            stride: шаг между окнами
        """
        self.logger = get_logger("TimeSeriesDataset")
        self.data = data.sort_values(['symbol', 'datetime']).reset_index(drop=True)
        self.context_window = context_window
        self.prediction_window = prediction_window
        self.target_window = prediction_window  # Добавляем target_window для совместимости
        self.stride = stride
        
        # Определение признаков и целевых переменных
        if feature_cols is None:
            self.feature_cols = [col for col in data.columns 
                               if col not in ['id', 'symbol', 'datetime', 'timestamp', 'sector']
                               and not col.startswith(('target_', 'future_', 'optimal_'))]
        else:
            self.feature_cols = feature_cols
            
        if target_cols is None:
            self.target_cols = [col for col in data.columns 
                              if col.startswith(('target_', 'future_return_'))]
        else:
            self.target_cols = target_cols
        
        # Создание индексов для эффективного доступа
        self._create_indices()
        
        self.logger.info(f"Dataset создан: {len(self)} примеров, "
                        f"{len(self.feature_cols)} признаков, "
                        f"{len(self.target_cols)} целевых переменных")
    
    def _create_indices(self):
        """Создание индексов для валидных окон"""
        self.indices = []
        
        # Группировка по символам
        for symbol in self.data['symbol'].unique():
            symbol_data = self.data[self.data['symbol'] == symbol]
            symbol_indices = symbol_data.index.tolist()
            
            # Создание окон с учетом stride
            for i in range(0, len(symbol_indices) - self.context_window - self.prediction_window + 1, self.stride):
                # Проверка непрерывности временного ряда
                window_indices = symbol_indices[i:i + self.context_window + self.prediction_window]
                
                # Проверяем, что все индексы последовательны
                if all(window_indices[j+1] - window_indices[j] == 1 for j in range(len(window_indices)-1)):
                    self.indices.append({
                        'symbol': symbol,
                        'start_idx': window_indices[0],
                        'context_end_idx': window_indices[self.context_window - 1],
                        'target_end_idx': window_indices[-1]
                    })
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Получение одного примера"""
        index_info = self.indices[idx]
        
        # Извлечение контекстных данных
        context_start = index_info['start_idx']
        context_end = index_info['context_end_idx'] + 1
        context_data = self.data.iloc[context_start:context_end]
        
        # ИСПРАВЛЕНО: Целевые переменные берем из последней строки контекста
        # а не из будущих строк!
        
        # Преобразование в тензоры
        X = torch.FloatTensor(context_data[self.feature_cols].values)
        
        # Обработка целевых переменных
        if len(self.target_cols) > 0:
            # Берем целевые значения из ПОСЛЕДНЕЙ строки контекста
            # Это значения future_return_X которые уже содержат будущую доходность
            y_values = context_data.iloc[-1][self.target_cols].values
            
            # Если модель ожидает несколько временных шагов, дублируем значение
            if self.target_window > 1:
                y = torch.FloatTensor([y_values] * self.target_window)
            else:
                y = torch.FloatTensor([y_values])
        else:
            y = torch.FloatTensor([])
        
        # Данные для анализа (оставляем для совместимости)
        target_start = index_info['context_end_idx'] + 1
        target_end = index_info['target_end_idx'] + 1
        target_data = self.data.iloc[target_start:target_end]
        
        # Дополнительная информация
        info = {
            'symbol': index_info['symbol'],
            'context_start_time': str(context_data.iloc[0]['datetime']),
            'context_end_time': str(context_data.iloc[-1]['datetime']),
            'target_start_time': str(target_data.iloc[0]['datetime']) if len(target_data) > 0 else None,
            'target_end_time': str(target_data.iloc[-1]['datetime']) if len(target_data) > 0 else None
        }
        
        return X, y, info


class TradingDataset(TimeSeriesDataset):
    """Специализированный Dataset для торговых сигналов"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 config: Dict = None,
                 context_window: int = 168,
                 prediction_window: int = 4,
                 feature_cols: List[str] = None,
                 target_cols: List[str] = None,
                 include_price_data: bool = True,
                 **kwargs):
        
        # КРИТИЧНОЕ ИСПРАВЛЕНИЕ: Правильная работа с одной целевой переменной
        if target_cols is None and config is not None:
            # Получаем целевую переменную из конфига
            model_config = config.get('model', {})
            target_variable = model_config.get('target_variable', 'future_return_4')
            task_type = model_config.get('task_type', 'regression')
            
            # ИСПРАВЛЕНИЕ: Проверяем что целевая переменная существует в данных
            if target_variable in data.columns:
                trading_targets = [target_variable]
                print(f"✅ Используется целевая переменная: {target_variable} (тип: {task_type})")
            else:
                # Fallback к доступным целевым переменным
                available_targets = [col for col in data.columns 
                                   if col.startswith(('target_', 'future_return_'))]
                
                if available_targets:
                    # Приоритет: future_return_4 > future_return_3 > future_return_2 > future_return_1
                    preferred_targets = ['future_return_4', 'future_return_3', 'future_return_2', 'future_return_1']
                    trading_targets = None
                    for pref_target in preferred_targets:
                        if pref_target in available_targets:
                            trading_targets = [pref_target]
                            break
                    
                    if trading_targets is None:
                        trading_targets = [available_targets[0]]  # Берем первую доступную
                    
                    print(f"⚠️  Целевая переменная {target_variable} не найдена. Используется: {trading_targets[0]}")
                else:
                    # КРИТИЧНОЕ ИСПРАВЛЕНИЕ: Создаем целевую переменную если её нет
                    print("❌ Целевые переменные не найдены в данных! Создаем future_return_4...")
                    
                    # Создаем целевую переменную для каждого символа
                    processed_dfs = []
                    for symbol in data['symbol'].unique():
                        symbol_data = data[data['symbol'] == symbol].copy().sort_values('datetime')
                        
                        # Создаем future_return_4
                        symbol_data['future_close_4'] = symbol_data['close'].shift(-4)
                        symbol_data['future_return_4'] = (
                            (symbol_data['future_close_4'] / symbol_data['close'] - 1) * 100
                        )
                        
                        processed_dfs.append(symbol_data)
                    
                    # Обновляем исходные данные
                    data = pd.concat(processed_dfs, ignore_index=True)
                    trading_targets = ['future_return_4']
                    print(f"✅ Создана целевая переменная: future_return_4")
                    
        elif target_cols is None:
            # Автоматически находим целевые переменные (старое поведение)
            available_targets = [col for col in data.columns 
                               if col.startswith(('target_', 'future_return_'))]
            
            if available_targets:
                # ИСПРАВЛЕНО: Используем только future_return_4 для совместимости с pred_len=4
                if 'future_return_4' in available_targets:
                    trading_targets = ['future_return_4']
                else:
                    # Fallback к любому future_return
                    future_returns = [col for col in available_targets if 'future_return_' in col]
                    trading_targets = future_returns[:1] if future_returns else [available_targets[0]]
            else:
                raise ValueError("❌ Целевые переменные не найдены в данных и config не предоставлен!")
        else:
            trading_targets = target_cols
        
        # ИСПРАВЛЕНИЕ: Проверка размерности целей с конфигурацией модели
        if config is not None:
            expected_output_size = config.get('model', {}).get('output_size', 1)
            if len(trading_targets) != expected_output_size:
                print(f"⚠️  Несоответствие размерностей: target_cols={len(trading_targets)}, output_size={expected_output_size}")
                
                # Автоматическое исправление: берем только нужное количество целей
                if len(trading_targets) > expected_output_size:
                    trading_targets = trading_targets[:expected_output_size]
                    print(f"🔧 Исправлено: используются первые {expected_output_size} целей: {trading_targets}")
        
        # ИСПРАВЛЕНИЕ: Обновляем data если были внесены изменения
        if 'future_return_4' in trading_targets and 'future_return_4' not in data.columns:
            print("⚠️  future_return_4 не найдена, создаем на лету...")
            processed_dfs = []
            for symbol in data['symbol'].unique():
                symbol_data = data[data['symbol'] == symbol].copy().sort_values('datetime')
                symbol_data['future_close_4'] = symbol_data['close'].shift(-4)
                symbol_data['future_return_4'] = (
                    (symbol_data['future_close_4'] / symbol_data['close'] - 1) * 100
                )
                processed_dfs.append(symbol_data)
            data = pd.concat(processed_dfs, ignore_index=True)
        
        # Добавляем target_window если он передан в kwargs
        if 'target_window' in kwargs:
            target_window = kwargs.pop('target_window')
        else:
            target_window = prediction_window
            
        super().__init__(
            data=data,
            context_window=context_window,
            prediction_window=target_window,
            feature_cols=feature_cols,
            target_cols=trading_targets,
            **kwargs
        )
        
        self.include_price_data = include_price_data
        
        # Индексы ценовых данных
        if include_price_data:
            self.price_cols = ['open', 'high', 'low', 'close', 'volume']
    
    def __getitem__(self, idx):
        """Получение примера с торговой информацией"""
        X, y, info = super().__getitem__(idx)
        
        index_info = self.indices[idx]
        
        # Добавляем ценовые данные если нужно
        if self.include_price_data:
            context_start = index_info['start_idx']
            context_end = index_info['context_end_idx'] + 1
            price_data = self.data.iloc[context_start:context_end][self.price_cols]
            info['price_data'] = torch.FloatTensor(price_data.values)
        
        # Добавляем последнюю цену для расчета уровней
        last_idx = index_info['context_end_idx']
        info['last_close'] = self.data.iloc[last_idx]['close']
        info['atr'] = self.data.iloc[last_idx].get('atr', info['last_close'] * 0.02)
        
        return X, y, info


class MultiSymbolDataset(Dataset):
    """Dataset для одновременной работы с несколькими символами"""
    
    def __init__(self,
                 data: pd.DataFrame,
                 symbols: List[str],
                 context_window: int = 168,
                 prediction_window: int = 4,
                 feature_cols: List[str] = None,
                 synchronize: bool = True):
        """
        Args:
            data: DataFrame со всеми данными
            symbols: список символов для батча
            context_window: размер входного окна
            prediction_window: размер окна предсказания
            feature_cols: список признаков
            synchronize: синхронизировать временные метки между символами
        """
        self.logger = get_logger("MultiSymbolDataset")
        self.symbols = symbols
        self.context_window = context_window
        self.prediction_window = prediction_window
        self.synchronize = synchronize
        
        # Фильтрация данных по символам
        self.data = data[data['symbol'].isin(symbols)].copy()
        
        # Определение признаков
        if feature_cols is None:
            self.feature_cols = [col for col in data.columns 
                               if col not in ['id', 'symbol', 'datetime', 'timestamp', 'sector']
                               and not col.startswith(('target_', 'future_', 'optimal_'))]
        else:
            self.feature_cols = feature_cols
        
        # Создание синхронизированных индексов
        if synchronize:
            self._create_synchronized_indices()
        else:
            self._create_independent_indices()
        
        self.logger.info(f"MultiSymbolDataset создан: {len(self)} батчей, "
                        f"{len(symbols)} символов")
    
    def _create_synchronized_indices(self):
        """Создание синхронизированных по времени индексов"""
        self.indices = []
        
        # Находим общие временные метки
        common_times = None
        for symbol in self.symbols:
            symbol_times = set(self.data[self.data['symbol'] == symbol]['datetime'].values)
            if common_times is None:
                common_times = symbol_times
            else:
                common_times = common_times.intersection(symbol_times)
        
        common_times = sorted(list(common_times))
        
        # Создаем окна на основе общих временных меток
        window_size = self.context_window + self.prediction_window
        for i in range(len(common_times) - window_size + 1):
            window_times = common_times[i:i + window_size]
            self.indices.append({
                'start_time': window_times[0],
                'context_end_time': window_times[self.context_window - 1],
                'end_time': window_times[-1]
            })
    
    def _create_independent_indices(self):
        """Создание независимых индексов для каждого символа"""
        # Используем TimeSeriesDataset для каждого символа
        self.symbol_datasets = {}
        
        for symbol in self.symbols:
            symbol_data = self.data[self.data['symbol'] == symbol]
            self.symbol_datasets[symbol] = TimeSeriesDataset(
                data=symbol_data,
                context_window=self.context_window,
                prediction_window=self.prediction_window,
                feature_cols=self.feature_cols
            )
        
        # Индексы - это минимальное количество примеров среди всех символов
        min_length = min(len(ds) for ds in self.symbol_datasets.values())
        self.indices = list(range(min_length))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Получение батча данных для всех символов"""
        batch_X = []
        batch_y = []
        batch_info = []
        
        if self.synchronize:
            # Синхронизированная выборка
            index_info = self.indices[idx]
            
            for symbol in self.symbols:
                # Получаем данные для временного окна
                symbol_data = self.data[
                    (self.data['symbol'] == symbol) &
                    (self.data['datetime'] >= index_info['start_time']) &
                    (self.data['datetime'] <= index_info['end_time'])
                ].sort_values('datetime')
                
                if len(symbol_data) == self.context_window + self.prediction_window:
                    # Разделение на контекст и цель
                    context_data = symbol_data.iloc[:self.context_window]
                    target_data = symbol_data.iloc[self.context_window:]
                    
                    X = torch.FloatTensor(context_data[self.feature_cols].values)
                    y = torch.FloatTensor(target_data[self.feature_cols].values)
                    
                    batch_X.append(X)
                    batch_y.append(y)
                    batch_info.append({
                        'symbol': symbol,
                        'start_time': index_info['start_time'],
                        'end_time': index_info['end_time']
                    })
        else:
            # Независимая выборка
            for symbol in self.symbols:
                X, y, info = self.symbol_datasets[symbol][idx]
                batch_X.append(X)
                batch_y.append(y)
                batch_info.append(info)
        
        # Стекирование в батч
        batch_X = torch.stack(batch_X)
        batch_y = torch.stack(batch_y) if len(batch_y[0]) > 0 else torch.tensor([])
        
        return batch_X, batch_y, batch_info


def create_data_loaders(train_data: pd.DataFrame,
                       val_data: pd.DataFrame,
                       test_data: pd.DataFrame,
                       config: Dict,
                       feature_cols: List[str] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Создание DataLoader'ов для обучения"""
    
    logger = get_logger("DataLoaders")
    
    batch_size = config['model']['batch_size']
    context_window = config['model']['context_window']
    pred_window = config['model']['pred_len']
    num_workers = config['performance']['num_workers']
    persistent_workers = config['performance'].get('persistent_workers', True) if num_workers > 0 else False
    prefetch_factor = config['performance'].get('prefetch_factor', 2)
    
    logger.info("Создание датасетов...")
    
    # Создание датасетов
    train_dataset = TradingDataset(
        data=train_data,
        config=config,
        context_window=context_window,
        prediction_window=pred_window,
        feature_cols=feature_cols,
        stride=1  # Для обучения используем все возможные окна
    )
    
    val_dataset = TradingDataset(
        data=val_data,
        config=config,
        context_window=context_window,
        prediction_window=pred_window,
        feature_cols=feature_cols,
        stride=4  # Для валидации можем использовать больший stride
    )
    
    test_dataset = TradingDataset(
        data=test_data,
        config=config,
        context_window=context_window,
        prediction_window=pred_window,
        feature_cols=feature_cols,
        stride=4
    )
    
    logger.info(f"Размеры датасетов - Train: {len(train_dataset)}, "
               f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Создание DataLoader'ов
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    return train_loader, val_loader, test_loader


def collate_trading_batch(batch):
    """Кастомная функция для объединения батчей с дополнительной информацией"""
    X_list, y_list, info_list = zip(*batch)
    
    # Стекирование тензоров
    X = torch.stack(X_list)
    y = torch.stack(y_list) if len(y_list[0]) > 0 else None
    
    # Объединение информации
    batch_info = {
        'symbols': [info['symbol'] for info in info_list],
        'last_closes': torch.tensor([info['last_close'] for info in info_list]),
        'atrs': torch.tensor([info['atr'] for info in info_list])
    }
    
    if 'price_data' in info_list[0]:
        batch_info['price_data'] = torch.stack([info['price_data'] for info in info_list])
    
    return X, y, batch_info