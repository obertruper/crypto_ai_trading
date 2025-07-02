"""
Унифицированная версия PatchTST для многозадачного обучения
Решает проблему несоответствия размерностей: модель выдает ровно 36 выходов
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import math

from models.patchtst import (
    PositionalEncoding,
    PatchEmbedding,
    EINOPS_AVAILABLE
)


class RevIN(nn.Module):
    """Reversible Instance Normalization для временных рядов"""
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        """
        x: [Batch, Length, Features]
        mode: 'norm' для нормализации, 'denorm' для денормализации
        """
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = torch.sqrt(x.var(dim=1, keepdim=True) + self.eps)
            x = (x - self.mean) / self.std
            
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / self.affine_weight
                
            x = x * self.std + self.mean
            
        return x


class PatchTSTEncoder(nn.Module):
    """Encoder для PatchTST с остаточными соединениями"""
    
    def __init__(self, 
                 e_layers: int = 3,
                 d_model: int = 256,
                 n_heads: int = 4,
                 d_ff: int = 512,
                 dropout: float = 0.1,
                 activation: str = 'gelu',
                 res_attention: bool = True):
        super().__init__()
        
        self.e_layers = e_layers
        self.d_model = d_model
        self.res_attention = res_attention
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                res_attention=res_attention
            ) for _ in range(e_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch, Length, Features]
        """
        # Encoder
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            
        x = self.norm(x)
        
        return x


class EncoderLayer(nn.Module):
    """Слой энкодера с multi-head attention"""
    
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.1,
                 activation: str = 'gelu',
                 res_attention: bool = True):
        super().__init__()
        
        self.res_attention = res_attention
        
        # Multi-head attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch, Length, Features]
        """
        # Self attention with residual
        attn_out, _ = self.self_attention(x, x, x)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        
        # Feed forward with residual
        ff_out = self.ff(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        
        return x


class UnifiedPatchTSTForTrading(nn.Module):
    """
    Единая модель PatchTST для торговли с 36 выходами
    
    Архитектура:
    1. Общий энкодер для извлечения признаков
    2. Многозадачные головы для разных типов предсказаний
    3. Правильное соответствие с целевыми переменными из датасета
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        model_config = config.get('model', {})
        
        # Базовые параметры
        self.n_features = model_config.get('input_size', 159)
        self.context_window = model_config.get('context_window', 168)
        self.patch_len = model_config.get('patch_len', 16)
        self.stride = model_config.get('stride', 8)
        
        # Параметры трансформера
        self.d_model = model_config.get('d_model', 256)
        self.n_heads = model_config.get('n_heads', 4)
        self.e_layers = model_config.get('e_layers', 3)
        self.d_ff = model_config.get('d_ff', 512)
        self.dropout = model_config.get('dropout', 0.1)
        self.activation = model_config.get('activation', 'gelu')
        
        # ВАЖНО: Фиксируем количество выходов = 36
        self.n_outputs = 36
        
        # Нормализация
        self.revin = RevIN(
            num_features=self.n_features,
            eps=1e-5,
            affine=True
        )
        
        # Патч эмбеддинги
        self.patch_embedding = nn.Conv1d(
            in_channels=self.n_features,
            out_channels=self.d_model,
            kernel_size=self.patch_len,
            stride=self.stride,
            padding=0,
            bias=False
        )
        
        # Позиционное кодирование
        self.n_patches = (self.context_window - self.patch_len) // self.stride + 1
        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            max_len=self.n_patches
        )
        
        # Основной энкодер
        self.encoder = PatchTSTEncoder(
            e_layers=self.e_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            res_attention=True
        )
        
        # Многозадачные выходные слои
        # Группируем выходы по типам для лучшего обучения
        
        # 1. Future returns (регрессия) - 4 выхода
        self.future_returns_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 4)
        )
        
        # 2. Long позиции - 15 выходов (включая time, price, improvement)
        self.long_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 15)
        )
        
        # 3. Short позиции - 15 выходов (включая time, price, improvement)
        self.short_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 15)
        )
        
        # 4. Направление и дополнительные - 2 выхода
        self.direction_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 2)  # best_direction + target_return_1h
        )
        
        # Финальный слой для объединения всех предсказаний
        self.output_projection = nn.Linear(self.d_model, self.d_model)
        
        # Layer normalization
        self.ln = nn.LayerNorm(self.d_model)
        
        # Инициализация весов
        self._init_weights()
        
    def _init_weights(self):
        """Инициализация весов модели с уменьшенной дисперсией"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Используем xavier_uniform с уменьшенным gain для стабильности
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                # Kaiming инициализация с уменьшенным масштабом
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                # Дополнительное масштабирование для стабильности
                with torch.no_grad():
                    module.weight.mul_(0.7)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (batch_size, seq_len, n_features)
            
        Returns:
            output: (batch_size, 36) - все целевые переменные
        """
        batch_size = x.shape[0]
        
        # Нормализация
        x = self.revin(x, 'norm')
        
        # Перестановка для Conv1d: (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)
        
        # Создание патчей
        x = self.patch_embedding(x)  # (B, d_model, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, d_model)
        
        # Позиционное кодирование
        x = self.positional_encoding(x)
        
        # Трансформер энкодер
        x = self.encoder(x)  # (B, n_patches, d_model)
        
        # Глобальное представление (среднее по патчам)
        x_global = x.mean(dim=1)  # (B, d_model)
        
        # Нормализация
        x_global = self.ln(x_global)
        
        # Проекция для лучшего представления
        x_projected = self.output_projection(x_global)
        
        # Многозадачные предсказания
        future_returns = self.future_returns_head(x_projected)  # (B, 4)
        long_outputs = self.long_head(x_projected)  # (B, 15)
        short_outputs = self.short_head(x_projected)  # (B, 15)
        direction_outputs = self.direction_head(x_projected)  # (B, 2)
        
        # Объединяем все выходы в один тензор
        outputs = torch.cat([
            future_returns,    # 0-3: future_return_1 to 4
            long_outputs,      # 4-18: long tp/sl/time/price/improvement
            short_outputs,     # 19-33: short tp/sl/time/price/improvement
            direction_outputs  # 34-35: best_direction, target_return_1h
        ], dim=1)
        
        # КРИТИЧНО: Клиппинг выходов для предотвращения взрыва градиентов
        # Ограничиваем выходы в разумных пределах перед возвратом
        outputs = torch.clamp(outputs, min=-10.0, max=10.0)
        
        # ИСПРАВЛЕНО: НЕ применяем sigmoid здесь для совместимости с BCEWithLogitsLoss
        # Future returns (0-3) - без активации (регрессия)
        # Hit/reached (индексы: 4,5,7,8,10,11,13,14,19,20,22,23,25,26,28,29) - логиты (sigmoid будет в loss)
        # Time (индексы: 6,9,12,15,21,24,27,30) - без активации (регрессия)
        # Price/improvement (индексы: 16,17,18,31,32,33) - без активации (регрессия)
        # best_direction (34) - без активации (будет обработан в loss как категориальная)
        # target_return_1h (35) - без активации (регрессия)
        
        # Возвращаем сырые логиты - sigmoid будет применен в BCEWithLogitsLoss
        
        return outputs
    
    def get_output_names(self) -> List[str]:
        """Возвращает имена всех выходов в правильном порядке (36 переменных)"""
        # Точный порядок из кэшированных данных
        return [
            # Базовые возвраты (0-3)
            'future_return_1', 'future_return_2', 'future_return_3', 'future_return_4',
            # Long позиции (4-18)
            'long_tp1_hit', 'long_tp1_reached', 'long_tp1_time',
            'long_tp2_hit', 'long_tp2_reached', 'long_tp2_time',
            'long_tp3_hit', 'long_tp3_reached', 'long_tp3_time',
            'long_sl_hit', 'long_sl_reached', 'long_sl_time',
            'long_optimal_entry_time', 'long_optimal_entry_price', 'long_optimal_entry_improvement',
            # Short позиции (19-33)
            'short_tp1_hit', 'short_tp1_reached', 'short_tp1_time',
            'short_tp2_hit', 'short_tp2_reached', 'short_tp2_time',
            'short_tp3_hit', 'short_tp3_reached', 'short_tp3_time',
            'short_sl_hit', 'short_sl_reached', 'short_sl_time',
            'short_optimal_entry_time', 'short_optimal_entry_price', 'short_optimal_entry_improvement',
            # Направление и целевая переменная (34-35)
            'best_direction', 'target_return_1h'
        ]


class UnifiedTradingLoss(nn.Module):
    """
    Унифицированная loss функция для 36 выходов
    Комбинирует regression и classification losses
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        loss_config = config.get('loss', {})
        
        # Веса для разных компонентов
        self.future_return_weight = loss_config.get('future_return_weight', 1.0)
        self.tp_weight = loss_config.get('tp_weight', 0.8)
        self.sl_weight = loss_config.get('sl_weight', 1.2)
        self.signal_weight = loss_config.get('signal_weight', 0.6)
        
        # Loss функции (исправлено для mixed precision)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction='none')  # Безопасно для autocast
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Вычисление loss
        
        Args:
            predictions: (batch_size, 36)
            targets: (batch_size, 36)
            
        Returns:
            loss: скаляр
        """
        assert predictions.shape == targets.shape, \
            f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
        
        batch_size = predictions.shape[0]
        losses = []
        
        # 1. Future returns loss (MSE) - индексы 0-3
        future_return_loss = self.mse_loss(
            predictions[:, :4], 
            targets[:, :4]
        ).mean()
        losses.append(future_return_loss * self.future_return_weight)
        
        # 2. Hit/Reached вероятности (BCE с логитами - безопасно для autocast)
        hit_reached_indices = [4,5,7,8,10,11,13,14,19,20,22,23,25,26,28,29]
        hit_reached_loss = self.bce_with_logits_loss(
            predictions[:, hit_reached_indices],
            targets[:, hit_reached_indices]
        ).mean()
        losses.append(hit_reached_loss * self.tp_weight)
        
        # 3. Time предсказания (MSE) - время до событий
        time_indices = [6,9,12,15,21,24,27,30]
        time_loss = self.mse_loss(
            predictions[:, time_indices],
            targets[:, time_indices]
        ).mean()
        losses.append(time_loss * self.signal_weight)
        
        # 4. Price/improvement предсказания (MSE)
        price_indices = [16,17,18,31,32,33]
        price_loss = self.mse_loss(
            predictions[:, price_indices],
            targets[:, price_indices]
        ).mean()
        losses.append(price_loss * self.signal_weight)
        
        # 5. Best direction (MSE) - индекс 34
        # ИСПРАВЛЕНО: используем MSE для простоты, так как значения 0/1/2 можно трактовать как числовые
        direction_loss = self.mse_loss(
            predictions[:, 34:35],
            targets[:, 34:35]
        ).mean()
        losses.append(direction_loss * self.signal_weight)
        
        # 6. Target return 1h (MSE) - индекс 35
        target_return_loss = self.mse_loss(
            predictions[:, 35:36],
            targets[:, 35:36]
        ).mean()
        losses.append(target_return_loss * self.future_return_weight)
        
        # Общий loss
        total_loss = sum(losses) / len(losses)
        
        return total_loss


def create_unified_model(config: Dict) -> UnifiedPatchTSTForTrading:
    """Создание унифицированной модели"""
    return UnifiedPatchTSTForTrading(config)