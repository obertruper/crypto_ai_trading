"""
Реализация PatchTST (Patch Time Series Transformer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
import math

# Обработка импорта einops
try:
    from einops import rearrange, repeat
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False
    print("Warning: einops not available. Using manual tensor operations.")


# Функции для ручной перестановки размерностей (fallback для einops)
def manual_rearrange_b_n_c_p_to_bc_n_p(tensor):
    """Ручная реализация rearrange('b n c p -> (b c) n p')"""
    b, n, c, p = tensor.shape
    # (B, N, C, P) -> (B, C, N, P) -> (B*C, N, P)
    return tensor.permute(0, 2, 1, 3).contiguous().view(b * c, n, p)


def manual_rearrange_bc_t_to_b_t_c(tensor, n_vars):
    """Ручная реализация rearrange('(b c) t -> b t c', c=n_vars)"""
    bc, t = tensor.shape
    b = bc // n_vars
    c = n_vars
    # (B*C, T) -> (B, C, T) -> (B, T, C)
    return tensor.view(b, c, t).transpose(1, 2)


def manual_repeat_t_n(tensor, n):
    """Ручная реализация repeat('t -> t n', n=n)"""
    # tensor shape: (T,) -> (T, N)
    return tensor.unsqueeze(1).expand(-1, n)

class PositionalEncoding(nn.Module):
    """Позиционное кодирование для трансформера"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class PatchEmbedding(nn.Module):
    """Преобразование временного ряда в патчи"""
    
    def __init__(self, 
                 patch_len: int,
                 stride: int,
                 in_channels: int,
                 embed_dim: int,
                 norm_layer: Optional[nn.Module] = None):
        super().__init__()
        
        self.patch_len = patch_len
        self.stride = stride
        self.in_channels = in_channels
        
        self.proj = nn.Linear(patch_len, embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x):
        B, L, C = x.shape
        
        num_patches = (L - self.patch_len) // self.stride + 1
        
        # Создание патчей
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # patches shape: (B, num_patches, C, patch_len)
        
        # Перестановка размерностей
        if EINOPS_AVAILABLE:
            patches = rearrange(patches, 'b n c p -> (b c) n p')
        else:
            # Ручная перестановка: (B, num_patches, C, patch_len) -> (B*C, num_patches, patch_len)
            patches = manual_rearrange_b_n_c_p_to_bc_n_p(patches)
        
        patches = self.proj(patches)
        patches = self.norm(patches)
        
        return patches, num_patches

class FlattenHead(nn.Module):
    """Голова для прогнозирования"""
    
    def __init__(self,
                 n_vars: int,
                 nf: int,
                 target_window: int,
                 head_dropout: float = 0.0):
        super().__init__()
        
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        
        if EINOPS_AVAILABLE:
            x = rearrange(x, '(b n) t -> b t n', n=self.n_vars)
        else:
            # Ручная перестановка: (B*n_vars, target_window) -> (B, target_window, n_vars)
            x = manual_rearrange_bc_t_to_b_t_c(x, self.n_vars)
        
        return x


class PatchTSTForPrediction(nn.Module):
    """PatchTST модель для предсказания целевых переменных с защитой от переобучения"""
    
    def __init__(self,
                 c_in: int,  # количество входных признаков
                 c_out: int,  # количество выходных переменных
                 context_window: int,
                 target_window: int,
                 patch_len: int = 16,
                 stride: int = 8,
                 n_layers: int = 3,
                 d_model: int = 128,
                 n_heads: int = 8,
                 d_ff: int = 256,
                 dropout: float = 0.0,
                 layer_dropout: float = 0.0,  # Stochastic depth
                 weight_noise: float = 0.0,  # Noise regularization
                 **kwargs):
        super().__init__()
        
        self.c_in = c_in
        self.c_out = c_out
        self.context_window = context_window
        self.target_window = target_window
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.layer_dropout = layer_dropout
        self.weight_noise = weight_noise
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Фиксированное вычисление num_patches с валидацией
        self.num_patches = self._calculate_num_patches(context_window, patch_len, stride)
        
        if self.num_patches <= 0:
            raise ValueError(
                f"Некорректные параметры патчей: context_window={context_window}, "
                f"patch_len={patch_len}, stride={stride} -> num_patches={self.num_patches}"
            )
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            patch_len=patch_len,
            stride=stride,
            in_channels=c_in,
            embed_dim=d_model,
            norm_layer=nn.LayerNorm
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.num_patches)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Output projection
        self.head_nf = d_model * self.num_patches
        self.output_projection = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.head_nf, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, c_out * target_window)
        )
        
        # Инициализация весов
        self._init_weights()
    
    def _calculate_num_patches(self, context_window: int, patch_len: int, stride: int) -> int:
        """ИСПРАВЛЕННОЕ вычисление количества патчей с валидацией"""
        if context_window < patch_len:
            raise ValueError(f"context_window ({context_window}) должен быть >= patch_len ({patch_len})")
        
        return max(1, (context_window - patch_len) // stride + 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, N = x.shape
        
        # Проверка входных размерностей
        if L != self.context_window:
            raise ValueError(f"Expected context_window={self.context_window}, got L={L}")
        if N != self.c_in:
            raise ValueError(f"Expected c_in={self.c_in}, got N={N}")
        
        # Нормализация входных данных
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + 1e-5
        x_norm = (x - x_mean) / x_std
        
        # Создание патчей
        x_patches, actual_num_patches = self.patch_embedding(x_norm)
        # x_patches: (B*N, actual_num_patches, d_model)
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Строгая проверка без изменения архитектуры
        if actual_num_patches != self.num_patches:
            raise ValueError(
                f"Несоответствие количества патчей: ожидалось {self.num_patches}, "
                f"получено {actual_num_patches}. Проверьте параметры модели."
            )
        
        # Позиционное кодирование
        x_patches = self.pos_encoding(x_patches)
        
        # Transformer encoder
        x_encoded = self.transformer_encoder(x_patches)
        
        # Агрегация по всем признакам
        # x_encoded имеет размерность (B*N, num_patches, d_model)
        # Нужно reshape обратно и усреднить по признакам
        d_model = x_encoded.size(-1)
        x_encoded = x_encoded.view(B, N, self.num_patches, d_model)  # (B, N, num_patches, d_model)
        x_encoded = x_encoded.mean(dim=1)  # (B, num_patches, d_model) - усреднение по признакам
        
        # Прогнозирование
        output = self.output_projection(x_encoded)
        
        # Reshape to (B, target_window, c_out)
        output = output.view(B, self.target_window, self.c_out)
        
        return output
    
    def _init_weights(self):
        """Инициализация весов модели"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def create_patchtst_model(config: Dict) -> PatchTSTForPrediction:
    """Создание модели из конфигурации с валидацией"""
    from utils.config_validator import ModelConfig
    
    # Валидация конфигурации модели
    if isinstance(config, dict):
        model_config = config.get('model', {})
        try:
            # Валидация через Pydantic
            validated_config = ModelConfig(**model_config)
            model_config = validated_config.dict()
        except Exception as e:
            raise ValueError(f"Неверная конфигурация модели: {e}")
    else:
        model_config = config['model']
    
    n_features = model_config.get('input_size', 100)
    n_targets = model_config.get('output_size', 1)  # количество целевых переменных
    
    model = PatchTSTForPrediction(
        c_in=n_features,
        c_out=n_targets,
        context_window=model_config.get('context_window', 168),
        target_window=model_config.get('pred_len', 4),
        patch_len=model_config.get('patch_len', 16),
        stride=model_config.get('stride', 8),
        n_layers=model_config.get('e_layers', 3),
        d_model=model_config.get('d_model', 128),
        n_heads=model_config.get('n_heads', 8),
        d_ff=model_config.get('d_ff', 512),
        dropout=model_config.get('dropout', 0.1)
    )
    
    return model