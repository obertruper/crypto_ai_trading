"""
Реализация PatchTST (Patch Time Series Transformer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
import math
from einops import rearrange, repeat

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
        
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        
        patches = rearrange(patches, 'b n c p -> (b c) n p')
        
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
        
        x = rearrange(x, '(b n) t -> b t n', n=self.n_vars)
        
        return x

class PatchTST(nn.Module):
    """PatchTST модель для многомерного прогнозирования временных рядов"""
    
    def __init__(self,
                 c_in: int,
                 context_window: int,
                 target_window: int,
                 patch_len: int = 16,
                 stride: int = 8,
                 n_layers: int = 3,
                 d_model: int = 128,
                 n_heads: int = 8,
                 d_ff: int = 256,
                 norm: str = 'LayerNorm',
                 attn_dropout: float = 0.0,
                 dropout: float = 0.0,
                 act: str = 'gelu',
                 individual: bool = False,
                 pre_norm: bool = False,
                 **kwargs):
        super().__init__()
        
        self.c_in = c_in
        self.context_window = context_window
        self.target_window = target_window
        self.patch_len = patch_len
        self.stride = stride
        self.individual = individual
        
        self.num_patches = (context_window - patch_len) // stride + 1
        
        self.patch_embedding = PatchEmbedding(
            patch_len=patch_len,
            stride=stride,
            in_channels=c_in,
            embed_dim=d_model,
            norm_layer=nn.LayerNorm if norm == 'LayerNorm' else None
        )
        
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.num_patches)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=act,
            batch_first=True,
            norm_first=pre_norm
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model) if not pre_norm else None
        )
        
        self.head_nf = d_model * self.num_patches
        
        if individual:
            self.heads = nn.ModuleList([
                FlattenHead(1, self.head_nf, target_window, dropout)
                for _ in range(c_in)
            ])
        else:
            self.head = FlattenHead(c_in, self.head_nf, target_window, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, N = x.shape
        
        # Нормализация входных данных
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + 1e-5
        x = (x - x_mean) / x_std
        
        # Создание патчей
        x_patches, num_patches = self.patch_embedding(x)
        
        # Позиционное кодирование
        x_patches = self.pos_encoding(x_patches)
        
        # Transformer encoder
        x_encoded = self.transformer_encoder(x_patches)
        
        # Прогнозирование
        if self.individual:
            x_out = []
            for i in range(self.c_in):
                z = x_encoded[i::self.c_in]
                z = self.heads[i](z)
                x_out.append(z)
            x_out = torch.cat(x_out, dim=-1)
        else:
            x_out = self.head(x_encoded)
        
        # Денормализация
        last_mean = x_mean[:, -1:, :]
        last_std = x_std[:, -1:, :]
        x_out = x_out * last_std + last_mean
        
        return x_out
    
    def configure_optimizers(self, learning_rate: float, weight_decay: float = 0.01):
        """Конфигурация оптимизатора"""
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'bias' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=learning_rate)
        
        return optimizer


class PatchTSTForTrading(PatchTST):
    """Расширенная версия PatchTST для торговых сигналов"""
    
    def __init__(self, 
                 c_in: int,
                 context_window: int,
                 target_window: int,
                 num_tp_levels: int = 3,
                 **kwargs):
        super().__init__(
            c_in=c_in,
            context_window=context_window,
            target_window=target_window,
            **kwargs
        )
        
        self.num_tp_levels = num_tp_levels
        
        hidden_size = self.head_nf // 2
        
        # Головы для take profit уровней
        self.tp_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.head_nf, hidden_size),
                nn.ReLU(),
                nn.Dropout(kwargs.get('dropout', 0.1)),
                nn.Linear(hidden_size, target_window),
                nn.Sigmoid()
            )
            for _ in range(num_tp_levels)
        ])
        
        # Голова для stop loss
        self.sl_head = nn.Sequential(
            nn.Linear(self.head_nf, hidden_size),
            nn.ReLU(),
            nn.Dropout(kwargs.get('dropout', 0.1)),
            nn.Linear(hidden_size, target_window),
            nn.Sigmoid()
        )
        
        # Голова для волатильности
        self.volatility_head = nn.Sequential(
            nn.Linear(self.head_nf, hidden_size),
            nn.ReLU(),
            nn.Dropout(kwargs.get('dropout', 0.1)),
            nn.Linear(hidden_size, target_window),
            nn.Softplus()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, L, N = x.shape
        
        # Нормализация входных данных
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + 1e-5
        x_norm = (x - x_mean) / x_std
        
        # Создание патчей и обработка через трансформер
        x_patches, _ = self.patch_embedding(x_norm)
        x_patches = self.pos_encoding(x_patches)
        x_encoded = self.transformer_encoder(x_patches)
        
        # Базовое прогнозирование цены
        price_pred = super().forward(x)
        
        # Агрегация для торговых сигналов
        x_pooled = x_encoded.view(B, N, -1).mean(dim=1)
        
        # Take profit вероятности
        tp_probs = []
        for tp_head in self.tp_heads:
            tp_prob = tp_head(x_pooled)
            tp_probs.append(tp_prob)
        
        tp_probs = torch.stack(tp_probs, dim=-1)
        
        # Stop loss вероятность
        sl_prob = self.sl_head(x_pooled)
        
        # Прогноз волатильности
        volatility = self.volatility_head(x_pooled)
        
        return {
            'price_pred': price_pred,
            'tp_probs': tp_probs,
            'sl_prob': sl_prob,
            'volatility': volatility
        }


class PatchTSTForPrediction(nn.Module):
    """PatchTST модель для предсказания целевых переменных"""
    
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
                 **kwargs):
        super().__init__()
        
        self.c_in = c_in
        self.c_out = c_out
        self.context_window = context_window
        self.target_window = target_window
        self.patch_len = patch_len
        self.stride = stride
        
        self.num_patches = (context_window - patch_len) // stride + 1
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, N = x.shape
        
        # Нормализация входных данных
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + 1e-5
        x_norm = (x - x_mean) / x_std
        
        # Создание патчей
        x_patches, _ = self.patch_embedding(x_norm)
        
        # Позиционное кодирование
        x_patches = self.pos_encoding(x_patches)
        
        # Transformer encoder
        x_encoded = self.transformer_encoder(x_patches)
        
        # Агрегация по всем признакам
        # x_patches имеет размерность (B*N, num_patches, d_model)
        # Нужно reshape обратно и усреднить по признакам
        x_encoded = x_encoded.view(B, N, self.num_patches, -1)
        x_encoded = x_encoded.mean(dim=1)  # Усреднение по признакам
        
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


def create_patchtst_model(config: Dict) -> PatchTSTForTrading:
    """Создание модели из конфигурации"""
    model_config = config['model']
    
    n_features = model_config.get('input_size', 100)
    
    model = PatchTSTForTrading(
        c_in=n_features,
        context_window=model_config.get('context_window', 168),
        target_window=model_config.get('pred_len', 4),
        patch_len=model_config.get('patch_len', 16),
        stride=model_config.get('stride', 8),
        n_layers=model_config.get('e_layers', 3),
        d_model=model_config.get('d_model', 128),
        n_heads=model_config.get('n_heads', 8),
        d_ff=model_config.get('d_ff', 512),
        dropout=model_config.get('dropout', 0.1),
        act=model_config.get('activation', 'gelu'),
        individual=model_config.get('individual', False),
        num_tp_levels=len(config['risk_management']['take_profit_targets'])
    )
    
    return model