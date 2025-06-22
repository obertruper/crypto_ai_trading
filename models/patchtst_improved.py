"""
Улучшенная версия PatchTST с feature attention и другими оптимизациями
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class FeatureAttention(nn.Module):
    """Механизм внимания для выбора важных признаков"""
    
    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_features),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # x: (B, L, C)
        # Вычисляем веса внимания для каждого временного шага
        weights = self.attention(x)  # (B, L, C)
        # Применяем веса
        return x * weights


class ImprovedPatchEmbedding(nn.Module):
    """Улучшенное создание патчей с многомасштабным подходом"""
    
    def __init__(self, d_model: int, patch_len: int, stride: int, 
                 n_features: int, dropout: float = 0.0):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        
        # Основная проекция патчей
        self.patch_proj = nn.Linear(patch_len, d_model)
        
        # Дополнительные проекции для разных масштабов
        self.multi_scale = nn.ModuleList([
            nn.Conv1d(n_features, d_model // 4, kernel_size=k, stride=1, padding=k//2)
            for k in [3, 5, 7, 9]
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        
        # Основные патчи
        num_patches = (L - self.patch_len) // self.stride + 1
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # patches: (B, num_patches, C, patch_len)
        
        # Проецируем патчи
        patches = patches.transpose(1, 2).reshape(B * C, num_patches, self.patch_len)
        patch_embed = self.patch_proj(patches)  # (B*C, num_patches, d_model)
        
        # Многомасштабные признаки
        x_conv = x.transpose(1, 2)  # (B, C, L)
        multi_scale_features = []
        for conv in self.multi_scale:
            feat = conv(x_conv)  # (B, d_model//4, L)
            # Интерполируем до размера патчей
            feat = F.interpolate(feat, size=num_patches, mode='linear', align_corners=False)
            feat = feat.transpose(1, 2)  # (B, num_patches, d_model//4)
            multi_scale_features.append(feat)
        
        # Объединяем многомасштабные признаки
        multi_scale = torch.cat(multi_scale_features, dim=-1)  # (B, num_patches, d_model)
        multi_scale = multi_scale.unsqueeze(1).repeat(1, C, 1, 1)  # (B, C, num_patches, d_model)
        multi_scale = multi_scale.reshape(B * C, num_patches, -1)
        
        # Комбинируем с основными патчами
        combined = patch_embed + multi_scale
        
        return self.norm(self.dropout(combined))


class ImprovedPatchTST(nn.Module):
    """Улучшенная версия PatchTST"""
    
    def __init__(self, 
                 c_in: int,
                 c_out: int,
                 context_window: int,
                 target_window: int,
                 patch_len: int = 16,
                 stride: int = 8,
                 d_model: int = 128,
                 n_heads: int = 8,
                 d_ff: int = 512,
                 n_layers: int = 3,
                 dropout: float = 0.1,
                 activation: str = 'gelu',
                 **kwargs):
        super().__init__()
        
        self.c_in = c_in
        self.c_out = c_out
        self.context_window = context_window
        self.target_window = target_window
        self.patch_len = patch_len
        self.stride = stride
        
        # Feature attention перед патчами
        self.feature_attention = FeatureAttention(c_in, d_model, dropout)
        
        # Улучшенное создание патчей
        self.patch_embedding = ImprovedPatchEmbedding(
            d_model, patch_len, stride, c_in, dropout
        )
        
        # Позиционное кодирование (learnable) с правильной инициализацией
        num_patches = (context_window - patch_len) // stride + 1
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))
        # Инициализация с малой дисперсией для стабильности
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Улучшенная проекция выхода с skip connection
        self.head_nf = d_model * num_patches
        self.output_projection = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(self.head_nf, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff // 2),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, c_out * target_window)
        )
        
        # Skip connection для прямого предсказания
        self.skip_projection = nn.Linear(c_in, c_out * target_window)
        
        # Layer для комбинирования основного выхода и skip
        self.combine_layer = nn.Sequential(
            nn.Linear(2 * c_out * target_window, c_out * target_window),
            nn.Tanh()
        )
        
        # Instance normalization для входа
        self.instance_norm = nn.InstanceNorm1d(c_in)
        
    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        
        # Проверка на NaN во входных данных
        if torch.isnan(x).any():
            print(f"⚠️ Обнаружены NaN во входных данных! Shape: {x.shape}")
            # Заменяем NaN на 0
            x = torch.nan_to_num(x, nan=0.0)
        
        # Instance normalization с проверкой
        x_transposed = x.transpose(1, 2)
        # Добавляем малое значение для стабильности нормализации
        x_norm = self.instance_norm(x_transposed + 1e-8).transpose(1, 2)
        
        # Feature attention
        x_attended = self.feature_attention(x_norm)
        
        # Skip connection - используем последнее значение
        skip_input = x[:, -1, :]  # (B, C)
        skip_output = self.skip_projection(skip_input)  # (B, c_out * target_window)
        
        # Создание патчей
        patch_embed = self.patch_embedding(x_attended)  # (B*C, num_patches, d_model)
        
        # Добавляем позиционное кодирование
        num_patches = patch_embed.shape[1]
        patch_embed = patch_embed + self.pos_embedding[:, :num_patches, :]
        
        # Transformer
        transformer_out = self.transformer(patch_embed)  # (B*C, num_patches, d_model)
        
        # Reshape обратно
        transformer_out = transformer_out.reshape(B, C, num_patches, -1)
        
        # Взвешенное усреднение по признакам с использованием внимания
        # Добавляем стабилизацию для softmax
        feature_logits = transformer_out.mean(dim=2).mean(dim=2, keepdim=True)
        # Clamp логиты для предотвращения overflow в softmax
        feature_logits = torch.clamp(feature_logits, min=-10, max=10)
        feature_weights = torch.softmax(feature_logits, dim=1)
        transformer_out = (transformer_out * feature_weights.unsqueeze(2)).sum(dim=1)
        
        # Проекция выхода
        main_output = self.output_projection(transformer_out)  # (B, c_out * target_window)
        
        # Комбинируем с skip connection
        combined = torch.cat([main_output, skip_output], dim=-1)
        output = self.combine_layer(combined)
        
        # Reshape в целевой формат
        output = output.reshape(B, self.target_window, self.c_out)
        
        return output