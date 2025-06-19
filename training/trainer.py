"""
Основной модуль для обучения моделей
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Union
from pathlib import Path
import time
from tqdm import tqdm
import json
from datetime import datetime

from utils.logger import get_logger
from models.losses import get_loss_function, MultiTaskLoss
from training.optimizer import get_optimizer, get_scheduler
from utils.metrics import MetricsTracker

class Trainer:
    """Основной класс для обучения моделей"""
    
    def __init__(self,
                 model: nn.Module,
                 config: Dict,
                 device: Optional[torch.device] = None):
        """
        Args:
            model: модель для обучения
            config: конфигурация обучения
            device: устройство для обучения
        """
        self.model = model
        self.config = config
        self.logger = get_logger("Trainer")
        
        # Устройство
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Параметры обучения
        self.epochs = config['model']['epochs']
        self.learning_rate = config['model']['learning_rate']
        self.gradient_clip = config['model'].get('gradient_clip', 1.0)
        self.early_stopping_patience = config['model'].get('early_stopping_patience', 10)
        
        # Mixed precision training
        self.use_amp = config['performance'].get('mixed_precision', False)
        if self.use_amp:
            self.scaler = GradScaler()
        
        # Оптимизатор и планировщик
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Функции потерь
        self.criterion = self._create_loss_function()
        
        # Метрики
        self.metrics_tracker = MetricsTracker(config)
        
        # Директории для сохранения
        self.checkpoint_dir = Path("models_saved")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # История обучения
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Создание оптимизатора"""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_name = optimizer_config.get('name', 'AdamW')
        
        return get_optimizer(
            optimizer_name,
            self.model.parameters(),
            lr=self.learning_rate,
            **optimizer_config.get('params', {})
        )
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Создание планировщика learning rate"""
        scheduler_config = self.config.get('scheduler', {})
        
        if not scheduler_config:
            return None
        
        scheduler_name = scheduler_config.get('name', 'CosineAnnealingLR')
        return get_scheduler(
            scheduler_name,
            self.optimizer,
            **scheduler_config.get('params', {})
        )
    
    def _create_loss_function(self) -> nn.Module:
        """Создание функции потерь"""
        loss_config = self.config.get('loss', {})
        
        # Многозадачная потеря для торговой модели
        if loss_config.get('multitask', False):
            task_losses = {}
            task_weights = {}
            
            # Потеря для предсказания цены
            task_losses['price'] = get_loss_function(
                loss_config.get('price_loss', 'mse')
            )
            task_weights['price'] = loss_config.get('price_weight', 1.0)
            
            # Потеря для вероятностей TP
            task_losses['tp'] = get_loss_function(
                loss_config.get('tp_loss', 'bce')
            )
            task_weights['tp'] = loss_config.get('tp_weight', 1.0)
            
            # Потеря для вероятности SL
            task_losses['sl'] = get_loss_function(
                loss_config.get('sl_loss', 'bce')
            )
            task_weights['sl'] = loss_config.get('sl_weight', 1.0)
            
            # Потеря для волатильности
            task_losses['volatility'] = get_loss_function(
                loss_config.get('volatility_loss', 'mse')
            )
            task_weights['volatility'] = loss_config.get('volatility_weight', 0.5)
            
            return MultiTaskLoss(
                task_weights=task_weights,
                uncertainty_weighting=loss_config.get('uncertainty_weighting', False)
            )
        else:
            # Одна функция потерь
            loss_name = loss_config.get('name', 'mse')
            return get_loss_function(loss_name, **loss_config.get('params', {}))
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Обучение одной эпохи"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_metrics = {}
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (inputs, targets, info) in enumerate(progress_bar):
            # Перенос на устройство
            inputs = inputs.to(self.device)
            if isinstance(targets, torch.Tensor):
                targets = targets.to(self.device)
            elif isinstance(targets, dict):
                targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self._compute_loss(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                self.optimizer.step()
            
            # Обновление метрик
            epoch_loss += loss.item()
            batch_metrics = self.metrics_tracker.metrics_calculator.compute_batch_metrics(outputs, targets)
            
            for key, value in batch_metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                epoch_metrics[key] += value
            
            # Обновление progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}'
            })
        
        # Усреднение метрик
        num_batches = len(train_loader)
        epoch_loss /= num_batches
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        epoch_metrics['loss'] = epoch_loss
        
        return epoch_metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Валидация модели"""
        self.model.eval()
        
        val_loss = 0.0
        val_metrics = {}
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, info) in enumerate(val_loader):
                # Перенос на устройство
                inputs = inputs.to(self.device)
                if isinstance(targets, torch.Tensor):
                    targets = targets.to(self.device)
                elif isinstance(targets, dict):
                    targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self._compute_loss(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self._compute_loss(outputs, targets)
                
                # Обновление метрик
                val_loss += loss.item()
                batch_metrics = self.metrics_tracker.metrics_calculator.compute_batch_metrics(outputs, targets)
                
                for key, value in batch_metrics.items():
                    if key not in val_metrics:
                        val_metrics[key] = 0.0
                    val_metrics[key] += value
        
        # Усреднение метрик
        num_batches = len(val_loader)
        val_loss /= num_batches
        val_metrics = {k: v / num_batches for k, v in val_metrics.items()}
        val_metrics['loss'] = val_loss
        
        return val_metrics
    
    def _compute_loss(self, outputs: Union[torch.Tensor, Dict], 
                     targets: Union[torch.Tensor, Dict]) -> torch.Tensor:
        """Вычисление потерь"""
        
        if isinstance(self.criterion, MultiTaskLoss):
            # Многозадачная потеря
            losses = {}
            
            # Обработка выходов и целей
            if isinstance(outputs, dict) and isinstance(targets, dict):
                # Полная многозадачная структура
                if 'price_pred' in outputs and 'future_returns' in targets:
                    losses['price'] = self.criterion.task_losses['price'](
                        outputs['price_pred'], targets['future_returns']
                    )
                
                if 'tp_probs' in outputs and 'tp_targets' in targets:
                    losses['tp'] = self.criterion.task_losses['tp'](
                        outputs['tp_probs'], targets['tp_targets']
                    )
                
                if 'sl_prob' in outputs and 'sl_target' in targets:
                    losses['sl'] = self.criterion.task_losses['sl'](
                        outputs['sl_prob'], targets['sl_target']
                    )
                
                if 'volatility' in outputs and 'volatility_target' in targets:
                    losses['volatility'] = self.criterion.task_losses['volatility'](
                        outputs['volatility'], targets['volatility_target']
                    )
            else:
                # Простая структура - используем базовую потерю для цены
                if isinstance(outputs, dict):
                    outputs = outputs.get('price_pred', outputs.get('prediction', list(outputs.values())[0]))
                if isinstance(targets, dict):
                    targets = targets.get('future_returns', targets.get('target', list(targets.values())[0]))
                
                # Проверка размерностей
                outputs, targets = self._align_dimensions(outputs, targets)
                
                losses['price'] = self.criterion.task_losses['price'](outputs, targets)
            
            return self.criterion(losses) if losses else torch.tensor(0.0, device=outputs.device if torch.is_tensor(outputs) else 'cpu')
        else:
            # Простая потеря
            if isinstance(outputs, dict):
                outputs = outputs.get('price_pred', outputs.get('prediction', list(outputs.values())[0]))
            if isinstance(targets, dict):
                targets = targets.get('future_returns', targets.get('target', list(targets.values())[0]))
            
            # Проверка и выравнивание размерностей
            outputs, targets = self._align_dimensions(outputs, targets)
            
            return self.criterion(outputs, targets)
    
    def _align_dimensions(self, outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Выравнивание размерностей выходов и целей"""
        if outputs.shape != targets.shape:
            # Логирование несоответствия
            self.logger.debug(f"Несоответствие размерностей: outputs {outputs.shape}, targets {targets.shape}")
            
            # Если у outputs есть дополнительное измерение по признакам
            if outputs.dim() == 3 and targets.dim() == 3 and outputs.shape[-1] != targets.shape[-1]:
                # Берем только нужное количество признаков из outputs
                if outputs.shape[-1] > targets.shape[-1]:
                    outputs = outputs[..., :targets.shape[-1]]
                else:
                    # Или наоборот, если targets больше
                    targets = targets[..., :outputs.shape[-1]]
            
            # Если размерности все еще не совпадают
            if outputs.shape != targets.shape:
                # Попытка выравнивания по батчам и временным шагам
                if outputs.dim() == 3 and targets.dim() == 2:
                    # outputs: (batch, time, features), targets: (batch, features)
                    # Берем последний временной шаг из outputs
                    outputs = outputs[:, -1, :]
                elif outputs.dim() == 2 and targets.dim() == 3:
                    # outputs: (batch, features), targets: (batch, time, features)
                    # Берем последний временной шаг из targets
                    targets = targets[:, -1, :]
                elif outputs.dim() == 3 and targets.dim() == 3:
                    # Оба имеют 3 измерения, но разные размеры по времени
                    min_time = min(outputs.shape[1], targets.shape[1])
                    outputs = outputs[:, :min_time, :]
                    targets = targets[:, :min_time, :]
                
                # Финальная попытка - flatten до 2D
                if outputs.shape != targets.shape:
                    if outputs.dim() > 2:
                        outputs = outputs.reshape(outputs.shape[0], -1)
                    if targets.dim() > 2:
                        targets = targets.reshape(targets.shape[0], -1)
                    
                    # Выравнивание по последнему измерению
                    min_size = min(outputs.shape[-1], targets.shape[-1])
                    outputs = outputs[..., :min_size]
                    targets = targets[..., :min_size]
        
        return outputs, targets
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              callbacks: Optional[List[Callable]] = None) -> Dict:
        """Полный цикл обучения"""
        
        self.logger.info(f"Начало обучения на {self.epochs} эпох")
        self.logger.info(f"Устройство: {self.device}")
        self.logger.info(f"Размер батча: {train_loader.batch_size}")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            
            # Обучение
            train_metrics = self.train_epoch(train_loader)
            
            # Валидация
            val_metrics = self.validate(val_loader)
            
            # Обновление learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Сохранение истории
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['learning_rates'].append(current_lr)
            
            # Early stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
                self._save_checkpoint(epoch, val_metrics['loss'], is_best=True)
            else:
                self.patience_counter += 1
            
            # Логирование
            epoch_time = time.time() - epoch_start_time
            self.logger.log_model_metrics(
                epoch=epoch + 1,
                metrics={
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'learning_rate': current_lr,
                    'epoch_time': epoch_time
                }
            )
            
            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self, epoch, train_metrics, val_metrics)
            
            # Early stopping check
            if self.patience_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Восстановление лучшей модели
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        total_time = time.time() - start_time
        self.logger.info(f"Обучение завершено за {total_time:.2f} секунд")
        
        # Финальный отчет
        self._save_training_report()
        
        return self.history
    
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Сохранение checkpoint'а"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'config': self.config,
            'history': self.history
        }
        
        if is_best:
            path = self.checkpoint_dir / f"best_model_{timestamp}.pth"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_{timestamp}.pth"
        
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint сохранен: {path}")
    
    def _save_training_report(self):
        """Сохранение отчета об обучении"""
        report = {
            'config': self.config,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'total_epochs': len(self.history['train_loss']),
            'model_architecture': str(self.model),
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.checkpoint_dir / f"training_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        self.logger.info(f"Отчет об обучении сохранен: {report_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Загрузка checkpoint'а"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        
        self.logger.info(f"Checkpoint загружен: {checkpoint_path}")
        
        return checkpoint['epoch']