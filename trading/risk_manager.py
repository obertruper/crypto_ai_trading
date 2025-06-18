"""
Система управления рисками для криптотрейдинга
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from utils.logger import get_logger

@dataclass
class Position:
    """Класс для представления торговой позиции"""
    symbol: str
    side: str  # 'long' или 'short'
    entry_price: float
    quantity: float
    stop_loss: float
    take_profits: List[float]
    entry_time: datetime
    confidence: float
    atr_value: float
    
    @property
    def position_value(self) -> float:
        return self.entry_price * self.quantity
    
    @property
    def risk_amount(self) -> float:
        return abs(self.entry_price - self.stop_loss) * self.quantity

@dataclass
class RiskMetrics:
    """Метрики риска портфеля"""
    total_exposure: float
    risk_per_trade: float
    portfolio_risk: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float

class RiskManager:
    """Основной класс управления рисками"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.risk_config = config['risk_management']
        self.logger = get_logger("RiskManager")
        
        # Параметры риска
        self.max_risk_per_trade = self.risk_config['risk_per_trade_pct'] / 100
        self.max_portfolio_risk = self.risk_config.get('max_portfolio_risk_pct', 10) / 100
        self.max_position_size = self.risk_config['position_sizing']['max_position_pct'] / 100
        self.max_concurrent_positions = self.risk_config['max_concurrent_positions']
        
        # Текущие позиции и история
        self.active_positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.equity_curve: List[float] = []
        self.initial_capital = config['backtesting']['initial_capital']
        self.current_capital = self.initial_capital
        
        # Корректировки для разных типов активов
        self.volatility_adjustments = self.risk_config['volatility_adjustment']
        
    def calculate_position_size(self, 
                              symbol: str, 
                              entry_price: float, 
                              stop_loss: float, 
                              atr_value: float,
                              confidence: float = 1.0) -> float:
        """Расчет размера позиции на основе волатильности и риска"""
        
        # Базовый риск на сделку
        base_risk = self.max_risk_per_trade * self.current_capital
        
        # Корректировка на тип актива
        risk_multiplier = self._get_risk_multiplier(symbol)
        adjusted_risk = base_risk * risk_multiplier
        
        # Корректировка на уверенность модели
        confidence_adjusted_risk = adjusted_risk * confidence
        
        # Размер позиции на основе стоп-лосса
        risk_per_unit = abs(entry_price - stop_loss)
        position_size_by_sl = confidence_adjusted_risk / risk_per_unit
        
        # Размер позиции на основе ATR (волатильность)
        atr_multiplier = 2.0  # Стандартный множитель ATR
        position_size_by_atr = confidence_adjusted_risk / (atr_value * atr_multiplier)
        
        # Берем минимальный из двух размеров
        position_size = min(position_size_by_sl, position_size_by_atr)
        
        # Ограничение по максимальному размеру позиции
        max_position_value = self.current_capital * self.max_position_size
        max_position_size = max_position_value / entry_price
        
        position_size = min(position_size, max_position_size)
        
        self.logger.debug(
            f"Расчет позиции {symbol}: "
            f"базовый_риск={base_risk:.2f}, "
            f"скорр_риск={confidence_adjusted_risk:.2f}, "
            f"размер_по_SL={position_size_by_sl:.4f}, "
            f"размер_по_ATR={position_size_by_atr:.4f}, "
            f"финальный_размер={position_size:.4f}"
        )
        
        return position_size
    
    def _get_risk_multiplier(self, symbol: str) -> float:
        """Получение множителя риска для разных типов активов"""
        major_coins = self.volatility_adjustments['major_coins']
        
        if symbol in major_coins:
            return self.volatility_adjustments['major_risk_multiplier']
        elif any(meme in symbol for meme in ['DOGE', 'SHIB', 'PEPE', 'WIF', 'POPCAT', 'PNUT']):
            return self.volatility_adjustments['meme_coin_risk_multiplier']
        else:
            return self.volatility_adjustments['altcoin_risk_multiplier']
    
    def validate_new_position(self, 
                            symbol: str, 
                            position_size: float, 
                            entry_price: float) -> Tuple[bool, str]:
        """Проверка возможности открытия новой позиции"""
        
        # Проверка максимального количества позиций
        if len(self.active_positions) >= self.max_concurrent_positions:
            return False, f"Достигнуто максимальное количество позиций ({self.max_concurrent_positions})"
        
        # Проверка существующей позиции по символу
        if symbol in self.active_positions:
            return False, f"Позиция по {symbol} уже открыта"
        
        # Проверка достаточности капитала
        position_value = position_size * entry_price
        if position_value > self.current_capital * 0.95:  # 95% от капитала
            return False, f"Недостаточно капитала для позиции (требуется {position_value:.2f})"
        
        # Проверка общего риска портфеля
        current_portfolio_risk = self._calculate_portfolio_risk()
        new_position_risk = position_size * entry_price * 0.02  # Примерный риск 2%
        
        if (current_portfolio_risk + new_position_risk) > self.current_capital * self.max_portfolio_risk:
            return False, f"Превышение максимального риска портфеля"
        
        return True, "Позиция одобрена"
    
    def open_position(self, 
                     symbol: str, 
                     side: str, 
                     entry_price: float, 
                     stop_loss: float, 
                     take_profits: List[float],
                     confidence: float,
                     atr_value: float) -> Optional[Position]:
        """Открытие новой позиции"""
        
        # Расчет размера позиции
        position_size = self.calculate_position_size(
            symbol, entry_price, stop_loss, atr_value, confidence
        )
        
        # Валидация позиции
        is_valid, message = self.validate_new_position(symbol, position_size, entry_price)
        
        if not is_valid:
            self.logger.warning(f"Отклонена позиция {symbol}: {message}")
            return None
        
        # Создание позиции
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=position_size,
            stop_loss=stop_loss,
            take_profits=take_profits.copy(),
            entry_time=datetime.now(),
            confidence=confidence,
            atr_value=atr_value
        )
        
        self.active_positions[symbol] = position
        
        self.logger.log_trade(
            symbol=symbol,
            action=f"OPEN_{side.upper()}",
            price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profits[0] if take_profits else 0,
            confidence=confidence
        )
        
        return position
    
    def update_position(self, symbol: str, current_price: float) -> List[Dict]:
        """Обновление позиции и проверка условий закрытия"""
        
        if symbol not in self.active_positions:
            return []
        
        position = self.active_positions[symbol]
        actions = []
        
        # Проверка stop loss
        if self._check_stop_loss(position, current_price):
            actions.append({
                'action': 'close_position',
                'reason': 'stop_loss',
                'price': current_price,
                'quantity': position.quantity
            })
            return actions
        
        # Проверка take profit уровней
        tp_actions = self._check_take_profits(position, current_price)
        actions.extend(tp_actions)
        
        # Обновление trailing stop
        self._update_trailing_stop(position, current_price)
        
        return actions
    
    def _check_stop_loss(self, position: Position, current_price: float) -> bool:
        """Проверка срабатывания stop loss"""
        if position.side == 'long':
            return current_price <= position.stop_loss
        else:
            return current_price >= position.stop_loss
    
    def _check_take_profits(self, position: Position, current_price: float) -> List[Dict]:
        """Проверка срабатывания take profit уровней"""
        actions = []
        
        if not position.take_profits:
            return actions
        
        # Конфигурация частичных закрытий
        tp_percentages = [0.2, 0.3, 0.3, 0.2]  # 20%, 30%, 30%, 20%
        
        for i, tp_price in enumerate(position.take_profits[:]):
            if position.side == 'long' and current_price >= tp_price:
                # Частичное закрытие для long позиции
                close_percentage = tp_percentages[min(i, len(tp_percentages)-1)]
                close_quantity = position.quantity * close_percentage
                
                actions.append({
                    'action': 'partial_close',
                    'reason': f'take_profit_{i+1}',
                    'price': current_price,
                    'quantity': close_quantity,
                    'percentage': close_percentage
                })
                
                # Удаляем сработавший TP
                position.take_profits.remove(tp_price)
                position.quantity -= close_quantity
                
                # Обновляем stop loss к breakeven после первого TP
                if i == 0:
                    position.stop_loss = position.entry_price
                    self.logger.info(f"Stop loss перемещен к breakeven для {position.symbol}")
                
            elif position.side == 'short' and current_price <= tp_price:
                # Аналогично для short позиций
                close_percentage = tp_percentages[min(i, len(tp_percentages)-1)]
                close_quantity = position.quantity * close_percentage
                
                actions.append({
                    'action': 'partial_close',
                    'reason': f'take_profit_{i+1}',
                    'price': current_price,
                    'quantity': close_quantity,
                    'percentage': close_percentage
                })
                
                position.take_profits.remove(tp_price)
                position.quantity -= close_quantity
                
                if i == 0:
                    position.stop_loss = position.entry_price
        
        return actions
    
    def _update_trailing_stop(self, position: Position, current_price: float):
        """Обновление trailing stop loss"""
        if position.side == 'long':
            # Trailing stop для long позиций
            unrealized_profit_pct = (current_price - position.entry_price) / position.entry_price
            
            if unrealized_profit_pct > 0.02:  # Если прибыль > 2%
                new_stop = current_price * 0.99  # Trailing на 1%
                if new_stop > position.stop_loss:
                    old_stop = position.stop_loss
                    position.stop_loss = new_stop
                    self.logger.debug(f"Trailing stop обновлен для {position.symbol}: {old_stop:.4f} -> {new_stop:.4f}")
        
        else:
            # Trailing stop для short позиций
            unrealized_profit_pct = (position.entry_price - current_price) / position.entry_price
            
            if unrealized_profit_pct > 0.02:
                new_stop = current_price * 1.01  # Trailing на 1%
                if new_stop < position.stop_loss:
                    old_stop = position.stop_loss
                    position.stop_loss = new_stop
                    self.logger.debug(f"Trailing stop обновлен для {position.symbol}: {old_stop:.4f} -> {new_stop:.4f}")
    
    def close_position(self, symbol: str, close_price: float, reason: str = "manual"):
        """Полное закрытие позиции"""
        if symbol not in self.active_positions:
            self.logger.warning(f"Попытка закрыть несуществующую позицию {symbol}")
            return
        
        position = self.active_positions[symbol]
        
        # Расчет P&L
        if position.side == 'long':
            pnl = (close_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - close_price) * position.quantity
        
        # Обновление капитала
        self.current_capital += pnl
        self.equity_curve.append(self.current_capital)
        
        # Логирование закрытия
        self.logger.info(
            f"Закрыта позиция {symbol}: "
            f"Entry={position.entry_price:.4f}, "
            f"Exit={close_price:.4f}, "
            f"PnL={pnl:.2f}, "
            f"Reason={reason}"
        )
        
        # Перемещение в закрытые позиции
        self.closed_positions.append(position)
        del self.active_positions[symbol]
    
    def _calculate_portfolio_risk(self) -> float:
        """Расчет текущего риска портфеля"""
        total_risk = 0
        
        for position in self.active_positions.values():
            position_risk = position.risk_amount
            total_risk += position_risk
        
        return total_risk
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Получение метрик риска портфеля"""
        
        if not self.closed_positions and not self.equity_curve:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Общая экспозиция
        total_exposure = sum(pos.position_value for pos in self.active_positions.values())
        
        # Риск на сделку
        current_portfolio_risk = self._calculate_portfolio_risk()
        
        # Максимальная просадка
        max_drawdown = self._calculate_max_drawdown()
        
        # Коэффициент Шарпа
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Win rate
        win_rate = self._calculate_win_rate()
        
        # Profit factor
        profit_factor = self._calculate_profit_factor()
        
        return RiskMetrics(
            total_exposure=total_exposure,
            risk_per_trade=self.max_risk_per_trade * 100,
            portfolio_risk=current_portfolio_risk,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Расчет максимальной просадки"""
        if len(self.equity_curve) < 2:
            return 0
        
        equity_series = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity_series)
        drawdown = (equity_series - peak) / peak
        
        return float(np.min(drawdown))
    
    def _calculate_sharpe_ratio(self) -> float:
        """Расчет коэффициента Шарпа"""
        if len(self.equity_curve) < 2:
            return 0
        
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        
        return float(np.mean(returns) / np.std(returns) * np.sqrt(252))  # Аннуализированный Шарп
    
    def _calculate_win_rate(self) -> float:
        """Расчет win rate"""
        if not self.closed_positions:
            return 0
        
        winning_trades = 0
        for position in self.closed_positions:
            # Примерный расчет - в реальности нужно сохранять PnL
            if hasattr(position, 'pnl') and position.pnl > 0:
                winning_trades += 1
        
        return winning_trades / len(self.closed_positions)
    
    def _calculate_profit_factor(self) -> float:
        """Расчет profit factor"""
        if not self.closed_positions:
            return 0
        
        gross_profit = 0
        gross_loss = 0
        
        for position in self.closed_positions:
            if hasattr(position, 'pnl'):
                if position.pnl > 0:
                    gross_profit += position.pnl
                else:
                    gross_loss += abs(position.pnl)
        
        return gross_profit / gross_loss if gross_loss > 0 else 0
    
    def get_portfolio_summary(self) -> Dict:
        """Получение сводки по портфелю"""
        metrics = self.get_risk_metrics()
        
        return {
            'current_capital': self.current_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'active_positions': len(self.active_positions),
            'total_exposure': metrics.total_exposure,
            'portfolio_risk': metrics.portfolio_risk,
            'max_drawdown': metrics.max_drawdown,
            'sharpe_ratio': metrics.sharpe_ratio,
            'win_rate': metrics.win_rate,
            'profit_factor': metrics.profit_factor,
            'total_trades': len(self.closed_positions)
        }