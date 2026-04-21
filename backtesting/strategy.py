from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Minimal strategy contract for the backtesting engine."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return a Series aligned with `data.index` containing -1, 0, or 1."""


class BuyAndHoldStrategy(Strategy):
    """Long from the first bar through the end of the dataset."""

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index, dtype="int8")
        if not signals.empty:
            signals.iloc[0:] = 1
        return signals


class MovingAverageCrossStrategy(Strategy):
    """Simple moving-average crossover strategy."""

    def __init__(self, fast_period: int = 20, slow_period: int = 50) -> None:
        if fast_period <= 0 or slow_period <= 0:
            raise ValueError("Moving-average periods must be positive")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].astype("float64")
        fast = close.rolling(self.fast_period).mean()
        slow = close.rolling(self.slow_period).mean()
        signals = pd.Series(0, index=data.index, dtype="int8")
        signals[fast > slow] = 1
        signals[fast < slow] = -1
        return signals.fillna(0).astype("int8")
