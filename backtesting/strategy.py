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


class UTBotStrategy(Strategy):
    """UT Bot ATR trailing-stop strategy."""

    def __init__(self, key_value: float = 1.0, atr_period: int = 10) -> None:
        if key_value <= 0:
            raise ValueError("key_value must be positive")
        if atr_period <= 0:
            raise ValueError("atr_period must be positive")
        self.key_value = float(key_value)
        self.atr_period = int(atr_period)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        high = data["high"].astype("float64")
        low = data["low"].astype("float64")
        close = data["close"].astype("float64")
        prev_close = close.shift(1)

        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = true_range.ewm(alpha=1.0 / float(self.atr_period), adjust=False).mean()
        n_loss = self.key_value * atr

        trailing_stop = pd.Series(index=close.index, dtype="float64")
        signals = pd.Series(0, index=close.index, dtype="int8")
        if len(close) == 0:
            return signals

        trailing_stop.iloc[0] = close.iloc[0] - n_loss.iloc[0]
        pos = 0
        for i in range(1, len(close)):
            src = float(close.iloc[i])
            prev_src = float(close.iloc[i - 1])
            prev_stop = float(trailing_stop.iloc[i - 1])
            loss = float(n_loss.iloc[i])

            if src > prev_stop and prev_src > prev_stop:
                stop = max(prev_stop, src - loss)
            elif src < prev_stop and prev_src < prev_stop:
                stop = min(prev_stop, src + loss)
            elif src > prev_stop:
                stop = src - loss
            else:
                stop = src + loss

            trailing_stop.iloc[i] = stop
            buy = prev_src <= prev_stop and src > stop and src > stop
            sell = prev_src >= prev_stop and src < stop and src < stop
            if buy:
                pos = 1
            elif sell:
                pos = -1
            signals.iloc[i] = pos

        return signals
