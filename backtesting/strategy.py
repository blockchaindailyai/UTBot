from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Minimal strategy contract for the backtesting engine."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return a Series aligned with `data.index` containing -1, 0, or 1."""


def compute_ut_bot_components(
    data: pd.DataFrame,
    key_value: float = 1.0,
    atr_period: int = 10,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute UT Bot trailing stop and directional signals."""
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
    atr = true_range.ewm(alpha=1.0 / float(atr_period), adjust=False).mean()
    n_loss = key_value * atr

    trailing_stop = pd.Series(index=close.index, dtype="float64")
    buy_signal = pd.Series(False, index=close.index, dtype="bool")
    sell_signal = pd.Series(False, index=close.index, dtype="bool")
    position_state = pd.Series(0, index=close.index, dtype="int8")
    if len(close) == 0:
        return trailing_stop, buy_signal, sell_signal, position_state

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
        buy = prev_src <= prev_stop and src > stop
        sell = prev_src >= prev_stop and src < stop
        buy_signal.iloc[i] = buy
        sell_signal.iloc[i] = sell
        if buy:
            pos = 1
        elif sell:
            pos = -1
        position_state.iloc[i] = pos

    return trailing_stop, buy_signal, sell_signal, position_state


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

    def __init__(
        self,
        key_value: float = 1.0,
        atr_period: int = 10,
        ma_filter_enabled: bool = False,
        ma_period: int = 60,
    ) -> None:
        if key_value <= 0:
            raise ValueError("key_value must be positive")
        if atr_period <= 0:
            raise ValueError("atr_period must be positive")
        if ma_period <= 0:
            raise ValueError("ma_period must be positive")
        self.key_value = float(key_value)
        self.atr_period = int(atr_period)
        self.ma_filter_enabled = bool(ma_filter_enabled)
        self.ma_period = int(ma_period)
        self.signal_fill_prices: pd.Series | None = None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        _, buy_signal, sell_signal, position_state = compute_ut_bot_components(
            data=data,
            key_value=self.key_value,
            atr_period=self.atr_period,
        )
        close = data["close"].astype("float64")
        if self.ma_filter_enabled:
            moving_average = close.rolling(self.ma_period).mean()
            buy_signal = buy_signal & (close > moving_average)
            sell_signal = sell_signal & (close < moving_average)

            filtered_position_state = pd.Series(0, index=data.index, dtype="int8")
            pos = 0
            for i in range(len(data.index)):
                if bool(buy_signal.iloc[i]):
                    pos = 1
                elif bool(sell_signal.iloc[i]):
                    pos = -1
                filtered_position_state.iloc[i] = pos
            position_state = filtered_position_state

        fills = pd.Series(float("nan"), index=data.index, dtype="float64")
        signal_rows = buy_signal | sell_signal
        fills.loc[signal_rows] = close.loc[signal_rows].astype("float64")

        self.signal_fill_prices = fills
        return position_state.astype("int8")
