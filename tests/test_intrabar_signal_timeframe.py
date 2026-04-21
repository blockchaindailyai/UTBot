from __future__ import annotations

import pandas as pd

from backtesting import BacktestConfig, BacktestEngine
from backtesting.strategy import Strategy


class DailyMomentumStrategy(Strategy):
    """Long when the current (possibly partial) 1D close is above that day's open."""

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(-1, index=data.index, dtype="int8")
        signals[data["close"] > data["open"]] = 1
        return signals


class CountingSignalStrategy(Strategy):
    def __init__(self) -> None:
        self.calls = 0
        self.max_seen_len = 0

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        self.calls += 1
        self.max_seen_len = max(self.max_seen_len, len(data))
        return pd.Series(1, index=data.index, dtype="int8")


def test_signal_timeframe_detects_intraday_flip_for_daily_bar() -> None:
    idx = pd.date_range("2024-01-01", periods=12, freq="4h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 100.0, 99.0, 98.0, 97.0, 96.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            "low": [99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0],
            "close": [99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 98.0, 99.0, 101.0, 103.0, 104.0, 105.0],
            "volume": [1_000] * 12,
        },
        index=idx,
    )

    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000.0,
            signal_timeframe="1D",
            execute_on_signal_bar=True,
        )
    )
    result = engine.run(data, DailyMomentumStrategy())

    assert result.trades, "Expected at least one trade from the intraday signal flip"
    first_long_trade = next(trade for trade in result.trades if trade.side == "long")
    # The second day flips bullish intraday at 08:00, not only after daily close.
    assert first_long_trade.entry_time == pd.Timestamp("2024-01-02T08:00:00Z")


def test_signal_timeframe_works_without_volume_column() -> None:
    idx = pd.date_range("2024-01-01", periods=8, freq="6h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 102.0, 101.0, 100.0, 99.0],
            "high": [101.0, 102.0, 103.0, 104.0, 103.0, 102.0, 101.0, 100.0],
            "low": [99.0, 100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0],
            "close": [101.0, 102.0, 103.0, 102.0, 101.0, 100.0, 99.0, 98.0],
        },
        index=idx,
    )

    engine = BacktestEngine(BacktestConfig(signal_timeframe="1D", execute_on_signal_bar=True))
    result = engine.run(data, DailyMomentumStrategy())

    assert len(result.equity_curve) == len(data)


def test_signal_timeframe_caps_intrabar_evaluations_per_bucket() -> None:
    idx = pd.date_range("2024-01-01", periods=36, freq="2h", tz="UTC")
    close = pd.Series(range(len(idx)), index=idx, dtype="float64") + 100.0
    data = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000.0,
        },
        index=idx,
    )

    strategy = CountingSignalStrategy()
    engine = BacktestEngine(
        BacktestConfig(
            signal_timeframe="1D",
            execute_on_signal_bar=True,
            max_intrabar_evaluations_per_signal_bar=4,
        )
    )
    engine.run(data, strategy)

    # 36 2h-bars = 3 days. With cap=4, strategy evaluations are bounded to 4/day.
    assert strategy.calls == 12


def test_signal_timeframe_history_bars_limits_strategy_input_length() -> None:
    idx = pd.date_range("2024-01-01", periods=72, freq="1h", tz="UTC")
    close = pd.Series(range(len(idx)), index=idx, dtype="float64") + 100.0
    data = pd.DataFrame(
        {
            "open": close - 0.25,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1_000.0,
        },
        index=idx,
    )
    strategy = CountingSignalStrategy()
    engine = BacktestEngine(
        BacktestConfig(
            signal_timeframe="1D",
            execute_on_signal_bar=True,
            max_intrabar_evaluations_per_signal_bar=8,
            signal_timeframe_history_bars=3,
        )
    )

    engine.run(data, strategy)

    assert strategy.max_seen_len <= 3
