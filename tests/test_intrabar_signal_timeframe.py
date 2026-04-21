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
