from __future__ import annotations

import pandas as pd

from backtesting import BacktestConfig, BacktestEngine
from backtesting.strategy import Strategy


class FixedFillStrategy(Strategy):
    def __init__(self, signals: pd.Series, fills: pd.Series) -> None:
        self._signals = signals
        self._fills = fills
        self.signal_fill_prices: pd.Series | None = None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        self.signal_fill_prices = self._fills.reindex(data.index)
        return self._signals.reindex(data.index).fillna(0).astype('int8')


def test_engine_uses_strategy_signal_fill_prices_for_execution() -> None:
    idx = pd.date_range('2024-01-01', periods=6, freq='1h', tz='UTC')
    close = pd.Series([100.0, 101.0, 102.0, 101.0, 100.0, 99.0], index=idx)
    data = pd.DataFrame(
        {
            'open': close,
            'high': close + 0.5,
            'low': close - 0.5,
            'close': close,
            'volume': 1000,
        },
        index=idx,
    )
    signals = pd.Series([0, 1, 1, -1, -1, 0], index=idx, dtype='int8')
    fills = pd.Series([float('nan'), 100.25, float('nan'), 101.75, float('nan'), float('nan')], index=idx, dtype='float64')

    strategy = FixedFillStrategy(signals, fills)
    result = BacktestEngine(BacktestConfig(execute_on_signal_bar=True)).run(data, strategy)

    assert len(result.trades) >= 1
    first_trade = result.trades[0]
    assert first_trade.entry_price == 100.25
    assert first_trade.exit_price == 101.75
