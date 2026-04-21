from __future__ import annotations

import pandas as pd

from backtesting import BacktestConfig, BacktestEngine
from backtesting.strategy import UTBotStrategy


def test_ut_bot_provides_intrabar_fill_prices_and_engine_uses_them() -> None:
    idx = pd.date_range('2024-01-01', periods=6, freq='1h', tz='UTC')
    data = pd.DataFrame(
        {
            'open': [100.0, 100.0, 99.8, 99.4, 99.1, 98.9],
            'high': [100.2, 100.1, 100.0, 99.6, 99.3, 99.0],
            'low': [99.8, 99.6, 99.2, 98.8, 98.6, 98.4],
            'close': [100.0, 99.7, 99.3, 99.0, 98.8, 98.6],
            'volume': [1000] * 6,
        },
        index=idx,
    )

    strategy = UTBotStrategy(key_value=1.0, atr_period=2)
    engine = BacktestEngine(BacktestConfig(execute_on_signal_bar=True))
    result = engine.run(data, strategy)

    fill_prices = strategy.signal_fill_prices
    assert isinstance(fill_prices, pd.Series)
    assert fill_prices.notna().any()

    if result.trades:
        first_trade = result.trades[0]
        expected_fill = float(fill_prices.loc[first_trade.entry_time])
        assert expected_fill == first_trade.entry_price
