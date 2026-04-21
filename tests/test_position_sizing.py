from __future__ import annotations

import pandas as pd
import pytest

from backtesting import BacktestConfig, BacktestEngine
from backtesting.strategy import Strategy


class _SingleEntryStrategy(Strategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index, dtype="int8")
        signals.iloc[1:] = 1
        return signals


def _sample_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
    close = pd.Series([100, 101, 103, 102, 106, 104, 107, 108], index=idx, dtype="float64")
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
        },
        index=idx,
    )


def test_static_usd_position_sizing_uses_fixed_notional() -> None:
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            position_size_mode="static_usd",
            position_size_value=2_000,
        )
    )
    result = engine.run(_sample_df(), _SingleEntryStrategy())
    assert len(result.trades) == 1
    assert result.trades[0].units == pytest.approx(2_000 / 103)


def test_equity_percent_position_sizing_scales_with_equity() -> None:
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            position_size_mode="equity_percent",
            position_size_value=0.5,
        )
    )
    result = engine.run(_sample_df(), _SingleEntryStrategy())
    assert len(result.trades) == 1
    assert result.trades[0].units == pytest.approx(5_000 / 103)


def test_volatility_scaled_position_sizing_applies_scale_bounds() -> None:
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            position_size_mode="volatility_scaled",
            position_size_value=0.5,
            volatility_target_annual=0.10,
            volatility_lookback=3,
            volatility_min_scale=0.5,
            volatility_max_scale=0.5,
        )
    )
    result = engine.run(_sample_df(), _SingleEntryStrategy())
    assert len(result.trades) == 1
    assert result.trades[0].units == pytest.approx(2_500 / 103)
