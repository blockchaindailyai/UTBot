from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting.stats import compute_performance_stats


def test_compute_performance_stats_cagr_uses_datetime_span() -> None:
    index = pd.to_datetime(["2024-01-01T00:00:00Z", "2024-12-31T00:00:00Z"])
    equity = pd.Series([100.0, 121.0], index=index, dtype="float64")
    returns = equity.pct_change().fillna(0.0)

    stats = compute_performance_stats(
        equity_curve=equity,
        returns=returns,
        trades=[],
        periods_per_year=252,
    )

    expected_years = (index[-1] - index[0]).total_seconds() / (365.25 * 24 * 60 * 60)
    expected_cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / expected_years) - 1
    assert stats["cagr"] == pytest.approx(expected_cagr)


def test_compute_performance_stats_sharpe_ignores_seed_return() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    equity = pd.Series([100.0, 110.0, 105.0, 120.0], index=index, dtype="float64")
    returns = equity.pct_change().fillna(0.0)

    stats = compute_performance_stats(
        equity_curve=equity,
        returns=returns,
        trades=[],
        periods_per_year=252,
    )

    realized = returns.iloc[1:]
    expected = (realized.mean() / realized.std(ddof=0)) * np.sqrt(252)
    assert stats["sharpe"] == pytest.approx(expected)
