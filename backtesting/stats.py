from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def infer_periods_per_year(index: pd.Index, default: int = 252) -> int:
    """Infer annualization factor from a DatetimeIndex.

    Examples:
    - Daily bars -> ~365
    - 1h bars -> ~8760
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return default

    deltas = index.to_series().diff().dropna().dt.total_seconds()
    if deltas.empty:
        return default

    median_seconds = float(deltas.median())
    if median_seconds <= 0:
        return default

    seconds_per_year = 365.25 * 24 * 60 * 60
    return max(1, int(round(seconds_per_year / median_seconds)))


def compute_performance_stats(
    equity_curve: pd.Series,
    returns: pd.Series,
    trades: Iterable,
    periods_per_year: int = 252,
    positions: pd.Series | None = None,
) -> dict[str, float]:
    trades = list(trades)
    total_return = _safe_div(equity_curve.iloc[-1], equity_curve.iloc[0]) - 1

    n_periods = len(returns)
    years = n_periods / periods_per_year if periods_per_year else 0
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0

    vol = returns.std(ddof=0) * np.sqrt(periods_per_year)
    downside = returns[returns < 0].std(ddof=0) * np.sqrt(periods_per_year)

    sharpe = _safe_div(returns.mean() * periods_per_year, returns.std(ddof=0) * np.sqrt(periods_per_year))
    sortino = _safe_div(returns.mean() * periods_per_year, downside)

    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve / rolling_max) - 1
    max_drawdown = drawdown.min()

    pnl_list = [float(t.pnl) for t in trades]
    wins = [p for p in pnl_list if p > 0]
    losses = [p for p in pnl_list if p < 0]

    total_trades = len(trades)
    win_rate = _safe_div(len(wins), total_trades)
    profit_factor = _safe_div(sum(wins), abs(sum(losses)))
    avg_pnl = float(np.mean(pnl_list)) if pnl_list else 0.0
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    avg_holding_bars = float(np.mean([t.holding_bars for t in trades])) if trades else 0.0
    if positions is not None and len(positions):
        aligned_positions = positions.reindex(equity_curve.index).fillna(0.0)
        exposure = float((aligned_positions != 0).sum() / len(aligned_positions))
    else:
        exposure = float((returns != 0).sum() / len(returns)) if len(returns) else 0.0

    return {
        "periods_per_year": float(periods_per_year),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "volatility": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_drawdown),
        "calmar": float(_safe_div(cagr, abs(max_drawdown))),
        "total_trades": float(total_trades),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "avg_trade_pnl": float(avg_pnl),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "expectancy": float(expectancy),
        "avg_holding_bars": float(avg_holding_bars),
        "exposure": float(exposure),
    }
