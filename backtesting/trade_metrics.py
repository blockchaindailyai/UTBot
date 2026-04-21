from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import ExecutionEvent


def estimate_total_slippage_paid(execution_events: list[ExecutionEvent], slippage_rate: float) -> float:
    if slippage_rate <= 0:
        return 0.0
    total_notional = 0.0
    for event in execution_events:
        units = abs(float(event.units))
        if units <= 0:
            continue
        total_notional += abs(float(event.price)) * units
    return float(total_notional * slippage_rate)


def compute_trade_diagnostics(
    trades_df: pd.DataFrame,
    initial_capital: float,
    total_fees_paid: float,
    execution_events: list[ExecutionEvent],
    slippage_rate: float,
) -> dict[str, float]:
    execution_notionals: list[float] = []
    for event in execution_events:
        units = abs(float(event.units))
        if units <= 0:
            continue
        execution_notionals.append(abs(float(event.price)) * units)
    total_volume = float(sum(execution_notionals))
    inferred_fee_rate = float(total_fees_paid / total_volume) if total_volume > 0 else 0.0

    trade_count = float(len(trades_df))
    if trades_df.empty:
        total_slippage = estimate_total_slippage_paid(execution_events, slippage_rate)
        return {
            "trade_count": 0.0,
            "mean_position_size_usd": 0.0,
            "median_position_size_usd": 0.0,
            "mean_trade_pnl_usd": 0.0,
            "median_trade_pnl_usd": 0.0,
            "mean_trade_pnl_pct": 0.0,
            "median_trade_pnl_pct": 0.0,
            "total_cumulative_volume": float(total_volume),
            "total_cumulative_fees": float(total_fees_paid),
            "total_cumulative_slippage": float(total_slippage),
            "mean_volume_per_trade": 0.0,
            "median_volume_per_trade": 0.0,
            "mean_fee_per_trade": 0.0,
            "median_fee_per_trade": 0.0,
            "mean_slippage_per_trade": 0.0,
            "median_slippage_per_trade": 0.0,
            "total_slippage_paid": float(total_slippage),
            "fees_per_trade": 0.0,
            "slippage_per_trade": 0.0,
        }

    position_sizes = (trades_df["entry_price"].astype(float).abs() * trades_df["units"].astype(float).abs()).astype("float64")
    pnl_usd = trades_df["pnl"].astype(float)
    if "return_pct" in trades_df.columns:
        pnl_pct = trades_df["return_pct"].astype(float)
    elif initial_capital > 0:
        pnl_pct = pnl_usd / float(initial_capital)
    else:
        pnl_pct = pd.Series(np.zeros(len(trades_df)), index=trades_df.index, dtype="float64")

    per_trade_volume = (
        trades_df["entry_price"].astype(float).abs() * trades_df["units"].astype(float).abs()
        + trades_df["exit_price"].astype(float).abs() * trades_df["units"].astype(float).abs()
    ).astype("float64")
    if {"entry_fee", "exit_fee"}.issubset(trades_df.columns):
        per_trade_fee = (trades_df["entry_fee"].astype(float) + trades_df["exit_fee"].astype(float)).astype("float64")
    else:
        per_trade_fee = (per_trade_volume * inferred_fee_rate).astype("float64")
    per_trade_slippage = (per_trade_volume * float(slippage_rate)).astype("float64")

    total_slippage = estimate_total_slippage_paid(execution_events, slippage_rate)
    fees_per_trade = float(total_fees_paid / trade_count) if trade_count else 0.0
    slippage_per_trade = float(total_slippage / trade_count) if trade_count else 0.0

    return {
        "trade_count": trade_count,
        "mean_position_size_usd": float(position_sizes.mean()),
        "median_position_size_usd": float(position_sizes.median()),
        "mean_trade_pnl_usd": float(pnl_usd.mean()),
        "median_trade_pnl_usd": float(pnl_usd.median()),
        "mean_trade_pnl_pct": float(pnl_pct.mean()),
        "median_trade_pnl_pct": float(pnl_pct.median()),
        "total_cumulative_volume": float(total_volume),
        "total_cumulative_fees": float(total_fees_paid),
        "total_cumulative_slippage": float(total_slippage),
        "mean_volume_per_trade": float(per_trade_volume.mean()),
        "median_volume_per_trade": float(per_trade_volume.median()),
        "mean_fee_per_trade": float(per_trade_fee.mean()),
        "median_fee_per_trade": float(per_trade_fee.median()),
        "mean_slippage_per_trade": float(per_trade_slippage.mean()),
        "median_slippage_per_trade": float(per_trade_slippage.median()),
        "total_slippage_paid": float(total_slippage),
        "fees_per_trade": float(fees_per_trade),
        "slippage_per_trade": float(slippage_per_trade),
    }
