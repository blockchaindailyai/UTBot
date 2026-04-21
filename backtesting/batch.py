from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .engine import BacktestEngine, BacktestResult
from .resample import infer_source_timeframe_label, normalize_timeframe, resample_ohlcv
from .stats import compute_performance_stats, infer_periods_per_year
from .strategy import Strategy


@dataclass(slots=True)
class BatchBacktestResult:
    run_results: dict[str, BacktestResult]
    run_data: dict[str, pd.DataFrame]
    summary: pd.DataFrame
    aggregate_equity_curve: pd.Series
    aggregate_returns: pd.Series
    aggregate_stats: dict[str, float]


def run_batch_backtest(
    data_by_asset: dict[str, pd.DataFrame],
    timeframes: list[str] | None,
    engine: BacktestEngine,
    strategy_factory: Callable[[str, str], Strategy],
) -> BatchBacktestResult:
    """Run many asset/timeframe backtests and aggregate results."""
    run_results: dict[str, BacktestResult] = {}
    run_data: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, float | str]] = []

    provided_timeframes = [tf.strip() for tf in (timeframes or []) if tf and tf.strip()]
    use_auto_timeframe = not provided_timeframes or any(tf.lower() == "auto" for tf in provided_timeframes)

    for asset, df in data_by_asset.items():
        source_timeframe = infer_source_timeframe_label(df.index)
        if use_auto_timeframe:
            if source_timeframe == "unknown":
                raise ValueError(
                    f"Unable to infer source timeframe for asset '{asset}'. "
                    "Specify --timeframes explicitly (e.g. 5m,15m,1h)."
                )
            asset_timeframes = [source_timeframe]
        else:
            asset_timeframes = provided_timeframes

        for timeframe in asset_timeframes:
            key = f"{asset}|{timeframe}"
            normalized_timeframe = normalize_timeframe(timeframe)
            normalized_source = normalize_timeframe(source_timeframe) if source_timeframe != "unknown" else source_timeframe
            if source_timeframe != "unknown" and normalized_timeframe == normalized_source:
                tf_data = df.copy()
            else:
                tf_data = resample_ohlcv(df, timeframe)
            if len(tf_data) < 2:
                continue
            strategy = strategy_factory(asset, timeframe)
            result = engine.run(tf_data, strategy)
            run_results[key] = result
            run_data[key] = tf_data

            row: dict[str, float | str] = {
                "run_key": key,
                "asset": asset,
                "timeframe": timeframe,
            }
            row.update(result.stats)
            summary_rows.append(row)

    if not run_results:
        raise ValueError("Batch run produced no valid backtest results")

    summary = pd.DataFrame(summary_rows).sort_values(["asset", "timeframe"]).reset_index(drop=True)

    returns_df = pd.concat(
        {k: v.returns for k, v in run_results.items()},
        axis=1,
    ).fillna(0.0)
    positions_df = pd.concat(
        {k: v.positions for k, v in run_results.items()},
        axis=1,
    ).fillna(0.0)
    aggregate_returns = returns_df.mean(axis=1)
    initial_capital = engine.config.initial_capital
    aggregate_equity_curve = (1 + aggregate_returns).cumprod() * initial_capital
    aggregate_positions = positions_df.mean(axis=1)

    periods_per_year = infer_periods_per_year(aggregate_returns.index)
    aggregate_stats = compute_performance_stats(
        equity_curve=aggregate_equity_curve,
        returns=aggregate_returns,
        trades=[],
        periods_per_year=periods_per_year,
        positions=aggregate_positions,
    )
    aggregate_stats["total_trades"] = float(summary["total_trades"].sum()) if "total_trades" in summary else 0.0
    aggregate_stats["total_runs"] = float(len(run_results))

    return BatchBacktestResult(
        run_results=run_results,
        run_data=run_data,
        summary=summary,
        aggregate_equity_curve=aggregate_equity_curve,
        aggregate_returns=aggregate_returns,
        aggregate_stats=aggregate_stats,
    )
