from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting import (
    AlligatorAOStrategy,
    BacktestConfig,
    BacktestEngine,
    BWStrategy,
    CombinedStrategy,
    NTDStrategy,
    WisemanStrategy,
    compute_trade_diagnostics,
    filter_ohlcv_by_date,
    load_ohlcv_csv,
    parse_trade_size_equity_milestones,
)
from examples.run_wiseman_parameter_sweep import _SimplePdfReport, _draw_bar_chart, _draw_histogram, _draw_line_chart, _draw_scatter_chart, _draw_table, _text_block


def _parse_strategy_selection(selection: str) -> tuple[str, ...]:
    allowed = {"alligator_ao", "wiseman", "ntd", "bw"}
    names = tuple(dict.fromkeys(part.strip().lower() for part in selection.split(",") if part.strip()))
    if not names:
        raise ValueError("--strategy must include at least one strategy name")
    invalid = [name for name in names if name not in allowed]
    if invalid:
        raise ValueError(f"Unsupported strategy name(s): {', '.join(invalid)}")
    if ("alligator_ao" in names or "bw" in names) and len(names) > 1:
        raise ValueError("alligator_ao and bw cannot be combined with other strategies")
    return names


def _build_strategy(args: argparse.Namespace) -> AlligatorAOStrategy | WisemanStrategy | NTDStrategy | BWStrategy | CombinedStrategy:
    strategy_names = _parse_strategy_selection(args.strategy)

    def _build_wiseman_strategy() -> WisemanStrategy:
        return WisemanStrategy(
            gator_width_lookback=args.gator_width_lookback,
            gator_width_mult=args.gator_width_mult,
            gator_width_valid_factor=args.gator_width_valid_factor,
            gator_direction_mode=args.wiseman_gator_direction_mode,
            first_wiseman_contracts=args.wiseman_1w_contracts,
            second_wiseman_contracts=args.wiseman_2w_contracts,
            third_wiseman_contracts=args.wiseman_3w_contracts,
            reversal_contracts_mult=args.wiseman_reversal_contracts_mult,
            first_wiseman_wait_bars_to_close=args.wiseman_1w_wait_bars_to_close,
            first_wiseman_divergence_filter_bars=args.wiseman_1w_divergence_filter_bars,
            first_wiseman_opposite_close_min_unrealized_return=args.wiseman_1w_opposite_close_min_unrealized_return,
            first_wiseman_reversal_cooldown=args.wiseman_reversal_cooldown,
            cancel_reversal_on_first_wiseman_exit=args.wiseman_cancel_reversal_on_first_exit,
            teeth_profit_protection_enabled=args.wiseman_profit_protection_teeth_exit,
            teeth_profit_protection_min_bars=args.wiseman_profit_protection_min_bars,
            teeth_profit_protection_min_unrealized_return=args.wiseman_profit_protection_min_unrealized_return,
            teeth_profit_protection_credit_unrealized_before_min_bars=args.wiseman_profit_protection_credit_unrealized_before_min_bars,
            teeth_profit_protection_require_gator_open=args.wiseman_profit_protection_require_gator_open,
            profit_protection_volatility_lookback=args.wiseman_profit_protection_volatility_lookback,
                profit_protection_annualized_volatility_scaler=args.wiseman_profit_protection_annualized_volatility_scaler,
            lips_profit_protection_enabled=args.wiseman_profit_protection_lips_exit,
            lips_profit_protection_volatility_trigger=args.wiseman_profit_protection_lips_volatility_trigger,
            lips_profit_protection_profit_trigger_mult=args.wiseman_profit_protection_lips_profit_trigger_mult,
            lips_profit_protection_volatility_lookback=args.wiseman_profit_protection_lips_volatility_lookback,
            lips_profit_protection_recent_trade_lookback=args.wiseman_profit_protection_lips_recent_trade_lookback,
            lips_profit_protection_min_unrealized_return=args.wiseman_profit_protection_lips_min_unrealized_return,
            lips_profit_protection_arm_on_min_unrealized_return=args.wiseman_profit_protection_lips_arm_on_min_unrealized_return,
            zone_profit_protection_enabled=args.wiseman_profit_protection_zones_exit,
            zone_profit_protection_min_unrealized_return=args.wiseman_profit_protection_zones_min_unrealized_return,
        )

    def _build_ntd_strategy() -> NTDStrategy:
        return NTDStrategy(
            gator_width_lookback=args.gator_width_lookback,
            gator_width_mult=args.gator_width_mult,
            require_gator_close_reset=(
                args.ntd_require_gator_close_reset
                if args.ntd_require_gator_close_reset is not None
                else "wiseman" not in strategy_names
            ),
            ao_ac_near_zero_lookback=args.ntd_ao_ac_near_zero_lookback,
            ao_ac_near_zero_factor=args.ntd_ao_ac_near_zero_factor,
            teeth_profit_protection_enabled=args.wiseman_profit_protection_teeth_exit,
            teeth_profit_protection_min_bars=args.wiseman_profit_protection_min_bars,
            teeth_profit_protection_min_unrealized_return=args.wiseman_profit_protection_min_unrealized_return,
            teeth_profit_protection_credit_unrealized_before_min_bars=args.wiseman_profit_protection_credit_unrealized_before_min_bars,
            teeth_profit_protection_require_gator_open=args.wiseman_profit_protection_require_gator_open,
            profit_protection_volatility_lookback=args.wiseman_profit_protection_volatility_lookback,
            profit_protection_annualized_volatility_scaler=args.wiseman_profit_protection_annualized_volatility_scaler,
            lips_profit_protection_enabled=args.wiseman_profit_protection_lips_exit,
            lips_profit_protection_volatility_trigger=args.wiseman_profit_protection_lips_volatility_trigger,
            lips_profit_protection_profit_trigger_mult=args.wiseman_profit_protection_lips_profit_trigger_mult,
            lips_profit_protection_volatility_lookback=args.wiseman_profit_protection_lips_volatility_lookback,
            lips_profit_protection_recent_trade_lookback=args.wiseman_profit_protection_lips_recent_trade_lookback,
            lips_profit_protection_min_unrealized_return=args.wiseman_profit_protection_lips_min_unrealized_return,
            lips_profit_protection_arm_on_min_unrealized_return=args.wiseman_profit_protection_lips_arm_on_min_unrealized_return,
            zone_profit_protection_enabled=args.wiseman_profit_protection_zones_exit,
            zone_profit_protection_min_unrealized_return=args.wiseman_profit_protection_zones_min_unrealized_return,
        )

    if strategy_names == ("alligator_ao",):
        return AlligatorAOStrategy()
    if strategy_names == ("bw",):
        return BWStrategy(
            divergence_filter_bars=args.bw_1w_divergence_filter_bars,
            gator_open_filter_lookback=args.bw_1w_gator_open_lookback,
            gator_open_filter_min_percentile=args.bw_1w_gator_open_percentile,
            ntd_initial_fractal_enabled=args.bw_ntd_initial_fractal_enabled,
            ntd_sleeping_gator_lookback=args.bw_ntd_sleeping_gator_lookback,
            ntd_sleeping_gator_tightness_mult=args.bw_ntd_sleeping_gator_tightness_mult,
            ntd_ranging_lookback=args.bw_ntd_ranging_lookback,
            ntd_ranging_max_span_pct=args.bw_ntd_ranging_max_span_pct,
            peak_drawdown_exit_enabled=args.bw_peak_drawdown_exit,
            peak_drawdown_exit_pct=args.bw_peak_drawdown_exit_pct,
            peak_drawdown_exit_volatility_lookback=args.bw_peak_drawdown_exit_volatility_lookback,
            peak_drawdown_exit_annualized_volatility_scaler=(
                args.bw_peak_drawdown_exit_annualized_volatility_scaler
            ),
            sigma_move_profit_protection_enabled=args.bw_profit_protection_sigma_move_exit,
            sigma_move_profit_protection_lookback=args.bw_profit_protection_sigma_move_lookback,
            sigma_move_profit_protection_sigma=args.bw_profit_protection_sigma_move_sigma,
        )
    strategies = []
    if "wiseman" in strategy_names:
        strategies.append(_build_wiseman_strategy())
    if "ntd" in strategy_names:
        strategies.append(_build_ntd_strategy())
    return strategies[0] if len(strategies) == 1 else CombinedStrategy(strategies)


def _align_timestamp(raw: str | None, fallback: pd.Timestamp, tz) -> pd.Timestamp:
    if raw is None:
        return fallback
    ts = pd.Timestamp(raw)
    if tz is not None:
        if ts.tzinfo is None:
            return ts.tz_localize(tz)
        return ts.tz_convert(tz)
    if ts.tzinfo is not None:
        return ts.tz_convert(None)
    return ts


def _window_ranges(index: pd.DatetimeIndex, start: str | None, end: str | None, window_months: int, jump_months: int) -> list[tuple[pd.Timestamp, pd.Timestamp, bool]]:
    tz = index.tz
    dataset_start = index[0]
    dataset_end = index[-1]
    bound_start = _align_timestamp(start, dataset_start, tz)

    # `--end` is treated as the final window *start* (not truncation bound).
    if end is None:
        bound_end = dataset_end - pd.DateOffset(months=window_months)
    else:
        bound_end = _align_timestamp(end, dataset_end, tz)

    windows: list[tuple[pd.Timestamp, pd.Timestamp, bool]] = []
    cursor = bound_start
    while cursor <= bound_end:
        planned_end = cursor + pd.DateOffset(months=window_months)
        actual_end = min(planned_end, dataset_end)
        is_cutoff = actual_end < planned_end
        windows.append((cursor, actual_end, is_cutoff))
        cursor = cursor + pd.DateOffset(months=jump_months)
        if cursor > bound_end:
            break
    return windows


def _kmeans_clusters(frame: pd.DataFrame, k: int = 3) -> np.ndarray:
    if frame.empty:
        return np.array([], dtype=int)
    cols = ["total_return", "max_drawdown", "sharpe"]
    x = frame[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    std = x.std(axis=0)
    std[std == 0] = 1.0
    z = (x - x.mean(axis=0)) / std
    k = max(1, min(k, len(z)))
    centers = z[:k].copy()
    labels = np.zeros(len(z), dtype=int)
    for _ in range(20):
        d = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        new_centers = np.array([z[labels == i].mean(axis=0) if np.any(labels == i) else centers[i] for i in range(k)])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels


def _write_svg_line(path: Path, values: list[float], title: str) -> None:
    w, h, pad = 900, 300, 30
    if not values:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='900' height='300'></svg>", encoding="utf-8")
        return
    vmin, vmax = min(values), max(values)
    span = (vmax - vmin) or 1.0
    pts = []
    for i, v in enumerate(values):
        x = pad + (i / max(1, len(values) - 1)) * (w - 2 * pad)
        y = h - pad - ((v - vmin) / span) * (h - 2 * pad)
        pts.append(f"{x:.1f},{y:.1f}")
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}'><text x='10' y='18' font-size='14'>{title}</text><polyline fill='none' stroke='#2b6cb0' stroke-width='2' points='{' '.join(pts)}'/></svg>"
    path.write_text(svg, encoding="utf-8")


def _write_svg_hist(path: Path, values: list[float], title: str, bins: int = 12) -> None:
    w, h, pad = 900, 300, 30
    hist, _ = np.histogram(np.array(values, dtype=float), bins=min(bins, max(1, len(values))))
    max_h = max(hist) if len(hist) else 1
    bar_w = (w - 2 * pad) / max(1, len(hist))
    rects = []
    for i, c in enumerate(hist):
        bh = (c / max_h) * (h - 2 * pad)
        x = pad + i * bar_w
        y = h - pad - bh
        rects.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{max(1.0, bar_w - 2):.1f}' height='{bh:.1f}' fill='#4a5568'/>")
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}'><text x='10' y='18' font-size='14'>{title}</text>{''.join(rects)}</svg>"
    path.write_text(svg, encoding="utf-8")




def _make_pdf_report(out_path: Path, df: pd.DataFrame, summary: dict[str, object], args: argparse.Namespace, data: pd.DataFrame) -> None:
    pdf = _SimplePdfReport()

    def _pct(value: float) -> str:
        return f"{float(value) * 100.0:.2f}%"

    def _num(value: float) -> str:
        return f"{float(value):.3f}"

    def _safe_quantile(series: pd.Series, q: float) -> float:
        if series.empty:
            return 0.0
        return float(series.quantile(q))

    worst = df.sort_values("total_return").head(5)
    best = df.sort_values("total_return", ascending=False).head(5)
    cluster_counts = df["cluster"].value_counts().sort_index()
    return_pct = df["total_return"].astype(float) * 100.0
    drawdown_pct = df["max_drawdown"].astype(float) * 100.0
    cagr_pct = df["cagr"].astype(float) * 100.0
    sharpe = df["sharpe"].astype(float)
    win_rate_pct = df["win_rate"].astype(float) * 100.0
    trades = df["trades"].astype(float)
    bars = df["bars"].astype(float)
    final_equity = df["final_equity"].astype(float)

    positive_return_windows = int((df["total_return"].astype(float) > 0).sum())
    negative_return_windows = int((df["total_return"].astype(float) < 0).sum())
    non_negative_windows = int((df["total_return"].astype(float) >= 0).sum())
    positive_sharpe_windows = int((sharpe > 0).sum())

    cluster_profile_rows: list[list[object]] = []
    for cluster_id, group in df.groupby("cluster", sort=True):
        cluster_profile_rows.append(
            [
                int(cluster_id),
                int(len(group)),
                _pct(float(group["total_return"].mean())),
                _pct(float(group["max_drawdown"].mean())),
                _num(float(group["sharpe"].mean())),
                _pct(float(group["win_rate"].mean())),
            ]
        )

    best_row = df.sort_values("total_return", ascending=False).iloc[0]
    worst_row = df.sort_values("total_return", ascending=True).iloc[0]

    # Page 1: executive summary + complete run configuration.
    y = 560.0
    pdf.text(40, y, "Rolling Multi-Timeframe Sweep Report", size=20)
    y -= 26
    y = _text_block(
        pdf,
        40,
        y,
        (
            "This report summarizes walk-forward rolling window outcomes with optional multi-threaded execution. "
            "Each window represents an independent strategy evaluation across a fixed horizon and is analyzed for "
            "return quality, drawdown pressure, consistency, and operational resilience."
        ),
        size=10,
        line_gap=13,
    )

    run_config_rows = [
        ["source_csv", args.csv],
        ["window_start_range", f"{args.start or data.index[0].isoformat()} -> {args.end or (data.index[-1] - pd.DateOffset(months=args.window_months)).isoformat()}"],
        ["window_months", args.window_months],
        ["data_available_through", data.index[-1].isoformat()],
        ["jump_forward_months", args.jump_forward_months],
        ["threads", args.threads],
        ["strategy", args.strategy],
        ["capital", args.capital],
        ["fee", args.fee],
        ["slippage", args.slippage],
        ["spread", args.spread],
        ["order_type", args.order_type],
        ["size_mode", args.size_mode],
        ["size_value", args.size_value],
        ["size_min_usd", args.size_min_usd],
        ["volatility_target_annual", args.volatility_target_annual],
        ["volatility_lookback", args.volatility_lookback],
        ["volatility_min_scale", args.volatility_min_scale],
        ["volatility_max_scale", args.volatility_max_scale],
        ["max_leverage", args.max_leverage],
        ["leverage_stop_out", args.leverage_stop_out],
        ["borrow_annual", args.borrow_annual],
        ["funding_per_period", args.funding_per_period],
        ["overnight_annual", args.overnight_annual],
        ["max_loss", args.max_loss],
        ["equity_cutoff", args.equity_cutoff],
    ]
    if args.strategy == "wiseman":
        run_config_rows.extend(
            [
                ["gator_width_lookback", args.gator_width_lookback],
                ["gator_width_mult", args.gator_width_mult],
                ["gator_width_valid_factor", args.gator_width_valid_factor],
                ["wiseman_1w_contracts", args.wiseman_1w_contracts],
                ["wiseman_2w_contracts", args.wiseman_2w_contracts],
                ["wiseman_3w_contracts", args.wiseman_3w_contracts],
                ["wiseman_reversal_contracts_mult", args.wiseman_reversal_contracts_mult],
                ["wiseman_1w_wait_bars_to_close", args.wiseman_1w_wait_bars_to_close],
                ["wiseman_1w_divergence_filter_bars", args.wiseman_1w_divergence_filter_bars],
                ["wiseman_reversal_cooldown", args.wiseman_reversal_cooldown],
                ["wiseman_gator_direction_mode", args.wiseman_gator_direction_mode],
                ["wiseman_cancel_reversal_on_first_exit", args.wiseman_cancel_reversal_on_first_exit],
                ["wiseman_profit_protection_teeth_exit", args.wiseman_profit_protection_teeth_exit],
                ["wiseman_profit_protection_min_bars", args.wiseman_profit_protection_min_bars],
                ["wiseman_profit_protection_min_unrealized_return", args.wiseman_profit_protection_min_unrealized_return],
                [
                    "wiseman_profit_protection_credit_unrealized_before_min_bars",
                    args.wiseman_profit_protection_credit_unrealized_before_min_bars,
                ],
                ["wiseman_profit_protection_require_gator_open", args.wiseman_profit_protection_require_gator_open],
            ]
        )

    run_config_row_height = 12
    run_config_table_y = 430
    first_page_max_rows = max(1, int((run_config_table_y - 20) / run_config_row_height))
    _draw_table(
        pdf,
        40,
        run_config_table_y,
        760,
        ["field", "value"],
        run_config_rows[:first_page_max_rows],
        "Run configuration",
        row_height=run_config_row_height,
        max_rows=first_page_max_rows,
    )

    remaining_run_config_rows = run_config_rows[first_page_max_rows:]
    run_config_rows_per_page = 44
    for offset in range(0, len(remaining_run_config_rows), run_config_rows_per_page):
        pdf.new_page()
        _draw_table(
            pdf,
            40,
            520,
            760,
            ["field", "value"],
            remaining_run_config_rows[offset : offset + run_config_rows_per_page],
            "Run configuration (continued)",
            row_height=run_config_row_height,
            max_rows=run_config_rows_per_page,
        )

    # Next page: headline diagnostics table (isolated to prevent overlap/cutoff).
    pdf.new_page()
    def _series_or_nan(col: str) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype="float64")

    mean_position_size_usd = _series_or_nan("mean_position_size_usd")
    median_position_size_usd = _series_or_nan("median_position_size_usd")
    mean_trade_pnl_usd = _series_or_nan("mean_trade_pnl_usd")
    median_trade_pnl_usd = _series_or_nan("median_trade_pnl_usd")
    mean_trade_pnl_pct = _series_or_nan("mean_trade_pnl_pct")
    median_trade_pnl_pct = _series_or_nan("median_trade_pnl_pct")
    total_slippage_paid = _series_or_nan("total_slippage_paid")
    fees_per_trade = _series_or_nan("fees_per_trade")
    slippage_per_trade = _series_or_nan("slippage_per_trade")
    total_cumulative_volume = _series_or_nan("total_cumulative_volume")
    total_cumulative_fees = _series_or_nan("total_cumulative_fees")
    total_cumulative_slippage = _series_or_nan("total_cumulative_slippage")
    mean_volume_per_trade = _series_or_nan("mean_volume_per_trade")
    median_volume_per_trade = _series_or_nan("median_volume_per_trade")
    mean_fee_per_trade = _series_or_nan("mean_fee_per_trade")
    median_fee_per_trade = _series_or_nan("median_fee_per_trade")
    mean_slippage_per_trade = _series_or_nan("mean_slippage_per_trade")
    median_slippage_per_trade = _series_or_nan("median_slippage_per_trade")

    _draw_table(
        pdf,
        40,
        510,
        760,
        ["metric", "value"],
        [
            ["windows_total", summary["windows_total"]],
            ["windows_liquidated", summary["windows_liquidated"]],
            ["windows_equity_cutoff_hit", summary["windows_equity_cutoff_hit"]],
            ["windows_time_cutoff", summary["windows_time_cutoff"]],
            ["mean_total_return", _pct(float(summary["mean_total_return"]))],
            ["median_total_return", _pct(float(summary["median_total_return"]))],
            ["return_std_dev", _pct(float(df["total_return"].astype(float).std(ddof=0)))],
            ["mean_sharpe", _num(float(summary["mean_sharpe"]))],
            ["median_sharpe", _num(float(summary["median_sharpe"]))],
            ["expected_win_rate", _pct(float(summary["expected_win_rate"]))],
            ["positive_return_windows", positive_return_windows],
            ["non_negative_windows", non_negative_windows],
            ["positive_sharpe_windows", positive_sharpe_windows],
            ["mean_position_size_usd", _num(float(mean_position_size_usd.mean()))],
            ["median_position_size_usd", _num(float(median_position_size_usd.median()))],
            ["mean_trade_pnl_usd", _num(float(mean_trade_pnl_usd.mean()))],
            ["median_trade_pnl_usd", _num(float(median_trade_pnl_usd.median()))],
            ["mean_trade_pnl_pct", _pct(float(mean_trade_pnl_pct.mean()))],
            ["median_trade_pnl_pct", _pct(float(median_trade_pnl_pct.median()))],
            ["total_slippage_paid", _num(float(total_slippage_paid.sum()))],
            ["fees_per_trade", _num(float(fees_per_trade.mean()))],
            ["slippage_per_trade", _num(float(slippage_per_trade.mean()))],
            ["total_cumulative_volume", _num(float(total_cumulative_volume.sum()))],
            ["total_cumulative_fees", _num(float(total_cumulative_fees.sum()))],
            ["total_cumulative_slippage", _num(float(total_cumulative_slippage.sum()))],
            ["mean_volume_per_trade", _num(float(mean_volume_per_trade.mean()))],
            ["median_volume_per_trade", _num(float(median_volume_per_trade.median()))],
            ["mean_fee_per_trade", _num(float(mean_fee_per_trade.mean()))],
            ["median_fee_per_trade", _num(float(median_fee_per_trade.median()))],
            ["mean_slippage_per_trade", _num(float(mean_slippage_per_trade.mean()))],
            ["median_slippage_per_trade", _num(float(median_slippage_per_trade.median()))],
        ],
        "Portfolio-level diagnostics",
        row_height=12,
        max_rows=33,
    )

    pdf.new_page()
    pdf.text(40, 560, "Volume / Fees / Slippage Turnover", size=16)
    _draw_line_chart(
        pdf,
        40,
        320,
        760,
        210,
        total_cumulative_volume.fillna(0.0).cumsum().tolist(),
        "Cumulative volume across windows",
        "USD",
        x_label="Window sequence",
        caption="Running sum of total cumulative volume from each rolling window.",
    )
    _draw_line_chart(
        pdf,
        40,
        60,
        360,
        210,
        total_cumulative_fees.fillna(0.0).cumsum().tolist(),
        "Cumulative fees across windows",
        "USD",
        x_label="Window sequence",
        caption="Running sum of total fees across rolling windows.",
        color=(0.80, 0.42, 0.15),
    )
    _draw_line_chart(
        pdf,
        430,
        60,
        370,
        210,
        total_cumulative_slippage.fillna(0.0).cumsum().tolist(),
        "Cumulative slippage across windows",
        "USD",
        x_label="Window sequence",
        caption="Running sum of estimated slippage across rolling windows.",
        color=(0.82, 0.12, 0.12),
    )

    # Page 3: return and risk distributions
    pdf.new_page()
    _draw_line_chart(
        pdf,
        40,
        315,
        360,
        230,
        return_pct.tolist(),
        "Rolling total return by window",
        "Total return (%)",
        x_label="Window sequence",
        caption="Window-by-window return trajectory with consistent scaling to highlight regime breaks and recovery phases.",
    )
    _draw_histogram(
        pdf,
        430,
        315,
        360,
        230,
        return_pct.tolist(),
        12,
        "Total return distribution",
        x_label="Total return (%)",
        y_label="Window count",
        caption="Distribution of rolling returns to evaluate central tendency, skew, and tail outcomes.",
    )
    _draw_line_chart(
        pdf,
        40,
        35,
        360,
        230,
        drawdown_pct.tolist(),
        "Max drawdown by window",
        "Max drawdown (%)",
        color=(0.55, 0.16, 0.16),
        x_label="Window sequence",
        caption="Drawdown depth per window to spot clustered stress periods and capital impairment risk.",
    )
    _draw_histogram(
        pdf,
        430,
        35,
        360,
        230,
        sharpe.tolist(),
        12,
        "Sharpe distribution",
        x_label="Sharpe ratio",
        y_label="Window count",
        caption="Risk-adjusted performance dispersion across windows, showing consistency of quality.",
    )

    # Page 3: cross-metric tradeoffs and execution consistency
    pdf.new_page()
    _draw_scatter_chart(
        pdf,
        40,
        315,
        360,
        230,
        drawdown_pct.tolist(),
        return_pct.tolist(),
        "Return vs drawdown tradeoff",
        "Max drawdown (%)",
        "Total return (%)",
        caption="Each point is a rolling window; preferred outcomes are higher return with shallower drawdown.",
    )
    _draw_bar_chart(
        pdf,
        430,
        315,
        360,
        230,
        [f"C{int(c)}" for c in cluster_counts.index.tolist()],
        cluster_counts.astype(float).tolist(),
        "Cluster membership",
        x_label="Cluster",
        y_label="Window count",
        caption="K-means grouping of return, drawdown, and Sharpe profiles across windows.",
    )
    _draw_scatter_chart(
        pdf,
        40,
        35,
        360,
        230,
        trades.tolist(),
        return_pct.tolist(),
        "Return vs trade count",
        "Trades per window",
        "Total return (%)",
        caption="Checks whether returns depend on over-trading or remain robust at moderate activity levels.",
    )
    _draw_line_chart(
        pdf,
        430,
        35,
        360,
        230,
        win_rate_pct.tolist(),
        "Win rate by window",
        "Win rate (%)",
        color=(0.15, 0.45, 0.28),
        x_label="Window sequence",
        caption="Execution consistency by period; structural dips can indicate regime mismatch.",
    )

    # Page 4: ranking, cluster diagnostics, and operational guidance
    pdf.new_page()
    _draw_table(
        pdf,
        40,
        330,
        760,
        ["window_id", "start", "end", "total_return", "max_drawdown", "sharpe", "cluster"],
        best[["window_id", "start", "end", "total_return", "max_drawdown", "sharpe", "cluster"]].values.tolist(),
        "Top windows (highest returns)",
        row_height=13,
        max_rows=6,
    )
    _draw_table(
        pdf,
        40,
        170,
        760,
        ["window_id", "start", "end", "total_return", "max_drawdown", "sharpe", "cluster"],
        worst[["window_id", "start", "end", "total_return", "max_drawdown", "sharpe", "cluster"]].values.tolist(),
        "Bottom windows (lowest returns)",
        row_height=13,
        max_rows=6,
    )
    _draw_table(
        pdf,
        40,
        44,
        760,
        ["cluster", "windows", "avg_return", "avg_max_dd", "avg_sharpe", "avg_win_rate"],
        cluster_profile_rows,
        "Cluster profile summary",
        row_height=13,
        max_rows=4,
    )

    # Page 5: extended diagnostics and interpretation
    pdf.new_page()
    _draw_table(
        pdf,
        40,
        360,
        760,
        ["metric", "value"],
        [
            ["best_window_id", int(best_row["window_id"])],
            ["best_window_return", _pct(float(best_row["total_return"]))],
            ["worst_window_id", int(worst_row["window_id"])],
            ["worst_window_return", _pct(float(worst_row["total_return"]))],
            ["median_cagr", _pct(float(cagr_pct.median() / 100.0))],
            ["p10_return", _pct(_safe_quantile(df["total_return"].astype(float), 0.10))],
            ["p90_return", _pct(_safe_quantile(df["total_return"].astype(float), 0.90))],
            ["p10_sharpe", _num(_safe_quantile(sharpe, 0.10))],
            ["p90_sharpe", _num(_safe_quantile(sharpe, 0.90))],
            ["median_drawdown", _pct(float(df["max_drawdown"].astype(float).median()))],
            ["average_trades_per_window", _num(float(trades.mean()))],
            ["average_bars_per_window", _num(float(bars.mean()))],
            ["average_final_equity", _num(float(final_equity.mean()))],
            ["negative_return_windows", negative_return_windows],
        ],
        "Distribution diagnostics and tail-risk checks",
        row_height=14,
        max_rows=14,
    )
    y = _text_block(
        pdf,
        40,
        150,
        (
            "Interpretation guidance:\n"
            "1) Favor parameterizations that keep P10 returns acceptable while maintaining positive median Sharpe.\n"
            "2) Examine windows with deep drawdown spikes for shared market structure and adjust leverage caps accordingly.\n"
            "3) Validate that top-performing clusters are not driven by a single outlier window before live deployment.\n"
            "4) If liquidation/equity-cutoff events cluster in one regime, reduce size and add stricter stop-out controls."
        ),
        size=9,
        line_gap=11,
    )
    _text_block(
        pdf,
        40,
        y - 6,
        "Metric glossary: P10/P90 = 10th/90th percentile across windows; win rate = winning trades / total trades per window.",
        size=8,
        line_gap=10,
    )

    # Final section: complete ledger of every tested rolling window sequence and outcomes.
    full_results_df = df.copy()
    full_results_df["start"] = pd.to_datetime(full_results_df["start"]).dt.strftime("%Y-%m-%d")
    full_results_df["end"] = pd.to_datetime(full_results_df["end"]).dt.strftime("%Y-%m-%d")
    full_results_rows = full_results_df[
        [
            "window_id",
            "start",
            "end",
            "total_return",
            "max_drawdown",
            "sharpe",
            "win_rate",
            "trades",
            "liquidated",
            "equity_cutoff_hit",
            "window_cutoff",
        ]
    ].values.tolist()
    rows_per_page = 38
    for offset in range(0, len(full_results_rows), rows_per_page):
        pdf.new_page()
        _draw_table(
            pdf,
            40,
            520,
            760,
            [
                "window_id",
                "start",
                "end",
                "total_return",
                "max_drawdown",
                "sharpe",
                "win_rate",
                "trades",
                "liquidated",
                "equity_cutoff_hit",
                "window_cutoff",
            ],
            full_results_rows[offset : offset + rows_per_page],
            "Complete rolling window sequence results",
            row_height=12,
            max_rows=rows_per_page,
        )

    pdf.finalize(out_path)


def _write_svg_scatter(path: Path, xvals: list[float], yvals: list[float], clusters: list[int], title: str) -> None:
    w, h, pad = 900, 320, 30
    if not xvals:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='900' height='320'></svg>", encoding="utf-8")
        return
    xmin, xmax = min(xvals), max(xvals)
    ymin, ymax = min(yvals), max(yvals)
    xspan = (xmax - xmin) or 1.0
    yspan = (ymax - ymin) or 1.0
    colors = ["#1a365d", "#2f855a", "#97266d", "#744210", "#2d3748"]
    circles = []
    for x, y, c in zip(xvals, yvals, clusters):
        px = pad + ((x - xmin) / xspan) * (w - 2 * pad)
        py = h - pad - ((y - ymin) / yspan) * (h - 2 * pad)
        circles.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='4' fill='{colors[c % len(colors)]}'/>")
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}'><text x='10' y='18' font-size='14'>{title}</text>{''.join(circles)}</svg>"
    path.write_text(svg, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run rolling-window backtests and generate a comprehensive analytics report.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--window-months", type=int, required=True)
    parser.add_argument("--jump-forward-months", type=int, required=True)
    parser.add_argument("--threads", type=int, default=1, help="Number of rolling windows to backtest concurrently")
    parser.add_argument("--capital", type=float, default=10_000)
    parser.add_argument("--fee", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument("--spread", type=float, default=0.0)
    parser.add_argument("--order-type", default="market", choices=["market", "limit", "stop", "stop_limit"])
    parser.add_argument(
        "--size-mode",
        default="percent_of_equity",
        choices=["percent_of_equity", "usd", "units", "hybrid_min_usd_percent", "volatility_scaled", "stop_loss_scaled", "equity_milestone_usd"],
    )
    parser.add_argument("--size-value", type=float, default=1.0)
    parser.add_argument("--size-min-usd", type=float, default=0.0)
    parser.add_argument("--size-equity-milestones", default="", help="Comma-separated EQUITY:USD step pairs for equity_milestone_usd sizing, e.g. 15000:1500,20000:2000")
    parser.add_argument("--volatility-target-annual", type=float, default=0.15)
    parser.add_argument("--volatility-lookback", type=int, default=20)
    parser.add_argument("--volatility-min-scale", type=float, default=0.25)
    parser.add_argument("--volatility-max-scale", type=float, default=3.0)
    parser.add_argument("--max-leverage", type=float, default=None)
    parser.add_argument("--max-position-size", type=float, default=None, help="Maximum absolute position notional in quote currency (e.g. USD)")
    parser.add_argument("--leverage-stop-out", type=float, default=0.0)
    parser.add_argument("--borrow-annual", type=float, default=0.0)
    parser.add_argument("--funding-per-period", type=float, default=0.0)
    parser.add_argument("--overnight-annual", type=float, default=0.0)
    parser.add_argument("--max-loss", type=float, default=None)
    parser.add_argument("--max-loss-pct-of-equity", type=float, default=None)
    parser.add_argument("--equity-cutoff", type=float, default=None)
    parser.add_argument("--strategy", default="alligator_ao", help="Strategy name(s): alligator_ao, wiseman, ntd, bw, or comma-separated wiseman,ntd")
    parser.add_argument("--bw-1w-divergence-filter", "--1W-divergence-filter-bw", dest="bw_1w_divergence_filter_bars", type=int, default=0)
    parser.add_argument("--bw-1w-gator-open-lookback", type=int, default=0)
    parser.add_argument("--bw-1w-gator-open-percentile", type=float, default=50.0)
    parser.add_argument("--bw-ntd-initial-fractal-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bw-ntd-sleeping-gator-lookback", type=int, default=50)
    parser.add_argument("--bw-ntd-sleeping-gator-tightness-mult", type=float, default=0.75)
    parser.add_argument("--bw-ntd-ranging-lookback", type=int, default=20)
    parser.add_argument("--bw-ntd-ranging-max-span-pct", type=float, default=0.025)
    parser.add_argument("--bw-peak-drawdown-exit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bw-peak-drawdown-exit-pct", type=float, default=0.01)
    parser.add_argument("--bw-peak-drawdown-exit-volatility-lookback", type=int, default=20)
    parser.add_argument("--bw-peak-drawdown-exit-annualized-volatility-scaler", type=float, default=1.0)
    parser.add_argument("--bw-profit-protection-sigma-move-exit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bw-profit-protection-sigma-move-lookback", type=int, default=20)
    parser.add_argument("--bw-profit-protection-sigma-move-sigma", type=float, default=2.0)
    parser.add_argument("--gator-width-lookback", type=int, default=50)
    parser.add_argument("--gator-width-mult", type=float, default=1.0)
    parser.add_argument("--gator-width-valid-factor", type=float, default=1.0)
    parser.add_argument("--ntd-ao-ac-near-zero-lookback", type=int, default=50)
    parser.add_argument("--ntd-ao-ac-near-zero-factor", type=float, default=0.25)
    parser.add_argument("--ntd-require-gator-close-reset", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wiseman-1w-contracts", type=int, default=1)
    parser.add_argument("--wiseman-2w-contracts", type=int, default=3)
    parser.add_argument("--wiseman-3w-contracts", type=int, default=5)
    parser.add_argument("--wiseman-reversal-contracts-mult", type=float, default=1.0)
    parser.add_argument("--1W-wait-bars-to-close", dest="wiseman_1w_wait_bars_to_close", type=int, default=0)
    parser.add_argument("--1W-divergence-filter", dest="wiseman_1w_divergence_filter_bars", type=int, default=0)
    parser.add_argument("--wiseman-1w-opposite-close-min-unrealized-return", type=float, default=0.0)
    parser.add_argument("--wiseman-reversal-cooldown", type=int, default=0)
    parser.add_argument("--wiseman-gator-direction-mode", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--wiseman-cancel-reversal-on-first-exit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wiseman-profit-protection-teeth-exit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wiseman-profit-protection-min-bars", type=int, default=3)
    parser.add_argument("--wiseman-profit-protection-min-unrealized-return", type=float, default=1.0)
    parser.add_argument("--wiseman-profit-protection-credit-unrealized-before-min-bars", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wiseman-profit-protection-require-gator-open", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wiseman-profit-protection-volatility-lookback", type=int, default=None)
    parser.add_argument("--wiseman-profit-protection-annualized-volatility-scaler", type=float, default=1.0)
    parser.add_argument("--wiseman-profit-protection-lips-exit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wiseman-profit-protection-lips-volatility-trigger", type=float, default=0.02)
    parser.add_argument("--wiseman-profit-protection-lips-profit-trigger-mult", type=float, default=2.0)
    parser.add_argument("--wiseman-profit-protection-lips-volatility-lookback", type=int, default=20)
    parser.add_argument("--wiseman-profit-protection-lips-recent-trade-lookback", type=int, default=5)
    parser.add_argument("--wiseman-profit-protection-lips-min-unrealized-return", type=float, default=1.0)
    parser.add_argument("--wiseman-profit-protection-lips-arm-on-min-unrealized-return", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wiseman-profit-protection-zones-exit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wiseman-profit-protection-zones-min-unrealized-return", type=float, default=1.0)
    parser.add_argument("--out", default="artifacts_rolling_sweep")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.window_months <= 0 or args.jump_forward_months <= 0 or args.threads <= 0:
        raise ValueError("--window-months, --jump-forward-months, and --threads must be positive")

    data = load_ohlcv_csv(args.csv)
    # `--start` defines earliest candidate window start, but `--end` defines the last window start.
    # We therefore keep full tail history after `--end` so windows can extend to full duration.
    data = filter_ohlcv_by_date(data, start=args.start, end=None)
    if len(data) < 2:
        raise ValueError("Filtered dataset has fewer than 2 bars; widen --start range")

    windows = _window_ranges(data.index, args.start, args.end, args.window_months, args.jump_forward_months)
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=args.capital,
            fee_rate=args.fee,
            slippage_rate=args.slippage,
            spread_rate=args.spread,
            order_type=args.order_type,
            trade_size_mode=args.size_mode,
            trade_size_value=args.size_value,
            trade_size_min_usd=args.size_min_usd,
            trade_size_equity_milestones=parse_trade_size_equity_milestones(args.size_equity_milestones),
            volatility_target_annual=args.volatility_target_annual,
            volatility_lookback=args.volatility_lookback,
            volatility_min_scale=args.volatility_min_scale,
            volatility_max_scale=args.volatility_max_scale,
            max_leverage=args.max_leverage,
            max_position_size=args.max_position_size,
            leverage_stop_out_pct=args.leverage_stop_out,
            borrow_rate_annual=args.borrow_annual,
            funding_rate_per_period=args.funding_per_period,
            overnight_rate_annual=args.overnight_annual,
            max_loss=args.max_loss,
            max_loss_pct_of_equity=args.max_loss_pct_of_equity,
            equity_cutoff=args.equity_cutoff,
        )
    )

    def _run_window(idx: int, w_start: pd.Timestamp, w_end: pd.Timestamp, window_cutoff: bool) -> dict[str, object] | None:
        subset = data[(data.index >= w_start) & (data.index <= w_end)]
        if len(subset) < 2:
            return None
        result = engine.run(subset, _build_strategy(args))
        trades_df = result.trades_dataframe()
        trade_diag = compute_trade_diagnostics(
            trades_df=trades_df,
            initial_capital=args.capital,
            total_fees_paid=result.total_fees_paid,
            execution_events=result.execution_events,
            slippage_rate=args.slippage,
        )
        win_rate = float((trades_df["pnl"] > 0).mean()) if result.trades else 0.0
        liquidated = any(e.event_type == "liquidation" for e in result.execution_events)
        cutoff_hit = any("equity_cutoff" in e.event_type for e in result.execution_events)
        return {
            "window_id": idx,
            "start": subset.index[0].isoformat(),
            "end": subset.index[-1].isoformat(),
            "planned_end": w_end.isoformat(),
            "bars": int(len(subset)),
            "trades": int(len(result.trades)),
            "total_return": float(result.stats.get("total_return", 0.0)),
            "cagr": float(result.stats.get("cagr", 0.0)),
            "sharpe": float(result.stats.get("sharpe", 0.0)),
            "max_drawdown": float(result.stats.get("max_drawdown", 0.0)),
            "final_equity": float(result.equity_curve.iloc[-1]),
            "win_rate": win_rate,
            "mean_position_size_usd": float(trade_diag["mean_position_size_usd"]),
            "median_position_size_usd": float(trade_diag["median_position_size_usd"]),
            "mean_trade_pnl_usd": float(trade_diag["mean_trade_pnl_usd"]),
            "median_trade_pnl_usd": float(trade_diag["median_trade_pnl_usd"]),
            "mean_trade_pnl_pct": float(trade_diag["mean_trade_pnl_pct"]),
            "median_trade_pnl_pct": float(trade_diag["median_trade_pnl_pct"]),
            "total_slippage_paid": float(trade_diag["total_slippage_paid"]),
            "fees_per_trade": float(trade_diag["fees_per_trade"]),
            "slippage_per_trade": float(trade_diag["slippage_per_trade"]),
            "total_cumulative_volume": float(trade_diag["total_cumulative_volume"]),
            "total_cumulative_fees": float(trade_diag["total_cumulative_fees"]),
            "total_cumulative_slippage": float(trade_diag["total_cumulative_slippage"]),
            "mean_volume_per_trade": float(trade_diag["mean_volume_per_trade"]),
            "median_volume_per_trade": float(trade_diag["median_volume_per_trade"]),
            "mean_fee_per_trade": float(trade_diag["mean_fee_per_trade"]),
            "median_fee_per_trade": float(trade_diag["median_fee_per_trade"]),
            "mean_slippage_per_trade": float(trade_diag["mean_slippage_per_trade"]),
            "median_slippage_per_trade": float(trade_diag["median_slippage_per_trade"]),
            "liquidated": liquidated,
            "equity_cutoff_hit": cutoff_hit,
            "window_cutoff": bool(window_cutoff),
        }

    rows: list[dict[str, object]] = []
    indexed_windows = [(idx, w_start, w_end, window_cutoff) for idx, (w_start, w_end, window_cutoff) in enumerate(windows, start=1)]
    if args.threads == 1:
        for idx, w_start, w_end, window_cutoff in indexed_windows:
            row = _run_window(idx, w_start, w_end, window_cutoff)
            if row is not None:
                rows.append(row)
    else:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            for row in executor.map(lambda win: _run_window(*win), indexed_windows):
                if row is not None:
                    rows.append(row)

    rows.sort(key=lambda r: int(r["window_id"]))

    if not rows:
        raise ValueError("No valid rolling windows produced at least 2 bars")

    df = pd.DataFrame(rows)
    df["cluster"] = _kmeans_clusters(df, k=3)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    windows_csv = out_dir / "rolling_window_results.csv"
    df.to_csv(windows_csv, index=False)

    summary = {
        "windows_total": int(len(df)),
        "windows_liquidated": int(df["liquidated"].sum()),
        "windows_equity_cutoff_hit": int(df["equity_cutoff_hit"].sum()),
        "windows_time_cutoff": int(df["window_cutoff"].sum()),
        "mean_total_return": float(df["total_return"].mean()),
        "median_total_return": float(df["total_return"].median()),
        "mean_sharpe": float(df["sharpe"].mean()),
        "median_sharpe": float(df["sharpe"].median()),
        "expected_win_rate": float(df["win_rate"].mean()),
        "best_window": df.sort_values("total_return", ascending=False).iloc[0].to_dict(),
        "worst_window": df.sort_values("total_return", ascending=True).iloc[0].to_dict(),
        "cluster_counts": {str(k): int(v) for k, v in df["cluster"].value_counts().sort_index().items()},
    }
    summary_path = out_dir / "rolling_window_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _write_svg_line(charts_dir / "rolling_total_return_line.svg", (df["total_return"] * 100.0).tolist(), "Total return (%) by rolling window")
    _write_svg_hist(charts_dir / "rolling_total_return_hist.svg", (df["total_return"] * 100.0).tolist(), "Total return (%) distribution")
    _write_svg_scatter(
        charts_dir / "rolling_return_vs_drawdown_cluster.svg",
        (df["max_drawdown"] * 100.0).tolist(),
        (df["total_return"] * 100.0).tolist(),
        df["cluster"].astype(int).tolist(),
        "Return (%) vs Max Drawdown (%) cluster map",
    )

    report_path = out_dir / "rolling_window_report.pdf"
    _make_pdf_report(report_path, df, summary, args, data)

    print(f"Rolling windows complete: {len(df)}")
    print(f"Window-level results: {windows_csv}")
    print(f"Summary JSON: {summary_path}")
    print(f"Comprehensive report: {report_path}")


if __name__ == "__main__":
    main()
