from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from textwrap import wrap

import numpy as np
import pandas as pd

from backtesting import (
    BacktestConfig,
    BacktestEngine,
    WisemanStrategy,
    filter_ohlcv_by_date,
    load_ohlcv_csv,
    parse_trade_size_equity_milestones,
    run_return_bootstrap_monte_carlo,
    infer_source_timeframe_label,
)
from backtesting.resample import normalize_timeframe, resample_ohlcv
from backtesting.trade_metrics import compute_trade_diagnostics


Combo = tuple[int, int, int, float, int, float, float, int, int, int, bool, bool, int, float, bool, bool]
_WORKER_DATA: pd.DataFrame | None = None
_WORKER_ENGINE: BacktestEngine | None = None


class _SimplePdfReport:
    def __init__(self, page_width: int = 842, page_height: int = 595) -> None:
        self.page_width = page_width
        self.page_height = page_height
        self._pages: list[str] = []
        self._ops: list[str] = []

    @staticmethod
    def _escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def new_page(self) -> None:
        if self._ops:
            self._pages.append("\n".join(self._ops))
        self._ops = []

    def text(self, x: float, y: float, value: str, size: int = 11) -> None:
        self._ops.append(f"BT /F1 {size} Tf {x:.2f} {y:.2f} Td ({self._escape(value)}) Tj ET")

    def line(self, x1: float, y1: float, x2: float, y2: float, width: float = 1.0) -> None:
        self._ops.append(f"{width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def rect(self, x: float, y: float, width: float, height: float, fill_gray: float = 0.75) -> None:
        self._ops.append(f"{fill_gray:.2f} g {x:.2f} {y:.2f} {width:.2f} {height:.2f} re f")
        self._ops.append("0 g")

    def finalize(self, path: Path) -> None:
        if self._ops:
            self._pages.append("\n".join(self._ops))
            self._ops = []

        objs: list[bytes] = []

        def add_obj(content: str | bytes) -> int:
            raw = content.encode("latin-1") if isinstance(content, str) else content
            objs.append(raw)
            return len(objs)

        font_obj = add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        page_objs: list[int] = []
        content_objs: list[int] = []

        for page_stream in self._pages:
            stream = page_stream.encode("latin-1", errors="replace")
            content_obj = add_obj(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")
            content_objs.append(content_obj)
            page_obj = add_obj(
                f"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 {self.page_width} {self.page_height}] "
                f"/Resources << /Font << /F1 {font_obj} 0 R >> >> /Contents {content_obj} 0 R >>"
            )
            page_objs.append(page_obj)

        kids_ref = " ".join(f"{obj_id} 0 R" for obj_id in page_objs)
        pages_obj = add_obj(f"<< /Type /Pages /Count {len(page_objs)} /Kids [{kids_ref}] >>")

        for page_obj in page_objs:
            page_txt = objs[page_obj - 1].decode("latin-1")
            objs[page_obj - 1] = page_txt.replace("/Parent 0 0 R", f"/Parent {pages_obj} 0 R").encode("latin-1")

        catalog_obj = add_obj(f"<< /Type /Catalog /Pages {pages_obj} 0 R >>")

        out = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for i, obj in enumerate(objs, start=1):
            offsets.append(len(out))
            out.extend(f"{i} 0 obj\n".encode("ascii"))
            out.extend(obj)
            out.extend(b"\nendobj\n")

        xref_pos = len(out)
        out.extend(f"xref\n0 {len(objs)+1}\n".encode("ascii"))
        out.extend(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
        out.extend(
            f"trailer\n<< /Size {len(objs)+1} /Root {catalog_obj} 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("ascii")
        )
        path.write_bytes(out)


def _init_worker(data: pd.DataFrame, engine_config: BacktestConfig) -> None:
    global _WORKER_DATA, _WORKER_ENGINE
    _WORKER_DATA = data
    _WORKER_ENGINE = BacktestEngine(engine_config)


def _run_combo(combo: Combo) -> dict[str, float | int | str | bool]:
    if _WORKER_DATA is None or _WORKER_ENGINE is None:
        raise RuntimeError("Worker not initialized")

    (
        first,
        second,
        third,
        reversal_mult,
        lookback,
        gator_mult,
        gator_valid_factor,
        first_wait_bars_to_close,
        first_divergence_filter_bars,
        gator_direction_mode,
        reversal_cooldown,
        cancel_reversal_on_first_exit,
        pp_enabled,
        pp_min_bars,
        pp_min_return,
        pp_volatility_lookback,
        pp_annualized_volatility_scaler,
        pp_credit_unrealized_before_min_bars,
        pp_require_gator_open,
        zone_pp_enabled,
        zone_pp_min_return,
    ) = combo

    strategy = WisemanStrategy(
        first_wiseman_contracts=first,
        second_wiseman_contracts=second,
        third_wiseman_contracts=third,
        reversal_contracts_mult=reversal_mult,
        gator_width_lookback=lookback,
        gator_width_mult=gator_mult,
        gator_width_valid_factor=gator_valid_factor,
        first_wiseman_wait_bars_to_close=first_wait_bars_to_close,
        first_wiseman_divergence_filter_bars=first_divergence_filter_bars,
        gator_direction_mode=gator_direction_mode,
        first_wiseman_reversal_cooldown=reversal_cooldown,
        cancel_reversal_on_first_wiseman_exit=cancel_reversal_on_first_exit,
        teeth_profit_protection_enabled=pp_enabled,
        teeth_profit_protection_min_bars=pp_min_bars,
        teeth_profit_protection_min_unrealized_return=pp_min_return,
        profit_protection_volatility_lookback=pp_volatility_lookback,
        profit_protection_annualized_volatility_scaler=pp_annualized_volatility_scaler,
        teeth_profit_protection_credit_unrealized_before_min_bars=pp_credit_unrealized_before_min_bars,
        teeth_profit_protection_require_gator_open=pp_require_gator_open,
        zone_profit_protection_enabled=zone_pp_enabled,
        zone_profit_protection_min_unrealized_return=zone_pp_min_return,
    )
    result = _WORKER_ENGINE.run(_WORKER_DATA, strategy)

    row: dict[str, float | int | str | bool] = {
        "first_wiseman_contracts": first,
        "second_wiseman_contracts": second,
        "third_wiseman_contracts": third,
        "reversal_contracts_mult": reversal_mult,
        "gator_width_lookback": lookback,
        "gator_width_mult": gator_mult,
        "gator_width_valid_factor": gator_valid_factor,
        "first_wiseman_wait_bars_to_close": first_wait_bars_to_close,
        "first_wiseman_divergence_filter_bars": first_divergence_filter_bars,
        "gator_direction_mode": gator_direction_mode,
        "first_wiseman_reversal_cooldown": reversal_cooldown,
        "cancel_reversal_on_first_wiseman_exit": cancel_reversal_on_first_exit,
        "profit_protection_enabled": pp_enabled,
        "profit_protection_min_bars": pp_min_bars,
        "profit_protection_min_unrealized_return": pp_min_return,
        "profit_protection_volatility_lookback": pp_volatility_lookback,
        "profit_protection_annualized_volatility_scaler": pp_annualized_volatility_scaler,
        "profit_protection_credit_unrealized_before_min_bars": pp_credit_unrealized_before_min_bars,
        "profit_protection_require_gator_open": pp_require_gator_open,
        "zone_profit_protection_enabled": zone_pp_enabled,
        "zone_profit_protection_min_unrealized_return": zone_pp_min_return,
        "final_equity": float(result.equity_curve.iloc[-1]),
        "trades": float(len(result.trades)),
    }
    row.update(result.stats)
    row.update(_compute_additional_metrics(result))
    row.update(_compute_regime_metrics(result, _WORKER_DATA["close"]))

    close = _WORKER_DATA["close"].reindex(result.returns.index).ffill().bfill().astype(float)
    benchmark_returns = close.pct_change().fillna(0.0)
    benchmark_total_return = float((1.0 + benchmark_returns).prod() - 1.0)
    strategy_total_return = float((1.0 + result.returns.astype(float)).prod() - 1.0)
    row["buy_hold_return"] = benchmark_total_return
    row["alpha_vs_buy_hold"] = strategy_total_return - benchmark_total_return
    row["return_over_maxdd"] = _safe_div(strategy_total_return, abs(float(result.stats.get("max_drawdown", 0.0))))

    trade_diag = compute_trade_diagnostics(
        trades_df=result.trades_dataframe(),
        initial_capital=float(_WORKER_ENGINE.config.initial_capital),
        total_fees_paid=float(result.total_fees_paid),
        execution_events=result.execution_events,
        slippage_rate=float(_WORKER_ENGINE.config.slippage_rate),
    )
    row.update({
        "total_cumulative_volume": float(trade_diag["total_cumulative_volume"]),
        "total_cumulative_fees": float(trade_diag["total_cumulative_fees"]),
        "total_cumulative_slippage": float(trade_diag["total_cumulative_slippage"]),
        "mean_volume_per_trade": float(trade_diag["mean_volume_per_trade"]),
        "median_volume_per_trade": float(trade_diag["median_volume_per_trade"]),
        "mean_fee_per_trade": float(trade_diag["mean_fee_per_trade"]),
        "median_fee_per_trade": float(trade_diag["median_fee_per_trade"]),
        "mean_slippage_per_trade": float(trade_diag["mean_slippage_per_trade"]),
        "median_slippage_per_trade": float(trade_diag["median_slippage_per_trade"]),
    })
    return row


def _parse_int_list(text: str) -> list[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def _parse_float_list(text: str) -> list[float]:
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def _parse_bool_list(text: str) -> list[bool]:
    values: list[bool] = []
    for raw in text.split(","):
        value = raw.strip().lower()
        if not value:
            continue
        if value in {"1", "true", "t", "yes", "y", "on"}:
            values.append(True)
        elif value in {"0", "false", "f", "no", "n", "off"}:
            values.append(False)
        else:
            raise ValueError(f"Unable to parse boolean value: {raw}")
    return values


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _compute_additional_metrics(result) -> dict[str, float]:
    returns = result.returns.astype(float)
    equity = result.equity_curve.astype(float)
    drawdown = (equity / equity.cummax()) - 1.0
    losses = returns[returns < 0]
    gains = returns[returns > 0]

    dd_duration = 0
    dd_duration_max = 0
    for v in drawdown.astype(float).tolist():
        if v < 0:
            dd_duration += 1
            dd_duration_max = max(dd_duration_max, dd_duration)
        else:
            dd_duration = 0

    downside_sq_mean = float((losses**2).mean()) if not losses.empty else 0.0
    downside_dev = np.sqrt(downside_sq_mean)
    sortino_alt = _safe_div(float(returns.mean()), downside_dev)

    ulcer_index = float(np.sqrt((drawdown.pow(2)).mean()))
    q95 = float(returns.quantile(0.95)) if not returns.empty else 0.0
    q5 = float(returns.quantile(0.05)) if not returns.empty else 0.0
    tail_ratio = _safe_div(abs(q95), abs(q5))

    gain_to_pain = _safe_div(float(gains.sum()), abs(float(losses.sum())))
    recovery_factor = _safe_div(float(equity.iloc[-1] - equity.iloc[0]), abs(float(drawdown.min()) * float(equity.iloc[0])))

    trades_df = result.trades_dataframe()
    pnl_values = trades_df["pnl"].astype(float).tolist() if not trades_df.empty else []
    win_streak = 0
    loss_streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    for pnl in pnl_values:
        if pnl > 0:
            win_streak += 1
            loss_streak = 0
        elif pnl < 0:
            loss_streak += 1
            win_streak = 0
        else:
            win_streak = 0
            loss_streak = 0
        max_win_streak = max(max_win_streak, win_streak)
        max_loss_streak = max(max_loss_streak, loss_streak)

    avg_trade_duration = float(trades_df["holding_bars"].mean()) if not trades_df.empty else 0.0
    avg_trade_return = float(trades_df["return"].mean()) if (not trades_df.empty and "return" in trades_df.columns) else 0.0
    worst_trade = float(trades_df["pnl"].min()) if not trades_df.empty else 0.0
    best_trade = float(trades_df["pnl"].max()) if not trades_df.empty else 0.0

    return {
        "ulcer_index": ulcer_index,
        "tail_ratio": tail_ratio,
        "gain_to_pain_ratio": gain_to_pain,
        "recovery_factor": recovery_factor,
        "sortino_alt": sortino_alt,
        "longest_winning_streak": float(max_win_streak),
        "longest_losing_streak": float(max_loss_streak),
        "avg_trade_duration_bars": avg_trade_duration,
        "avg_trade_return": avg_trade_return,
        "worst_trade": worst_trade,
        "largest_winning_trade": best_trade,
        "largest_drawdown_duration": float(dd_duration_max),
    }


def _compute_regime_metrics(result, close: pd.Series) -> dict[str, float]:
    aligned_close = close.reindex(result.returns.index).ffill().bfill().astype(float)
    market_returns = aligned_close.pct_change().fillna(0.0)
    strategy_returns = result.returns.astype(float)
    rolling_vol = market_returns.rolling(20, min_periods=10).std().fillna(market_returns.std())
    vol_threshold = float(rolling_vol.median()) if not rolling_vol.empty else 0.0

    regimes = {
        "bull": market_returns > 0,
        "bear": market_returns < 0,
        "high_vol": rolling_vol >= vol_threshold,
        "low_vol": rolling_vol < vol_threshold,
    }
    out: dict[str, float] = {}
    for name, mask in regimes.items():
        if mask.sum() == 0:
            out[f"regime_{name}_return"] = 0.0
            continue
        subset = strategy_returns[mask]
        out[f"regime_{name}_return"] = float((1.0 + subset).prod() - 1.0)
    return out


def _format_value(value: object, digits: int = 4) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:d}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _text_block(pdf: _SimplePdfReport, x: float, y: float, text: str, size: int = 11, line_gap: int = 14) -> float:
    for raw_line in text.splitlines():
        lines = wrap(raw_line, width=110) or [""]
        for line in lines:
            pdf.text(x, y, line, size=size)
            y -= line_gap
    return y


def _fmt_tick(value: float) -> str:
    av = abs(float(value))
    if av >= 1000:
        return f"{value:,.0f}"
    if av >= 10:
        return f"{value:,.1f}"
    return f"{value:,.3f}"




def _detect_log_growth(values: list[float]) -> bool:
    series = pd.Series(values, dtype='float64').replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 8 or (series <= 0).any():
        return False
    q10 = float(series.quantile(0.10))
    q90 = float(series.quantile(0.90))
    if q10 <= 0 or (q90 / q10) < 8.0:
        return False
    idx = np.arange(len(series), dtype='float64')
    linear_corr = abs(float(np.corrcoef(idx, series.to_numpy())[0, 1]))
    log_corr = abs(float(np.corrcoef(idx, np.log(series.to_numpy()))[0, 1]))
    return bool(np.isfinite(log_corr) and log_corr > linear_corr * 1.05)
def _chart_frame(
    pdf: _SimplePdfReport,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    caption: str,
    x_label: str,
    y_label: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_ticks: int = 6,
    y_ticks: int = 6,
    y_scale: str = "linear",
) -> tuple[float, float, float, float]:
    pdf.line(x, y, x + width, y)
    pdf.line(x, y, x, y + height)
    pdf.line(x + width, y, x + width, y + height)
    pdf.line(x, y + height, x + width, y + height)

    title_y = y + height + 22
    pdf.text(x, title_y, title[:100], size=11)
    caption_lines = wrap(caption, width=max(48, int(width / 6)))[:2]
    caption_y = title_y - 11
    for line in caption_lines:
        pdf.text(x, caption_y, line, size=7)
        caption_y -= 9

    left_pad = 48.0
    right_pad = 8.0
    top_pad = 34.0
    bottom_pad = 30.0
    ix, iy = x + left_pad, y + bottom_pad
    iw, ih = max(12.0, width - left_pad - right_pad), max(12.0, height - top_pad - bottom_pad)

    # grid
    pdf._ops.append('0.85 G 0.20 w')
    for i in range(max(2, x_ticks)):
        gx = ix + (i / max(1, x_ticks - 1)) * iw
        pdf._ops.append(f"{gx:.2f} {iy:.2f} m {gx:.2f} {iy+ih:.2f} l S")
    for i in range(max(2, y_ticks)):
        gy = iy + (i / max(1, y_ticks - 1)) * ih
        pdf._ops.append(f"{ix:.2f} {gy:.2f} m {ix+iw:.2f} {gy:.2f} l S")
    pdf._ops.append('0 G 0.40 w')
    pdf._ops.append(f"{ix:.2f} {iy:.2f} m {ix+iw:.2f} {iy:.2f} l S")
    pdf._ops.append(f"{ix:.2f} {iy:.2f} m {ix:.2f} {iy+ih:.2f} l S")

    xs = (x_max - x_min) or 1.0
    ys = (y_max - y_min) or 1.0
    for i in range(max(2, x_ticks)):
        ratio = i / max(1, x_ticks - 1)
        tx = ix + ratio * iw
        tv = x_min + ratio * xs
        pdf.text(tx - 11, iy - 13, _fmt_tick(tv), size=6)
    y_safe_min = max(float(y_min), 1e-12)
    y_safe_max = max(float(y_max), y_safe_min * 1.0000001)
    y_log_span = (np.log(y_safe_max) - np.log(y_safe_min)) or 1.0
    for i in range(max(2, y_ticks)):
        ratio = i / max(1, y_ticks - 1)
        ty = iy + ratio * ih
        if y_scale == "log":
            tv = float(np.exp(np.log(y_safe_min) + ratio * y_log_span))
        else:
            tv = y_min + ratio * ys
        pdf.text(x + 2, ty - 2, _fmt_tick(tv), size=6)
    pdf.text(ix + iw * 0.35, y + 5, x_label[:48], size=7)
    pdf.text(x + 3, iy + ih + 5, y_label[:48], size=7)
    return ix, iy, iw, ih


def _draw_bar_chart(
    pdf: _SimplePdfReport,
    x: float,
    y: float,
    width: float,
    height: float,
    labels: list[str],
    values: list[float],
    title: str,
    x_label: str = 'Category',
    y_label: str = 'Value',
    caption: str = 'Bar chart comparing category magnitudes.',
) -> None:
    if not values:
        _chart_frame(pdf, x, y, width, height, title, caption, x_label, y_label, 0.0, 1.0, 0.0, 1.0)
        pdf.text(x + 10, y + height / 2, 'No data', size=9)
        return
    ymax = max(values)
    ix, iy, iw, ih = _chart_frame(
        pdf,
        x,
        y,
        width,
        height,
        title,
        caption,
        x_label,
        y_label,
        1.0,
        float(max(1, len(values))),
        0.0,
        float(ymax if ymax != 0 else 1.0),
    )
    bar_w = iw / max(1, len(values)) * 0.7
    gap = iw / max(1, len(values)) * 0.3
    cursor = ix + gap / 2
    denom = ymax if ymax != 0 else 1.0
    step = max(1, len(labels) // 8)
    for idx, (label, val) in enumerate(zip(labels, values)):
        bar_h = (val / denom) * ih
        pdf.rect(cursor, iy, bar_w, bar_h, fill_gray=0.65)
        if idx % step == 0:
            pdf.text(cursor, iy - 11, str(label)[:10], size=6)
        cursor += bar_w + gap


def _draw_line_chart(
    pdf: _SimplePdfReport,
    x: float,
    y: float,
    width: float,
    height: float,
    values: list[float],
    title: str,
    y_label: str,
    color: tuple[float, float, float] = (0.12, 0.32, 0.72),
    x_label: str = 'Index',
    caption: str = 'Line chart showing temporal evolution of the selected metric.',
) -> None:
    if len(values) < 2:
        _chart_frame(pdf, x, y, width, height, title, caption, x_label, y_label, 1.0, 2.0, 0.0, 1.0)
        pdf.text(x + 8, y + height / 2, 'Not enough data', size=9)
        return
    y_scale = 'log' if _detect_log_growth(values) else 'linear'
    vmin, vmax = float(min(values)), float(max(values))
    ix, iy, iw, ih = _chart_frame(pdf, x, y, width, height, title, caption, x_label, y_label, 1.0, float(len(values)), vmin, vmax, y_scale=y_scale)
    if y_scale == 'log':
        safe_values = [max(float(v), 1e-12) for v in values]
        safe_min = max(vmin, 1e-12)
        safe_max = max(vmax, safe_min * 1.0000001)
        draw_values = [float(np.log(v)) for v in safe_values]
        lo = float(np.log(safe_min))
        hi = float(np.log(safe_max))
    else:
        draw_values = [float(v) for v in values]
        lo = vmin
        hi = vmax
    span = (hi - lo) or 1.0
    r, g, b = color
    pdf._ops.append(f"{r:.2f} {g:.2f} {b:.2f} RG 1.10 w")
    for i, value in enumerate(draw_values):
        px = ix + (i / (len(draw_values) - 1)) * iw
        py = iy + ((float(value) - lo) / span) * ih
        pdf._ops.append(f"{px:.2f} {py:.2f} {'m' if i == 0 else 'l'}")
    pdf._ops.append('S 0 G')


def _draw_histogram(
    pdf: _SimplePdfReport,
    x: float,
    y: float,
    width: float,
    height: float,
    values: list[float],
    bins: int,
    title: str,
    x_label: str = 'Value',
    y_label: str = 'Frequency',
    caption: str = 'Histogram summarizing the distribution shape and concentration.',
) -> None:
    if not values:
        _draw_bar_chart(pdf, x, y, width, height, [], [], title, x_label=x_label, y_label=y_label, caption=caption)
        return
    arr = np.asarray(values, dtype='float64')
    hist, edges = np.histogram(arr, bins=bins)
    y_max = float(max(1.0, np.max(hist)))
    ix, iy, iw, ih = _chart_frame(
        pdf,
        x,
        y,
        width,
        height,
        title,
        caption,
        x_label,
        y_label,
        float(edges[0]),
        float(edges[-1]),
        0.0,
        y_max,
        x_ticks=5,
        y_ticks=5,
    )
    bar_w = iw / max(1, len(hist))
    for i, count in enumerate(hist):
        bh = (float(count) / y_max) * ih
        px = ix + i * bar_w
        pdf.rect(px + 0.5, iy, max(0.5, bar_w - 1.0), bh, fill_gray=0.65)


def _draw_heatmap(
    pdf: _SimplePdfReport,
    x: float,
    y: float,
    width: float,
    height: float,
    matrix: pd.DataFrame,
    title: str,
    value_fmt: str = '{:.2f}',
    caption: str = 'Heatmap of aggregated metric values; darker blocks indicate stronger readings.',
    x_axis_label: str = 'Second Wiseman contracts',
    y_axis_label: str = 'First Wiseman contracts',
) -> None:
    pdf.line(x, y, x + width, y)
    pdf.line(x, y, x, y + height)
    pdf.line(x + width, y, x + width, y + height)
    pdf.line(x, y + height, x + width, y + height)
    pdf.text(x, y + height + 22, title, size=11)
    cap_lines = wrap(caption, width=max(44, int(width / 6)))[:2]
    cap_y = y + height + 12
    for line in cap_lines:
        pdf.text(x, cap_y, line, size=7)
        cap_y -= 8
    if matrix.empty:
        pdf.text(x + 6, y + height / 2, 'No data', size=9)
        return

    rows, cols = matrix.shape
    left_pad = 34.0
    right_pad = 18.0
    bottom_pad = 24.0
    top_pad = 10.0
    ix = x + left_pad
    iy = y + bottom_pad
    iw = max(10.0, width - left_pad - right_pad)
    ih = max(10.0, height - top_pad - bottom_pad)

    cell_w = iw / max(1, cols)
    cell_h = ih / max(1, rows)
    vals = matrix.astype(float).to_numpy()
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    span = (vmax - vmin) or 1.0

    for i in range(rows):
        for j in range(cols):
            value = float(vals[i, j])
            ratio = (value - vmin) / span
            gray = 0.92 - (0.72 * ratio)
            px = ix + j * cell_w
            py = iy + (rows - 1 - i) * cell_h
            pdf.rect(px, py, max(0.5, cell_w - 0.6), max(0.5, cell_h - 0.6), fill_gray=max(0.08, min(0.96, gray)))
            if rows <= 6 and cols <= 6:
                pdf.text(px + 2, py + (cell_h / 2) - 3, value_fmt.format(value)[:10], size=6)

    for j, col_name in enumerate(matrix.columns):
        pdf.text(ix + j * cell_w + 1, y + 6, str(col_name)[:10], size=6)
    for i, row_name in enumerate(matrix.index):
        py = iy + (rows - 1 - i) * cell_h + (cell_h / 2) - 3
        pdf.text(x + 2, py, str(row_name)[:8], size=6)

    pdf.text(ix + iw * 0.30, y - 6, x_axis_label[:50], size=7)
    pdf.text(x + 2, iy + ih + 2, y_axis_label[:50], size=7)
    pdf.text(x + width - 76, y + 8, f"Scale {value_fmt.format(vmin)} to {value_fmt.format(vmax)}", size=6)


def _draw_table(pdf: _SimplePdfReport, x: float, y: float, width: float, headers: list[str], rows: list[list[object]], title: str, row_height: int = 12, max_rows: int = 22) -> None:
    pdf.text(x, y + 22, title, size=12)
    cols = max(1, len(headers))
    col_w = width / cols
    cell_char_limit = max(8, int(col_w / 6))
    pdf.rect(x, y, width, row_height, fill_gray=0.86)
    for i, h in enumerate(headers):
        pdf.text(x + i * col_w + 2, y + 2, str(h)[:cell_char_limit], size=8)
    pdf.line(x, y, x + width, y)
    yy = y - row_height
    for row in rows[:max_rows]:
        for i, value in enumerate(row):
            pdf.text(x + i * col_w + 2, yy + 2, _format_value(value, digits=3)[:cell_char_limit], size=7)
        pdf.line(x, yy, x + width, yy, width=0.3)
        yy -= row_height


def _draw_scatter_chart(
    pdf: _SimplePdfReport,
    x: float,
    y: float,
    width: float,
    height: float,
    x_values: list[float],
    y_values: list[float],
    title: str,
    x_label: str,
    y_label: str,
    caption: str = 'Scatter chart mapping joint distribution and tradeoffs between two metrics.',
) -> None:
    if not x_values or not y_values:
        _chart_frame(pdf, x, y, width, height, title, caption, x_label, y_label, 0.0, 1.0, 0.0, 1.0)
        pdf.text(x + 4, y + 8, 'No data', size=8)
        return
    xmin, xmax = min(x_values), max(x_values)
    ymin, ymax = min(y_values), max(y_values)
    xspan = (xmax - xmin) or 1.0
    yspan = (ymax - ymin) or 1.0
    ix, iy, iw, ih = _chart_frame(pdf, x, y, width, height, title, caption, x_label, y_label, xmin, xmax, ymin, ymax)
    pdf._ops.append('0.10 0.42 0.10 rg')
    for xv, yv in zip(x_values, y_values):
        px = ix + ((xv - xmin) / xspan) * iw
        py = iy + ((yv - ymin) / yspan) * ih
        pdf._ops.append(f"{px:.2f} {py:.2f} 1.8 1.8 re f")
    pdf._ops.append('0 g')


def _efficient_frontier(points: pd.DataFrame, ret_col: str = 'total_return', dd_col: str = 'max_drawdown') -> pd.Series:
    if points.empty or ret_col not in points.columns or dd_col not in points.columns:
        return pd.Series(dtype='bool')
    risk = points[dd_col].abs().astype(float)
    ret = points[ret_col].astype(float)
    frontier = []
    for i in range(len(points)):
        dominated = ((risk <= risk.iloc[i]) & (ret >= ret.iloc[i]) & ((risk < risk.iloc[i]) | (ret > ret.iloc[i]))).any()
        frontier.append(not dominated)
    return pd.Series(frontier, index=points.index)


def _neighbor_table(df: pd.DataFrame, center_idx: int, param_cols: list[str], metric: str, top_k: int = 12) -> pd.DataFrame:
    if not param_cols:
        return pd.DataFrame()
    base = df.loc[center_idx, param_cols].astype(float)
    work = df.copy()
    for col in param_cols:
        work[f'd_{col}'] = (work[col].astype(float) - float(base[col])).abs()
    work['distance'] = work[[f'd_{c}' for c in param_cols]].sum(axis=1)
    cols = param_cols + [metric, 'sharpe', 'max_drawdown', 'distance']
    cols = [c for c in cols if c in work.columns]
    return work.sort_values('distance').head(top_k)[cols]


def _make_pdf_report(
    out_path: Path,
    leaderboard: pd.DataFrame,
    sort_by: str,
    top_n: int,
    cli_args: argparse.Namespace,
    total_grid_size: int,
    best_result,
    benchmark_equity: pd.Series,
) -> None:
    pdf = _SimplePdfReport()
    work = leaderboard.copy()
    best = work.iloc[0]

    # Composite diagnostics for institutional ranking.
    for col, default in [("total_return", 0.0), ("sharpe", 0.0), ("calmar", 0.0), ("max_drawdown", 0.0), ("trades", 0.0), ("return_over_maxdd", 0.0)]:
        if col not in work.columns:
            work[col] = default
    risk = work["max_drawdown"].abs().replace(0, np.nan)
    work["return_drawdown"] = work["total_return"] / risk
    work["robustness_score"] = (
        work["sharpe"].rank(pct=True)
        + work["calmar"].rank(pct=True)
        + work["return_drawdown"].fillna(0.0).rank(pct=True)
        + (1.0 - work["max_drawdown"].abs().rank(pct=True))
        + work["trades"].rank(pct=True)
    ) / 5.0
    work["overfit_risk"] = (work["total_return"].rank(pct=True) - work["robustness_score"]).clip(lower=0.0)
    work["abs_max_drawdown"] = work["max_drawdown"].abs()

    best_return = work.sort_values("total_return", ascending=False).iloc[0]
    best_risk_adj = work.sort_values("sharpe", ascending=False).iloc[0]
    best_robust = work.sort_values("robustness_score", ascending=False).iloc[0]
    lowest_dd = work.sort_values("max_drawdown", ascending=False).head(3)

    param_cols = [
        "first_wiseman_contracts",
        "second_wiseman_contracts",
        "third_wiseman_contracts",
        "reversal_contracts_mult",
        "gator_width_lookback",
        "gator_width_mult",
        "gator_width_valid_factor",
        "profit_protection_enabled",
        "profit_protection_min_bars",
        "profit_protection_min_unrealized_return",
        "profit_protection_credit_unrealized_before_min_bars",
        "profit_protection_require_gator_open",
    ]
    param_cols = [c for c in param_cols if c in work.columns]

    # Executive summary
    pdf.new_page()
    pdf.text(40, 560, "Institutional Parameter Sweep Research Report", size=18)
    y = 538
    start_dt = str(best_result.equity_curve.index.min()) if len(best_result.equity_curve.index) else "n/a"
    end_dt = str(best_result.equity_curve.index.max()) if len(best_result.equity_curve.index) else "n/a"
    summary_lines = [
        f"Dataset: {cli_args.csv}",
        f"Backtest horizon: {start_dt} -> {end_dt}",
        f"Total trades analyzed: {int(work['trades'].sum())}",
        f"Parameter combinations tested: {len(work)}",
        f"Best performing set (return): {best_return.get('timeframe','n/a')} | return={best_return['total_return']:.2%} | max drawdown={best_return['max_drawdown']:.2%}",
        f"Best risk-adjusted set (Sharpe): {best_risk_adj.get('timeframe','n/a')} | Sharpe={best_risk_adj['sharpe']:.3f}",
        f"Most robust set: score={best_robust['robustness_score']:.3f} | return={best_robust['total_return']:.2%} | max drawdown={best_robust['max_drawdown']:.2%}",
        f"Lowest drawdown sets (percent): {', '.join(f'{v:.2%}' for v in lowest_dd['max_drawdown'].tolist())}",
        "Tradeoff: highest returns generally coincide with deeper drawdowns and lower breadth of stability.",
        "Interpretation: the strategy appears deployable only in robust parameter clusters, not at single-point optima.",
    ]
    for line in summary_lines:
        pdf.text(40, y, line, size=10)
        y -= 14

    # Parameter sweep overview + parameter ranges table
    pdf.new_page()
    pdf.text(40, 560, "1) Parameter Sweep Overview", size=16)
    y = 540
    overview = [
        f"Sampling method: grid search across {total_grid_size} combinations per timeframe.",
        f"Timeframes: {cli_args.timeframes}",
        f"Data period used: {start_dt} -> {end_dt}",
        f"Assumptions: fee={cli_args.fee:.4%}, slippage={cli_args.slippage:.4%}, spread={cli_args.spread:.4%}, size_mode={cli_args.size_mode}, size_value={cli_args.size_value}",
        f"Sizing extras: size_min_usd={cli_args.size_min_usd}, vol_target_annual={cli_args.volatility_target_annual}, vol_lookback={cli_args.volatility_lookback}, vol_scale_range=[{cli_args.volatility_min_scale}, {cli_args.volatility_max_scale}]",
    ]
    for line in overview:
        pdf.text(40, y, line, size=10)
        y -= 14

    table_rows = []
    for c in param_cols:
        vals = sorted(work[c].dropna().unique().tolist())
        if not vals:
            continue
        step = vals[1] - vals[0] if len(vals) >= 2 and isinstance(vals[0], (int, float, np.integer, np.floating)) else "n/a"
        table_rows.append([c, vals[0], vals[-1], step, ",".join(str(v) for v in vals[:8]) + ("..." if len(vals) > 8 else "")])
    _draw_table(pdf, 40, 430, 760, ["parameter", "min", "max", "step", "tested values"], table_rows, "Parameter range matrix", row_height=14, max_rows=20)

    # Core results table
    pdf.new_page()
    pdf.text(40, 560, "2) Core Results Table (sortable in CSV)", size=16)
    core_cols = [
        "timeframe",
        "final_equity",
        "total_return",
        "cagr",
        "max_drawdown",
        "return_drawdown",
        "sharpe",
        "sortino",
        "calmar",
        "profit_factor",
        "win_rate",
        "trades",
        "avg_trade_return",
        "worst_trade",
        "largest_drawdown_duration",
        "robustness_score",
    ]
    core_cols = [c for c in core_cols if c in work.columns]
    rows = [[r[c] for c in core_cols] for _, r in work.sort_values(sort_by, ascending=False).head(18).iterrows()]
    _draw_table(pdf, 40, 500, 760, core_cols, rows, "Top rows from full results universe", row_height=12, max_rows=18)
    pdf.text(40, 70, "Highlights: best return, best Sharpe, best Calmar, lowest drawdown, highest robustness are explicitly tracked in text sections.", size=9)

    # Top strategy configurations pages
    categories = [
        ("Top 10 by Return", "total_return", False),
        ("Top 10 by Sharpe", "sharpe", False),
        ("Top 10 by Calmar", "calmar", False),
        ("Top 10 lowest Drawdown", "max_drawdown", True),
        ("Top 10 by Smallest Absolute Drawdown", "abs_max_drawdown", True),
        ("Top 10 by Return/Drawdown", "return_drawdown", False),
    ]
    for title, metric, asc in categories:
        if metric not in work.columns:
            continue
        pdf.new_page()
        pdf.text(40, 560, f"3) {title}", size=16)
        slice_df = work.sort_values(metric, ascending=asc).head(10)
        cols = [c for c in ["timeframe", metric, "total_return", "sharpe", "calmar", "max_drawdown", "trades", "robustness_score"] if c in slice_df.columns]
        rows = [[r[c] for c in cols] for _, r in slice_df.iterrows()]
        _draw_table(pdf, 40, 500, 760, cols, rows, f"{title} table", row_height=13, max_rows=12)

        yy = 120
        for rank, (_, leader) in enumerate(slice_df.iterrows(), start=1):
            line = (
                f"{rank}. TF={leader.get('timeframe', 'n/a')} | "
                f"(1W/2W/3W): {int(leader.get('first_wiseman_contracts', 0))}/{int(leader.get('second_wiseman_contracts', 0))}/{int(leader.get('third_wiseman_contracts', 0))} | "
                f"gator width mult={float(leader.get('gator_width_mult', 0.0)):.2f}, valid-factor={float(leader.get('gator_width_valid_factor', 0.0)):.2f}, lookback={int(leader.get('gator_width_lookback', 0))} | "
                f"Protection: enabled={bool(leader.get('profit_protection_enabled', False))}, min bars={int(leader.get('profit_protection_min_bars', 0))}, min unrealized return={float(leader.get('profit_protection_min_unrealized_return', 0.0)):.2%}, credit unrealized before min bars={bool(leader.get('profit_protection_credit_unrealized_before_min_bars', False))}, require gator open={bool(leader.get('profit_protection_require_gator_open', True))} | "
                f"total return={float(leader.get('total_return', 0.0)):.2%}, max DD={float(leader.get('max_drawdown', 0.0)):.2%}, trades={int(float(leader.get('trades', 0.0)))}"
            )
            pdf.text(40, yy, line, size=8)
            yy -= 10
            if yy < 20:
                break

    # Robustness and heatmaps
    pdf.new_page()
    pdf.text(40, 560, "4) Robustness Analysis", size=16)
    pivot_a = work.pivot_table(index="first_wiseman_contracts", columns="second_wiseman_contracts", values="total_return", aggfunc="median") if {"first_wiseman_contracts", "second_wiseman_contracts", "total_return"}.issubset(work.columns) else pd.DataFrame()
    pivot_b = work.pivot_table(index="first_wiseman_contracts", columns="second_wiseman_contracts", values="sharpe", aggfunc="median") if {"first_wiseman_contracts", "second_wiseman_contracts", "sharpe"}.issubset(work.columns) else pd.DataFrame()
    pivot_c = work.pivot_table(index="first_wiseman_contracts", columns="second_wiseman_contracts", values="max_drawdown", aggfunc="median") if {"first_wiseman_contracts", "second_wiseman_contracts", "max_drawdown"}.issubset(work.columns) else pd.DataFrame()
    pivot_d = work.pivot_table(index="first_wiseman_contracts", columns="second_wiseman_contracts", values="return_drawdown", aggfunc="median") if {"first_wiseman_contracts", "second_wiseman_contracts", "return_drawdown"}.issubset(work.columns) else pd.DataFrame()

    _draw_heatmap(
        pdf,
        40,
        250,
        360,
        230,
        pivot_a * 100.0 if not pivot_a.empty else pivot_a,
        "Return stability heatmap",
        value_fmt='{:.1f}%',
        caption="Median total return (%) across 1W/2W contract combinations. Prefer broad positive zones over isolated outliers.",
    )
    _draw_heatmap(
        pdf,
        430,
        250,
        360,
        230,
        pivot_b,
        "Sharpe stability heatmap",
        value_fmt='{:.2f}',
        caption="Median Sharpe by 1W/2W contracts. Larger contiguous areas indicate stronger risk-adjusted stability.",
    )
    pdf.text(40, 220, "Interpretation: rows are first Wiseman contracts (1W), columns are second Wiseman contracts (2W).", size=8)

    pdf.new_page()
    pdf.text(40, 560, "4) Robustness Analysis (drawdown and efficiency)", size=16)
    _draw_heatmap(
        pdf,
        40,
        250,
        360,
        230,
        pivot_c * 100.0 if not pivot_c.empty else pivot_c,
        "Max drawdown heatmap",
        value_fmt='{:.1f}%',
        caption="Median max drawdown (%) by 1W/2W contracts. Prefer shallower drawdown cells with consistent neighboring values.",
    )
    _draw_heatmap(
        pdf,
        430,
        250,
        360,
        230,
        pivot_d,
        "Return/Drawdown heatmap",
        value_fmt='{:.2f}',
        caption="Median return per unit drawdown by 1W/2W contracts. Higher values with stability are preferred.",
    )

    neigh = _neighbor_table(work, best_return.name, [c for c in ["first_wiseman_contracts", "second_wiseman_contracts", "gator_width_mult"] if c in work.columns], "total_return")
    collapse_ratio = float((neigh["total_return"] < 0).mean()) if "total_return" in neigh.columns and not neigh.empty else 0.0
    pdf.text(40, 220, f"Sensitivity note: {collapse_ratio*100:.1f}% of nearest-neighbor configs lose money. High values imply narrow overfit regions.", size=9)

    # Distribution + frontier + clustering
    pdf.new_page()
    pdf.text(40, 560, "5) Risk Distribution, Efficiency Frontier, and Clustering", size=16)
    _draw_histogram(pdf, 40, 300, 360, 190, (work["cagr"].astype(float) * 100.0).tolist() if "cagr" in work.columns else [], 12, "CAGR distribution", x_label="CAGR (%)", y_label="Count", caption="Distribution of annualized return outcomes. Focus on median and downside tails, not only extremes.")
    _draw_histogram(pdf, 430, 300, 360, 190, (work["max_drawdown"].astype(float) * 100.0).tolist() if "max_drawdown" in work.columns else [], 12, "Max drawdown distribution", x_label="Max drawdown (%)", y_label="Count", caption="Distribution of peak-to-trough losses. This estimates capital impairment risk concentration.")
    _draw_histogram(pdf, 40, 70, 360, 190, work["sharpe"].astype(float).tolist() if "sharpe" in work.columns else [], 12, "Sharpe distribution", x_label="Sharpe ratio", y_label="Count", caption="Distribution of risk-adjusted returns across all tested parameter sets.")
    _draw_histogram(pdf, 430, 70, 360, 190, work["return_drawdown"].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float).tolist(), 12, "Return/DD distribution", x_label="Return per unit drawdown", y_label="Count", caption="Efficiency ratio dispersion; higher and stable values generally indicate more deployable profiles.")

    pdf.new_page()
    pdf.text(40, 560, "5) Efficiency Frontier and Cluster Summary", size=16)
    frontier_mask = _efficient_frontier(work)
    frontier = work[frontier_mask]
    dominated = work[~frontier_mask]
    _draw_scatter_chart(
        pdf,
        40,
        250,
        760,
        250,
        (dominated["max_drawdown"].abs().astype(float) * 100.0).tolist(),
        (dominated["total_return"].astype(float) * 100.0).tolist(),
        "Return vs Drawdown (efficient frontier context)",
        "|Max Drawdown| (%)",
        "Total Return (%)",
        caption="Each dot is a dominated configuration. Frontier candidates should combine higher return with lower drawdown.",
    )
    pdf.text(40, 228, f"Efficient frontier count: {len(frontier)} | Dominated configurations: {len(dominated)}", size=9)

    if {"sharpe", "max_drawdown", "total_return"}.issubset(work.columns):
        cluster_df = work[["sharpe", "max_drawdown", "total_return"]].astype(float).copy()
        cluster_df["cluster"] = pd.qcut(cluster_df["sharpe"].rank(method="first"), q=min(3, len(cluster_df)), labels=False, duplicates="drop")
        y_cluster = 208
        for cid in sorted(cluster_df["cluster"].dropna().unique().tolist()):
            subset = work.loc[cluster_df[cluster_df["cluster"] == cid].index]
            pdf.text(y= y_cluster, x=40, value=f"Cluster {int(cid)}: n={len(subset)}, mean Sharpe={subset['sharpe'].mean():.2f}, mean max DD={subset['max_drawdown'].mean():.2%}", size=9)
            y_cluster -= 14

    # Trade diagnostics, stability across time, and Monte Carlo cross-validation
    pdf.new_page()
    pdf.text(40, 560, "6) Trade Diagnostics and Benchmark Context", size=16)
    trades_df = best_result.trades_dataframe()
    trade_lines = [
        f"Average trade PnL ($): {(float(trades_df['pnl'].mean()) if not trades_df.empty else 0.0):,.2f}",
        f"Median trade PnL ($): {(float(trades_df['pnl'].median()) if not trades_df.empty else 0.0):,.2f}",
        f"Worst trade ($): {(float(trades_df['pnl'].min()) if not trades_df.empty else 0.0):,.2f}",
        f"Largest winning trade ($): {(float(trades_df['pnl'].max()) if not trades_df.empty else 0.0):,.2f}",
        f"Win rate: {float(best.get('win_rate', 0.0)):.2%}",
        f"Payoff ratio (avg win/avg loss): {_safe_div(float(trades_df[trades_df['pnl']>0]['pnl'].mean()) if not trades_df.empty else 0.0, abs(float(trades_df[trades_df['pnl']<0]['pnl'].mean())) if not trades_df.empty and (trades_df['pnl']<0).any() else 0.0):.3f}",
    ]
    yy = 538
    for line in trade_lines:
        pdf.text(40, yy, line, size=10)
        yy -= 13

    _draw_histogram(pdf, 40, 250, 360, 220, trades_df["pnl"].astype(float).tolist() if not trades_df.empty else [], 14, "Trade PnL distribution", x_label="Trade PnL ($)", y_label="Count", caption="Distribution of individual trade outcomes for the selected reference configuration.")
    _draw_line_chart(pdf, 430, 250, 360, 220, benchmark_equity.astype(float).tolist(), "Benchmark equity path", "Equity ($)", color=(0.15, 0.30, 0.75), x_label="Bar index", caption="Buy-and-hold benchmark path used as baseline context against strategy behavior.")

    pdf.new_page()
    pdf.text(40, 560, "6) Volume / Fees / Slippage Turnover", size=16)
    total_cumulative_volume_sum = float(work["total_cumulative_volume"].astype(float).sum()) if "total_cumulative_volume" in work.columns else 0.0
    total_cumulative_fees_sum = float(work["total_cumulative_fees"].astype(float).sum()) if "total_cumulative_fees" in work.columns else 0.0
    total_cumulative_slippage_sum = float(work["total_cumulative_slippage"].astype(float).sum()) if "total_cumulative_slippage" in work.columns else 0.0
    volume_lines = [
        f"Total cumulative volume (all runs): {total_cumulative_volume_sum:,.2f}",
        f"Total cumulative fees (all runs): {total_cumulative_fees_sum:,.2f}",
        f"Total cumulative slippage (all runs): {total_cumulative_slippage_sum:,.2f}",
        f"Mean/median volume per trade: {float(work['mean_volume_per_trade'].mean()) if 'mean_volume_per_trade' in work.columns else 0.0:,.2f} / {float(work['median_volume_per_trade'].median()) if 'median_volume_per_trade' in work.columns else 0.0:,.2f}",
        f"Mean/median fee per trade: {float(work['mean_fee_per_trade'].mean()) if 'mean_fee_per_trade' in work.columns else 0.0:,.2f} / {float(work['median_fee_per_trade'].median()) if 'median_fee_per_trade' in work.columns else 0.0:,.2f}",
        f"Mean/median slippage per trade: {float(work['mean_slippage_per_trade'].mean()) if 'mean_slippage_per_trade' in work.columns else 0.0:,.2f} / {float(work['median_slippage_per_trade'].median()) if 'median_slippage_per_trade' in work.columns else 0.0:,.2f}",
    ]
    yv = 538
    for line in volume_lines:
        pdf.text(40, yv, line, size=10)
        yv -= 13

    _draw_line_chart(pdf, 40, 250, 360, 220, work["total_cumulative_volume"].astype(float).cumsum().tolist() if "total_cumulative_volume" in work.columns else [], "Cumulative volume across ranked sets", "USD", color=(0.12, 0.35, 0.82), x_label="Ranked configuration", caption="Running turnover by ranked parameter set.")
    _draw_line_chart(pdf, 430, 250, 360, 220, work["total_cumulative_fees"].astype(float).cumsum().tolist() if "total_cumulative_fees" in work.columns else [], "Cumulative fees across ranked sets", "USD", color=(0.78, 0.42, 0.10), x_label="Ranked configuration", caption="Running fees by ranked parameter set.")
    _draw_line_chart(pdf, 40, 40, 360, 180, work["total_cumulative_slippage"].astype(float).cumsum().tolist() if "total_cumulative_slippage" in work.columns else [], "Cumulative slippage across ranked sets", "USD", color=(0.80, 0.12, 0.12), x_label="Ranked configuration", caption="Running slippage by ranked parameter set.")
    _draw_scatter_chart(pdf, 430, 40, 360, 180, work["mean_volume_per_trade"].astype(float).tolist() if "mean_volume_per_trade" in work.columns else [], work["total_return"].astype(float).tolist(), "Mean volume per trade vs return", "Mean volume/trade", "Total return", caption="Assess whether turnover intensity is associated with higher returns.")

    pdf.new_page()
    pdf.text(40, 560, "7) Time Stability and Monte Carlo Cross-Validation", size=16)
    if "timeframe" in work.columns:
        tf_stats = work.groupby("timeframe")[[c for c in ["total_return", "sharpe", "max_drawdown"] if c in work.columns]].mean().reset_index()
        cols = tf_stats.columns.tolist()
        _draw_table(pdf, 40, 470, 760, cols, tf_stats.values.tolist(), "Stability across timeframes (mean metrics)", row_height=13, max_rows=10)

    mc_lines: list[str] = []
    mc_candidates = [best_return, best_robust]
    for idx, row in enumerate(mc_candidates, start=1):
        tf = str(row.get("timeframe", best.get("timeframe", "1h")))
        if tf not in [v.strip() for v in str(cli_args.timeframes).split(',') if v.strip()]:
            continue
        data_full = load_ohlcv_csv(cli_args.csv)
        tf_data = resample_ohlcv(data_full, tf)
        strat = _build_strategy_from_row(row)
        bt = BacktestEngine(
            BacktestConfig(
                initial_capital=cli_args.capital,
                fee_rate=cli_args.fee,
                slippage_rate=cli_args.slippage,
                spread_rate=cli_args.spread,
                order_type=cli_args.order_type,
                trade_size_mode=cli_args.size_mode,
                trade_size_value=cli_args.size_value,
                trade_size_min_usd=cli_args.size_min_usd,
                trade_size_equity_milestones=parse_trade_size_equity_milestones(cli_args.size_equity_milestones),
                volatility_target_annual=cli_args.volatility_target_annual,
                volatility_lookback=cli_args.volatility_lookback,
                volatility_min_scale=cli_args.volatility_min_scale,
                volatility_max_scale=cli_args.volatility_max_scale,
                max_leverage=cli_args.max_leverage,
                max_position_size=cli_args.max_position_size,
                leverage_stop_out_pct=cli_args.leverage_stop_out,
                borrow_rate_annual=cli_args.borrow_annual,
                funding_rate_per_period=cli_args.funding_per_period,
                overnight_rate_annual=cli_args.overnight_annual,
                max_loss=cli_args.max_loss,
                max_loss_pct_of_equity=cli_args.max_loss_pct_of_equity,
                equity_cutoff=cli_args.equity_cutoff,
            )
        ).run(tf_data, strat)
        mc = run_return_bootstrap_monte_carlo(
            bt.returns,
            initial_capital=cli_args.capital,
            simulations=250,
            horizon_bars=min(500, len(bt.returns)),
            seed=42,
            threads=1,
            equity_cutoff=cli_args.equity_cutoff,
        )
        mc_lines.append(
            f"MC #{idx} ({tf}): expected return={mc.summary['expected_return']:.2%}, median return={mc.summary['return_median']:.2%}, P(loss)={(1-mc.summary['probability_profit']):.2%}, P(max DD < -30%)={mc.summary['probability_drawdown_worse_than_30pct']:.2%}"
        )

    y_mc = 260
    pdf.text(40, y_mc + 24, "Monte Carlo validation for top return and top robustness configurations:", size=10)
    for line in mc_lines[:4]:
        pdf.text(40, y_mc, line, size=9)
        y_mc -= 14
    # CLI settings used for this run
    cli_items = sorted(vars(cli_args).items(), key=lambda kv: kv[0])
    cli_rows = [[f"--{k.replace('_','-')}", v] for k, v in cli_items]
    per_page = 34
    for page_idx, start in enumerate(range(0, len(cli_rows), per_page), start=1):
        pdf.new_page()
        title_suffix = "" if len(cli_rows) <= per_page else f" (page {page_idx})"
        pdf.text(40, 560, f"7) Full Run Settings (CLI Flags){title_suffix}", size=16)
        _draw_table(
            pdf,
            40,
            510,
            760,
            ["flag", "value"],
            cli_rows[start:start + per_page],
            "Complete CLI argument set used to generate this report",
            row_height=12,
            max_rows=per_page,
        )

    # Final recommendation + constraints + warning flags
    pdf.new_page()
    pdf.text(40, 560, "8) Final Analyst Recommendation", size=16)
    dd_abs = work["max_drawdown"].abs().replace(0, np.nan)
    cap10 = (float(cli_args.capital) * dd_abs / 0.10).replace([np.inf, -np.inf], np.nan).median()
    cap20 = (float(cli_args.capital) * dd_abs / 0.20).replace([np.inf, -np.inf], np.nan).median()
    cap30 = (float(cli_args.capital) * dd_abs / 0.30).replace([np.inf, -np.inf], np.nan).median()

    low_trade = int((work["trades"] < 30).sum())
    high_dd = int((work["max_drawdown"] < -0.35).sum())
    high_var = int((work["overfit_risk"] > 0.5).sum())
    fragile = int((work["robustness_score"] < 0.35).sum())

    final_lines = [
        f"Most promising configurations: return leader at {best_return['total_return']:.2%}, robust leader score {best_robust['robustness_score']:.3f}.",
        f"Best risk-adjusted strategy: Sharpe leader {best_risk_adj['sharpe']:.3f} with max drawdown {best_risk_adj['max_drawdown']:.2%}.",
        f"Safest strategy candidates are from lowest drawdown set (best max drawdown {lowest_dd.iloc[0]['max_drawdown']:.2%}).",
        "Recommended ranges: prioritize neighborhoods around robust leader where neighbors preserve positive return and acceptable drawdown.",
        "Live-trading readiness: conditionally positive, but requires out-of-sample walk-forward and live slippage stress validation.",
        f"Capital realism (median required capital for DD caps): 10% cap=${(float(cap10) if pd.notna(cap10) else 0.0):,.2f}, 20% cap=${(float(cap20) if pd.notna(cap20) else 0.0):,.2f}, 30% cap=${(float(cap30) if pd.notna(cap30) else 0.0):,.2f}.",
        f"Warning flags: low-trade={low_trade}, high-drawdown={high_dd}, high-overfit-risk={high_var}, fragile-robustness={fragile}.",
    ]
    y = 536
    for line in final_lines:
        y = _text_block(pdf, 40, y, line, size=10, line_gap=13) - 4

    pdf.text(40, 86, "Conclusion: deploy from robust clusters, not isolated winners; keep position sizing tied to drawdown tolerance and monitor drift continuously.", size=10)
    pdf.finalize(out_path)


def _build_strategy_from_row(row: pd.Series) -> WisemanStrategy:
    return WisemanStrategy(
        first_wiseman_contracts=int(row["first_wiseman_contracts"]),
        second_wiseman_contracts=int(row["second_wiseman_contracts"]),
        third_wiseman_contracts=int(row["third_wiseman_contracts"]),
        reversal_contracts_mult=float(row["reversal_contracts_mult"]),
        gator_width_lookback=int(row["gator_width_lookback"]),
        gator_width_mult=float(row["gator_width_mult"]),
        gator_width_valid_factor=float(row["gator_width_valid_factor"]),
        first_wiseman_wait_bars_to_close=int(row.get("first_wiseman_wait_bars_to_close", 0)),
        first_wiseman_divergence_filter_bars=int(row.get("first_wiseman_divergence_filter_bars", 0)),
        gator_direction_mode=int(row.get("gator_direction_mode", 1)),
        first_wiseman_reversal_cooldown=int(row.get("first_wiseman_reversal_cooldown", 0)),
        cancel_reversal_on_first_wiseman_exit=bool(row.get("cancel_reversal_on_first_wiseman_exit", False)),
        teeth_profit_protection_enabled=bool(row["profit_protection_enabled"]),
        teeth_profit_protection_min_bars=int(row["profit_protection_min_bars"]),
        teeth_profit_protection_min_unrealized_return=float(row["profit_protection_min_unrealized_return"]),
        profit_protection_volatility_lookback=int(row.get("profit_protection_volatility_lookback", 20)),
        profit_protection_annualized_volatility_scaler=float(row.get("profit_protection_annualized_volatility_scaler", 1.0)),
        teeth_profit_protection_credit_unrealized_before_min_bars=bool(row.get("profit_protection_credit_unrealized_before_min_bars", False)),
        teeth_profit_protection_require_gator_open=bool(row.get("profit_protection_require_gator_open", True)),
        zone_profit_protection_enabled=bool(row.get("zone_profit_protection_enabled", False)),
        zone_profit_protection_min_unrealized_return=float(row.get("zone_profit_protection_min_unrealized_return", 1.0)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grid-search Wiseman parameters and rank setups by profitability."
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--timeframes", default="auto", help="Comma-separated list like 5m,1h,4h,1d,1w. Use auto to run at detected source cadence.")
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
    parser.add_argument("--equity-cutoff", type=float, default=None, help="Stop each sweep/MC leg when equity falls to this value")
    parser.add_argument("--first-contracts", default="1,2")
    parser.add_argument("--second-contracts", default="2,3,5")
    parser.add_argument("--third-contracts", default="3,5,8")
    parser.add_argument("--reversal-contracts-mult", default="0.0,0.5,1.0")
    parser.add_argument("--gator-width-lookback", default="34,50,89")
    parser.add_argument("--gator-width-mult", default="0.75,1.0,1.25")
    parser.add_argument("--gator-width-valid-factor", default="0.75,1.0,1.5")
    parser.add_argument("--1W-wait-bars-to-close", dest="wiseman_1w_wait_bars_to_close", type=int, default=0)
    parser.add_argument("--1W-divergence-filter", dest="wiseman_1w_divergence_filter_bars", type=int, default=0)
    parser.add_argument("--wiseman-reversal-cooldown", type=int, default=0)
    parser.add_argument("--wiseman-gator-direction-mode", default="1")
    parser.add_argument("--wiseman-cancel-reversal-on-first-exit", default="false,true")
    parser.add_argument("--profit-protection-enabled", default="false,true")
    parser.add_argument("--profit-protection-min-bars", default="3,4,5")
    parser.add_argument("--profit-protection-min-unrealized-return", default="0.005,0.01,0.02")
    parser.add_argument("--wiseman-profit-protection-volatility-lookback", default="20")
    parser.add_argument("--wiseman-profit-protection-annualized-volatility-scaler", default="1.0")
    parser.add_argument(
        "--profit-protection-credit-unrealized-before-min-bars",
        "--wiseman-profit-protection-credit-unrealized-before-min-bars",
        dest="profit_protection_credit_unrealized_before_min_bars",
        default="false,true",
    )
    parser.add_argument("--profit-protection-require-gator-open", default="true,false")
    parser.add_argument("--wiseman-profit-protection-zones-exit", default="false")
    parser.add_argument("--wiseman-profit-protection-zones-min-unrealized-return", default="1.0")
    parser.add_argument("--sort-by", default="total_return", choices=["total_return", "cagr", "sharpe", "profit_factor"])
    parser.add_argument("--top", type=int, default=15, help="Top N rows to print to stdout")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes to use per timeframe")
    parser.add_argument("--out", default="artifacts_sweep")
    parser.add_argument("--start", default=None, help="Inclusive start date/time (e.g. 2024-01-01)")
    parser.add_argument("--end", default=None, help="Inclusive end date/time (e.g. 2025-12-12)")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    first_contracts = _parse_int_list(args.first_contracts)
    second_contracts = _parse_int_list(args.second_contracts)
    third_contracts = _parse_int_list(args.third_contracts)
    reversal_mults = _parse_float_list(args.reversal_contracts_mult)
    gator_lookbacks = _parse_int_list(args.gator_width_lookback)
    gator_mults = _parse_float_list(args.gator_width_mult)
    gator_valid_factors = _parse_float_list(args.gator_width_valid_factor)
    gator_direction_modes = _parse_int_list(args.wiseman_gator_direction_mode)
    if any(mode not in {1, 2, 3} for mode in gator_direction_modes):
        raise ValueError("--wiseman-gator-direction-mode values must be in {1,2,3}")
    cancel_reversal_on_first_exit = _parse_bool_list(args.wiseman_cancel_reversal_on_first_exit)
    profit_protection_enabled = _parse_bool_list(args.profit_protection_enabled)
    profit_protection_min_bars = _parse_int_list(args.profit_protection_min_bars)
    profit_protection_min_unrealized_return = _parse_float_list(args.profit_protection_min_unrealized_return)
    profit_protection_volatility_lookback = _parse_int_list(args.wiseman_profit_protection_volatility_lookback)
    profit_protection_annualized_volatility_scaler = _parse_float_list(args.wiseman_profit_protection_annualized_volatility_scaler)
    profit_protection_credit_unrealized_before_min_bars = _parse_bool_list(args.profit_protection_credit_unrealized_before_min_bars)
    profit_protection_require_gator_open = _parse_bool_list(args.profit_protection_require_gator_open)
    zone_profit_protection_enabled = _parse_bool_list(args.wiseman_profit_protection_zones_exit)
    zone_profit_protection_min_unrealized_return = _parse_float_list(args.wiseman_profit_protection_zones_min_unrealized_return)
    timeframes = [v.strip() for v in args.timeframes.split(",") if v.strip()]

    grid = list(
        itertools.product(
            first_contracts,
            second_contracts,
            third_contracts,
            reversal_mults,
            gator_lookbacks,
            gator_mults,
            gator_valid_factors,
            [args.wiseman_1w_wait_bars_to_close],
            [args.wiseman_1w_divergence_filter_bars],
            gator_direction_modes,
            [args.wiseman_reversal_cooldown],
            cancel_reversal_on_first_exit,
            profit_protection_enabled,
            profit_protection_min_bars,
            profit_protection_min_unrealized_return,
            profit_protection_volatility_lookback,
            profit_protection_annualized_volatility_scaler,
            profit_protection_credit_unrealized_before_min_bars,
            profit_protection_require_gator_open,
            zone_profit_protection_enabled,
            zone_profit_protection_min_unrealized_return,
        )
    )
    data = load_ohlcv_csv(args.csv)
    data = filter_ohlcv_by_date(data, start=args.start, end=args.end)
    if len(data) < 2:
        raise ValueError("Filtered dataset has fewer than 2 bars; widen --start/--end range")
    source_tf = infer_source_timeframe_label(data.index)
    print(f"Detected source bar interval: {source_tf}")
    if args.timeframes.strip().lower() == "auto":
        if source_tf == "unknown":
            raise ValueError(
                "Unable to infer source timeframe from OHLCV index. "
                "Pass --timeframes explicitly (for example: 5m,15m,1h)."
            )
        timeframes = [source_tf]
        print(f"Auto-selected timeframe(s): {','.join(timeframes)}")
    total_runs = len(grid) * len(timeframes)
    print(f"Running {total_runs} parameter combinations across {len(timeframes)} timeframe(s)")

    if args.start or args.end:
        print(f"Date range used: {data.index[0].isoformat()} -> {data.index[-1].isoformat()}")
    workers = max(1, args.workers)
    cpu_count = os.cpu_count() or 1
    if workers > cpu_count:
        print(f"Requested workers ({workers}) exceeds CPU count ({cpu_count}); using {cpu_count}")
        workers = cpu_count

    engine_config = BacktestConfig(
        initial_capital=args.capital,
        fee_rate=args.fee,
        slippage_rate=args.slippage,
        spread_rate=args.spread,
        order_type=args.order_type,
        trade_size_mode=args.size_mode,
        trade_size_value=args.size_value,
        trade_size_min_usd=args.size_min_usd,
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

    rows: list[dict[str, float | int | str | bool]] = []

    normalized_source_tf = normalize_timeframe(source_tf) if source_tf != "unknown" else source_tf

    if grid and timeframes:
        sample_timeframe = timeframes[0]
        normalized_sample_tf = normalize_timeframe(sample_timeframe)
        if source_tf != "unknown" and normalized_sample_tf == normalized_source_tf:
            sample_data = data.copy()
        else:
            sample_data = resample_ohlcv(data, sample_timeframe)

        if len(sample_data) >= 2:
            _init_worker(sample_data, engine_config)
            estimate_started = time.perf_counter()
            _run_combo(grid[0])
            estimate_elapsed = time.perf_counter() - estimate_started
            efficiency = 0.85 if workers > 1 else 1.0
            estimated_total_seconds = (estimate_elapsed * total_runs) / max(1.0, workers * efficiency)
            print(
                "Sweep runtime estimate: "
                f"~{_format_eta(estimated_total_seconds)} "
                f"(pilot 1 combo in {estimate_elapsed:.2f}s, workers={workers}, total runs={total_runs:,})"
            )

    for timeframe in timeframes:
        normalized_target_tf = normalize_timeframe(timeframe)
        if source_tf != "unknown" and normalized_target_tf == normalized_source_tf:
            tf_data = data.copy()
        else:
            tf_data = resample_ohlcv(data, timeframe)
        if len(tf_data) < 2:
            print(f"Skipping timeframe '{timeframe}' due to insufficient bars after alignment/resampling")
            continue

        timeframe_rows: list[dict[str, float | int | str | bool]]
        if workers == 1:
            _init_worker(tf_data, engine_config)
            timeframe_rows = [_run_combo(combo) for combo in grid]
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(tf_data, engine_config),
            ) as executor:
                timeframe_rows = list(executor.map(_run_combo, grid, chunksize=10))

        for row in timeframe_rows:
            row["timeframe"] = timeframe
        rows.extend(timeframe_rows)

    if not rows:
        raise ValueError("No valid parameter runs completed. Check your data and timeframe inputs.")

    leaderboard = pd.DataFrame(rows).sort_values(by=args.sort_by, ascending=False).reset_index(drop=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    leaderboard_path = out_dir / "wiseman_sweep_results.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    top_df = leaderboard.head(args.top)
    top_path = out_dir / "wiseman_sweep_top.csv"
    top_df.to_csv(top_path, index=False)

    best_summary = {
        "total_runs": int(len(leaderboard)),
        "sort_by": args.sort_by,
        "best_setup": top_df.iloc[0].to_dict(),
    }
    summary_path = out_dir / "wiseman_sweep_summary.json"
    summary_path.write_text(json.dumps(best_summary, indent=2), encoding="utf-8")

    best_row = leaderboard.iloc[0]
    best_timeframe = str(best_row["timeframe"])
    normalized_best_tf = normalize_timeframe(best_timeframe)
    if source_tf != "unknown" and normalized_best_tf == normalized_source_tf:
        best_data = data.copy()
    else:
        best_data = resample_ohlcv(data, best_timeframe)
    best_strategy = _build_strategy_from_row(best_row)
    best_result = BacktestEngine(engine_config).run(best_data, best_strategy)
    benchmark_close = best_data["close"].reindex(best_result.returns.index).ffill().bfill().astype(float)
    benchmark_equity = (1.0 + benchmark_close.pct_change().fillna(0.0)).cumprod() * float(best_result.equity_curve.iloc[0])

    pdf_path = out_dir / "wiseman_sweep_report.pdf"
    _make_pdf_report(
        out_path=pdf_path,
        leaderboard=leaderboard,
        sort_by=args.sort_by,
        top_n=args.top,
        cli_args=args,
        total_grid_size=len(grid),
        best_result=best_result,
        benchmark_equity=benchmark_equity,
    )

    print("Sweep complete")
    print(f"Leaderboard written to: {leaderboard_path}")
    print(f"Top setups written to: {top_path}")
    print(f"Best setup summary written to: {summary_path}")
    print(f"Comprehensive PDF report written to: {pdf_path}")
    print("Top setups preview:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(top_df)


if __name__ == "__main__":
    main()
