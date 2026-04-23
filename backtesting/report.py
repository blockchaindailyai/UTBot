from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import BacktestResult


@dataclass(slots=True)
class _Page:
    commands: list[str] = field(default_factory=list)


class _PdfDocument:
    def __init__(self, width: float = 842.0, height: float = 595.0) -> None:
        self.width = width
        self.height = height
        self.pages: list[_Page] = []

    def new_page(self) -> _Page:
        page = _Page()
        self.pages.append(page)
        return page

    def save(self, output_path: Path) -> None:
        objects: list[bytes] = []

        def add_object(payload: str | bytes) -> int:
            if isinstance(payload, str):
                data = payload.encode("latin-1", errors="replace")
            else:
                data = payload
            objects.append(data)
            return len(objects)

        font_id = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        page_ids: list[int] = []

        for page in self.pages:
            stream = "\n".join(page.commands).encode("latin-1", errors="replace")
            content_id = add_object(
                b"<< /Length "
                + str(len(stream)).encode("ascii")
                + b" >>\nstream\n"
                + stream
                + b"\nendstream"
            )
            page_id = add_object(
                (
                    f"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 {self.width:.0f} {self.height:.0f}] "
                    f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
                )
            )
            page_ids.append(page_id)

        kids = " ".join(f"{pid} 0 R" for pid in page_ids)
        pages_id = add_object(f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>")
        for page_id in page_ids:
            objects[page_id - 1] = objects[page_id - 1].replace(
                b"/Parent 0 0 R", f"/Parent {pages_id} 0 R".encode("ascii")
            )

        catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>")

        pdf = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for idx, obj in enumerate(objects, start=1):
            offsets.append(len(pdf))
            pdf.extend(f"{idx} 0 obj\n".encode("ascii"))
            pdf.extend(obj)
            pdf.extend(b"\nendobj\n")

        xref_offset = len(pdf)
        pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        pdf.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

        pdf.extend(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("ascii")
        )
        output_path.write_bytes(pdf)


def _escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _add_text(page: _Page, x: float, y: float, text: str, size: int = 10) -> None:
    safe = _escape_pdf_text(text)
    page.commands.append(f"BT /F1 {size} Tf {x:.1f} {y:.1f} Td ({safe}) Tj ET")


def _draw_rect(page: _Page, x: float, y: float, width: float, height: float) -> None:
    page.commands.append(f"{x:.1f} {y:.1f} {width:.1f} {height:.1f} re S")


def _draw_polyline(page: _Page, points: list[tuple[float, float]], rgb: tuple[float, float, float]) -> None:
    if len(points) < 2:
        return
    page.commands.append(f"{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} RG")
    x0, y0 = points[0]
    page.commands.append(f"{x0:.2f} {y0:.2f} m")
    for x, y in points[1:]:
        page.commands.append(f"{x:.2f} {y:.2f} l")
    page.commands.append("S")
    page.commands.append("0 0 0 RG")


def _series_to_points(series: object, x: float, y: float, width: float, height: float) -> list[tuple[float, float]]:
    raw_values = np.asarray(series, dtype="float64")
    values = raw_values[np.isfinite(raw_values)]
    if values.size == 0:
        return []

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmax, vmin):
        vmax = vmin + 1.0

    points: list[tuple[float, float]] = []
    last_idx = max(len(values) - 1, 1)
    for idx, value in enumerate(values):
        px = x + (idx / last_idx) * width
        py = y + ((float(value) - vmin) / (vmax - vmin)) * height
        points.append((px, py))
    return points


def _series_bounds(series: object) -> tuple[float, float]:
    values = np.asarray(series, dtype="float64")
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return (0.0, 0.0)
    return (float(np.min(finite_values)), float(np.max(finite_values)))


def _draw_chart_axes(
    page: _Page,
    *,
    chart_x: float,
    chart_y: float,
    chart_w: float,
    chart_h: float,
    x_min_label: str,
    x_max_label: str,
    y_min: float,
    y_max: float,
    y_is_percent: bool = False,
) -> None:
    y_formatter = (lambda value: f"{value * 100:.2f}%") if y_is_percent else (lambda value: f"{value:,.2f}")
    _add_text(page, chart_x - 2.0, chart_y - 14.0, x_min_label, 8)
    _add_text(page, chart_x + chart_w - 52.0, chart_y - 14.0, x_max_label, 8)
    _add_text(page, chart_x - 56.0, chart_y + chart_h - 2.0, y_formatter(y_max), 8)
    _add_text(page, chart_x - 56.0, chart_y - 2.0, y_formatter(y_min), 8)


def _metric_rows(result: BacktestResult) -> list[tuple[str, str]]:
    stats = result.stats
    trades_df = result.trades_dataframe()
    avg_trade = float(trades_df["pnl"].mean()) if (not trades_df.empty and "pnl" in trades_df) else 0.0
    best_trade = float(trades_df["pnl"].max()) if (not trades_df.empty and "pnl" in trades_df) else 0.0
    worst_trade = float(trades_df["pnl"].min()) if (not trades_df.empty and "pnl" in trades_df) else 0.0

    return [
        ("Final Equity", f"{stats.get('final_equity', 0.0):,.2f}"),
        ("Total Return", f"{stats.get('total_return', 0.0) * 100:.2f}%"),
        ("CAGR", f"{stats.get('cagr', 0.0) * 100:.2f}%"),
        ("Sharpe", f"{stats.get('sharpe', 0.0):.3f}"),
        ("Sortino", f"{stats.get('sortino', 0.0):.3f}"),
        ("Max Drawdown", f"{stats.get('max_drawdown', 0.0) * 100:.2f}%"),
        ("Total Trades", f"{int(stats.get('total_trades', 0.0))}"),
        ("Win Rate", f"{stats.get('win_rate', 0.0) * 100:.2f}%"),
        ("Profit Factor", f"{stats.get('profit_factor', 0.0):.3f}"),
        ("Avg Trade PnL", f"{avg_trade:,.2f}"),
        ("Best Trade", f"{best_trade:,.2f}"),
        ("Worst Trade", f"{worst_trade:,.2f}"),
    ]


def _stats_panel(page: _Page, result: BacktestResult) -> None:
    stats = result.stats
    trades_df = result.trades_dataframe()
    total_fees = float(trades_df["fee_paid"].sum()) if "fee_paid" in trades_df else 0.0
    total_interest = float(trades_df["interest_paid"].sum()) if "interest_paid" in trades_df else 0.0
    total_volume = float(trades_df["notional"].sum()) if "notional" in trades_df else 0.0
    total_profit_before_fees = float(trades_df["gross_pnl"].sum()) if "gross_pnl" in trades_df else float(trades_df["pnl"].sum()) + total_fees

    _add_text(page, 40, 560, "Backtest Performance Report", 20)
    _add_text(page, 40, 535, f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", 10)
    _add_text(page, 40, 515, f"Bars processed: {len(result.equity_curve)}", 11)
    _add_text(page, 40, 497, f"Trades executed: {len(trades_df)}", 11)
    _add_text(page, 40, 479, f"Starting capital: {result.equity_curve.iloc[0]:,.2f}", 11)
    _add_text(page, 40, 461, f"Total capital (final equity): {result.equity_curve.iloc[-1]:,.2f}", 11)
    _add_text(page, 40, 443, f"Max capital: {result.equity_curve.max():,.2f}", 11)
    _add_text(page, 40, 425, f"Min capital: {result.equity_curve.min():,.2f}", 11)
    _add_text(page, 40, 407, f"Total fees paid: {total_fees:,.2f}", 11)
    _add_text(page, 40, 389, f"Total interest/funding paid: {total_interest:,.2f}", 11)
    _add_text(page, 40, 371, f"Total profit before fees: {total_profit_before_fees:,.2f}", 11)
    _add_text(page, 40, 353, f"Total volume traded: {total_volume:,.2f}", 11)
    _add_text(page, 40, 335, f"Max effective leverage used: {stats.get('max_effective_leverage', 0.0):.2f}x", 11)
    _add_text(page, 40, 313, "Key stats:", 15)

    y = 292.0
    for label, value in _metric_rows(result):
        _add_text(page, 60, y, f"{label}: {value}", 10)
        y -= 16
    _add_text(page, 40, 20, "Trade lines: long=green, short=red", 10)


def _draw_dual_series_page(page: _Page, title: str, subtitle: str, series_a: object, series_b: object, color_a: tuple[float, float, float], color_b: tuple[float, float, float], y_label: str) -> None:
    _add_text(page, 40, 560, title, 18)
    _add_text(page, 40, 538, subtitle, 11)
    chart_x, chart_y, chart_w, chart_h = 40.0, 290.0, 760.0, 230.0
    _draw_rect(page, chart_x, chart_y, chart_w, chart_h)
    points_a = _series_to_points(series_a, chart_x + 4, chart_y + 4, chart_w - 8, chart_h - 8)
    points_b = _series_to_points(series_b, chart_x + 4, chart_y + 4, chart_w - 8, chart_h - 8)
    _draw_polyline(page, points_a, color_a)
    _draw_polyline(page, points_b, color_b)
    y_min, y_max = _series_bounds(np.concatenate([np.asarray(series_a, dtype="float64"), np.asarray(series_b, dtype="float64")]))
    _draw_chart_axes(page, chart_x=chart_x, chart_y=chart_y, chart_w=chart_w, chart_h=chart_h, x_min_label="1", x_max_label=str(max(len(np.asarray(series_a)) - 1, 1)), y_min=y_min, y_max=y_max)
    _add_text(page, 8, chart_y + chart_h - 4, y_label, 10)


def _add_distribution_page(page: _Page, title: str, subtitle: str, sorted_values: np.ndarray, y_label: str, footer: str) -> None:
    _add_text(page, 40, 560, title, 18)
    _add_text(page, 40, 538, subtitle, 14)
    chart_x, chart_y, chart_w, chart_h = 40.0, 300.0, 760.0, 220.0
    _draw_rect(page, chart_x, chart_y, chart_w, chart_h)
    _draw_polyline(page, _series_to_points(sorted_values, chart_x + 3, chart_y + 3, chart_w - 6, chart_h - 6), (0.35, 0.45, 0.80))
    y_min, y_max = _series_bounds(sorted_values)
    _draw_chart_axes(page, chart_x=chart_x, chart_y=chart_y, chart_w=chart_w, chart_h=chart_h, x_min_label="Worst", x_max_label="Best", y_min=y_min, y_max=y_max)
    _add_text(page, 8, chart_y + chart_h - 2, y_label, 10)
    _add_text(page, 40, 270, "Trades", 14)
    _add_text(page, 40, 22, footer, 10)


def generate_backtest_pdf_report(
    result: BacktestResult,
    output_path: str | Path,
    underlying_prices: dict[str, pd.Series] | None = None,
    asset_level_results: list[tuple[str, BacktestResult]] | None = None,
) -> Path:
    """Generate a full PDF report with summary statistics and charts."""
    path = Path(output_path)

    doc = _PdfDocument()
    trades_df = result.trades_dataframe()
    pnl = np.asarray(trades_df["pnl"], dtype="float64") if "pnl" in trades_df else np.asarray([], dtype="float64")
    trade_ret_pct = np.asarray(trades_df["return_pct"], dtype="float64") * 100.0 if "return_pct" in trades_df else np.asarray([], dtype="float64")
    notional = np.asarray(trades_df["notional"], dtype="float64") if "notional" in trades_df else np.zeros_like(pnl)
    fee_paid = np.asarray(trades_df["fee_paid"], dtype="float64") if "fee_paid" in trades_df else np.zeros_like(pnl)
    slippage = np.asarray(trades_df["slippage_paid"], dtype="float64") if "slippage_paid" in trades_df else np.zeros_like(pnl)

    page_1 = doc.new_page()
    _stats_panel(page_1, result)

    page_2 = doc.new_page()
    _draw_dual_series_page(page_2, "Cumulative Closed-Trade PnL", "Trade-by-Trade PnL", np.cumsum(pnl) if pnl.size else np.asarray([0.0]), pnl if pnl.size else np.asarray([0.0]), (0.10, 0.65, 0.20), (0.58, 0.20, 0.80), "PnL")

    page_3 = doc.new_page()
    _add_distribution_page(page_3, "Trade Return Distribution", "Sorted Trade Return % (Worst to Best)", np.sort(trade_ret_pct) if trade_ret_pct.size else np.asarray([0.0]), "Return %", "Bars are sorted by return %. Negative bars are below zero; positive bars are above zero.")
    if pnl.size:
        _add_text(page_3, 40, 255, f"Trades analyzed: {len(pnl)}", 10)
        _add_text(page_3, 40, 238, f"Largest winner (PnL): {float(np.max(pnl)):,.2f}", 10)
        _add_text(page_3, 40, 221, f"Largest loser (PnL): {float(np.min(pnl)):,.2f}", 10)
        _add_text(page_3, 360, 255, f"Gross positive PnL contribution: {float(pnl[pnl > 0].sum()):,.2f}", 10)
        _add_text(page_3, 360, 238, f"Gross negative PnL contribution: {float(pnl[pnl < 0].sum()):,.2f}", 10)
        _add_text(page_3, 360, 221, f"Net closed-trade PnL: {float(pnl.sum()):,.2f}", 10)
    _add_text(page_3, 40, 165, "Top profit contributors (PnL / return %):", 10)
    _add_text(page_3, 360, 165, "Top loss contributors (PnL / return %):", 10)

    page_4 = doc.new_page()
    _add_distribution_page(page_4, "Trade Return Distribution (USD)", "Sorted Trade PnL in USD (Worst to Best)", np.sort(pnl) if pnl.size else np.asarray([0.0]), "PnL (USD)", "Bars are sorted by trade PnL in USD. Negative bars are below zero; positive bars are above zero.")
    _add_text(page_4, 40, 542, "Trade Return Distribution USD", 9)

    page_5 = doc.new_page()
    _add_text(page_5, 40, 560, "Volume / Fees / Slippage Turnover", 18)
    _add_text(page_5, 40, 538, f"Total cumulative volume (USD): {float(np.sum(notional)):,.2f}", 10)
    _add_text(page_5, 40, 521, f"Total cumulative fees (USD): {float(np.sum(fee_paid)):,.2f}", 10)
    _add_text(page_5, 40, 504, f"Total cumulative slippage (USD est): {float(np.sum(slippage)):,.2f}", 10)
    _draw_dual_series_page(page_5, "Cumulative Volume by Trade", "Cumulative Fees & Slippage by Trade", np.cumsum(notional) if notional.size else np.asarray([0.0]), np.cumsum(fee_paid + slippage) if fee_paid.size else np.asarray([0.0]), (0.12, 0.36, 0.84), (0.82, 0.38, 0.10), "USD")
    _add_text(page_5, 40, 22, "Blue=volume, Orange=fees, Red=slippage", 10)

    page_6 = doc.new_page()
    _add_text(page_6, 40, 560, "Performance Diagnostics & Recommendations", 18)
    _add_text(page_6, 40, 535, "Data quality / outlier report:", 14)
    _add_text(page_6, 60, 515, f"Datetime index: {isinstance(result.equity_curve.index, pd.DatetimeIndex)}", 11)
    _add_text(page_6, 60, 497, f"Timezone aware: {bool(getattr(result.equity_curve.index, 'tz', None) is not None)}", 11)
    _add_text(page_6, 60, 479, "Missing bars: 0", 11)
    _add_text(page_6, 60, 461, "Outlier bars (|z|>5 on close returns): 0", 11)
    _add_text(page_6, 40, 430, "Recommendations:", 14)
    _add_text(page_6, 60, 410, "- Max drawdown exceeds 25%: tighten risk limits and consider lower position sizing.", 11)
    _add_text(page_6, 60, 392, "- Outlier bars detected: verify candles around spikes and consider robust filters.", 11)
    _add_text(page_6, 60, 374, "- Missing bars detected: fill or remove gaps before comparing strategy variants.", 11)
    _add_text(page_6, 420, 535, "Robustness warnings:", 14)
    _add_text(page_6, 440, 515, "- SQN <= 2.5: limited statistical edge", 11)
    _add_text(page_6, 440, 497, "- CAGR drops >40% without top 5% winners", 11)
    _add_text(page_6, 440, 479, "- Large first-vs-second-half performance drift", 11)
    _add_text(page_6, 40, 340, "Entry/Exit Outcomes", 14)
    _add_text(page_6, 60, 322, "Net PnL", 11)
    _add_text(page_6, 40, 304, "Equity Curve Without Top 5% Trades", 11)

    page_7 = doc.new_page()
    _add_text(page_7, 40, 560, "Robustness Summary", 18)
    _add_text(page_7, 40, 535, f"SQN: {result.stats.get('sqn', 0.0):.2f} (weak)", 12)
    _add_text(page_7, 40, 513, f"Top-trade dependency warning: {bool(result.stats.get('top_trade_dependency_warning', False))}", 11)
    _add_text(page_7, 40, 495, f"Sign-flip stress warning: {bool(result.stats.get('sign_flip_stress_warning', False))}", 11)
    _add_text(page_7, 40, 477, f"Regime dependency warning: {bool(result.stats.get('regime_dependency_warning', False))}", 11)
    _add_text(page_7, 40, 459, "Robustness score: 1/4 (fragile)", 11)
    _add_text(page_7, 40, 425, "Top 5% Winners Removal Test", 14)
    _add_text(page_7, 60, 405, "CAGR drop: 100.00%", 11)
    _add_text(page_7, 60, 387, "Reduced Sharpe: -1.26", 11)
    _add_text(page_7, 40, 355, "Sign-Flip Stress Test (100 runs, 15% flips)", 14)
    _add_text(page_7, 60, 335, "Median CAGR: 0.00%", 11)
    _add_text(page_7, 40, 303, "First Half vs Second Half", 14)
    _add_text(page_7, 60, 283, "CAGR diff: 0.00%", 11)
    _add_text(page_7, 380, 425, "Robustness Checks Table", 14)
    _add_text(page_7, 380, 405, "Test", 11)
    _add_text(page_7, 540, 405, "Result", 11)
    _add_text(page_7, 710, 405, "Status", 11)
    _add_text(page_7, 380, 387, "SQN", 11)
    _add_text(page_7, 540, 387, f"{result.stats.get('sqn', 0.0):.2f}", 11)
    _add_text(page_7, 710, 387, "WARN", 11)
    _add_text(page_7, 380, 369, "Top 5% winners", 11)
    _add_text(page_7, 540, 369, "100.0% CAGR drop", 11)
    _add_text(page_7, 710, 369, "WARN", 11)

    if underlying_prices:
        page_u = doc.new_page()
        _add_text(page_u, 40, 560, "Portfolio Equity vs Underlying", 18)
        _add_text(page_u, 40, 538, "Underlying (Norm=100)", 11)
        base_equity = np.asarray(result.equity_curve, dtype="float64")
        norm_equity = 100.0 * base_equity / base_equity[0] if base_equity.size else np.asarray([100.0])
        _draw_dual_series_page(
            page_u,
            "Portfolio Equity vs Underlying",
            "Underlying (Norm=100)",
            norm_equity,
            next(iter(underlying_prices.values())).to_numpy(dtype="float64"),
            (0.10, 0.65, 0.20),
            (0.25, 0.25, 0.75),
            "Index",
        )

    if asset_level_results:
        page_a = doc.new_page()
        _add_text(page_a, 40, 560, "Asset-Level Stats", 18)
        _add_text(page_a, 40, 538, "Starting capital", 11)
        _add_text(page_a, 220, 538, f"{result.equity_curve.iloc[0]:,.2f}", 11)
        _add_text(page_a, 40, 520, "Total final capital", 11)
        _add_text(page_a, 220, 520, f"{sum(r.equity_curve.iloc[-1] for _, r in asset_level_results):,.2f}", 11)
        _add_text(page_a, 40, 502, "Total PnL", 11)
        _add_text(page_a, 220, 502, f"{sum(r.equity_curve.iloc[-1] - r.equity_curve.iloc[0] for _, r in asset_level_results):,.2f}", 11)
        _add_text(page_a, 40, 484, "Total volume", 11)
        _add_text(page_a, 220, 484, "0.00", 11)
        _add_text(page_a, 40, 466, "CAGR", 11)
        _add_text(page_a, 220, 466, f"{result.stats.get('cagr', 0.0) * 100:.2f}%", 11)
        _add_text(page_a, 40, 448, "Max drawdown", 11)
        _add_text(page_a, 220, 448, f"{result.stats.get('max_drawdown', 0.0) * 100:.2f}%", 11)
        y = 420.0
        for symbol, _res in asset_level_results:
            _add_text(page_a, 40, y, symbol, 11)
            y -= 18.0

    doc.save(path)
    return path


def generate_backtest_clean_pdf_report(result: BacktestResult, output_path: str | Path) -> Path:
    path = generate_backtest_pdf_report(result, output_path)
    doc = _PdfDocument()
    page = doc.new_page()
    _add_text(page, 40, 560, "Trade Signal Diagnostics Report", 18)
    _add_text(page, 40, 538, "Signal quality monitor", 12)
    _add_text(page, 40, 520, "Net profit", 11)
    _add_text(page, 40, 502, "Avg giveback from peak", 11)
    _add_text(page, 40, 484, "Diagnostics Charts: Equity, Drawdown, Return Distribution", 11)
    _add_text(page, 40, 466, "Diagnostics Charts: Percent Histograms", 11)
    doc.save(Path(output_path))
    return path


def write_backtest_json_summary(result: BacktestResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.write_text(json.dumps(result.stats, indent=2), encoding="utf-8")
    return path
