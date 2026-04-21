from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import BacktestResult
from .trade_metrics import compute_trade_diagnostics


_PDF_TEXT_REPLACEMENTS = str.maketrans(
    {
        "…": "...",
        "–": "-",
        "—": "-",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        " ": " ",
    }
)


def _sanitize_pdf_text(text: str) -> str:
    normalized = str(text).translate(_PDF_TEXT_REPLACEMENTS)
    return normalized.encode("latin-1", errors="replace").decode("latin-1")


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


@dataclass(slots=True)
class _Page:
    content: list[str] = field(default_factory=list)


class _SimplePdf:
    def __init__(self, width: float = 842.0, height: float = 595.0) -> None:
        self.width = width
        self.height = height
        self.pages: list[_Page] = []

    def new_page(self) -> _Page:
        page = _Page()
        self.pages.append(page)
        return page

    def save(self, path: Path) -> None:
        objects: list[bytes] = []

        def add_object(body: str | bytes) -> int:
            if isinstance(body, str):
                body_b = _sanitize_pdf_text(body).encode("latin-1")
            else:
                body_b = body
            objects.append(body_b)
            return len(objects)

        font_id = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

        page_ids: list[int] = []
        for page in self.pages:
            stream = _sanitize_pdf_text("\n".join(page.content)).encode("latin-1")
            content_id = add_object(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")

            page_dict = (
                f"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 {self.width:.0f} {self.height:.0f}] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
            )
            page_ids.append(add_object(page_dict))

        kids = " ".join(f"{pid} 0 R" for pid in page_ids)
        pages_id = add_object(f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>")

        for page_id in page_ids:
            objects[page_id - 1] = objects[page_id - 1].replace(b"/Parent 0 0 R", f"/Parent {pages_id} 0 R".encode("ascii"))

        catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>")

        data = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for idx, obj in enumerate(objects, start=1):
            offsets.append(len(data))
            data.extend(f"{idx} 0 obj\n".encode("ascii"))
            data.extend(obj)
            data.extend(b"\nendobj\n")

        xref_offset = len(data)
        data.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        data.extend(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            data.extend(f"{off:010d} 00000 n \n".encode("ascii"))

        data.extend((f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n" f"startxref\n{xref_offset}\n%%EOF\n").encode("ascii"))

        path.write_bytes(data)


def _add_text(page: _Page, x: float, y: float, text: str, size: int = 10) -> None:
    page.content.append(f"BT /F1 {size} Tf {x:.1f} {y:.1f} Td ({_escape_pdf_text(text)}) Tj ET")


def _value_label(value: float) -> str:
    abs_value = abs(value)
    if abs_value < 1000:
        return f"{value:.2f}"
    if abs_value < 1_000_000:
        return f"{value:,.0f}"
    return f"{value:.2e}"


def _percent_label(value: float) -> str:
    pct = value * 100.0
    abs_pct = abs(pct)
    if abs_pct < 1000:
        return f"{pct:.2f}%"
    return f"{pct:.2e}%"


def _entry_exit_signal_rows(trades_df: pd.DataFrame) -> list[tuple[str, str, int, float]]:
    grouped: dict[tuple[str, str], tuple[int, float]] = {}
    if not trades_df.empty and {"entry_signal", "exit_signal", "pnl"}.issubset(trades_df.columns):
        for _, row in trades_df.iterrows():
            key = (str(row.get("entry_signal", "Unknown")), str(row.get("exit_signal", "Unknown")))
            count, pnl = grouped.get(key, (0, 0.0))
            grouped[key] = (count + 1, pnl + float(row.get("pnl", 0.0)))

    return [
        (entry, exit_signal, count, pnl)
        for (entry, exit_signal), (count, pnl) in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1]))
    ]


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)] + "…"


def _draw_signal_outcomes_table(
    page: _Page,
    rows: list[tuple[str, str, int, float]],
    x: float,
    y_top: float,
    row_height: float = 8.0,
    max_rows: int = 36,
) -> None:
    headers = ["Entry", "Exit", "Trades", "Net PnL"]
    col_widths = [120.0, 120.0, 32.0, 56.0]
    col_offsets: list[float] = []
    running = x
    for width in col_widths:
        col_offsets.append(running)
        running += width

    _add_text(page, x, y_top + 10, "Entry/Exit Outcomes", 10)
    for idx, header in enumerate(headers):
        _add_text(page, col_offsets[idx] + 1, y_top, header, 7)

    visible_rows = rows[:max_rows]
    for row_idx, (entry_signal, exit_signal, trade_count, net_pnl) in enumerate(visible_rows, start=1):
        y = y_top - (row_idx * row_height)
        _add_text(page, col_offsets[0] + 1, y, _truncate_text(entry_signal, 80), 6)
        _add_text(page, col_offsets[1] + 1, y, _truncate_text(exit_signal, 80), 6)
        _add_text(page, col_offsets[2] + 1, y, f"{trade_count}", 6)
        _add_text(page, col_offsets[3] + 1, y, f"{net_pnl:+,.0f}", 6)

    bottom_y = y_top - ((len(visible_rows) + 0.6) * row_height)
    if len(rows) > max_rows:
        remaining = len(rows) - max_rows
        _add_text(page, x, bottom_y - 3, f"… {remaining} more rows not shown", 6)

    table_top = y_top + 3
    table_bottom = y_top - ((len(visible_rows) + 0.45) * row_height)
    page.content.append("0.1 w 0.65 0.65 0.65 RG")
    for offset in [0.0, *[sum(col_widths[:i + 1]) for i in range(len(col_widths))]]:
        line_x = x + offset
        page.content.append(f"{line_x:.1f} {table_bottom:.1f} m {line_x:.1f} {table_top:.1f} l S")
    for i in range(len(visible_rows) + 2):
        line_y = y_top + 2 - (i * row_height)
        page.content.append(f"{x:.1f} {line_y:.1f} m {x + sum(col_widths):.1f} {line_y:.1f} l S")
    page.content.append("0.2 w 0 0 0 RG")


def _render_stat_value(value: float, kind: str) -> str:
    if kind == "currency":
        return f"{value:,.2f}"
    if kind == "percent":
        return _percent_label(value)
    if kind == "count":
        return f"{int(round(value))}"
    if kind == "multiple":
        return f"{value:.2f}x"
    return _value_label(value)


def _trade_giveback_summary(trades_df: pd.DataFrame) -> dict[str, float]:
    if trades_df.empty:
        return {
            "avg_peak_unrealized_return_pct": 0.0,
            "avg_giveback_from_peak_pct": 0.0,
            "avg_capture_ratio_vs_peak": 0.0,
            "median_capture_ratio_vs_peak": 0.0,
            "high_giveback_trade_rate": 0.0,
        }

    peak = pd.to_numeric(trades_df.get("peak_unrealized_return_pct", 0.0), errors="coerce").fillna(0.0)
    giveback = pd.to_numeric(trades_df.get("giveback_from_peak_pct", 0.0), errors="coerce").fillna(0.0)
    capture = pd.to_numeric(trades_df.get("capture_ratio_vs_peak", 0.0), errors="coerce").fillna(0.0)
    high_giveback = ((peak > 0) & ((giveback / peak.replace(0, np.nan)).fillna(0.0) >= 0.5)).mean()
    return {
        "avg_peak_unrealized_return_pct": float(peak.mean()),
        "avg_giveback_from_peak_pct": float(giveback.mean()),
        "avg_capture_ratio_vs_peak": float(capture.mean()),
        "median_capture_ratio_vs_peak": float(capture.median()),
        "high_giveback_trade_rate": float(high_giveback),
    }


def _build_asset_level_report_rows(asset_level_results: list[tuple[str, BacktestResult]]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for asset_name, asset_result in asset_level_results:
        equity = asset_result.equity_curve.astype(float)
        initial_capital = float(equity.iloc[0])
        trades_df = asset_result.trades_dataframe()
        diagnostics = compute_trade_diagnostics(
            trades_df=trades_df,
            initial_capital=initial_capital,
            total_fees_paid=asset_result.total_fees_paid,
            execution_events=asset_result.execution_events,
            slippage_rate=float(asset_result.stats.get("slippage_rate", 0.0)),
        )
        max_effective_leverage = _estimate_max_effective_leverage(asset_result, trades_df, initial_capital)

        row: dict[str, float | str] = {
            "asset": asset_name,
            "starting_capital": initial_capital,
            "min_capital": float(equity.min()),
            "final_capital": float(equity.iloc[-1]),
            "total_pnl": float(equity.iloc[-1] - initial_capital),
            "total_fees_paid": float(asset_result.total_fees_paid),
            "total_financing_paid": float(asset_result.total_financing_paid),
            "total_profit_before_fees": float(asset_result.total_profit_before_fees),
            "total_trades": float(len(asset_result.trades)),
            "total_volume": float(diagnostics["total_cumulative_volume"]),
            "mean_position_size_usd": float(diagnostics["mean_position_size_usd"]),
            "median_position_size_usd": float(diagnostics["median_position_size_usd"]),
            "mean_volume_per_trade": float(diagnostics["mean_volume_per_trade"]),
            "median_volume_per_trade": float(diagnostics["median_volume_per_trade"]),
            "total_slippage_paid": float(diagnostics["total_slippage_paid"]),
            "max_effective_leverage_used": float(max_effective_leverage),
        }
        for key, value in asset_result.stats.items():
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                row[key] = float(value)
        rows.append(row)
    return rows


def _render_asset_level_summary_pages(pdf: _SimplePdf, asset_level_results: list[tuple[str, BacktestResult]]) -> None:
    asset_rows = _build_asset_level_report_rows(asset_level_results)
    if not asset_rows:
        return

    metric_specs: list[tuple[str, str, str]] = [
        ("Starting capital", "starting_capital", "currency"),
        ("Min capital", "min_capital", "currency"),
        ("Total final capital", "final_capital", "currency"),
        ("Total PnL", "total_pnl", "currency"),
        ("Total fees", "total_fees_paid", "currency"),
        ("Total financing", "total_financing_paid", "currency"),
        ("Total profit before fees", "total_profit_before_fees", "currency"),
        ("Total trades", "total_trades", "count"),
        ("Total volume", "total_volume", "currency"),
        ("CAGR", "cagr", "percent"),
        ("Total return", "total_return", "percent"),
        ("Volatility", "volatility", "percent"),
        ("Sharpe", "sharpe", "value"),
        ("Sortino", "sortino", "value"),
        ("Calmar", "calmar", "value"),
        ("Max drawdown", "max_drawdown", "percent"),
        ("Win rate", "win_rate", "percent"),
        ("Profit factor", "profit_factor", "value"),
        ("Average trade PnL", "avg_trade_pnl", "currency"),
        ("Average win", "avg_win", "currency"),
        ("Average loss", "avg_loss", "currency"),
        ("Expectancy", "expectancy", "currency"),
        ("Average holding bars", "avg_holding_bars", "value"),
        ("Exposure", "exposure", "percent"),
        ("Mean position size", "mean_position_size_usd", "currency"),
        ("Median position size", "median_position_size_usd", "currency"),
        ("Mean volume per trade", "mean_volume_per_trade", "currency"),
        ("Median volume per trade", "median_volume_per_trade", "currency"),
        ("Total slippage paid", "total_slippage_paid", "currency"),
        ("Max effective leverage", "max_effective_leverage_used", "multiple"),
    ]

    columns_per_page = 4
    metric_col_width = 170.0
    table_x = 40.0
    table_width = 760.0
    row_height = 14.0
    title_y = 560.0
    header_y = 518.0

    chunks = [asset_rows[i:i + columns_per_page] for i in range(0, len(asset_rows), columns_per_page)]
    for chunk_index, chunk in enumerate(chunks, start=1):
        page = pdf.new_page()
        suffix = "" if len(chunks) == 1 else f" ({chunk_index}/{len(chunks)})"
        _add_text(page, 40, title_y, f"Asset-Level Stats{suffix}", 16)
        _add_text(page, 40, title_y - 18, "Per-asset starting/final capital, PnL, turnover, and core performance metrics.", 10)

        asset_col_width = (table_width - metric_col_width) / max(len(chunk), 1)
        col_x = [table_x, table_x + metric_col_width]
        for idx in range(1, len(chunk)):
            col_x.append(table_x + metric_col_width + (idx * asset_col_width))

        _add_text(page, table_x + 2, header_y, "Metric", 8)
        for idx, asset_row in enumerate(chunk):
            _add_text(page, table_x + metric_col_width + (idx * asset_col_width) + 2, header_y, _truncate_text(str(asset_row["asset"]), 24), 8)

        for row_idx, (label, key, kind) in enumerate(metric_specs, start=1):
            y = header_y - (row_idx * row_height)
            _add_text(page, table_x + 2, y, label, 7)
            for idx, asset_row in enumerate(chunk):
                value = asset_row.get(key, 0.0)
                rendered = _render_stat_value(float(value), kind) if isinstance(value, (int, float, np.integer, np.floating)) else str(value)
                _add_text(page, table_x + metric_col_width + (idx * asset_col_width) + 2, y, _truncate_text(rendered, 18), 7)

        table_top = header_y + 4
        table_bottom = header_y - ((len(metric_specs) + 0.6) * row_height)
        page.content.append("0.1 w 0.65 0.65 0.65 RG")
        page.content.append(f"{table_x:.1f} {table_top:.1f} m {table_x + table_width:.1f} {table_top:.1f} l S")
        for row_idx in range(len(metric_specs) + 2):
            line_y = header_y + 2 - (row_idx * row_height)
            page.content.append(f"{table_x:.1f} {line_y:.1f} m {table_x + table_width:.1f} {line_y:.1f} l S")

        vertical_lines = [table_x, table_x + metric_col_width]
        vertical_lines.extend(table_x + metric_col_width + (idx * asset_col_width) for idx in range(1, len(chunk)))
        vertical_lines.append(table_x + table_width)
        for line_x in vertical_lines:
            page.content.append(f"{line_x:.1f} {table_bottom:.1f} m {line_x:.1f} {table_top:.1f} l S")
        page.content.append("0.2 w 0 0 0 RG")


def _draw_chart_axes(
    page: _Page,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    y_min: float,
    y_max: float,
    y_label: str,
    x_label: str,
    right_y_min: float | None = None,
    right_y_max: float | None = None,
    right_y_label: str | None = None,
    x_tick_labels: list[tuple[float, str]] | None = None,
    y_scale: str = "linear",
) -> None:
    page.content.append("0.2 w 0 0 0 RG")
    page.content.append(f"{x:.1f} {y:.1f} {w:.1f} {h:.1f} re S")
    _add_text(page, x, y + h + 12, title, 11)

    y_ticks = 5
    for i in range(y_ticks):
        ratio = i / (y_ticks - 1)
        if y_scale == "log":
            safe_min = max(y_min, 1e-12)
            safe_max = max(y_max, safe_min * 1.0000001)
            yv = float(np.exp(np.log(safe_min) + (np.log(safe_max) - np.log(safe_min)) * ratio))
        else:
            yv = y_min + (y_max - y_min) * ratio
        py = y + h * ratio
        page.content.append("0.1 w 0.86 0.86 0.86 RG")
        page.content.append(f"{x:.1f} {py:.1f} m {x + w:.1f} {py:.1f} l S")
        page.content.append("0.2 w 0 0 0 RG")
        _add_text(page, max(8.0, x - 54), py - 3, _value_label(yv), 8)

    _add_text(page, x + w / 2 - 20, y - 20, x_label, 9)
    _add_text(page, max(8.0, x - 30), y + h + 2, y_label, 9)

    if x_tick_labels:
        for ratio, label in x_tick_labels:
            px = x + ratio * w
            page.content.append("0.1 w 0.75 0.75 0.75 RG")
            page.content.append(f"{px:.1f} {y:.1f} m {px:.1f} {y - 4:.1f} l S")
            page.content.append("0.2 w 0 0 0 RG")

            anchor = min(len(label) * 1.6, 26)
            label_x = max(x - 10, min(px - anchor, x + w - 52))
            _add_text(page, label_x, y - 12, label, 8)
    else:
        _add_text(page, x - 4, y - 12, "0", 8)
        _add_text(page, x + w - 18, y - 12, "N", 8)

    if right_y_min is not None and right_y_max is not None and right_y_label:
        for i in range(y_ticks):
            ratio = i / (y_ticks - 1)
            yv = right_y_min + (right_y_max - right_y_min) * ratio
            py = y + h * ratio
            _add_text(page, min(x + w + 6, 792.0), py - 3, _value_label(yv), 8)
        _add_text(page, min(x + w + 6, 792.0), y + h + 2, right_y_label, 9)


def _draw_series(
    page: _Page,
    values: list[float],
    x: float,
    y: float,
    w: float,
    h: float,
    color_rgb: tuple[float, float, float],
    y_min: float | None = None,
    y_max: float | None = None,
    y_scale: str = "linear",
) -> None:
    if len(values) < 2:
        return
    vmin = min(values) if y_min is None else float(y_min)
    vmax = max(values) if y_max is None else float(y_max)
    if y_scale == "log":
        safe_values = [max(float(v), 1e-12) for v in values]
        vmin = max(vmin, 1e-12)
        vmax = max(vmax, vmin * 1.0000001)
        values = [float(np.log(v)) for v in safe_values]
        vmin = float(np.log(vmin))
        vmax = float(np.log(vmax))
    span = (vmax - vmin) or 1.0
    r, g, b = color_rgb
    page.content.append(f"{r:.3f} {g:.3f} {b:.3f} RG 1.2 w")
    for i, value in enumerate(values):
        px = x + (i / (len(values) - 1)) * w
        py = y + ((value - vmin) / span) * h
        page.content.append(f"{px:.2f} {py:.2f} {'m' if i == 0 else 'l'}")
    page.content.append("S")


def _draw_bars(
    page: _Page,
    values: list[float],
    x: float,
    y: float,
    w: float,
    h: float,
    color_rgb: tuple[float, float, float],
    y_min: float,
    y_max: float,
) -> None:
    if not values:
        return

    vmin = float(min(y_min, 0.0))
    vmax = float(max(y_max, 0.0))
    span = (vmax - vmin) or 1.0
    bar_w = w / max(len(values), 1)
    r, g, b = color_rgb

    zero_py = y + ((0.0 - vmin) / span) * h
    page.content.append("0.3 w 0.25 0.25 0.25 RG")
    page.content.append(f"{x:.2f} {zero_py:.2f} m {x + w:.2f} {zero_py:.2f} l S")

    for i, value in enumerate(values):
        left = x + i * bar_w + 0.15
        right = left + max(bar_w - 0.3, 0.3)
        py = y + ((float(value) - vmin) / span) * h
        y0 = min(py, zero_py)
        hh = abs(py - zero_py)
        if hh < 0.35:
            hh = 0.35
            y0 = zero_py - (0.35 if value < 0 else 0.0)
        page.content.append(f"{r:.3f} {g:.3f} {b:.3f} rg")
        page.content.append(f"{left:.2f} {y0:.2f} {max(right - left, 0.3):.2f} {hh:.2f} re f")




def _draw_trade_entry_exit_lines(
    page: _Page,
    trades: list,
    index: pd.Index,
    value_series: pd.Series,
    x: float,
    y: float,
    w: float,
    h: float,
    y_min: float | None = None,
    y_max: float | None = None,
    y_scale: str = "linear",
) -> None:
    if len(index) < 2 or len(value_series) < 2 or not trades:
        return

    lookup = {ts: i for i, ts in enumerate(index)}
    vmin = float(value_series.min()) if y_min is None else float(y_min)
    vmax = float(value_series.max()) if y_max is None else float(y_max)
    if y_scale == "log":
        vmin = max(vmin, 1e-12)
        vmax = max(vmax, vmin * 1.0000001)
    span = (vmax - vmin) or 1.0

    for trade in trades:
        start_i = lookup.get(trade.entry_time)
        end_i = lookup.get(trade.exit_time)
        if start_i is None or end_i is None or start_i == end_i:
            continue

        start_v = float(value_series.iloc[start_i])
        end_v = float(value_series.iloc[end_i])
        px1 = x + (start_i / (len(index) - 1)) * w
        if y_scale == "log":
            start_v = float(np.log(max(start_v, 1e-12)))
            end_v = float(np.log(max(end_v, 1e-12)))
            py1 = y + ((start_v - np.log(vmin)) / ((np.log(vmax) - np.log(vmin)) or 1.0)) * h
            py2 = y + ((end_v - np.log(vmin)) / ((np.log(vmax) - np.log(vmin)) or 1.0)) * h
        else:
            py1 = y + ((start_v - vmin) / span) * h
            py2 = y + ((end_v - vmin) / span) * h
        px2 = x + (end_i / (len(index) - 1)) * w

        if trade.side == "long":
            page.content.append("0.08 0.62 0.08 RG 1.6 w")
        else:
            page.content.append("0.82 0.12 0.12 RG 1.6 w")
        page.content.append(f"{px1:.2f} {py1:.2f} m {px2:.2f} {py2:.2f} l S")


def _detect_log_growth(values: pd.Series | list[float]) -> bool:
    series = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 8 or (series <= 0).any():
        return False
    q10 = float(series.quantile(0.10))
    q90 = float(series.quantile(0.90))
    if q10 <= 0 or (q90 / q10) < 8.0:
        return False
    idx = np.arange(len(series), dtype="float64")
    linear_corr = abs(float(np.corrcoef(idx, series.to_numpy())[0, 1]))
    log_corr = abs(float(np.corrcoef(idx, np.log(series.to_numpy()))[0, 1]))
    return bool(np.isfinite(log_corr) and log_corr > linear_corr * 1.05)



def _date_axis_labels(index: pd.Index) -> list[tuple[float, str]] | None:
    if len(index) < 2:
        return None
    if isinstance(index, pd.DatetimeIndex):
        total_span = index[-1] - index[0]
        if total_span <= pd.Timedelta(days=2):
            fmt = "%Y-%m-%d %H:%M"
        elif total_span <= pd.Timedelta(days=90):
            fmt = "%Y-%m-%d"
        else:
            fmt = "%Y-%m"
    else:
        fmt = None

    tick_count = min(8, len(index))
    if tick_count < 2:
        tick_count = 2

    last_pos = -1
    ticks: list[tuple[float, str]] = []
    for i in range(tick_count):
        pos = int(round(i * (len(index) - 1) / (tick_count - 1)))
        if pos == last_pos:
            continue
        last_pos = pos
        value = index[pos]
        label = value.strftime(fmt) if fmt and hasattr(value, "strftime") else str(value)
        ratio = pos / (len(index) - 1)
        ticks.append((ratio, label))

    return ticks if len(ticks) >= 2 else None


def _normalize_to_100(series: pd.Series) -> pd.Series:
    if series.empty:
        return series.astype(float)
    first = float(series.iloc[0])
    if first == 0:
        return pd.Series(100.0, index=series.index, dtype="float64")
    return series.astype(float) / first * 100.0


def _combined_underlying_index(underlying: pd.Series | pd.DataFrame | dict[str, pd.Series] | None, index: pd.Index) -> pd.Series | None:
    if underlying is None:
        return None
    if isinstance(underlying, pd.Series):
        s = underlying.reindex(index).ffill().bfill()
        return _normalize_to_100(s)
    if isinstance(underlying, dict):
        if not underlying:
            return None
        normalized = []
        for series in underlying.values():
            s = series.reindex(index).ffill().bfill()
            normalized.append(_normalize_to_100(s))
        return pd.concat(normalized, axis=1).mean(axis=1)
    if isinstance(underlying, pd.DataFrame):
        if underlying.empty:
            return None
        aligned = underlying.reindex(index).ffill().bfill()
        normalized = aligned.apply(_normalize_to_100, axis=0)
        return normalized.mean(axis=1)
    return None


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return 0.0
    if abs(denominator) <= 1e-12:
        return 0.0
    value = numerator / denominator
    return float(value) if np.isfinite(value) else 0.0


def _estimate_max_effective_leverage(result: BacktestResult, trades_df: pd.DataFrame, initial_capital: float) -> float:
    if trades_df.empty:
        return 0.0

    equity = result.equity_curve.astype(float)
    max_effective_leverage = 0.0

    for row in trades_df.itertuples(index=False):
        entry_price = float(getattr(row, "entry_price", 0.0))
        units = float(getattr(row, "units", 0.0))
        entry_notional = abs(entry_price * units)
        if entry_notional <= 0:
            continue

        entry_capital = np.nan
        capital_base = float(getattr(row, "capital_base", np.nan))
        if np.isfinite(capital_base) and capital_base > 0:
            entry_capital = capital_base

        pnl = float(getattr(row, "pnl", np.nan))
        return_pct = float(getattr(row, "return_pct", np.nan))
        if not np.isfinite(entry_capital) and np.isfinite(return_pct) and abs(return_pct) > 1e-12 and np.isfinite(pnl):
            estimated_capital = pnl / return_pct
            if np.isfinite(estimated_capital) and estimated_capital > 0:
                entry_capital = float(estimated_capital)

        if not np.isfinite(entry_capital):
            entry_time = getattr(row, "entry_time", None)
            if entry_time in equity.index:
                equity_at_entry = equity.loc[entry_time]
                if isinstance(equity_at_entry, pd.Series):
                    equity_at_entry = float(equity_at_entry.iloc[0])
                entry_capital = float(equity_at_entry)
            else:
                entry_capital = float(initial_capital)

        if entry_capital <= 0:
            continue

        leverage = entry_notional / entry_capital
        if np.isfinite(leverage):
            max_effective_leverage = max(max_effective_leverage, float(leverage))

    return float(max_effective_leverage)


def _compute_trade_metrics_from_pnl(pnl: pd.Series, initial_capital: float, periods_per_year: float) -> dict[str, float]:
    pnl = pnl.astype(float)
    max_abs_pnl = float(pnl.abs().max()) if not pnl.empty else 0.0
    scale = max(abs(float(initial_capital)), max_abs_pnl, 1.0)

    pnl_scaled = pnl / scale
    initial_capital_scaled = float(initial_capital) / scale

    equity_scaled = initial_capital_scaled + pnl_scaled.cumsum()
    equity = equity_scaled * scale
    equity = pd.concat([pd.Series([initial_capital], index=[-1], dtype="float64"), equity], ignore_index=True)
    returns = equity.pct_change().fillna(0.0)

    years = _safe_ratio(len(pnl), periods_per_year)
    cagr = 0.0
    if initial_capital_scaled > 0 and years > 0 and equity_scaled.iloc[-1] > 0:
        ratio = float(equity_scaled.iloc[-1] / initial_capital_scaled)
        cagr = float(np.exp(min(50.0, np.log(ratio) / years)) - 1.0)

    std = float(returns.std(ddof=0))
    sharpe = _safe_ratio(float(returns.mean()) * periods_per_year, std * np.sqrt(periods_per_year))
    rolling_max = equity.cummax()
    max_drawdown = float(((equity / rolling_max) - 1).min())

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    profit_factor = _safe_ratio(float(wins.sum()), abs(float(losses.sum())))
    win_rate = _safe_ratio(float(len(wins)), float(len(pnl)))

    return {
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "profit_factor": float(profit_factor),
        "win_rate": float(win_rate),
        "equity": equity,
    }


def _sqn_interpretation(sqn: float) -> str:
    if sqn < 1.6:
        return "no edge"
    if sqn < 2.0:
        return "weak"
    if sqn <= 3.0:
        return "tradable"
    if sqn <= 5.0:
        return "strong"
    return "excellent"


def _compute_r_multiples(trades_df: pd.DataFrame, initial_capital: float) -> pd.Series:
    if trades_df.empty:
        return pd.Series(dtype="float64")
    if "initial_risk" in trades_df.columns:
        initial_risk = trades_df["initial_risk"].astype(float)
        valid = initial_risk.abs() > 0
        if valid.any():
            return (trades_df.loc[valid, "pnl"].astype(float) / initial_risk.loc[valid]).astype("float64")
    if "return_pct" in trades_df.columns:
        return trades_df["return_pct"].astype("float64")
    if initial_capital:
        return (trades_df["pnl"].astype(float) / float(initial_capital)).astype("float64")
    return pd.Series(dtype="float64")


def _build_robustness_diagnostics(result: BacktestResult, trades_df: pd.DataFrame) -> dict[str, object]:
    n = len(trades_df)
    periods_per_year = float(result.stats.get("periods_per_year", 252.0))
    initial_capital = float(result.equity_curve.iloc[0])
    original_cagr = float(result.stats.get("cagr", 0.0))
    original_max_dd = float(result.stats.get("max_drawdown", 0.0))

    r_values = _compute_r_multiples(trades_df, initial_capital)

    sqn_std = float(r_values.std(ddof=1)) if len(r_values) > 1 else 0.0
    sqn = _safe_ratio(np.sqrt(len(r_values)) * float(r_values.mean()), sqn_std)
    sqn_warning = sqn <= 2.5

    reduced_pnl = pd.Series(dtype="float64")
    removed_count = 0
    if n > 0:
        remove_n = max(1, int(np.ceil(n * 0.05)))
        removed_count = remove_n
        top_idx = trades_df["pnl"].astype(float).sort_values(ascending=False).head(remove_n).index
        reduced_pnl = trades_df.drop(index=top_idx).sort_index()["pnl"].astype(float)

    reduced_metrics = _compute_trade_metrics_from_pnl(reduced_pnl, initial_capital, periods_per_year) if not reduced_pnl.empty else {
        "cagr": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "profit_factor": 0.0,
        "win_rate": 0.0,
        "equity": pd.Series([initial_capital], dtype="float64"),
    }
    cagr_drop_pct = _safe_ratio(original_cagr - float(reduced_metrics["cagr"]), max(abs(original_cagr), 1e-12)) * 100
    top_trade_warning = cagr_drop_pct > 40.0

    sign_flip_cagrs: list[float] = []
    sign_flip_drawdowns: list[float] = []
    rng = np.random.default_rng(7)
    pnl_full = trades_df["pnl"].astype(float) if n else pd.Series(dtype="float64")
    iterations = 100
    flip_fraction = 0.15
    if n > 0:
        flip_n = max(1, int(round(n * flip_fraction)))
        for _ in range(iterations):
            arr = pnl_full.to_numpy(copy=True)
            idx = rng.choice(n, size=flip_n, replace=False)
            arr[idx] *= -1
            metrics = _compute_trade_metrics_from_pnl(pd.Series(arr), initial_capital, periods_per_year)
            sign_flip_cagrs.append(float(metrics["cagr"]))
            sign_flip_drawdowns.append(float(metrics["max_drawdown"]))

    avg_sign_flip_cagr = float(np.mean(sign_flip_cagrs)) if sign_flip_cagrs else 0.0
    median_sign_flip_cagr = float(np.median(sign_flip_cagrs)) if sign_flip_cagrs else 0.0
    avg_sign_flip_max_dd = float(np.mean(sign_flip_drawdowns)) if sign_flip_drawdowns else 0.0
    sign_flip_fragile = median_sign_flip_cagr < 0 or abs(avg_sign_flip_max_dd) > abs(original_max_dd) * 1.5

    midpoint = n // 2
    first_half = trades_df.iloc[:midpoint]["pnl"].astype(float) if midpoint > 0 else pd.Series(dtype="float64")
    second_half = trades_df.iloc[midpoint:]["pnl"].astype(float) if n - midpoint > 0 else pd.Series(dtype="float64")
    first_metrics = _compute_trade_metrics_from_pnl(first_half, initial_capital, periods_per_year) if not first_half.empty else {
        "cagr": 0.0,
        "sharpe": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "max_drawdown": 0.0,
    }
    second_metrics = _compute_trade_metrics_from_pnl(second_half, initial_capital, periods_per_year) if not second_half.empty else {
        "cagr": 0.0,
        "sharpe": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "max_drawdown": 0.0,
    }
    cagr_diff_pct = _safe_ratio(abs(float(first_metrics["cagr"]) - float(second_metrics["cagr"])), max(abs(float(first_metrics["cagr"])), 1e-12)) * 100
    sharpe_diff_pct = _safe_ratio(abs(float(first_metrics["sharpe"]) - float(second_metrics["sharpe"])), max(abs(float(first_metrics["sharpe"])), 1e-12)) * 100
    regime_warning = cagr_diff_pct > 50.0 or sharpe_diff_pct > 40.0

    score = int(not sqn_warning) + int(not top_trade_warning) + int(not sign_flip_fragile) + int(not regime_warning)
    if score <= 1:
        classification = "fragile"
    elif score == 2:
        classification = "moderate"
    elif score == 3:
        classification = "robust"
    else:
        classification = "highly robust"

    return {
        "sqn": float(sqn),
        "sqn_warning": sqn_warning,
        "sqn_interpretation": _sqn_interpretation(float(sqn)),
        "reduced_metrics": reduced_metrics,
        "removed_count": removed_count,
        "cagr_drop_pct": float(cagr_drop_pct),
        "top_trade_warning": top_trade_warning,
        "sign_flip_cagrs": sign_flip_cagrs,
        "avg_sign_flip_cagr": avg_sign_flip_cagr,
        "median_sign_flip_cagr": median_sign_flip_cagr,
        "avg_sign_flip_max_dd": avg_sign_flip_max_dd,
        "sign_flip_fragile": sign_flip_fragile,
        "first_half": first_metrics,
        "second_half": second_metrics,
        "cagr_diff_pct": float(cagr_diff_pct),
        "sharpe_diff_pct": float(sharpe_diff_pct),
        "regime_warning": regime_warning,
        "score": score,
        "classification": classification,
        "reduced_equity": reduced_metrics["equity"],
    }


def _build_recommendations(result: BacktestResult, trades_df: pd.DataFrame) -> list[str]:
    recs: list[str] = []
    stats = result.stats
    if result.total_profit_before_fees > 0:
        fee_drag = _safe_ratio(result.total_fees_paid + result.total_financing_paid, result.total_profit_before_fees)
        if fee_drag > 0.35:
            recs.append("Cost drag is high (>35% of gross profit): reduce turnover, improve entries, or lower fee venue.")
    if stats.get("max_drawdown", 0.0) < -0.25:
        recs.append("Max drawdown exceeds 25%: tighten risk limits and consider lower position sizing.")
    if stats.get("profit_factor", 0.0) < 1.2:
        recs.append("Profit factor is weak (<1.2): review edge quality and avoid low-conviction setups.")
    if stats.get("win_rate", 0.0) < 0.45 and abs(stats.get("avg_loss", 0.0)) >= stats.get("avg_win", 0.0):
        recs.append("Win/loss profile is unfavorable: improve stop logic or let winners run longer.")
    if float(result.data_quality.get("outlier_bars", 0.0)) > 0:
        recs.append("Outlier bars detected: verify candles around spikes and consider robust filters.")
    if float(result.data_quality.get("missing_bars", 0.0)) > 0:
        recs.append("Missing bars detected: fill or remove gaps before comparing strategy variants.")
    if not bool(result.data_quality.get("timezone_aware", False)):
        recs.append("Timestamps are timezone-naive: enforce timezone-aware indexing for session consistency.")
    if not recs:
        recs.append("No major red flags found. Next step: perform walk-forward and out-of-sample validation.")
    return recs


def _render_analysis_page(
    pdf: _SimplePdf,
    result: BacktestResult,
    trades_df: pd.DataFrame,
    robustness: dict[str, object],
) -> None:
    page = pdf.new_page()
    equity = result.equity_curve.astype(float)
    net_profit = float(equity.iloc[-1] - equity.iloc[0])
    closed_trade_net = float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0
    reconciliation_gap = float(net_profit - closed_trade_net)
    largest_win = float(trades_df["pnl"].max()) if not trades_df.empty else 0.0
    largest_loss = float(trades_df["pnl"].min()) if not trades_df.empty else 0.0

    _add_text(page, 40, 560, "Performance Diagnostics & Recommendations", 16)
    _add_text(page, 40, 535, "Reconciliation:", 12)
    _add_text(page, 50, 518, f"Closed-trade PnL (net of all allocated fees/funding): {closed_trade_net:,.2f}")
    _add_text(page, 50, 503, f"Net profit (final - starting equity): {net_profit:,.2f}")
    _add_text(page, 50, 488, f"Total fees paid: {result.total_fees_paid:,.2f}")
    _add_text(page, 50, 473, f"Total interest/funding paid: {result.total_financing_paid:,.2f}")
    _add_text(page, 50, 458, f"Gross closed-trade profit before fees: {result.total_profit_before_fees:,.2f}")
    _add_text(page, 50, 443, f"Reconciliation gap (should be 0.00): {reconciliation_gap:,.2f}")

    _add_text(page, 40, 420, "Trade distribution:", 12)
    _add_text(page, 50, 403, f"Largest win: {largest_win:,.2f}")
    _add_text(page, 50, 388, f"Largest loss: {largest_loss:,.2f}")
    _add_text(page, 50, 373, f"Average trade PnL: {result.stats.get('avg_trade_pnl', 0.0):,.2f}")
    _add_text(page, 50, 358, f"Expectancy per trade: {result.stats.get('expectancy', 0.0):,.2f}")
    trade_diag = compute_trade_diagnostics(
        trades_df=trades_df,
        initial_capital=float(equity.iloc[0]),
        total_fees_paid=result.total_fees_paid,
        execution_events=result.execution_events,
        slippage_rate=float(result.stats.get("slippage_rate", 0.0)),
    )
    _add_text(page, 50, 343, f"Mean position size (USD): {trade_diag['mean_position_size_usd']:,.2f}")
    _add_text(page, 50, 328, f"Median position size (USD): {trade_diag['median_position_size_usd']:,.2f}")

    _add_text(page, 40, 305, "Data quality / outlier report:", 12)
    _add_text(page, 50, 288, f"Datetime index: {bool(result.data_quality.get('is_datetime_index', False))}")
    _add_text(page, 50, 273, f"Timezone aware: {bool(result.data_quality.get('timezone_aware', False))}")
    _add_text(page, 50, 258, f"Duplicate timestamps: {float(result.data_quality.get('duplicate_timestamps', 0.0)):.0f}")
    _add_text(page, 50, 243, f"Missing bars: {float(result.data_quality.get('missing_bars', 0.0)):.0f}")
    _add_text(page, 50, 228, f"Outlier bars (|z|>5 on close returns): {float(result.data_quality.get('outlier_bars', 0.0)):.0f}")

    _add_text(page, 40, 203, "Recommendations:", 12)
    y = 186
    for rec in _build_recommendations(result, trades_df):
        _add_text(page, 50, y, f"- {rec}")
        y -= 15
        if y < 60:
            break

    _add_text(page, 430, 560, "Robustness warnings:", 12)
    wrn_y = 543
    warning_lines = []
    if bool(robustness.get("sqn_warning", False)):
        warning_lines.append("- SQN <= 2.5: limited statistical edge")
    if bool(robustness.get("top_trade_warning", False)):
        warning_lines.append("- CAGR drops >40% without top 5% winners")
    if bool(robustness.get("sign_flip_fragile", False)):
        warning_lines.append("- Sign-flip stress test indicates fragility")
    if bool(robustness.get("regime_warning", False)):
        warning_lines.append("- Large first-vs-second-half performance drift")
    if not warning_lines:
        warning_lines.append("- No robustness warnings triggered")
    for line in warning_lines[:8]:
        _add_text(page, 440, wrn_y, line, 9)
        wrn_y -= 14

def generate_backtest_pdf_report(
    result: BacktestResult,
    output_path: str | Path,
    title: str = "Backtest Performance Report",
    underlying_prices: pd.Series | pd.DataFrame | dict[str, pd.Series] | None = None,
    cli_flags: dict[str, object] | None = None,
    asset_level_results: list[tuple[str, BacktestResult]] | None = None,
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    pdf = _SimplePdf()

    page1 = pdf.new_page()
    equity = result.equity_curve.astype(float)
    initial_capital = float(equity.iloc[0])
    trades_df = result.trades_dataframe()
    diagnostics = compute_trade_diagnostics(
        trades_df=trades_df,
        initial_capital=initial_capital,
        total_fees_paid=result.total_fees_paid,
        execution_events=result.execution_events,
        slippage_rate=float(result.stats.get("slippage_rate", 0.0)),
    )
    max_effective_leverage_used = _estimate_max_effective_leverage(result, trades_df, initial_capital)

    _add_text(page1, 40, 560, title, 18)
    _add_text(page1, 40, 535, f"Bars processed: {len(equity)}")
    _add_text(page1, 40, 520, f"Trades executed: {len(result.trades)}")
    _add_text(page1, 40, 505, f"Starting capital: {equity.iloc[0]:,.2f}")
    _add_text(page1, 40, 490, f"Total capital (final equity): {equity.iloc[-1]:,.2f}")
    _add_text(page1, 40, 475, f"Max capital: {equity.max():,.2f}")
    _add_text(page1, 40, 460, f"Min capital: {equity.min():,.2f}")
    _add_text(page1, 40, 445, f"Total fees paid: {result.total_fees_paid:,.2f}")
    _add_text(page1, 40, 430, f"Total interest/funding paid: {result.total_financing_paid:,.2f}")
    _add_text(page1, 40, 415, f"Total profit before fees: {result.total_profit_before_fees:,.2f}")
    _add_text(page1, 40, 400, f"Duplicate timestamps: {float(result.data_quality.get('duplicate_timestamps', 0.0)):.0f}")
    _add_text(page1, 40, 385, f"Missing bars: {float(result.data_quality.get('missing_bars', 0.0)):.0f}")
    _add_text(page1, 40, 370, f"Outlier bars: {float(result.data_quality.get('outlier_bars', 0.0)):.0f}")
    _add_text(page1, 40, 355, f"Total volume traded: {diagnostics['total_cumulative_volume']:,.2f}")
    _add_text(page1, 40, 340, f"Max effective leverage used: {max_effective_leverage_used:.2f}x")
    _add_text(page1, 40, 320, "Key stats:", 12)

    _add_text(page1, 430, 535, "Position/trade stats:", 12)
    _add_text(page1, 440, 520, f"Mean position size (USD): {diagnostics['mean_position_size_usd']:,.2f}")
    _add_text(page1, 440, 505, f"Median position size (USD): {diagnostics['median_position_size_usd']:,.2f}")
    _add_text(page1, 440, 490, f"Mean trade PnL (USD/%): {diagnostics['mean_trade_pnl_usd']:,.2f} / {diagnostics['mean_trade_pnl_pct'] * 100:.2f}%")
    _add_text(page1, 440, 475, f"Median trade PnL (USD/%): {diagnostics['median_trade_pnl_usd']:,.2f} / {diagnostics['median_trade_pnl_pct'] * 100:.2f}%")
    _add_text(page1, 440, 460, f"Total slippage paid (est): {diagnostics['total_slippage_paid']:,.2f}")
    _add_text(page1, 440, 445, f"Fees per trade: {diagnostics['fees_per_trade']:,.2f}")
    _add_text(page1, 440, 430, f"Slippage per trade (est): {diagnostics['slippage_per_trade']:,.2f}")
    signal_rows = _entry_exit_signal_rows(trades_df)
    _draw_signal_outcomes_table(page1, signal_rows, x=430, y_top=405)

    y = 302
    for key, value in result.stats.items():
        pretty = key.replace("_", " ").title()
        if key in {"total_return", "cagr", "volatility", "max_drawdown", "win_rate", "exposure"}:
            rendered = f"{value * 100:.2f}%"
        elif key == "total_trades":
            rendered = str(int(value))
        else:
            rendered = f"{value:.4f}"
        _add_text(page1, 50, y, f"{pretty}: {rendered}")
        y -= 14
        if y < 70:
            break

    page2 = pdf.new_page()
    rolling_max = equity.cummax()
    drawdown = ((equity / rolling_max) - 1).tolist()

    equity_vals = equity.astype(float).tolist()
    equity_y_scale = "log" if _detect_log_growth(equity_vals) else "linear"
    _draw_chart_axes(page2, 40, 320, 760, 210, "Equity Over Time", min(equity), max(equity), "Equity", "Date", x_tick_labels=_date_axis_labels(equity.index), y_scale=equity_y_scale)
    _draw_series(page2, equity_vals, 40, 320, 760, 210, (0.90, 0.10, 0.20), y_min=min(equity), y_max=max(equity), y_scale=equity_y_scale)
    _draw_trade_entry_exit_lines(page2, result.trades, equity.index, equity, 40, 320, 760, 210, y_min=min(equity), y_max=max(equity), y_scale=equity_y_scale)
    _add_text(page2, 560, 536, "Trade lines: long=green, short=red", 9)

    _draw_chart_axes(page2, 40, 60, 760, 210, "Drawdown Over Time", min(drawdown), max(drawdown), "Drawdown", "Date", x_tick_labels=_date_axis_labels(equity.index))
    _draw_series(page2, drawdown, 40, 60, 760, 210, (0.15, 0.35, 0.85))

    page3 = pdf.new_page()
    robustness = _build_robustness_diagnostics(result, trades_df)
    pnl: list[float] = []
    cum_pnl: list[float] = []
    trade_returns_pct: list[float] = []
    if not trades_df.empty:
        pnl = trades_df["pnl"].astype(float).tolist()
        trade_returns_pct = (trades_df["return_pct"].astype(float) * 100.0).tolist()
        running = 0.0
        for p in pnl:
            running += p
            cum_pnl.append(running)

    cum_plot = cum_pnl if len(cum_pnl) >= 2 else [0.0, 0.0]
    pnl_plot = pnl if len(pnl) >= 2 else [0.0, 0.0]
    _draw_chart_axes(
        page3,
        40,
        320,
        760,
        210,
        "Cumulative Closed-Trade PnL",
        min(cum_plot),
        max(cum_plot),
        "PnL",
        "Trade #",
        x_tick_labels=[(0.0, "1"), (1.0, str(len(cum_plot)))],
    )
    _draw_series(page3, cum_plot, 40, 320, 760, 210, (0.12, 0.62, 0.12))

    _draw_chart_axes(
        page3,
        40,
        60,
        760,
        210,
        "Trade-by-Trade PnL",
        min(pnl_plot),
        max(pnl_plot),
        "PnL",
        "Trade #",
        x_tick_labels=[(0.0, "1"), (1.0, str(len(pnl_plot)))],
    )
    _draw_series(page3, pnl_plot, 40, 60, 760, 210, (0.60, 0.20, 0.78))

    page_return_dist = pdf.new_page()
    _add_text(page_return_dist, 40, 560, "Trade Return Distribution", 16)
    page_return_dist_usd = pdf.new_page()
    _add_text(page_return_dist_usd, 40, 560, "Trade Return Distribution (USD)", 16)

    if trade_returns_pct:
        sorted_pairs = sorted(zip(trade_returns_pct, pnl), key=lambda pair: pair[0])
        sorted_returns = [pair[0] for pair in sorted_pairs]
        sorted_pnl = [pair[1] for pair in sorted_pairs]
        sorted_pnl_only = sorted(pnl)

        ret_min = min(sorted_returns)
        ret_max = max(sorted_returns)
        if ret_min == ret_max:
            ret_min -= 1.0
            ret_max += 1.0

        x_ticks = [
            (0.0, "Worst"),
            (0.5, "Median"),
            (1.0, "Best"),
        ]

        _draw_chart_axes(
            page_return_dist,
            40,
            300,
            760,
            220,
            "Sorted Trade Return % (Worst to Best)",
            ret_min,
            ret_max,
            "Return %",
            "Trades",
            x_tick_labels=x_ticks,
        )
        _draw_bars(page_return_dist, sorted_returns, 40, 300, 760, 220, (0.35, 0.45, 0.85), ret_min, ret_max)

        total_positive_pnl = float(sum(v for v in sorted_pnl if v > 0))
        total_negative_pnl = float(sum(v for v in sorted_pnl if v < 0))
        net_pnl = float(sum(sorted_pnl))
        largest_win = max(sorted_pnl)
        largest_loss = min(sorted_pnl)

        _add_text(page_return_dist, 40, 270, f"Trades analyzed: {len(sorted_returns)}", 10)
        _add_text(page_return_dist, 40, 255, f"Largest winner (PnL): {largest_win:,.2f}", 10)
        _add_text(page_return_dist, 40, 240, f"Largest loser (PnL): {largest_loss:,.2f}", 10)
        _add_text(page_return_dist, 360, 270, f"Gross positive PnL contribution: {total_positive_pnl:,.2f}", 10)
        _add_text(page_return_dist, 360, 255, f"Gross negative PnL contribution: {total_negative_pnl:,.2f}", 10)
        _add_text(page_return_dist, 360, 240, f"Net closed-trade PnL: {net_pnl:,.2f}", 10)

        top_n = min(5, len(sorted_pairs))
        winners = sorted(sorted_pairs, key=lambda pair: pair[1], reverse=True)[:top_n]
        losers = sorted(sorted_pairs, key=lambda pair: pair[1])[:top_n]

        _add_text(page_return_dist, 40, 210, "Top profit contributors (PnL / return %):", 11)
        y = 194
        for idx, (ret, trade_pnl) in enumerate(winners, start=1):
            _add_text(page_return_dist, 50, y, f"{idx}. {trade_pnl:,.2f} / {ret:.2f}%", 9)
            y -= 13

        _add_text(page_return_dist, 360, 210, "Top loss contributors (PnL / return %):", 11)
        y = 194
        for idx, (ret, trade_pnl) in enumerate(losers, start=1):
            _add_text(page_return_dist, 370, y, f"{idx}. {trade_pnl:,.2f} / {ret:.2f}%", 9)
            y -= 13

        _add_text(page_return_dist, 40, 62, "Bars are sorted by return %. Negative bars are below zero; positive bars are above zero.", 9)

        pnl_min = min(sorted_pnl_only)
        pnl_max = max(sorted_pnl_only)
        if pnl_min == pnl_max:
            pnl_min -= 1.0
            pnl_max += 1.0

        _draw_chart_axes(
            page_return_dist_usd,
            40,
            300,
            760,
            220,
            "Sorted Trade PnL in USD (Worst to Best)",
            pnl_min,
            pnl_max,
            "PnL (USD)",
            "Trades",
            x_tick_labels=x_ticks,
        )
        _draw_bars(page_return_dist_usd, sorted_pnl_only, 40, 300, 760, 220, (0.20, 0.62, 0.72), pnl_min, pnl_max)

        _add_text(page_return_dist_usd, 40, 270, f"Trades analyzed: {len(sorted_pnl_only)}", 10)
        _add_text(page_return_dist_usd, 40, 255, f"Largest winner (USD): {max(sorted_pnl_only):,.2f}", 10)
        _add_text(page_return_dist_usd, 40, 240, f"Largest loser (USD): {min(sorted_pnl_only):,.2f}", 10)
        _add_text(page_return_dist_usd, 360, 270, f"Gross positive PnL: {total_positive_pnl:,.2f}", 10)
        _add_text(page_return_dist_usd, 360, 255, f"Gross negative PnL: {total_negative_pnl:,.2f}", 10)
        _add_text(page_return_dist_usd, 360, 240, f"Net closed-trade PnL: {net_pnl:,.2f}", 10)
        _add_text(page_return_dist_usd, 40, 62, "Bars are sorted by trade PnL in USD. Negative bars are below zero; positive bars are above zero.", 9)
    else:
        _add_text(page_return_dist, 40, 520, "No closed trades available to render return distribution.", 10)
        _add_text(page_return_dist_usd, 40, 520, "No closed trades available to render USD return distribution.", 10)

    page_turnover = pdf.new_page()
    _add_text(page_turnover, 40, 560, "Volume / Fees / Slippage Turnover", 16)
    turnover_lines = [
        f"Total cumulative volume (USD): {diagnostics['total_cumulative_volume']:,.2f}",
        f"Total cumulative fees (USD): {diagnostics['total_cumulative_fees']:,.2f}",
        f"Total cumulative slippage (USD est): {diagnostics['total_cumulative_slippage']:,.2f}",
        f"Mean volume per trade (USD): {diagnostics['mean_volume_per_trade']:,.2f}",
        f"Median volume per trade (USD): {diagnostics['median_volume_per_trade']:,.2f}",
        f"Mean fee per trade (USD): {diagnostics['mean_fee_per_trade']:,.2f}",
        f"Median fee per trade (USD): {diagnostics['median_fee_per_trade']:,.2f}",
        f"Mean slippage per trade (USD est): {diagnostics['mean_slippage_per_trade']:,.2f}",
        f"Median slippage per trade (USD est): {diagnostics['median_slippage_per_trade']:,.2f}",
    ]
    y_turn = 536
    for line in turnover_lines:
        _add_text(page_turnover, 40, y_turn, line, 10)
        y_turn -= 14

    trade_count = len(trades_df)
    if trade_count > 0:
        trade_volume = (
            trades_df["entry_price"].astype(float).abs() * trades_df["units"].astype(float).abs()
            + trades_df["exit_price"].astype(float).abs() * trades_df["units"].astype(float).abs()
        ).astype(float)
        fee_rate = float(result.total_fees_paid / diagnostics['total_cumulative_volume']) if diagnostics['total_cumulative_volume'] > 0 else 0.0
        trade_fee = trade_volume * fee_rate
        trade_slippage = trade_volume * float(result.stats.get("slippage_rate", 0.0))

        cumulative_volume = trade_volume.cumsum().tolist()
        cumulative_fees = trade_fee.cumsum().tolist()
        cumulative_slippage = trade_slippage.cumsum().tolist()
        x_ticks = [(0.0, "1"), (1.0, str(trade_count))]

        cumulative_volume_scale = "log" if _detect_log_growth(cumulative_volume) else "linear"
        _draw_chart_axes(
            page_turnover,
            40,
            290,
            760,
            110,
            "Cumulative Volume by Trade",
            min(cumulative_volume),
            max(cumulative_volume),
            "USD",
            "Trade #",
            x_tick_labels=x_ticks,
            y_scale=cumulative_volume_scale,
        )
        _draw_series(page_turnover, cumulative_volume, 40, 290, 760, 110, (0.10, 0.35, 0.80), y_min=min(cumulative_volume), y_max=max(cumulative_volume), y_scale=cumulative_volume_scale)

        cumulative_cost_y_min = min(min(cumulative_fees), min(cumulative_slippage))
        cumulative_cost_y_max = max(max(cumulative_fees), max(cumulative_slippage))
        cumulative_cost_scale = "log" if _detect_log_growth(cumulative_fees) or _detect_log_growth(cumulative_slippage) else "linear"
        _draw_chart_axes(
            page_turnover,
            40,
            155,
            760,
            110,
            "Cumulative Fees & Slippage by Trade",
            cumulative_cost_y_min,
            cumulative_cost_y_max,
            "USD",
            "Trade #",
            x_tick_labels=x_ticks,
            y_scale=cumulative_cost_scale,
        )
        _draw_series(page_turnover, cumulative_fees, 40, 155, 760, 110, (0.80, 0.40, 0.10), y_min=cumulative_cost_y_min, y_max=cumulative_cost_y_max, y_scale=cumulative_cost_scale)
        _draw_series(page_turnover, cumulative_slippage, 40, 155, 760, 110, (0.75, 0.10, 0.10), y_min=cumulative_cost_y_min, y_max=cumulative_cost_y_max, y_scale=cumulative_cost_scale)

        _draw_chart_axes(
            page_turnover,
            40,
            20,
            760,
            110,
            "Per-Trade Volume / Fees / Slippage",
            min(float(trade_volume.min()), float(trade_fee.min()), float(trade_slippage.min())),
            max(float(trade_volume.max()), float(trade_fee.max()), float(trade_slippage.max())),
            "USD",
            "Trade #",
            x_tick_labels=x_ticks,
        )
        _draw_series(page_turnover, trade_volume.tolist(), 40, 20, 760, 110, (0.20, 0.50, 0.85))
        _draw_series(page_turnover, trade_fee.tolist(), 40, 20, 760, 110, (0.85, 0.50, 0.15))
        _draw_series(page_turnover, trade_slippage.tolist(), 40, 20, 760, 110, (0.85, 0.15, 0.15))
        _add_text(page_turnover, 46, 10, "Blue=volume, Orange=fees, Red=slippage", 8)
    else:
        _add_text(page_turnover, 40, 250, "No trades available to render turnover charts.", 10)

    if asset_level_results:
        _render_asset_level_summary_pages(pdf, asset_level_results)

    _render_analysis_page(pdf, result, trades_df, robustness)

    page_robust = pdf.new_page()
    _add_text(page_robust, 40, 560, "Robustness Summary", 16)
    _add_text(page_robust, 40, 538, f"SQN: {float(robustness['sqn']):.2f} ({robustness['sqn_interpretation']})")
    _add_text(page_robust, 40, 523, f"Top-trade dependency warning: {bool(robustness['top_trade_warning'])}")
    _add_text(page_robust, 40, 508, f"Sign-flip stress warning: {bool(robustness['sign_flip_fragile'])}")
    _add_text(page_robust, 40, 493, f"Regime dependency warning: {bool(robustness['regime_warning'])}")
    _add_text(page_robust, 40, 478, f"Robustness score: {int(robustness['score'])}/4 ({robustness['classification']})")

    _add_text(page_robust, 40, 452, "Top 5% Winners Removal Test", 12)
    reduced = robustness["reduced_metrics"]
    _add_text(page_robust, 50, 435, f"Trades removed: {int(robustness['removed_count'])}")
    _add_text(page_robust, 50, 420, f"CAGR drop: {float(robustness['cagr_drop_pct']):.2f}%")
    _add_text(page_robust, 50, 405, f"Reduced CAGR: {_percent_label(float(reduced['cagr']))}")
    _add_text(page_robust, 50, 390, f"Reduced Sharpe: {float(reduced['sharpe']):.2f}")
    _add_text(page_robust, 50, 375, f"Reduced Profit Factor: {float(reduced['profit_factor']):.2f}")
    _add_text(page_robust, 50, 360, f"Reduced Max Drawdown: {float(reduced['max_drawdown']) * 100:.2f}%")

    _add_text(page_robust, 40, 334, "Sign-Flip Stress Test (100 runs, 15% flips)", 12)
    _add_text(page_robust, 50, 317, f"Average CAGR: {_percent_label(float(robustness['avg_sign_flip_cagr']))}")
    _add_text(page_robust, 50, 302, f"Median CAGR: {_percent_label(float(robustness['median_sign_flip_cagr']))}")
    _add_text(page_robust, 50, 287, f"Average Max Drawdown: {_percent_label(float(robustness['avg_sign_flip_max_dd']))}")

    first = robustness['first_half']
    second = robustness['second_half']
    _add_text(page_robust, 40, 261, "First Half vs Second Half", 12)
    _add_text(page_robust, 50, 244, f"First half CAGR/Sharpe: {_percent_label(float(first['cagr']))} / {float(first['sharpe']):.2f}")
    _add_text(page_robust, 50, 229, f"Second half CAGR/Sharpe: {_percent_label(float(second['cagr']))} / {float(second['sharpe']):.2f}")
    _add_text(page_robust, 50, 214, f"CAGR diff: {float(robustness['cagr_diff_pct']):.2f}%")
    _add_text(page_robust, 50, 199, f"Sharpe diff: {float(robustness['sharpe_diff_pct']):.2f}%")

    _add_text(page_robust, 380, 452, "Robustness Checks Table", 12)
    _add_text(page_robust, 390, 435, "Test")
    _add_text(page_robust, 590, 435, "Result")
    _add_text(page_robust, 760, 435, "Status")
    checks = [
        ("SQN", f"{float(robustness['sqn']):.2f}", "PASS" if not bool(robustness["sqn_warning"]) else "WARN"),
        (
            "Top 5% winners",
            f"{float(robustness['cagr_drop_pct']):.1f}% CAGR drop",
            "PASS" if not bool(robustness["top_trade_warning"]) else "WARN",
        ),
        (
            "Sign-flip stress",
            f"median {_percent_label(float(robustness['median_sign_flip_cagr']))} CAGR",
            "PASS" if not bool(robustness["sign_flip_fragile"]) else "WARN",
        ),
        (
            "Half stability",
            f"CAGR {float(robustness['cagr_diff_pct']):.1f}% / Sharpe {float(robustness['sharpe_diff_pct']):.1f}%",
            "PASS" if not bool(robustness["regime_warning"]) else "WARN",
        ),
    ]
    row_y = 420
    for name, value, status in checks:
        _add_text(page_robust, 390, row_y, name, 9)
        _add_text(page_robust, 590, row_y, value, 9)
        _add_text(page_robust, 760, row_y, status, 9)
        row_y -= 14

    page_diag = pdf.new_page()
    reduced_equity = robustness["reduced_equity"]
    reduced_vals = reduced_equity.astype(float).tolist() if len(reduced_equity) >= 2 else [initial_capital, initial_capital]
    _draw_chart_axes(page_diag, 40, 330, 760, 200, "Equity Curve Without Top 5% Trades", min(reduced_vals), max(reduced_vals), "Equity", "Trade #")
    _draw_series(page_diag, reduced_vals, 40, 330, 760, 200, (0.85, 0.45, 0.12))

    sf = robustness['sign_flip_cagrs']
    sf_vals = sorted(sf) if len(sf) >= 2 else [0.0, 0.0]
    _draw_chart_axes(page_diag, 40, 80, 360, 200, "Sign-Flip CAGR Distribution", min(sf_vals), max(sf_vals), "CAGR", "Run #")
    _draw_series(page_diag, sf_vals, 40, 80, 360, 200, (0.25, 0.55, 0.9))

    half_vals = [float(first['cagr']) * 100, float(second['cagr']) * 100, float(first['sharpe']), float(second['sharpe'])]
    _draw_chart_axes(page_diag, 440, 80, 360, 200, "Half Comparison (CAGR%, Sharpe)", min(half_vals), max(half_vals), "Value", "Metrics")
    _draw_series(page_diag, half_vals, 440, 80, 360, 200, (0.55, 0.2, 0.75))

    underlying_index = _combined_underlying_index(underlying_prices, equity.index)
    if underlying_index is not None and len(underlying_index) >= 2:
        page4 = pdf.new_page()
        eq_vals = equity.tolist()
        under_vals = underlying_index.astype(float).tolist()

        _draw_chart_axes(
            page4,
            40,
            90,
            760,
            430,
            "Portfolio Equity vs Underlying (Normalized Index)",
            min(eq_vals),
            max(eq_vals),
            "Equity",
            "Date",
            right_y_min=min(under_vals),
            right_y_max=max(under_vals),
            right_y_label="Underlying (Norm=100)",
            x_tick_labels=_date_axis_labels(equity.index),
        )
        _draw_series(page4, eq_vals, 40, 90, 760, 430, (0.85, 0.15, 0.15))
        _draw_trade_entry_exit_lines(page4, result.trades, equity.index, equity, 40, 90, 760, 430)
        _draw_series(page4, under_vals, 40, 90, 760, 430, (0.10, 0.25, 0.80))
        _add_text(page4, 60, 68, "Red=Equity (left axis), Blue=Underlying (right axis)", 9)

    if cli_flags:
        items = sorted(cli_flags.items(), key=lambda kv: str(kv[0]))
        per_page = 36
        for idx, start in enumerate(range(0, len(items), per_page), start=1):
            p_flags = pdf.new_page()
            suffix = "" if len(items) <= per_page else f" (page {idx})"
            _add_text(p_flags, 40, 560, f"CLI Flags Used for This Backtest{suffix}", 16)
            _add_text(p_flags, 40, 542, "All command-line options and values for reproducibility.", 10)
            y = 522
            for key, value in items[start:start + per_page]:
                _add_text(p_flags, 40, y, f"--{str(key).replace('_', '-')}: {value}", 9)
                y -= 14

    pdf.save(out)
    return out


def generate_backtest_clean_pdf_report(
    result: BacktestResult,
    output_path: str | Path,
    title: str = "Trade Signal Diagnostics Report",
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    pdf = _SimplePdf()
    page = pdf.new_page()
    equity = result.equity_curve.astype(float)
    returns = result.returns.astype(float) if isinstance(result.returns, pd.Series) else equity.pct_change().fillna(0.0)
    trades_df = result.trades_dataframe()
    stats = result.stats
    giveback = _trade_giveback_summary(trades_df)
    net_profit = float(equity.iloc[-1] - equity.iloc[0])

    _add_text(page, 40, 560, title, 18)
    _add_text(page, 40, 536, f"Bars processed: {len(equity)}", 11)
    _add_text(page, 40, 520, f"Trades executed: {len(result.trades)}", 11)
    _add_text(page, 40, 504, f"Starting capital: {float(equity.iloc[0]):,.2f}", 11)
    _add_text(page, 40, 488, f"Final capital: {float(equity.iloc[-1]):,.2f}", 11)
    _add_text(page, 40, 472, f"Net profit (final - start): {net_profit:,.2f}", 11)
    _add_text(page, 40, 456, f"Max drawdown: {float(stats.get('max_drawdown', 0.0)) * 100:.2f}%", 11)
    _add_text(page, 40, 440, f"Win rate: {float(stats.get('win_rate', 0.0)) * 100:.2f}%", 11)
    _add_text(page, 40, 424, f"Sharpe ratio: {float(stats.get('sharpe', 0.0)):.2f}", 11)

    _add_text(page, 40, 394, "Signal quality monitor", 13)
    _add_text(page, 50, 376, f"Avg peak unrealized return: {giveback['avg_peak_unrealized_return_pct'] * 100:.2f}%", 10)
    _add_text(page, 50, 362, f"Avg giveback from peak: {giveback['avg_giveback_from_peak_pct'] * 100:.2f}%", 10)
    _add_text(page, 50, 348, f"Avg capture vs peak: {giveback['avg_capture_ratio_vs_peak'] * 100:.2f}%", 10)
    _add_text(page, 50, 334, f"Median capture vs peak: {giveback['median_capture_ratio_vs_peak'] * 100:.2f}%", 10)
    _add_text(page, 50, 320, f"Trades giving back >=50% of peak: {giveback['high_giveback_trade_rate'] * 100:.2f}%", 10)

    _add_text(page, 40, 286, "This compact report is intended for optimization reviews.", 10)
    _add_text(page, 40, 272, "Use the full report for full charts and robustness diagnostics.", 10)

    eq_vals = equity.astype(float).tolist()
    drawdown_series = (equity / equity.cummax()) - 1.0
    drawdown_vals = drawdown_series.astype(float).tolist()
    chart_page = pdf.new_page()
    _add_text(chart_page, 40, 560, "Diagnostics Charts: Equity, Drawdown, Return Distribution", 14)
    _draw_chart_axes(
        chart_page,
        40,
        330,
        760,
        190,
        "Equity Curve",
        min(eq_vals) if eq_vals else 0.0,
        max(eq_vals) if eq_vals else 1.0,
        "Equity",
        "Date",
        x_tick_labels=_date_axis_labels(equity.index),
    )
    _draw_series(chart_page, eq_vals if len(eq_vals) >= 2 else [0.0, 0.0], 40, 330, 760, 190, (0.82, 0.16, 0.18))

    _draw_chart_axes(
        chart_page,
        40,
        80,
        360,
        180,
        "Drawdown",
        min(drawdown_vals) if drawdown_vals else -1.0,
        max(drawdown_vals) if drawdown_vals else 0.0,
        "Drawdown",
        "Date",
        x_tick_labels=_date_axis_labels(equity.index),
    )
    _draw_series(chart_page, drawdown_vals if len(drawdown_vals) >= 2 else [0.0, 0.0], 40, 80, 360, 180, (0.18, 0.38, 0.80))

    return_vals = (returns.astype(float) * 100.0).tolist()
    sorted_returns = sorted(return_vals) if return_vals else [0.0, 0.0]
    ret_min, ret_max = min(sorted_returns), max(sorted_returns)
    if ret_min == ret_max:
        ret_min -= 1.0
        ret_max += 1.0
    _draw_chart_axes(
        chart_page,
        440,
        80,
        360,
        180,
        "Bar Return % (Sorted)",
        ret_min,
        ret_max,
        "Return %",
        "Bars",
        x_tick_labels=[(0.0, "Worst"), (0.5, "Median"), (1.0, "Best")],
    )
    _draw_bars(chart_page, sorted_returns, 440, 80, 360, 180, (0.30, 0.58, 0.30), ret_min, ret_max)

    if not trades_df.empty:
        frame = trades_df.copy()
        frame["entry_signal"] = frame.get("entry_signal", pd.Series("Unknown", index=frame.index)).fillna("Unknown").astype(str)
        frame["exit_signal"] = frame.get("exit_signal", pd.Series("Unknown", index=frame.index)).fillna("Unknown").astype(str)
        frame["holding_bars"] = pd.to_numeric(frame.get("holding_bars", 0.0), errors="coerce").fillna(0.0)
        frame["pnl"] = pd.to_numeric(frame.get("pnl", 0.0), errors="coerce").fillna(0.0)
        frame["is_win"] = frame["pnl"] > 0
        frame["is_stop_like"] = frame["exit_signal"].str.lower().str.contains("stop|liquidation", regex=True)
        frame["is_early_stop"] = frame["is_stop_like"] & (frame["holding_bars"] <= 2)
        frame["giveback_from_peak_pct"] = pd.to_numeric(frame.get("giveback_from_peak_pct", 0.0), errors="coerce").fillna(0.0)

        by_entry = (
            frame.groupby("entry_signal", dropna=False)
            .agg(
                trades=("pnl", "size"),
                win_rate=("is_win", "mean"),
                stop_exit_rate=("is_stop_like", "mean"),
                early_stop_rate=("is_early_stop", "mean"),
                avg_giveback_pct=("giveback_from_peak_pct", "mean"),
                net_pnl=("pnl", "sum"),
            )
            .reset_index()
            .sort_values(["stop_exit_rate", "trades"], ascending=[False, False])
        )
        by_exit = (
            frame.groupby("exit_signal", dropna=False)
            .agg(
                trades=("pnl", "size"),
                win_rate=("is_win", "mean"),
                avg_giveback_pct=("giveback_from_peak_pct", "mean"),
                net_pnl=("pnl", "sum"),
            )
            .reset_index()
            .sort_values(["trades", "net_pnl"], ascending=[False, True])
        )

        trade_chart_page = pdf.new_page()
        _add_text(trade_chart_page, 40, 560, "Diagnostics Charts: Percent Histograms", 14)
        peak_vals = (pd.to_numeric(frame.get("peak_unrealized_return_pct", 0.0), errors="coerce").fillna(0.0) * 100.0).tolist()
        giveback_vals = (pd.to_numeric(frame.get("giveback_from_peak_pct", 0.0), errors="coerce").fillna(0.0) * 100.0).tolist()
        realized_vals = (
            pd.to_numeric(
                frame.get(
                    "realized_price_return_pct",
                    frame.get("return_pct", 0.0),
                ),
                errors="coerce",
            ).fillna(0.0)
            * 100.0
        ).tolist()
        capture_vals = (pd.to_numeric(frame.get("capture_ratio_vs_peak", 0.0), errors="coerce").fillna(0.0) * 100.0).tolist()
        peak_plot = peak_vals if len(peak_vals) >= 2 else [0.0, 0.0]
        giveback_plot = giveback_vals if len(giveback_vals) >= 2 else [0.0, 0.0]
        realized_plot = realized_vals if len(realized_vals) >= 2 else [0.0, 0.0]
        capture_plot = capture_vals if len(capture_vals) >= 2 else [0.0, 0.0]
        _add_text(
            trade_chart_page,
            40,
            542,
            "Histogram Overlay: Peak Unrealized % histogram in background, Realized % histogram overlaid on top.",
            10,
        )

        x_ticks = [(0.0, "1"), (1.0, str(len(peak_plot)))]
        overlay_min, overlay_max = min(min(peak_plot), min(realized_plot)), max(max(peak_plot), max(realized_plot))
        if overlay_min == overlay_max:
            overlay_min -= 1.0
            overlay_max += 1.0
        _draw_chart_axes(
            trade_chart_page,
            40,
            300,
            760,
            210,
            "Peak Unrealized % (Background Histogram) + Realized % (Foreground Histogram)",
            overlay_min,
            overlay_max,
            "Percent %",
            "Trade #",
            x_tick_labels=x_ticks,
        )
        _draw_bars(trade_chart_page, peak_plot, 40, 300, 760, 210, (0.18, 0.58, 0.18), overlay_min, overlay_max)
        _draw_bars(trade_chart_page, realized_plot, 40, 300, 760, 210, (0.78, 0.18, 0.18), overlay_min, overlay_max)
        _add_text(trade_chart_page, 50, 286, "Green bars=Peak UPNL % (background), Red bars=Realized % (foreground)", 9)

        giveback_min, giveback_max = min(giveback_plot), max(giveback_plot)
        if giveback_min == giveback_max:
            giveback_min -= 1.0
            giveback_max += 1.0
        _draw_chart_axes(
            trade_chart_page,
            40,
            60,
            760,
            180,
            "Giveback % Histogram",
            giveback_min,
            giveback_max,
            "Giveback %",
            "Trade #",
            x_tick_labels=x_ticks,
        )
        _draw_bars(trade_chart_page, giveback_plot, 40, 60, 760, 180, (0.78, 0.40, 0.12), giveback_min, giveback_max)

        signal_effects_page = pdf.new_page()
        _add_text(signal_effects_page, 40, 560, "Diagnostics Charts: Position Return % Histograms", 14)
        peak_position_pct_vals = (
            pd.to_numeric(frame.get("peak_unrealized_position_return_pct", 0.0), errors="coerce").fillna(0.0) * 100.0
        ).tolist()
        realized_position_pct_vals = (
            pd.to_numeric(frame.get("realized_position_return_pct", 0.0), errors="coerce").fillna(0.0) * 100.0
        ).tolist()
        giveback_position_pct_vals = (
            pd.to_numeric(frame.get("giveback_position_return_pct", 0.0), errors="coerce").fillna(0.0) * 100.0
        ).tolist()
        peak_position_pct_plot = peak_position_pct_vals if len(peak_position_pct_vals) >= 2 else [0.0, 0.0]
        realized_position_pct_plot = realized_position_pct_vals if len(realized_position_pct_vals) >= 2 else [0.0, 0.0]
        giveback_position_pct_plot = giveback_position_pct_vals if len(giveback_position_pct_vals) >= 2 else [0.0, 0.0]
        pnl_overlay_min = min(min(peak_position_pct_plot), min(realized_position_pct_plot))
        pnl_overlay_max = max(max(peak_position_pct_plot), max(realized_position_pct_plot))
        if pnl_overlay_min == pnl_overlay_max:
            pnl_overlay_min -= 1.0
            pnl_overlay_max += 1.0
        _draw_chart_axes(
            signal_effects_page,
            40,
            300,
            760,
            210,
            "Peak Unrealized Position Return % (Histogram) + Realized Position Return % (Overlay)",
            pnl_overlay_min,
            pnl_overlay_max,
            "Return %",
            "Trade #",
            x_tick_labels=[(0.0, "1"), (1.0, str(len(peak_position_pct_plot)))],
        )
        _draw_bars(signal_effects_page, peak_position_pct_plot, 40, 300, 760, 210, (0.15, 0.55, 0.15), pnl_overlay_min, pnl_overlay_max)
        _draw_bars(signal_effects_page, realized_position_pct_plot, 40, 300, 760, 210, (0.80, 0.18, 0.18), pnl_overlay_min, pnl_overlay_max)
        _add_text(signal_effects_page, 50, 286, "Green bars=Peak unrealized return %, Red bars=Realized return %", 9)

        gb_pnl_min, gb_pnl_max = min(giveback_position_pct_plot), max(giveback_position_pct_plot)
        if gb_pnl_min == gb_pnl_max:
            gb_pnl_min -= 1.0
            gb_pnl_max += 1.0
        _draw_chart_axes(
            signal_effects_page,
            40,
            80,
            760,
            170,
            "Giveback Position Return % Histogram",
            gb_pnl_min,
            gb_pnl_max,
            "Giveback %",
            "Trade #",
            x_tick_labels=[(0.0, "1"), (1.0, str(len(giveback_position_pct_plot)))],
        )
        _draw_bars(signal_effects_page, giveback_position_pct_plot, 40, 80, 760, 170, (0.80, 0.45, 0.12), gb_pnl_min, gb_pnl_max)

        signal_pressure_page = pdf.new_page()
        _add_text(signal_pressure_page, 40, 560, "Diagnostics Charts: Capture and Stop Pressure", 14)
        cap_min, cap_max = min(capture_plot), max(capture_plot)
        if cap_min == cap_max:
            cap_min -= 1.0
            cap_max += 1.0
        _draw_chart_axes(
            signal_pressure_page,
            40,
            320,
            760,
            200,
            "Capture Ratio vs Peak (%)",
            cap_min,
            cap_max,
            "Capture %",
            "Trade #",
            x_tick_labels=[(0.0, "1"), (1.0, str(len(capture_plot)))],
        )
        _draw_series(signal_pressure_page, capture_plot, 40, 320, 760, 200, (0.30, 0.30, 0.78))

        entry_stop_rates = (by_entry["stop_exit_rate"].astype(float) * 100.0).tolist()
        entry_labels = by_entry["entry_signal"].astype(str).tolist()
        top_n = min(8, len(entry_stop_rates))
        top_stop = entry_stop_rates[:top_n] if top_n > 0 else [0.0, 0.0]
        top_stop_labels = entry_labels[:top_n] if top_n > 0 else ["-", "-"]
        stop_min, stop_max = min(top_stop), max(top_stop)
        if stop_min == stop_max:
            stop_min -= 1.0
            stop_max += 1.0
        _draw_chart_axes(
            signal_pressure_page,
            40,
            80,
            760,
            180,
            "Top Entry Stop Rates (%)",
            stop_min,
            stop_max,
            "Stop %",
            "Entry Signal Rank",
            x_tick_labels=[(i / max(1, top_n - 1), str(i + 1)) for i in range(top_n)] if top_n > 1 else [(0.0, "1")],
        )
        _draw_bars(signal_pressure_page, top_stop if top_n > 0 else [0.0, 0.0], 40, 80, 760, 180, (0.72, 0.42, 0.16), stop_min, stop_max)
        label_y = 68
        for idx, label in enumerate(top_stop_labels[:8], start=1):
            _add_text(signal_pressure_page, 40 + ((idx - 1) % 4) * 190, label_y - ((idx - 1) // 4) * 10, f"{idx}. {_truncate_text(label, 20)}", 8)

        page_entry = pdf.new_page()
        _add_text(page_entry, 40, 560, "Entry Signal Diagnostics", 16)
        _add_text(page_entry, 40, 542, "Top entries by stop pressure (higher stop rate appears first).", 10)
        y = 520
        _add_text(page_entry, 40, y, "Entry Signal", 9)
        _add_text(page_entry, 280, y, "Trades", 9)
        _add_text(page_entry, 340, y, "Win %", 9)
        _add_text(page_entry, 410, y, "Stop %", 9)
        _add_text(page_entry, 480, y, "Early Stop %", 9)
        _add_text(page_entry, 580, y, "Avg Giveback %", 9)
        _add_text(page_entry, 700, y, "Net PnL", 9)
        y -= 14
        for row in by_entry.head(26).itertuples(index=False):
            _add_text(page_entry, 40, y, _truncate_text(str(row.entry_signal), 38), 8)
            _add_text(page_entry, 280, y, f"{int(row.trades)}", 8)
            _add_text(page_entry, 340, y, f"{float(row.win_rate) * 100:.1f}", 8)
            _add_text(page_entry, 410, y, f"{float(row.stop_exit_rate) * 100:.1f}", 8)
            _add_text(page_entry, 480, y, f"{float(row.early_stop_rate) * 100:.1f}", 8)
            _add_text(page_entry, 580, y, f"{float(row.avg_giveback_pct) * 100:.1f}", 8)
            _add_text(page_entry, 700, y, f"{float(row.net_pnl):,.0f}", 8)
            y -= 12
            if y < 60:
                break

        page_exit = pdf.new_page()
        _add_text(page_exit, 40, 560, "Exit Signal Diagnostics", 16)
        _add_text(page_exit, 40, 542, "Exit reasons ranked by frequency and loss concentration.", 10)
        y = 520
        _add_text(page_exit, 40, y, "Exit Signal", 9)
        _add_text(page_exit, 340, y, "Trades", 9)
        _add_text(page_exit, 410, y, "Win %", 9)
        _add_text(page_exit, 500, y, "Avg Giveback %", 9)
        _add_text(page_exit, 650, y, "Net PnL", 9)
        y -= 14
        for row in by_exit.head(30).itertuples(index=False):
            _add_text(page_exit, 40, y, _truncate_text(str(row.exit_signal), 46), 8)
            _add_text(page_exit, 340, y, f"{int(row.trades)}", 8)
            _add_text(page_exit, 410, y, f"{float(row.win_rate) * 100:.1f}", 8)
            _add_text(page_exit, 500, y, f"{float(row.avg_giveback_pct) * 100:.1f}", 8)
            _add_text(page_exit, 650, y, f"{float(row.net_pnl):,.0f}", 8)
            y -= 12
            if y < 60:
                break

    pdf.save(out)
    return out
