from __future__ import annotations

import json
from datetime import date, datetime
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

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


def _draw_line(page: _Page, x1: float, y1: float, x2: float, y2: float) -> None:
    page.commands.append(f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")


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
    x_ticks: list[tuple[float, str]],
    y_min: float,
    y_max: float,
    y_tick_count: int = 6,
    y_is_percent: bool = False,
) -> None:
    y_formatter = (lambda value: f"{value * 100:.2f}%") if y_is_percent else (lambda value: f"{value:,.2f}")
    for x_ratio, x_label in x_ticks:
        tick_x = chart_x + (x_ratio * chart_w)
        _draw_line(page, tick_x, chart_y, tick_x, chart_y - 4.0)
        _add_text(page, tick_x - 26.0, chart_y - 14.0, x_label, 7)

    if y_tick_count < 2:
        y_tick_count = 2
    for i in range(y_tick_count):
        y_ratio = i / (y_tick_count - 1)
        tick_y = chart_y + (y_ratio * chart_h)
        tick_value = y_min + (y_ratio * (y_max - y_min))
        _draw_line(page, chart_x - 4.0, tick_y, chart_x, tick_y)
        _add_text(page, chart_x - 58.0, tick_y - 2.0, y_formatter(tick_value), 8)


def _coerce_datetime(value: object) -> datetime | None:
    converted = value
    if hasattr(converted, "to_pydatetime"):
        converted = converted.to_pydatetime()
    if isinstance(converted, np.datetime64):
        converted = converted.astype("datetime64[ms]").tolist()
    return converted if isinstance(converted, datetime) else None


def _format_x_label(value: object, include_time: bool) -> str:
    converted = value
    if hasattr(converted, "to_pydatetime"):
        converted = converted.to_pydatetime()
    if isinstance(converted, np.datetime64):
        converted = converted.astype("datetime64[ms]").tolist()

    if isinstance(converted, datetime):
        return converted.strftime("%Y-%m-%d %H:%M") if include_time else converted.strftime("%Y-%m-%d")
    if isinstance(converted, date):
        return converted.strftime("%Y-%m-%d")

    label = str(value)
    return label[:16] if include_time else label[:10]


def _build_x_ticks(series: object, tick_count: int = 6) -> list[tuple[float, str]]:
    index = getattr(series, "index", None)
    if index is None or len(index) == 0:
        return [(0.0, "0"), (1.0, "0")]

    length = len(index)
    if length == 1:
        return [(0.0, _format_x_label(index[0], include_time=False))]

    tick_count = max(2, min(tick_count, length))
    first_dt = _coerce_datetime(index[0])
    last_dt = _coerce_datetime(index[-1])
    include_time = False
    if first_dt is not None and last_dt is not None:
        include_time = abs((last_dt - first_dt).total_seconds()) <= (3 * 24 * 60 * 60)

    ticks: list[tuple[float, str]] = []
    for raw_position in np.linspace(0, length - 1, tick_count):
        idx = int(round(float(raw_position)))
        x_ratio = idx / max(length - 1, 1)
        ticks.append((x_ratio, _format_x_label(index[idx], include_time=include_time)))
    return ticks


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


def _equity_drawdown_chart(page: _Page, result: BacktestResult) -> None:
    chart_x = 380.0
    chart_y = 320.0
    chart_w = 430.0
    chart_h = 220.0

    _add_text(page, chart_x, chart_y + chart_h + 16, "Equity Curve", 11)
    _draw_rect(page, chart_x, chart_y, chart_w, chart_h)

    equity_points = _series_to_points(result.equity_curve, chart_x + 6, chart_y + 8, chart_w - 12, chart_h - 16)
    _draw_polyline(page, equity_points, (0.10, 0.70, 0.30))
    equity_min, equity_max = _series_bounds(result.equity_curve)
    _draw_chart_axes(
        page,
        chart_x=chart_x,
        chart_y=chart_y,
        chart_w=chart_w,
        chart_h=chart_h,
        x_ticks=_build_x_ticks(result.equity_curve, tick_count=6),
        y_min=equity_min,
        y_max=equity_max,
        y_tick_count=7,
    )

    running_max = result.equity_curve.cummax()
    drawdown = (result.equity_curve / running_max) - 1.0
    dd_x = chart_x
    dd_y = 55.0
    dd_w = chart_w
    dd_h = 220.0

    _add_text(page, dd_x, dd_y + dd_h + 16, "Drawdown", 11)
    _draw_rect(page, dd_x, dd_y, dd_w, dd_h)
    dd_points = _series_to_points(drawdown, dd_x + 6, dd_y + 8, dd_w - 12, dd_h - 16)
    _draw_polyline(page, dd_points, (0.85, 0.20, 0.20))
    dd_min, dd_max = _series_bounds(drawdown)
    _draw_chart_axes(
        page,
        chart_x=dd_x,
        chart_y=dd_y,
        chart_w=dd_w,
        chart_h=dd_h,
        x_ticks=_build_x_ticks(drawdown, tick_count=6),
        y_min=dd_min,
        y_max=dd_max,
        y_tick_count=7,
        y_is_percent=True,
    )


def _stats_panel(page: _Page, result: BacktestResult) -> None:
    _add_text(page, 40, 560, "Backtest Report", 16)
    _add_text(page, 40, 540, "Performance Summary", 12)

    rows = _metric_rows(result)
    y = 520.0
    for label, value in rows:
        _add_text(page, 40, y, label, 10)
        _add_text(page, 220, y, value, 10)
        y -= 18

    if not result.trades:
        _add_text(page, 40, y - 8, "No closed trades were generated for this run.", 10)


def _wrap_text(text: str, width: int) -> list[str]:
    if len(text) <= width:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > width:
        split_at = remaining.rfind(" ", 0, width + 1)
        if split_at <= 0:
            split_at = width
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _cli_flags_panel(page: _Page, cli_flags: dict[str, object] | None) -> None:
    _add_text(page, 40, 560, "Backtest Report", 16)
    _add_text(page, 40, 540, "CLI Flags Set", 12)

    if not cli_flags:
        _add_text(page, 40, 515, "No CLI flags were provided for this run.", 10)
        return

    y = 515.0
    for name in sorted(cli_flags):
        value = cli_flags[name]
        flag_name = f"--{name.replace('_', '-')}"
        if isinstance(value, bool):
            line = flag_name if value else f"{flag_name}=false"
        elif isinstance(value, (list, tuple)):
            joined = ", ".join(str(item) for item in value)
            line = f"{flag_name}={joined}"
        else:
            line = f"{flag_name}={value}"
        for wrapped in _wrap_text(line, width=95):
            _add_text(page, 40, y, wrapped, 10)
            y -= 14
            if y < 40:
                return


def generate_backtest_pdf_report(
    result: BacktestResult,
    output_path: str | Path,
    cli_flags: dict[str, object] | None = None,
) -> Path:
    """Generate a full PDF report with summary statistics and charts."""
    path = Path(output_path)

    doc = _PdfDocument()
    page = doc.new_page()
    _stats_panel(page, result)
    _equity_drawdown_chart(page, result)
    flags_page = doc.new_page()
    _cli_flags_panel(flags_page, cli_flags)

    doc.save(path)
    return path


def generate_backtest_clean_pdf_report(result: BacktestResult, output_path: str | Path) -> Path:
    return generate_backtest_pdf_report(result, output_path)


def write_backtest_json_summary(result: BacktestResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.write_text(json.dumps(result.stats, indent=2), encoding="utf-8")
    return path
