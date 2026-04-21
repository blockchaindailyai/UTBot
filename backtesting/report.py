from __future__ import annotations

import json
from pathlib import Path

from .engine import BacktestResult


def _escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_simple_pdf(lines: list[str]) -> bytes:
    page_width = 612
    page_height = 792
    font_size = 12
    leading = 16
    start_y = 760

    content_lines = ["BT", f"/F1 {font_size} Tf"]
    y = start_y
    for line in lines:
        safe = _escape_pdf_text(line)
        content_lines.append(f"1 0 0 1 50 {y} Tm ({safe}) Tj")
        y -= leading
        if y < 60:
            break
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1", errors="replace")

    objects: list[bytes] = []

    def add_object(payload: bytes | str) -> int:
        body = payload if isinstance(payload, bytes) else payload.encode("ascii")
        objects.append(body)
        return len(objects)

    font_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    content_id = add_object(
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream"
    )
    page_id = add_object(
        (
            f"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 {page_width} {page_height}] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("ascii")
    )
    pages_id = add_object(f"<< /Type /Pages /Count 1 /Kids [{page_id} 0 R] >>".encode("ascii"))
    objects[page_id - 1] = objects[page_id - 1].replace(
        b"/Parent 0 0 R", f"/Parent {pages_id} 0 R".encode("ascii")
    )
    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("ascii"))

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
    return bytes(pdf)


def _lines(result: BacktestResult) -> list[str]:
    stats = result.stats
    return [
        "Backtest Report",
        "================",
        f"Final equity: {stats.get('final_equity', 0.0):,.2f}",
        f"Total return: {stats.get('total_return', 0.0) * 100:.2f}%",
        f"CAGR: {stats.get('cagr', 0.0) * 100:.2f}%",
        f"Sharpe: {stats.get('sharpe', 0.0):.3f}",
        f"Max drawdown: {stats.get('max_drawdown', 0.0) * 100:.2f}%",
        f"Trades: {int(stats.get('total_trades', 0.0))}",
    ]


def generate_backtest_pdf_report(result: BacktestResult, output_path: str | Path) -> Path:
    """Write a valid single-page PDF report."""
    path = Path(output_path)
    path.write_bytes(_build_simple_pdf(_lines(result)))
    return path


def generate_backtest_clean_pdf_report(result: BacktestResult, output_path: str | Path) -> Path:
    return generate_backtest_pdf_report(result, output_path)


def write_backtest_json_summary(result: BacktestResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.write_text(json.dumps(result.stats, indent=2), encoding="utf-8")
    return path
