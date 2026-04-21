from __future__ import annotations

import json
from pathlib import Path

from .engine import BacktestResult


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
    """Write a minimal text-based report with a .pdf extension for compatibility."""
    path = Path(output_path)
    path.write_text("\n".join(_lines(result)) + "\n", encoding="utf-8")
    return path


def generate_backtest_clean_pdf_report(result: BacktestResult, output_path: str | Path) -> Path:
    return generate_backtest_pdf_report(result, output_path)


def write_backtest_json_summary(result: BacktestResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.write_text(json.dumps(result.stats, indent=2), encoding="utf-8")
    return path
