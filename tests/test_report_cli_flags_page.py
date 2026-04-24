from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtesting.engine import BacktestConfig, BacktestEngine
from backtesting.report import _Page, _build_x_ticks, _draw_chart_axes, generate_backtest_pdf_report
from backtesting.strategy import MovingAverageCrossStrategy


def test_pdf_report_adds_cli_flags_page(tmp_path) -> None:
    idx = pd.date_range("2024-01-01", periods=80, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0 + i * 0.1 for i in range(80)],
            "high": [100.5 + i * 0.1 for i in range(80)],
            "low": [99.5 + i * 0.1 for i in range(80)],
            "close": [100.0 + i * 0.1 for i in range(80)],
            "volume": [1_000.0] * 80,
        },
        index=idx,
    )

    engine = BacktestEngine(BacktestConfig(initial_capital=10_000.0))
    result = engine.run(data, MovingAverageCrossStrategy(fast_period=5, slow_period=20))
    out = tmp_path / "report.pdf"

    generate_backtest_pdf_report(
        result=result,
        output_path=out,
        cli_flags={"csv": "examples/sample_ohlcv.csv", "strategy": "ma_cross", "ut_ma_filter": True},
    )

    pdf_text = out.read_bytes().decode("latin-1", errors="ignore")
    assert "CLI Flags Set" in pdf_text
    assert "--csv=examples/sample_ohlcv.csv" in pdf_text
    assert "--strategy=ma_cross" in pdf_text
    assert "--ut-ma-filter" in pdf_text
    assert "2024-01-02" in pdf_text
    assert "2024-01-03" in pdf_text
    assert "Mean trade PnL \\(USD/%\\)" in pdf_text
    assert "Median trade PnL \\(USD/%\\)" in pdf_text
    assert "Total slippage paid \\(est\\)" in pdf_text
    assert "Max effective leverage used" in pdf_text
    assert "Avg Holding Bars" in pdf_text
    assert "Exposure" in pdf_text


def test_collect_set_cli_flags_returns_only_non_default_values() -> None:
    module_path = REPO_ROOT / "examples" / "run_backtest.py"
    spec = importlib.util.spec_from_file_location("run_backtest", module_path)
    assert spec and spec.loader
    run_backtest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_backtest)

    parser = run_backtest.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--strategy",
            "ut_bot",
            "--ut-ma-filter",
        ]
    )

    set_flags = run_backtest._collect_set_cli_flags(parser, args)

    assert set_flags["csv"] == "examples/sample_ohlcv.csv"
    assert set_flags["strategy"] == "ut_bot"
    assert set_flags["ut_ma_filter"] is True
    assert "ut_ma_period" not in set_flags


def test_build_x_ticks_uses_real_date_labels() -> None:
    series = pd.Series(
        [float(i) for i in range(10)],
        index=pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
    )

    ticks = _build_x_ticks(series, tick_count=6)

    assert len(ticks) == 6
    assert ticks[0][1] == "2024-01-01"
    assert ticks[-1][1] == "2024-01-10"


def test_draw_chart_axes_adds_granular_y_ticks() -> None:
    page = _Page()
    _draw_chart_axes(
        page,
        chart_x=100.0,
        chart_y=100.0,
        chart_w=200.0,
        chart_h=100.0,
        x_ticks=[(0.0, "2024-01-01"), (1.0, "2024-01-10")],
        y_min=0.0,
        y_max=6.0,
        y_tick_count=7,
    )

    text_commands = [command for command in page.commands if " Tj ET" in command]
    assert len(text_commands) == 9
