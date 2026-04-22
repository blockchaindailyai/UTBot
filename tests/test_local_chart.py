from __future__ import annotations

import json
import re

import pandas as pd

from backtesting.engine import BacktestResult, Trade
from backtesting.local_chart import generate_local_tradingview_chart


def test_local_chart_includes_ut_bot_overlay(tmp_path) -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")
    close = pd.Series(
        [
            100,
            101,
            103,
            102,
            104,
            106,
            105,
            107,
            106,
            108,
            110,
            109,
            111,
            113,
            112,
            114,
            113,
            115,
            117,
            116,
        ],
        index=idx,
        dtype="float64",
    )
    data = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        },
        index=idx,
    )
    result = BacktestResult(
        equity_curve=close.copy(),
        returns=close.pct_change().fillna(0.0),
        positions=pd.Series(0, index=idx, dtype="int8"),
        trades=[],
        stats={},
    )
    out = tmp_path / "chart.html"
    generate_local_tradingview_chart(data, result, out)

    html = out.read_text(encoding="utf-8")
    assert "payload.utBot" in html
    assert "payload.tradeMarkers" in html
    assert "payload.tradeEventLines" in html
    assert "createSeriesMarkers" in html
    assert "LineStyle.Dashed" in html
    assert "trailing_stop" in html


def test_local_chart_uses_unique_intraday_timestamps_for_candles(tmp_path) -> None:
    idx = pd.date_range("2024-01-01", periods=12, freq="2h", tz="UTC")
    close = pd.Series(range(len(idx)), index=idx, dtype="float64") + 100.0
    data = pd.DataFrame(
        {
            "open": close - 0.25,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
        },
        index=idx,
    )
    result = BacktestResult(
        equity_curve=close.copy(),
        returns=close.pct_change().fillna(0.0),
        positions=pd.Series(0, index=idx, dtype="int8"),
        trades=[],
        stats={},
    )
    out = tmp_path / "chart_intraday.html"
    generate_local_tradingview_chart(data, result, out)

    html = out.read_text(encoding="utf-8")
    match = re.search(r"const payload = (\{.*?\});\nconst statusEl", html, re.DOTALL)
    assert match is not None
    payload = json.loads(match.group(1))
    candle_times = [item["time"] for item in payload["candles"]]
    assert len(candle_times) == len(idx)
    assert len(set(candle_times)) == len(idx)


def test_local_chart_snaps_intraday_execution_to_daily_candles(tmp_path) -> None:
    chart_idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    intraday_idx = pd.date_range("2024-01-01", periods=5 * 24, freq="1h", tz="UTC")

    chart_close = pd.Series(range(len(chart_idx)), index=chart_idx, dtype="float64") + 100.0
    intraday_close = pd.Series(range(len(intraday_idx)), index=intraday_idx, dtype="float64") + 100.0

    data = pd.DataFrame(
        {
            "open": chart_close - 0.25,
            "high": chart_close + 0.5,
            "low": chart_close - 0.5,
            "close": chart_close,
        },
        index=chart_idx,
    )
    result = BacktestResult(
        equity_curve=intraday_close.copy(),
        returns=intraday_close.pct_change().fillna(0.0),
        positions=pd.Series(0, index=intraday_idx, dtype="int8"),
        trades=[
            Trade(
                side="long",
                entry_time=pd.Timestamp("2024-01-02T09:30:00Z"),
                exit_time=pd.Timestamp("2024-01-03T15:45:00Z"),
                entry_price=101.0,
                exit_price=103.0,
                units=1.0,
                pnl=2.0,
                return_pct=0.02,
                holding_bars=10,
            )
        ],
        stats={},
    )
    out = tmp_path / "chart_daily.html"
    generate_local_tradingview_chart(data, result, out)

    html = out.read_text(encoding="utf-8")
    match = re.search(r"const payload = (\{.*?\});\nconst statusEl", html, re.DOTALL)
    assert match is not None
    payload = json.loads(match.group(1))
    candle_times = [item["time"] for item in payload["candles"]]
    assert len(payload["equity"]) == len(payload["candles"])
    assert {item["time"] for item in payload["tradeMarkers"]}.issubset(set(candle_times))
