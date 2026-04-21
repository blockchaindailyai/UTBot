from __future__ import annotations

import pandas as pd

from backtesting.engine import BacktestResult
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
