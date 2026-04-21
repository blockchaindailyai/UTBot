from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .engine import BacktestResult
from .strategy import compute_ut_bot_components


def _to_points(series: pd.Series) -> list[dict[str, float | str]]:
    return [
        {"time": str(ts), "value": float(value)}
        for ts, value in series.items()
    ]


def _ut_bot_payload(
    data: pd.DataFrame,
    key_value: float = 1.0,
    atr_period: int = 10,
) -> dict[str, list[dict[str, float | str]]]:
    trailing_stop, buy_signal, sell_signal, _ = compute_ut_bot_components(
        data=data,
        key_value=key_value,
        atr_period=atr_period,
    )
    if len(trailing_stop) == 0:
        return {"trailing_stop": [], "markers": []}

    markers: list[dict[str, str]] = []
    for ts in trailing_stop.index[buy_signal]:
        markers.append({"time": str(ts.date()), "position": "belowBar", "color": "#22c55e", "shape": "arrowUp", "text": "UT Buy"})
    for ts in trailing_stop.index[sell_signal]:
        markers.append({"time": str(ts.date()), "position": "aboveBar", "color": "#ef4444", "shape": "arrowDown", "text": "UT Sell"})
    markers.sort(key=lambda marker: marker["time"])

    return {
        "trailing_stop": _to_points(trailing_stop),
        "markers": markers,
    }


def _trade_markers_payload(result: BacktestResult) -> list[dict[str, str]]:
    markers: list[dict[str, str]] = []
    for trade in result.trades:
        side = str(trade.side).lower()
        is_long = side == "long"
        entry_time = str(pd.Timestamp(trade.entry_time).date())
        exit_time = str(pd.Timestamp(trade.exit_time).date())
        markers.append(
            {
                "time": entry_time,
                "position": "belowBar" if is_long else "aboveBar",
                "color": "#16a34a" if is_long else "#dc2626",
                "shape": "arrowUp" if is_long else "arrowDown",
                "text": "LE" if is_long else "SE",
            }
        )
        markers.append(
            {
                "time": exit_time,
                "position": "aboveBar" if is_long else "belowBar",
                "color": "#22c55e" if is_long else "#ef4444",
                "shape": "arrowDown" if is_long else "arrowUp",
                "text": "LX" if is_long else "SX",
            }
        )
    markers.sort(key=lambda marker: marker["time"])
    return markers


def generate_local_tradingview_chart(
    data: pd.DataFrame,
    result: BacktestResult,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    candles = [
        {
            "time": str(ts.date()),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        for ts, row in data[["open", "high", "low", "close"]].iterrows()
    ]
    payload = {
        "candles": candles,
        "price": _to_points(data["close"].astype("float64")),
        "equity": _to_points(result.equity_curve.astype("float64")),
        "position": _to_points(result.positions.astype("float64")),
        "tradeMarkers": _trade_markers_payload(result),
    }
    payload["utBot"] = _ut_bot_payload(data)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>Barebones Backtest Chart</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; }}
    h2 {{ margin: 16px; }}
    #chart {{ height: 560px; margin: 0 16px 16px 16px; }}
    #status {{ margin: 0 16px 16px 16px; color: #94a3b8; }}
    #fallback {{ margin: 0 16px 16px 16px; display: none; white-space: pre-wrap; font-size: 12px; }}
  </style>
</head>
<body>
<h2>Barebones Backtest Chart</h2>
<div id='chart'></div>
<div id='status'></div>
<pre id='fallback'></pre>
<script src='https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js'></script>
<script>
const payload = {json.dumps(payload)};
const statusEl = document.getElementById('status');
const fallbackEl = document.getElementById('fallback');
const chartEl = document.getElementById('chart');

function renderFallback(message) {{
  statusEl.textContent = message;
  fallbackEl.style.display = 'block';
  fallbackEl.textContent = JSON.stringify(payload, null, 2);
}}

if (!window.LightweightCharts) {{
  renderFallback('Could not load Lightweight Charts (CDN unavailable). Showing raw data fallback below.');
}} else {{
  const chart = LightweightCharts.createChart(chartEl, {{
    layout: {{ background: {{ color: '#0f172a' }}, textColor: '#e2e8f0' }},
    grid: {{ vertLines: {{ color: '#1e293b' }}, horzLines: {{ color: '#1e293b' }} }},
    rightPriceScale: {{ borderColor: '#334155' }},
    leftPriceScale: {{ borderColor: '#334155', visible: true }},
    timeScale: {{ borderColor: '#334155' }},
    width: chartEl.clientWidth,
    height: 560,
  }});

  const candleSeries = chart.addCandlestickSeries();
  candleSeries.setData(payload.candles);
  const allMarkers = [...payload.utBot.markers, ...payload.tradeMarkers];
  if (typeof candleSeries.setMarkers === 'function') {{
    candleSeries.setMarkers(allMarkers);
  }} else if (typeof LightweightCharts.createSeriesMarkers === 'function') {{
    LightweightCharts.createSeriesMarkers(candleSeries, allMarkers);
  }}

  const equitySeries = chart.addLineSeries({{
    color: '#22c55e',
    lineWidth: 2,
    priceScaleId: 'left',
    title: 'Equity',
  }});
  equitySeries.setData(payload.equity.map(point => ({{ time: point.time.slice(0, 10), value: point.value }})));
  const utStopSeries = chart.addLineSeries({{ color: '#f59e0b', lineWidth: 2 }});
  utStopSeries.setData(payload.utBot.trailing_stop.map(point => ({{ time: point.time.slice(0, 10), value: point.value }})));

  statusEl.textContent = 'Candles + equity + UT Bot signals + trade execution markers rendered with Lightweight Charts.';
  window.addEventListener('resize', () => {{
    chart.applyOptions({{ width: chartEl.clientWidth }});
  }});
}}
</script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return path


def generate_batch_local_tradingview_chart(*args, **kwargs) -> Path:
    raise NotImplementedError("Batch charting was intentionally removed in the barebones rewrite")
