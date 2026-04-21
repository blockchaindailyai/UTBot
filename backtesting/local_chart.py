from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .engine import BacktestResult


def _to_points(series: pd.Series) -> list[dict[str, float | str]]:
    return [
        {"time": str(ts), "value": float(value)}
        for ts, value in series.items()
    ]


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
    }

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
    timeScale: {{ borderColor: '#334155' }},
    width: chartEl.clientWidth,
    height: 560,
  }});

  const candleSeries = chart.addCandlestickSeries();
  candleSeries.setData(payload.candles);

  const equitySeries = chart.addLineSeries({{ color: '#22c55e', lineWidth: 2 }});
  equitySeries.setData(payload.equity.map(point => ({{ time: point.time.slice(0, 10), value: point.value }})));

  statusEl.textContent = 'Candles + equity rendered with Lightweight Charts.';
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
