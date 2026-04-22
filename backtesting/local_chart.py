from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .engine import BacktestResult
from .strategy import compute_ut_bot_components


def _to_chart_time(ts: pd.Timestamp) -> int:
    timestamp = pd.Timestamp(ts)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp())


def _align_to_candle_time(ts: pd.Timestamp, candle_times: list[int]) -> int:
    if not candle_times:
        return _to_chart_time(ts)
    target = _to_chart_time(ts)
    for candle_time in reversed(candle_times):
        if candle_time <= target:
            return candle_time
    return candle_times[0]


def _dedupe_sorted_points(points: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    deduped: dict[int, float] = {}
    for point in points:
        ts = int(point["time"])
        value = float(point["value"])
        if not pd.notna(value):
            continue
        deduped[ts] = value
    return [{"time": ts, "value": deduped[ts]} for ts in sorted(deduped)]


def _to_points(series: pd.Series) -> list[dict[str, float | int]]:
    raw = [
        {"time": _to_chart_time(pd.Timestamp(ts)), "value": float(value)}
        for ts, value in series.items()
    ]
    return _dedupe_sorted_points(raw)


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

    markers: list[dict[str, str | int]] = []
    for ts in trailing_stop.index[buy_signal]:
        markers.append(
            {
                "time": _to_chart_time(pd.Timestamp(ts)),
                "position": "belowBar",
                "color": "#22c55e",
                "shape": "arrowUp",
                "text": "UT Buy",
            }
        )
    for ts in trailing_stop.index[sell_signal]:
        markers.append(
            {
                "time": _to_chart_time(pd.Timestamp(ts)),
                "position": "aboveBar",
                "color": "#ef4444",
                "shape": "arrowDown",
                "text": "UT Sell",
            }
        )
    markers.sort(key=lambda marker: marker["time"])

    return {
        "trailing_stop": _to_points(trailing_stop),
        "markers": markers,
    }


def _trade_markers_payload(result: BacktestResult, candle_times: list[int]) -> list[dict[str, str | int]]:
    markers: list[dict[str, str | int]] = []
    for trade in result.trades:
        side = str(trade.side).lower()
        is_long = side == "long"
        entry_time = _align_to_candle_time(pd.Timestamp(trade.entry_time), candle_times)
        exit_time = _align_to_candle_time(pd.Timestamp(trade.exit_time), candle_times)
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


def _trade_event_lines_payload(data: pd.DataFrame, result: BacktestResult) -> list[dict[str, object]]:
    if data.empty:
        return []

    candle_times = [_to_chart_time(pd.Timestamp(ts)) for ts in data.index]
    candle_index_by_time = {time: idx for idx, time in enumerate(candle_times)}
    lines: list[dict[str, object]] = []

    def _line_points(anchor_time: int, price: float) -> list[dict[str, float | int]]:
        anchor_index = candle_index_by_time.get(anchor_time)
        if anchor_index is None:
            return [{"time": anchor_time, "value": float(price)}]
        left_index = max(0, anchor_index - 1)
        right_index = min(len(candle_times) - 1, anchor_index + 1)
        return [
            {"time": candle_times[left_index], "value": float(price)},
            {"time": candle_times[right_index], "value": float(price)},
        ]

    for trade in result.trades:
        side = str(trade.side).lower()
        is_long = side == "long"
        entry_time = _align_to_candle_time(pd.Timestamp(trade.entry_time), candle_times)
        exit_time = _align_to_candle_time(pd.Timestamp(trade.exit_time), candle_times)
        lines.append(
            {
                "label": "LE" if is_long else "SE",
                "color": "#16a34a" if is_long else "#dc2626",
                "points": _line_points(entry_time, float(trade.entry_price)),
            }
        )
        lines.append(
            {
                "label": "LX" if is_long else "SX",
                "color": "#22c55e" if is_long else "#ef4444",
                "points": _line_points(exit_time, float(trade.exit_price)),
            }
        )
    return lines


def generate_local_tradingview_chart(
    data: pd.DataFrame,
    result: BacktestResult,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    candles = [
        {
            "time": _to_chart_time(pd.Timestamp(ts)),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        for ts, row in data[["open", "high", "low", "close"]].iterrows()
    ]
    candles_by_time: dict[int, dict[str, float | int]] = {int(item["time"]): item for item in candles}
    candles = [candles_by_time[ts] for ts in sorted(candles_by_time)]
    candle_times = [int(item["time"]) for item in candles]
    equity = result.equity_curve.astype("float64").reindex(data.index, method="ffill").bfill()
    positions = result.positions.astype("float64").reindex(data.index, method="ffill").fillna(0.0)
    payload = {
        "candles": candles,
        "price": _to_points(data["close"].astype("float64")),
        "equity": _to_points(equity),
        "position": _to_points(positions),
        "tradeMarkers": _trade_markers_payload(result, candle_times),
        "tradeEventLines": _trade_event_lines_payload(data, result),
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
  const tradeEventLineSeries = [];
  const renderTradeEventLines = (eventLines) => {{
    tradeEventLineSeries.forEach((series) => chart.removeSeries(series));
    tradeEventLineSeries.length = 0;
    (Array.isArray(eventLines) ? eventLines : []).forEach((eventLine) => {{
      const series = chart.addLineSeries({{
        color: eventLine.color || '#94a3b8',
        lineWidth: 3,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        lastValueVisible: false,
        priceLineVisible: false,
        crosshairMarkerVisible: false,
        title: eventLine.label || 'Execution',
      }});
      series.setData(Array.isArray(eventLine.points) ? eventLine.points : []);
      tradeEventLineSeries.push(series);
    }});
  }};

  const allMarkers = [...payload.utBot.markers, ...payload.tradeMarkers];
  if (typeof candleSeries.setMarkers === 'function') {{
    candleSeries.setMarkers(allMarkers);
  }} else if (typeof LightweightCharts.createSeriesMarkers === 'function') {{
    LightweightCharts.createSeriesMarkers(candleSeries, allMarkers);
  }}
  renderTradeEventLines(payload.tradeEventLines);

  const equitySeries = chart.addLineSeries({{
    color: '#22c55e',
    lineWidth: 2,
    priceScaleId: 'left',
    title: 'Equity',
  }});
  equitySeries.setData(payload.equity);
  const utStopSeries = chart.addLineSeries({{ color: '#f59e0b', lineWidth: 2 }});
  utStopSeries.setData(payload.utBot.trailing_stop);

  statusEl.textContent = 'Candles + equity + UT Bot signals + execution markers/lines rendered with Lightweight Charts.';
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
