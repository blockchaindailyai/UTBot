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
    payload = {
        "price": _to_points(data["close"].astype("float64")),
        "equity": _to_points(result.equity_curve.astype("float64")),
        "position": _to_points(result.positions.astype("float64")),
    }

    html = f"""<!doctype html>
<html>
<head><meta charset='utf-8'><title>Barebones Backtest Chart</title></head>
<body>
<h2>Barebones Backtest Chart</h2>
<pre id='data'></pre>
<script>
const payload = {json.dumps(payload)};
document.getElementById('data').textContent = JSON.stringify(payload, null, 2);
</script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return path


def generate_batch_local_tradingview_chart(*args, **kwargs) -> Path:
    raise NotImplementedError("Batch charting was intentionally removed in the barebones rewrite")
