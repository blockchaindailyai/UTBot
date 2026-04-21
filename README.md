# UTBot (Barebones Edition)

A minimal backtesting toolkit with:
- simple strategy interfaces
- lightweight execution engine
- basic HTML chart output
- plain-text report output

## Quick start (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python .\examples\run_backtest.py --csv .\examples\sample_ohlcv.csv --strategy ma_cross --fast 20 --slow 50 --out .\artifacts
```

## Outputs

- `stats.json`
- `trades.csv`
- `chart.html` (includes UT Bot trailing stop + Buy/Sell signal markers)
- `report.pdf`
- `ut_bot_strategy.pine` (paste into TradingView Pine Editor, then click **Add to chart**)
