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
- `chart.html`
- `report.pdf`
