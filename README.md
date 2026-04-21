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

UT Bot strategy example with position sizing controls:

```powershell
python .\examples\run_backtest.py --csv .\examples\sample_ohlcv.csv --strategy ut_bot --ut-key-value 1 --ut-atr-period 10 --size-mode equity_percent --size-value 0.5 --out .\artifacts_ut
```

Intrabar-style higher-timeframe signals (for example, build progressive 1D bars from 5m/1h source data while still executing on source bars):

```powershell
python .\examples\run_backtest.py --csv .\examples\sample_ohlcv.csv --strategy ut_bot --signal-timeframe 1D --out .\artifacts_intrabar
```

For dense source bars (like 5m), cap intrabar evaluations per 1D bar to speed up runs:

```powershell
python .\examples\run_backtest.py --csv .\examples\sample_ohlcv.csv --strategy ut_bot --signal-timeframe 1D --max-intrabar-evaluations-per-signal-bar 24 --out .\artifacts_intrabar_fast
```

You can also cap higher-timeframe history used in each intrabar re-evaluation:

```powershell
python .\examples\run_backtest.py --csv .\examples\sample_ohlcv.csv --strategy ut_bot --signal-timeframe 1D --max-intrabar-evaluations-per-signal-bar 24 --signal-timeframe-history-bars 20 --out .\artifacts_intrabar_faster
```

Available sizing modes:
- `static_usd`: fixed USD notional per trade (`--size-value` is USD amount).
- `equity_percent`: scales trade notional with account equity (`--size-value` is a decimal fraction, e.g. `0.5` = 50%).
- `volatility_scaled`: starts from equity % sizing and rescales using realized annualized volatility toward `--volatility-target-annual`.

## Outputs

- `stats.json`
- `trades.csv`
- `chart.html` (includes UT Bot trailing stop + Buy/Sell signal markers)
- `report.pdf`
- `ut_bot_strategy.pine` (paste into TradingView Pine Editor, then click **Add to chart**)
