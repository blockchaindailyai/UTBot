# WilliamsBot Backtesting Engine

A modular Python backtesting engine for testing multiple trading strategies against OHLCV data, producing comprehensive performance statistics, and generating TradingView Pine Script visualizations of historical trades.

## Features

- **Pluggable strategy interface** (`Strategy` base class)
- **Order types**: market, limit, stop, stop-limit with OHLC intrabar trigger checks
- **Execution cost models**: fees, slippage, spread, borrow, funding, overnight financing
- **Trade sizing modes**: percent of equity, fixed USD notional, fixed underlying units, hybrid min-USD + percent-of-equity, equity-milestone USD steps, volatility-scaled percent-of-equity, or stop-loss-scaled risk sizing
- **Configurable starting capital** for single and batch runs
- **Comprehensive analytics**: CAGR, Sharpe, Sortino, drawdown stats, trade stats, expectancy, exposure, and more
- **Human-readable PDF report**: summary page + charts for equity/PnL, drawdown, and trade-level outcomes
- **Data quality report per run**: duplicate timestamps, missing bars, outliers, timezone awareness
- **Resampling support**: convert source data to `2h`, `4h`, `12h`, `1d`, `1w`, etc.
- **Batch runs**: run many asset/timeframe combinations in one command and aggregate results
- **TradingView integration**: generate Pine Script markers for TradingView and local TradingView-style HTML charts
- **Built-in Bill Williams strategy**: Alligator (13/8, 8/5, 5/3 on median price) + AO histogram (MACD 5-34-5 on median price)
- **Williams visual overlays**: AO + AC histograms and Williams Zones candle coloring (green/red/gray) in local charts
- **Paper live-execution engine**: signal-driven paper broker for real-time market/limit/stop/stop-limit workflows, protective exits, reversals, and open-position monitoring

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Run a single backtest:

```powershell
python examples/run_backtest.py --csv examples/sample_ohlcv.csv --capital 10000 --order-type market --size-mode usd --size-value 2000
```

Start the real-time paper-trading engine with live-style flags:

```powershell
python .\examples\run_paper_trading.py --symbol BTC_USDT --interval Min60 --capital 10000 --order-type market --size-mode units --size-value 1 --max-position-size 5000 --max-leverage 2 --equity-cutoff 7000 --strategy wiseman --wiseman-1w-contracts 1 --wiseman-2w-contracts 0 --wiseman-3w-contracts 0 --gator-width-lookback 180 --gator-width-mult 0.5 --gator-width-valid-factor 1.0 --wiseman-profit-protection-teeth-exit --wiseman-profit-protection-min-bars 15 --wiseman-profit-protection-min-unrealized-return 0.01 --no-wiseman-profit-protection-require-gator-open --wiseman-cancel-reversal-on-first-exit --wiseman-gator-direction-mode 1 --wiseman-reversal-cooldown 1 --wiseman-profit-protection-lips-exit --wiseman-profit-protection-lips-volatility-trigger 0.04 --wiseman-profit-protection-lips-profit-trigger-mult 5 --wiseman-profit-protection-lips-volatility-lookback 30 --wiseman-profit-protection-lips-recent-trade-lookback 10 --wiseman-profit-protection-lips-min-unrealized-return 0.03 --wiseman-profit-protection-lips-arm-on-min-unrealized-return --1W-divergence-filter 240 --out .\artifacts_paper
```

The live paper runner keeps updating these paper-trading artifacts while it runs:
- `artifacts_paper\paper_dashboard.html` — a live-refreshing dashboard page for current position, working orders, trades, fills, equity, and drawdown
- `artifacts_paper\paper_dashboard_data.json` — the JSON payload for machine-readable monitoring
- `artifacts_paper\paper_dashboard_data.js` — the browser-friendly data script that lets `paper_dashboard.html` work when opened directly from disk
- `artifacts_paper\paper_status.md` — human-readable current account and position snapshot
- `artifacts_paper\paper_trades.md` — human-readable historical trades and fill ledger

By default, the paper runner now polls live candles every second, updates the dashboard files on that cadence, and keeps the dashboard chart/mark price synced to the latest fetched candle. Setup formation logic that depends on a bar close still waits for that close, but once a qualifying order or stop level exists it can now be triggered intrabar on the next live market snapshot without waiting for another close.

Open the live page in PowerShell with:

```powershell
Start-Process .\artifacts_paper\paper_dashboard.html
```

For rolling sweeps, `--end` is interpreted as the **last window start date** (the final month to begin a window), not as a truncation date for window end points.

This command produces:
- `artifacts/stats.json`
- `artifacts/data_quality.json`
- `artifacts/trades.csv`
- `artifacts/trade_markers.pine`
- `artifacts/first_wiseman_bearish.pine`
- `artifacts/first_wiseman_bullish.pine`
- `artifacts/tradingview_local_chart.html`
- `artifacts/report.pdf`

When `--csv` includes multiple files, each artifact uses an asset suffix (for example `report-BINANCE_ETHUSD1h.pdf`, `trades-BINANCE_ETHUSD1h.csv`) and the run also emits `multi_asset_summary.csv`, `consolidated_stats.json`, `consolidated_trades.csv`, and `consolidated_report.pdf`. By default, **each asset run starts with the full `--capital` value**, so BTC/ETH results stay identical whether you run them alone or alongside other assets. Multi-asset CLI runs are also executed serially for deterministic per-asset results. `consolidated_stats.json` reports the true total deployed capital in `initial_capital_total`, plus nominal-sum cashflow fields (`*_nominal_sum`) across all runs. Consolidation aligns runs on the union of timestamps, carrying each run's equity forward and filling pre-start periods with that run's initial capital to avoid artificial equity drop-offs when datasets have different start/end dates.

Run an indicator-only dry-run chart (no trade entry/exit markers):

```powershell
python examples/run_indicator_dry_run.py --csv examples/sample_ohlcv.csv
```

For rolling sweeps, `--end` is interpreted as the **last window start date** (the final month to begin a window), not as a truncation date for window end points.

This command produces:
- `artifacts_dry_run/first_wiseman_bearish.pine`
- `artifacts_dry_run/first_wiseman_bullish.pine`
- `artifacts_dry_run/tradingview_indicator_dry_run.html`

Download Binance OHLCV candles in this project's CSV format (hourly by default, supports more granular intervals like `1m`):

```powershell
python examples/scrape_binance_ohlcv.py --symbol BTCUSDT --interval 1h --start 2017-01-01T00:00:00+00:00 --out examples/BINANCE_BTCUSDT_1h.csv
```

Common variants:

```powershell
# More granular, minute candles
python examples/scrape_binance_ohlcv.py --symbol ETHUSDT --interval 1m --start 2017-01-01T00:00:00+00:00 --out examples/BINANCE_ETHUSDT_1m.csv

# Bound to a fixed date range
python examples/scrape_binance_ohlcv.py --symbol SOLUSDT --interval 15m --start 2020-01-01T00:00:00+00:00 --end 2021-01-01T00:00:00+00:00 --out examples/BINANCE_SOLUSDT_15m_2020.csv
```

Run a batch backtest across assets/timeframes:

```powershell
python examples/run_batch_backtest.py --assets BTC=examples/sample_ohlcv.csv,ETH=examples/sample_ohlcv.csv --timeframes 1h,2h,4h,12h,1d,1w --capital 10000 --order-type stop_limit
```

For rolling sweeps, `--end` is interpreted as the **last window start date** (the final month to begin a window), not as a truncation date for window end points.

This command produces:
- `artifacts_batch/batch_summary.csv`
- `artifacts_batch/aggregate_stats.json`
- `artifacts_batch/aggregate_equity.csv`
- `artifacts_batch/batch_tradingview_chart.html` (includes a dropdown to switch asset/timeframe view)


Run a Monte Carlo distribution analysis from one strategy run:

```powershell
python examples/run_monte_carlo.py --csv examples/sample_ohlcv.csv --strategy wiseman --capital 10000 --simulations 2000 --block-size 3 --threads 8 --wiseman-profit-protection-teeth-exit --no-wiseman-profit-protection-require-gator-open --start 2023-01-01 --end 2023-03-31
```

For rolling sweeps, `--end` is interpreted as the **last window start date** (the final month to begin a window), not as a truncation date for window end points.

This command produces:
- `artifacts_monte_carlo/monte_carlo_summary.json`
- `artifacts_monte_carlo/monte_carlo_equity_paths.csv`
- `artifacts_monte_carlo/monte_carlo_report.pdf`


### Parameter sweep for best Wiseman setups

Use this script to brute-force different 1W/2W/3W contract sizes, reversal behavior, profit-protection settings, gator filters, and timeframes, then rank by profitability:

```powershell
python examples/run_wiseman_parameter_sweep.py --csv examples/sample_ohlcv.csv --timeframes 1h,4h,1d --first-contracts 1,2 --second-contracts 2,3,5 --third-contracts 3,5,8 --reversal-contracts-mult 0.0,0.5,1.0 --profit-protection-enabled false,true --profit-protection-require-gator-open true,false --gator-width-valid-factor 0.75,1.0,1.5 --sort-by total_return --top 20 --start 2023-01-01 --end 2023-03-31
```

For rolling sweeps, `--end` is interpreted as the **last window start date** (the final month to begin a window), not as a truncation date for window end points.

This command produces:
- `artifacts_sweep/wiseman_sweep_results.csv` (all runs)
- `artifacts_sweep/wiseman_sweep_top.csv` (top-ranked runs)
- `artifacts_sweep/wiseman_sweep_summary.json` (best single setup)

Tip: 1m/5m data is supported. Use `--timeframes 1m` or `--timeframes 5m` to run at native cadence; sweep now defaults to `auto`, which uses the source OHLCV cadence directly.


### Rolling timeframe sweep (walk-forward windows)

Run sequential rolling windows (for example 6-month windows stepping forward 1 month), then generate an analytics-rich report with window-level outcomes, liquidation/cutoff counts, summary stats, cluster analysis, and charts:

```powershell
python examples/run_rolling_timeframe_sweep.py --csv examples/sample_ohlcv.csv --start 2018-01-01 --window-months 6 --jump-forward-months 1 --strategy wiseman --out artifacts_rolling_sweep
```

For rolling sweeps, `--end` is interpreted as the **last window start date** (the final month to begin a window), not as a truncation date for window end points.

This command produces:
- `artifacts_rolling_sweep/rolling_window_results.csv`
- `artifacts_rolling_sweep/rolling_window_summary.json`
- `artifacts_rolling_sweep/rolling_window_report.pdf`
- `artifacts_rolling_sweep/charts/*.svg`


## CLI option reference (PowerShell)

All three main workflows (`run_backtest.py`, `run_monte_carlo.py`, and `run_wiseman_parameter_sweep.py`) support `--start` and `--end` for inclusive date filtering.

This is the same information as before, but in a faster-to-read format.

### Single run: `examples/run_backtest.py`

```powershell
python .\examples\run_backtest.py --csv <path.csv> [options]
```

**Required**
- `--csv <path ...>`: one or more OHLCV CSV input files. Example: `--csv .\examples\sample_ohlcv.csv .\examples\BINANCE_XRPUSD60.csv`
- `--asset-size <KEY=MULTIPLIER>` (repeatable): per-asset capital multiplier in multi-CSV mode. Each run starts with `--capital * MULTIPLIER`. `KEY` can be the full path, filename, or filename stem. Unspecified assets default to `1.0`, meaning they keep the full baseline capital.
- Multi-CSV outputs are written as unique files in `--out`, e.g. `report-BINANCE_ETHUSD1h.pdf`, `trades-BINANCE_ETHUSD1h.csv`, plus consolidated files (`consolidated_report.pdf`, `consolidated_stats.json`, `consolidated_trades.csv`). Consolidated report equity starts from the sum of per-asset starting capitals; check `initial_capital_total` and `*_nominal_sum` fields in consolidated stats to see total nominal capital/cashflow across all assets.

**Core execution settings**
- `--capital <number>` (default `10000`): starting equity in quote currency.
- `--order-type <market|limit|stop|stop_limit>` (default `market`): fill model.
- `--fee <rate>` (default `0.0005`): fee rate on traded notional per fill.
  - `0.0005 = 0.05%`, `0.01 = 1%`, `0.05 = 5%`.
  - This is **not** USD and **not** units.
- `--slippage <rate>` (default `0.0002`): price-impact rate applied to fills.
  - `0.01` means **1% price slippage**, not `$0.01`, not `0.01` units.
- `--spread <rate>` (default `0.0`): bid/ask spread model as fractional rate.

**Position sizing**
- `--size-mode <percent_of_equity|usd|units|hybrid_min_usd_percent|equity_milestone_usd|volatility_scaled|stop_loss_scaled>` (default `percent_of_equity`)
- `--size-value <number>` (default `1.0`), interpreted by `--size-mode`:
  - `percent_of_equity`: fraction of equity (`1.0`=100%, `0.5`=50%)
  - `usd`: fixed notional in quote currency (`1000` = ~$1000 notional)
  - `units`: fixed asset units (`1` = 1 unit)
  - `hybrid_min_usd_percent`: uses the greater of `equity * size-value` and `--size-min-usd`
  - `equity_milestone_usd`: keeps the USD notional from `--size-value` until equity reaches a milestone, then steps up to that milestone's USD size until the next milestone is reached
  - `volatility_scaled`: starts from `equity * size-value`, then scales notional by `target_vol / realized_vol`
  - `stop_loss_scaled`: sizes units so `abs(entry - stop_loss) * units` equals `equity * size-value`; requires strategy-provided stop-loss prices for entry/add signals
- `--size-min-usd <number>` (default `0.0`): minimum USD notional floor used by `hybrid_min_usd_percent`.
- `--size-equity-milestones <equity:usd,...>` (default empty): comma-separated step schedule for `equity_milestone_usd`, for example `15000:1500,20000:2000`.
- `--volatility-target-annual <float>` (default `0.15`): annualized volatility target for `volatility_scaled`.
- `--volatility-lookback <int>` (default `20`): max lookback bars for realized volatility in `volatility_scaled` (uses all available prior bars when history is shorter; never uses forward bars).
- `--volatility-min-scale <float>` (default `0.25`): lower clamp on volatility size multiplier.
- `--volatility-max-scale <float>` (default `3.0`): upper clamp on volatility size multiplier.
- `volatility_scaled` sizing formula:
  - `base_notional = equity * size_value`
  - `scale = clip(volatility_target_annual / realized_annual_volatility, volatility_min_scale, volatility_max_scale)`
  - `notional = base_notional * scale`
  - `units = notional / fill_price`
  - Practical implication: when trading low-priced assets (for example, DOGE near `0.002`), even modest USD notional maps to very large unit counts (millions of units), and units can change materially from one signal to the next as realized volatility changes bar by bar.

**Financing / carry**
- `--borrow-annual <rate>` (default `0.0`): annual short borrow rate.
- `--funding-per-period <rate>` (default `0.0`): funding rate applied each bar.
- `--overnight-annual <rate>` (default `0.0`): annual overnight financing rate.
- `--max-loss <number>` (default disabled): max per-position absolute loss in quote currency before an automatic stop-out.
- `--max-loss-pct-of-equity <float>` (default disabled): max per-position loss as a fraction of entry equity (`0.01` = 1%). If both max-loss settings are set, the tighter threshold is used.

**Strategy / output**
- `--out <folder>` (default `artifacts`): output folder.
- `--start <yyyy-mm-dd|datetime>` (default unset): inclusive start of data window (example: `--start 2024-01-01`).
- `--end <yyyy-mm-dd|datetime>` (default unset): inclusive end of data window (example: `--end 2025-12-12`).
- `--strategy <alligator_ao|wiseman>` (default `alligator_ao`): strategy selector for the single-run script.
- `--gator-width-lookback <int>` (default `50`): Wiseman gator-closed rolling median lookback.
- `--gator-width-mult <float>` (default `1.0`): Wiseman gator-closed strictness multiplier.
- `--gator-width-valid-factor <float>` (default `1.0`): scales the 1W setup filter `|teeth-lips| < |lips-midpoint| * factor`; values `>1` loosen, values `<1` tighten.
- `--wiseman-gator-direction-mode <1|2|3>` (default `1`): 1=classic `lips>teeth>jaw` / `lips<teeth<jaw`; 2=relaxed `lips>teeth OR jaw` / `lips<teeth OR jaw`; 3=price-vs-teeth using setup-bar midpoint (`price>teeth` / `price<teeth`).
- `--wiseman-1w-contracts <int>` (default `1`): contracts for 1W entries (`0` disables 1W entries).
- `--wiseman-2w-contracts <int>` (default `3`): additional contracts added when 2W fills (`0` disables 2W adds).
- `--wiseman-3w-contracts <int>` (default `5`): additional contracts added when 3W fills (`0` disables 3W adds).
- `--wiseman-reversal-contracts-mult <float>` (default `1.0`): multiplier applied to current contracts when a 1W reversal triggers (`0` disables reversals, `0.5` halves, `2.0` doubles). Reversal contracts preserve fractional values (for example, `0.5` on a 1-contract source yields `0.5`), and this contract multiplier is applied before engine size-mode notional sizing.
- `--1W-wait-bars-to-close <int>` (default `0`): delay 1W entry activation until this many bars after setup-bar close; when delay elapses, place a resting limit at setup high/low after that bar closes. The order becomes fill-eligible on subsequent bars and remains working until filled or an opposite 1W setup forms.
- `--1W-divergence-filter <int>` (default `0`, disabled): require a qualifying AO/price divergence within the last N bars before arming a 1W setup, measured with midpoint price `(H+L)/2`. Bullish 1W requires AO lower-low → higher-low and midpoint price higher-low → lower-low; bearish 1W requires AO higher-high → higher-low and midpoint price lower-high → higher-high.
- `--wiseman-1w-opposite-close-min-unrealized-return <float>` (default `0.0`): minimum favorable unrealized return required before an opposite 1W level is allowed to close/reverse an already-open trade. Example: set `0.05` to require +5% unrealized excursion; if not met, opposite 1W close is deferred and profit-protection/other exits must close the trade.
- `--wiseman-cancel-reversal-on-first-exit` / `--no-wiseman-cancel-reversal-on-first-exit` (default disabled): when enabled, a resting 1W-R level is canceled immediately if the original 1W trade exits (profit protection, stop, or opposite signal).
- `--wiseman-profit-protection-teeth-exit` / `--no-wiseman-profit-protection-teeth-exit` (default disabled): toggle teeth-based profit-protection exits.
- `--wiseman-profit-protection-min-bars <int>` (default `3`): minimum bars in position before teeth profit protection can arm.
- `--wiseman-profit-protection-min-unrealized-return <float>` (default `0.01`): minimum unrealized return required before teeth profit protection can arm.
- `--wiseman-profit-protection-credit-unrealized-before-min-bars` / `--no-wiseman-profit-protection-credit-unrealized-before-min-bars` (default disabled): if enabled, a trade can permanently credit the min-unrealized gate as soon as intrabar favorable excursion reaches threshold, even before min-bars is reached.
- `--wiseman-profit-protection-require-gator-open` / `--no-wiseman-profit-protection-require-gator-open` (default enabled): require gator-open regime before teeth profit protection can arm.
- `--wiseman-profit-protection-lips-exit` / `--no-wiseman-profit-protection-lips-exit` (default disabled): enable aggressive mode that uses the Alligator lips as the protection level when aggressive trigger conditions are met.
- `--wiseman-profit-protection-lips-volatility-trigger <float>` (default `0.02`): rolling realized-volatility threshold (close-return stdev) that enables aggressive lips exits when reached/exceeded.
- `--wiseman-profit-protection-lips-profit-trigger-mult <float>` (default `2.0`): deep-profit trigger multiplier; aggressive lips mode activates when unrealized return reaches this multiple of the chosen baseline.
- `--wiseman-profit-protection-lips-volatility-lookback <int>` (default `20`): lookback window used for rolling realized-volatility estimation.
- `--wiseman-profit-protection-lips-recent-trade-lookback <int>` (default `5`): number of recently closed trades used when building the deep-profit baseline.
- `--wiseman-profit-protection-lips-min-unrealized-return <float>` (default `0.01`): separate minimum unrealized-return threshold used by lips-mode when min-unrealized arming is enabled.
- `--wiseman-profit-protection-lips-arm-on-min-unrealized-return` / `--no-wiseman-profit-protection-lips-arm-on-min-unrealized-return` (default disabled): when enabled, lips-mode uses `--wiseman-profit-protection-lips-min-unrealized-return` as its profit trigger instead of the lips profit multiple baseline comparison.
- `--wiseman-profit-protection-zones-exit` / `--no-wiseman-profit-protection-zones-exit` (default disabled): enable Williams Zones trailing profit protection that arms after 5 consecutive AO+AC zone bars in the trade direction.
- `--wiseman-profit-protection-zones-min-unrealized-return <float>` (default `0.01`): minimum unrealized return required before the Williams Zones trailing stop can arm.

The default example strategy is `AlligatorAOStrategy`:
- Alligator lines use SMMA of `(H+L)/2` with Bill Williams defaults: Jaw `13` shifted `8`, Teeth `8` shifted `5`, Lips `5` shifted `3`.
- AO histogram is implemented as MACD histogram on median price with settings `5-34-5`.
- AC histogram is also plotted using `5-34-5` on median price, on a separate lower scale beneath AO.
- Williams Zones coloring is applied to candles: green when AO+AC are green, red when AO+AC are red, gray when they differ.
- The local TradingView-style charts plot candles, trade markers, Alligator lines, AO histogram, and AC histogram for visual backtest validation.

**Examples**

```powershell
# USD notional sizing
python examples/run_backtest.py --csv BINANCE_BTC1H.csv --size-mode usd --size-value 1000 --slippage 0.01 --fee 0.0005

# Units sizing
python examples/run_backtest.py --csv examples/sample_ohlcv.csv --size-mode units --size-value 1

# 50% equity per trade
python examples/run_backtest.py --csv examples/sample_ohlcv.csv --size-mode percent_of_equity --size-value 0.5

# Hybrid sizing: at least $500, otherwise 5% equity
python examples/run_backtest.py --csv examples/sample_ohlcv.csv --size-mode hybrid_min_usd_percent --size-value 0.05 --size-min-usd 500

# Step up USD notional after equity milestones
python examples/run_backtest.py --csv examples/sample_ohlcv.csv --size-mode equity_milestone_usd --size-value 1000 --size-equity-milestones 15000:1500,20000:2000

# Volatility-scaled sizing targeting 12% annualized vol
python examples/run_backtest.py --csv examples/sample_ohlcv.csv --size-mode volatility_scaled --size-value 0.1 --volatility-target-annual 0.12 --volatility-lookback 30

# Wiseman strategy with 10% equity sizing
python examples/run_backtest.py --csv examples/sample_ohlcv.csv --strategy wiseman --order-type stop --size-mode percent_of_equity --size-value 0.1

# Wiseman strategy with looser gator-width-valid setup filter
python examples/run_backtest.py --csv examples/sample_ohlcv.csv --strategy wiseman --gator-width-valid-factor 2.0

# Wiseman strategy with custom contract sizing and half-size reversals
python examples/run_backtest.py --csv examples/sample_ohlcv.csv --strategy wiseman --wiseman-1w-contracts 2 --wiseman-2w-contracts 4 --wiseman-3w-contracts 8 --wiseman-reversal-contracts-mult 0.5

# Wiseman with teeth-based profit protection enabled
python examples/run_backtest.py --csv examples/sample_ohlcv.csv --strategy wiseman --wiseman-profit-protection-teeth-exit --wiseman-profit-protection-min-bars 4 --wiseman-profit-protection-min-unrealized-return 0.015

# Wiseman with aggressive lips-based profit protection (volatility + deep-profit triggers)
python examples/run_backtest.py --csv examples/sample_ohlcv.csv --strategy wiseman --wiseman-profit-protection-teeth-exit --wiseman-profit-protection-lips-exit --wiseman-profit-protection-min-bars 4 --wiseman-profit-protection-min-unrealized-return 0.015 --wiseman-profit-protection-lips-volatility-trigger 0.025 --wiseman-profit-protection-lips-profit-trigger-mult 2.5 --wiseman-profit-protection-lips-volatility-lookback 30 --wiseman-profit-protection-lips-recent-trade-lookback 8 --wiseman-profit-protection-lips-min-unrealized-return 0.012 --wiseman-profit-protection-lips-arm-on-min-unrealized-return

# Wiseman: require at least +5% unrealized excursion before opposite 1W can close/reverse an open trade
python examples/run_backtest.py --csv examples/sample_ohlcv.csv --strategy wiseman --wiseman-1w-opposite-close-min-unrealized-return 0.05 --wiseman-profit-protection-teeth-exit --wiseman-profit-protection-min-bars 4 --wiseman-profit-protection-min-unrealized-return 0.01

# Wiseman with Williams Zones trailing profit protection enabled
python examples/run_backtest.py --csv examples/sample_ohlcv.csv --strategy wiseman --wiseman-profit-protection-zones-exit --wiseman-profit-protection-zones-min-unrealized-return 0.02

# Limit the backtest to a date range
python examples/run_backtest.py --csv examples/sample_ohlcv.csv --start 2024-01-01 --end 2024-01-31

# Multi-asset normal backtest with per-asset capital allocation
python .\examples\run_backtest.py --csv .\examples\BINANCE_XRPUSD60.csv .\examples\BINANCE_SOLUSD60.csv --size-mode percent_of_equity --size-value 0.25 --asset-size BINANCE_XRPUSD60=0.10 --asset-size BINANCE_SOLUSD60.csv=0.40 --out .\artifacts_multi

```

### Indicator-only dry run: `examples/run_indicator_dry_run.py`

```powershell
python examples/run_indicator_dry_run.py --csv <path.csv> [options]
```

**Required**
- `--csv <path>`: OHLCV CSV input file.

**Optional**
- `--out <folder>` (default `artifacts_dry_run`): output folder.

This mode generates a clean local chart with indicator overlays and Wiseman markers while intentionally omitting trade entry/exit markers.

### Batch run: `examples/run_batch_backtest.py`

```powershell
python examples/run_batch_backtest.py --assets <ASSET=path.csv,...> [options]
```

**Required**
- `--assets <list>`: comma-separated `ASSET=path.csv` entries.
  - Example: `--assets BTC=BINANCE_BTC1H.csv,ETH=examples/sample_ohlcv.csv`

**Batch-specific**
- `--timeframes <list>` (default `auto`): comma-separated intervals. With `auto`, each asset runs on its detected source OHLCV cadence.
  - Example: `--timeframes 1h,1d,1w`

**All remaining options**
- Same semantics as single-run for: `--capital`, `--fee`, `--slippage`, `--spread`, `--order-type`, `--size-mode`, `--size-value`, `--borrow-annual`, `--funding-per-period`, `--overnight-annual`, `--max-loss`, `--max-loss-pct-of-equity`, `--out`.
- Batch default output folder is `artifacts_batch`.

**Examples**

```powershell
# Mixed assets/timeframes with USD sizing
python examples/run_batch_backtest.py --assets BTC=BINANCE_BTC1H.csv,ETH=sample_ohlcv.csv --timeframes 1h,1d,1w --size-mode usd --size-value 1000 --slippage 0.01 --fee 0.0005

# Units sizing across two assets
python examples/run_batch_backtest.py --assets BTC=examples/sample_ohlcv.csv,ETH=examples/sample_ohlcv.csv --timeframes 1h,4h --size-mode units --size-value 1
```


## Wiseman execution strategy (state-machine spec)

The logic below is internally consistent and can be implemented as a deterministic bar-by-bar state machine.

### Core definitions

- **Bearish wiseman bar**: a bearish 1st Wiseman setup candle (your detector output).
- **Bullish wiseman bar**: a bullish 1st Wiseman setup candle (your detector output).
- **Signal confirmation**:
  - Bearish wiseman confirms when price breaks the wiseman **low** before its **high** is broken.
  - Bullish wiseman confirms when price breaks the wiseman **high** before its **low** is broken.
- **Risk allocation**: each new entry uses **10% of current account equity**.

### Bearish wiseman trade flow

1. When a bearish wiseman bar closes, place a **stop-sell** at the wiseman low sized to 10% of equity.
2. If the wiseman high breaks before the wiseman low, **cancel** that pending short setup.
3. If stop-sell triggers, open short and immediately place stop-loss at the wiseman high.
4. If the wiseman high is broken:
   - before 3 full bars after entry: treat as stop-out (close short).
   - after more than 2 subsequent bars and reversal is confirmed: close short and open long at the wiseman high with same notional sizing basis.
5. For a long reversal opened from this bearish setup, set reversal stop-loss to the **lowest low between setup bar and reversal trigger bar** (inclusive).

### Bullish wiseman trade flow

1. When a bullish wiseman bar closes, place a **stop-buy** at the wiseman high sized to 10% of equity.
2. If the wiseman low breaks before the wiseman high, **cancel** that pending long setup.
3. If stop-buy triggers, open long and immediately place stop-loss at the wiseman low.
4. If the wiseman low is broken:
   - before 3 full bars after entry: treat as stop-out (close long).
   - after more than 2 subsequent bars and reversal is confirmed: close long and open short at the wiseman low with same notional sizing basis.
5. For a short reversal opened from this bullish setup, set reversal stop-loss to the **highest high between setup bar and reversal trigger bar** (inclusive).

### Opposite confirmed signal handling

- If currently **short** and a **bullish wiseman** confirms, close short and open long.
- If currently **long** and a **bearish wiseman** confirms, close long and open short.
- Never hold long and short simultaneously.
- Direction always follows the most recent **confirmed** wiseman signal.

### Optional gate for opposite 1W closes

Use `--wiseman-1w-opposite-close-min-unrealized-return` to require a minimum favorable unrealized excursion before an opposite 1W level is allowed to close/reverse an already-open trade.

- `0.0` (default): keep current behavior (opposite 1W closes/reverses immediately when level is crossed).
- `0.05`: requires +5% favorable excursion first (long uses intrabar high vs entry, short uses intrabar low vs entry).
- If the threshold is not met, the trade remains open and must be closed by other logic (for example profit protection).

### Optional teeth-based profit protection

When `--wiseman-profit-protection-teeth-exit` is enabled, an additional protective exit can occur **without overriding stop-loss or opposite-signal logic**:

1. A live position must first show meaningful progress in the trade direction:
   - held at least `--wiseman-profit-protection-min-bars`,
   - unrealized return reaches at least `--wiseman-profit-protection-min-unrealized-return` based on intrabar favorable excursion (high for longs, low for shorts),
   - min-unrealized credit can be restricted to bars at/after min-bars (default), or allowed at any time with `--wiseman-profit-protection-credit-unrealized-before-min-bars`,
   - and gator is open (not in the `gator_closed` regime) when `--wiseman-profit-protection-require-gator-open` is enabled.
2. After those conditions are met, protection is armed for that trade.
3. Armed exit triggers on a close across the red teeth:
   - long exits when a bar closes **below** teeth,
   - short exits when a bar closes **above** teeth.

This is intentionally a late-stage profit-protection rule and is not intended for immediate post-entry exits.

### Optional aggressive lips-based profit protection

When `--wiseman-profit-protection-lips-exit` is enabled, the strategy keeps the same arming behavior as teeth protection, but can switch the active protection line from **teeth** to **lips** whenever either aggressive trigger is true:

1. **Higher volatility trigger**: rolling close-return volatility (using `--wiseman-profit-protection-lips-volatility-lookback`) is greater than or equal to `--wiseman-profit-protection-lips-volatility-trigger`. This volatility is computed as the rolling standard deviation of close-to-close percentage returns, so lower thresholds make lips mode activate more often and higher thresholds make it rarer.
2. **Profit trigger** (configurable):
   - default behavior uses a deep-profit multiple: unrealized return must be at least `--wiseman-profit-protection-lips-profit-trigger-mult` times a baseline derived from recent volatility and recently closed trade returns (from `--wiseman-profit-protection-lips-recent-trade-lookback`),
   - optional behavior (`--wiseman-profit-protection-lips-arm-on-min-unrealized-return`) uses `--wiseman-profit-protection-lips-min-unrealized-return` directly instead of the multiple baseline comparison.

If neither aggressive trigger is active, protection still uses teeth; if either trigger is active, protection uses lips for earlier/more-defensive exits.

### Optional Williams Zones trailing profit protection

When `--wiseman-profit-protection-zones-exit` is enabled, the strategy adds a separate trailing stop that is based on Williams Zones bar coloring (`AO` green/red together with `AC` green/red on the same bar):

1. Long trades arm after unrealized return reaches `--wiseman-profit-protection-zones-min-unrealized-return` and the trade records at least 5 consecutive green zone bars. The initial stop is placed at the low of the qualifying bar.
2. Short trades arm after unrealized return reaches `--wiseman-profit-protection-zones-min-unrealized-return` and the trade records at least 5 consecutive red zone bars. The initial stop is placed at the high of the qualifying bar.
3. After arming, each subsequent bar that closes without breaking the active stop advances the stop to that bar's low (longs) or high (shorts).
4. If price breaks the active stop on a later bar, the strategy exits flat immediately and clears the trailing state for the next trade.

### Intrabar ambiguity rule (low tick specificity)

When one OHLC bar contains both:
1. a valid wiseman **reversal event** for the active trade, and
2. a **new wiseman setup** in the opposite direction,

then process in this order:
- First honor the reversal of the active signal and trade that direction.
- Keep trading that direction until the active wiseman level (high for bearish-source trades, low for bullish-source trades) is broken.
- Only then allow direction to switch to the newly formed wiseman signal.

### Practical implementation notes

- Track explicit states: `flat`, `pending_long`, `pending_short`, `long`, `short`.
- Store per-signal reference levels: `wiseman_high`, `wiseman_low`, `signal_bar_index`.
- Enforce deterministic tie-breakers for same-bar high+low touches because only OHLC is available.
- Use a single-position model and atomic flips (`close` then `open`) on confirmed opposite signals.

### Why a green `1W` marker can appear without a trade

If you see a green `1W` marker on the chart but no matching trade entry, that's expected in some states:

- The marker is drawn by indicator-style detection logic and is anchored to the **setup candle** (`candidate_ts` / `candidateIndex`), not the eventual execution candle.
- The strategy enters only when the breakout trigger actually occurs (break of candidate high for bullish / candidate low for bearish) and when state-machine guards permit it.
- Pending Wiseman setups are intentionally blocked while the gator is considered closed, and only become executable once it re-opens.
- A setup can be canceled before execution if the invalidation side is touched first (for bullish: low first; for bearish: high first).
- If both sides are touched inside one OHLC bar, deterministic tie-break rules in the state machine can flatten or suppress the setup instead of opening a position.

So, a `1W` label should be interpreted as "qualified setup observed" on that prior candle; execution still depends on subsequent bar-by-bar trigger and state conditions.

When you run `--strategy wiseman`, the local chart now flags non-triggered first Wiseman setups with amber labels:
- `1W-I`: setup invalidated before trigger (opposite side broke first).
- `1W-C`: same-bar stop conflict before reversal window opened.
- `1W-N`: no breakout trigger before end of available data.
- `1W-G`: setup canceled because gator was closed.
- `1W-S`: earlier active setup superseded by a newer, more extreme same-side setup.
- `1W-W`: newer same-side setup was weaker than the currently active setup and ignored.
- `1W-D`: setup filtered out because no qualifying AO/price divergence was found within the configured divergence-filter lookback.

## Data format

Input CSV must contain:
- `timestamp` (ISO datetime) or `time`
- `open`
- `high`
- `low`
- `close`
- `volume` or `Volume`

## Extending with your own strategy

Create a class inheriting `Strategy` and implement `generate_signals(dataframe)` to return a pandas `Series` with values:
- `1` = long
- `0` = flat
- `-1` = short

Then pass it into `BacktestEngine.run(...)` or use `run_batch_backtest(...)` for multi-asset/timeframe runs.
