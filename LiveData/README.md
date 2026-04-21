# LiveData KCEX chart

You now have **two ways** to run the chart.

## 1) Direct mode (no local API server)

If KCEX allows cross-origin browser requests from your machine/network, you can open the direct chart and hit KCEX from the browser directly.

```powershell
Start-Process .\LiveData\index.direct.html
```

If that page shows CORS/network errors, use API mode below.

## 2) API mode (recommended fallback)

This runs a tiny local Python server that proxies/normalizes KCEX responses for the chart.

```powershell
python .\LiveData\live_kcex_chart.py --port 8765
```

Then open:

```powershell
Start-Process http://127.0.0.1:8765
```

## Is a local API server necessary?

**Not always.**

- If KCEX endpoints are reachable directly from your browser and CORS is allowed, direct mode works and no local API is required.
- If CORS/network restrictions block direct browser calls, the local API mode is the practical way to keep a live chart working reliably.

## What this app infers automatically

- Top 10 USDT assets inferred from KCEX metadata endpoints (falls back to a sensible default list if unavailable).
- Typical timeframes inferred by probing common intervals (falls back to defaults if probing fails).
- API mode now refreshes only the currently selected symbol/timeframe, with the chart polling snapshot updates every 1 second while strategy evaluation runs separately in the background.


## If the page loads but no candles appear

Use this quick API check while the server is running:

```powershell
(Invoke-RestMethod "http://127.0.0.1:8765/api/klines?symbol=BTC_USDT&interval=Min1&limit=5").candles
```

- If this returns candles, the frontend should now render them (timestamp normalization and resize handling are built in).
- If this returns an empty array, KCEX returned no candle payload for that request window; switch symbol/timeframe and retry.


## Indicators now available in server mode chart

`index.html` now supports Bill Williams/Wiseman-style overlays and pane layouts:

- **Alligator lines** (Jaw/Teeth/Lips) using SMMA with classic periods/shifts on the main price pane.
- **AO (Awesome Oscillator)** in its own lower pane with an independent scale.
- **AC (Acceleration/Deceleration)** in its own lower pane with an independent scale.
- **Williams Zones** candle coloring (green/red/gray) from AO + AC state.
- **Wiseman 1W / 1W-R signals** plotted as up/down markers on the price pane.
- The chart now preserves the current visible range during live polling, so it does **not** shift right on every refresh.
- API mode now keeps chart market-data updates and strategy/backtest recalculation on separate background workers, so the chart can refresh faster without blocking on full strategy payload rebuilds.
- After the initial load, API mode applies incremental series updates while the strategy worker only refreshes markers and execution overlays after new closed bars arrive.
- The price, AO, and AC panes now use synchronized time ranges and fixed right-scale widths for tighter TradingView-style visual alignment.

Use the checkboxes in the top bar to toggle each indicator on/off.

## Paper-trading execution engine foundation

The repo now includes `backtesting.live_execution.PaperTradingEngine`, a reusable paper broker that can:

- accept interpreted live signals (`enter`, `exit`, `scale`, `reverse`, `cancel`)
- simulate `market`, `limit`, `stop`, and `stop_limit` orders
- monitor open positions with optional stop-loss / take-profit levels
- track cash, equity, realized PnL, unrealized PnL, fees, fills, and working orders

This is intended to be the execution-layer bridge between live candle polling and a future Selenium trading adapter.

Example startup command using real-time strategy and execution flags:

```powershell
python .\examples\run_paper_trading.py --symbol BTC_USDT --interval Min60 --capital 10000 --order-type market --size-mode units --size-value 1 --max-position-size 5000 --max-leverage 2 --equity-cutoff 7000 --strategy wiseman --wiseman-1w-contracts 1 --wiseman-2w-contracts 0 --wiseman-3w-contracts 0 --gator-width-lookback 180 --gator-width-mult 0.5 --gator-width-valid-factor 1.0 --wiseman-profit-protection-teeth-exit --wiseman-profit-protection-min-bars 15 --wiseman-profit-protection-min-unrealized-return 0.01 --no-wiseman-profit-protection-require-gator-open --wiseman-profit-protection-volatility-lookback 30 --wiseman-profit-protection-annualized-volatility-scaler 0.85 --wiseman-cancel-reversal-on-first-exit --wiseman-gator-direction-mode 1 --wiseman-reversal-cooldown 1 --wiseman-profit-protection-lips-exit --wiseman-profit-protection-lips-volatility-trigger 0.04 --wiseman-profit-protection-lips-profit-trigger-mult 5 --wiseman-profit-protection-lips-recent-trade-lookback 10 --wiseman-profit-protection-lips-min-unrealized-return 0.03 --wiseman-profit-protection-lips-arm-on-min-unrealized-return --1W-divergence-filter 240 --out .\artifacts_paper
```

While that process runs, it continuously refreshes `paper_dashboard.html`, `paper_dashboard_data.json`, `paper_dashboard_data.js`, `paper_status.md`, and `paper_trades.md` inside the output folder so you can watch the current position, historical trades/fills, equity curve, and drawdown in near real time. The paper runner now polls every second by default, keeps the dashboard mark price/chart aligned with the latest fetched candle, and allows already-qualified stop/entry/exit levels to trigger intrabar on live snapshots while close-dependent setup formation still waits for bar closes. The dashboard now loads from the local `.js` data artifact so it works even when you open the HTML file directly from disk.

```powershell
Start-Process .\artifacts_paper\paper_dashboard.html
```
