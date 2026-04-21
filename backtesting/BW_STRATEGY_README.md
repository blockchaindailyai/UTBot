# BW Strategy (Bill Williams Unified Mode)

This document describes the current `BWStrategy` implementation in `backtesting/strategy.py` and the CLI flags that configure it in `examples/run_backtest.py`.

---

## What BW strategy is

`BWStrategy` is a **single strategy that blends**:

1. **1W (First Wiseman) breakout logic** for primary directional entries.
2. **1W-R reversal logic** around the same 1W setup levels.
3. **NTD-style fractal entries** (optional) when the gator wakes up from compression.
4. **Fractal add-on contracts** (optional) to pyramid in-trend.
5. **Red Gator Teeth profit protection** with optional volatility scaling.

In short: BW is not just 1W anymore. It now supports a hybrid 1W + fractal workflow with explicit controls for each component.

---

## Data and indicators used

The strategy expects OHLCV-like bars with at least:
- `open`
- `high`
- `low`
- `close`

Internally it computes:
- **Alligator lines** (`jaw`, `teeth`, `lips`).
- **Awesome Oscillator (AO)**.
- **Williams fractals**.
- **Sleeping gator / waking gator regime state**.
- **Ranging filter state**.
- **Annualized realized volatility** (for red-teeth threshold scaling when enabled).

---

## Feature summary (new behavior included)

## 1) 1W setup, trigger, stop, and optional AO divergence filter

- Bullish and bearish 1W setup candles are detected and tracked.
- Entry is still breakout-based (break setup high for long / setup low for short) before invalidation.
- Initial stop uses opposite side of setup candle.
- `divergence_filter_bars` can require recent AO divergence confirmation.

## 2) 1W-R reversal handling

- If the active 1W setup is structurally reversal-armed, breaking the opposite setup extreme flips the position.
- Reversal stop is anchored to extrema over the setup-to-trigger window.
- Reversal stop exit reason remains: `"1W Reversal Stop"`.

## 3) Optional NTD initial fractal entries (new)

When enabled, BW can place/trigger initial fractal entries if:
- gator regime transitions indicate wake-up from compression,
- market is not filtered out as ranging,
- qualifying fractal levels exist.

Notes:
- Fractal-driven entries can open from flat **or** reverse an existing opposite position.
- Fractal-driven position source is tracked separately from 1W/reversal source.

## 4) Optional fractal add-on contracts (new)

After a directional position is active, additional same-direction contracts can be added using validated fractal breakouts.

This allows BW to pyramid beyond the base entry size without requiring additional 1W triggers.

## 5) Red Gator Teeth profit protection (expanded controls)

Red-teeth PP is enabled by default in BW.

An exit is allowed only when all configured gates pass:
- minimum bars in position,
- minimum favorable excursion (optionally volatility-scaled),
- close breaches teeth against trade direction.

Exit reason: `"Red Gator Teeth PP"`.

---

## State outputs exposed by BW strategy

BW publishes the following strategy series for execution/charting layers:

- `signal_fill_prices`
- `signal_stop_loss_prices`
- `signal_contracts`
- `signal_first_wiseman_setup_side`
- `signal_first_wiseman_setup_marker_side`
- `signal_first_wiseman_ignored_reason`
- `signal_first_wiseman_reversal_side`
- `signal_add_on_fractal_fill_side`
- `signal_fractal_position_side`
- `signal_exit_reason`
- `signal_intrabar_events`

These are useful for debugging state-machine transitions and chart annotations.

---

## CLI flags for BW strategy (`examples/run_backtest.py`)

Use `--strategy bw` to run this strategy mode.

## Core selection

- `--strategy bw`

## 1W controls

- `--bw-1w-divergence-filter <int>` (alias: `--1W-divergence-filter-bw`, default `0`)  
  AO divergence lookback bars for 1W setup filtering (`0` disables).
- `--bw-1w-gator-open-lookback <int>` (default `0`)  
  Rolling lookback used by the percentile-based gator-open strength filter for 1W (`0` disables).
- `--bw-1w-gator-open-percentile <float>` (default `50.0`)  
  Percentile threshold (0-100) of recent gator widths; current gator width must be strictly greater than this rolling percentile to allow a 1W. Example: `lookback=100`, `percentile=50` means the current gator width must be wider than the median of the last 100 bars.
- `--bw-1w-contracts <int>` (default `1`)  
  Base contracts used for 1W entries.
- `--allow-close-on-1w-d` / `--no-allow-close-on-1w-d` (default disabled)
- `--allow-close-on-1w-d-min-unrealized-return <float>` (default `0.0`)  
  Allow a filtered opposite `1W-D` signal (divergence-filter rejected) to flatten an open BW position once favorable unrealized return reaches this threshold.
- `--allow-close-on-1w-a` / `--no-allow-close-on-1w-a` (default disabled)
- `--allow-close-on-1w-a-min-unrealized-return <float>` (default `0.0`)  
  Allow a filtered opposite `1W-A` signal (alligator/gator gate rejected) to flatten an open BW position once favorable unrealized return reaches this threshold.

## NTD initial fractal controls

- `--bw-ntd-initial-fractal-enabled` / `--no-bw-ntd-initial-fractal-enabled` (default disabled)
- `--bw-ntd-initial-fractal-contracts <int>` (default `1`)
- `--bw-ntd-sleeping-gator-lookback <int>` (default `50`)
- `--bw-ntd-sleeping-gator-tightness-mult <float>` (default `0.75`)
- `--bw-ntd-ranging-lookback <int>` (default `20`)
- `--bw-ntd-ranging-max-span-pct <float>` (default `0.025`)

## Fractal add-on controls

- `--bw-fractal-add-on-contracts <int>` (alias: `--Fractal-add-on-contracts`, default `0`)  
  Additional contracts to add on validated same-direction fractal add-on fills.

## Red-teeth profit-protection controls

- `--bw-profit-protection-red-teeth-exit` / `--no-bw-profit-protection-red-teeth-exit` (default enabled)
- `--bw-profit-protection-red-teeth-min-bars <int>` (default `3`)
- `--bw-profit-protection-red-teeth-min-unrealized-return <float>` (default `1.0`)
- `--bw-profit-protection-red-teeth-volatility-lookback <int>` (default `20`)
- `--bw-profit-protection-red-teeth-annualized-volatility-scaler <float>` (default `1.0`)

## Green-lips profit-protection controls

- `--bw-profit-protection-green-lips-exit` / `--no-bw-profit-protection-green-lips-exit` (default enabled)
- `--bw-profit-protection-green-lips-min-bars <int>` (default `3`)
- `--bw-profit-protection-green-lips-min-unrealized-return <float>` (default `1.1`)
- `--bw-profit-protection-green-lips-volatility-lookback <int>` (default `20`)
- `--bw-profit-protection-green-lips-annualized-volatility-scaler <float>` (default `1.0`)
- `--bw-profit-protection-red-teeth-latch-min-unrealized-return` / `--no-bw-profit-protection-red-teeth-latch-min-unrealized-return` (default disabled)  
  When enabled, the red-teeth unrealized-return gate stays armed after first being reached during a position.
- `--bw-profit-protection-green-lips-latch-min-unrealized-return` / `--no-bw-profit-protection-green-lips-latch-min-unrealized-return` (default disabled)  
  When enabled, the green-lips unrealized-return gate stays armed after first being reached during a position.

## Williams Zones profit-protection controls

- `--bw-profit-protection-zones-exit` / `--no-bw-profit-protection-zones-exit` (default disabled)
- `--bw-profit-protection-zones-min-bars <int>` (default `3`)
- `--bw-profit-protection-zones-min-unrealized-return <float>` (default `1.0`)
- `--bw-profit-protection-zones-volatility-lookback <int>` (default `20`)
- `--bw-profit-protection-zones-annualized-volatility-scaler <float>` (default `1.0`)
- `--bw-profit-protection-zones-min-same-color-bars <int>` (default `5`)

> For BW profit-protection controls, setting an annualized volatility scaler to `0` disables volatility scaling and uses the raw configured percentage threshold.

## Peak-drawdown profit-protection controls

- `--bw-peak-drawdown-exit` / `--no-bw-peak-drawdown-exit` (default disabled)
- `--bw-peak-drawdown-exit-pct <float>` (default `0.01`)  
  Exit when drawdown from peak favorable return exceeds this base percent scaled by annualized volatility(20).
- `--bw-peak-drawdown-exit-volatility-lookback <int>` (default `20`)  
  Rolling bar lookback used to compute annualized volatility for the peak-drawdown threshold.
- `--bw-peak-drawdown-exit-annualized-volatility-scaler <float>` (default `1.0`)  
  Reference annualized volatility used to scale the peak-drawdown threshold (`0` disables volatility scaling and uses the raw percent).

## Sigma-move profit-protection controls

- `--bw-profit-protection-sigma-move-exit` / `--no-bw-profit-protection-sigma-move-exit` (default disabled)
- `--bw-profit-protection-sigma-move-lookback <int>` (default `20`)  
  Rolling close lookback used to compute the sigma baseline (`mean +/- sigma * std`).
- `--bw-profit-protection-sigma-move-sigma <float>` (default `2.0`)  
  Number of standard deviations required for a same-bar sigma touch before flattening at bar close.

---

## Example commands

### Basic BW run

```powershell
python .\examples\run_backtest.py --csv .\examples\sample_ohlcv.csv --strategy bw --out .\artifacts_bw
```

### BW with NTD initial fractals and add-ons

```powershell
python .\examples\run_backtest.py --csv .\examples\sample_ohlcv.csv --strategy bw --bw-ntd-initial-fractal-enabled --bw-ntd-initial-fractal-contracts 2 --bw-fractal-add-on-contracts 1 --out .\artifacts_bw
```

### BW with stricter 1W divergence + custom red-teeth protection

```powershell
python .\examples\run_backtest.py --csv .\examples\sample_ohlcv.csv --strategy bw --bw-1w-divergence-filter 240 --bw-profit-protection-red-teeth-min-bars 5 --bw-profit-protection-red-teeth-min-unrealized-return 0.03 --bw-profit-protection-red-teeth-volatility-lookback 40 --bw-profit-protection-red-teeth-annualized-volatility-scaler 0.85 --out .\artifacts_bw
```

---

## Practical tuning tips

- Start with defaults and only change **one group at a time** (1W, then NTD, then PP).
- If trade count is too low, reduce filtering intensity:
  - lower `--bw-1w-divergence-filter`,
  - increase `--bw-ntd-ranging-max-span-pct`,
  - raise `--bw-ntd-sleeping-gator-tightness-mult`.
- If exits are too loose, tighten red-teeth PP with:
  - lower `--bw-profit-protection-red-teeth-min-bars`,
  - lower `--bw-profit-protection-red-teeth-min-unrealized-return`.
- If exits are too aggressive, do the opposite or disable red-teeth PP for baseline comparisons.

---

## Maintenance note

Treat this file as the BW strategy changelog-style README.
When strategy behavior changes, update:
1. the feature summary,
2. state-machine behavior,
3. CLI flags + defaults,
4. example commands.
