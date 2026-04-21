# Wiseman Strategy Trade State Machine (Code-Level Breakdown)

This document explains every setup and trade path implemented in `WisemanStrategy.generate_signals` and how the backtest engine executes those signals.

## 1) Indicator and regime calculations

On each run, the strategy computes:

- **Alligator lines** on median price `(high + low)/2`
  - Jaw = SMMA(13) shifted 8
  - Teeth = SMMA(8) shifted 5
  - Lips = SMMA(5) shifted 3
- **Gator-width filter components**
  - `red_to_green_distance = |teeth - lips|`
  - `green_to_midpoint_distance = |lips - midpoint|`
  - `gator_width_valid = red_to_green_distance < (green_to_midpoint_distance * gator_width_valid_factor)`
  - `gator_width_valid_factor` defaults to `1.0` and is configurable from CLI via `--gator-width-valid-factor`
- **Gator closed filter**
  - `gator_range = max(jaw,teeth,lips) - min(jaw,teeth,lips)`
  - `gator_slope = |Δjaw| + |Δteeth| + |Δlips|`
  - Baselines are rolling medians over `gator_width_lookback`
  - `gator_closed` when both range and slope are below baseline × multiplier
- **Directional Alligator alignment**
  - `gator_up` when `lips > teeth > jaw`
  - `gator_down` when `lips < teeth < jaw`
- **AO momentum color**
  - `ao = SMA5(median) - SMA34(median)`
  - `ao_green` when AO rises vs prior bar
  - `ao_red` when AO falls vs prior bar
- **Williams fractals** used for 3rd Wiseman setup confirmation.

## 2) Signals and sizing model

The strategy writes a **directional signal** (`-1`, `0`, `1`) and a **contract multiplier** series.

Position sizing by phase:

- First Wiseman only: **1 contract**
- + Second Wiseman triggered: **4 contracts total**
- + Third Wiseman triggered: **6 contracts total**
- Second + Third both triggered: **10 contracts total**

This is encoded by `_position_size(second_triggered, third_triggered)`.

## 3) First Wiseman setup detection (raw candidates)

Starting from bar index `i >= 2`, the strategy inspects bar `i-1` as potential setup with bar `i` as right-side confirmer:

### Bearish 1W candidate (`side = -1`)

Requires ALL on bar `i-1`:

1. **Local peak**: `high[i-1] > high[i-2]` and `high[i-1] > high[i]`
2. **Bear candle body**: `open[i-1] > close[i-1]`
3. **Alligator up alignment**: `gator_up[i-1]`
4. **Gator width valid**: `gator_width_valid[i-1]`
5. **AO green**: `ao_green[i-1]`

Candidate stores:

- trigger low (`p_low = low[i-1]`) for short breakout,
- invalidation high (`p_high = high[i-1]`),
- setup bar (`p_bar = i-1`).

### Bullish 1W candidate (`side = +1`)

Requires ALL on bar `i-1`:

1. **Local trough**: `low[i-1] < low[i-2]` and `low[i-1] < low[i]`
2. **Bull candle body**: `close[i-1] > open[i-1]`
3. **Alligator down alignment**: `gator_down[i-1]`
4. **Gator width valid**: `gator_width_valid[i-1]`
5. **AO red**: `ao_red[i-1]`

Candidate stores:

- trigger high (`p_high = high[i-1]`) for long breakout,
- invalidation low (`p_low = low[i-1]`),
- setup bar (`p_bar = i-1`).

## 4) First Wiseman queueing and immediate rejection rules

Each new candidate is tagged in `first_wiseman_setup_side`.

Then one of these happens:

- **Queued** into `pending_setups`, OR
- **Immediately ignored** with reason `weaker_than_active_setup` if:
  - current live position is same side as the new setup, and
  - either setup levels are still armed (`active_levels_armed=True`) **or** a reversal trade is active (`reversal_position_active=True`).

Interpretation: this condition says *"ignore same-side replacements while an existing same-side trade regime is already active, including post-reversal trades."*

Why that matters:
- The original triggered setup has its `active_high`/`active_low` levels armed.
- A reversal trade has its own active risk anchor (`reversal_stop_level`).
- Replacing either regime with every newer same-side setup would constantly move protective/reversal references.

So while either regime is active, newer same-side setups are treated as weaker/informational and tagged `weaker_than_active_setup` instead of re-anchoring risk logic.

## 5) Active-level reversal logic from a previously triggered 1W

When `active_levels_armed` is true, the prior setup’s opposite level acts like a hard reversal trigger:

- If currently short and `high_now >= active_high`: force reverse to long.
- If currently long and `low_now <= active_low`: force reverse to short.

On this forced reversal:

- `first_wiseman_reversal_side` is marked.
- 2W/3W triggers are reset.
- `reversal_position_active = True`.
- A **reversal stop level** is computed from extreme since setup bar:
  - Short→Long reversal stop = minimum low from setup bar to current bar.
  - Long→Short reversal stop = maximum high from setup bar to current bar.
- New 2W/3W setup pipelines are disabled.

## 6) Pending setup lifecycle: activate, cancel, invalidate

For each pending setup on each bar:

### Global cancel condition

- If `gator_closed` on current bar: setup is canceled with `gator_closed_canceled`.

### Bearish pending setup (`side=-1`)

- **Activation trigger**: `low_now <= p_low` (short breakout)
- **Pre-trigger invalidation**: if `high_now >= p_high` before breakout, mark `invalidation_before_trigger` and drop setup.

### Bullish pending setup (`side=+1`)

- **Activation trigger**: `high_now >= p_high` (long breakout)
- **Pre-trigger invalidation**: if `low_now <= p_low` before breakout, mark `invalidation_before_trigger` and drop setup.

### End-of-data unresolved setup

After loop, any still-pending setup gets `no_breakout_until_end_of_data`.

## 7) Same-bar breakout-and-stop behavior (special path)

When a setup activates, strategy enters side immediately and arms opposite level.

But on the **same bar**, if price also crosses the opposite level:

- If reversal window is open (`i >= p_bar + 3`), it performs same-bar reversal (marks `first_wiseman_reversal_side`, sets reversal-stop regime).
- Otherwise, it flat-closes and records reason `same_bar_stop_before_reversal_window`.

This creates a strict 3-bar minimum before a valid same-bar reversal from the original setup bar.

## 8) Second Wiseman (Super AO) add-on setup and fill

Second Wiseman is only eligible when:

- position exists,
- `second_wiseman_triggered` is still false (2W has not filled yet),
- `second_wiseman_allowed` is true,
- no existing super AO order,
- and at least 3 AO bars are available.

Important nuance: this means 2W can still set up/fill even if 3W has already filled first and increased contracts (for example from 1x to 6x).

### Setup condition

- In short: AO red 3 bars in a row → place 2W short stop at `low_now`.
- In long: AO green 3 bars in a row → place 2W long stop at `high_now`.

`second_wiseman_setup_side` is tagged on setup bar.

### Fill condition

On subsequent bars (`i > setup_bar`):

- Short 2W fills if `low_now <= trigger` and still short.
- Long 2W fills if `high_now >= trigger` and still long.

On fill:

- `second_wiseman_triggered = True`
- contract size escalates to 4 or 10 depending on 3W state
- fill recorded in `signal_fill_prices_second` and `signal_second_wiseman_fill_side`

## 9) Third Wiseman fractal add-on setup and fill

After a first-W activation, strategy creates `third_wiseman_watch = {side, setup_bar}`.

It waits for confirmed fractals (`fractal_i = i-2`) after setup bar.

### Setup condition by side

- In short watch:
  - need a **down fractal** at `fractal_i`
  - fractal low must be **below Teeth**
  - then place 3W short stop at that fractal low
- In long watch:
  - need an **up fractal** at `fractal_i`
  - fractal high must be **above Teeth**
  - then place 3W long stop at that fractal high

`third_wiseman_setup_side` is tagged at fractal bar.

### Fill condition

On subsequent bars (`i > order_bar`):

- Short 3W fills if `low_now <= trigger` and still short.
- Long 3W fills if `high_now >= trigger` and still long.

On fill:

- `third_wiseman_triggered = True`
- contract size escalates to 6 or 10 depending on 2W state
- fill recorded in `signal_fill_prices_third` and `signal_third_wiseman_fill_side`

## 10) Reversal-position stop-out logic (post-reversal kill switch)

When in reversal regime (`reversal_position_active=True`), strategy monitors `reversal_stop_level`.

After entry bar only (`i > entry_i`), if crossed against reversal position:

- close position to flat,
- reset all Wiseman flags (2W/3W allowed/armed/orders/watch),
- stamp fill in first-fill channel,
- clear reversal regime.

This is a distinct close path from normal opposite signal flips.

## 11) Optional teeth-based profit protection exit

When `teeth_profit_protection_enabled=True` on `WisemanStrategy`, an additional exit path is active.

Arming conditions (all required):
- position is open and at least `teeth_profit_protection_min_bars` bars old,
- unrealized return has reached the annualized-volatility-scaled threshold derived from `teeth_profit_protection_min_unrealized_return` and `profit_protection_annualized_volatility_scaler`,
- gator is open (`gator_closed == False`) when `teeth_profit_protection_require_gator_open=True`.

Once armed for the active trade:
- long exits to flat when bar **close < teeth**,
- short exits to flat when bar **close > teeth**.

Guardrails:
- this logic only acts when no other same-bar fill already occurred,
- it does not override first-Wiseman stop/reversal handling,
- it is intended as late-stage profit protection, not immediate post-entry stop behavior.
- when lips-based profit protection is enabled, `lips_profit_protection_volatility_trigger`
  is interpreted relative to `profit_protection_annualized_volatility_scaler`, so the
  effective annualized-volatility threshold is `lips_profit_protection_volatility_trigger * profit_protection_annualized_volatility_scaler`.

## 12) Engine execution behavior that determines actual trade records

`BacktestEngine` translates strategy outputs into real trade events:

- If strategy sets `execute_on_signal_bar=True` (Wiseman does), signal index is current bar `i`; otherwise previous bar.
- Strategy-provided `signal_fill_prices` is preferred over generic order model.
- If desired side changes, engine exits current position then enters new one on same bar when fills are available.
- If side stays same but desired contracts change, engine emits **add** or **reduce** executions (rebalancing path).
- Financing costs are debited each loop while position is open.
- Any open position is forcibly closed on the last bar.

So Wiseman’s staged size upgrades (1→4→6→10) become tangible add/reduce events through `signal_contracts` deltas.

## 13) Complete catalog of explicit 1W ignore/cancel reasons

The strategy writes these strings into `first_wiseman_ignored_reason`:

1. `weaker_than_active_setup`
2. `gator_closed_canceled`
3. `invalidation_before_trigger`
4. `same_bar_stop_before_reversal_window`
5. `no_breakout_until_end_of_data`

These reasons are exhaustive for first-setup non-activation outcomes in current code.

## 14) Every distinct trade path the system can take

At runtime, all observed trades are combinations of the following atomic paths:

1. **1W breakout entry only**, hold until later reversal/close.
2. **1W breakout then same-bar opposite touch**:
   - reverse immediately (if eligible), or
   - flat immediately (if not eligible).
3. **1W breakout then active-level hard reversal** on later bar.
4. **1W breakout + 2W add-on fill** (size jump).
5. **1W breakout + 3W add-on fill** (size jump).
6. **1W breakout + both 2W and 3W fills** (size reaches 10x).
7. **Any reversal-position stopped out** by reversal-stop kill switch.
8. **Pending 1W setup canceled/invalidated**, producing no trade.
9. **Open position forced final-bar close** by engine.

Because the core strategy is a state machine with queued setups + armed levels + staged add-ons, all trade outcomes are deterministic transitions between those states.
