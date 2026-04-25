from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .resample import normalize_timeframe
from .stats import compute_performance_stats, infer_periods_per_year
from .strategy import Strategy


@dataclass(slots=True)
class BacktestConfig:
    initial_capital: float = 10_000.0
    fee_rate: float = 0.0
    slippage_rate: float = 0.0
    size: float = 1.0
    contracts: float = 1.0
    position_size_mode: str = "equity_percent"
    position_size_value: float = 1.0
    volatility_target_annual: float = 0.20
    volatility_lookback: int = 20
    volatility_min_scale: float = 0.25
    volatility_max_scale: float = 3.0
    execute_on_signal_bar: bool = False
    signal_timeframe: str | None = None
    signal_timeframe_progressive: bool = True
    max_intrabar_evaluations_per_signal_bar: int = 24
    signal_timeframe_history_bars: int | None = None


@dataclass(slots=True)
class Trade:
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    units: float
    pnl: float
    return_pct: float
    holding_bars: int


@dataclass(slots=True)
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: list[Trade]
    stats: dict[str, float]

    def trades_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(t) for t in self.trades])


class BacktestEngine:
    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(self, data: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        required = {"open", "high", "low", "close"}
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        if len(data) < 2:
            raise ValueError("At least two rows are required")
        if self.config.size <= 0:
            raise ValueError("size must be positive")
        if self.config.contracts <= 0:
            raise ValueError("contracts must be positive")
        if self.config.position_size_value <= 0:
            raise ValueError("position_size_value must be positive")
        if self.config.volatility_lookback <= 1:
            raise ValueError("volatility_lookback must be greater than 1")
        if self.config.volatility_target_annual <= 0:
            raise ValueError("volatility_target_annual must be positive")
        if self.config.volatility_min_scale <= 0 or self.config.volatility_max_scale <= 0:
            raise ValueError("volatility scale bounds must be positive")
        if self.config.volatility_min_scale > self.config.volatility_max_scale:
            raise ValueError("volatility_min_scale must be <= volatility_max_scale")
        if self.config.max_intrabar_evaluations_per_signal_bar <= 0:
            raise ValueError("max_intrabar_evaluations_per_signal_bar must be positive")
        if (
            self.config.signal_timeframe_history_bars is not None
            and self.config.signal_timeframe_history_bars <= 0
        ):
            raise ValueError("signal_timeframe_history_bars must be positive when provided")

        signals, signal_fill_prices = self._generate_signals(data=data, strategy=strategy)
        close = data["close"].astype("float64")
        close_returns = close.pct_change().fillna(0.0)

        capital = float(self.config.initial_capital)
        equity_values = [capital]
        positions = [0]

        units = 0.0
        entry_price = 0.0
        entry_time: pd.Timestamp | None = None
        entry_index = 0
        trades: list[Trade] = []
        total_fees_paid = 0.0
        total_slippage_paid = 0.0
        total_profit_before_fees = 0.0
        total_volume_traded = 0.0
        max_effective_leverage_used = 0.0

        for i in range(1, len(data)):
            signal_now = int(signals.iloc[i])
            signal_fill_price_now = float(signal_fill_prices.iloc[i]) if pd.notna(signal_fill_prices.iloc[i]) else float("nan")
            bar_close = float(close.iloc[i])
            bar_high = float(data["high"].iloc[i])
            bar_low = float(data["low"].iloc[i])
            if pd.notna(signal_fill_price_now) and bar_low <= signal_fill_price_now <= bar_high:
                price = signal_fill_price_now
            else:
                price = bar_close
            ts = data.index[i]
            desired_signal = signal_now if self.config.execute_on_signal_bar else int(signals.iloc[i - 1])
            current_signal = 0 if units == 0 else (1 if units > 0 else -1)

            if units != 0 and desired_signal != current_signal:
                fill = price * (1.0 - self.config.slippage_rate * current_signal)
                slippage_paid = abs(price - fill) * abs(units)
                total_slippage_paid += float(slippage_paid)
                gross = (fill - entry_price) * units
                fee = abs(fill * units) * self.config.fee_rate
                total_fees_paid += float(fee)
                total_profit_before_fees += float(gross)
                pnl = gross - fee
                capital += pnl
                total_volume_traded += abs(fill * units)
                trade = Trade(
                    side="long" if units > 0 else "short",
                    entry_time=entry_time if entry_time is not None else ts,
                    exit_time=ts,
                    entry_price=entry_price,
                    exit_price=fill,
                    units=abs(units),
                    pnl=float(pnl),
                    return_pct=float(pnl / self.config.initial_capital) if self.config.initial_capital else 0.0,
                    holding_bars=i - entry_index,
                )
                trades.append(trade)
                units = 0.0
                current_signal = 0

            if units == 0 and desired_signal != 0:
                fill = price * (1.0 + self.config.slippage_rate * desired_signal)
                notional = self._compute_position_notional(
                    capital=capital,
                    close_returns=close_returns,
                    bar_index=i,
                )
                fee = notional * self.config.fee_rate
                units = (notional / fill) * desired_signal
                slippage_paid = abs(fill - price) * abs(units)
                total_slippage_paid += float(slippage_paid)
                total_fees_paid += float(fee)
                total_volume_traded += abs(fill * units)
                entry_price = fill
                entry_time = ts
                entry_index = i
                capital -= fee

            mark_to_market = capital if units == 0 else capital + (price - entry_price) * units
            if units != 0 and mark_to_market != 0:
                current_leverage = abs(price * units) / abs(mark_to_market)
                max_effective_leverage_used = max(max_effective_leverage_used, float(current_leverage))
            equity_values.append(float(mark_to_market))
            positions.append(0 if units == 0 else (1 if units > 0 else -1))

        if units != 0:
            final_price = float(close.iloc[-1])
            final_ts = data.index[-1]
            side = 1 if units > 0 else -1
            fill = final_price * (1.0 - self.config.slippage_rate * side)
            slippage_paid = abs(final_price - fill) * abs(units)
            total_slippage_paid += float(slippage_paid)
            gross = (fill - entry_price) * units
            fee = abs(fill * units) * self.config.fee_rate
            total_fees_paid += float(fee)
            total_profit_before_fees += float(gross)
            pnl = gross - fee
            capital += pnl
            total_volume_traded += abs(fill * units)
            trade = Trade(
                side="long" if units > 0 else "short",
                entry_time=entry_time if entry_time is not None else final_ts,
                exit_time=final_ts,
                entry_price=entry_price,
                exit_price=fill,
                units=abs(units),
                pnl=float(pnl),
                return_pct=float(pnl / self.config.initial_capital) if self.config.initial_capital else 0.0,
                holding_bars=(len(data) - 1) - entry_index,
            )
            trades.append(trade)
            units = 0.0
            positions[-1] = 0
            equity_values[-1] = float(capital)

        equity = pd.Series(equity_values, index=data.index, dtype="float64")
        returns = equity.pct_change().fillna(0.0)
        positions_series = pd.Series(positions, index=data.index, dtype="int8")
        periods_per_year = infer_periods_per_year(data.index, default=252)
        stats = compute_performance_stats(
            equity_curve=equity,
            returns=returns,
            trades=trades,
            periods_per_year=periods_per_year,
            positions=positions_series,
        )
        stats["final_equity"] = float(equity.iloc[-1])
        stats["total_trades"] = float(len(trades))
        stats["total_slippage_paid"] = float(total_slippage_paid)
        stats["total_fees_paid"] = float(total_fees_paid)
        stats["total_financing_paid"] = 0.0
        stats["total_profit_before_fees"] = float(total_profit_before_fees)
        stats["total_volume_traded"] = float(total_volume_traded)
        stats["max_effective_leverage_used"] = float(max_effective_leverage_used)

        return BacktestResult(
            equity_curve=equity,
            returns=returns,
            positions=positions_series,
            trades=trades,
            stats=stats,
        )

    def _generate_signals(self, data: pd.DataFrame, strategy: Strategy) -> tuple[pd.Series, pd.Series]:
        signal_timeframe = self.config.signal_timeframe
        if signal_timeframe is None or signal_timeframe.strip() == "":
            signals = strategy.generate_signals(data).reindex(data.index).fillna(0).astype("int8")
            fill_prices = getattr(strategy, "signal_fill_prices", None)
            if isinstance(fill_prices, pd.Series):
                fills = fill_prices.reindex(data.index).astype("float64")
            else:
                fills = pd.Series(float("nan"), index=data.index, dtype="float64")
            return signals, fills

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("signal_timeframe requires a DatetimeIndex")

        if not data.index.is_monotonic_increasing:
            raise ValueError("Data index must be sorted ascending for signal_timeframe simulation")

        if not self.config.signal_timeframe_progressive:
            return self._generate_signals_on_closed_timeframe(data=data, strategy=strategy, signal_timeframe=signal_timeframe)

        rule = normalize_timeframe(signal_timeframe)
        source_freq = data.index.to_series().diff().dropna().median()
        try:
            target_freq = pd.to_timedelta(rule)
        except (ValueError, TypeError):
            target_freq = None
        if pd.notna(source_freq) and target_freq is not None and target_freq <= source_freq:
            signals = strategy.generate_signals(data).reindex(data.index).fillna(0).astype("int8")
            fill_prices = getattr(strategy, "signal_fill_prices", None)
            if isinstance(fill_prices, pd.Series):
                fills = fill_prices.reindex(data.index).astype("float64")
            else:
                fills = pd.Series(float("nan"), index=data.index, dtype="float64")
            return signals, fills

        signals = pd.Series(index=data.index, dtype="int8")
        fill_prices = pd.Series(float("nan"), index=data.index, dtype="float64")
        closed_bars = pd.DataFrame(columns=data.columns)
        aggregation_has_volume = "volume" in data.columns
        prior_bucket_side = 0

        for bucket_start, bucket in data.groupby(pd.Grouper(freq=rule), sort=True):
            if bucket.empty:
                continue

            bucket_signals = pd.Series(index=bucket.index, dtype="float64")
            bucket_fill_prices = pd.Series(float("nan"), index=bucket.index, dtype="float64")
            bucket_open = float(bucket["open"].iloc[0])
            bucket_high = bucket["high"].astype("float64").cummax()
            bucket_low = bucket["low"].astype("float64").cummin()
            bucket_close = bucket["close"].astype("float64")
            bucket_volume = (
                bucket["volume"].astype("float64").cumsum()
                if aggregation_has_volume
                else pd.Series(0.0, index=bucket.index, dtype="float64")
            )
            if len(bucket.index) <= self.config.max_intrabar_evaluations_per_signal_bar:
                eval_indices = list(range(len(bucket.index)))
            else:
                eval_count = self.config.max_intrabar_evaluations_per_signal_bar
                eval_indices = sorted(
                    set(
                        [
                            int(round(v))
                            for v in np.linspace(0, len(bucket.index) - 1, num=eval_count)
                        ]
                    )
                )
                if 0 not in eval_indices:
                    eval_indices.insert(0, 0)
                last_idx = len(bucket.index) - 1
                if last_idx not in eval_indices:
                    eval_indices.append(last_idx)

            for i in eval_indices:
                ts = bucket.index[i]
                partial_bar = {
                    "open": bucket_open,
                    "high": float(bucket_high.loc[ts]),
                    "low": float(bucket_low.loc[ts]),
                    "close": float(bucket_close.loc[ts]),
                }
                if aggregation_has_volume:
                    partial_bar["volume"] = float(bucket_volume.loc[ts])

                partial_df = pd.DataFrame([partial_bar], index=pd.DatetimeIndex([bucket_start]))
                snapshot_agg = pd.concat([closed_bars, partial_df]) if len(closed_bars) else partial_df
                history_bars = self.config.signal_timeframe_history_bars
                strategy_input = snapshot_agg.tail(history_bars) if history_bars is not None else snapshot_agg
                aggregated_signal = strategy.generate_signals(strategy_input).iloc[-1]
                bucket_signals.loc[ts] = int(aggregated_signal) if pd.notna(aggregated_signal) else 0
                intrabar_fill_series = getattr(strategy, "signal_fill_prices", None)
                if isinstance(intrabar_fill_series, pd.Series) and len(intrabar_fill_series):
                    maybe_fill = intrabar_fill_series.iloc[-1]
                    if pd.notna(maybe_fill):
                        bucket_fill_prices.loc[ts] = float(maybe_fill)

            bucket_signals = bucket_signals.ffill().bfill().fillna(0.0)
            bucket_signals_int = bucket_signals.astype("int8")
            filtered_bucket_signals = pd.Series(index=bucket.index, dtype="int8")
            filtered_bucket_fills = bucket_fill_prices.copy()
            current_side = int(prior_bucket_side)
            side_changes = 0

            for ts in bucket.index:
                proposed_side = int(bucket_signals_int.loc[ts])
                if proposed_side != current_side:
                    if side_changes >= 1:
                        filtered_bucket_signals.loc[ts] = current_side
                        filtered_bucket_fills.loc[ts] = float("nan")
                        continue
                    current_side = proposed_side
                    side_changes += 1
                filtered_bucket_signals.loc[ts] = current_side

            prior_bucket_side = current_side
            signals.loc[bucket.index] = filtered_bucket_signals
            fill_prices.loc[bucket.index] = filtered_bucket_fills

            final_bar = {
                "open": bucket_open,
                "high": float(bucket_high.iloc[-1]),
                "low": float(bucket_low.iloc[-1]),
                "close": float(bucket_close.iloc[-1]),
            }
            if aggregation_has_volume:
                final_bar["volume"] = float(bucket_volume.iloc[-1])
            closed_bars.loc[bucket_start, list(final_bar.keys())] = list(final_bar.values())

        return (
            signals.reindex(data.index).fillna(0).astype("int8"),
            fill_prices.reindex(data.index).astype("float64"),
        )

    def _generate_signals_on_closed_timeframe(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        signal_timeframe: str,
    ) -> tuple[pd.Series, pd.Series]:
        rule = normalize_timeframe(signal_timeframe)
        source_freq = data.index.to_series().diff().dropna().median()
        try:
            target_freq = pd.to_timedelta(rule)
        except (ValueError, TypeError):
            target_freq = None

        if pd.notna(source_freq) and target_freq is not None and target_freq <= source_freq:
            signals = strategy.generate_signals(data).reindex(data.index).fillna(0).astype("int8")
            fill_prices = getattr(strategy, "signal_fill_prices", None)
            if isinstance(fill_prices, pd.Series):
                fills = fill_prices.reindex(data.index).astype("float64")
            else:
                fills = pd.Series(float("nan"), index=data.index, dtype="float64")
            return signals, fills

        htf = data.resample(rule).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                **({"volume": "sum"} if "volume" in data.columns else {}),
            }
        ).dropna(subset=["open", "high", "low", "close"])

        htf_signals = strategy.generate_signals(htf).reindex(htf.index).fillna(0).astype("int8")
        htf_fill_prices = getattr(strategy, "signal_fill_prices", None)
        htf_fills = (
            htf_fill_prices.reindex(htf.index).astype("float64")
            if isinstance(htf_fill_prices, pd.Series)
            else pd.Series(float("nan"), index=htf.index, dtype="float64")
        )

        signals = pd.Series(0, index=data.index, dtype="int8")
        fills = pd.Series(float("nan"), index=data.index, dtype="float64")
        previous_side = 0
        grouped = list(data.groupby(pd.Grouper(freq=rule), sort=True))
        for bucket_start, bucket in grouped:
            if bucket.empty or bucket_start not in htf_signals.index:
                continue
            new_side = int(htf_signals.loc[bucket_start])
            bucket_signal = pd.Series(previous_side, index=bucket.index, dtype="int8")
            if new_side != previous_side:
                bucket_signal.iloc[-1] = new_side
                maybe_fill = htf_fills.loc[bucket_start]
                if pd.notna(maybe_fill):
                    fills.loc[bucket.index[-1]] = float(maybe_fill)
            signals.loc[bucket.index] = bucket_signal
            previous_side = new_side

        return signals.astype("int8"), fills.astype("float64")

    def _compute_position_notional(self, capital: float, close_returns: pd.Series, bar_index: int) -> float:
        mode = self.config.position_size_mode.strip().lower()
        if mode == "static_usd":
            return float(self.config.position_size_value)

        base_equity_notional = capital * self.config.position_size_value
        if mode == "equity_percent":
            return float(base_equity_notional)

        if mode == "volatility_scaled":
            window = close_returns.iloc[max(0, bar_index - self.config.volatility_lookback) : bar_index]
            realized_vol = float(window.std(ddof=0) * (252.0 ** 0.5)) if len(window) > 1 else 0.0
            if realized_vol <= 0:
                scale = 1.0
            else:
                scale = self.config.volatility_target_annual / realized_vol
            scale = max(self.config.volatility_min_scale, min(scale, self.config.volatility_max_scale))
            return float(base_equity_notional * scale)

        raise ValueError(
            "position_size_mode must be one of: static_usd, equity_percent, volatility_scaled"
        )
