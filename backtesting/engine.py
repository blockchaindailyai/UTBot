from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .quality import generate_data_quality_report
from .stats import compute_performance_stats, infer_periods_per_year
from .strategy import Strategy


def parse_trade_size_equity_milestones(spec: str | None) -> tuple[tuple[float, float], ...]:
    if spec is None:
        return ()
    text = str(spec).strip()
    if not text:
        return ()

    milestones: list[tuple[float, float]] = []
    for raw_entry in text.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        equity_text, separator, usd_text = entry.partition(":")
        if separator != ":":
            raise ValueError(
                "Equity milestone sizing entries must use EQUITY:USD pairs, e.g. '15000:1500,20000:2000'"
            )
        try:
            equity_threshold = float(equity_text)
            usd_notional = float(usd_text)
        except ValueError as exc:
            raise ValueError(
                "Equity milestone sizing entries must contain numeric EQUITY:USD pairs, e.g. '15000:1500'"
            ) from exc
        if not np.isfinite(equity_threshold) or not np.isfinite(usd_notional):
            raise ValueError("Equity milestone sizing entries must be finite numeric values")
        if equity_threshold <= 0 or usd_notional < 0:
            raise ValueError("Equity milestone thresholds must be > 0 and milestone USD notionals must be >= 0")
        milestones.append((equity_threshold, usd_notional))

    milestones.sort(key=lambda item: item[0])
    thresholds = [threshold for threshold, _ in milestones]
    if len(thresholds) != len(set(thresholds)):
        raise ValueError("Equity milestone sizing thresholds must be unique")
    return tuple(milestones)


@dataclass(slots=True)
class BacktestConfig:
    initial_capital: float = 10_000.0
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0002
    spread_rate: float = 0.0

    order_type: str = "market"  # market|limit|stop|stop_limit
    limit_offset_pct: float = 0.001
    stop_offset_pct: float = 0.001
    stop_limit_offset_pct: float = 0.0005

    borrow_rate_annual: float = 0.0
    funding_rate_per_period: float = 0.0
    overnight_rate_annual: float = 0.0

    trade_size_mode: str = "percent_of_equity"  # percent_of_equity|usd|units|hybrid_min_usd_percent|volatility_scaled|stop_loss_scaled|equity_milestone_usd
    trade_size_value: float = 1.0
    trade_size_min_usd: float = 0.0
    trade_size_equity_milestones: tuple[tuple[float, float], ...] = ()
    volatility_target_annual: float = 0.15
    volatility_lookback: int = 20
    volatility_min_scale: float = 0.25
    volatility_max_scale: float = 3.0
    max_leverage: float | None = None
    max_position_size: float | None = None  # absolute USD notional cap per position
    leverage_stop_out_pct: float = 0.0
    max_loss: float | None = None
    max_loss_pct_of_equity: float | None = None
    equity_cutoff: float | None = None
    close_open_position_on_last_bar: bool = True


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
    entry_signal: str = "Unknown"
    exit_signal: str = "Unknown"
    signal_intent_flat_time: pd.Timestamp | None = None
    gross_pnl: float = 0.0
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    financing_cost: float = 0.0
    capital_base: float = 0.0
    peak_unrealized_return_pct: float = 0.0
    worst_unrealized_return_pct: float = 0.0
    giveback_from_peak_pct: float = 0.0
    capture_ratio_vs_peak: float = 0.0
    realized_price_return_pct: float = 0.0
    peak_unrealized_pnl: float = 0.0
    giveback_from_peak_pnl: float = 0.0
    peak_unrealized_position_return_pct: float = 0.0
    realized_position_return_pct: float = 0.0
    giveback_position_return_pct: float = 0.0



@dataclass(slots=True)
class ExecutionEvent:
    event_type: str
    time: pd.Timestamp
    side: str
    price: float
    units: float
    strategy_reason: str | None = None
    sizing_mode: str | None = None
    capital_snapshot: float | None = None
    base_notional: float | None = None
    volatility_scale: float | None = None
    realized_vol_annual: float | None = None
    scaled_notional: float | None = None

@dataclass(slots=True)
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: list[Trade]
    stats: dict[str, float]
    data_quality: dict[str, float | bool]
    execution_events: list[ExecutionEvent]
    total_fees_paid: float
    total_financing_paid: float
    total_profit_before_fees: float

    def trades_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "side": t.side,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "units": t.units,
                    "pnl": t.pnl,
                    "return_pct": t.return_pct,
                    "holding_bars": t.holding_bars,
                    "entry_signal": t.entry_signal,
                    "exit_signal": t.exit_signal,
                    "signal_intent_flat_time": t.signal_intent_flat_time,
                    "gross_pnl": t.gross_pnl,
                    "entry_fee": t.entry_fee,
                    "exit_fee": t.exit_fee,
                    "financing_cost": t.financing_cost,
                    "capital_base": t.capital_base,
                    "peak_unrealized_return_pct": t.peak_unrealized_return_pct,
                    "worst_unrealized_return_pct": t.worst_unrealized_return_pct,
                    "giveback_from_peak_pct": t.giveback_from_peak_pct,
                    "capture_ratio_vs_peak": t.capture_ratio_vs_peak,
                    "realized_price_return_pct": t.realized_price_return_pct,
                    "peak_unrealized_pnl": t.peak_unrealized_pnl,
                    "giveback_from_peak_pnl": t.giveback_from_peak_pnl,
                    "peak_unrealized_position_return_pct": t.peak_unrealized_position_return_pct,
                    "realized_position_return_pct": t.realized_position_return_pct,
                    "giveback_position_return_pct": t.giveback_position_return_pct,
                }
                for t in self.trades
            ]
        )


class BacktestEngine:
    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()
        self._last_sizing_context: dict[str, float | str | None] = {}

    def run(self, data: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        if len(data) < 2:
            raise ValueError("At least two bars are required for backtesting")
        for col in ["open", "high", "low", "close"]:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        raw_signals = strategy.generate_signals(data)
        signals = raw_signals.reindex(data.index).fillna(0)
        def _aligned_series(attr_name: str, *, fill_value: float | None = None) -> pd.Series | None:
            value = getattr(strategy, attr_name, None)
            if not isinstance(value, pd.Series):
                return None
            aligned = value.reindex(data.index)
            if fill_value is not None:
                aligned = aligned.fillna(fill_value)
            return aligned

        signal_fill_prices = _aligned_series("signal_fill_prices")
        signal_fill_prices_first = _aligned_series("signal_fill_prices_first")
        signal_stop_loss_prices = _aligned_series("signal_stop_loss_prices")
        signal_exit_reasons = _aligned_series("signal_exit_reason")
        signal_contracts = _aligned_series("signal_contracts", fill_value=0)
        signal_first_wiseman_setup_side = _aligned_series("signal_first_wiseman_setup_side")
        signal_first_wiseman_reversal_side = _aligned_series("signal_first_wiseman_reversal_side")
        signal_intrabar_events = getattr(strategy, "signal_intrabar_events", None)
        signal_first_wiseman_reversal_side = getattr(strategy, "signal_first_wiseman_reversal_side", None)
        if isinstance(signal_fill_prices, pd.Series):
            signal_fill_prices = signal_fill_prices.reindex(data.index)
        else:
            signal_fill_prices = None
        if isinstance(signal_stop_loss_prices, pd.Series):
            signal_stop_loss_prices = signal_stop_loss_prices.reindex(data.index)
        else:
            signal_stop_loss_prices = None
        if isinstance(signal_fill_prices_first, pd.Series):
            signal_fill_prices_first = signal_fill_prices_first.reindex(data.index)
        else:
            signal_fill_prices_first = None
        if isinstance(signal_exit_reasons, pd.Series):
            signal_exit_reasons = signal_exit_reasons.reindex(data.index)
        else:
            signal_exit_reasons = None
        if isinstance(signal_contracts, pd.Series):
            signal_contracts = signal_contracts.reindex(data.index).fillna(0)
        else:
            signal_contracts = None
        if isinstance(signal_first_wiseman_setup_side, pd.Series):
            signal_first_wiseman_setup_side = signal_first_wiseman_setup_side.reindex(data.index).fillna(0).astype("int8")
        else:
            signal_first_wiseman_setup_side = None
        if isinstance(signal_first_wiseman_reversal_side, pd.Series):
            signal_first_wiseman_reversal_side = (
                signal_first_wiseman_reversal_side.reindex(data.index).fillna(0).astype("int8")
            )
        else:
            signal_first_wiseman_reversal_side = None
        if not isinstance(signal_intrabar_events, dict):
            signal_intrabar_events = {}

        periods_per_year = infer_periods_per_year(data.index, default=252)
        data_quality = generate_data_quality_report(data)

        capital = self.config.initial_capital
        equity_values: list[float] = [capital]

        position = 0
        entry_price = 0.0
        entry_time = None
        entry_index: int | None = None
        entry_capital = 0.0
        entry_notional = 0.0
        liquidation_price: float | None = None
        active_stop_loss_price: float | None = None
        units = 0.0
        trades: list[Trade] = []
        execution_events: list[ExecutionEvent] = []
        positions = pd.Series(0, index=data.index, dtype="int8")
        total_fees_paid = 0.0
        total_financing_paid = 0.0
        total_profit_before_fees = 0.0
        open_entry_fee_balance = 0.0
        open_financing_balance = 0.0
        open_entry_signal = "Unknown"

        execute_on_signal_bar = bool(getattr(strategy, "execute_on_signal_bar", False))
        bankruptcy_index: int | None = None
        cutoff_index: int | None = None
        equity_cutoff = self.config.equity_cutoff
        if equity_cutoff is not None and equity_cutoff <= 0:
            equity_cutoff = None

        def record_closed_trade(
            *,
            close_units: float,
            exit_time: pd.Timestamp,
            exit_fill: float,
            holding_bars: int,
            exit_signal: str,
            exit_index: int,
            signal_intent_flat_time_value: pd.Timestamp | None = None,
        ) -> None:
            nonlocal capital
            nonlocal total_fees_paid
            nonlocal total_profit_before_fees
            nonlocal open_entry_fee_balance
            nonlocal open_financing_balance

            if close_units <= 0 or units <= 0:
                return

            units_before_close = float(units)
            allocation_ratio = close_units / units_before_close
            allocated_entry_fee = open_entry_fee_balance * allocation_ratio
            allocated_financing = open_financing_balance * allocation_ratio

            gross_pnl = (exit_fill - entry_price) * close_units * position
            exit_fee = abs(exit_fill * close_units) * self.config.fee_rate
            pnl = gross_pnl - allocated_entry_fee - exit_fee - allocated_financing

            total_fees_paid += exit_fee
            total_profit_before_fees += gross_pnl
            capital += gross_pnl - exit_fee
            self._ensure_finite(capital, f"capital became non-finite after exit at {exit_time}")

            capital_base = entry_capital * allocation_ratio if entry_capital > 0 else 0.0
            return_pct = (pnl / capital_base) if capital_base > 0 else 0.0
            peak_unrealized_return_pct = 0.0
            worst_unrealized_return_pct = 0.0
            giveback_from_peak_pct = 0.0
            capture_ratio_vs_peak = 0.0
            realized_price_return_pct = 0.0
            peak_unrealized_pnl = 0.0
            giveback_from_peak_pnl = 0.0
            peak_unrealized_position_return_pct = 0.0
            realized_position_return_pct = 0.0
            giveback_position_return_pct = 0.0
            if (
                entry_index is not None
                and entry_index >= 0
                and exit_index >= entry_index
                and entry_price > 0
            ):
                trade_slice = data.iloc[entry_index : exit_index + 1]
                trade_high = float(pd.to_numeric(trade_slice.get("high"), errors="coerce").max())
                trade_low = float(pd.to_numeric(trade_slice.get("low"), errors="coerce").min())
                if position == 1:
                    peak_unrealized_return_pct = (trade_high - entry_price) / entry_price
                    worst_unrealized_return_pct = (trade_low - entry_price) / entry_price
                    realized_price_return_pct = (exit_fill - entry_price) / entry_price
                    peak_unrealized_pnl = (trade_high - entry_price) * close_units
                else:
                    peak_unrealized_return_pct = (entry_price - trade_low) / entry_price
                    worst_unrealized_return_pct = (entry_price - trade_high) / entry_price
                    realized_price_return_pct = (entry_price - exit_fill) / entry_price
                    peak_unrealized_pnl = (entry_price - trade_low) * close_units
                peak_unrealized_return_pct = float(max(0.0, peak_unrealized_return_pct))
                giveback_from_peak_pct = float(max(0.0, peak_unrealized_return_pct - realized_price_return_pct))
                peak_unrealized_pnl = float(max(0.0, peak_unrealized_pnl))
                giveback_from_peak_pnl = float(max(0.0, peak_unrealized_pnl - gross_pnl))
                initial_position_notional = abs(entry_price * close_units)
                if initial_position_notional > 0:
                    peak_unrealized_position_return_pct = float(peak_unrealized_pnl / initial_position_notional)
                    realized_position_return_pct = float(gross_pnl / initial_position_notional)
                    giveback_position_return_pct = float(giveback_from_peak_pnl / initial_position_notional)
                if peak_unrealized_return_pct > 0:
                    capture_ratio_vs_peak = float(realized_price_return_pct / peak_unrealized_return_pct)

            trades.append(
                Trade(
                    side="long" if position == 1 else "short",
                    entry_time=entry_time,
                    exit_time=exit_time,
                    entry_price=entry_price,
                    exit_price=exit_fill,
                    units=close_units,
                    pnl=pnl,
                    return_pct=return_pct,
                    holding_bars=holding_bars,
                    entry_signal=open_entry_signal,
                    exit_signal=exit_signal,
                    signal_intent_flat_time=signal_intent_flat_time_value,
                    gross_pnl=gross_pnl,
                    entry_fee=allocated_entry_fee,
                    exit_fee=exit_fee,
                    financing_cost=allocated_financing,
                    capital_base=capital_base,
                    peak_unrealized_return_pct=peak_unrealized_return_pct,
                    worst_unrealized_return_pct=worst_unrealized_return_pct,
                    giveback_from_peak_pct=giveback_from_peak_pct,
                    capture_ratio_vs_peak=capture_ratio_vs_peak,
                    realized_price_return_pct=realized_price_return_pct,
                    peak_unrealized_pnl=peak_unrealized_pnl,
                    giveback_from_peak_pnl=giveback_from_peak_pnl,
                    peak_unrealized_position_return_pct=peak_unrealized_position_return_pct,
                    realized_position_return_pct=realized_position_return_pct,
                    giveback_position_return_pct=giveback_position_return_pct,
                )
            )

            open_entry_fee_balance -= allocated_entry_fee
            open_financing_balance -= allocated_financing
            if abs(open_entry_fee_balance) <= 1e-12:
                open_entry_fee_balance = 0.0
            if abs(open_financing_balance) <= 1e-12:
                open_financing_balance = 0.0

        for i in range(1, len(data)):
            ts = data.index[i]
            bar = data.iloc[i]
            prev_close = float(data["close"].iloc[i - 1])
            signal_index = i if execute_on_signal_bar else i - 1
            signal_time = data.index[signal_index]
            signal_value = float(signals.iloc[signal_index])
            contracts_signal_value = None
            if signal_contracts is not None:
                contracts_signal_raw = float(signal_contracts.iloc[signal_index])
                if np.isfinite(contracts_signal_raw):
                    contracts_signal_value = contracts_signal_raw
            signal_side = int(np.sign(signal_value))
            if signal_side == 0:
                # Explicit flat signal intent must dominate any stale/non-zero
                # contract magnitude artifacts so the engine can fully flatten.
                desired_position = 0
            elif contracts_signal_value is not None and abs(contracts_signal_value) > 1e-12:
                # Contract direction is the authoritative intent when present.
                # This keeps execution aligned with contract-based strategies
                # even if the coarse signal direction is stale.
                desired_position = int(np.sign(contracts_signal_value))
            else:
                desired_position = signal_side
            entry_signal_label = self._infer_entry_signal(strategy, signal_index=signal_index, desired_position=desired_position)
            exit_signal_label = self._infer_exit_signal(
                strategy,
                signal_index=signal_index,
                prior_position=position,
                desired_position=desired_position,
                exit_reason_series=signal_exit_reasons,
                first_wiseman_reversal_series=signal_first_wiseman_reversal_side,
            )
            signal_intent_flat_time = self._signal_intent_flat_timestamp(exit_signal_label, signal_time)
            reversal_side_marker_now = (
                int(signal_first_wiseman_reversal_side.iloc[signal_index])
                if signal_first_wiseman_reversal_side is not None
                else 0
            )
            desired_contracts = 0.0
            if desired_position != 0:
                if signal_contracts is not None:
                    desired_contracts = abs(float(signal_contracts.iloc[signal_index]))
                else:
                    desired_contracts = abs(signal_value)
                desired_contracts = desired_contracts if desired_contracts > 0 else 1.0

            signal_fill = None
            has_explicit_signal_fill_series = signal_fill_prices is not None
            if signal_fill_prices is not None:
                signal_fill_raw = signal_fill_prices.iloc[signal_index]
                if pd.notna(signal_fill_raw):
                    signal_fill = float(signal_fill_raw)
            explicit_exit_reason_now = ""
            if signal_exit_reasons is not None:
                exit_reason_raw = signal_exit_reasons.iloc[signal_index]
                if pd.notna(exit_reason_raw):
                    explicit_exit_reason_now = str(exit_reason_raw).strip()
            has_explicit_exit_reason = explicit_exit_reason_now != ""
            first_wiseman_fill_price = None
            if signal_fill_prices_first is not None:
                first_wiseman_fill_raw = signal_fill_prices_first.iloc[signal_index]
                if pd.notna(first_wiseman_fill_raw):
                    first_wiseman_fill_price = float(first_wiseman_fill_raw)
            first_wiseman_setup_side_now = (
                int(signal_first_wiseman_setup_side.iloc[signal_index])
                if signal_first_wiseman_setup_side is not None
                else 0
            )
            if (
                position != 0
                and desired_position == position
                and first_wiseman_fill_price is not None
                and np.isfinite(first_wiseman_fill_price)
                and first_wiseman_setup_side_now != 0
                and first_wiseman_setup_side_now != position
            ):
                desired_position = first_wiseman_setup_side_now
            has_pending_alternate_flip_fill = bool(
                position != 0
                and desired_position != position
                and first_wiseman_fill_price is not None
                and np.isfinite(first_wiseman_fill_price)
            )
            if (
                position != 0
                and desired_position != position
                and desired_position == 0
                and has_explicit_signal_fill_series
                and signal_fill is None
                and not has_explicit_exit_reason
                and not has_pending_alternate_flip_fill
            ):
                # Hold until a concrete execution hint arrives. This prevents
                # phantom closes/flips from transient flat intent with no fill.
                desired_position = position
                desired_contracts = abs(float(signal_contracts.iloc[signal_index])) if signal_contracts is not None else 1.0
                desired_contracts = desired_contracts if desired_contracts > 0 else 1.0
            signal_stop_loss = None
            if signal_stop_loss_prices is not None:
                signal_stop_loss_raw = signal_stop_loss_prices.iloc[signal_index]
                if pd.notna(signal_stop_loss_raw):
                    signal_stop_loss = float(signal_stop_loss_raw)
            if (
                position != 0
                and desired_position == position
                and signal_stop_loss is not None
                and np.isfinite(signal_stop_loss)
            ):
                active_stop_loss_price = signal_stop_loss
            sizing_stop_loss = signal_stop_loss
            if (
                (sizing_stop_loss is None or not np.isfinite(sizing_stop_loss))
                and desired_position != 0
                and position != 0
                and desired_position == position
                and active_stop_loss_price is not None
                and np.isfinite(active_stop_loss_price)
            ):
                sizing_stop_loss = active_stop_loss_price

            prev_signal_index = max(0, signal_index - 1)
            prev_signal_value = float(signals.iloc[prev_signal_index])
            prev_contracts_signal_value = None
            if signal_contracts is not None:
                prev_contracts_signal_raw = float(signal_contracts.iloc[prev_signal_index])
                if np.isfinite(prev_contracts_signal_raw):
                    prev_contracts_signal_value = prev_contracts_signal_raw
            if prev_signal_value == 0:
                prev_position_intent = 0
            elif prev_contracts_signal_value is not None and abs(prev_contracts_signal_value) > 1e-12:
                prev_position_intent = int(np.sign(prev_contracts_signal_value))
            else:
                prev_position_intent = int(np.sign(prev_signal_value))
            if prev_position_intent != 0:
                if signal_contracts is not None:
                    prev_contracts = abs(float(signal_contracts.iloc[prev_signal_index]))
                else:
                    prev_contracts = abs(prev_signal_value)
                prev_contracts = prev_contracts if prev_contracts > 0 else 1.0
            else:
                prev_contracts = 0.0
            contract_intent_changed = (
                desired_position != 0
                and desired_position == prev_position_intent
                and abs(desired_contracts - prev_contracts) > 1e-12
            )
            increasing_contract_intent = contract_intent_changed and desired_contracts > prev_contracts + 1e-12
            decreasing_contract_intent = contract_intent_changed and desired_contracts < prev_contracts - 1e-12

            target_units = units
            target_sizing_context: dict[str, float | str | None] = {}
            if desired_position != 0:
                mode = self.config.trade_size_mode
                if (
                    position != 0
                    and desired_position == position
                    and (
                        not contract_intent_changed
                        or (has_explicit_signal_fill_series and signal_fill is None)
                    )
                ):
                    target_units = units
                else:
                    sizing_fill = signal_fill if signal_fill is not None else float(bar["open"])
                    if (
                        mode == "stop_loss_scaled"
                        and (sizing_stop_loss is None or not np.isfinite(sizing_stop_loss))
                    ):
                        target_units = 0.0
                    else:
                        target_units = self._resolve_units(
                            capital=capital,
                            fill_price=sizing_fill,
                            stop_loss_price=sizing_stop_loss,
                            bar_index=i,
                            closes=data["close"],
                            periods_per_year=periods_per_year,
                        ) * desired_contracts
                        target_sizing_context = dict(self._last_sizing_context)
                    if self.config.max_leverage is not None and self.config.max_leverage > 0 and sizing_fill > 0:
                        max_units = (capital * self.config.max_leverage) / sizing_fill
                        target_units = min(target_units, max_units)
                    if self.config.max_position_size is not None and self.config.max_position_size > 0 and sizing_fill > 0:
                        max_units_by_notional = self.config.max_position_size / sizing_fill
                        target_units = min(target_units, max_units_by_notional)
                    if position != 0 and desired_position == position and contract_intent_changed:
                        if increasing_contract_intent and target_units < units:
                            target_units = units
                        elif decreasing_contract_intent and target_units > units:
                            target_units = units
            else:
                target_units = 0.0

            carried_position = position
            carried_units = units
            financing_cost = self._financing_cost(
                units=carried_units,
                price=prev_close,
                position=carried_position,
                periods_per_year=periods_per_year,
            )
            total_financing_paid += financing_cost
            if carried_position != 0 and carried_units > 0:
                open_financing_balance += financing_cost
            capital -= financing_cost
            self._ensure_finite(capital, f"capital became non-finite after financing at bar index {i}")

            intrabar_bar_events = signal_intrabar_events.get(signal_index, [])
            has_explicit_intrabar_execution = bool(
                isinstance(intrabar_bar_events, list)
                and any(
                    isinstance(raw_event, dict)
                    and str(raw_event.get("event_type", "")).strip().lower() in {"entry", "exit"}
                    for raw_event in intrabar_bar_events
                )
            )

            stop_out_triggered = False
            strategy_stop_closed_side = 0
            skip_strategy_stop_execution = False
            pending_signal_exit_fill: float | None = None
            if position != 0 and desired_position != position:
                if signal_fill is None and has_pending_alternate_flip_fill:
                    pending_signal_exit_fill = first_wiseman_fill_price
                else:
                    pending_signal_exit_fill = self._resolve_signal_fill(
                        bar=bar,
                        prev_close=prev_close,
                        side=-position,
                        signal_fill=signal_fill,
                    )
            if position != 0:
                bar_open = float(bar["open"])
                gap_liquidation = liquidation_price is not None and (
                    self._price_le(bar_open, liquidation_price) if position > 0 else self._price_ge(bar_open, liquidation_price)
                )

                max_loss_fill: float | None = None
                max_loss_price: float | None = None
                cutoff_fill: float | None = None
                cutoff_price: float | None = None
                liquidation_fill: float | None = None

                if gap_liquidation:
                    liquidation_fill = self._resolve_liquidation_fill(
                        bar=bar,
                        position=position,
                        liquidation_price=liquidation_price,
                    )
                    selected_event = "liquidation" if liquidation_fill is not None else None
                elif has_explicit_intrabar_execution:
                    selected_event = None
                else:
                    max_loss_fill = self._resolve_max_loss_exit_fill(
                        bar=bar,
                        position=position,
                        entry_price=entry_price,
                        units=units,
                        capital_base=entry_capital,
                    )
                    effective_max_loss = self._effective_max_loss(capital_base=entry_capital)
                    if max_loss_fill is not None and effective_max_loss is not None and units > 0:
                        max_loss_price = entry_price - (effective_max_loss / units) if position > 0 else entry_price + (effective_max_loss / units)

                    if equity_cutoff is not None:
                        cutoff_fill, cutoff_price = self._resolve_equity_cutoff_exit_fill(
                            bar=bar,
                            position=position,
                            entry_price=entry_price,
                            units=units,
                            capital=capital,
                            equity_cutoff=equity_cutoff,
                        )

                    liquidation_fill = self._resolve_liquidation_fill(
                        bar=bar,
                        position=position,
                        liquidation_price=liquidation_price,
                    )

                    risk_candidates: list[tuple[str, float]] = []
                    if max_loss_fill is not None and max_loss_price is not None:
                        risk_candidates.append(("stop_out", max_loss_price))
                    if cutoff_fill is not None and cutoff_price is not None:
                        risk_candidates.append(("equity_cutoff", cutoff_price))
                    if liquidation_fill is not None and liquidation_price is not None:
                        risk_candidates.append(("liquidation", liquidation_price))

                    ordering_candidates: list[tuple[str, float, int]] = []
                    if pending_signal_exit_fill is not None and self._is_fill_reached_this_bar(
                        bar=bar,
                        position=position,
                        fill_price=pending_signal_exit_fill,
                    ):
                        ordering_candidates.append(("signal_fill", pending_signal_exit_fill, 0))
                    if (
                        not has_explicit_intrabar_execution
                        and active_stop_loss_price is not None
                        and np.isfinite(active_stop_loss_price)
                    ):
                        if position > 0:
                            stop_hit_pre = bool(
                                self._price_le(float(bar["low"]), active_stop_loss_price)
                                and (
                                    self._price_ge(prev_close, active_stop_loss_price)
                                    or self._price_ge(float(bar["high"]), active_stop_loss_price)
                                )
                            )
                        else:
                            stop_hit_pre = bool(
                                self._price_ge(float(bar["high"]), active_stop_loss_price)
                                and (
                                    self._price_le(prev_close, active_stop_loss_price)
                                    or self._price_le(float(bar["low"]), active_stop_loss_price)
                                )
                            )
                        if stop_hit_pre:
                            ordering_candidates.append(("strategy_stop", float(active_stop_loss_price), 1))
                    for risk_name, risk_price in risk_candidates:
                        ordering_candidates.append((risk_name, float(risk_price), 2))

                    adverse_candidates = [
                        item
                        for item in ordering_candidates
                        if (
                            self._price_le(item[1], bar_open)
                            if position > 0
                            else self._price_ge(item[1], bar_open)
                        )
                    ]
                    first_event = (
                        min(
                            adverse_candidates,
                            key=lambda item: (abs(item[1] - bar_open), item[2]),
                        )
                        if adverse_candidates
                        else None
                    )

                    if first_event is not None and first_event[0] == "signal_fill":
                        selected_event = None
                        skip_strategy_stop_execution = True
                    elif first_event is not None and first_event[0] == "strategy_stop":
                        selected_event = None
                    elif first_event is not None:
                        selected_event = first_event[0]
                        skip_strategy_stop_execution = True
                    elif risk_candidates:
                        if position > 0:
                            selected_event = max(risk_candidates, key=lambda item: item[1])[0]
                        else:
                            selected_event = min(risk_candidates, key=lambda item: item[1])[0]
                    else:
                        selected_event = None

                if selected_event is not None:
                    exit_fill = {
                        "stop_out": max_loss_fill,
                        "equity_cutoff": cutoff_fill,
                        "liquidation": liquidation_fill,
                    }[selected_event]
                    if exit_fill is not None:
                        closing_units = units
                        record_closed_trade(
                            close_units=closing_units,
                            exit_time=ts,
                            exit_fill=exit_fill,
                            holding_bars=i - entry_index if entry_index is not None else 0,
                            exit_signal=self._finalize_exit_signal_label(self._engine_event_exit_reason(selected_event), open_entry_signal),
                            exit_index=i,
                        )
                        if selected_event == "liquidation":
                            capital = max(capital, 0.0)
                        self._ensure_finite(capital, f"capital became non-finite after {selected_event} at bar index {i}")
                        execution_events.append(
                            ExecutionEvent(
                                event_type="equity_cutoff_exit" if selected_event == "equity_cutoff" else selected_event,
                                time=ts,
                                side="sell" if position == 1 else "buy",
                                price=exit_fill,
                                units=closing_units,
                                strategy_reason=self._finalize_exit_signal_label(self._engine_event_exit_reason(selected_event), open_entry_signal),
                            )
                        )
                        if selected_event == "equity_cutoff":
                            execution_events.append(
                                ExecutionEvent(
                                    event_type="equity_cutoff",
                                    time=ts,
                                    side="flat",
                                    price=exit_fill,
                                    units=0.0,
                                )
                            )
                            cutoff_index = i

                        position = 0
                        units = 0.0
                        liquidation_price = None
                        active_stop_loss_price = None
                        entry_index = None
                        open_entry_signal = "Unknown"
                        stop_out_triggered = True

            # Strategy-controlled stop execution:
            # if the strategy publishes a stop via `signal_stop_loss_prices`, the engine
            # can execute that stop when price touches it intrabar. This models realistic
            # stop execution while keeping stop placement under strategy control.
            if (
                not stop_out_triggered
                and not skip_strategy_stop_execution
                and not has_explicit_intrabar_execution
                and position != 0
                and active_stop_loss_price is not None
                and np.isfinite(active_stop_loss_price)
            ):
                bar_low = float(bar["low"])
                bar_high = float(bar["high"])
                if position > 0:
                    stop_hit = bool(
                        self._price_le(bar_low, active_stop_loss_price)
                        and (
                            self._price_ge(prev_close, active_stop_loss_price)
                            or self._price_ge(bar_high, active_stop_loss_price)
                        )
                    )
                else:
                    stop_hit = bool(
                        self._price_ge(bar_high, active_stop_loss_price)
                        and (
                            self._price_le(prev_close, active_stop_loss_price)
                            or self._price_le(bar_low, active_stop_loss_price)
                        )
                    )
                if stop_hit:
                    if (
                        pending_signal_exit_fill is not None
                        and self._fill_occurs_before_adverse_risk(
                            bar_open=float(bar["open"]),
                            position=position,
                            fill_price=pending_signal_exit_fill,
                            risk_prices=[active_stop_loss_price],
                        )
                    ):
                        stop_hit = False
                if stop_hit:
                    strategy_stop_closed_side = position
                    stop_fill = min(float(bar["open"]), active_stop_loss_price) if position > 0 else max(float(bar["open"]), active_stop_loss_price)
                    closing_units = units
                    resolved_stop_exit_label = self._finalize_exit_signal_label("Strategy Stop Loss", open_entry_signal)
                    record_closed_trade(
                        close_units=closing_units,
                        exit_time=ts,
                        exit_fill=stop_fill,
                        holding_bars=i - entry_index if entry_index is not None else 0,
                        exit_signal=resolved_stop_exit_label,
                        exit_index=i,
                    )
                    execution_events.append(
                        ExecutionEvent(
                            event_type="exit",
                            time=ts,
                            side="sell" if position == 1 else "buy",
                            price=stop_fill,
                            units=closing_units,
                            strategy_reason=resolved_stop_exit_label,
                        )
                    )
                    position = 0
                    units = 0.0
                    liquidation_price = None
                    active_stop_loss_price = None
                    entry_index = None
                    open_entry_fee_balance = 0.0
                    open_financing_balance = 0.0
                    open_entry_signal = "Unknown"
                    stop_out_triggered = True

            if isinstance(intrabar_bar_events, list) and intrabar_bar_events:
                processed_intrabar_event = False
                seen_intrabar_event_keys: set[tuple[str, int, float, float, str, float | None]] = set()
                consumed_entry_reasons: set[str] = set()
                first_non_reversal_entry_side: int | None = None
                intrabar_entry_side_opened: int | None = None
                intrabar_exit_seen = False
                intrabar_reversal_executed = False
                ordered_intrabar_events: list[dict[str, float | int | str]] = []
                sortable_intrabar_events: list[tuple[float, int, dict[str, float | int | str]]] = []
                for event_order, candidate_event in enumerate(intrabar_bar_events):
                    if not isinstance(candidate_event, dict):
                        continue
                    candidate_price = candidate_event.get("price")
                    if candidate_price is None:
                        continue
                    candidate_price_float = float(candidate_price)
                    if not np.isfinite(candidate_price_float):
                        continue
                    sortable_intrabar_events.append(
                        (
                            abs(candidate_price_float - float(bar["open"])),
                            event_order,
                            candidate_event,
                        )
                    )
                if sortable_intrabar_events:
                    ordered_intrabar_events = [
                        event
                        for _, _, event in sorted(
                            sortable_intrabar_events,
                            key=lambda item: (item[0], item[1]),
                        )
                    ]
                for event_idx, raw_event in enumerate(ordered_intrabar_events):
                    if not isinstance(raw_event, dict):
                        continue
                    event_type = str(raw_event.get("event_type", "")).strip().lower()
                    event_side = int(raw_event.get("side", 0))
                    event_price_raw = raw_event.get("price")
                    event_reason = str(raw_event.get("reason", "")).strip() or None
                    is_1w_reason = bool(event_reason and "1W" in event_reason)
                    if event_price_raw is None:
                        continue
                    event_price = float(event_price_raw)
                    if not np.isfinite(event_price):
                        continue
                    event_contracts = abs(float(raw_event.get("contracts", 1.0)))
                    event_stop_loss = raw_event.get("stop_loss_price")
                    event_stop_loss_price = None
                    if event_stop_loss is not None:
                        event_stop_loss_float = float(event_stop_loss)
                        if np.isfinite(event_stop_loss_float):
                            event_stop_loss_price = event_stop_loss_float
                    event_key = (
                        event_type,
                        event_side,
                        float(event_price),
                        float(event_contracts),
                        event_reason or "",
                        event_stop_loss_price,
                    )
                    if event_key in seen_intrabar_event_keys:
                        continue
                    seen_intrabar_event_keys.add(event_key)
                    if event_type == "exit":
                        if position == 0 or units <= 0:
                            continue
                        if event_reason and intrabar_reversal_executed:
                            reason_lower = event_reason.lower()
                            if "stop" in reason_lower:
                                if position > 0 and "bearish" in reason_lower:
                                    continue
                                if position < 0 and "bullish" in reason_lower:
                                    continue
                        closing_units = units
                        normalized_event_reason = self._normalize_intrabar_stop_label(event_reason, position)
                        resolved_exit_label = self._finalize_exit_signal_label(normalized_event_reason, open_entry_signal)
                        record_closed_trade(
                            close_units=closing_units,
                            exit_time=ts,
                            exit_fill=event_price,
                            holding_bars=i - entry_index if entry_index is not None else 0,
                            exit_signal=resolved_exit_label,
                            exit_index=i,
                        )
                        execution_events.append(
                            ExecutionEvent(
                                event_type="exit",
                                time=ts,
                                side="sell" if position == 1 else "buy",
                                price=event_price,
                                units=closing_units,
                                strategy_reason=resolved_exit_label,
                            )
                        )
                        position = 0
                        units = 0.0
                        liquidation_price = None
                        active_stop_loss_price = None
                        entry_index = None
                        open_entry_fee_balance = 0.0
                        open_financing_balance = 0.0
                        open_entry_signal = "Unknown"
                        intrabar_entry_side_opened = None
                        intrabar_exit_seen = True
                        processed_intrabar_event = True
                        continue
                    if event_type != "entry":
                        continue
                    if event_side == 0:
                        continue
                    if event_reason and "Add-on Fractal" in event_reason:
                        # Add-on fractals are strictly pyramiding events and must never
                        # open a fresh position or reverse direction intrabar.
                        if position == 0 or event_side != position:
                            continue
                    if carried_position == 0 and desired_position == 0 and intrabar_exit_seen:
                        allow_post_exit_reversal = bool(
                            event_reason and "1W-R" in event_reason and intrabar_reversal_executed
                        )
                        if not allow_post_exit_reversal:
                            continue
                    if carried_position != 0 and desired_position == 0:
                        # Flatten-intent guardrail:
                        # - never allow any entry after an intrabar exit has already happened,
                        #   except explicit same-bar 1W-R re-entry requested by strategy
                        # - before the first intrabar exit, allow same-side adds and only allow
                        #   opposite-side entries when they are explicit 1W reversal events
                        allow_post_exit_reversal = bool(
                            event_reason and "1W-R" in event_reason and intrabar_reversal_executed
                        )
                        if intrabar_exit_seen:
                            if not allow_post_exit_reversal:
                                continue
                            if position != 0:
                                continue
                        else:
                            if position == 0:
                                continue
                            allow_pre_exit_reversal = bool(
                                event_reason
                                and "1W" in event_reason
                                and event_side == -position
                            )
                            if event_side != carried_position and not allow_pre_exit_reversal:
                                continue
                    can_reverse_open_position = position != 0 and event_side != position
                    if (
                        intrabar_entry_side_opened is not None
                        and event_side != intrabar_entry_side_opened
                        and not intrabar_exit_seen
                        and not can_reverse_open_position
                    ):
                        continue
                    if event_reason and event_reason in consumed_entry_reasons:
                        continue
                    if desired_position != 0 and event_side != desired_position:
                        has_follow_on_stop = any(
                            isinstance(next_event, dict)
                            and str(next_event.get("event_type", "")).strip().lower() == "exit"
                            and "stop" in str(next_event.get("reason", "")).strip().lower()
                            for next_event in ordered_intrabar_events[event_idx + 1 :]
                        )
                        allow_same_bar_reversal_override = bool(
                            carried_position != 0
                            and event_side == -carried_position
                            and event_reason
                            and "1W" in event_reason
                            and not processed_intrabar_event
                            and not intrabar_exit_seen
                            and has_follow_on_stop
                        )
                        if not allow_same_bar_reversal_override:
                            continue
                    if (
                        reversal_side_marker_now == 0
                        and first_non_reversal_entry_side is not None
                        and event_side != first_non_reversal_entry_side
                        and not intrabar_exit_seen
                    ):
                        continue
                    if position != 0:
                        if event_side == position:
                            if event_contracts <= 0:
                                continue
                            if self.config.trade_size_mode == "stop_loss_scaled" and (
                                event_stop_loss_price is None or not np.isfinite(event_stop_loss_price)
                            ):
                                continue
                            add_units = self._resolve_units(
                                capital=capital,
                                fill_price=event_price,
                                stop_loss_price=event_stop_loss_price,
                                bar_index=i,
                                closes=data["close"],
                                periods_per_year=periods_per_year,
                            ) * event_contracts
                            sizing_context = dict(self._last_sizing_context)
                            if self.config.max_leverage is not None and self.config.max_leverage > 0 and event_price > 0:
                                max_units = (capital * self.config.max_leverage) / event_price
                                add_units = min(add_units, max_units)
                            if self.config.max_position_size is not None and self.config.max_position_size > 0 and event_price > 0:
                                max_units_by_notional = self.config.max_position_size / event_price
                                add_units = min(add_units, max_units_by_notional)
                            if add_units <= 0:
                                continue
                            add_fee = abs(event_price * add_units) * self.config.fee_rate
                            total_fees_paid += add_fee
                            capital -= add_fee
                            self._ensure_finite(capital, f"capital became non-finite after intrabar add fee at bar index {i}")
                            if units + add_units > 0:
                                entry_price = ((entry_price * units) + (event_price * add_units)) / (units + add_units)
                            units += add_units
                            entry_capital = capital
                            entry_notional = abs(entry_price * units)
                            open_entry_fee_balance += add_fee
                            liquidation_price = self._compute_liquidation_price(
                                side=position,
                                entry_price=entry_price,
                                entry_capital=entry_capital,
                                entry_notional=entry_notional,
                            )
                            if event_stop_loss_price is not None and np.isfinite(event_stop_loss_price):
                                active_stop_loss_price = event_stop_loss_price
                            execution_events.append(
                                ExecutionEvent(
                                    event_type="add",
                                    time=ts,
                                    side="buy" if position == 1 else "sell",
                                    price=event_price,
                                    units=add_units,
                                    strategy_reason=event_reason,
                                    sizing_mode=sizing_context.get("mode"),
                                    capital_snapshot=sizing_context.get("capital_snapshot"),
                                    base_notional=sizing_context.get("base_notional"),
                                    volatility_scale=sizing_context.get("volatility_scale"),
                                    realized_vol_annual=sizing_context.get("realized_vol_annual"),
                                    scaled_notional=sizing_context.get("scaled_notional"),
                                )
                            )
                            if event_reason:
                                consumed_entry_reasons.add(event_reason)
                            intrabar_entry_side_opened = position
                            processed_intrabar_event = True
                            continue
                        closing_units = units
                        reversal_exit_label = self._finalize_exit_signal_label(
                            f"Strategy Reversal to {event_reason}" if event_reason else "Strategy Reversal",
                            open_entry_signal,
                        )
                        record_closed_trade(
                            close_units=closing_units,
                            exit_time=ts,
                            exit_fill=event_price,
                            holding_bars=i - entry_index if entry_index is not None else 0,
                            exit_signal=reversal_exit_label,
                            exit_index=i,
                        )
                        execution_events.append(
                            ExecutionEvent(
                                event_type="exit",
                                time=ts,
                                side="sell" if position == 1 else "buy",
                                price=event_price,
                                units=closing_units,
                                strategy_reason=reversal_exit_label,
                            )
                        )
                        position = 0
                        units = 0.0
                        liquidation_price = None
                        active_stop_loss_price = None
                        entry_index = None
                        open_entry_fee_balance = 0.0
                        open_financing_balance = 0.0
                        open_entry_signal = "Unknown"
                        intrabar_entry_side_opened = None
                        intrabar_exit_seen = True
                        processed_intrabar_event = True
                        intrabar_reversal_executed = True
                    if event_contracts <= 0:
                        continue
                    if self.config.trade_size_mode == "stop_loss_scaled" and (
                        event_stop_loss_price is None or not np.isfinite(event_stop_loss_price)
                    ):
                        continue
                    entry_units = self._resolve_units(
                        capital=capital,
                        fill_price=event_price,
                        stop_loss_price=event_stop_loss_price,
                        bar_index=i,
                        closes=data["close"],
                        periods_per_year=periods_per_year,
                    ) * event_contracts
                    sizing_context = dict(self._last_sizing_context)
                    if self.config.max_leverage is not None and self.config.max_leverage > 0 and event_price > 0:
                        max_units = (capital * self.config.max_leverage) / event_price
                        entry_units = min(entry_units, max_units)
                    if self.config.max_position_size is not None and self.config.max_position_size > 0 and event_price > 0:
                        max_units_by_notional = self.config.max_position_size / event_price
                        entry_units = min(entry_units, max_units_by_notional)
                    if entry_units <= 0:
                        continue
                    entry_fee = abs(event_price * entry_units) * self.config.fee_rate
                    total_fees_paid += entry_fee
                    capital -= entry_fee
                    self._ensure_finite(capital, f"capital became non-finite after intrabar entry fee at bar index {i}")
                    units = entry_units
                    entry_price = event_price
                    entry_time = ts
                    entry_index = i
                    entry_capital = capital
                    entry_notional = abs(event_price * units)
                    open_entry_fee_balance = entry_fee
                    open_financing_balance = 0.0
                    position = 1 if event_side > 0 else -1
                    active_stop_loss_price = None
                    open_entry_signal = event_reason or self._infer_entry_signal(
                        strategy,
                        signal_index=signal_index,
                        desired_position=position,
                    )
                    liquidation_price = self._compute_liquidation_price(
                        side=position,
                        entry_price=entry_price,
                        entry_capital=entry_capital,
                        entry_notional=entry_notional,
                    )
                    execution_events.append(
                        ExecutionEvent(
                            event_type="entry",
                            time=ts,
                            side="buy" if position == 1 else "sell",
                            price=event_price,
                            units=units,
                            strategy_reason=open_entry_signal,
                            sizing_mode=sizing_context.get("mode"),
                            capital_snapshot=sizing_context.get("capital_snapshot"),
                            base_notional=sizing_context.get("base_notional"),
                            volatility_scale=sizing_context.get("volatility_scale"),
                            realized_vol_annual=sizing_context.get("realized_vol_annual"),
                            scaled_notional=sizing_context.get("scaled_notional"),
                        )
                    )
                    if reversal_side_marker_now == 0 and first_non_reversal_entry_side is None:
                        first_non_reversal_entry_side = position
                    intrabar_entry_side_opened = position
                    if event_reason:
                        consumed_entry_reasons.add(event_reason)
                    processed_intrabar_event = True
                if processed_intrabar_event:
                    if position != 0:
                        positions.iloc[i] = position
                        equity_values.append(capital + ((float(bar["close"]) - entry_price) * units * position))
                    else:
                        positions.iloc[i] = 0
                        equity_values.append(capital)
                    continue

            if desired_position != position:
                # Exit current position first
                if position != 0:
                    exit_fill = self._resolve_signal_fill(bar=bar, prev_close=prev_close, side=-position, signal_fill=signal_fill)
                    if exit_fill is not None:
                        closing_units = units
                        record_closed_trade(
                            close_units=closing_units,
                            exit_time=ts,
                            exit_fill=exit_fill,
                            holding_bars=i - entry_index if entry_index is not None else 0,
                            exit_signal=self._finalize_exit_signal_label(exit_signal_label, open_entry_signal),
                            exit_index=i,
                            signal_intent_flat_time_value=signal_intent_flat_time,
                        )
                        execution_events.append(
                            ExecutionEvent(
                                event_type="exit",
                                time=ts,
                                side="sell" if position == 1 else "buy",
                                price=exit_fill,
                                units=closing_units,
                                strategy_reason=self._finalize_exit_signal_label(exit_signal_label, open_entry_signal),
                            )
                        )
                        position = 0
                        units = 0.0
                        liquidation_price = None
                        active_stop_loss_price = None
                        entry_index = None
                        open_entry_fee_balance = 0.0
                        open_financing_balance = 0.0
                        open_entry_signal = "Unknown"

                # Enter desired position
                allow_reentry_after_strategy_stop = (
                    stop_out_triggered
                    and strategy_stop_closed_side != 0
                    and desired_position == -strategy_stop_closed_side
                    and reversal_side_marker_now == desired_position
                )
                if (
                    desired_position != 0
                    and position == 0
                    and (not stop_out_triggered or allow_reentry_after_strategy_stop)
                ):
                    entry_fill = self._resolve_signal_fill(bar=bar, prev_close=prev_close, side=desired_position, signal_fill=signal_fill)
                    if entry_fill is not None:
                        units = target_units
                        if units > 0:
                            entry_fee = abs(entry_fill * units) * self.config.fee_rate
                            total_fees_paid += entry_fee
                            capital -= entry_fee
                            self._ensure_finite(capital, f"capital became non-finite after entry fee at bar index {i}")
                            entry_price = entry_fill
                            entry_time = ts
                            entry_index = i
                            entry_capital = capital
                            entry_notional = abs(entry_fill * units)
                            open_entry_fee_balance = entry_fee
                            open_financing_balance = 0.0
                            position = desired_position
                            active_stop_loss_price = sizing_stop_loss if sizing_stop_loss is not None and np.isfinite(sizing_stop_loss) else None
                            open_entry_signal = entry_signal_label
                            liquidation_price = self._compute_liquidation_price(
                                side=position,
                                entry_price=entry_price,
                                entry_capital=entry_capital,
                                entry_notional=entry_notional,
                            )
                            execution_events.append(
                                ExecutionEvent(
                                    event_type="entry",
                                    time=ts,
                                    side="buy" if desired_position == 1 else "sell",
                                    price=entry_fill,
                                    units=units,
                                    strategy_reason=entry_signal_label,
                                    sizing_mode=target_sizing_context.get("mode"),
                                    capital_snapshot=target_sizing_context.get("capital_snapshot"),
                                    base_notional=target_sizing_context.get("base_notional"),
                                    volatility_scale=target_sizing_context.get("volatility_scale"),
                                    realized_vol_annual=target_sizing_context.get("realized_vol_annual"),
                                    scaled_notional=target_sizing_context.get("scaled_notional"),
                                )
                            )
            elif position != 0 and target_units > 0 and contract_intent_changed and abs(target_units - units) > 1e-12:
                delta_units = target_units - units
                rebalance_side = position if delta_units > 0 else -position
                rebalance_fill = self._resolve_signal_fill(
                    bar=bar,
                    prev_close=prev_close,
                    side=rebalance_side,
                    signal_fill=signal_fill,
                )
                if rebalance_fill is not None:
                    if delta_units > 0:
                        add_units = delta_units
                        fee = abs(rebalance_fill * add_units) * self.config.fee_rate
                        total_fees_paid += fee
                        capital -= fee
                        self._ensure_finite(capital, f"capital became non-finite after add/rebalance fee at bar index {i}")
                        if units + add_units > 0:
                            entry_price = ((entry_price * units) + (rebalance_fill * add_units)) / (units + add_units)
                        units += add_units
                        if sizing_stop_loss is not None and np.isfinite(sizing_stop_loss):
                            active_stop_loss_price = sizing_stop_loss
                        entry_capital = capital
                        entry_notional = abs(entry_price * units)
                        open_entry_fee_balance += fee
                        liquidation_price = self._compute_liquidation_price(
                            side=position,
                            entry_price=entry_price,
                            entry_capital=entry_capital,
                            entry_notional=entry_notional,
                        )
                        execution_events.append(
                            ExecutionEvent(
                                event_type="add",
                                time=ts,
                                side="buy" if position == 1 else "sell",
                                price=rebalance_fill,
                                units=add_units,
                                sizing_mode=target_sizing_context.get("mode"),
                                capital_snapshot=target_sizing_context.get("capital_snapshot"),
                                base_notional=target_sizing_context.get("base_notional"),
                                volatility_scale=target_sizing_context.get("volatility_scale"),
                                realized_vol_annual=target_sizing_context.get("realized_vol_annual"),
                                scaled_notional=target_sizing_context.get("scaled_notional"),
                            )
                        )
                    else:
                        reduce_units = min(units, -delta_units)
                        record_closed_trade(
                            close_units=reduce_units,
                            exit_time=ts,
                            exit_fill=rebalance_fill,
                            holding_bars=i - entry_index if entry_index is not None else 0,
                            exit_signal=self._finalize_exit_signal_label(exit_signal_label, open_entry_signal),
                            exit_index=i,
                            signal_intent_flat_time_value=signal_intent_flat_time,
                        )
                        self._ensure_finite(capital, f"capital became non-finite after reduce/rebalance at bar index {i}")
                        units -= reduce_units
                        execution_events.append(
                            ExecutionEvent(
                                event_type="reduce",
                                time=ts,
                                side="sell" if position == 1 else "buy",
                                price=rebalance_fill,
                                units=reduce_units,
                            )
                        )
                        if units <= 1e-12:
                            position = 0
                            units = 0.0
                            liquidation_price = None
                            active_stop_loss_price = None
                            open_entry_fee_balance = 0.0
                            open_financing_balance = 0.0
                            open_entry_signal = "Unknown"
                        else:
                            entry_notional = abs(entry_price * units)
                            liquidation_price = self._compute_liquidation_price(
                                side=position,
                                entry_price=entry_price,
                                entry_capital=entry_capital,
                                entry_notional=entry_notional,
                            )

            mark_to_market = capital
            if position != 0:
                unrealized = (float(bar["close"]) - entry_price) * units * position
                # `capital` already includes all realized effects up to this bar
                # (fees, prior realized PnL, and cumulative financing debits).
                # Equity should therefore be cash plus open-position unrealized PnL.
                mark_to_market = capital + unrealized

            self._ensure_finite(mark_to_market, f"equity became non-finite at bar index {i}")

            if mark_to_market <= 0.0:
                mark_to_market = 0.0
                capital = 0.0
                position = 0
                units = 0.0
                active_stop_loss_price = None
                bankruptcy_index = i

            if cutoff_index is None and bankruptcy_index is None and equity_cutoff is not None and mark_to_market <= equity_cutoff:
                if position != 0:
                    exit_fill = self._apply_execution_adjustment(float(bar["close"]), -position)
                    closing_units = units
                    record_closed_trade(
                        close_units=closing_units,
                        exit_time=ts,
                        exit_fill=exit_fill,
                        holding_bars=i - entry_index if entry_index is not None else 0,
                        exit_signal=self._finalize_exit_signal_label(exit_signal_label, open_entry_signal),
                        exit_index=i,
                        signal_intent_flat_time_value=signal_intent_flat_time,
                    )
                    execution_events.append(
                        ExecutionEvent(
                            event_type="equity_cutoff_exit",
                            time=ts,
                            side="sell" if position == 1 else "buy",
                            price=exit_fill,
                            units=closing_units,
                        )
                    )
                    position = 0
                    units = 0.0
                    liquidation_price = None
                    active_stop_loss_price = None
                    entry_index = None
                    open_entry_fee_balance = 0.0
                    open_financing_balance = 0.0
                    open_entry_signal = "Unknown"
                    mark_to_market = capital

                execution_events.append(
                    ExecutionEvent(
                        event_type="equity_cutoff",
                        time=ts,
                        side="flat",
                        price=float(bar["close"]),
                        units=0.0,
                    )
                )
                cutoff_index = i

            equity_values.append(mark_to_market)
            positions.iloc[i] = position

            if bankruptcy_index is not None:
                break
            if cutoff_index is not None:
                break

        if bankruptcy_index is not None and bankruptcy_index < len(data) - 1:
            remaining_index = data.index[bankruptcy_index + 1 :]
            equity_values.extend([0.0] * len(remaining_index))
            positions.loc[remaining_index] = 0
        elif cutoff_index is not None and cutoff_index < len(data) - 1:
            remaining_index = data.index[cutoff_index + 1 :]
            equity_values.extend([capital] * len(remaining_index))
            positions.loc[remaining_index] = 0

        if position != 0 and self.config.close_open_position_on_last_bar:
            ts = data.index[-1]
            bar = data.iloc[-1]
            # Forced end-of-backtest flatten should execute at the observable bar close,
            # not via the configured entry order type mechanics.
            exit_fill = self._apply_execution_adjustment(float(bar["close"]), -position)
            closing_units = units
            record_closed_trade(
                close_units=closing_units,
                exit_time=ts,
                exit_fill=exit_fill,
                holding_bars=len(data) - 1 - entry_index if entry_index is not None else 0,
                exit_signal=self._finalize_exit_signal_label(self._engine_event_exit_reason("end_of_backtest"), open_entry_signal),
                exit_index=len(data) - 1,
            )
            self._ensure_finite(capital, "capital became non-finite during forced end-of-backtest flatten")
            execution_events.append(
                ExecutionEvent(
                    event_type="exit",
                    time=ts,
                    side="sell" if position == 1 else "buy",
                    price=exit_fill,
                    units=closing_units,
                    strategy_reason=self._finalize_exit_signal_label(self._engine_event_exit_reason("end_of_backtest"), open_entry_signal),
                )
            )
            equity_values[-1] = capital

        equity_curve = pd.Series(equity_values, index=data.index)
        returns = equity_curve.pct_change().fillna(0)
        stats = compute_performance_stats(
            equity_curve=equity_curve,
            returns=returns,
            trades=trades,
            periods_per_year=periods_per_year,
            positions=positions,
        )
        stats["slippage_rate"] = float(self.config.slippage_rate)

        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            positions=positions,
            trades=trades,
            stats=stats,
            data_quality=data_quality,
            execution_events=execution_events,
            total_fees_paid=total_fees_paid,
            total_financing_paid=total_financing_paid,
            total_profit_before_fees=total_profit_before_fees,
        )

    def _infer_entry_signal(self, strategy: Strategy, signal_index: int, desired_position: int) -> str:
        if desired_position == 0:
            return "Flat"
        if bool(getattr(strategy, "use_generic_trade_labels", False)):
            return "LE" if desired_position > 0 else "SE"
        inferred = self._infer_signal_label_for_side(strategy, signal_index=signal_index, side=desired_position, for_entry=True)
        if inferred is not None:
            return inferred
        return f"{'Bullish' if desired_position > 0 else 'Bearish'} 1W"

    def _infer_exit_signal(
        self,
        strategy: Strategy,
        signal_index: int,
        prior_position: int,
        desired_position: int,
        exit_reason_series: pd.Series | None = None,
        first_wiseman_reversal_series: pd.Series | None = None,
    ) -> str:
        if prior_position == 0:
            return "No Exit (No Open Position)"
        if bool(getattr(strategy, "use_generic_trade_labels", False)):
            return "LX" if prior_position > 0 else "SX"

        exit_reason = exit_reason_series if exit_reason_series is not None else getattr(strategy, "signal_exit_reason", None)
        if isinstance(exit_reason, pd.Series) and signal_index < len(exit_reason):
            reason_raw = exit_reason.iloc[signal_index]
            if pd.notna(reason_raw):
                reason = str(reason_raw).strip()
                if reason:
                    return self._normalize_exit_reason_label(reason=reason, prior_position=prior_position)

        reversal = (
            first_wiseman_reversal_series
            if first_wiseman_reversal_series is not None
            else getattr(strategy, "signal_first_wiseman_reversal_side", None)
        )
        if isinstance(reversal, pd.Series) and signal_index < len(reversal):
            reversal_side = int(reversal.iloc[signal_index])
            if reversal_side != 0 and reversal_side != prior_position:
                return f"Strategy Reversal to {'Bullish' if reversal_side > 0 else 'Bearish'} 1W"

        if desired_position == 0:
            prior_label = self._infer_signal_label_for_side(strategy, signal_index=signal_index, side=prior_position, for_entry=False)
            if prior_label is None:
                prior_label = f"{'Bullish' if prior_position > 0 else 'Bearish'} 1W"
            return f"Signal Intent Flat from {prior_label}"

        if desired_position != prior_position:
            target_label = self._infer_signal_label_for_side(strategy, signal_index=signal_index, side=desired_position, for_entry=False)
            if target_label is None:
                target_label = f"{'Bullish' if desired_position > 0 else 'Bearish'} 1W"
            return f"Signal Intent Flip to {target_label}"

        held_label = self._infer_signal_label_for_side(strategy, signal_index=signal_index, side=prior_position, for_entry=False)
        if held_label is None:
            held_label = f"{'Bullish' if prior_position > 0 else 'Bearish'} 1W"
        return f"Signal Intent Reduce from {held_label}"

    def _normalize_exit_reason_label(self, reason: str, prior_position: int) -> str:
        normalized = reason.strip()
        if normalized in {"Red Gator PP", "Red Gator"}:
            return "Strategy Profit Protection Red Gator"
        if normalized in {"Green Gator PP", "Green Gator"}:
            return "Strategy Profit Protection Green Gator"
        if normalized in {"Williams Zone PP", "Zone PP"}:
            return "Strategy Profit Protection Williams Zone"
        if normalized in {"Peak Drawdown PP", "Peak Drawdown"}:
            return "Strategy Profit Protection Peak Drawdown"
        if normalized == "1W Reversal Stop":
            return "Strategy Stop Loss Bullish 1W" if prior_position > 0 else "Strategy Stop Loss Bearish 1W"
        if normalized in {"Bullish 1W Reversal", "Bullish 1W-R"}:
            return "Strategy Reversal to Bullish 1W"
        if normalized in {"Bearish 1W Reversal", "Bearish 1W-R"}:
            return "Strategy Reversal to Bearish 1W"
        return f"Strategy Exit Reason: {normalized}"

    def _finalize_exit_signal_label(self, raw_exit_signal: str, entry_signal: str) -> str:
        normalized_exit = str(raw_exit_signal or "Unspecified Exit Reason")
        normalized_entry = str(entry_signal or "Unknown")
        if normalized_exit == "Strategy Stop Loss":
            if normalized_entry.startswith("Bullish 1W"):
                normalized_exit = "Strategy Stop Loss Bullish 1W"
            elif normalized_entry.startswith("Bearish 1W"):
                normalized_exit = "Strategy Stop Loss Bearish 1W"
        if normalized_exit in {"Strategy Stop Loss Bullish 1W", "Strategy Stop Loss Bearish 1W"} and normalized_entry in {"Bullish 1W-R", "Bearish 1W-R"}:
            return f"Strategy Stop Loss {normalized_entry}"
        if "Fractal" in normalized_entry:
            if normalized_exit == "Signal Intent Flat from Bullish 1W" and normalized_entry.startswith("Bullish"):
                return "Signal Intent Flat from Bullish Fractal"
            if normalized_exit == "Signal Intent Flat from Bearish 1W" and normalized_entry.startswith("Bearish"):
                return "Signal Intent Flat from Bearish Fractal"
            if normalized_exit == "Signal Intent Reduce from Bullish 1W" and normalized_entry.startswith("Bullish"):
                return "Signal Intent Reduce from Bullish Fractal"
            if normalized_exit == "Signal Intent Reduce from Bearish 1W" and normalized_entry.startswith("Bearish"):
                return "Signal Intent Reduce from Bearish Fractal"
        return normalized_exit

    def _normalize_intrabar_stop_label(self, reason: str | None, position: int) -> str | None:
        if reason is None:
            return None
        normalized = str(reason).strip()
        if normalized in {"Strategy Stop Loss Bullish 1W", "Strategy Stop Loss Bearish 1W", "1W Reversal Stop"}:
            return "Strategy Stop Loss Bullish 1W" if position > 0 else "Strategy Stop Loss Bearish 1W"
        return normalized or None

    def _signal_intent_flat_timestamp(self, exit_signal_label: str, signal_time: pd.Timestamp) -> pd.Timestamp | None:
        if str(exit_signal_label).startswith("Signal Intent Flat from "):
            return signal_time
        return None

    def _engine_event_exit_reason(self, event_name: str) -> str:
        mapping = {
            "stop_out": "Engine Risk Stop Out",
            "equity_cutoff": "Engine Risk Equity Cutoff",
            "liquidation": "Engine Risk Liquidation",
            "end_of_backtest": "Engine End of Backtest",
        }
        return mapping.get(str(event_name), f"Engine Event: {event_name}")

    def _infer_signal_label_for_side(self, strategy: Strategy, signal_index: int, side: int, for_entry: bool) -> str | None:
        if side == 0:
            return None

        direction = "Bullish" if side > 0 else "Bearish"

        def _series(name: str) -> pd.Series | None:
            series = getattr(strategy, name, None)
            if isinstance(series, pd.Series) and signal_index < len(series):
                return series
            return None

        first_rev = _series("signal_first_wiseman_reversal_side")
        if first_rev is not None and int(first_rev.iloc[signal_index]) == side:
            return f"{direction} 1W-R" if for_entry else f"{direction} 1W"

        second_side = _series("signal_second_wiseman_fill_side")
        if second_side is not None and int(second_side.iloc[signal_index]) == side:
            return f"{direction} 2W"

        third_side = _series("signal_third_wiseman_fill_side")
        if third_side is not None and int(third_side.iloc[signal_index]) == side:
            return f"{direction} 3W"

        add_on_fractal_side = _series("signal_add_on_fractal_fill_side")
        if add_on_fractal_side is not None and int(add_on_fractal_side.iloc[signal_index]) == side:
            return f"{direction} Add-on Fractal"

        fractal_side = _series("signal_fractal_position_side")
        if fractal_side is not None and int(fractal_side.iloc[signal_index]) == side and not for_entry:
            return f"{direction} Fractal"

        first_setup = _series("signal_first_wiseman_setup_side")
        if first_setup is not None and int(first_setup.iloc[signal_index]) == side:
            ignored_reason = _series("signal_first_wiseman_ignored_reason")
            ignored_text = str(ignored_reason.iloc[signal_index]).strip() if ignored_reason is not None else ""
            if ignored_text == "":
                if for_entry:
                    first_fill = _series("signal_fill_prices_first")
                    has_first_fill = first_fill is not None and pd.notna(first_fill.iloc[signal_index])
                    if first_fill is not None and not has_first_fill:
                        # A retained 1W setup marker can coexist with a fractal-led fill.
                        # When no explicit first 1W fill is present on this bar, prefer
                        # fractal/fallback labeling instead of claiming a 1W entry.
                        pass
                    else:
                        return f"{direction} 1W"
                else:
                    return f"{direction} 1W"

        if fractal_side is not None and int(fractal_side.iloc[signal_index]) == side:
            return f"{direction} Fractal"

        return None



    def _resolve_signal_fill(self, bar: pd.Series, prev_close: float, side: int, signal_fill: float | None) -> float | None:
        if signal_fill is not None and np.isfinite(signal_fill) and signal_fill > 0:
            open_px = float(bar["open"])
            high_px = float(bar["high"])
            low_px = float(bar["low"])
            if low_px <= signal_fill <= high_px:
                return self._apply_execution_adjustment(signal_fill, side)

            ot = self.config.order_type.lower()
            if ot == "stop":
                if side > 0 and open_px >= signal_fill:
                    return self._apply_execution_adjustment(open_px, side)
                if side < 0 and open_px <= signal_fill:
                    return self._apply_execution_adjustment(open_px, side)
        return self._resolve_order_fill(bar=bar, prev_close=prev_close, side=side)

    def _resolve_units(
        self,
        capital: float,
        fill_price: float,
        stop_loss_price: float | None,
        bar_index: int,
        closes: pd.Series,
        periods_per_year: int,
    ) -> float:
        if not np.isfinite(capital) or not np.isfinite(fill_price):
            self._last_sizing_context = {}
            return 0.0
        mode = self.config.trade_size_mode
        value = self.config.trade_size_value
        self._last_sizing_context = {
            "mode": mode,
            "capital_snapshot": float(capital),
            "base_notional": None,
            "volatility_scale": None,
            "realized_vol_annual": None,
            "scaled_notional": None,
        }
        if mode == "usd":
            notional = max(0.0, value)
            self._last_sizing_context["base_notional"] = float(notional)
            self._last_sizing_context["scaled_notional"] = float(notional)
            return notional / fill_price if fill_price > 0 else 0.0
        if mode == "units":
            return max(0.0, value)
        if mode == "hybrid_min_usd_percent":
            percent_notional = max(0.0, capital * value)
            min_notional = max(0.0, self.config.trade_size_min_usd)
            notional = max(percent_notional, min_notional)
            return notional / fill_price if fill_price > 0 else 0.0
        if mode == "equity_milestone_usd":
            notional = max(0.0, value)
            for equity_threshold, milestone_notional in self.config.trade_size_equity_milestones:
                if capital >= equity_threshold:
                    notional = max(0.0, milestone_notional)
                else:
                    break
            return notional / fill_price if fill_price > 0 else 0.0
        if mode == "volatility_scaled":
            base_notional = max(0.0, capital * value)
            scale, realized_vol_annual = self._volatility_scale_details(
                closes=closes,
                bar_index=bar_index,
                periods_per_year=periods_per_year,
            )
            scaled_notional = base_notional * scale
            self._last_sizing_context["base_notional"] = float(base_notional)
            self._last_sizing_context["volatility_scale"] = float(scale)
            self._last_sizing_context["realized_vol_annual"] = (
                float(realized_vol_annual) if realized_vol_annual is not None else None
            )
            self._last_sizing_context["scaled_notional"] = float(scaled_notional)
            return scaled_notional / fill_price if fill_price > 0 else 0.0
        if mode == "stop_loss_scaled":
            if stop_loss_price is None or not np.isfinite(stop_loss_price):
                raise ValueError("stop_loss_scaled trade sizing requires strategy.signal_stop_loss_prices for entry signals")
            risk_per_unit = abs(fill_price - stop_loss_price)
            if risk_per_unit <= 0:
                raise ValueError("stop_loss_scaled trade sizing requires stop-loss prices different from fill prices")
            max_risk = max(0.0, capital * value)
            return max_risk / risk_per_unit
        notional = max(0.0, capital * value)
        self._last_sizing_context["base_notional"] = float(notional)
        self._last_sizing_context["scaled_notional"] = float(notional)
        return notional / fill_price if fill_price > 0 else 0.0

    def _volatility_scale(self, closes: pd.Series, bar_index: int, periods_per_year: int) -> float:
        scale, _ = self._volatility_scale_details(closes=closes, bar_index=bar_index, periods_per_year=periods_per_year)
        return scale

    def _volatility_scale_details(
        self,
        closes: pd.Series,
        bar_index: int,
        periods_per_year: int,
    ) -> tuple[float, float | None]:
        lookback = int(self.config.volatility_lookback)
        if lookback < 2 or periods_per_year <= 0 or bar_index < 1:
            return 1.0, None

        # Sizing at `bar_index` uses an order fill from that bar's open (or a known
        # signal fill set before execution). The current bar close is therefore not
        # observable yet and must be excluded to avoid look-ahead bias.
        start = max(0, bar_index - lookback)
        window = closes.iloc[start:bar_index].astype("float64")
        returns = window.pct_change().dropna()
        if returns.empty:
            return 1.0, None

        realized_vol_annual = float(returns.std(ddof=0) * np.sqrt(periods_per_year))
        if not np.isfinite(realized_vol_annual) or realized_vol_annual <= 0:
            return 1.0, None

        target_vol = float(self.config.volatility_target_annual)
        if not np.isfinite(target_vol) or target_vol <= 0:
            return 1.0, None

        raw_scale = target_vol / realized_vol_annual
        min_scale = max(0.0, float(self.config.volatility_min_scale))
        max_scale = max(min_scale, float(self.config.volatility_max_scale))
        return float(np.clip(raw_scale, min_scale, max_scale)), realized_vol_annual

    @staticmethod
    def _ensure_finite(value: float, message: str) -> None:
        if not np.isfinite(value):
            raise ValueError(message)

    def _resolve_order_fill(self, bar: pd.Series, prev_close: float, side: int) -> float | None:
        ot = self.config.order_type.lower()
        open_px, high_px, low_px = float(bar["open"]), float(bar["high"]), float(bar["low"])

        if ot == "market":
            return self._apply_execution_adjustment(open_px, side)

        if ot == "limit":
            limit_px = prev_close * (1 - self.config.limit_offset_pct if side > 0 else 1 + self.config.limit_offset_pct)
            touched = low_px <= limit_px if side > 0 else high_px >= limit_px
            if not touched:
                return None
            raw_fill = min(open_px, limit_px) if side > 0 else max(open_px, limit_px)
            return self._apply_execution_adjustment(raw_fill, side)

        if ot == "stop":
            stop_px = prev_close * (1 + self.config.stop_offset_pct if side > 0 else 1 - self.config.stop_offset_pct)
            triggered = high_px >= stop_px if side > 0 else low_px <= stop_px
            if not triggered:
                return None
            raw_fill = max(open_px, stop_px) if side > 0 else min(open_px, stop_px)
            return self._apply_execution_adjustment(raw_fill, side)

        if ot == "stop_limit":
            stop_px = prev_close * (1 + self.config.stop_offset_pct if side > 0 else 1 - self.config.stop_offset_pct)
            triggered = high_px >= stop_px if side > 0 else low_px <= stop_px
            if not triggered:
                return None
            limit_px = stop_px * (1 + self.config.stop_limit_offset_pct if side > 0 else 1 - self.config.stop_limit_offset_pct)
            touched = low_px <= limit_px if side > 0 else high_px >= limit_px
            if not touched:
                return None
            raw_fill = min(open_px, limit_px) if side > 0 else max(open_px, limit_px)
            return self._apply_execution_adjustment(raw_fill, side)

        raise ValueError(f"Unsupported order_type: {self.config.order_type}")

    def _financing_cost(self, units: float, price: float, position: int, periods_per_year: int) -> float:
        if position == 0 or units <= 0:
            return 0.0
        notional = abs(units * price)
        borrow = notional * (self.config.borrow_rate_annual / periods_per_year) if position < 0 else 0.0
        overnight = notional * (self.config.overnight_rate_annual / periods_per_year)
        funding = notional * self.config.funding_rate_per_period
        return borrow + overnight + funding

    def _effective_max_loss(self, capital_base: float | None) -> float | None:
        thresholds: list[float] = []
        if self.config.max_loss is not None and self.config.max_loss > 0:
            thresholds.append(float(self.config.max_loss))
        if (
            self.config.max_loss_pct_of_equity is not None
            and self.config.max_loss_pct_of_equity > 0
            and capital_base is not None
            and capital_base > 0
        ):
            thresholds.append(float(capital_base) * float(self.config.max_loss_pct_of_equity))
        if not thresholds:
            return None
        return min(thresholds)

    def _price_tolerance(self, a: float, b: float) -> float:
        scale = max(abs(a), abs(b), 1.0)
        return max(1e-12, scale * 1e-9)

    def _price_le(self, a: float, b: float) -> bool:
        return a <= b + self._price_tolerance(a, b)

    def _price_ge(self, a: float, b: float) -> bool:
        return a + self._price_tolerance(a, b) >= b

    def _resolve_max_loss_exit_fill(
        self,
        bar: pd.Series,
        position: int,
        entry_price: float,
        units: float,
        capital_base: float | None,
    ) -> float | None:
        max_loss = self._effective_max_loss(capital_base)
        if max_loss is None or max_loss <= 0 or position == 0 or units <= 0:
            return None

        loss_per_unit = max_loss / units
        if position > 0:
            stop_price = entry_price - loss_per_unit
            if stop_price <= 0 or not self._price_le(float(bar["low"]), stop_price):
                return None
            raw_fill = min(float(bar["open"]), stop_price)
        else:
            stop_price = entry_price + loss_per_unit
            if not self._price_ge(float(bar["high"]), stop_price):
                return None
            raw_fill = max(float(bar["open"]), stop_price)

        return self._apply_execution_adjustment(raw_fill, -position)

    def _resolve_equity_cutoff_exit_fill(
        self,
        bar: pd.Series,
        position: int,
        entry_price: float,
        units: float,
        capital: float,
        equity_cutoff: float,
    ) -> tuple[float | None, float | None]:
        if position == 0 or units <= 0 or equity_cutoff <= 0:
            return None, None

        if position > 0:
            threshold_price = entry_price + ((equity_cutoff - capital) / units)
            if threshold_price <= 0 or not self._price_le(float(bar["low"]), threshold_price):
                return None, None
            raw_fill = min(float(bar["open"]), threshold_price)
        else:
            threshold_price = entry_price - ((equity_cutoff - capital) / units)
            if not self._price_ge(float(bar["high"]), threshold_price):
                return None, None
            raw_fill = max(float(bar["open"]), threshold_price)

        return self._apply_execution_adjustment(raw_fill, -position), threshold_price

    def _compute_liquidation_price(self, side: int, entry_price: float, entry_capital: float, entry_notional: float) -> float | None:
        if side == 0 or entry_price <= 0 or entry_capital <= 0 or entry_notional <= 0:
            return None
        leverage = entry_notional / entry_capital
        if leverage <= 1.0:
            return None
        stop_out_pct = min(max(self.config.leverage_stop_out_pct, 0.0), 0.99)
        adverse_move = (1.0 / leverage) - stop_out_pct
        if adverse_move <= 0:
            adverse_move = 1e-6
        if side > 0:
            liq = entry_price * (1.0 - adverse_move)
            return liq if liq > 0 else 0.0
        return entry_price * (1.0 + adverse_move)

    def _resolve_liquidation_fill(self, bar: pd.Series, position: int, liquidation_price: float | None) -> float | None:
        if position == 0 or liquidation_price is None or liquidation_price <= 0:
            return None
        if position > 0:
            if self._price_le(float(bar["low"]), liquidation_price):
                raw_fill = min(float(bar["open"]), liquidation_price)
                return self._apply_execution_adjustment(raw_fill, -position)
            return None
        if self._price_ge(float(bar["high"]), liquidation_price):
            raw_fill = max(float(bar["open"]), liquidation_price)
            return self._apply_execution_adjustment(raw_fill, -position)
        return None

    def _fill_occurs_before_adverse_risk(
        self,
        *,
        bar_open: float,
        position: int,
        fill_price: float,
        risk_prices: list[float],
    ) -> bool:
        if position == 0 or not np.isfinite(fill_price):
            return False
        finite_risks = [price for price in risk_prices if np.isfinite(price)]
        if not finite_risks:
            return False

        if position > 0:
            relevant_risks = [price for price in finite_risks if self._price_le(price, bar_open)]
            if not relevant_risks:
                return False
            nearest_risk = max(relevant_risks)
            if self._price_le(bar_open, fill_price):
                return True
            return self._price_ge(bar_open, fill_price) and self._price_ge(fill_price, nearest_risk)

        relevant_risks = [price for price in finite_risks if self._price_ge(price, bar_open)]
        if not relevant_risks:
            return False
        nearest_risk = min(relevant_risks)
        if self._price_ge(bar_open, fill_price):
            return True
        return self._price_le(bar_open, fill_price) and self._price_le(fill_price, nearest_risk)

    def _is_fill_reached_this_bar(self, *, bar: pd.Series, position: int, fill_price: float) -> bool:
        if not np.isfinite(fill_price):
            return False
        bar_open = float(bar["open"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        if position > 0:
            return self._price_ge(bar_open, fill_price) or self._price_le(bar_low, fill_price)
        if position < 0:
            return self._price_le(bar_open, fill_price) or self._price_ge(bar_high, fill_price)
        return False

    def _apply_execution_adjustment(self, price: float, side: int) -> float:
        spread_component = self.config.spread_rate / 2
        if side > 0:
            return price * (1 + self.config.slippage_rate + spread_component)
        if side < 0:
            return price * (1 - self.config.slippage_rate - spread_component)
        return price
