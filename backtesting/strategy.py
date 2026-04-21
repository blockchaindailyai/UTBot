from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import pandas as pd

from .fractals import detect_williams_fractals
from .stats import infer_periods_per_year


class Strategy(ABC):
    """Base class for pluggable strategies."""

    execute_on_signal_bar: bool = False
    signal_fill_prices: pd.Series | None = None
    signal_stop_loss_prices: pd.Series | None = None
    signal_fill_prices_first: pd.Series | None = None
    signal_fill_prices_second: pd.Series | None = None
    signal_fill_prices_third: pd.Series | None = None
    signal_second_wiseman_setup_side: pd.Series | None = None
    signal_second_wiseman_fill_side: pd.Series | None = None
    signal_third_wiseman_fill_side: pd.Series | None = None
    signal_third_wiseman_setup_side: pd.Series | None = None
    signal_contracts: pd.Series | None = None
    signal_first_wiseman_setup_side: pd.Series | None = None
    signal_first_wiseman_ignored_reason: pd.Series | None = None
    signal_first_wiseman_reversal_side: pd.Series | None = None
    signal_add_on_fractal_fill_side: pd.Series | None = None
    signal_exit_reason: pd.Series | None = None

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return a Series aligned to `data.index` with values in {-1, 0, 1}."""


def _alligator_lines(data: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    median_price = (data["high"] + data["low"]) / 2
    jaw = _smma(median_price, 13).shift(8)
    teeth = _smma(median_price, 8).shift(5)
    lips = _smma(median_price, 5).shift(3)
    return jaw, teeth, lips


def _williams_ao(data: pd.DataFrame) -> pd.Series:
    median_price = (data["high"] + data["low"]) / 2
    return median_price.rolling(5).mean() - median_price.rolling(34).mean()


def _annualized_volatility_scaled_return_threshold(
    base_return: float,
    annualized_volatility: float | None,
    annualized_volatility_scaler: float,
) -> float:
    """Scale a base return threshold by annualized volatility relative to a target annual volatility."""
    if base_return <= 0:
        return 0.0
    if (
        annualized_volatility is None
        or pd.isna(annualized_volatility)
        or annualized_volatility_scaler <= 0
    ):
        return np.inf
    return base_return * max(float(annualized_volatility), 0.0) / annualized_volatility_scaler


def _scaled_annualized_volatility_trigger(
    base_trigger: float,
    annualized_volatility_scaler: float,
) -> float:
    """Scale an annualized-volatility trigger relative to the configured reference volatility."""
    if base_trigger <= 0:
        return 0.0
    if annualized_volatility_scaler <= 0:
        return np.inf
    return base_trigger * annualized_volatility_scaler


class SmaCrossoverStrategy(Strategy):
    def __init__(self, fast: int = 10, slow: int = 30) -> None:
        if fast >= slow:
            raise ValueError("fast period must be smaller than slow period")
        self.fast = fast
        self.slow = slow

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        fast_ma = data["close"].rolling(self.fast).mean()
        slow_ma = data["close"].rolling(self.slow).mean()

        signal = pd.Series(0, index=data.index, dtype="int8")
        signal[fast_ma > slow_ma] = 1
        signal[fast_ma < slow_ma] = -1
        return signal.fillna(0)


def _smma(series: pd.Series, period: int) -> pd.Series:
    """Bill Williams smoothed moving average (SMMA)."""
    return series.ewm(alpha=1 / period, adjust=False).mean()


class AlligatorAOStrategy(Strategy):
    """Trend-following strategy using Bill Williams Alligator and AO alignment."""

    def __init__(self) -> None:
        self.jaw_period = 13
        self.jaw_shift = 8
        self.teeth_period = 8
        self.teeth_shift = 5
        self.lips_period = 5
        self.lips_shift = 3
        self.ao_fast = 5
        self.ao_slow = 34
        self.ao_signal = 5

    def _alligator(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        median_price = (data["high"] + data["low"]) / 2
        jaw = _smma(median_price, self.jaw_period).shift(self.jaw_shift)
        teeth = _smma(median_price, self.teeth_period).shift(self.teeth_shift)
        lips = _smma(median_price, self.lips_period).shift(self.lips_shift)
        return jaw, teeth, lips

    def _ao_histogram(self, data: pd.DataFrame) -> pd.Series:
        median_price = (data["high"] + data["low"]) / 2
        macd_line = median_price.rolling(self.ao_fast).mean() - median_price.rolling(self.ao_slow).mean()
        ao_signal = macd_line.rolling(self.ao_signal).mean()
        return macd_line - ao_signal

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        jaw, teeth, lips = self._alligator(data)
        ao_hist = self._ao_histogram(data)

        signal = pd.Series(0, index=data.index, dtype="int8")
        bullish = (lips > teeth) & (teeth > jaw) & (ao_hist > 0)
        bearish = (lips < teeth) & (teeth < jaw) & (ao_hist < 0)
        signal[bullish] = 1
        signal[bearish] = -1
        return signal.fillna(0)


class WisemanStrategy(Strategy):
    """State-machine strategy for trading 1st Wiseman bullish/bearish breakouts and reversals."""
    # Wiseman entries/exits are stop-triggered at pre-known setup levels (high/low),
    # so confirmed breaks can be executed on the same bar without waiting an extra bar.
    execute_on_signal_bar = True
    _GATOR_DIRECTION_MODES = {1, 2, 3}

    def __init__(
        self,
        gator_width_lookback: int = 50,
        gator_width_mult: float = 1.0,
        gator_width_valid_factor: float = 1.0,
        gator_direction_mode: int = 1,
        first_wiseman_contracts: int = 1,
        second_wiseman_contracts: int = 3,
        third_wiseman_contracts: int = 5,
        reversal_contracts_mult: float = 1.0,
        first_wiseman_wait_bars_to_close: int = 0,
        first_wiseman_divergence_filter_bars: int = 0,
        first_wiseman_opposite_close_min_unrealized_return: float = 0.0,
        first_wiseman_reversal_cooldown: int = 0,
        cancel_reversal_on_first_wiseman_exit: bool = False,
        teeth_profit_protection_enabled: bool = False,
        teeth_profit_protection_min_bars: int = 3,
        teeth_profit_protection_min_unrealized_return: float = 1.0,
        teeth_profit_protection_credit_unrealized_before_min_bars: bool = False,
        teeth_profit_protection_require_gator_open: bool = True,
        profit_protection_volatility_lookback: int | None = None,
        profit_protection_annualized_volatility_scaler: float = 1.0,
        lips_profit_protection_enabled: bool = False,
        lips_profit_protection_volatility_trigger: float = 0.02,
        lips_profit_protection_profit_trigger_mult: float = 2.0,
        lips_profit_protection_volatility_lookback: int = 20,
        lips_profit_protection_recent_trade_lookback: int = 5,
        lips_profit_protection_min_unrealized_return: float = 1.0,
        lips_profit_protection_arm_on_min_unrealized_return: bool = False,
        zone_profit_protection_enabled: bool = False,
        zone_profit_protection_min_unrealized_return: float = 1.0,
    ) -> None:
        if gator_width_lookback <= 0:
            raise ValueError("gator_width_lookback must be positive")
        if gator_width_mult <= 0:
            raise ValueError("gator_width_mult must be positive")
        if gator_width_valid_factor <= 0:
            raise ValueError("gator_width_valid_factor must be positive")
        if gator_direction_mode not in self._GATOR_DIRECTION_MODES:
            raise ValueError("gator_direction_mode must be one of {1, 2, 3}")
        if first_wiseman_contracts < 0:
            raise ValueError("first_wiseman_contracts must be >= 0")
        if second_wiseman_contracts < 0:
            raise ValueError("second_wiseman_contracts must be >= 0")
        if third_wiseman_contracts < 0:
            raise ValueError("third_wiseman_contracts must be >= 0")
        if reversal_contracts_mult < 0:
            raise ValueError("reversal_contracts_mult must be >= 0")
        if first_wiseman_wait_bars_to_close < 0:
            raise ValueError("first_wiseman_wait_bars_to_close must be >= 0")
        if first_wiseman_divergence_filter_bars < 0:
            raise ValueError("first_wiseman_divergence_filter_bars must be >= 0")
        if first_wiseman_opposite_close_min_unrealized_return < 0:
            raise ValueError("first_wiseman_opposite_close_min_unrealized_return must be >= 0")
        if first_wiseman_reversal_cooldown < 0:
            raise ValueError("first_wiseman_reversal_cooldown must be >= 0")
        if teeth_profit_protection_min_bars < 1:
            raise ValueError("teeth_profit_protection_min_bars must be >= 1")
        if teeth_profit_protection_min_unrealized_return < 0:
            raise ValueError("teeth_profit_protection_min_unrealized_return must be >= 0")
        if profit_protection_volatility_lookback is not None and profit_protection_volatility_lookback < 2:
            raise ValueError("profit_protection_volatility_lookback must be >= 2")
        if profit_protection_annualized_volatility_scaler <= 0:
            raise ValueError("profit_protection_annualized_volatility_scaler must be > 0")
        if lips_profit_protection_volatility_trigger < 0:
            raise ValueError("lips_profit_protection_volatility_trigger must be >= 0")
        if lips_profit_protection_profit_trigger_mult < 0:
            raise ValueError("lips_profit_protection_profit_trigger_mult must be >= 0")
        if lips_profit_protection_volatility_lookback < 2:
            raise ValueError("lips_profit_protection_volatility_lookback must be >= 2")
        if lips_profit_protection_recent_trade_lookback < 1:
            raise ValueError("lips_profit_protection_recent_trade_lookback must be >= 1")
        if lips_profit_protection_min_unrealized_return < 0:
            raise ValueError("lips_profit_protection_min_unrealized_return must be >= 0")
        if zone_profit_protection_min_unrealized_return < 0:
            raise ValueError("zone_profit_protection_min_unrealized_return must be >= 0")

        self.gator_width_lookback = gator_width_lookback
        self.gator_width_mult = gator_width_mult
        self.gator_width_valid_factor = gator_width_valid_factor
        self.gator_direction_mode = gator_direction_mode
        self.first_wiseman_contracts = first_wiseman_contracts
        self.second_wiseman_contracts = second_wiseman_contracts
        self.third_wiseman_contracts = third_wiseman_contracts
        self.reversal_contracts_mult = reversal_contracts_mult
        self.first_wiseman_wait_bars_to_close = first_wiseman_wait_bars_to_close
        self.first_wiseman_divergence_filter_bars = first_wiseman_divergence_filter_bars
        self.first_wiseman_opposite_close_min_unrealized_return = first_wiseman_opposite_close_min_unrealized_return
        self.first_wiseman_reversal_cooldown = first_wiseman_reversal_cooldown
        self.cancel_reversal_on_first_wiseman_exit = cancel_reversal_on_first_wiseman_exit
        self.teeth_profit_protection_enabled = teeth_profit_protection_enabled
        self.teeth_profit_protection_min_bars = teeth_profit_protection_min_bars
        self.teeth_profit_protection_min_unrealized_return = teeth_profit_protection_min_unrealized_return
        self.teeth_profit_protection_credit_unrealized_before_min_bars = teeth_profit_protection_credit_unrealized_before_min_bars
        self.teeth_profit_protection_require_gator_open = teeth_profit_protection_require_gator_open
        self.profit_protection_volatility_lookback = (
            profit_protection_volatility_lookback
            if profit_protection_volatility_lookback is not None
            else lips_profit_protection_volatility_lookback
        )
        self.profit_protection_annualized_volatility_scaler = profit_protection_annualized_volatility_scaler
        self.lips_profit_protection_enabled = lips_profit_protection_enabled
        self.lips_profit_protection_volatility_trigger = lips_profit_protection_volatility_trigger
        self.lips_profit_protection_profit_trigger_mult = lips_profit_protection_profit_trigger_mult
        self.lips_profit_protection_volatility_lookback = lips_profit_protection_volatility_lookback
        self.lips_profit_protection_recent_trade_lookback = lips_profit_protection_recent_trade_lookback
        self.lips_profit_protection_min_unrealized_return = lips_profit_protection_min_unrealized_return
        self.lips_profit_protection_arm_on_min_unrealized_return = lips_profit_protection_arm_on_min_unrealized_return
        self.zone_profit_protection_enabled = zone_profit_protection_enabled
        self.zone_profit_protection_min_unrealized_return = zone_profit_protection_min_unrealized_return

    def _williams_zone_bars(self, data: pd.DataFrame, ao: pd.Series) -> tuple[pd.Series, pd.Series]:
        ac = ao - ao.rolling(5).mean()
        ao_green = ao >= ao.shift(1)
        ao_red = ao < ao.shift(1)
        ac_green = ac >= ac.shift(1)
        ac_red = ac < ac.shift(1)
        zone_green = (ao_green & ac_green).fillna(False)
        zone_red = (ao_red & ac_red).fillna(False)
        return zone_green.astype(bool), zone_red.astype(bool)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        median_price = (data["high"] + data["low"]) / 2
        jaw = _smma(median_price, 13).shift(8)
        teeth = _smma(median_price, 8).shift(5)
        lips = _smma(median_price, 5).shift(3)

        candidate_midpoint = (data["high"] + data["low"]) / 2
        red_to_green_distance = (teeth - lips).abs()
        green_to_midpoint_distance = (lips - candidate_midpoint).abs()
        gator_range = pd.concat([jaw, teeth, lips], axis=1).max(axis=1) - pd.concat([jaw, teeth, lips], axis=1).min(axis=1)
        gator_slope = jaw.diff().abs() + teeth.diff().abs() + lips.diff().abs()
        range_baseline = gator_range.rolling(self.gator_width_lookback, min_periods=1).median()
        slope_baseline = gator_slope.rolling(self.gator_width_lookback, min_periods=1).median()
        gator_closed = (gator_range <= (range_baseline * self.gator_width_mult)) & (
            gator_slope <= (slope_baseline * self.gator_width_mult)
        )
        if self.gator_direction_mode == 1:
            gator_up = (lips > teeth) & (teeth > jaw)
            gator_down = (lips < teeth) & (teeth < jaw)
        elif self.gator_direction_mode == 2:
            gator_up = (lips > teeth) | (lips > jaw)
            gator_down = (lips < teeth) | (lips < jaw)
        else:
            gator_up = candidate_midpoint > teeth
            gator_down = candidate_midpoint < teeth
        gator_width_valid = red_to_green_distance < (green_to_midpoint_distance * self.gator_width_valid_factor)

        ao = median_price.rolling(5).mean() - median_price.rolling(34).mean()
        close_returns = data["close"].pct_change()
        periods_per_year = infer_periods_per_year(data.index, default=252)
        rolling_volatility = (
            close_returns.rolling(self.profit_protection_volatility_lookback, min_periods=2).std(ddof=0)
            * np.sqrt(periods_per_year)
        )
        ao_green = ao > ao.shift(1)
        ao_red = ao < ao.shift(1)
        zone_green, zone_red = self._williams_zone_bars(data, ao)

        def _has_recent_ao_divergence(setup_bar: int, side: int) -> bool:
            if self.first_wiseman_divergence_filter_bars <= 0:
                return True
            if setup_bar < 1:
                return False

            lookback_start = max(1, setup_bar - self.first_wiseman_divergence_filter_bars)
            setup_ao = float(ao.iloc[setup_bar])
            if np.isnan(setup_ao):
                return False
            setup_mid = float(candidate_midpoint.iloc[setup_bar])
            if np.isnan(setup_mid):
                return False

            if side == 1:
                # Bullish setup candidates are AO-red by construction. Keep this
                # gate aligned so valid setups are not rejected a priori.
                if not bool(ao_red.iloc[setup_bar]):
                    return False
                for j in range(lookback_start, setup_bar):
                    current_ao = float(ao.iloc[j])
                    if np.isnan(current_ao):
                        continue
                    if (
                        current_ao < 0
                        and bool(ao_red.iloc[j])
                        and current_ao < setup_ao
                        and current_ao < float(ao.iloc[j - 1])
                        and current_ao < float(ao.iloc[j + 1])
                    ):
                        prior_mid = float(candidate_midpoint.iloc[j])
                        if setup_mid < prior_mid:
                            return True
                return False

            # Bearish setup candidates are AO-green by construction. Keep this
            # gate aligned so valid setups are not rejected a priori.
            if not bool(ao_green.iloc[setup_bar]):
                return False
            for j in range(lookback_start, setup_bar):
                current_ao = float(ao.iloc[j])
                if np.isnan(current_ao):
                    continue
                if (
                    current_ao > 0
                    and bool(ao_green.iloc[j])
                    and current_ao > setup_ao
                    and current_ao > float(ao.iloc[j - 1])
                    and current_ao > float(ao.iloc[j + 1])
                ):
                    prior_mid = float(candidate_midpoint.iloc[j])
                    if setup_mid > prior_mid:
                        return True
            return False


        def _first_wiseman_signal_allowed(signal_bar: int, signal_side: int) -> tuple[bool, str]:
            if (
                self.first_wiseman_reversal_cooldown > 0
                and last_first_wiseman_reversal_bar >= 0
                and signal_bar <= (last_first_wiseman_reversal_bar + self.first_wiseman_reversal_cooldown)
            ):
                return False, "reversal_cooldown_active"
            if not _has_recent_ao_divergence(signal_bar, signal_side):
                return False, "ao_divergence_filter"
            return True, ""

        def _can_exit_or_reverse_on_new_first_wiseman_signal(signal_side: int, signal_bar: int) -> bool:
            # Keep suppression logic symmetric: if a 1W signal is filtered,
            # it should not open, close, or reverse positions.
            allowed, _ = _first_wiseman_signal_allowed(signal_bar, signal_side)
            return allowed

        signals = pd.Series(0, index=data.index, dtype="int8")
        fill_prices = pd.Series(np.nan, index=data.index, dtype="float64")
        stop_loss_prices = pd.Series(np.nan, index=data.index, dtype="float64")
        fill_prices_first = pd.Series(np.nan, index=data.index, dtype="float64")
        fill_prices_second = pd.Series(np.nan, index=data.index, dtype="float64")
        fill_prices_third = pd.Series(np.nan, index=data.index, dtype="float64")
        second_wiseman_setup_side = pd.Series(0, index=data.index, dtype="int8")
        second_wiseman_fill_side = pd.Series(0, index=data.index, dtype="int8")
        third_wiseman_fill_side = pd.Series(0, index=data.index, dtype="int8")
        third_wiseman_setup_side = pd.Series(0, index=data.index, dtype="int8")
        contracts = pd.Series(0.0, index=data.index, dtype="float64")
        first_wiseman_setup_side = pd.Series(0, index=data.index, dtype="int8")
        first_wiseman_ignored_reason = pd.Series("", index=data.index, dtype="object")
        first_wiseman_reversal_side = pd.Series(0, index=data.index, dtype="int8")
        exit_reason = pd.Series("", index=data.index, dtype="object")

        fractals = detect_williams_fractals(data)

        position = 0
        position_contracts = 0.0
        entry_i = -1
        active_high = 0.0
        active_low = 0.0
        active_setup_bar = -1
        active_setup_side = 0
        active_levels_armed = False
        reversal_position_active = False
        reversal_stop_level = np.nan
        active_reversal_source_contracts = 0.0
        second_wiseman_allowed = False
        super_ao_order: dict[str, float | int] | None = None
        third_wiseman_watch: dict[str, float | int] | None = None
        third_wiseman_order: dict[str, float | int] | None = None
        third_wiseman_locked_until_new_first_trade = False  # lock only after a 3W setup is actually traded (filled)

        pending_setups: list[dict[str, float | int]] = []

        second_wiseman_triggered = False
        third_wiseman_triggered = False
        teeth_profit_protection_armed = False
        teeth_profit_protection_unrealized_gate_met = False
        profit_protection_entry_i = -1
        zone_profit_protection_active = False
        zone_profit_protection_stop_level = np.nan
        zone_green_streak = 0
        zone_red_streak = 0
        last_first_wiseman_reversal_bar = -1
        recent_closed_trade_returns: deque[float] = deque(maxlen=self.lips_profit_protection_recent_trade_lookback)

        def _position_size(second_triggered: bool, third_triggered: bool) -> int:
            return (
                self.first_wiseman_contracts
                + (self.second_wiseman_contracts if second_triggered else 0)
                + (self.third_wiseman_contracts if third_triggered else 0)
            )

        def _reversal_size(source_contracts: float) -> float:
            if self.reversal_contracts_mult <= 0:
                return 0.0
            return source_contracts * self.reversal_contracts_mult

        def _opposite_close_unrealized_gate_met(position_side: int, entry_bar: int) -> bool:
            min_unrealized_return = self.first_wiseman_opposite_close_min_unrealized_return
            if min_unrealized_return <= 0:
                return True
            if entry_bar < 0:
                return False
            entry_price = float(fill_prices.iloc[entry_bar]) if entry_bar < len(fill_prices) else np.nan
            if not np.isfinite(entry_price) or entry_price <= 0:
                return False
            favorable_price = high_now if position_side == 1 else low_now
            unrealized_return = (
                (favorable_price - entry_price) / entry_price
                if position_side == 1
                else (entry_price - favorable_price) / entry_price
            )
            return unrealized_return >= min_unrealized_return

        def _clear_add_on_orders() -> None:
            nonlocal super_ao_order, third_wiseman_watch, third_wiseman_order
            super_ao_order = None
            third_wiseman_watch = None
            third_wiseman_order = None

        for i in range(len(data)):
            open_now = float(data["open"].iloc[i])
            high_now = float(data["high"].iloc[i])
            low_now = float(data["low"].iloc[i])
            close_now = float(data["close"].iloc[i])
            midpoint_now = (high_now + low_now) / 2.0
            gator_is_closed = bool(gator_closed.iloc[i])

            bearish_setup: dict[str, float | int] | None = None
            bullish_setup: dict[str, float | int] | None = None

            if i >= 3:
                pivot_bar = i - 1
                high_pivot = float(data["high"].iloc[pivot_bar])
                low_pivot = float(data["low"].iloc[pivot_bar])

                left_high_1 = float(data["high"].iloc[pivot_bar - 1])
                left_high_2 = float(data["high"].iloc[pivot_bar - 2])
                right_high_1 = float(data["high"].iloc[pivot_bar + 1])

                left_low_1 = float(data["low"].iloc[pivot_bar - 1])
                left_low_2 = float(data["low"].iloc[pivot_bar - 2])
                right_low_1 = float(data["low"].iloc[pivot_bar + 1])
                pivot_gator_lines = np.array(
                    [jaw.iloc[pivot_bar], teeth.iloc[pivot_bar], lips.iloc[pivot_bar]],
                    dtype="float64",
                )
                pivot_finite_gator_lines = pivot_gator_lines[np.isfinite(pivot_gator_lines)]
                pivot_gator_top = (
                    float(np.max(pivot_finite_gator_lines))
                    if pivot_finite_gator_lines.size > 0
                    else np.nan
                )
                pivot_gator_bottom = (
                    float(np.min(pivot_finite_gator_lines))
                    if pivot_finite_gator_lines.size > 0
                    else np.nan
                )
                bearish_setup_respects_gator = (
                    np.isfinite(pivot_gator_top) and low_pivot >= pivot_gator_top
                )
                bullish_setup_respects_gator = (
                    np.isfinite(pivot_gator_bottom) and high_pivot <= pivot_gator_bottom
                )

                has_two_left_and_one_right_lower_highs = (
                    high_pivot > left_high_1
                    and high_pivot > left_high_2
                    and high_pivot > right_high_1
                )
                bearish_body = float(data["open"].iloc[pivot_bar]) > float(data["close"].iloc[pivot_bar])
                if (
                    has_two_left_and_one_right_lower_highs
                    and bearish_body
                    and bool(gator_up.iloc[pivot_bar])
                    and (not bool(gator_closed.iloc[pivot_bar]))
                    and bool(gator_width_valid.iloc[pivot_bar])
                    and bool(ao_green.iloc[pivot_bar])
                    and bearish_setup_respects_gator
                ):
                    bearish_setup = {"side": -1, "high": high_pivot, "low": low_pivot, "bar": pivot_bar}

                has_two_left_and_one_right_higher_lows = (
                    low_pivot < left_low_1
                    and low_pivot < left_low_2
                    and low_pivot < right_low_1
                )
                bullish_body = float(data["close"].iloc[pivot_bar]) > float(data["open"].iloc[pivot_bar])
                if (
                    has_two_left_and_one_right_higher_lows
                    and bullish_body
                    and bool(gator_down.iloc[pivot_bar])
                    and (not bool(gator_closed.iloc[pivot_bar]))
                    and bool(gator_width_valid.iloc[pivot_bar])
                    and bool(ao_red.iloc[pivot_bar])
                    and bullish_setup_respects_gator
                ):
                    bullish_setup = {
                        "side": 1,
                        "high": high_pivot,
                        "low": low_pivot,
                        "bar": pivot_bar,
                        "gator_blocked": 0,
                    }

            new_setup = bearish_setup or bullish_setup
            if new_setup is not None:
                new_side = int(new_setup["side"])
                new_bar = int(new_setup["bar"])
                first_wiseman_setup_side.iloc[new_bar] = new_side
                setup_allowed, setup_ignored_reason = _first_wiseman_signal_allowed(new_bar, new_side)
                if not setup_allowed:
                    first_wiseman_ignored_reason.iloc[new_bar] = setup_ignored_reason
                else:
                    if self.first_wiseman_wait_bars_to_close > 0:
                        for pending in pending_setups:
                            if int(pending.get("limit_placed", 0)) and int(pending["side"]) != new_side:
                                pending["canceled_by_opposite_setup"] = 1
                    if active_levels_armed and position == 0 and active_setup_side == new_side:
                        # If a fresh same-side setup forms while flat, retire the prior setup's
                        # armed reversal level and let the new setup become the active reference.
                        active_levels_armed = False
                    if position == new_side and (active_levels_armed or reversal_position_active):
                        # Once a first Wiseman setup has already triggered a live position,
                        # subsequent same-side setups are informational only.
                        #
                        # This applies both while the original setup levels are armed and
                        # while a post-reversal position is active: in both states we avoid
                        # re-anchoring active risk references with weaker same-side setups.
                        first_wiseman_ignored_reason.iloc[new_bar] = "weaker_than_active_setup"
                    else:
                        if self.first_wiseman_wait_bars_to_close > 0:
                            new_setup["wait_until_close_bar"] = int(new_setup["bar"]) + self.first_wiseman_wait_bars_to_close
                            new_setup["limit_placed"] = 0
                        pending_setups.append(new_setup)

            prior_position = position
            prior_position_contracts = position_contracts
            active_reversal_eligible = active_setup_bar >= 0 and i >= active_setup_bar + 3
            bearish_active_level_touched = (
                active_levels_armed
                and active_setup_side == -1
                and position == -1
                and high_now >= active_high
            )
            if bearish_active_level_touched:
                if active_reversal_eligible:
                    position = 1
                    first_wiseman_reversal_side.iloc[i] = 1
                    last_first_wiseman_reversal_bar = i
                    reversal_source_contracts = (
                        active_reversal_source_contracts
                        if active_reversal_source_contracts > 0
                        else position_contracts
                    )
                    position_contracts = _reversal_size(reversal_source_contracts)
                    if position_contracts == 0:
                        position = 0
                    second_wiseman_triggered = False
                    third_wiseman_triggered = False
                    entry_i = i
                    reversal_position_active = position != 0
                    reversal_stop_level = float(data["low"].iloc[active_setup_bar : i + 1].min()) if active_setup_side == -1 and active_setup_bar >= 0 else np.nan
                    second_wiseman_allowed = False
                    _clear_add_on_orders()
                    fill_prices.iloc[i] = active_high
                    fill_prices_first.iloc[i] = active_high
                else:
                    # If setup high is reclaimed before reversal eligibility (t+3),
                    # stop out any active short and cancel reversal permanently.
                    if position == -1:
                        position = 0
                        exit_reason.iloc[i] = "1W Reversal Stop"
                        position_contracts = 0
                        second_wiseman_triggered = False
                        third_wiseman_triggered = False
                        reversal_position_active = False
                        reversal_stop_level = np.nan
                        second_wiseman_allowed = False
                        _clear_add_on_orders()

                # Opposite touch has been consumed (reversal or stop-out).
                active_levels_armed = False
                active_reversal_source_contracts = 0
                _clear_add_on_orders()
            bullish_active_level_touched = (
                active_levels_armed
                and active_setup_side == 1
                and position == 1
                and low_now <= active_low
            )
            if bullish_active_level_touched:
                if active_reversal_eligible:
                    position = -1
                    first_wiseman_reversal_side.iloc[i] = -1
                    last_first_wiseman_reversal_bar = i
                    reversal_source_contracts = (
                        active_reversal_source_contracts
                        if active_reversal_source_contracts > 0
                        else position_contracts
                    )
                    position_contracts = _reversal_size(reversal_source_contracts)
                    if position_contracts == 0:
                        position = 0
                    second_wiseman_triggered = False
                    third_wiseman_triggered = False
                    entry_i = i
                    reversal_position_active = position != 0
                    reversal_stop_level = float(data["high"].iloc[active_setup_bar : i + 1].max()) if active_setup_side == 1 and active_setup_bar >= 0 else np.nan
                    second_wiseman_allowed = False
                    _clear_add_on_orders()
                    fill_prices.iloc[i] = active_low
                    fill_prices_first.iloc[i] = active_low
                else:
                    # If setup low is lost before reversal eligibility (t+3),
                    # stop out any active long and cancel reversal permanently.
                    if position == 1:
                        position = 0
                        exit_reason.iloc[i] = "1W Reversal Stop"
                        position_contracts = 0
                        second_wiseman_triggered = False
                        third_wiseman_triggered = False
                        reversal_position_active = False
                        reversal_stop_level = np.nan
                        second_wiseman_allowed = False
                        _clear_add_on_orders()

                # Opposite touch has been consumed (reversal or stop-out).
                active_levels_armed = False
                active_reversal_source_contracts = 0
                _clear_add_on_orders()

            triggered_setup = False
            surviving_setups: list[dict[str, float | int]] = []
            for setup in pending_setups:
                side = int(setup["side"])
                p_high = float(setup["high"])
                p_low = float(setup["low"])
                p_bar = int(setup["bar"])
                reversal_eligible = i >= p_bar + 3

                if side == -1:
                    if int(setup.get("canceled_by_opposite_setup", 0)):
                        continue
                    if self.first_wiseman_wait_bars_to_close > 0:
                        wait_until_close_bar = int(setup.get("wait_until_close_bar", p_bar))
                        limit_placed = bool(int(setup.get("limit_placed", 0)))
                        if not limit_placed and i < wait_until_close_bar:
                            if high_now >= p_high:
                                first_wiseman_ignored_reason.iloc[p_bar] = "invalidation_before_trigger"
                                continue
                            surviving_setups.append(setup)
                            continue
                        if not limit_placed:
                            setup["limit_placed"] = 1
                            # Waiting uses close-based activation. After the configured bar closes,
                            # if the close is already through the limit price, fill immediately at
                            # that close; otherwise leave a resting limit for subsequent bars.
                            if close_now >= p_low and not triggered_setup and _can_exit_or_reverse_on_new_first_wiseman_signal(-1, p_bar):
                                if self.first_wiseman_contracts <= 0:
                                    first_wiseman_ignored_reason.iloc[p_bar] = "signal_disabled_zero_contracts"
                                    continue
                                position = -1
                                second_wiseman_triggered = False
                                third_wiseman_triggered = False
                                position_contracts = _position_size(second_wiseman_triggered, third_wiseman_triggered)
                                active_reversal_source_contracts = position_contracts
                                entry_i = i
                                reversal_position_active = False
                                second_wiseman_allowed = True
                                active_high = p_high
                                active_low = p_low
                                active_setup_bar = p_bar
                                active_setup_side = side
                                active_levels_armed = True
                                fill_prices.iloc[i] = close_now
                                fill_prices_first.iloc[i] = close_now
                                triggered_setup = True
                                third_wiseman_locked_until_new_first_trade = False
                                super_ao_order = None
                                third_wiseman_watch = {"side": side, "setup_bar": p_bar}
                                third_wiseman_order = None

                                if high_now >= p_high:
                                    if reversal_eligible:
                                        position = 1
                                        first_wiseman_reversal_side.iloc[i] = 1
                                        last_first_wiseman_reversal_bar = i
                                        position_contracts = _reversal_size(position_contracts)
                                        if position_contracts == 0:
                                            position = 0
                                        second_wiseman_triggered = False
                                        third_wiseman_triggered = False
                                        entry_i = i
                                        reversal_position_active = position != 0
                                        reversal_stop_level = float(data["low"].iloc[p_bar : i + 1].min())
                                        second_wiseman_allowed = False
                                        fill_prices.iloc[i] = p_high
                                        fill_prices_first.iloc[i] = p_high
                                        third_wiseman_locked_until_new_first_trade = False
                                        third_wiseman_watch = None
                                        third_wiseman_order = None
                                        active_reversal_source_contracts = 0
                                        active_levels_armed = False
                                    else:
                                        position = 0
                                        exit_reason.iloc[i] = "1W Reversal Stop"
                                        position_contracts = 0
                                        second_wiseman_triggered = False
                                        third_wiseman_triggered = False
                                        reversal_position_active = False
                                        reversal_stop_level = np.nan
                                        second_wiseman_allowed = False
                                        third_wiseman_watch = None
                                        third_wiseman_order = None
                                        first_wiseman_ignored_reason.iloc[p_bar] = "same_bar_stop_before_reversal_window"
                                        active_reversal_source_contracts = 0
                                        active_levels_armed = False
                                    super_ao_order = None
                                continue
                            surviving_setups.append(setup)
                            continue
                        if low_now <= p_low <= high_now and not triggered_setup and _can_exit_or_reverse_on_new_first_wiseman_signal(-1, p_bar):
                            if self.first_wiseman_contracts <= 0:
                                first_wiseman_ignored_reason.iloc[p_bar] = "signal_disabled_zero_contracts"
                                continue
                            position = -1
                            second_wiseman_triggered = False
                            third_wiseman_triggered = False
                            position_contracts = _position_size(second_wiseman_triggered, third_wiseman_triggered)
                            active_reversal_source_contracts = position_contracts
                            entry_i = i
                            reversal_position_active = False
                            second_wiseman_allowed = True
                            active_high = p_high
                            active_low = p_low
                            active_setup_bar = p_bar
                            active_setup_side = side
                            active_levels_armed = True
                            fill_prices.iloc[i] = p_low
                            fill_prices_first.iloc[i] = p_low
                            triggered_setup = True
                            third_wiseman_locked_until_new_first_trade = False
                            super_ao_order = None
                            third_wiseman_watch = {"side": side, "setup_bar": p_bar}
                            third_wiseman_order = None
                            continue
                        surviving_setups.append(setup)
                        continue
                    if low_now <= p_low and not triggered_setup and _can_exit_or_reverse_on_new_first_wiseman_signal(-1, p_bar):
                        if self.first_wiseman_contracts <= 0:
                            first_wiseman_ignored_reason.iloc[p_bar] = "signal_disabled_zero_contracts"
                            continue
                        position = -1
                        second_wiseman_triggered = False
                        third_wiseman_triggered = False
                        position_contracts = _position_size(second_wiseman_triggered, third_wiseman_triggered)
                        active_reversal_source_contracts = position_contracts
                        entry_i = i
                        reversal_position_active = False
                        second_wiseman_allowed = True
                        active_high = p_high
                        active_low = p_low
                        active_setup_bar = p_bar
                        active_setup_side = side
                        active_levels_armed = True
                        fill_prices.iloc[i] = p_low
                        fill_prices_first.iloc[i] = p_low
                        triggered_setup = True
                        third_wiseman_locked_until_new_first_trade = False  # lock only after a 3W setup is actually traded (filled)
                        super_ao_order = None
                        third_wiseman_watch = {"side": side, "setup_bar": p_bar}
                        third_wiseman_order = None

                        if high_now >= p_high:
                            if reversal_eligible:
                                position = 1
                                first_wiseman_reversal_side.iloc[i] = 1
                                last_first_wiseman_reversal_bar = i
                                position_contracts = _reversal_size(position_contracts)
                                if position_contracts == 0:
                                    position = 0
                                second_wiseman_triggered = False
                                third_wiseman_triggered = False
                                entry_i = i
                                reversal_position_active = position != 0
                                reversal_stop_level = float(data["low"].iloc[p_bar : i + 1].min())
                                second_wiseman_allowed = False
                                fill_prices.iloc[i] = p_high
                                fill_prices_first.iloc[i] = p_high
                                third_wiseman_locked_until_new_first_trade = False  # lock only after a 3W setup is actually traded (filled)
                                third_wiseman_watch = None
                                third_wiseman_order = None
                                active_reversal_source_contracts = 0
                            else:
                                position = 0
                                exit_reason.iloc[i] = "1W Reversal Stop"
                                position_contracts = 0
                                second_wiseman_triggered = False
                                third_wiseman_triggered = False
                                reversal_position_active = False
                                reversal_stop_level = np.nan
                                second_wiseman_allowed = False
                                third_wiseman_watch = None
                                third_wiseman_order = None
                                first_wiseman_ignored_reason.iloc[p_bar] = "same_bar_stop_before_reversal_window"
                                active_reversal_source_contracts = 0
                                active_levels_armed = False
                            # Keep original setup reversal level armed even when profit
                            # protection exits the position, so the one-time reversal entry
                            # can still fire later if price reaches the setup stop level.
                            super_ao_order = None
                    elif high_now >= p_high:
                        first_wiseman_ignored_reason.iloc[p_bar] = "invalidation_before_trigger"
                        continue
                    else:
                        surviving_setups.append(setup)
                else:
                    if int(setup.get("canceled_by_opposite_setup", 0)):
                        continue
                    if self.first_wiseman_wait_bars_to_close > 0:
                        wait_until_close_bar = int(setup.get("wait_until_close_bar", p_bar))
                        limit_placed = bool(int(setup.get("limit_placed", 0)))
                        if not limit_placed and i < wait_until_close_bar:
                            if low_now <= p_low:
                                first_wiseman_ignored_reason.iloc[p_bar] = "invalidation_before_trigger"
                                continue
                            surviving_setups.append(setup)
                            continue
                        if not limit_placed:
                            setup["limit_placed"] = 1
                            # Waiting uses close-based activation. After the configured bar closes,
                            # if the close is already through the limit price, fill immediately at
                            # that close; otherwise leave a resting limit for subsequent bars.
                            if close_now <= p_high and not triggered_setup and _can_exit_or_reverse_on_new_first_wiseman_signal(1, p_bar):
                                if self.first_wiseman_contracts <= 0:
                                    first_wiseman_ignored_reason.iloc[p_bar] = "signal_disabled_zero_contracts"
                                    continue
                                position = 1
                                second_wiseman_triggered = False
                                third_wiseman_triggered = False
                                position_contracts = _position_size(second_wiseman_triggered, third_wiseman_triggered)
                                active_reversal_source_contracts = position_contracts
                                entry_i = i
                                reversal_position_active = False
                                second_wiseman_allowed = True
                                active_high = p_high
                                active_low = p_low
                                active_setup_bar = p_bar
                                active_setup_side = side
                                active_levels_armed = True
                                fill_prices.iloc[i] = close_now
                                fill_prices_first.iloc[i] = close_now
                                triggered_setup = True
                                third_wiseman_locked_until_new_first_trade = False
                                super_ao_order = None
                                third_wiseman_watch = {"side": side, "setup_bar": p_bar}
                                third_wiseman_order = None

                                if low_now <= p_low:
                                    if reversal_eligible:
                                        position = -1
                                        first_wiseman_reversal_side.iloc[i] = -1
                                        last_first_wiseman_reversal_bar = i
                                        position_contracts = _reversal_size(position_contracts)
                                        if position_contracts == 0:
                                            position = 0
                                        second_wiseman_triggered = False
                                        third_wiseman_triggered = False
                                        entry_i = i
                                        reversal_position_active = position != 0
                                        reversal_stop_level = float(data["high"].iloc[p_bar : i + 1].max())
                                        second_wiseman_allowed = False
                                        fill_prices.iloc[i] = p_low
                                        fill_prices_first.iloc[i] = p_low
                                        third_wiseman_locked_until_new_first_trade = False
                                        third_wiseman_watch = None
                                        third_wiseman_order = None
                                        active_reversal_source_contracts = 0
                                        active_levels_armed = False
                                    else:
                                        position = 0
                                        exit_reason.iloc[i] = "1W Reversal Stop"
                                        position_contracts = 0
                                        second_wiseman_triggered = False
                                        third_wiseman_triggered = False
                                        reversal_position_active = False
                                        reversal_stop_level = np.nan
                                        second_wiseman_allowed = False
                                        third_wiseman_watch = None
                                        third_wiseman_order = None
                                        first_wiseman_ignored_reason.iloc[p_bar] = "same_bar_stop_before_reversal_window"
                                        active_reversal_source_contracts = 0
                                        active_levels_armed = False
                                    super_ao_order = None
                                continue
                            surviving_setups.append(setup)
                            continue
                        if low_now <= p_high <= high_now and not triggered_setup and _can_exit_or_reverse_on_new_first_wiseman_signal(1, p_bar):
                            if self.first_wiseman_contracts <= 0:
                                first_wiseman_ignored_reason.iloc[p_bar] = "signal_disabled_zero_contracts"
                                continue
                            position = 1
                            second_wiseman_triggered = False
                            third_wiseman_triggered = False
                            position_contracts = _position_size(second_wiseman_triggered, third_wiseman_triggered)
                            active_reversal_source_contracts = position_contracts
                            entry_i = i
                            reversal_position_active = False
                            second_wiseman_allowed = True
                            active_high = p_high
                            active_low = p_low
                            active_setup_bar = p_bar
                            active_setup_side = side
                            active_levels_armed = True
                            fill_prices.iloc[i] = p_high
                            fill_prices_first.iloc[i] = p_high
                            triggered_setup = True
                            third_wiseman_locked_until_new_first_trade = False
                            super_ao_order = None
                            third_wiseman_watch = {"side": side, "setup_bar": p_bar}
                            third_wiseman_order = None
                            continue
                        surviving_setups.append(setup)
                        continue
                    if high_now >= p_high and not triggered_setup and _can_exit_or_reverse_on_new_first_wiseman_signal(1, p_bar):
                        if self.first_wiseman_contracts <= 0:
                            first_wiseman_ignored_reason.iloc[p_bar] = "signal_disabled_zero_contracts"
                            continue
                        position = 1
                        second_wiseman_triggered = False
                        third_wiseman_triggered = False
                        position_contracts = _position_size(second_wiseman_triggered, third_wiseman_triggered)
                        active_reversal_source_contracts = position_contracts
                        entry_i = i
                        reversal_position_active = False
                        second_wiseman_allowed = True
                        active_high = p_high
                        active_low = p_low
                        active_setup_bar = p_bar
                        active_setup_side = side
                        active_levels_armed = True
                        fill_prices.iloc[i] = p_high
                        fill_prices_first.iloc[i] = p_high
                        triggered_setup = True
                        third_wiseman_locked_until_new_first_trade = False  # lock only after a 3W setup is actually traded (filled)
                        super_ao_order = None
                        third_wiseman_watch = {"side": side, "setup_bar": p_bar}
                        third_wiseman_order = None

                        if low_now <= p_low:
                            if reversal_eligible:
                                position = -1
                                first_wiseman_reversal_side.iloc[i] = -1
                                last_first_wiseman_reversal_bar = i
                                position_contracts = _reversal_size(position_contracts)
                                if position_contracts == 0:
                                    position = 0
                                second_wiseman_triggered = False
                                third_wiseman_triggered = False
                                entry_i = i
                                reversal_position_active = position != 0
                                reversal_stop_level = float(data["high"].iloc[p_bar : i + 1].max())
                                second_wiseman_allowed = False
                                fill_prices.iloc[i] = p_low
                                fill_prices_first.iloc[i] = p_low
                                third_wiseman_locked_until_new_first_trade = False  # lock only after a 3W setup is actually traded (filled)
                                third_wiseman_watch = None
                                third_wiseman_order = None
                                active_reversal_source_contracts = 0
                            else:
                                position = 0
                                exit_reason.iloc[i] = "1W Reversal Stop"
                                position_contracts = 0
                                second_wiseman_triggered = False
                                third_wiseman_triggered = False
                                reversal_position_active = False
                                reversal_stop_level = np.nan
                                second_wiseman_allowed = False
                                third_wiseman_watch = None
                                third_wiseman_order = None
                                first_wiseman_ignored_reason.iloc[p_bar] = "same_bar_stop_before_reversal_window"
                                active_reversal_source_contracts = 0
                            active_levels_armed = False
                            super_ao_order = None
                    elif low_now <= p_low:
                        first_wiseman_ignored_reason.iloc[p_bar] = "invalidation_before_trigger"
                        continue
                    else:
                        surviving_setups.append(setup)

            pending_setups = surviving_setups

            if (
                i >= 2
                and position == -1
                and not second_wiseman_triggered
                and super_ao_order is None
                and second_wiseman_allowed
                and self.second_wiseman_contracts > 0
            ):
                if bool(ao_red.iloc[i]) and bool(ao_red.iloc[i - 1]) and bool(ao_red.iloc[i - 2]):
                    super_ao_order = {"side": -1, "trigger": low_now, "bar": i}
                    second_wiseman_setup_side.iloc[i] = -1
            elif (
                i >= 2
                and position == 1
                and not second_wiseman_triggered
                and super_ao_order is None
                and second_wiseman_allowed
                and self.second_wiseman_contracts > 0
            ):
                if bool(ao_green.iloc[i]) and bool(ao_green.iloc[i - 1]) and bool(ao_green.iloc[i - 2]):
                    super_ao_order = {"side": 1, "trigger": high_now, "bar": i}
                    second_wiseman_setup_side.iloc[i] = 1

            if super_ao_order is not None and i > int(super_ao_order["bar"]):
                if int(super_ao_order["side"]) == -1 and low_now <= float(super_ao_order["trigger"]) and position == -1:
                    second_wiseman_triggered = True
                    position_contracts = _position_size(second_wiseman_triggered, third_wiseman_triggered)
                    active_reversal_source_contracts = position_contracts
                    fill_prices.iloc[i] = float(super_ao_order["trigger"])
                    fill_prices_second.iloc[i] = float(super_ao_order["trigger"])
                    second_wiseman_fill_side.iloc[i] = -1
                    super_ao_order = None
                elif int(super_ao_order["side"]) == 1 and high_now >= float(super_ao_order["trigger"]) and position == 1:
                    second_wiseman_triggered = True
                    position_contracts = _position_size(second_wiseman_triggered, third_wiseman_triggered)
                    active_reversal_source_contracts = position_contracts
                    fill_prices.iloc[i] = float(super_ao_order["trigger"])
                    fill_prices_second.iloc[i] = float(super_ao_order["trigger"])
                    second_wiseman_fill_side.iloc[i] = 1
                    super_ao_order = None

            if (
                third_wiseman_watch is not None
                and not third_wiseman_locked_until_new_first_trade
                and i >= 4
                and self.third_wiseman_contracts > 0
            ):
                fractal_i = i - 2
                if fractal_i > int(third_wiseman_watch["setup_bar"]):
                    watch_side = int(third_wiseman_watch["side"])
                    if watch_side == -1 and bool(fractals["down_fractal"].iloc[fractal_i]):
                        fractal_price = float(data["low"].iloc[fractal_i])
                        if pd.notna(teeth.iloc[fractal_i]) and fractal_price < float(teeth.iloc[fractal_i]):
                            if third_wiseman_order is not None:
                                prior_fractal_bar = int(third_wiseman_order["fractal_bar"])
                                third_wiseman_setup_side.iloc[prior_fractal_bar] = 0
                            third_wiseman_order = {
                                "side": -1,
                                "trigger": fractal_price,
                                "bar": i,
                                "fractal_bar": fractal_i,
                            }
                            third_wiseman_setup_side.iloc[fractal_i] = -1
                    elif watch_side == 1 and bool(fractals["up_fractal"].iloc[fractal_i]):
                        fractal_price = float(data["high"].iloc[fractal_i])
                        if pd.notna(teeth.iloc[fractal_i]) and fractal_price > float(teeth.iloc[fractal_i]):
                            if third_wiseman_order is not None:
                                prior_fractal_bar = int(third_wiseman_order["fractal_bar"])
                                third_wiseman_setup_side.iloc[prior_fractal_bar] = 0
                            third_wiseman_order = {
                                "side": 1,
                                "trigger": fractal_price,
                                "bar": i,
                                "fractal_bar": fractal_i,
                            }
                            third_wiseman_setup_side.iloc[fractal_i] = 1

            if third_wiseman_order is not None and i > int(third_wiseman_order["bar"]):
                if int(third_wiseman_order["side"]) == -1 and low_now <= float(third_wiseman_order["trigger"]) and position == -1:
                    third_wiseman_triggered = True
                    position_contracts = _position_size(second_wiseman_triggered, third_wiseman_triggered)
                    active_reversal_source_contracts = position_contracts
                    fill_prices.iloc[i] = float(third_wiseman_order["trigger"])
                    fill_prices_third.iloc[i] = float(third_wiseman_order["trigger"])
                    third_wiseman_fill_side.iloc[i] = -1
                    third_wiseman_order = None
                    third_wiseman_locked_until_new_first_trade = True
                elif int(third_wiseman_order["side"]) == 1 and high_now >= float(third_wiseman_order["trigger"]) and position == 1:
                    third_wiseman_triggered = True
                    position_contracts = _position_size(second_wiseman_triggered, third_wiseman_triggered)
                    active_reversal_source_contracts = position_contracts
                    fill_prices.iloc[i] = float(third_wiseman_order["trigger"])
                    fill_prices_third.iloc[i] = float(third_wiseman_order["trigger"])
                    third_wiseman_fill_side.iloc[i] = 1
                    third_wiseman_order = None
                    third_wiseman_locked_until_new_first_trade = True

            reversal_stop_triggered = (
                reversal_position_active
                and i > entry_i
                and not np.isnan(reversal_stop_level)
                and ((position == -1 and high_now >= reversal_stop_level) or (position == 1 and low_now <= reversal_stop_level))
            )
            if reversal_stop_triggered:
                position = 0
                exit_reason.iloc[i] = "1W Reversal Stop"
                position_contracts = 0
                second_wiseman_triggered = False
                third_wiseman_triggered = False
                reversal_position_active = False
                stopped_price = reversal_stop_level
                reversal_stop_level = np.nan
                second_wiseman_allowed = False
                active_levels_armed = False
                active_reversal_source_contracts = 0
                super_ao_order = None
                fill_prices.iloc[i] = stopped_price
                fill_prices_first.iloc[i] = stopped_price
                third_wiseman_watch = None
                third_wiseman_order = None

            zone_profit_protection_triggered = (
                self.zone_profit_protection_enabled
                and zone_profit_protection_active
                and position == prior_position
                and position != 0
                and i > entry_i
                and np.isnan(fill_prices.iloc[i])
                and np.isfinite(zone_profit_protection_stop_level)
                and (
                    (position == 1 and low_now <= zone_profit_protection_stop_level)
                    or (position == -1 and high_now >= zone_profit_protection_stop_level)
                )
            )
            if zone_profit_protection_triggered:
                stopped_price = (
                    min(open_now, zone_profit_protection_stop_level)
                    if position == 1
                    else max(open_now, zone_profit_protection_stop_level)
                )
                position = 0
                exit_reason.iloc[i] = "Williams Zone PP"
                position_contracts = 0
                second_wiseman_triggered = False
                third_wiseman_triggered = False
                reversal_position_active = False
                reversal_stop_level = np.nan
                second_wiseman_allowed = False
                _clear_add_on_orders()
                fill_prices.iloc[i] = stopped_price
                fill_prices_first.iloc[i] = stopped_price
                zone_profit_protection_active = False
                zone_profit_protection_stop_level = np.nan

            if np.isnan(fill_prices.iloc[i]) and position != prior_position:
                if prior_position == -1 and position in (0, 1):
                    fill_prices.iloc[i] = active_high
                    fill_prices_first.iloc[i] = active_high
                elif prior_position == 1 and position in (0, -1):
                    fill_prices.iloc[i] = active_low
                    fill_prices_first.iloc[i] = active_low
                elif prior_position == 0 and position == 1 and np.isfinite(active_high):
                    # A one-time reversal can trigger after a flat exit (e.g. PP close).
                    # Stamp the active setup level as the reversal fill so later logic
                    # (including profit-protection entry reference) has a valid entry.
                    fill_prices.iloc[i] = active_high
                    fill_prices_first.iloc[i] = active_high
                elif prior_position == 0 and position == -1 and np.isfinite(active_low):
                    fill_prices.iloc[i] = active_low
                    fill_prices_first.iloc[i] = active_low

            if (
                position != prior_position
                and prior_position != 0
                and entry_i >= 0
                and np.isfinite(fill_prices.iloc[i])
            ):
                entry_fill = fill_prices.iloc[entry_i]
                exit_fill = fill_prices.iloc[i]
                if np.isfinite(entry_fill) and entry_fill > 0 and np.isfinite(exit_fill):
                    realized_return = (
                        (exit_fill - entry_fill) / entry_fill
                        if prior_position == 1
                        else (entry_fill - exit_fill) / entry_fill
                    )
                    recent_closed_trade_returns.append(float(realized_return))

            if position != prior_position:
                # Profit-protection arming is trade-specific and must not carry across
                # a new entry/reversal; otherwise the newly opened trade can be closed
                # before its own arming gates (min bars/return) are satisfied.
                teeth_profit_protection_armed = False
                teeth_profit_protection_unrealized_gate_met = False
                profit_protection_entry_i = entry_i if position != 0 else -1
                zone_profit_protection_active = False
                zone_profit_protection_stop_level = np.nan
                zone_green_streak = 0
                zone_red_streak = 0

            if position == 0:
                reversal_position_active = False
                reversal_stop_level = np.nan
                second_wiseman_allowed = False
                position_contracts = 0
                second_wiseman_triggered = False
                third_wiseman_triggered = False
                _clear_add_on_orders()
                teeth_profit_protection_armed = False
                teeth_profit_protection_unrealized_gate_met = False
                profit_protection_entry_i = -1
                zone_profit_protection_active = False
                zone_profit_protection_stop_level = np.nan
                zone_green_streak = 0
                zone_red_streak = 0

            if (
                self.cancel_reversal_on_first_wiseman_exit
                and active_levels_armed
                and not reversal_position_active
                and prior_position == active_setup_side
                and position != active_setup_side
            ):
                active_levels_armed = False
                active_reversal_source_contracts = 0

            if (
                self.teeth_profit_protection_enabled
                or self.lips_profit_protection_enabled
                or self.zone_profit_protection_enabled
            ) and position != 0 and entry_i >= 0 and i >= entry_i:
                if profit_protection_entry_i != entry_i:
                    teeth_profit_protection_armed = False
                    teeth_profit_protection_unrealized_gate_met = False
                    zone_profit_protection_active = False
                    zone_profit_protection_stop_level = np.nan
                    zone_green_streak = 0
                    zone_red_streak = 0
                    profit_protection_entry_i = entry_i
                bars_in_position = i - entry_i
                entry_price_for_check = fill_prices.iloc[entry_i] if entry_i >= 0 else np.nan
                if np.isfinite(entry_price_for_check) and entry_price_for_check > 0:
                    close_now = float(data["close"].iloc[i])
                    favorable_price = high_now if position == 1 else low_now
                    unrealized_return = (
                        (favorable_price - entry_price_for_check) / entry_price_for_check
                        if position == 1
                        else (entry_price_for_check - favorable_price) / entry_price_for_check
                    )

                    teeth_now = teeth.iloc[i]
                    lips_now = lips.iloc[i]

                    if position == prior_position and np.isnan(fill_prices.iloc[i]):
                        if position == 1:
                            zone_green_streak = zone_green_streak + 1 if bool(zone_green.iloc[i]) else 0
                            zone_red_streak = 0
                        else:
                            zone_red_streak = zone_red_streak + 1 if bool(zone_red.iloc[i]) else 0
                            zone_green_streak = 0

                    volatility_now = rolling_volatility.iloc[i]
                    teeth_min_unrealized_return = _annualized_volatility_scaled_return_threshold(
                        self.teeth_profit_protection_min_unrealized_return,
                        volatility_now,
                        self.profit_protection_annualized_volatility_scaler,
                    )
                    zone_min_unrealized_return = _annualized_volatility_scaled_return_threshold(
                        self.zone_profit_protection_min_unrealized_return,
                        volatility_now,
                        self.profit_protection_annualized_volatility_scaler,
                    )
                    lips_min_unrealized_return = _annualized_volatility_scaled_return_threshold(
                        self.lips_profit_protection_min_unrealized_return,
                        volatility_now,
                        self.profit_protection_annualized_volatility_scaler,
                    )

                    if self.teeth_profit_protection_enabled or self.lips_profit_protection_enabled:
                        if (
                            self.teeth_profit_protection_credit_unrealized_before_min_bars
                            or bars_in_position >= self.teeth_profit_protection_min_bars
                        ) and unrealized_return >= teeth_min_unrealized_return:
                            teeth_profit_protection_unrealized_gate_met = True

                        gator_open = not gator_is_closed
                        gator_gate_ok = (not self.teeth_profit_protection_require_gator_open) or gator_open
                        meets_teeth_gate = (
                            bars_in_position >= self.teeth_profit_protection_min_bars
                            and teeth_profit_protection_unrealized_gate_met
                            and gator_gate_ok
                        )
                        if meets_teeth_gate:
                            teeth_profit_protection_armed = True

                    if (
                        self.zone_profit_protection_enabled
                        and position == prior_position
                        and np.isnan(fill_prices.iloc[i])
                    ):
                        streak = zone_green_streak if position == 1 else zone_red_streak
                        if zone_profit_protection_active:
                            zone_profit_protection_stop_level = low_now if position == 1 else high_now
                        elif (
                            streak >= 5
                            and unrealized_return >= zone_min_unrealized_return
                        ):
                            zone_profit_protection_active = True
                            zone_profit_protection_stop_level = low_now if position == 1 else high_now

                    use_lips_exit = False
                    if self.lips_profit_protection_enabled and pd.notna(lips_now):
                        recent_trade_baseline = (
                            float(np.median(np.abs(np.array(recent_closed_trade_returns, dtype="float64"))))
                            if recent_closed_trade_returns
                            else 0.0
                        )
                        volatility_baseline = float(volatility_now) if pd.notna(volatility_now) else 0.0
                        lips_volatility_trigger = _scaled_annualized_volatility_trigger(
                            self.lips_profit_protection_volatility_trigger,
                            self.profit_protection_annualized_volatility_scaler,
                        )
                        deep_profit_reference = max(volatility_baseline, recent_trade_baseline)
                        deep_profit_triggered = (
                            deep_profit_reference > 0
                            and unrealized_return > (deep_profit_reference * self.lips_profit_protection_profit_trigger_mult)
                        )
                        min_unrealized_triggered = (
                            unrealized_return > lips_min_unrealized_return
                        )
                        high_volatility = (
                            pd.notna(volatility_now)
                            and float(volatility_now) >= lips_volatility_trigger
                        )
                        use_lips_exit = (
                            teeth_profit_protection_armed
                            and (high_volatility or min_unrealized_triggered or deep_profit_triggered)
                        )

                    use_teeth_exit = (
                        self.teeth_profit_protection_enabled
                        and teeth_profit_protection_armed
                        and pd.notna(teeth_now)
                    )
                    if (
                        (use_teeth_exit or use_lips_exit)
                        and np.isnan(fill_prices.iloc[i])
                        and position == prior_position
                    ):
                        protection_level = float(lips_now) if use_lips_exit else float(teeth_now)
                        if (position == 1 and close_now < protection_level) or (position == -1 and close_now > protection_level):
                            position = 0
                            exit_reason.iloc[i] = "Green Gator PP" if use_lips_exit else "Red Gator PP"
                            position_contracts = 0
                            second_wiseman_triggered = False
                            third_wiseman_triggered = False
                            reversal_position_active = False
                            reversal_stop_level = np.nan
                            second_wiseman_allowed = False
                            _clear_add_on_orders()
                            fill_prices.iloc[i] = close_now
                            fill_prices_first.iloc[i] = close_now
                            teeth_profit_protection_armed = False

            if (
                self.cancel_reversal_on_first_wiseman_exit
                and active_levels_armed
                and not reversal_position_active
                and prior_position == active_setup_side
                and position != active_setup_side
            ):
                active_levels_armed = False
                active_reversal_source_contracts = 0

            signals.iloc[i] = position
            contracts.iloc[i] = position * position_contracts
            if position != 0:
                stop_reference = (
                    reversal_stop_level
                    if reversal_position_active and np.isfinite(reversal_stop_level)
                    else (active_low if position == 1 else active_high)
                )
                if np.isfinite(stop_reference):
                    stop_loss_prices.iloc[i] = float(stop_reference)

        for setup in pending_setups:
            p_bar = int(setup["bar"])
            if first_wiseman_ignored_reason.iloc[p_bar] == "":
                first_wiseman_ignored_reason.iloc[p_bar] = "no_breakout_until_end_of_data"

        self.signal_fill_prices = fill_prices
        self.signal_stop_loss_prices = stop_loss_prices
        self.signal_fill_prices_first = fill_prices_first
        self.signal_fill_prices_second = fill_prices_second
        self.signal_fill_prices_third = fill_prices_third
        self.signal_second_wiseman_setup_side = second_wiseman_setup_side
        self.signal_second_wiseman_fill_side = second_wiseman_fill_side
        self.signal_third_wiseman_fill_side = third_wiseman_fill_side
        self.signal_third_wiseman_setup_side = third_wiseman_setup_side
        self.signal_contracts = contracts
        self.signal_first_wiseman_setup_side = first_wiseman_setup_side
        self.signal_first_wiseman_ignored_reason = first_wiseman_ignored_reason
        self.signal_first_wiseman_reversal_side = first_wiseman_reversal_side
        self.signal_exit_reason = exit_reason
        return signals.fillna(0)


class BWStrategy(Strategy):
    """Fresh Bill Williams 1W-only strategy scaffold."""

    execute_on_signal_bar = True
    use_generic_trade_labels = False

    def __init__(
        self,
        divergence_filter_bars: int = 0,
        first_wiseman_contracts: int = 1,
        ntd_initial_fractal_enabled: bool = False,
        ntd_initial_fractal_contracts: int = 1,
        fractal_add_on_contracts: int = 0,
        ntd_sleeping_gator_lookback: int = 50,
        ntd_sleeping_gator_tightness_mult: float = 0.75,
        ntd_ranging_lookback: int = 20,
        ntd_ranging_max_span_pct: float = 0.025,
        gator_open_filter_lookback: int = 0,
        gator_open_filter_min_percentile: float = 50.0,
        red_teeth_profit_protection_enabled: bool = True,
        red_teeth_profit_protection_min_bars: int = 3,
        red_teeth_profit_protection_min_unrealized_return: float = 1.0,
        red_teeth_profit_protection_volatility_lookback: int = 20,
        red_teeth_profit_protection_annualized_volatility_scaler: float = 1.0,
        red_teeth_profit_protection_require_gator_direction_alignment: bool = False,
        green_lips_profit_protection_enabled: bool = True,
        green_lips_profit_protection_min_bars: int = 3,
        green_lips_profit_protection_min_unrealized_return: float = 1.1,
        green_lips_profit_protection_volatility_lookback: int = 20,
        green_lips_profit_protection_annualized_volatility_scaler: float = 1.0,
        green_lips_profit_protection_require_gator_direction_alignment: bool = False,
        red_teeth_latch_min_unrealized_return: bool = False,
        green_lips_latch_min_unrealized_return: bool = False,
        zones_profit_protection_enabled: bool = False,
        zones_profit_protection_min_bars: int = 3,
        zones_profit_protection_min_unrealized_return: float = 1.0,
        zones_profit_protection_volatility_lookback: int = 20,
        zones_profit_protection_annualized_volatility_scaler: float = 1.0,
        zones_profit_protection_min_same_color_bars: int = 5,
        peak_drawdown_exit_enabled: bool = False,
        peak_drawdown_exit_pct: float = 0.01,
        peak_drawdown_exit_volatility_lookback: int = 20,
        peak_drawdown_exit_annualized_volatility_scaler: float = 1.0,
        sigma_move_profit_protection_enabled: bool = False,
        sigma_move_profit_protection_lookback: int = 20,
        sigma_move_profit_protection_sigma: float = 2.0,
        close_on_underlying_gain_pct: float = 0.0,
        allow_close_on_1w_d: bool = False,
        allow_close_on_1w_d_min_unrealized_return: float = 0.0,
        allow_close_on_1w_a: bool = False,
        allow_close_on_1w_a_min_unrealized_return: float = 0.0,
        only_trade_1w_reversals: bool = False,
    ) -> None:
        if divergence_filter_bars < 0:
            raise ValueError("divergence_filter_bars must be >= 0")
        if first_wiseman_contracts < 0:
            raise ValueError("first_wiseman_contracts must be >= 0")
        if ntd_initial_fractal_contracts < 0:
            raise ValueError("ntd_initial_fractal_contracts must be >= 0")
        if fractal_add_on_contracts < 0:
            raise ValueError("fractal_add_on_contracts must be >= 0")
        if ntd_sleeping_gator_lookback <= 0:
            raise ValueError("ntd_sleeping_gator_lookback must be positive")
        if ntd_sleeping_gator_tightness_mult <= 0:
            raise ValueError("ntd_sleeping_gator_tightness_mult must be positive")
        if ntd_ranging_lookback <= 0:
            raise ValueError("ntd_ranging_lookback must be positive")
        if ntd_ranging_max_span_pct < 0:
            raise ValueError("ntd_ranging_max_span_pct must be >= 0")
        if gator_open_filter_lookback < 0:
            raise ValueError("gator_open_filter_lookback must be >= 0")
        if gator_open_filter_min_percentile < 0 or gator_open_filter_min_percentile > 100:
            raise ValueError("gator_open_filter_min_percentile must be in [0, 100]")
        if red_teeth_profit_protection_min_bars < 1:
            raise ValueError("red_teeth_profit_protection_min_bars must be >= 1")
        if red_teeth_profit_protection_min_unrealized_return < 0:
            raise ValueError("red_teeth_profit_protection_min_unrealized_return must be >= 0")
        if red_teeth_profit_protection_volatility_lookback < 2:
            raise ValueError("red_teeth_profit_protection_volatility_lookback must be >= 2")
        if red_teeth_profit_protection_annualized_volatility_scaler < 0:
            raise ValueError("red_teeth_profit_protection_annualized_volatility_scaler must be >= 0")
        if green_lips_profit_protection_min_bars < 1:
            raise ValueError("green_lips_profit_protection_min_bars must be >= 1")
        if green_lips_profit_protection_min_unrealized_return < 0:
            raise ValueError("green_lips_profit_protection_min_unrealized_return must be >= 0")
        if green_lips_profit_protection_volatility_lookback < 2:
            raise ValueError("green_lips_profit_protection_volatility_lookback must be >= 2")
        if green_lips_profit_protection_annualized_volatility_scaler < 0:
            raise ValueError("green_lips_profit_protection_annualized_volatility_scaler must be >= 0")
        if zones_profit_protection_min_bars < 1:
            raise ValueError("zones_profit_protection_min_bars must be >= 1")
        if zones_profit_protection_min_unrealized_return < 0:
            raise ValueError("zones_profit_protection_min_unrealized_return must be >= 0")
        if zones_profit_protection_volatility_lookback < 2:
            raise ValueError("zones_profit_protection_volatility_lookback must be >= 2")
        if zones_profit_protection_annualized_volatility_scaler < 0:
            raise ValueError("zones_profit_protection_annualized_volatility_scaler must be >= 0")
        if zones_profit_protection_min_same_color_bars < 1:
            raise ValueError("zones_profit_protection_min_same_color_bars must be >= 1")
        if peak_drawdown_exit_pct < 0:
            raise ValueError("peak_drawdown_exit_pct must be >= 0")
        if peak_drawdown_exit_volatility_lookback < 2:
            raise ValueError("peak_drawdown_exit_volatility_lookback must be >= 2")
        if peak_drawdown_exit_annualized_volatility_scaler < 0:
            raise ValueError("peak_drawdown_exit_annualized_volatility_scaler must be >= 0")
        if sigma_move_profit_protection_lookback < 2:
            raise ValueError("sigma_move_profit_protection_lookback must be >= 2")
        if sigma_move_profit_protection_sigma <= 0:
            raise ValueError("sigma_move_profit_protection_sigma must be > 0")
        if close_on_underlying_gain_pct < 0:
            raise ValueError("close_on_underlying_gain_pct must be >= 0")
        if allow_close_on_1w_d_min_unrealized_return < 0:
            raise ValueError("allow_close_on_1w_d_min_unrealized_return must be >= 0")
        if allow_close_on_1w_a_min_unrealized_return < 0:
            raise ValueError("allow_close_on_1w_a_min_unrealized_return must be >= 0")
        self.divergence_filter_bars = divergence_filter_bars
        self.first_wiseman_contracts = first_wiseman_contracts
        self.ntd_initial_fractal_enabled = ntd_initial_fractal_enabled
        self.ntd_initial_fractal_contracts = ntd_initial_fractal_contracts
        self.fractal_add_on_contracts = fractal_add_on_contracts
        self.ntd_sleeping_gator_lookback = ntd_sleeping_gator_lookback
        self.ntd_sleeping_gator_tightness_mult = ntd_sleeping_gator_tightness_mult
        self.ntd_ranging_lookback = ntd_ranging_lookback
        self.ntd_ranging_max_span_pct = ntd_ranging_max_span_pct
        self.gator_open_filter_lookback = gator_open_filter_lookback
        self.gator_open_filter_min_percentile = gator_open_filter_min_percentile
        self.red_teeth_profit_protection_enabled = red_teeth_profit_protection_enabled
        self.red_teeth_profit_protection_min_bars = red_teeth_profit_protection_min_bars
        self.red_teeth_profit_protection_min_unrealized_return = red_teeth_profit_protection_min_unrealized_return
        self.red_teeth_profit_protection_volatility_lookback = red_teeth_profit_protection_volatility_lookback
        self.red_teeth_profit_protection_annualized_volatility_scaler = (
            red_teeth_profit_protection_annualized_volatility_scaler
        )
        self.red_teeth_profit_protection_require_gator_direction_alignment = (
            red_teeth_profit_protection_require_gator_direction_alignment
        )
        self.green_lips_profit_protection_enabled = green_lips_profit_protection_enabled
        self.green_lips_profit_protection_min_bars = green_lips_profit_protection_min_bars
        self.green_lips_profit_protection_min_unrealized_return = green_lips_profit_protection_min_unrealized_return
        self.green_lips_profit_protection_volatility_lookback = green_lips_profit_protection_volatility_lookback
        self.green_lips_profit_protection_annualized_volatility_scaler = (
            green_lips_profit_protection_annualized_volatility_scaler
        )
        self.green_lips_profit_protection_require_gator_direction_alignment = (
            green_lips_profit_protection_require_gator_direction_alignment
        )
        self.red_teeth_latch_min_unrealized_return = red_teeth_latch_min_unrealized_return
        self.green_lips_latch_min_unrealized_return = green_lips_latch_min_unrealized_return
        self.zones_profit_protection_enabled = zones_profit_protection_enabled
        self.zones_profit_protection_min_bars = zones_profit_protection_min_bars
        self.zones_profit_protection_min_unrealized_return = zones_profit_protection_min_unrealized_return
        self.zones_profit_protection_volatility_lookback = zones_profit_protection_volatility_lookback
        self.zones_profit_protection_annualized_volatility_scaler = (
            zones_profit_protection_annualized_volatility_scaler
        )
        self.zones_profit_protection_min_same_color_bars = zones_profit_protection_min_same_color_bars
        self.peak_drawdown_exit_enabled = peak_drawdown_exit_enabled
        self.peak_drawdown_exit_pct = peak_drawdown_exit_pct
        self.peak_drawdown_exit_volatility_lookback = peak_drawdown_exit_volatility_lookback
        self.peak_drawdown_exit_annualized_volatility_scaler = (
            peak_drawdown_exit_annualized_volatility_scaler
        )
        self.sigma_move_profit_protection_enabled = sigma_move_profit_protection_enabled
        self.sigma_move_profit_protection_lookback = sigma_move_profit_protection_lookback
        self.sigma_move_profit_protection_sigma = sigma_move_profit_protection_sigma
        self.close_on_underlying_gain_pct = close_on_underlying_gain_pct
        self.allow_close_on_1w_d = allow_close_on_1w_d
        self.allow_close_on_1w_d_min_unrealized_return = allow_close_on_1w_d_min_unrealized_return
        self.allow_close_on_1w_a = allow_close_on_1w_a
        self.allow_close_on_1w_a_min_unrealized_return = allow_close_on_1w_a_min_unrealized_return
        self.only_trade_1w_reversals = only_trade_1w_reversals

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        index = data.index
        signals = pd.Series(0, index=index, dtype="int8")
        fill_prices = pd.Series(np.nan, index=index, dtype="float64")
        stop_loss_prices = pd.Series(np.nan, index=index, dtype="float64")
        contracts = pd.Series(0.0, index=index, dtype="float64")
        intrabar_events: dict[int, list[dict[str, float | int | str]]] = {}
        first_setup_side = pd.Series(0, index=index, dtype="int8")
        first_setup_marker_side = pd.Series(0, index=index, dtype="int8")
        first_ignored_reason = pd.Series("", index=index, dtype="object")
        first_reversal_side = pd.Series(0, index=index, dtype="int8")
        add_on_fractal_fill_side = pd.Series(0, index=index, dtype="int8")
        fractal_position_side = pd.Series(0, index=index, dtype="int8")
        exit_reason = pd.Series("", index=index, dtype="object")

        jaw, teeth, lips = _alligator_lines(data)
        ao = _williams_ao(data)
        ac = ao - ao.rolling(5).mean()
        zone_green = ((ao >= ao.shift(1)) & (ac >= ac.shift(1))).fillna(False)
        zone_red = ((ao < ao.shift(1)) & (ac < ac.shift(1))).fillna(False)
        fractals = detect_williams_fractals(data)
        gator_range = pd.concat([jaw, teeth, lips], axis=1).max(axis=1) - pd.concat([jaw, teeth, lips], axis=1).min(axis=1)
        gator_slope = jaw.diff().abs() + teeth.diff().abs() + lips.diff().abs()
        gator_open_filter_enabled = self.gator_open_filter_lookback > 0
        gator_open_threshold = pd.Series(np.nan, index=index, dtype="float64")
        if gator_open_filter_enabled:
            min_periods = min(self.gator_open_filter_lookback, 2)
            gator_open_threshold = gator_range.rolling(
                self.gator_open_filter_lookback,
                min_periods=min_periods,
            ).quantile(self.gator_open_filter_min_percentile / 100.0)
        close_returns = data["close"].pct_change()
        periods_per_year = infer_periods_per_year(data.index)
        red_teeth_annualized_volatility = (
            close_returns.rolling(self.red_teeth_profit_protection_volatility_lookback, min_periods=2).std(ddof=0)
            * np.sqrt(periods_per_year)
        )
        green_lips_annualized_volatility = (
            close_returns.rolling(self.green_lips_profit_protection_volatility_lookback, min_periods=2).std(ddof=0)
            * np.sqrt(periods_per_year)
        )
        zones_annualized_volatility = (
            close_returns.rolling(self.zones_profit_protection_volatility_lookback, min_periods=2).std(ddof=0)
            * np.sqrt(periods_per_year)
        )
        peak_drawdown_annualized_volatility = (
            close_returns.rolling(self.peak_drawdown_exit_volatility_lookback, min_periods=2).std(ddof=0)
            * np.sqrt(periods_per_year)
        )
        sigma_move_rolling_mean = data["close"].rolling(
            self.sigma_move_profit_protection_lookback,
            min_periods=self.sigma_move_profit_protection_lookback,
        ).mean().shift(1)
        sigma_move_rolling_std = data["close"].rolling(
            self.sigma_move_profit_protection_lookback,
            min_periods=self.sigma_move_profit_protection_lookback,
        ).std(ddof=0).shift(1)
        range_baseline = gator_range.rolling(self.ntd_sleeping_gator_lookback, min_periods=1).median()
        slope_baseline = gator_slope.rolling(self.ntd_sleeping_gator_lookback, min_periods=1).median()
        sleeping_gator = (
            (gator_range <= (range_baseline * self.ntd_sleeping_gator_tightness_mult))
            & (gator_slope <= (slope_baseline * self.ntd_sleeping_gator_tightness_mult))
        ).fillna(False)
        gator_waking_up = ((~sleeping_gator) & sleeping_gator.shift(1, fill_value=False)).fillna(False)
        ranging_span = (
            data["high"].rolling(self.ntd_ranging_lookback, min_periods=self.ntd_ranging_lookback).max()
            - data["low"].rolling(self.ntd_ranging_lookback, min_periods=self.ntd_ranging_lookback).min()
        )
        ranging_reference = data["close"].rolling(self.ntd_ranging_lookback, min_periods=self.ntd_ranging_lookback).mean().abs()
        ranging_reference = ranging_reference.where(ranging_reference > 1e-12, np.nan)
        price_ranging = ((ranging_span / ranging_reference) <= self.ntd_ranging_max_span_pct).fillna(False)

        pending_setup: dict[str, float | int] | None = None
        opposite_setup: dict[str, float | int] | None = None
        pending_ntd_initial_long: dict[str, float | int] | None = None
        pending_ntd_initial_short: dict[str, float | int] | None = None
        pending_filtered_opposite_close_setup: dict[str, float | int | str] | None = None
        pending_fractal_add_on: dict[str, float | int] | None = None
        latest_bearish_fractal_under_teeth_bar = -1
        latest_bearish_fractal_under_teeth_price = np.nan
        latest_bullish_fractal_above_teeth_bar = -1
        latest_bullish_fractal_above_teeth_price = np.nan
        position = 0
        position_entry_bar = -1
        reversal_position_active = False
        reversal_stop_loss: float | None = None
        position_entry_source = ""
        active_base_contract_size = 0.0
        active_add_on_contracts = 0.0
        first_wiseman_contract_size = float(self.first_wiseman_contracts)
        ntd_initial_fractal_contract_size = float(self.ntd_initial_fractal_contracts)
        ntd_initial_fractal_active = self.ntd_initial_fractal_enabled and ntd_initial_fractal_contract_size > 0
        fractal_add_on_contract_size = float(self.fractal_add_on_contracts)
        red_teeth_gator_alignment_latched = False
        red_teeth_gator_alignment_latched_entry_bar = -1
        green_lips_gator_alignment_latched = False
        green_lips_gator_alignment_latched_entry_bar = -1
        red_teeth_unrealized_return_latched = False
        red_teeth_unrealized_return_latched_entry_bar = -1
        green_lips_unrealized_return_latched = False
        green_lips_unrealized_return_latched_entry_bar = -1
        zones_profit_protection_active = False
        zones_profit_protection_entry_bar = -1
        zones_profit_protection_stop_level = np.nan
        zones_profit_protection_stop_set_bar = -1
        zones_green_streak = 0
        zones_red_streak = 0
        peak_drawdown_entry_bar = -1
        peak_favorable_return = 0.0

        def _append_intrabar_event(
            bar_index: int,
            event_type: str,
            side: int,
            price: float,
            contracts_value: float,
            reason: str,
            stop_loss_price: float | None = None,
        ) -> None:
            if not np.isfinite(price):
                return
            events = intrabar_events.setdefault(bar_index, [])
            normalized_stop = (
                float(stop_loss_price)
                if stop_loss_price is not None and np.isfinite(float(stop_loss_price))
                else np.nan
            )
            events.append(
                {
                    "event_type": str(event_type),
                    "side": int(side),
                    "price": float(price),
                    "contracts": float(max(contracts_value, 0.0)),
                    "reason": str(reason),
                    "stop_loss_price": normalized_stop,
                }
            )

        def _active_signed_contracts(side: int) -> float:
            if side == 0:
                return 0.0
            total = active_base_contract_size + active_add_on_contracts
            if total <= 0:
                return 0.0
            return total if side > 0 else -total

        def _opposite_close_unrealized_gate_met(
            position_side: int,
            entry_bar: int,
            high_now: float,
            low_now: float,
            min_unrealized_return: float,
        ) -> bool:
            if min_unrealized_return <= 0:
                return True
            if entry_bar < 0:
                return False
            entry_price = float(data["close"].iloc[entry_bar])
            if not np.isfinite(entry_price) or entry_price <= 0:
                return False
            if position_side > 0:
                favorable_price = float(high_now)
                if not np.isfinite(favorable_price):
                    return False
                unrealized_return = (favorable_price / entry_price) - 1.0
            elif position_side < 0:
                favorable_price = float(low_now)
                if not np.isfinite(favorable_price) or favorable_price <= 0:
                    return False
                unrealized_return = (entry_price / favorable_price) - 1.0
            else:
                return False
            return unrealized_return >= min_unrealized_return

        def _ntd_initial_long_stop(current_bar: int | None = None) -> float:
            if latest_bearish_fractal_under_teeth_bar < 0 or not np.isfinite(latest_bearish_fractal_under_teeth_price):
                return np.nan
            if current_bar is None or current_bar < latest_bearish_fractal_under_teeth_bar:
                return float(latest_bearish_fractal_under_teeth_price)
            return float(data["low"].iloc[latest_bearish_fractal_under_teeth_bar : current_bar + 1].min())

        def _ntd_initial_short_stop(current_bar: int | None = None) -> float:
            if latest_bullish_fractal_above_teeth_bar < 0 or not np.isfinite(latest_bullish_fractal_above_teeth_price):
                return np.nan
            if current_bar is None or current_bar < latest_bullish_fractal_above_teeth_bar:
                return float(latest_bullish_fractal_above_teeth_price)
            return float(data["high"].iloc[latest_bullish_fractal_above_teeth_bar : current_bar + 1].max())

        def _try_trigger_ntd_initial_entry(i: int, open_now: float, high_now: float, low_now: float) -> None:
            nonlocal position, position_entry_bar, pending_setup, opposite_setup, pending_ntd_initial_long, pending_ntd_initial_short
            nonlocal reversal_position_active, reversal_stop_loss
            nonlocal position_entry_source, active_base_contract_size, active_add_on_contracts, pending_fractal_add_on
            if (
                not ntd_initial_fractal_active
            ):
                return
            # While an NTD initial fractal position is active, stop-and-reverse
            # is owned by the active fractal stop (`pending_setup`) so fills
            # happen on the exact stop level instead of drifting to later
            # pending-trigger processing.
            if (
                position != 0
                and isinstance(pending_setup, dict)
                and str(pending_setup.get("source", "1w")) != "1w"
            ):
                return
            long_trigger_price = (
                float(pending_ntd_initial_long["trigger_price"])
                if pending_ntd_initial_long is not None and i > int(pending_ntd_initial_long["placed_bar"])
                else np.nan
            )
            short_trigger_price = (
                float(pending_ntd_initial_short["trigger_price"])
                if pending_ntd_initial_short is not None and i > int(pending_ntd_initial_short["placed_bar"])
                else np.nan
            )
            long_hit = np.isfinite(long_trigger_price) and high_now >= long_trigger_price
            short_hit = np.isfinite(short_trigger_price) and low_now <= short_trigger_price
            if not long_hit and not short_hit:
                return

            if position != 0:
                if position > 0 and short_hit and np.isfinite(short_trigger_price):
                    ntd_stop_price = _ntd_initial_short_stop(i)
                    if np.isfinite(ntd_stop_price) and ntd_stop_price > short_trigger_price:
                        position = -1
                        position_entry_bar = i
                        opposite_setup = None
                        reversal_position_active = False
                        reversal_stop_loss = None
                        position_entry_source = "fractal"
                        active_base_contract_size = ntd_initial_fractal_contract_size
                        active_add_on_contracts = 0.0
                        pending_fractal_add_on = None
                        fill_price = short_trigger_price
                        signals.iloc[i] = -1
                        contracts.iloc[i] = _active_signed_contracts(position)
                        fill_prices.iloc[i] = fill_price
                        stop_loss_prices.iloc[i] = ntd_stop_price
                        fractal_position_side.iloc[i] = -1
                        pending_setup = {
                            "index": i,
                            "high": ntd_stop_price,
                            "low": fill_price,
                            "side": -1,
                            "reversal_armed": 0,
                            "source": "fractal",
                        }
                        pending_ntd_initial_short = None
                    return
                if position < 0 and long_hit and np.isfinite(long_trigger_price):
                    ntd_stop_price = _ntd_initial_long_stop(i)
                    if np.isfinite(ntd_stop_price) and ntd_stop_price < long_trigger_price:
                        position = 1
                        position_entry_bar = i
                        opposite_setup = None
                        reversal_position_active = False
                        reversal_stop_loss = None
                        position_entry_source = "fractal"
                        active_base_contract_size = ntd_initial_fractal_contract_size
                        active_add_on_contracts = 0.0
                        pending_fractal_add_on = None
                        fill_price = long_trigger_price
                        signals.iloc[i] = 1
                        contracts.iloc[i] = _active_signed_contracts(position)
                        fill_prices.iloc[i] = fill_price
                        stop_loss_prices.iloc[i] = ntd_stop_price
                        fractal_position_side.iloc[i] = 1
                        pending_setup = {
                            "index": i,
                            "high": fill_price,
                            "low": ntd_stop_price,
                            "side": 1,
                            "reversal_armed": 0,
                            "source": "fractal",
                        }
                        pending_ntd_initial_long = None
                    return
                return

            if pending_setup is not None and int(pending_setup.get("index", -1)) != i:
                return

            trigger_side = 0
            if long_hit and not short_hit:
                trigger_side = 1
            elif short_hit and not long_hit:
                trigger_side = -1
            else:
                long_distance = abs(long_trigger_price - open_now)
                short_distance = abs(open_now - short_trigger_price)
                trigger_side = 1 if long_distance <= short_distance else -1

            if trigger_side == 1 and np.isfinite(long_trigger_price):
                ntd_stop_price = _ntd_initial_long_stop(i)
                if np.isfinite(ntd_stop_price) and ntd_stop_price < long_trigger_price:
                    prior_same_bar_setup = dict(pending_setup) if isinstance(pending_setup, dict) else None
                    position = 1
                    position_entry_bar = i
                    position_entry_source = "fractal"
                    active_base_contract_size = ntd_initial_fractal_contract_size
                    active_add_on_contracts = 0.0
                    pending_fractal_add_on = None
                    fill_price = long_trigger_price
                    signals.iloc[i] = 1
                    contracts.iloc[i] = _active_signed_contracts(position)
                    fill_prices.iloc[i] = fill_price
                    stop_loss_prices.iloc[i] = ntd_stop_price
                    fractal_position_side.iloc[i] = 1
                    pending_setup = {
                        "index": i,
                        "high": fill_price,
                        "low": ntd_stop_price,
                        "side": 1,
                        "reversal_armed": 0,
                        "source": "fractal",
                    }
                    if prior_same_bar_setup is not None and int(prior_same_bar_setup.get("side", 0)) == -1:
                        opposite_setup = prior_same_bar_setup
                pending_ntd_initial_long = None
                return

            if trigger_side == -1 and np.isfinite(short_trigger_price):
                ntd_stop_price = _ntd_initial_short_stop(i)
                if np.isfinite(ntd_stop_price) and ntd_stop_price > short_trigger_price:
                    prior_same_bar_setup = dict(pending_setup) if isinstance(pending_setup, dict) else None
                    position = -1
                    position_entry_bar = i
                    position_entry_source = "fractal"
                    active_base_contract_size = ntd_initial_fractal_contract_size
                    active_add_on_contracts = 0.0
                    pending_fractal_add_on = None
                    fill_price = short_trigger_price
                    signals.iloc[i] = -1
                    contracts.iloc[i] = _active_signed_contracts(position)
                    fill_prices.iloc[i] = fill_price
                    stop_loss_prices.iloc[i] = ntd_stop_price
                    fractal_position_side.iloc[i] = -1
                    pending_setup = {
                        "index": i,
                        "high": ntd_stop_price,
                        "low": fill_price,
                        "side": -1,
                        "reversal_armed": 0,
                        "source": "fractal",
                    }
                    if prior_same_bar_setup is not None and int(prior_same_bar_setup.get("side", 0)) == 1:
                        opposite_setup = prior_same_bar_setup
                pending_ntd_initial_short = None

        def _try_trigger_red_teeth_exit(i: int, high_now: float, low_now: float, close_now: float) -> bool:
            nonlocal position, position_entry_bar, reversal_position_active, reversal_stop_loss
            nonlocal position_entry_source, active_base_contract_size, active_add_on_contracts
            nonlocal pending_setup, opposite_setup, pending_ntd_initial_long, pending_ntd_initial_short, pending_fractal_add_on
            nonlocal red_teeth_gator_alignment_latched, red_teeth_gator_alignment_latched_entry_bar
            nonlocal red_teeth_unrealized_return_latched, red_teeth_unrealized_return_latched_entry_bar
            if (
                not self.red_teeth_profit_protection_enabled
                or position == 0
                or position_entry_bar < 0
                or not pd.notna(teeth.iloc[i])
            ):
                return False

            bars_in_position = i - position_entry_bar
            entry_price = (
                float(fill_prices.iloc[position_entry_bar])
                if np.isfinite(fill_prices.iloc[position_entry_bar])
                else float(data["open"].iloc[position_entry_bar])
            )
            favorable_excursion = (
                (high_now / entry_price) - 1.0
                if position == 1
                else (entry_price / low_now) - 1.0
            )
            min_unrealized_return = _annualized_volatility_scaled_return_threshold(
                self.red_teeth_profit_protection_min_unrealized_return,
                red_teeth_annualized_volatility.iloc[i],
                self.red_teeth_profit_protection_annualized_volatility_scaler,
            )
            if self.red_teeth_latch_min_unrealized_return:
                if red_teeth_unrealized_return_latched_entry_bar != position_entry_bar:
                    red_teeth_unrealized_return_latched = False
                    red_teeth_unrealized_return_latched_entry_bar = position_entry_bar
                if favorable_excursion >= min_unrealized_return:
                    red_teeth_unrealized_return_latched = True
                min_unrealized_gate_met = red_teeth_unrealized_return_latched
            else:
                min_unrealized_gate_met = favorable_excursion >= min_unrealized_return
            close_prev = float(data["close"].iloc[i - 1]) if i > 0 else close_now
            teeth_now = float(teeth.iloc[i])
            teeth_prev = float(teeth.iloc[i - 1]) if i > 0 and pd.notna(teeth.iloc[i - 1]) else teeth_now
            close_breached_teeth = (
                (position == 1 and close_now < teeth_now and close_prev >= teeth_prev)
                or (position == -1 and close_now > teeth_now and close_prev <= teeth_prev)
            )
            if self.red_teeth_profit_protection_require_gator_direction_alignment:
                if red_teeth_gator_alignment_latched_entry_bar != position_entry_bar:
                    red_teeth_gator_alignment_latched = False
                    red_teeth_gator_alignment_latched_entry_bar = position_entry_bar
                gator_alignment_now = (
                    pd.notna(lips.iloc[i])
                    and pd.notna(teeth.iloc[i])
                    and pd.notna(jaw.iloc[i])
                    and (
                        (
                            position == 1
                            and float(lips.iloc[i]) > float(teeth.iloc[i]) > float(jaw.iloc[i])
                        )
                        or (
                            position == -1
                            and float(lips.iloc[i]) < float(teeth.iloc[i]) < float(jaw.iloc[i])
                        )
                    )
                )
                if gator_alignment_now:
                    red_teeth_gator_alignment_latched = True
                if not red_teeth_gator_alignment_latched:
                    return False
            if not (
                bars_in_position > self.red_teeth_profit_protection_min_bars
                and min_unrealized_gate_met
                and close_breached_teeth
            ):
                return False

            signals.iloc[i] = 0
            contracts.iloc[i] = 0.0
            fill_prices.iloc[i] = close_now
            stop_loss_prices.iloc[i] = close_now
            exit_reason.iloc[i] = "Red Gator Teeth PP"
            position = 0
            position_entry_bar = -1
            reversal_position_active = False
            reversal_stop_loss = None
            position_entry_source = ""
            active_base_contract_size = 0.0
            active_add_on_contracts = 0.0
            pending_setup = None
            opposite_setup = None
            pending_ntd_initial_long = None
            pending_ntd_initial_short = None
            pending_fractal_add_on = None
            return True

        def _try_trigger_green_lips_exit(i: int, high_now: float, low_now: float, close_now: float) -> bool:
            nonlocal position, position_entry_bar, reversal_position_active, reversal_stop_loss
            nonlocal position_entry_source, active_base_contract_size, active_add_on_contracts
            nonlocal pending_setup, opposite_setup, pending_ntd_initial_long, pending_ntd_initial_short, pending_fractal_add_on
            nonlocal green_lips_gator_alignment_latched, green_lips_gator_alignment_latched_entry_bar
            nonlocal green_lips_unrealized_return_latched, green_lips_unrealized_return_latched_entry_bar
            if (
                not self.green_lips_profit_protection_enabled
                or position == 0
                or position_entry_bar < 0
                or not pd.notna(lips.iloc[i])
            ):
                return False

            bars_in_position = i - position_entry_bar
            entry_price = (
                float(fill_prices.iloc[position_entry_bar])
                if np.isfinite(fill_prices.iloc[position_entry_bar])
                else float(data["open"].iloc[position_entry_bar])
            )
            favorable_excursion = (
                (high_now / entry_price) - 1.0
                if position == 1
                else (entry_price / low_now) - 1.0
            )
            min_unrealized_return = _annualized_volatility_scaled_return_threshold(
                self.green_lips_profit_protection_min_unrealized_return,
                green_lips_annualized_volatility.iloc[i],
                self.green_lips_profit_protection_annualized_volatility_scaler,
            )
            if self.green_lips_latch_min_unrealized_return:
                if green_lips_unrealized_return_latched_entry_bar != position_entry_bar:
                    green_lips_unrealized_return_latched = False
                    green_lips_unrealized_return_latched_entry_bar = position_entry_bar
                if favorable_excursion >= min_unrealized_return:
                    green_lips_unrealized_return_latched = True
                min_unrealized_gate_met = green_lips_unrealized_return_latched
            else:
                min_unrealized_gate_met = favorable_excursion >= min_unrealized_return
            close_prev = float(data["close"].iloc[i - 1]) if i > 0 else close_now
            lips_now = float(lips.iloc[i])
            lips_prev = float(lips.iloc[i - 1]) if i > 0 and pd.notna(lips.iloc[i - 1]) else lips_now
            close_breached_lips = (
                (position == 1 and close_now < lips_now and close_prev >= lips_prev)
                or (position == -1 and close_now > lips_now and close_prev <= lips_prev)
            )
            if self.green_lips_profit_protection_require_gator_direction_alignment:
                if green_lips_gator_alignment_latched_entry_bar != position_entry_bar:
                    green_lips_gator_alignment_latched = False
                    green_lips_gator_alignment_latched_entry_bar = position_entry_bar
                gator_alignment_now = (
                    pd.notna(lips.iloc[i])
                    and pd.notna(teeth.iloc[i])
                    and pd.notna(jaw.iloc[i])
                    and (
                        (
                            position == 1
                            and float(lips.iloc[i]) > float(teeth.iloc[i]) > float(jaw.iloc[i])
                        )
                        or (
                            position == -1
                            and float(lips.iloc[i]) < float(teeth.iloc[i]) < float(jaw.iloc[i])
                        )
                    )
                )
                if gator_alignment_now:
                    green_lips_gator_alignment_latched = True
                if not green_lips_gator_alignment_latched:
                    return False
            if not (
                bars_in_position > self.green_lips_profit_protection_min_bars
                and min_unrealized_gate_met
                and close_breached_lips
            ):
                return False

            signals.iloc[i] = 0
            contracts.iloc[i] = 0.0
            fill_prices.iloc[i] = close_now
            stop_loss_prices.iloc[i] = close_now
            exit_reason.iloc[i] = "Green Gator Lips PP"
            position = 0
            position_entry_bar = -1
            reversal_position_active = False
            reversal_stop_loss = None
            position_entry_source = ""
            active_base_contract_size = 0.0
            active_add_on_contracts = 0.0
            pending_setup = None
            opposite_setup = None
            pending_ntd_initial_long = None
            pending_ntd_initial_short = None
            pending_fractal_add_on = None
            return True

        def _try_trigger_zones_exit(i: int, open_now: float, high_now: float, low_now: float) -> bool:
            nonlocal position, position_entry_bar, reversal_position_active, reversal_stop_loss
            nonlocal position_entry_source, active_base_contract_size, active_add_on_contracts
            nonlocal pending_setup, opposite_setup, pending_ntd_initial_long, pending_ntd_initial_short, pending_fractal_add_on
            nonlocal zones_profit_protection_active, zones_profit_protection_entry_bar, zones_profit_protection_stop_level
            nonlocal zones_profit_protection_stop_set_bar, zones_green_streak, zones_red_streak
            if (
                not self.zones_profit_protection_enabled
                or position == 0
                or position_entry_bar < 0
            ):
                return False

            if zones_profit_protection_entry_bar != position_entry_bar:
                zones_profit_protection_active = False
                zones_profit_protection_entry_bar = position_entry_bar
                zones_profit_protection_stop_level = np.nan
                zones_profit_protection_stop_set_bar = -1
                zones_green_streak = 0
                zones_red_streak = 0

            entry_price = (
                float(fill_prices.iloc[position_entry_bar])
                if np.isfinite(fill_prices.iloc[position_entry_bar])
                else float(data["open"].iloc[position_entry_bar])
            )
            favorable_excursion = (
                (high_now / entry_price) - 1.0
                if position == 1
                else (entry_price / low_now) - 1.0
            )
            bars_in_position = i - position_entry_bar
            min_unrealized_return = _annualized_volatility_scaled_return_threshold(
                self.zones_profit_protection_min_unrealized_return,
                zones_annualized_volatility.iloc[i],
                self.zones_profit_protection_annualized_volatility_scaler,
            )

            if position == 1:
                zones_green_streak = zones_green_streak + 1 if bool(zone_green.iloc[i]) else 0
                zones_red_streak = 0
            else:
                zones_red_streak = zones_red_streak + 1 if bool(zone_red.iloc[i]) else 0
                zones_green_streak = 0

            if (
                not zones_profit_protection_active
                and bars_in_position >= self.zones_profit_protection_min_bars
                and favorable_excursion >= min_unrealized_return
            ):
                streak = zones_green_streak if position == 1 else zones_red_streak
                if streak >= self.zones_profit_protection_min_same_color_bars:
                    zones_profit_protection_active = True
                    zones_profit_protection_stop_level = low_now if position == 1 else high_now
                    zones_profit_protection_stop_set_bar = i

            if (
                zones_profit_protection_active
                and np.isfinite(zones_profit_protection_stop_level)
                and i > zones_profit_protection_stop_set_bar
                and (
                    (position == 1 and low_now <= zones_profit_protection_stop_level)
                    or (position == -1 and high_now >= zones_profit_protection_stop_level)
                )
            ):
                stop_fill_price = (
                    min(open_now, zones_profit_protection_stop_level)
                    if position == 1
                    else max(open_now, zones_profit_protection_stop_level)
                )
                signals.iloc[i] = 0
                contracts.iloc[i] = 0.0
                fill_prices.iloc[i] = stop_fill_price
                stop_loss_prices.iloc[i] = stop_fill_price
                exit_reason.iloc[i] = "Williams Zones PP"
                position = 0
                position_entry_bar = -1
                reversal_position_active = False
                reversal_stop_loss = None
                position_entry_source = ""
                active_base_contract_size = 0.0
                active_add_on_contracts = 0.0
                pending_setup = None
                opposite_setup = None
                pending_ntd_initial_long = None
                pending_ntd_initial_short = None
                pending_fractal_add_on = None
                zones_profit_protection_active = False
                zones_profit_protection_entry_bar = -1
                zones_profit_protection_stop_level = np.nan
                zones_profit_protection_stop_set_bar = -1
                zones_green_streak = 0
                zones_red_streak = 0
                return True

            if zones_profit_protection_active:
                zones_profit_protection_stop_level = low_now if position == 1 else high_now
                zones_profit_protection_stop_set_bar = i
            return False

        def _try_trigger_peak_drawdown_exit(i: int, close_now: float, high_now: float, low_now: float) -> bool:
            nonlocal position, position_entry_bar, reversal_position_active, reversal_stop_loss
            nonlocal position_entry_source, active_base_contract_size, active_add_on_contracts
            nonlocal pending_setup, opposite_setup, pending_ntd_initial_long, pending_ntd_initial_short, pending_fractal_add_on
            nonlocal peak_drawdown_entry_bar, peak_favorable_return
            if (
                position == 0
                or position_entry_bar < 0
                or (not self.peak_drawdown_exit_enabled)
                or self.peak_drawdown_exit_pct <= 0
            ):
                peak_drawdown_entry_bar = -1
                peak_favorable_return = 0.0
                return False

            if peak_drawdown_entry_bar != position_entry_bar:
                peak_drawdown_entry_bar = position_entry_bar
                peak_favorable_return = 0.0

            entry_price = (
                float(fill_prices.iloc[position_entry_bar])
                if np.isfinite(fill_prices.iloc[position_entry_bar])
                else float(data["open"].iloc[position_entry_bar])
            )
            if not np.isfinite(entry_price) or entry_price <= 0 or low_now <= 0 or close_now <= 0:
                return False

            favorable_return = (
                (high_now / entry_price) - 1.0
                if position == 1
                else (entry_price / low_now) - 1.0
            )
            if np.isfinite(favorable_return):
                peak_favorable_return = max(peak_favorable_return, favorable_return)

            current_return = (
                (close_now / entry_price) - 1.0
                if position == 1
                else (entry_price / close_now) - 1.0
            )
            drawdown_from_peak = peak_favorable_return - current_return
            max_allowed_drawdown = _annualized_volatility_scaled_return_threshold(
                self.peak_drawdown_exit_pct,
                peak_drawdown_annualized_volatility.iloc[i],
                self.peak_drawdown_exit_annualized_volatility_scaler,
            )
            if not np.isfinite(max_allowed_drawdown) or drawdown_from_peak <= max_allowed_drawdown:
                return False

            signals.iloc[i] = 0
            contracts.iloc[i] = 0.0
            fill_prices.iloc[i] = close_now
            stop_loss_prices.iloc[i] = close_now
            exit_reason.iloc[i] = "Peak Drawdown PP"
            position = 0
            position_entry_bar = -1
            reversal_position_active = False
            reversal_stop_loss = None
            position_entry_source = ""
            active_base_contract_size = 0.0
            active_add_on_contracts = 0.0
            pending_setup = None
            opposite_setup = None
            pending_ntd_initial_long = None
            pending_ntd_initial_short = None
            pending_fractal_add_on = None
            peak_drawdown_entry_bar = -1
            peak_favorable_return = 0.0
            return True

        def _try_trigger_sigma_move_exit(i: int, close_now: float, high_now: float, low_now: float) -> bool:
            nonlocal position, position_entry_bar, reversal_position_active, reversal_stop_loss
            nonlocal position_entry_source, active_base_contract_size, active_add_on_contracts
            nonlocal pending_setup, opposite_setup, pending_ntd_initial_long, pending_ntd_initial_short, pending_fractal_add_on
            if (
                position == 0
                or position_entry_bar < 0
                or (not self.sigma_move_profit_protection_enabled)
            ):
                return False
            baseline = float(sigma_move_rolling_mean.iloc[i]) if pd.notna(sigma_move_rolling_mean.iloc[i]) else np.nan
            dispersion = float(sigma_move_rolling_std.iloc[i]) if pd.notna(sigma_move_rolling_std.iloc[i]) else np.nan
            if not np.isfinite(baseline) or not np.isfinite(dispersion) or dispersion <= 0:
                return False
            sigma_band = self.sigma_move_profit_protection_sigma * dispersion
            reached_sigma_move = (
                high_now >= (baseline + sigma_band)
                if position > 0
                else low_now <= (baseline - sigma_band)
            )
            if not reached_sigma_move:
                return False
            signals.iloc[i] = 0
            contracts.iloc[i] = 0.0
            fill_prices.iloc[i] = close_now
            stop_loss_prices.iloc[i] = close_now
            exit_reason.iloc[i] = "Sigma Move PP"
            position = 0
            position_entry_bar = -1
            reversal_position_active = False
            reversal_stop_loss = None
            position_entry_source = ""
            active_base_contract_size = 0.0
            active_add_on_contracts = 0.0
            pending_setup = None
            opposite_setup = None
            pending_ntd_initial_long = None
            pending_ntd_initial_short = None
            pending_fractal_add_on = None
            return True

        def _try_trigger_underlying_gain_exit(
            i: int,
            open_now: float,
            high_now: float | None = None,
            low_now: float | None = None,
        ) -> bool:
            nonlocal position, position_entry_bar, reversal_position_active, reversal_stop_loss
            nonlocal position_entry_source, active_base_contract_size, active_add_on_contracts
            nonlocal pending_setup, opposite_setup, pending_ntd_initial_long, pending_ntd_initial_short, pending_fractal_add_on
            if position == 0 or position_entry_bar < 0 or self.close_on_underlying_gain_pct <= 0:
                return False
            high_eval = float(high_now) if high_now is not None else float(open_now)
            low_eval = float(low_now) if low_now is not None else float(open_now)

            entry_price = (
                float(fill_prices.iloc[position_entry_bar])
                if np.isfinite(fill_prices.iloc[position_entry_bar])
                else float(data["open"].iloc[position_entry_bar])
            )
            if not np.isfinite(entry_price) or entry_price <= 0 or high_eval <= 0 or low_eval <= 0:
                return False

            target_price = (
                entry_price * (1.0 + self.close_on_underlying_gain_pct)
                if position == 1
                else entry_price / (1.0 + self.close_on_underlying_gain_pct)
            )
            if not np.isfinite(target_price) or target_price <= 0:
                return False

            target_hit = (
                high_eval >= target_price
                if position == 1
                else low_eval <= target_price
            )
            if not target_hit:
                return False

            signals.iloc[i] = 0
            contracts.iloc[i] = 0.0
            fill_price = (
                target_price
                if (
                    (position == 1 and open_now <= target_price)
                    or (position == -1 and open_now >= target_price)
                )
                else open_now
            )
            fill_prices.iloc[i] = fill_price
            stop_loss_prices.iloc[i] = fill_price
            exit_reason.iloc[i] = "Underlying Gain Target"
            position = 0
            position_entry_bar = -1
            reversal_position_active = False
            reversal_stop_loss = None
            position_entry_source = ""
            active_base_contract_size = 0.0
            active_add_on_contracts = 0.0
            pending_setup = None
            opposite_setup = None
            pending_ntd_initial_long = None
            pending_ntd_initial_short = None
            pending_fractal_add_on = None
            return True

        for i in range(2, len(data)):
            position_at_bar_open = position
            add_on_fill_candidate_side = 0
            add_on_fill_candidate_price = np.nan
            open_now = float(data["open"].iloc[i])
            high_now = float(data["high"].iloc[i])
            low_now = float(data["low"].iloc[i])
            close_now = float(data["close"].iloc[i])
            if position == 0:
                opposite_setup = None
                pending_filtered_opposite_close_setup = None
                position_entry_bar = -1
                pending_fractal_add_on = None
            midpoint_now = (high_now + low_now) / 2.0
            ao_now = float(ao.iloc[i]) if pd.notna(ao.iloc[i]) else np.nan
            ao_prev = float(ao.iloc[i - 1]) if pd.notna(ao.iloc[i - 1]) else np.nan

            confirmed_fractal_bar = i - 2
            if confirmed_fractal_bar >= 0:
                if bool(fractals.iloc[confirmed_fractal_bar]["up_fractal"]):
                    pending_ntd_initial_long = None
                if bool(fractals.iloc[confirmed_fractal_bar]["down_fractal"]):
                    pending_ntd_initial_short = None
                if (
                    bool(fractals.iloc[confirmed_fractal_bar]["down_fractal"])
                    and pd.notna(teeth.iloc[confirmed_fractal_bar])
                    and float(data["low"].iloc[confirmed_fractal_bar]) < float(teeth.iloc[confirmed_fractal_bar])
                ):
                    latest_bearish_fractal_under_teeth_bar = confirmed_fractal_bar
                    latest_bearish_fractal_under_teeth_price = float(data["low"].iloc[confirmed_fractal_bar])
                if (
                    bool(fractals.iloc[confirmed_fractal_bar]["up_fractal"])
                    and pd.notna(teeth.iloc[confirmed_fractal_bar])
                    and float(data["high"].iloc[confirmed_fractal_bar]) > float(teeth.iloc[confirmed_fractal_bar])
                ):
                    latest_bullish_fractal_above_teeth_bar = confirmed_fractal_bar
                    latest_bullish_fractal_above_teeth_price = float(data["high"].iloc[confirmed_fractal_bar])
                if (
                    ntd_initial_fractal_active
                    and bool(fractals.iloc[confirmed_fractal_bar]["up_fractal"])
                    and pd.notna(teeth.iloc[confirmed_fractal_bar])
                    and float(data["high"].iloc[confirmed_fractal_bar]) > float(teeth.iloc[confirmed_fractal_bar])
                    and bool(sleeping_gator.iloc[confirmed_fractal_bar] or gator_waking_up.iloc[confirmed_fractal_bar])
                    and bool(price_ranging.iloc[confirmed_fractal_bar])
                ):
                    pending_ntd_initial_long = {
                        "fractal_bar": confirmed_fractal_bar,
                        "trigger_price": float(data["high"].iloc[confirmed_fractal_bar]),
                        "placed_bar": i,
                    }
                if (
                    ntd_initial_fractal_active
                    and bool(fractals.iloc[confirmed_fractal_bar]["down_fractal"])
                    and pd.notna(teeth.iloc[confirmed_fractal_bar])
                    and float(data["low"].iloc[confirmed_fractal_bar]) < float(teeth.iloc[confirmed_fractal_bar])
                    and bool(sleeping_gator.iloc[confirmed_fractal_bar] or gator_waking_up.iloc[confirmed_fractal_bar])
                    and bool(price_ranging.iloc[confirmed_fractal_bar])
                ):
                    pending_ntd_initial_short = {
                        "fractal_bar": confirmed_fractal_bar,
                        "trigger_price": float(data["low"].iloc[confirmed_fractal_bar]),
                        "placed_bar": i,
                    }
                if (
                    fractal_add_on_contract_size > 0
                    and position != 0
                    and position_entry_source in {"1w", "fractal", "reversal"}
                ):
                    long_alignment_at_fractal = (
                        pd.notna(lips.iloc[confirmed_fractal_bar])
                        and pd.notna(teeth.iloc[confirmed_fractal_bar])
                        and pd.notna(jaw.iloc[confirmed_fractal_bar])
                        and float(lips.iloc[confirmed_fractal_bar]) > float(teeth.iloc[confirmed_fractal_bar]) > float(jaw.iloc[confirmed_fractal_bar])
                    )
                    short_alignment_at_fractal = (
                        pd.notna(lips.iloc[confirmed_fractal_bar])
                        and pd.notna(teeth.iloc[confirmed_fractal_bar])
                        and pd.notna(jaw.iloc[confirmed_fractal_bar])
                        and float(lips.iloc[confirmed_fractal_bar]) < float(teeth.iloc[confirmed_fractal_bar]) < float(jaw.iloc[confirmed_fractal_bar])
                    )
                    long_alignment_now = (
                        pd.notna(lips.iloc[i])
                        and pd.notna(teeth.iloc[i])
                        and pd.notna(jaw.iloc[i])
                        and float(lips.iloc[i]) > float(teeth.iloc[i]) > float(jaw.iloc[i])
                    )
                    short_alignment_now = (
                        pd.notna(lips.iloc[i])
                        and pd.notna(teeth.iloc[i])
                        and pd.notna(jaw.iloc[i])
                        and float(lips.iloc[i]) < float(teeth.iloc[i]) < float(jaw.iloc[i])
                    )
                    if (
                        position == 1
                        and bool(fractals.iloc[confirmed_fractal_bar]["up_fractal"])
                        and float(data["high"].iloc[confirmed_fractal_bar]) > float(teeth.iloc[confirmed_fractal_bar])
                        and (long_alignment_at_fractal or long_alignment_now)
                    ):
                        pending_fractal_add_on = {
                            "side": 1,
                            "trigger_price": float(data["high"].iloc[confirmed_fractal_bar]),
                            "placed_bar": i,
                        }
                    elif (
                        position == -1
                        and bool(fractals.iloc[confirmed_fractal_bar]["down_fractal"])
                        and float(data["low"].iloc[confirmed_fractal_bar]) < float(teeth.iloc[confirmed_fractal_bar])
                        and (short_alignment_at_fractal or short_alignment_now)
                    ):
                        pending_fractal_add_on = {
                            "side": -1,
                            "trigger_price": float(data["low"].iloc[confirmed_fractal_bar]),
                            "placed_bar": i,
                        }

            if (
                pending_fractal_add_on is not None
                and position != 0
                and int(pending_fractal_add_on["side"]) == position
                and (
                    i > int(pending_fractal_add_on["placed_bar"])
                    or (
                        i == int(pending_fractal_add_on["placed_bar"])
                        and position_entry_source == "fractal"
                    )
                )
            ):
                add_trigger = float(pending_fractal_add_on["trigger_price"])
                add_triggered = (
                    (position == 1 and high_now >= add_trigger)
                    or (position == -1 and low_now <= add_trigger)
                )
                if add_triggered:
                    can_apply_now = (
                        position == position_at_bar_open
                        and str(exit_reason.iloc[i]).strip() == ""
                        and (int(np.sign(signals.iloc[i])) in (0, position))
                        and not np.isfinite(fill_prices.iloc[i])
                    )
                    if can_apply_now:
                        active_add_on_contracts += fractal_add_on_contract_size
                        add_on_fractal_fill_side.iloc[i] = int(position)
                        fill_prices.iloc[i] = float(add_trigger)
                        if int(np.sign(signals.iloc[i])) == 0:
                            signals.iloc[i] = int(position)
                        contracts.iloc[i] = _active_signed_contracts(position)
                    else:
                        add_on_fill_candidate_side = int(position)
                        add_on_fill_candidate_price = float(add_trigger)
                    pending_fractal_add_on = None

            bullish_divergence_ok = self.divergence_filter_bars <= 0
            bearish_divergence_ok = self.divergence_filter_bars <= 0
            if self.divergence_filter_bars > 0 and pd.notna(ao_now):
                left = max(0, i - self.divergence_filter_bars)
                for j in range(left, i):
                    if pd.isna(ao.iloc[j]):
                        continue
                    if float(data["low"].iloc[j]) > low_now and float(ao.iloc[j]) < ao_now:
                        bullish_divergence_ok = True
                    if float(data["high"].iloc[j]) < high_now and float(ao.iloc[j]) > ao_now:
                        bearish_divergence_ok = True
                    if bullish_divergence_ok and bearish_divergence_ok:
                        break

            bullish_alligator_ok = (
                pd.notna(lips.iloc[i])
                and pd.notna(teeth.iloc[i])
                and pd.notna(jaw.iloc[i])
                and lips.iloc[i] < teeth.iloc[i] < jaw.iloc[i]
                and midpoint_now < float(teeth.iloc[i])
            )
            bearish_alligator_ok = (
                pd.notna(lips.iloc[i])
                and pd.notna(teeth.iloc[i])
                and pd.notna(jaw.iloc[i])
                and lips.iloc[i] > teeth.iloc[i] > jaw.iloc[i]
                and midpoint_now > float(teeth.iloc[i])
            )
            bullish_price_action_ok = close_now > open_now and low_now < float(data["low"].iloc[i - 1])
            bearish_price_action_ok = close_now < open_now and high_now > float(data["high"].iloc[i - 1])
            bullish_ao_ok = pd.notna(ao_now) and pd.notna(ao_prev) and ao_now < ao_prev
            bearish_ao_ok = pd.notna(ao_now) and pd.notna(ao_prev) and ao_now > ao_prev
            gator_lines_now = np.array([jaw.iloc[i], teeth.iloc[i], lips.iloc[i]], dtype="float64")
            finite_gator_lines_now = gator_lines_now[np.isfinite(gator_lines_now)]
            gator_top_now = (
                float(np.max(finite_gator_lines_now)) if finite_gator_lines_now.size > 0 else np.nan
            )
            gator_bottom_now = (
                float(np.min(finite_gator_lines_now)) if finite_gator_lines_now.size > 0 else np.nan
            )
            bullish_extreme_outside_gator_ok = np.isfinite(gator_bottom_now) and high_now <= gator_bottom_now
            bearish_extreme_outside_gator_ok = np.isfinite(gator_top_now) and low_now >= gator_top_now

            bullish_candidate = bullish_price_action_ok and bullish_ao_ok
            bearish_candidate = bearish_price_action_ok and bearish_ao_ok
            gator_open_strength_ok = True
            if gator_open_filter_enabled:
                threshold_now = float(gator_open_threshold.iloc[i])
                gator_open_strength_ok = (
                    np.isfinite(threshold_now)
                    and np.isfinite(float(gator_range.iloc[i]))
                    and float(gator_range.iloc[i]) > threshold_now
                )
            bullish_armed = (
                bullish_candidate
                and bullish_alligator_ok
                and bullish_divergence_ok
                and bullish_extreme_outside_gator_ok
                and gator_open_strength_ok
            )
            bearish_armed = (
                bearish_candidate
                and bearish_alligator_ok
                and bearish_divergence_ok
                and bearish_extreme_outside_gator_ok
                and gator_open_strength_ok
            )

            if bullish_armed:
                first_setup_marker_side.iloc[i] = 1
            elif bearish_armed:
                first_setup_marker_side.iloc[i] = -1
            elif bullish_candidate:
                first_setup_marker_side.iloc[i] = 1
                if not bullish_alligator_ok:
                    first_ignored_reason.iloc[i] = "gator_closed_canceled"
                elif not gator_open_strength_ok:
                    first_ignored_reason.iloc[i] = "gator_open_percentile_filter"
                else:
                    first_ignored_reason.iloc[i] = "ao_divergence_filter"
            elif bearish_candidate:
                first_setup_marker_side.iloc[i] = -1
                if not bearish_alligator_ok:
                    first_ignored_reason.iloc[i] = "gator_closed_canceled"
                elif not gator_open_strength_ok:
                    first_ignored_reason.iloc[i] = "gator_open_percentile_filter"
                else:
                    first_ignored_reason.iloc[i] = "ao_divergence_filter"

            if position == 0 and pending_setup is None:
                if bullish_armed:
                    pending_setup = {
                        "index": i,
                        "high": high_now,
                        "low": low_now,
                        "side": 1,
                        "reversal_armed": 0,
                        "source": "1w",
                    }
                    first_setup_side.iloc[i] = 1
                elif bearish_armed:
                    pending_setup = {
                        "index": i,
                        "high": high_now,
                        "low": low_now,
                        "side": -1,
                        "reversal_armed": 0,
                        "source": "1w",
                    }
                    first_setup_side.iloc[i] = -1
            elif (
                position == 0
                and pending_setup is not None
                and bullish_armed
                and int(pending_setup.get("side", 0)) == 1
            ):
                pending_low = float(pending_setup.get("low", low_now))
                if low_now < pending_low:
                    pending_setup = {
                        "index": i,
                        "high": high_now,
                        "low": low_now,
                        "side": 1,
                        "reversal_armed": 0,
                        "source": "1w",
                    }
                    first_setup_side.iloc[i] = 1
                elif first_ignored_reason.iloc[i] == "":
                    first_ignored_reason.iloc[i] = "weaker_than_active_setup"
            elif (
                position == 0
                and pending_setup is not None
                and bearish_armed
                and int(pending_setup.get("side", 0)) == -1
            ):
                pending_high = float(pending_setup.get("high", high_now))
                if high_now > pending_high:
                    pending_setup = {
                        "index": i,
                        "high": high_now,
                        "low": low_now,
                        "side": -1,
                        "reversal_armed": 0,
                        "source": "1w",
                    }
                    first_setup_side.iloc[i] = -1
                elif first_ignored_reason.iloc[i] == "":
                    first_ignored_reason.iloc[i] = "weaker_than_active_setup"
            elif (
                position == 1
                and bearish_armed
            ):
                should_replace_opposite = True
                if opposite_setup is not None and int(opposite_setup.get("side", 0)) == -1:
                    active_opposite_high = float(opposite_setup.get("high", high_now))
                    should_replace_opposite = high_now > active_opposite_high
                    if not should_replace_opposite and first_ignored_reason.iloc[i] == "":
                        first_ignored_reason.iloc[i] = "weaker_than_active_setup"
                if should_replace_opposite:
                    opposite_setup = {"index": i, "high": high_now, "low": low_now, "side": -1, "reversal_armed": 0, "source": "1w"}
                    first_setup_side.iloc[i] = -1
            elif (
                position == -1
                and bullish_armed
            ):
                should_replace_opposite = True
                if opposite_setup is not None and int(opposite_setup.get("side", 0)) == 1:
                    active_opposite_low = float(opposite_setup.get("low", low_now))
                    should_replace_opposite = low_now < active_opposite_low
                    if not should_replace_opposite and first_ignored_reason.iloc[i] == "":
                        first_ignored_reason.iloc[i] = "weaker_than_active_setup"
                if should_replace_opposite:
                    opposite_setup = {"index": i, "high": high_now, "low": low_now, "side": 1, "reversal_armed": 0, "source": "1w"}
                    first_setup_side.iloc[i] = 1
            elif position == 1 and bullish_armed and first_ignored_reason.iloc[i] == "":
                first_ignored_reason.iloc[i] = "weaker_than_active_setup"
            elif position == -1 and bearish_armed and first_ignored_reason.iloc[i] == "":
                first_ignored_reason.iloc[i] = "weaker_than_active_setup"

            opposite_filtered_reason = ""
            if position == 1 and bearish_candidate and not bearish_armed:
                opposite_filtered_reason = str(first_ignored_reason.iloc[i])
            elif position == -1 and bullish_candidate and not bullish_armed:
                opposite_filtered_reason = str(first_ignored_reason.iloc[i])

            filtered_close_enabled = False
            filtered_close_reason = ""
            filtered_close_min_unrealized_return = 0.0
            if opposite_filtered_reason == "ao_divergence_filter" and self.allow_close_on_1w_d:
                filtered_close_enabled = True
                filtered_close_reason = "1W-D Opposite Close"
                filtered_close_min_unrealized_return = self.allow_close_on_1w_d_min_unrealized_return
            elif opposite_filtered_reason == "gator_closed_canceled" and self.allow_close_on_1w_a:
                filtered_close_enabled = True
                filtered_close_reason = "1W-A Opposite Close"
                filtered_close_min_unrealized_return = self.allow_close_on_1w_a_min_unrealized_return

            if filtered_close_enabled:
                filtered_setup_side = -1 if position > 0 else 1
                should_replace_filtered_close = True
                if (
                    pending_filtered_opposite_close_setup is not None
                    and int(pending_filtered_opposite_close_setup.get("side", 0)) == filtered_setup_side
                ):
                    if filtered_setup_side > 0:
                        active_low = float(pending_filtered_opposite_close_setup.get("low", low_now))
                        should_replace_filtered_close = low_now < active_low
                    else:
                        active_high = float(pending_filtered_opposite_close_setup.get("high", high_now))
                        should_replace_filtered_close = high_now > active_high
                if should_replace_filtered_close:
                    pending_filtered_opposite_close_setup = {
                        "index": i,
                        "high": high_now,
                        "low": low_now,
                        "side": filtered_setup_side,
                        "reason": filtered_close_reason,
                        "min_unrealized_return": float(filtered_close_min_unrealized_return),
                    }

            if _try_trigger_underlying_gain_exit(i, open_now, high_now, low_now):
                continue
            if _try_trigger_red_teeth_exit(i, high_now, low_now, close_now):
                continue
            if _try_trigger_green_lips_exit(i, high_now, low_now, close_now):
                continue
            if _try_trigger_zones_exit(i, open_now, high_now, low_now):
                continue
            if _try_trigger_peak_drawdown_exit(i, close_now, high_now, low_now):
                continue
            if _try_trigger_sigma_move_exit(i, close_now, high_now, low_now):
                continue

            if pending_filtered_opposite_close_setup is not None and position != 0:
                filtered_setup_index = int(pending_filtered_opposite_close_setup.get("index", -1))
                if filtered_setup_index < position_entry_bar:
                    pending_filtered_opposite_close_setup = None
                if pending_filtered_opposite_close_setup is not None:
                    if i == int(pending_filtered_opposite_close_setup["index"]):
                        signals.iloc[i] = int(position)
                        contracts.iloc[i] = _active_signed_contracts(position)
                    else:
                        setup_high = float(pending_filtered_opposite_close_setup["high"])
                        setup_low = float(pending_filtered_opposite_close_setup["low"])
                        setup_side = int(pending_filtered_opposite_close_setup["side"])
                        setup_reason = str(pending_filtered_opposite_close_setup.get("reason", ""))
                        min_unrealized_return = float(
                            pending_filtered_opposite_close_setup.get("min_unrealized_return", 0.0)
                        )
                        gate_met = _opposite_close_unrealized_gate_met(
                            position,
                            position_entry_bar,
                            high_now,
                            low_now,
                            min_unrealized_return,
                        )
                        if position == 1 and setup_side == -1:
                            if low_now <= setup_low and gate_met:
                                position = 0
                                position_entry_bar = -1
                                position_entry_source = ""
                                active_base_contract_size = 0.0
                                active_add_on_contracts = 0.0
                                pending_fractal_add_on = None
                                pending_ntd_initial_long = None
                                pending_ntd_initial_short = None
                                pending_setup = None
                                opposite_setup = None
                                reversal_position_active = False
                                reversal_stop_loss = None
                                signals.iloc[i] = 0
                                contracts.iloc[i] = 0.0
                                fill_prices.iloc[i] = setup_low
                                stop_loss_prices.iloc[i] = setup_low
                                exit_reason.iloc[i] = setup_reason
                                pending_filtered_opposite_close_setup = None
                                continue
                            if high_now >= setup_high:
                                pending_filtered_opposite_close_setup = None
                        elif position == -1 and setup_side == 1:
                            if high_now >= setup_high and gate_met:
                                position = 0
                                position_entry_bar = -1
                                position_entry_source = ""
                                active_base_contract_size = 0.0
                                active_add_on_contracts = 0.0
                                pending_fractal_add_on = None
                                pending_ntd_initial_long = None
                                pending_ntd_initial_short = None
                                pending_setup = None
                                opposite_setup = None
                                reversal_position_active = False
                                reversal_stop_loss = None
                                signals.iloc[i] = 0
                                contracts.iloc[i] = 0.0
                                fill_prices.iloc[i] = setup_high
                                stop_loss_prices.iloc[i] = setup_high
                                exit_reason.iloc[i] = setup_reason
                                pending_filtered_opposite_close_setup = None
                                continue
                            if low_now <= setup_low:
                                pending_filtered_opposite_close_setup = None

            if opposite_setup is not None and position != 0:
                setup_index = int(opposite_setup.get("index", -1))
                if setup_index < position_entry_bar:
                    opposite_setup = None
                if opposite_setup is not None:
                    if i == int(opposite_setup["index"]):
                        if position == 1:
                            signals.iloc[i] = 1
                            contracts.iloc[i] = _active_signed_contracts(position)
                        else:
                            signals.iloc[i] = -1
                            contracts.iloc[i] = _active_signed_contracts(position)
                    else:
                        setup_high = float(opposite_setup["high"])
                        setup_low = float(opposite_setup["low"])
                        setup_side = int(opposite_setup["side"])
                        if position == 1 and setup_side == -1:
                            if low_now <= setup_low:
                                if reversal_position_active:
                                    exit_reason.iloc[i] = "1W-R Flattened by Opposite 1W"
                                pending_setup = dict(opposite_setup)
                                opposite_setup = None
                                reversal_position_active = False
                                reversal_stop_loss = None
                                if self.only_trade_1w_reversals:
                                    position = 0
                                    position_entry_bar = -1
                                    position_entry_source = ""
                                    active_base_contract_size = 0.0
                                    active_add_on_contracts = 0.0
                                    pending_fractal_add_on = None
                                    signals.iloc[i] = 0
                                    contracts.iloc[i] = 0.0
                                    fill_prices.iloc[i] = setup_low
                                    stop_loss_prices.iloc[i] = setup_low
                                else:
                                    position = -1
                                    position_entry_bar = i
                                    position_entry_source = "1w"
                                    active_base_contract_size = first_wiseman_contract_size
                                    active_add_on_contracts = 0.0
                                    pending_fractal_add_on = None
                                    signals.iloc[i] = -1
                                    contracts.iloc[i] = _active_signed_contracts(position)
                                    fill_prices.iloc[i] = setup_low
                                    stop_loss_prices.iloc[i] = setup_high
                                if high_now >= setup_high and not self.only_trade_1w_reversals:
                                    _append_intrabar_event(
                                        i,
                                        "entry",
                                        -1,
                                        setup_low,
                                        first_wiseman_contract_size,
                                        "Bearish 1W",
                                        setup_high,
                                    )
                                    _append_intrabar_event(
                                        i,
                                        "exit",
                                        1,
                                        setup_high,
                                        0.0,
                                        "Strategy Stop Loss Bearish 1W",
                                    )
                                    reversal_window_open = setup_index >= 0 and i >= setup_index + 3
                                    if reversal_window_open:
                                        reversal_stop = float(data["low"].iloc[setup_index : i + 1].min())
                                        _append_intrabar_event(
                                            i,
                                            "entry",
                                            1,
                                            setup_high,
                                            first_wiseman_contract_size,
                                            "Bullish 1W-R",
                                            reversal_stop,
                                        )
                                        position = 1
                                        position_entry_bar = i
                                        position_entry_source = "reversal"
                                        active_base_contract_size = first_wiseman_contract_size
                                        active_add_on_contracts = 0.0
                                        pending_fractal_add_on = None
                                        signals.iloc[i] = 1
                                        contracts.iloc[i] = _active_signed_contracts(position)
                                        fill_prices.iloc[i] = setup_high
                                        first_reversal_side.iloc[i] = 1
                                        reversal_stop_loss = reversal_stop
                                        stop_loss_prices.iloc[i] = reversal_stop_loss
                                        reversal_position_active = True
                                    else:
                                        if setup_index >= 0 and first_ignored_reason.iloc[setup_index] == "":
                                            first_ignored_reason.iloc[setup_index] = "same_bar_stop_before_reversal_window"
                                        position = 0
                                        position_entry_bar = -1
                                        position_entry_source = ""
                                        active_base_contract_size = 0.0
                                        active_add_on_contracts = 0.0
                                        pending_fractal_add_on = None
                                        pending_ntd_initial_long = None
                                        pending_ntd_initial_short = None
                                        signals.iloc[i] = 0
                                        contracts.iloc[i] = 0.0
                                        fill_prices.iloc[i] = setup_high
                                        stop_loss_prices.iloc[i] = setup_high
                                        pending_setup = None
                            if opposite_setup is not None and high_now >= setup_high:
                                setup_bar = int(opposite_setup.get("index", -1))
                                if setup_bar >= 0 and first_ignored_reason.iloc[setup_bar] == "":
                                    first_ignored_reason.iloc[setup_bar] = "invalidation_before_trigger"
                                opposite_setup = None
                        elif position == -1 and setup_side == 1:
                            if high_now >= setup_high:
                                if reversal_position_active:
                                    exit_reason.iloc[i] = "1W-R Flattened by Opposite 1W"
                                pending_setup = dict(opposite_setup)
                                opposite_setup = None
                                reversal_position_active = False
                                reversal_stop_loss = None
                                if self.only_trade_1w_reversals:
                                    position = 0
                                    position_entry_bar = -1
                                    position_entry_source = ""
                                    active_base_contract_size = 0.0
                                    active_add_on_contracts = 0.0
                                    pending_fractal_add_on = None
                                    signals.iloc[i] = 0
                                    contracts.iloc[i] = 0.0
                                    fill_prices.iloc[i] = setup_high
                                    stop_loss_prices.iloc[i] = setup_high
                                else:
                                    position = 1
                                    position_entry_bar = i
                                    position_entry_source = "1w"
                                    active_base_contract_size = first_wiseman_contract_size
                                    active_add_on_contracts = 0.0
                                    pending_fractal_add_on = None
                                    signals.iloc[i] = 1
                                    contracts.iloc[i] = _active_signed_contracts(position)
                                    fill_prices.iloc[i] = setup_high
                                    stop_loss_prices.iloc[i] = setup_low
                                if low_now <= setup_low and not self.only_trade_1w_reversals:
                                    _append_intrabar_event(
                                        i,
                                        "entry",
                                        1,
                                        setup_high,
                                        first_wiseman_contract_size,
                                        "Bullish 1W",
                                        setup_low,
                                    )
                                    _append_intrabar_event(
                                        i,
                                        "exit",
                                        -1,
                                        setup_low,
                                        0.0,
                                        "Strategy Stop Loss Bullish 1W",
                                    )
                                    reversal_window_open = setup_index >= 0 and i >= setup_index + 3
                                    if reversal_window_open:
                                        reversal_stop = float(data["high"].iloc[setup_index : i + 1].max())
                                        _append_intrabar_event(
                                            i,
                                            "entry",
                                            -1,
                                            setup_low,
                                            first_wiseman_contract_size,
                                            "Bearish 1W-R",
                                            reversal_stop,
                                        )
                                        position = -1
                                        position_entry_bar = i
                                        position_entry_source = "reversal"
                                        active_base_contract_size = first_wiseman_contract_size
                                        active_add_on_contracts = 0.0
                                        pending_fractal_add_on = None
                                        signals.iloc[i] = -1
                                        contracts.iloc[i] = _active_signed_contracts(position)
                                        fill_prices.iloc[i] = setup_low
                                        first_reversal_side.iloc[i] = -1
                                        reversal_stop_loss = reversal_stop
                                        stop_loss_prices.iloc[i] = reversal_stop_loss
                                        reversal_position_active = True
                                    else:
                                        if setup_index >= 0 and first_ignored_reason.iloc[setup_index] == "":
                                            first_ignored_reason.iloc[setup_index] = "same_bar_stop_before_reversal_window"
                                        position = 0
                                        position_entry_bar = -1
                                        position_entry_source = ""
                                        active_base_contract_size = 0.0
                                        active_add_on_contracts = 0.0
                                        pending_fractal_add_on = None
                                        pending_ntd_initial_long = None
                                        pending_ntd_initial_short = None
                                        signals.iloc[i] = 0
                                        contracts.iloc[i] = 0.0
                                        fill_prices.iloc[i] = setup_low
                                        stop_loss_prices.iloc[i] = setup_low
                                        pending_setup = None
                            if opposite_setup is not None and low_now <= setup_low:
                                setup_bar = int(opposite_setup.get("index", -1))
                                if setup_bar >= 0 and first_ignored_reason.iloc[setup_bar] == "":
                                    first_ignored_reason.iloc[setup_bar] = "invalidation_before_trigger"
                                opposite_setup = None

            _try_trigger_ntd_initial_entry(i, open_now, high_now, low_now)

            if pending_setup is not None and position == 0:
                if i == int(pending_setup["index"]):
                    continue
                setup_index = int(pending_setup["index"])
                setup_high = float(pending_setup["high"])
                setup_low = float(pending_setup["low"])
                setup_side = int(pending_setup["side"])
                reversal_armed = bool(int(pending_setup.get("reversal_armed", 0)))
                if setup_side > 0:
                    if i >= setup_index + 2:
                        higher_left = bool(
                            float(data["low"].iloc[setup_index - 1]) > setup_low
                            and float(data["low"].iloc[setup_index - 2]) > setup_low
                        )
                        higher_right = bool(
                            float(data["low"].iloc[setup_index + 1]) > setup_low
                            and float(data["low"].iloc[setup_index + 2]) > setup_low
                        )
                        if higher_left and higher_right:
                            pending_setup["reversal_armed"] = 1
                            reversal_armed = True
                    if high_now >= setup_high:
                        position = 1
                        position_entry_bar = i
                        position_entry_source = "synthetic_1w" if self.only_trade_1w_reversals else "1w"
                        active_base_contract_size = 0.0 if self.only_trade_1w_reversals else first_wiseman_contract_size
                        active_add_on_contracts = 0.0
                        pending_fractal_add_on = None
                        if not self.only_trade_1w_reversals:
                            signals.iloc[i] = 1
                            contracts.iloc[i] = _active_signed_contracts(position)
                            fill_prices.iloc[i] = setup_high
                            stop_loss_prices.iloc[i] = setup_low
                    elif low_now <= setup_low:
                        reversal_window_open = setup_index >= 0 and i >= setup_index + 3
                        if self.only_trade_1w_reversals and reversal_armed and reversal_window_open:
                            position = -1
                            position_entry_bar = i
                            reversal_position_active = True
                            position_entry_source = "reversal"
                            active_base_contract_size = first_wiseman_contract_size
                            active_add_on_contracts = 0.0
                            pending_fractal_add_on = None
                            signals.iloc[i] = -1
                            contracts.iloc[i] = _active_signed_contracts(position)
                            fill_prices.iloc[i] = setup_low
                            first_reversal_side.iloc[i] = -1
                            reversal_stop_loss = float(data["high"].iloc[setup_index : i + 1].max())
                            stop_loss_prices.iloc[i] = reversal_stop_loss
                        elif not self.only_trade_1w_reversals:
                            setup_bar = int(pending_setup.get("index", -1))
                            if setup_bar >= 0 and first_ignored_reason.iloc[setup_bar] == "":
                                first_ignored_reason.iloc[setup_bar] = "invalidation_before_trigger"
                            pending_setup = None
                else:
                    if i >= setup_index + 2:
                        lower_left = bool(
                            float(data["high"].iloc[setup_index - 1]) < setup_high
                            and float(data["high"].iloc[setup_index - 2]) < setup_high
                        )
                        lower_right = bool(
                            float(data["high"].iloc[setup_index + 1]) < setup_high
                            and float(data["high"].iloc[setup_index + 2]) < setup_high
                        )
                        if lower_left and lower_right:
                            pending_setup["reversal_armed"] = 1
                            reversal_armed = True
                    if low_now <= setup_low:
                        position = -1
                        position_entry_bar = i
                        position_entry_source = "synthetic_1w" if self.only_trade_1w_reversals else "1w"
                        active_base_contract_size = 0.0 if self.only_trade_1w_reversals else first_wiseman_contract_size
                        active_add_on_contracts = 0.0
                        pending_fractal_add_on = None
                        if not self.only_trade_1w_reversals:
                            signals.iloc[i] = -1
                            contracts.iloc[i] = _active_signed_contracts(position)
                            fill_prices.iloc[i] = setup_low
                            stop_loss_prices.iloc[i] = setup_high
                    elif high_now >= setup_high:
                        reversal_window_open = setup_index >= 0 and i >= setup_index + 3
                        if self.only_trade_1w_reversals and reversal_armed and reversal_window_open:
                            position = 1
                            position_entry_bar = i
                            reversal_position_active = True
                            position_entry_source = "reversal"
                            active_base_contract_size = first_wiseman_contract_size
                            active_add_on_contracts = 0.0
                            pending_fractal_add_on = None
                            signals.iloc[i] = 1
                            contracts.iloc[i] = _active_signed_contracts(position)
                            fill_prices.iloc[i] = setup_high
                            first_reversal_side.iloc[i] = 1
                            reversal_stop_loss = float(data["low"].iloc[setup_index : i + 1].min())
                            stop_loss_prices.iloc[i] = reversal_stop_loss
                        elif not self.only_trade_1w_reversals:
                            setup_bar = int(pending_setup.get("index", -1))
                            if setup_bar >= 0 and first_ignored_reason.iloc[setup_bar] == "":
                                first_ignored_reason.iloc[setup_bar] = "invalidation_before_trigger"
                            pending_setup = None

            stopped_by_setup_stop = False
            if position != 0 and pending_setup is not None:
                setup_index = int(pending_setup["index"])
                setup_side = int(pending_setup["side"])
                setup_high = float(pending_setup["high"])
                setup_low = float(pending_setup["low"])
                setup_source = str(pending_setup.get("source", "1w"))
                reversal_armed = bool(int(pending_setup["reversal_armed"]))

                if setup_source != "1w":
                    if position == 1:
                        updated_stop = _ntd_initial_long_stop(i)
                        if np.isfinite(updated_stop):
                            setup_low = float(updated_stop)
                            pending_setup["low"] = setup_low
                        if np.isfinite(setup_low) and low_now <= setup_low:
                            _append_intrabar_event(
                                i,
                                "exit",
                                1,
                                setup_low,
                                0.0,
                                "Strategy Stop Loss Bullish Fractal",
                            )
                            reversal_stop = _ntd_initial_short_stop(i)
                            if not np.isfinite(reversal_stop):
                                reversal_stop = setup_high
                            if np.isfinite(reversal_stop) and reversal_stop > setup_low:
                                _append_intrabar_event(
                                    i,
                                    "entry",
                                    -1,
                                    setup_low,
                                    ntd_initial_fractal_contract_size,
                                    "Bearish Fractal",
                                    reversal_stop,
                                )
                                position = -1
                                position_entry_bar = i
                                position_entry_source = "fractal"
                                active_base_contract_size = ntd_initial_fractal_contract_size
                                active_add_on_contracts = 0.0
                                pending_fractal_add_on = None
                                signals.iloc[i] = -1
                                contracts.iloc[i] = _active_signed_contracts(position)
                                fill_prices.iloc[i] = setup_low
                                stop_loss_prices.iloc[i] = reversal_stop
                                fractal_position_side.iloc[i] = -1
                                pending_setup = {
                                    "index": i,
                                    "high": reversal_stop,
                                    "low": setup_low,
                                    "side": -1,
                                    "reversal_armed": 0,
                                    "source": "fractal",
                                }
                            else:
                                position = 0
                                position_entry_bar = -1
                                position_entry_source = ""
                                active_base_contract_size = 0.0
                                active_add_on_contracts = 0.0
                                pending_fractal_add_on = None
                                signals.iloc[i] = 0
                                contracts.iloc[i] = 0.0
                                fill_prices.iloc[i] = setup_low
                                stop_loss_prices.iloc[i] = setup_low
                                pending_setup = None
                            stopped_by_setup_stop = True
                        else:
                            signals.iloc[i] = 1
                            contracts.iloc[i] = _active_signed_contracts(position)
                            stop_loss_prices.iloc[i] = setup_low
                    elif position == -1:
                        updated_stop = _ntd_initial_short_stop(i)
                        if np.isfinite(updated_stop):
                            setup_high = float(updated_stop)
                            pending_setup["high"] = setup_high
                        if np.isfinite(setup_high) and high_now >= setup_high:
                            _append_intrabar_event(
                                i,
                                "exit",
                                -1,
                                setup_high,
                                0.0,
                                "Strategy Stop Loss Bearish Fractal",
                            )
                            reversal_stop = _ntd_initial_long_stop(i)
                            if not np.isfinite(reversal_stop):
                                reversal_stop = setup_low
                            if np.isfinite(reversal_stop) and reversal_stop < setup_high:
                                _append_intrabar_event(
                                    i,
                                    "entry",
                                    1,
                                    setup_high,
                                    ntd_initial_fractal_contract_size,
                                    "Bullish Fractal",
                                    reversal_stop,
                                )
                                position = 1
                                position_entry_bar = i
                                position_entry_source = "fractal"
                                active_base_contract_size = ntd_initial_fractal_contract_size
                                active_add_on_contracts = 0.0
                                pending_fractal_add_on = None
                                signals.iloc[i] = 1
                                contracts.iloc[i] = _active_signed_contracts(position)
                                fill_prices.iloc[i] = setup_high
                                stop_loss_prices.iloc[i] = reversal_stop
                                fractal_position_side.iloc[i] = 1
                                pending_setup = {
                                    "index": i,
                                    "high": setup_high,
                                    "low": reversal_stop,
                                    "side": 1,
                                    "reversal_armed": 0,
                                    "source": "fractal",
                                }
                            else:
                                position = 0
                                position_entry_bar = -1
                                position_entry_source = ""
                                active_base_contract_size = 0.0
                                active_add_on_contracts = 0.0
                                pending_fractal_add_on = None
                                signals.iloc[i] = 0
                                contracts.iloc[i] = 0.0
                                fill_prices.iloc[i] = setup_high
                                stop_loss_prices.iloc[i] = setup_high
                                pending_setup = None
                            stopped_by_setup_stop = True
                        else:
                            signals.iloc[i] = -1
                            contracts.iloc[i] = _active_signed_contracts(position)
                            stop_loss_prices.iloc[i] = setup_high
                    continue

                if not reversal_position_active and setup_side == 1 and position == 1:
                    higher_left = bool(
                        float(data["low"].iloc[setup_index - 1]) > setup_low
                        and float(data["low"].iloc[setup_index - 2]) > setup_low
                    )
                    if i >= setup_index + 2:
                        higher_right = bool(
                            float(data["low"].iloc[setup_index + 1]) > setup_low
                            and float(data["low"].iloc[setup_index + 2]) > setup_low
                        )
                        if higher_left and higher_right:
                            pending_setup["reversal_armed"] = 1
                            reversal_armed = True

                    if low_now <= setup_low:
                        same_bar_entry = position_entry_bar == i and position_entry_source != "synthetic_1w"
                        original_entry_price = float(fill_prices.iloc[i]) if np.isfinite(fill_prices.iloc[i]) else setup_low
                        if reversal_armed:
                            if same_bar_entry:
                                _append_intrabar_event(
                                    i,
                                    "entry",
                                    1,
                                    original_entry_price,
                                    first_wiseman_contract_size,
                                    "Bullish 1W",
                                    setup_low,
                                )
                                _append_intrabar_event(
                                    i,
                                    "exit",
                                    -1,
                                    setup_low,
                                    0.0,
                                    "Strategy Stop Loss Bullish 1W",
                                )
                                _append_intrabar_event(
                                    i,
                                    "entry",
                                    -1,
                                    setup_low,
                                    first_wiseman_contract_size,
                                    "Bearish 1W-R",
                                    float(data["high"].iloc[setup_index : i + 1].max()),
                                )
                            position = -1
                            position_entry_bar = i
                            reversal_position_active = True
                            position_entry_source = "reversal"
                            active_base_contract_size = first_wiseman_contract_size
                            active_add_on_contracts = 0.0
                            pending_fractal_add_on = None
                            signals.iloc[i] = -1
                            contracts.iloc[i] = _active_signed_contracts(position)
                            fill_prices.iloc[i] = setup_low
                            first_reversal_side.iloc[i] = -1
                            reversal_stop_loss = float(data["high"].iloc[setup_index : i + 1].max())
                            stop_loss_prices.iloc[i] = reversal_stop_loss
                        else:
                            if same_bar_entry:
                                _append_intrabar_event(
                                    i,
                                    "entry",
                                    1,
                                    original_entry_price,
                                    first_wiseman_contract_size,
                                    "Bullish 1W",
                                    setup_low,
                                )
                                _append_intrabar_event(
                                    i,
                                    "exit",
                                    -1,
                                    setup_low,
                                    0.0,
                                    "Strategy Stop Loss Bullish 1W",
                                )
                            position = 0
                            position_entry_bar = -1
                            position_entry_source = ""
                            active_base_contract_size = 0.0
                            active_add_on_contracts = 0.0
                            pending_fractal_add_on = None
                            signals.iloc[i] = 0
                            contracts.iloc[i] = 0.0
                            fill_prices.iloc[i] = setup_low
                            stop_loss_prices.iloc[i] = setup_low
                            pending_setup = None
                            stopped_by_setup_stop = True
                    elif position == 1:
                        signals.iloc[i] = 1
                        contracts.iloc[i] = _active_signed_contracts(position)
                        stop_loss_prices.iloc[i] = setup_low

                elif not reversal_position_active and setup_side == -1 and position == -1:
                    lower_left = bool(
                        float(data["high"].iloc[setup_index - 1]) < setup_high
                        and float(data["high"].iloc[setup_index - 2]) < setup_high
                    )
                    if i >= setup_index + 2:
                        lower_right = bool(
                            float(data["high"].iloc[setup_index + 1]) < setup_high
                            and float(data["high"].iloc[setup_index + 2]) < setup_high
                        )
                        if lower_left and lower_right:
                            pending_setup["reversal_armed"] = 1
                            reversal_armed = True

                    if high_now >= setup_high:
                        same_bar_entry = position_entry_bar == i and position_entry_source != "synthetic_1w"
                        original_entry_price = float(fill_prices.iloc[i]) if np.isfinite(fill_prices.iloc[i]) else setup_high
                        if reversal_armed:
                            if same_bar_entry:
                                _append_intrabar_event(
                                    i,
                                    "entry",
                                    -1,
                                    original_entry_price,
                                    first_wiseman_contract_size,
                                    "Bearish 1W",
                                    setup_high,
                                )
                                _append_intrabar_event(
                                    i,
                                    "exit",
                                    1,
                                    setup_high,
                                    0.0,
                                    "Strategy Stop Loss Bearish 1W",
                                )
                                _append_intrabar_event(
                                    i,
                                    "entry",
                                    1,
                                    setup_high,
                                    first_wiseman_contract_size,
                                    "Bullish 1W-R",
                                    float(data["low"].iloc[setup_index : i + 1].min()),
                                )
                            position = 1
                            position_entry_bar = i
                            reversal_position_active = True
                            position_entry_source = "reversal"
                            active_base_contract_size = first_wiseman_contract_size
                            active_add_on_contracts = 0.0
                            pending_fractal_add_on = None
                            signals.iloc[i] = 1
                            contracts.iloc[i] = _active_signed_contracts(position)
                            fill_prices.iloc[i] = setup_high
                            first_reversal_side.iloc[i] = 1
                            reversal_stop_loss = float(data["low"].iloc[setup_index : i + 1].min())
                            stop_loss_prices.iloc[i] = reversal_stop_loss
                        else:
                            if same_bar_entry:
                                _append_intrabar_event(
                                    i,
                                    "entry",
                                    -1,
                                    original_entry_price,
                                    first_wiseman_contract_size,
                                    "Bearish 1W",
                                    setup_high,
                                )
                                _append_intrabar_event(
                                    i,
                                    "exit",
                                    1,
                                    setup_high,
                                    0.0,
                                    "Strategy Stop Loss Bearish 1W",
                                )
                            position = 0
                            position_entry_bar = -1
                            position_entry_source = ""
                            active_base_contract_size = 0.0
                            active_add_on_contracts = 0.0
                            pending_fractal_add_on = None
                            signals.iloc[i] = 0
                            contracts.iloc[i] = 0.0
                            fill_prices.iloc[i] = setup_high
                            stop_loss_prices.iloc[i] = setup_high
                            pending_setup = None
                            stopped_by_setup_stop = True
                    elif position == -1:
                        signals.iloc[i] = -1
                        contracts.iloc[i] = _active_signed_contracts(position)
                        stop_loss_prices.iloc[i] = setup_high

            # After a setup stop-out flatten, defer any new NTD initial trigger
            # to the next bar so a single bar cannot both flatten and reopen.
            if stopped_by_setup_stop:
                continue

            if reversal_position_active:
                stopped_by_reversal_stop = False
                if position_entry_bar == i:
                    if position == -1:
                        signals.iloc[i] = -1
                        contracts.iloc[i] = _active_signed_contracts(position)
                        stop_loss_prices.iloc[i] = reversal_stop_loss
                    elif position == 1:
                        signals.iloc[i] = 1
                        contracts.iloc[i] = _active_signed_contracts(position)
                        stop_loss_prices.iloc[i] = reversal_stop_loss
                    continue
                if position == -1 and reversal_stop_loss is not None and high_now >= reversal_stop_loss:
                    position = 0
                    position_entry_bar = -1
                    position_entry_source = ""
                    active_base_contract_size = 0.0
                    active_add_on_contracts = 0.0
                    pending_fractal_add_on = None
                    signals.iloc[i] = 0
                    contracts.iloc[i] = 0.0
                    exit_reason.iloc[i] = "1W Reversal Stop"
                    fill_prices.iloc[i] = reversal_stop_loss
                    stop_loss_prices.iloc[i] = reversal_stop_loss
                    reversal_stop_loss = None
                    reversal_position_active = False
                    pending_setup = None
                    stopped_by_reversal_stop = True
                elif position == 1 and reversal_stop_loss is not None and low_now <= reversal_stop_loss:
                    position = 0
                    position_entry_bar = -1
                    position_entry_source = ""
                    active_base_contract_size = 0.0
                    active_add_on_contracts = 0.0
                    pending_fractal_add_on = None
                    signals.iloc[i] = 0
                    contracts.iloc[i] = 0.0
                    exit_reason.iloc[i] = "1W Reversal Stop"
                    fill_prices.iloc[i] = reversal_stop_loss
                    stop_loss_prices.iloc[i] = reversal_stop_loss
                    reversal_stop_loss = None
                    reversal_position_active = False
                    pending_setup = None
                    stopped_by_reversal_stop = True
                elif position == -1:
                    signals.iloc[i] = -1
                    contracts.iloc[i] = _active_signed_contracts(position)
                    stop_loss_prices.iloc[i] = reversal_stop_loss
                elif position == 1:
                    signals.iloc[i] = 1
                    contracts.iloc[i] = _active_signed_contracts(position)
                    stop_loss_prices.iloc[i] = reversal_stop_loss
                # After a 1W reversal-stop flatten, defer any NTD re-entry
                # to the next bar to keep the stop bar purely flattening.
                if stopped_by_reversal_stop:
                    continue

            if (
                (self.red_teeth_profit_protection_enabled or self.green_lips_profit_protection_enabled)
                and position != 0
                and position_entry_bar >= 0
                and (pd.notna(teeth.iloc[i]) or pd.notna(lips.iloc[i]))
            ):
                bars_in_position = i - position_entry_bar
                entry_price = (
                    float(fill_prices.iloc[position_entry_bar])
                    if np.isfinite(fill_prices.iloc[position_entry_bar])
                    else float(data["open"].iloc[position_entry_bar])
                )
                favorable_excursion = (
                    (high_now / entry_price) - 1.0
                    if position == 1
                    else (entry_price / low_now) - 1.0
                )
                min_unrealized_return = _annualized_volatility_scaled_return_threshold(
                    self.red_teeth_profit_protection_min_unrealized_return,
                    red_teeth_annualized_volatility.iloc[i],
                    self.red_teeth_profit_protection_annualized_volatility_scaler,
                )
                green_lips_min_unrealized_return = _annualized_volatility_scaled_return_threshold(
                    self.green_lips_profit_protection_min_unrealized_return,
                    green_lips_annualized_volatility.iloc[i],
                    self.green_lips_profit_protection_annualized_volatility_scaler,
                )
                close_breached_teeth = pd.notna(teeth.iloc[i]) and (
                    (position == 1 and close_now < float(teeth.iloc[i]))
                    or (position == -1 and close_now > float(teeth.iloc[i]))
                )
                close_breached_lips = pd.notna(lips.iloc[i]) and (
                    (position == 1 and close_now < float(lips.iloc[i]))
                    or (position == -1 and close_now > float(lips.iloc[i]))
                )
                if (
                    bars_in_position >= self.red_teeth_profit_protection_min_bars
                    and favorable_excursion >= min_unrealized_return
                    and close_breached_teeth
                    and self.red_teeth_profit_protection_enabled
                ):
                    signals.iloc[i] = 0
                    contracts.iloc[i] = 0.0
                    fill_prices.iloc[i] = close_now
                    stop_loss_prices.iloc[i] = close_now
                    exit_reason.iloc[i] = "Red Gator Teeth PP"
                    position = 0
                    position_entry_bar = -1
                    reversal_position_active = False
                    reversal_stop_loss = None
                    position_entry_source = ""
                    active_base_contract_size = 0.0
                    active_add_on_contracts = 0.0
                    pending_setup = None
                    opposite_setup = None
                    pending_ntd_initial_long = None
                    pending_ntd_initial_short = None
                    pending_fractal_add_on = None
                    continue
                if (
                    bars_in_position >= self.green_lips_profit_protection_min_bars
                    and favorable_excursion >= green_lips_min_unrealized_return
                    and close_breached_lips
                    and self.green_lips_profit_protection_enabled
                ):
                    signals.iloc[i] = 0
                    contracts.iloc[i] = 0.0
                    fill_prices.iloc[i] = close_now
                    stop_loss_prices.iloc[i] = close_now
                    exit_reason.iloc[i] = "Green Gator Lips PP"
                    position = 0
                    position_entry_bar = -1
                    reversal_position_active = False
                    reversal_stop_loss = None
                    position_entry_source = ""
                    active_base_contract_size = 0.0
                    active_add_on_contracts = 0.0
                    pending_setup = None
                    opposite_setup = None
                    pending_ntd_initial_long = None
                    pending_ntd_initial_short = None
                    pending_fractal_add_on = None
                    continue

            if _try_trigger_underlying_gain_exit(i, open_now, high_now, low_now):
                continue

            if position != 0 and str(exit_reason.iloc[i]).strip() == "":
                signal_side_now = int(np.sign(signals.iloc[i]))
                if signal_side_now == 0 and not np.isfinite(fill_prices.iloc[i]):
                    signals.iloc[i] = int(position)
                    contracts.iloc[i] = _active_signed_contracts(position)
                elif signal_side_now != int(position):
                    signals.iloc[i] = int(position)
                    contracts.iloc[i] = _active_signed_contracts(position)

            should_apply_add_on_fill = (
                add_on_fill_candidate_side != 0
                and position != 0
                and position == add_on_fill_candidate_side
                and position == position_at_bar_open
                and str(exit_reason.iloc[i]).strip() == ""
                and (int(np.sign(signals.iloc[i])) in (0, position))
                and not np.isfinite(fill_prices.iloc[i])
            )
            if should_apply_add_on_fill:
                active_add_on_contracts += fractal_add_on_contract_size
                add_on_fractal_fill_side.iloc[i] = int(position)
                fill_prices.iloc[i] = float(add_on_fill_candidate_price)
                if int(np.sign(signals.iloc[i])) == 0:
                    signals.iloc[i] = int(position)
                contracts.iloc[i] = _active_signed_contracts(position)

            if str(exit_reason.iloc[i]).strip() == "":
                if _try_trigger_underlying_gain_exit(i, open_now, high_now, low_now):
                    continue

        if self.only_trade_1w_reversals:
            zero_contract_mask = contracts.abs() <= 1e-12
            signals.loc[zero_contract_mask] = 0

        self.signal_fill_prices = fill_prices
        self.signal_stop_loss_prices = stop_loss_prices
        self.signal_contracts = contracts
        sanitized_intrabar_events: dict[int, list[dict[str, float | int | str]]] = {}
        for bar_index, raw_events in intrabar_events.items():
            if not isinstance(raw_events, list) or not raw_events:
                continue
            reversal_marker = int(first_reversal_side.iloc[bar_index]) if 0 <= bar_index < len(index) else 0
            desired_side = int(np.sign(signals.iloc[bar_index])) if 0 <= bar_index < len(index) else 0
            first_entry_side: int | None = None
            saw_intrabar_exit = False
            filtered_events: list[dict[str, float | int | str]] = []
            for raw_event in raw_events:
                if not isinstance(raw_event, dict):
                    continue
                event_type = str(raw_event.get("event_type", "")).strip().lower()
                if event_type == "exit":
                    filtered_events.append(raw_event)
                    saw_intrabar_exit = True
                    continue
                if event_type != "entry":
                    filtered_events.append(raw_event)
                    continue
                entry_side = int(raw_event.get("side", 0))
                entry_reason = str(raw_event.get("reason", "")).strip()
                if entry_side == 0:
                    continue
                if reversal_marker == 0 and desired_side != 0 and entry_side != desired_side:
                    continue
                if first_entry_side is None:
                    first_entry_side = entry_side
                    filtered_events.append(raw_event)
                    continue
                if entry_side == first_entry_side:
                    filtered_events.append(raw_event)
                    continue
                if reversal_marker != 0 and saw_intrabar_exit and "1W-R" in entry_reason:
                    filtered_events.append(raw_event)
                    first_entry_side = entry_side
            if filtered_events:
                sanitized_intrabar_events[bar_index] = filtered_events
        # Execution consumers should reference armed/active first setups, while
        # charting can still use marker-side candidates (including ignored ones).
        self.signal_first_wiseman_setup_side = first_setup_side
        self.signal_first_wiseman_setup_marker_side = first_setup_marker_side
        self.signal_first_wiseman_ignored_reason = first_ignored_reason
        self.signal_first_wiseman_reversal_side = first_reversal_side
        self.signal_add_on_fractal_fill_side = add_on_fractal_fill_side
        self.signal_fractal_position_side = fractal_position_side
        self.signal_exit_reason = exit_reason
        self.signal_intrabar_events = sanitized_intrabar_events
        return signals


class NTDStrategy(Strategy):
    """NTD fractal pyramid strategy with Bill Williams-style profit protection."""

    execute_on_signal_bar = True

    def __init__(
        self,
        gator_width_lookback: int = 50,
        gator_width_mult: float = 1.0,
        require_gator_close_reset: bool = True,
        ao_ac_near_zero_lookback: int = 50,
        ao_ac_near_zero_factor: float = 0.25,
        teeth_profit_protection_enabled: bool = False,
        teeth_profit_protection_min_bars: int = 3,
        teeth_profit_protection_min_unrealized_return: float = 1.0,
        teeth_profit_protection_credit_unrealized_before_min_bars: bool = False,
        teeth_profit_protection_require_gator_open: bool = True,
        profit_protection_volatility_lookback: int | None = None,
        profit_protection_annualized_volatility_scaler: float = 1.0,
        lips_profit_protection_enabled: bool = False,
        lips_profit_protection_volatility_trigger: float = 0.02,
        lips_profit_protection_profit_trigger_mult: float = 2.0,
        lips_profit_protection_volatility_lookback: int = 20,
        lips_profit_protection_recent_trade_lookback: int = 5,
        lips_profit_protection_min_unrealized_return: float = 1.0,
        lips_profit_protection_arm_on_min_unrealized_return: bool = False,
        zone_profit_protection_enabled: bool = False,
        zone_profit_protection_min_unrealized_return: float = 1.0,
    ) -> None:
        if gator_width_lookback <= 0:
            raise ValueError("gator_width_lookback must be positive")
        if gator_width_mult <= 0:
            raise ValueError("gator_width_mult must be positive")
        if ao_ac_near_zero_lookback <= 0:
            raise ValueError("ao_ac_near_zero_lookback must be positive")
        if ao_ac_near_zero_factor <= 0:
            raise ValueError("ao_ac_near_zero_factor must be positive")
        if teeth_profit_protection_min_bars < 1:
            raise ValueError("teeth_profit_protection_min_bars must be >= 1")
        if teeth_profit_protection_min_unrealized_return < 0:
            raise ValueError("teeth_profit_protection_min_unrealized_return must be >= 0")
        if profit_protection_volatility_lookback is not None and profit_protection_volatility_lookback < 2:
            raise ValueError("profit_protection_volatility_lookback must be >= 2")
        if profit_protection_annualized_volatility_scaler <= 0:
            raise ValueError("profit_protection_annualized_volatility_scaler must be > 0")
        if lips_profit_protection_volatility_trigger < 0:
            raise ValueError("lips_profit_protection_volatility_trigger must be >= 0")
        if lips_profit_protection_profit_trigger_mult < 0:
            raise ValueError("lips_profit_protection_profit_trigger_mult must be >= 0")
        if lips_profit_protection_volatility_lookback < 2:
            raise ValueError("lips_profit_protection_volatility_lookback must be >= 2")
        if lips_profit_protection_recent_trade_lookback < 1:
            raise ValueError("lips_profit_protection_recent_trade_lookback must be >= 1")
        if lips_profit_protection_min_unrealized_return < 0:
            raise ValueError("lips_profit_protection_min_unrealized_return must be >= 0")
        if zone_profit_protection_min_unrealized_return < 0:
            raise ValueError("zone_profit_protection_min_unrealized_return must be >= 0")

        self.gator_width_lookback = gator_width_lookback
        self.gator_width_mult = gator_width_mult
        self.require_gator_close_reset = require_gator_close_reset
        self.ao_ac_near_zero_lookback = ao_ac_near_zero_lookback
        self.ao_ac_near_zero_factor = ao_ac_near_zero_factor
        self.teeth_profit_protection_enabled = teeth_profit_protection_enabled
        self.teeth_profit_protection_min_bars = teeth_profit_protection_min_bars
        self.teeth_profit_protection_min_unrealized_return = teeth_profit_protection_min_unrealized_return
        self.teeth_profit_protection_credit_unrealized_before_min_bars = (
            teeth_profit_protection_credit_unrealized_before_min_bars
        )
        self.teeth_profit_protection_require_gator_open = teeth_profit_protection_require_gator_open
        self.profit_protection_volatility_lookback = (
            profit_protection_volatility_lookback
            if profit_protection_volatility_lookback is not None
            else lips_profit_protection_volatility_lookback
        )
        self.profit_protection_annualized_volatility_scaler = profit_protection_annualized_volatility_scaler
        self.lips_profit_protection_enabled = lips_profit_protection_enabled
        self.lips_profit_protection_volatility_trigger = lips_profit_protection_volatility_trigger
        self.lips_profit_protection_profit_trigger_mult = lips_profit_protection_profit_trigger_mult
        self.lips_profit_protection_volatility_lookback = lips_profit_protection_volatility_lookback
        self.lips_profit_protection_recent_trade_lookback = lips_profit_protection_recent_trade_lookback
        self.lips_profit_protection_min_unrealized_return = lips_profit_protection_min_unrealized_return
        self.lips_profit_protection_arm_on_min_unrealized_return = lips_profit_protection_arm_on_min_unrealized_return
        self.zone_profit_protection_enabled = zone_profit_protection_enabled
        self.zone_profit_protection_min_unrealized_return = zone_profit_protection_min_unrealized_return

    def _williams_zone_bars(self, data: pd.DataFrame, ao: pd.Series) -> tuple[pd.Series, pd.Series]:
        ac = ao - ao.rolling(5).mean()
        ao_green = ao >= ao.shift(1)
        ao_red = ao < ao.shift(1)
        ac_green = ac >= ac.shift(1)
        ac_red = ac < ac.shift(1)
        zone_green = (ao_green & ac_green).fillna(False)
        zone_red = (ao_red & ac_red).fillna(False)
        return zone_green.astype(bool), zone_red.astype(bool)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        jaw, teeth, lips = _alligator_lines(data)
        gator_range = pd.concat([jaw, teeth, lips], axis=1).max(axis=1) - pd.concat([jaw, teeth, lips], axis=1).min(axis=1)
        gator_slope = jaw.diff().abs() + teeth.diff().abs() + lips.diff().abs()
        range_baseline = gator_range.rolling(self.gator_width_lookback, min_periods=1).median()
        slope_baseline = gator_slope.rolling(self.gator_width_lookback, min_periods=1).median()
        gator_closed = (gator_range <= (range_baseline * self.gator_width_mult)) & (
            gator_slope <= (slope_baseline * self.gator_width_mult)
        )

        ao = _williams_ao(data)
        ac = ao - ao.rolling(5).mean()
        close_returns = data["close"].pct_change()
        periods_per_year = infer_periods_per_year(data.index, default=252)
        rolling_volatility = (
            close_returns.rolling(self.profit_protection_volatility_lookback, min_periods=2).std(ddof=0)
            * np.sqrt(periods_per_year)
        )
        zone_green, zone_red = self._williams_zone_bars(data, ao)
        fractals = detect_williams_fractals(data)
        ao_abs_baseline = ao.abs().rolling(self.ao_ac_near_zero_lookback, min_periods=1).median()
        ac_abs_baseline = ac.abs().rolling(self.ao_ac_near_zero_lookback, min_periods=1).median()
        ao_near_zero = ao.abs() <= np.maximum(ao_abs_baseline * self.ao_ac_near_zero_factor, 1e-12)
        ac_near_zero = ac.abs() <= np.maximum(ac_abs_baseline * self.ao_ac_near_zero_factor, 1e-12)
        sleeping_gator = gator_closed & ao_near_zero.fillna(False) & ac_near_zero.fillna(False)

        signals = pd.Series(0, index=data.index, dtype="int8")
        contracts = pd.Series(0.0, index=data.index, dtype="float64")
        fill_prices = pd.Series(np.nan, index=data.index, dtype="float64")
        stop_loss_prices = pd.Series(np.nan, index=data.index, dtype="float64")
        exit_reason = pd.Series("", index=data.index, dtype="object")
        fractal_position_side = pd.Series(0, index=data.index, dtype="int8")

        position = 0
        position_contracts = 0.0
        entry_i = -1
        pending_add: dict[str, float | int] | None = None
        active_entry_stop_price = np.nan
        active_reversal_side = 0
        waiting_for_sleeping_gator = self.require_gator_close_reset
        sleeping_setup_active = False
        current_up_initial_trigger = np.nan
        current_down_initial_trigger = np.nan
        latest_up_fractal_price = np.nan
        latest_down_fractal_price = np.nan
        latest_up_above_gator_price = np.nan
        latest_up_above_gator_bar = -1
        latest_down_below_gator_price = np.nan
        latest_down_below_gator_bar = -1
        recent_closed_trade_returns: deque[float] = deque(maxlen=self.lips_profit_protection_recent_trade_lookback)
        teeth_profit_protection_armed = False
        teeth_profit_protection_unrealized_gate_met = False
        profit_protection_entry_i = -1
        zone_profit_protection_active = False
        zone_profit_protection_stop_level = np.nan
        zone_green_streak = 0
        zone_red_streak = 0

        def _qualifies_initial_fractal(fractal_bar: int, side: int) -> bool:
            if fractal_bar < 0:
                return False
            if side == 1:
                return bool(fractals.iloc[fractal_bar]["up_fractal"])
            return bool(fractals.iloc[fractal_bar]["down_fractal"])

        def _qualifies_add_on_fractal(fractal_bar: int, side: int) -> bool:
            if fractal_bar < 0:
                return False
            if side == 1:
                return (
                    bool(fractals.iloc[fractal_bar]["up_fractal"])
                    and pd.notna(teeth.iloc[fractal_bar])
                    and float(data["high"].iloc[fractal_bar]) > float(teeth.iloc[fractal_bar])
                )
            return (
                bool(fractals.iloc[fractal_bar]["down_fractal"])
                and pd.notna(teeth.iloc[fractal_bar])
                and float(data["low"].iloc[fractal_bar]) < float(teeth.iloc[fractal_bar])
            )

        def _qualifies_gator_breakout_fractal(fractal_bar: int, side: int) -> bool:
            if fractal_bar < 0:
                return False
            gator_lines = np.array([jaw.iloc[fractal_bar], teeth.iloc[fractal_bar], lips.iloc[fractal_bar]], dtype="float64")
            finite_lines = gator_lines[np.isfinite(gator_lines)]
            if finite_lines.size == 0:
                return False
            gator_max = float(np.max(finite_lines))
            gator_min = float(np.min(finite_lines))
            if side == 1:
                return bool(fractals.iloc[fractal_bar]["up_fractal"]) and float(data["high"].iloc[fractal_bar]) > gator_max
            return bool(fractals.iloc[fractal_bar]["down_fractal"]) and float(data["low"].iloc[fractal_bar]) < gator_min

        def _current_reversal_stop_for_side(side: int) -> tuple[float, int]:
            if side == 1 and np.isfinite(latest_down_below_gator_price):
                return latest_down_below_gator_price, -1
            if side == -1 and np.isfinite(latest_up_above_gator_price):
                return latest_up_above_gator_price, 1
            return np.nan, 0

        def _select_initial_breakout_side(open_price: float, up_trigger: float, down_trigger: float, bar_high: float, bar_low: float) -> int:
            up_hit = np.isfinite(up_trigger) and bar_high >= up_trigger
            down_hit = np.isfinite(down_trigger) and bar_low <= down_trigger
            if up_hit and not down_hit:
                return 1
            if down_hit and not up_hit:
                return -1
            if not up_hit and not down_hit:
                return 0
            up_distance = abs(up_trigger - open_price)
            down_distance = abs(open_price - down_trigger)
            return 1 if up_distance <= down_distance else -1

        for i in range(len(data)):
            open_now = float(data["open"].iloc[i])
            high_now = float(data["high"].iloc[i])
            low_now = float(data["low"].iloc[i])
            close_now = float(data["close"].iloc[i])
            gator_is_closed = bool(gator_closed.iloc[i])
            confirmed_up_above_gator = False
            confirmed_down_below_gator = False

            confirmed_fractal_bar = i - 2
            if confirmed_fractal_bar >= 0:
                if _qualifies_initial_fractal(confirmed_fractal_bar, 1):
                    latest_up_fractal_price = float(data["high"].iloc[confirmed_fractal_bar])
                if _qualifies_initial_fractal(confirmed_fractal_bar, -1):
                    latest_down_fractal_price = float(data["low"].iloc[confirmed_fractal_bar])
                if _qualifies_gator_breakout_fractal(confirmed_fractal_bar, 1):
                    latest_up_above_gator_price = float(data["high"].iloc[confirmed_fractal_bar])
                    latest_up_above_gator_bar = confirmed_fractal_bar
                    confirmed_up_above_gator = True
                if _qualifies_gator_breakout_fractal(confirmed_fractal_bar, -1):
                    latest_down_below_gator_price = float(data["low"].iloc[confirmed_fractal_bar])
                    latest_down_below_gator_bar = confirmed_fractal_bar
                    confirmed_down_below_gator = True
                if position != 0 and _qualifies_add_on_fractal(confirmed_fractal_bar, position):
                    pending_add = {
                        "side": position,
                        "price": float(data["high"].iloc[confirmed_fractal_bar]) if position == 1 else float(data["low"].iloc[confirmed_fractal_bar]),
                        "placed_bar": i,
                    }

            if position == -1 and confirmed_up_above_gator:
                active_entry_stop_price = latest_up_above_gator_price
                active_reversal_side = 1
            elif position == 1 and confirmed_down_below_gator:
                active_entry_stop_price = latest_down_below_gator_price
                active_reversal_side = -1

            prior_position = position
            prior_contracts = position_contracts
            closed_by_profit_protection = False

            if position == 0:
                if bool(sleeping_gator.iloc[i]) and (waiting_for_sleeping_gator or not self.require_gator_close_reset):
                    sleeping_setup_active = True
                    waiting_for_sleeping_gator = False
                    pending_add = None
                elif not bool(sleeping_gator.iloc[i]) and waiting_for_sleeping_gator:
                    sleeping_setup_active = False

                if sleeping_setup_active:
                    if np.isfinite(latest_up_fractal_price):
                        current_up_initial_trigger = latest_up_fractal_price
                    if np.isfinite(latest_down_fractal_price):
                        current_down_initial_trigger = latest_down_fractal_price

                    breakout_side = _select_initial_breakout_side(
                        open_now,
                        current_up_initial_trigger,
                        current_down_initial_trigger,
                        high_now,
                        low_now,
                    )
                    if breakout_side != 0:
                        trigger_price = current_up_initial_trigger if breakout_side == 1 else current_down_initial_trigger
                        position = breakout_side
                        position_contracts = 1.0
                        entry_i = i
                        active_entry_stop_price, active_reversal_side = _current_reversal_stop_for_side(breakout_side)
                        fill_prices.iloc[i] = trigger_price
                        pending_add = None
                        sleeping_setup_active = False
                        current_up_initial_trigger = np.nan
                        current_down_initial_trigger = np.nan

            if pending_add is not None and position != 0 and position == int(pending_add["side"]) and i > int(pending_add["placed_bar"]):
                add_price = float(pending_add["price"])
                triggered = (position == 1 and high_now >= add_price) or (position == -1 and low_now <= add_price)
                if triggered:
                    position_contracts += 1.0
                    fill_prices.iloc[i] = add_price
                    pending_add = None

            stop_and_reverse_triggered = (
                position != 0
                and entry_i >= 0
                and i > entry_i
                and np.isfinite(active_entry_stop_price)
                and active_reversal_side != 0
                and (
                    (position == 1 and low_now <= active_entry_stop_price)
                    or (position == -1 and high_now >= active_entry_stop_price)
                )
            )
            if stop_and_reverse_triggered:
                reversal_price = float(active_entry_stop_price)
                position = active_reversal_side
                entry_i = i
                fill_prices.iloc[i] = reversal_price
                pending_add = None
                active_entry_stop_price, active_reversal_side = _current_reversal_stop_for_side(
                    position,
                )

            if position != 0 and entry_i >= 0:
                entry_price = float(fill_prices.iloc[entry_i]) if np.isfinite(fill_prices.iloc[entry_i]) else np.nan
                if np.isfinite(entry_price) and entry_price > 0:
                    favorable_price = high_now if position == 1 else low_now
                    unrealized_return = (
                        (favorable_price - entry_price) / entry_price
                        if position == 1
                        else (entry_price - favorable_price) / entry_price
                    )
                    bars_in_position = i - entry_i

                    if prior_position == position and np.isnan(fill_prices.iloc[i]):
                        if position == 1:
                            zone_green_streak = zone_green_streak + 1 if bool(zone_green.iloc[i]) else 0
                            zone_red_streak = 0
                        else:
                            zone_red_streak = zone_red_streak + 1 if bool(zone_red.iloc[i]) else 0
                            zone_green_streak = 0

                    volatility_now = rolling_volatility.iloc[i]
                    teeth_min_unrealized_return = _annualized_volatility_scaled_return_threshold(
                        self.teeth_profit_protection_min_unrealized_return,
                        volatility_now,
                        self.profit_protection_annualized_volatility_scaler,
                    )
                    zone_min_unrealized_return = _annualized_volatility_scaled_return_threshold(
                        self.zone_profit_protection_min_unrealized_return,
                        volatility_now,
                        self.profit_protection_annualized_volatility_scaler,
                    )
                    lips_min_unrealized_return = _annualized_volatility_scaled_return_threshold(
                        self.lips_profit_protection_min_unrealized_return,
                        volatility_now,
                        self.profit_protection_annualized_volatility_scaler,
                    )

                    if self.teeth_profit_protection_enabled or self.lips_profit_protection_enabled:
                        if profit_protection_entry_i != entry_i:
                            teeth_profit_protection_armed = False
                            teeth_profit_protection_unrealized_gate_met = False
                            zone_profit_protection_active = False
                            zone_profit_protection_stop_level = np.nan
                            zone_green_streak = 0
                            zone_red_streak = 0
                            profit_protection_entry_i = entry_i
                        gator_open = not gator_is_closed
                        if (
                            self.teeth_profit_protection_credit_unrealized_before_min_bars
                            or bars_in_position >= self.teeth_profit_protection_min_bars
                        ) and unrealized_return >= teeth_min_unrealized_return:
                            teeth_profit_protection_unrealized_gate_met = True
                        if (
                            bars_in_position >= self.teeth_profit_protection_min_bars
                            and teeth_profit_protection_unrealized_gate_met
                            and ((not self.teeth_profit_protection_require_gator_open) or gator_open)
                        ):
                            teeth_profit_protection_armed = True

                    if (
                        self.zone_profit_protection_enabled
                        and zone_profit_protection_active
                        and prior_position == position
                        and np.isnan(fill_prices.iloc[i])
                        and np.isfinite(zone_profit_protection_stop_level)
                        and (
                            (position == 1 and low_now <= zone_profit_protection_stop_level)
                            or (position == -1 and high_now >= zone_profit_protection_stop_level)
                        )
                    ):
                        position = 0
                        fill_prices.iloc[i] = (
                            min(open_now, zone_profit_protection_stop_level)
                            if prior_position == 1
                            else max(open_now, zone_profit_protection_stop_level)
                        )
                        exit_reason.iloc[i] = "Williams Zone PP"
                    else:
                        if (
                            self.zone_profit_protection_enabled
                            and prior_position == position
                            and np.isnan(fill_prices.iloc[i])
                        ):
                            streak = zone_green_streak if position == 1 else zone_red_streak
                            if zone_profit_protection_active:
                                zone_profit_protection_stop_level = low_now if position == 1 else high_now
                            elif streak >= 5 and unrealized_return >= zone_min_unrealized_return:
                                zone_profit_protection_active = True
                                zone_profit_protection_stop_level = low_now if position == 1 else high_now

                        use_lips_exit = False
                        recent_trade_baseline = (
                            float(np.median(np.abs(np.array(recent_closed_trade_returns, dtype="float64"))))
                            if recent_closed_trade_returns
                            else 0.0
                        )
                        volatility_baseline = float(volatility_now) if pd.notna(volatility_now) else 0.0
                        lips_volatility_trigger = _scaled_annualized_volatility_trigger(
                            self.lips_profit_protection_volatility_trigger,
                            self.profit_protection_annualized_volatility_scaler,
                        )
                        deep_profit_reference = max(volatility_baseline, recent_trade_baseline)
                        if self.lips_profit_protection_enabled:
                            deep_profit_triggered = (
                                deep_profit_reference > 0
                                and unrealized_return > (deep_profit_reference * self.lips_profit_protection_profit_trigger_mult)
                            )
                            min_unrealized_triggered = (
                                self.lips_profit_protection_arm_on_min_unrealized_return
                                and unrealized_return > lips_min_unrealized_return
                            )
                            high_volatility = (
                                pd.notna(volatility_now)
                                and float(volatility_now) >= lips_volatility_trigger
                            )
                            use_lips_exit = (
                                teeth_profit_protection_armed
                                and (high_volatility or min_unrealized_triggered or deep_profit_triggered)
                            )

                        teeth_now = teeth.iloc[i]
                        lips_now = lips.iloc[i]
                        use_teeth_exit = (
                            self.teeth_profit_protection_enabled
                            and teeth_profit_protection_armed
                            and pd.notna(teeth_now)
                        )
                        protection_level = np.nan
                        reason = ""
                        if use_lips_exit and use_teeth_exit and pd.notna(lips_now):
                            protection_level = float(lips_now)
                            reason = "Green Gator PP"
                        elif use_teeth_exit:
                            protection_level = float(teeth_now)
                            reason = "Red Gator PP"
                        if (
                            np.isfinite(protection_level)
                            and np.isnan(fill_prices.iloc[i])
                            and ((position == 1 and close_now < protection_level) or (position == -1 and close_now > protection_level))
                        ):
                            position = 0
                            fill_prices.iloc[i] = close_now
                            exit_reason.iloc[i] = reason
                            closed_by_profit_protection = True

            if prior_position != 0 and position == 0 and np.isfinite(fill_prices.iloc[i]) and entry_i >= 0:
                entry_price = float(fill_prices.iloc[entry_i])
                if np.isfinite(entry_price) and entry_price > 0:
                    realized_return = (
                        (float(fill_prices.iloc[i]) - entry_price) / entry_price
                        if prior_position == 1
                        else (entry_price - float(fill_prices.iloc[i])) / entry_price
                    )
                    recent_closed_trade_returns.append(realized_return)
                position_contracts = 0.0
                pending_add = None
                active_entry_stop_price = np.nan
                active_reversal_side = 0
                sleeping_setup_active = False
                waiting_for_sleeping_gator = self.require_gator_close_reset
                current_up_initial_trigger = np.nan
                current_down_initial_trigger = np.nan
                teeth_profit_protection_armed = False
                teeth_profit_protection_unrealized_gate_met = False
                profit_protection_entry_i = -1
                zone_profit_protection_active = False
                zone_profit_protection_stop_level = np.nan
                zone_green_streak = 0
                zone_red_streak = 0

            if closed_by_profit_protection:
                active_entry_stop_price = np.nan
                active_reversal_side = 0

            signals.iloc[i] = position
            contracts.iloc[i] = position * position_contracts
            label_side = position
            if label_side == 0 and prior_position != 0 and np.isfinite(fill_prices.iloc[i]):
                label_side = prior_position
            fractal_position_side.iloc[i] = int(label_side)
            if position != 0 and np.isfinite(active_entry_stop_price):
                stop_loss_prices.iloc[i] = active_entry_stop_price

        self.signal_fill_prices = fill_prices
        self.signal_stop_loss_prices = stop_loss_prices
        self.signal_contracts = contracts
        self.signal_exit_reason = exit_reason
        self.signal_fractal_position_side = fractal_position_side
        return signals.fillna(0)


class CombinedStrategy(Strategy):
    """Combine multiple strategies into a single net signal/contracts stream."""

    execute_on_signal_bar = True
    _PROFIT_PROTECTION_REASONS = {
        "Red Gator PP",
        "Red Gator Teeth PP",
        "Green Gator PP",
        "Green Gator Lips PP",
        "Williams Zone PP",
        "Sigma Move PP",
    }

    def __init__(self, strategies: list[Strategy]) -> None:
        if not strategies:
            raise ValueError("strategies must not be empty")
        self.strategies = strategies

    def _generate_wiseman_ntd_mode(
        self,
        data: pd.DataFrame,
        *,
        component_contracts: list[pd.Series],
        component_fills: list[pd.Series | None],
        component_stop_losses: list[pd.Series | None],
        component_exit_reasons: list[pd.Series | None],
        component_first_wiseman_setup_sides: list[pd.Series | None],
        component_first_wiseman_ignored_reasons: list[pd.Series | None],
        component_first_wiseman_reversal_sides: list[pd.Series | None],
        component_first_wiseman_fill_prices: list[pd.Series | None],
        component_second_wiseman_fill_sides: list[pd.Series | None],
        component_third_wiseman_fill_sides: list[pd.Series | None],
        component_fractal_position_sides: list[pd.Series | None],
        wiseman_index: int,
        ntd_index: int,
    ) -> pd.Series:
        net_contracts = pd.Series(0.0, index=data.index, dtype="float64")
        fill_prices = pd.Series(np.nan, index=data.index, dtype="float64")
        stop_loss_prices = pd.Series(np.nan, index=data.index, dtype="float64")
        exit_reason = pd.Series("", index=data.index, dtype="object")
        combined_second_wiseman_fill_side = pd.Series(0, index=data.index, dtype="int8")
        combined_third_wiseman_fill_side = pd.Series(0, index=data.index, dtype="int8")
        combined_fractal_position_side = pd.Series(0, index=data.index, dtype="int8")

        wiseman_setup_series = component_first_wiseman_setup_sides[wiseman_index]
        wiseman_ignored_reason_series = component_first_wiseman_ignored_reasons[wiseman_index]
        wiseman_reversal_series = component_first_wiseman_reversal_sides[wiseman_index]
        wiseman_first_fill_series = component_first_wiseman_fill_prices[wiseman_index]
        wiseman_fill_series = component_fills[wiseman_index]
        wiseman_stop_series = component_stop_losses[wiseman_index]
        wiseman_exit_reason_series = component_exit_reasons[wiseman_index]
        wiseman_second_fill_series = component_second_wiseman_fill_sides[wiseman_index]
        wiseman_third_fill_series = component_third_wiseman_fill_sides[wiseman_index]

        ntd_fill_series = component_fills[ntd_index]
        ntd_stop_series = component_stop_losses[ntd_index]
        ntd_exit_reason_series = component_exit_reasons[ntd_index]
        ntd_fractal_side_series = component_fractal_position_sides[ntd_index]
        wiseman_strategy = self.strategies[wiseman_index]
        wiseman_base_contracts = max(int(getattr(wiseman_strategy, "first_wiseman_contracts", 1)), 0)
        wiseman_third_enabled = int(getattr(wiseman_strategy, "third_wiseman_contracts", 0)) > 0
        _, ntd_teeth, _ = _alligator_lines(data)
        ntd_fractals = detect_williams_fractals(data)

        unified_contracts = 0.0
        regime = ""
        wiseman_activated = False
        synthetic_ntd_add_order: dict[str, float | int] | None = None
        wiseman_reversal_lock_active = False
        last_wiseman_setup_bar_by_side: dict[int, int] = {1: -1_000_000, -1: -1_000_000}
        unified_entry_bar = -1
        wiseman_last_entry_bar = -1
        ntd_last_entry_bar = -1

        def _series_value(series: pd.Series | None, bar_index: int, default: float | int | str) -> float | int | str:
            if isinstance(series, pd.Series):
                value = series.iloc[bar_index]
                if pd.notna(value):
                    return value
            return default

        def _fill_value(series: pd.Series | None, bar_index: int) -> float:
            if isinstance(series, pd.Series):
                value = series.iloc[bar_index]
                if pd.notna(value):
                    return float(value)
            return np.nan

        for i, _ in enumerate(data.index):
            open_now = float(data["open"].iloc[i])
            high_now = float(data["high"].iloc[i])
            low_now = float(data["low"].iloc[i])
            current_side = int(np.sign(unified_contracts))
            current_abs_contracts = abs(unified_contracts)

            wiseman_contracts_now = float(component_contracts[wiseman_index].iloc[i])
            wiseman_contracts_prev = float(component_contracts[wiseman_index].iloc[i - 1]) if i > 0 else 0.0
            wiseman_side_now = int(np.sign(wiseman_contracts_now))
            wiseman_side_prev = int(np.sign(wiseman_contracts_prev))
            wiseman_first_fill_now = _fill_value(wiseman_first_fill_series, i)
            wiseman_fill_now = _fill_value(wiseman_fill_series, i)
            wiseman_exit_now = str(_series_value(wiseman_exit_reason_series, i, "")).strip()
            wiseman_reversal_now = int(_series_value(wiseman_reversal_series, i, 0))
            wiseman_second_side_now = int(_series_value(wiseman_second_fill_series, i, 0))
            wiseman_third_side_now = int(_series_value(wiseman_third_fill_series, i, 0))
            wiseman_setup_now = int(_series_value(wiseman_setup_series, i, 0))
            wiseman_ignored_reason_now = str(_series_value(wiseman_ignored_reason_series, i, "")).strip()
            if wiseman_setup_now in {-1, 1} and wiseman_ignored_reason_now == "":
                last_wiseman_setup_bar_by_side[wiseman_setup_now] = i

            ntd_contracts_now = float(component_contracts[ntd_index].iloc[i])
            ntd_contracts_prev = float(component_contracts[ntd_index].iloc[i - 1]) if i > 0 else 0.0
            ntd_side_now = int(np.sign(ntd_contracts_now))
            ntd_side_prev = int(np.sign(ntd_contracts_prev))
            ntd_fill_now = _fill_value(ntd_fill_series, i)
            ntd_exit_now = str(_series_value(ntd_exit_reason_series, i, "")).strip()
            ntd_fractal_side_now = int(_series_value(ntd_fractal_side_series, i, 0))
            wiseman_directional_entry = (
                np.isfinite(wiseman_fill_now)
                and wiseman_side_now != 0
                and wiseman_side_now != wiseman_side_prev
            )
            ntd_directional_entry = (
                np.isfinite(ntd_fill_now)
                and ntd_side_now != 0
                and ntd_side_now != ntd_side_prev
            )
            if wiseman_directional_entry:
                wiseman_last_entry_bar = i
            if ntd_directional_entry:
                ntd_last_entry_bar = i
            wiseman_entry_triggered = (
                np.isfinite(wiseman_first_fill_now)
                and wiseman_side_now != 0
                and wiseman_side_prev == 0
                and abs(wiseman_contracts_now) > 0
            )
            ntd_entry_triggered = (
                np.isfinite(ntd_fill_now)
                and ntd_side_now != 0
                and ntd_side_prev == 0
                and abs(ntd_contracts_now) > 0
            )

            combined_second_wiseman_fill_side.iloc[i] = 0
            combined_third_wiseman_fill_side.iloc[i] = 0
            combined_fractal_position_side.iloc[i] = 0

            next_contracts = unified_contracts
            next_regime = regime
            bar_fill_price = np.nan
            bar_exit_reason = ""

            if current_side == 0:
                synthetic_ntd_add_order = None
                initial_candidates: list[tuple[float, int, str, float]] = []
                if wiseman_entry_triggered:
                    initial_candidates.append((abs(wiseman_first_fill_now - open_now), 0, "wiseman", wiseman_first_fill_now))
                if ntd_entry_triggered:
                    initial_candidates.append((abs(ntd_fill_now - open_now), 1, "ntd", ntd_fill_now))

                if initial_candidates:
                    _, _, initial_source, trigger_fill = min(initial_candidates, key=lambda item: (item[0], item[1]))
                    if initial_source == "wiseman":
                        next_contracts = wiseman_contracts_now
                        next_regime = "wiseman"
                        wiseman_activated = True
                        unified_entry_bar = wiseman_last_entry_bar if wiseman_last_entry_bar >= 0 else i
                        bar_fill_price = trigger_fill
                        if ntd_entry_triggered and ntd_side_now == wiseman_side_now:
                            next_contracts += float(abs(ntd_contracts_now))
                            combined_fractal_position_side.iloc[i] = ntd_side_now
                    else:
                        next_contracts = ntd_contracts_now
                        next_regime = "ntd"
                        wiseman_activated = False
                        unified_entry_bar = ntd_last_entry_bar if ntd_last_entry_bar >= 0 else i
                        bar_fill_price = trigger_fill
                        combined_fractal_position_side.iloc[i] = ntd_side_now
            else:
                confirmed_fractal_bar = i - 2
                if confirmed_fractal_bar >= 0:
                    if (
                        current_side == 1
                        and bool(ntd_fractals["up_fractal"].iloc[confirmed_fractal_bar])
                        and pd.notna(ntd_teeth.iloc[confirmed_fractal_bar])
                        and float(data["high"].iloc[confirmed_fractal_bar]) > float(ntd_teeth.iloc[confirmed_fractal_bar])
                    ):
                        synthetic_ntd_add_order = {"side": 1, "price": float(data["high"].iloc[confirmed_fractal_bar]), "placed_bar": i}
                    elif (
                        current_side == -1
                        and bool(ntd_fractals["down_fractal"].iloc[confirmed_fractal_bar])
                        and pd.notna(ntd_teeth.iloc[confirmed_fractal_bar])
                        and float(data["low"].iloc[confirmed_fractal_bar]) < float(ntd_teeth.iloc[confirmed_fractal_bar])
                    ):
                        synthetic_ntd_add_order = {"side": -1, "price": float(data["low"].iloc[confirmed_fractal_bar]), "placed_bar": i}

                wiseman_pp_exit = (
                    wiseman_exit_now in self._PROFIT_PROTECTION_REASONS
                    and np.isfinite(wiseman_fill_now)
                    and wiseman_side_prev == current_side
                    and wiseman_side_now == 0
                    and wiseman_last_entry_bar >= unified_entry_bar
                )
                ntd_pp_exit = (
                    ntd_exit_now in self._PROFIT_PROTECTION_REASONS
                    and np.isfinite(ntd_fill_now)
                    and ntd_side_prev == current_side
                    and ntd_side_now == 0
                    and ntd_last_entry_bar >= unified_entry_bar
                    and not wiseman_reversal_lock_active
                )
                wiseman_component_flat_exit = (
                    wiseman_exit_now != ""
                    and np.isfinite(wiseman_fill_now)
                    and wiseman_side_prev == current_side
                    and wiseman_side_now == 0
                    and abs(wiseman_contracts_prev) > 1e-12
                    and abs(wiseman_contracts_now) <= 1e-12
                    and wiseman_last_entry_bar >= unified_entry_bar
                )
                ntd_component_flat_exit = (
                    ntd_exit_now != ""
                    and np.isfinite(ntd_fill_now)
                    and ntd_side_prev == current_side
                    and ntd_side_now == 0
                    and abs(ntd_contracts_prev) > 1e-12
                    and abs(ntd_contracts_now) <= 1e-12
                    and ntd_last_entry_bar >= unified_entry_bar
                    and not wiseman_reversal_lock_active
                )
                opposite_wiseman_signal = (
                    np.isfinite(wiseman_fill_now)
                    and wiseman_side_now == -current_side
                    and wiseman_side_prev != -current_side
                    and wiseman_last_entry_bar >= unified_entry_bar
                    and not (
                        regime == "ntd"
                        and last_wiseman_setup_bar_by_side.get(wiseman_side_now, -1_000_000) > unified_entry_bar
                    )
                )
                opposite_ntd_signal = (
                    np.isfinite(ntd_fill_now)
                    and ntd_side_now == -current_side
                    and ntd_side_prev != -current_side
                    and not wiseman_reversal_lock_active
                )

                if (
                    wiseman_exit_now == "1W Reversal Stop"
                    and np.isfinite(wiseman_fill_now)
                ):
                    next_contracts = 0.0
                    next_regime = ""
                    wiseman_activated = False
                    wiseman_reversal_lock_active = False
                    synthetic_ntd_add_order = None
                    bar_fill_price = wiseman_fill_now
                    bar_exit_reason = wiseman_exit_now
                elif wiseman_pp_exit:
                    next_contracts = 0.0
                    next_regime = ""
                    wiseman_activated = False
                    bar_fill_price = wiseman_fill_now
                    bar_exit_reason = wiseman_exit_now
                elif ntd_pp_exit:
                    next_contracts = 0.0
                    next_regime = ""
                    wiseman_activated = False
                    bar_fill_price = ntd_fill_now
                    bar_exit_reason = ntd_exit_now
                    combined_fractal_position_side.iloc[i] = current_side
                elif wiseman_component_flat_exit:
                    next_contracts = 0.0
                    next_regime = ""
                    wiseman_activated = False
                    bar_fill_price = wiseman_fill_now
                    bar_exit_reason = wiseman_exit_now
                elif ntd_component_flat_exit:
                    next_contracts = 0.0
                    next_regime = ""
                    wiseman_activated = False
                    bar_fill_price = ntd_fill_now
                    bar_exit_reason = ntd_exit_now
                    combined_fractal_position_side.iloc[i] = current_side
                elif opposite_wiseman_signal:
                    reversal_contracts = (
                        current_abs_contracts
                        if wiseman_reversal_now == -current_side
                        else abs(wiseman_contracts_now)
                    )
                    if reversal_contracts <= 0:
                        reversal_contracts = float(wiseman_base_contracts)
                    next_contracts = float(wiseman_side_now * reversal_contracts)
                    next_regime = "wiseman"
                    wiseman_activated = True
                    if wiseman_reversal_now == -current_side:
                        wiseman_reversal_lock_active = True
                    synthetic_ntd_add_order = None
                    unified_entry_bar = wiseman_last_entry_bar if wiseman_last_entry_bar >= 0 else i
                    bar_fill_price = wiseman_fill_now if np.isfinite(wiseman_fill_now) else wiseman_first_fill_now
                elif opposite_ntd_signal:
                    next_contracts = float(ntd_side_now)
                    next_regime = "ntd"
                    wiseman_activated = False
                    unified_entry_bar = ntd_last_entry_bar if ntd_last_entry_bar >= 0 else i
                    bar_fill_price = ntd_fill_now
                    combined_fractal_position_side.iloc[i] = ntd_side_now
                else:
                    total_add_on_contracts = 0.0
                    ntd_add_on_triggered = False
                    if wiseman_activated and np.isfinite(wiseman_fill_now) and wiseman_side_now == current_side:
                        wiseman_add_on_contracts = max(abs(wiseman_contracts_now) - abs(wiseman_contracts_prev), 0.0)
                        if wiseman_add_on_contracts > 0 and wiseman_second_side_now == current_side:
                            total_add_on_contracts += wiseman_add_on_contracts
                            bar_fill_price = wiseman_fill_now
                            combined_second_wiseman_fill_side.iloc[i] = current_side
                        elif wiseman_add_on_contracts > 0 and wiseman_third_enabled and wiseman_third_side_now == current_side:
                            total_add_on_contracts += wiseman_add_on_contracts
                            bar_fill_price = wiseman_fill_now
                            combined_third_wiseman_fill_side.iloc[i] = current_side
                    if np.isfinite(ntd_fill_now) and ntd_side_now == current_side:
                        ntd_add_on_contracts = max(abs(ntd_contracts_now) - abs(ntd_contracts_prev), 0.0)
                        if ntd_add_on_contracts > 0:
                            ntd_add_on_triggered = True
                            ntd_increment = ntd_add_on_contracts
                            if ntd_side_prev == current_side and abs(ntd_contracts_prev) > 1e-12:
                                # Once NTD is already active in the same direction,
                                # each new fractal add-on should contribute one step
                                # per bar. Clamp malformed contract jumps to preserve
                                # single-fill coherence.
                                ntd_increment = min(ntd_increment, 1.0)
                            total_add_on_contracts += ntd_increment
                            if not np.isfinite(bar_fill_price):
                                bar_fill_price = ntd_fill_now
                            combined_fractal_position_side.iloc[i] = current_side
                    if (
                        not ntd_add_on_triggered
                        and synthetic_ntd_add_order is not None
                        and int(synthetic_ntd_add_order["side"]) == current_side
                        and i > int(synthetic_ntd_add_order["placed_bar"])
                    ):
                        synthetic_trigger_price = float(synthetic_ntd_add_order["price"])
                        synthetic_triggered = (
                            current_side == 1 and high_now >= synthetic_trigger_price
                        ) or (
                            current_side == -1 and low_now <= synthetic_trigger_price
                        )
                        if synthetic_triggered:
                            ntd_add_on_triggered = True
                            total_add_on_contracts += 1.0
                            bar_fill_price = synthetic_trigger_price
                            combined_fractal_position_side.iloc[i] = current_side
                            synthetic_ntd_add_order = None
                    if ntd_add_on_triggered and combined_third_wiseman_fill_side.iloc[i] == current_side:
                        combined_third_wiseman_fill_side.iloc[i] = 0
                        if np.isfinite(ntd_fill_now):
                            bar_fill_price = ntd_fill_now
                    if total_add_on_contracts > 0:
                        next_contracts = float(current_side * (current_abs_contracts + total_add_on_contracts))

            unified_contracts = next_contracts
            regime = next_regime
            net_contracts.iloc[i] = unified_contracts
            if np.isfinite(bar_fill_price) and not np.isclose(next_contracts, current_side * current_abs_contracts, atol=1e-12):
                fill_prices.iloc[i] = bar_fill_price
            elif np.isfinite(bar_fill_price) and current_side != 0 and next_contracts == 0:
                fill_prices.iloc[i] = bar_fill_price
            elif np.isfinite(bar_fill_price) and current_side != int(np.sign(next_contracts)):
                fill_prices.iloc[i] = bar_fill_price
            if bar_exit_reason:
                exit_reason.iloc[i] = bar_exit_reason

            effective_side = int(np.sign(unified_contracts))
            if effective_side == 0 or effective_side != current_side:
                synthetic_ntd_add_order = None
            if effective_side == 0:
                wiseman_reversal_lock_active = False
            if effective_side != 0:
                stop_series = wiseman_stop_series if regime == "wiseman" else ntd_stop_series
                stop_value = _fill_value(stop_series, i)
                if np.isfinite(stop_value):
                    stop_loss_prices.iloc[i] = stop_value

        signals = pd.Series(np.sign(net_contracts), index=data.index, dtype="int8")
        self.signal_contracts = net_contracts
        self.signal_fill_prices = fill_prices
        self.signal_stop_loss_prices = stop_loss_prices
        self.signal_exit_reason = exit_reason
        self.signal_first_wiseman_setup_side = (
            wiseman_setup_series.copy()
            if isinstance(wiseman_setup_series, pd.Series)
            else pd.Series(0, index=data.index, dtype="int8")
        )
        self.signal_first_wiseman_ignored_reason = (
            wiseman_ignored_reason_series.copy()
            if isinstance(wiseman_ignored_reason_series, pd.Series)
            else pd.Series("", index=data.index, dtype="object")
        )
        self.signal_first_wiseman_reversal_side = (
            wiseman_reversal_series.copy()
            if isinstance(wiseman_reversal_series, pd.Series)
            else pd.Series(0, index=data.index, dtype="int8")
        )
        self.signal_second_wiseman_fill_side = combined_second_wiseman_fill_side
        self.signal_third_wiseman_fill_side = combined_third_wiseman_fill_side
        self.signal_fill_prices_first = wiseman_first_fill_series
        self.signal_fractal_position_side = combined_fractal_position_side
        return signals

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        component_signals: list[pd.Series] = []
        component_contracts: list[pd.Series] = []
        component_fills: list[pd.Series | None] = []
        component_stop_losses: list[pd.Series | None] = []
        component_exit_reasons: list[pd.Series | None] = []
        component_first_wiseman_setup_sides: list[pd.Series | None] = []
        component_first_wiseman_ignored_reasons: list[pd.Series | None] = []
        component_first_wiseman_reversal_sides: list[pd.Series | None] = []
        component_first_wiseman_fill_prices: list[pd.Series | None] = []
        component_second_wiseman_fill_sides: list[pd.Series | None] = []
        component_third_wiseman_fill_sides: list[pd.Series | None] = []
        component_fractal_position_sides: list[pd.Series | None] = []
        combined_first_wiseman_setup_side = pd.Series(0, index=data.index, dtype="int8")
        combined_first_wiseman_ignored_reason = pd.Series("", index=data.index, dtype="object")
        combined_first_wiseman_reversal_side = pd.Series(0, index=data.index, dtype="int8")
        combined_second_wiseman_fill_side = pd.Series(0, index=data.index, dtype="int8")
        combined_third_wiseman_fill_side = pd.Series(0, index=data.index, dtype="int8")
        combined_fractal_position_side = pd.Series(0, index=data.index, dtype="int8")

        def _aligned_side_series(strategy: Strategy, attr_name: str) -> pd.Series | None:
            series = getattr(strategy, attr_name, None)
            if not isinstance(series, pd.Series):
                return None
            return series.reindex(data.index).fillna(0).astype("int8")

        def _copy_component_labels(component_index: int | None, bar_index: int) -> None:
            combined_first_wiseman_setup_side.iloc[bar_index] = 0
            combined_first_wiseman_ignored_reason.iloc[bar_index] = ""
            combined_first_wiseman_reversal_side.iloc[bar_index] = 0
            combined_second_wiseman_fill_side.iloc[bar_index] = 0
            combined_third_wiseman_fill_side.iloc[bar_index] = 0
            combined_fractal_position_side.iloc[bar_index] = 0
            if component_index is None:
                return

            first_setup = component_first_wiseman_setup_sides[component_index]
            first_ignored = component_first_wiseman_ignored_reasons[component_index]
            first_reversal = component_first_wiseman_reversal_sides[component_index]
            second_fill = component_second_wiseman_fill_sides[component_index]
            third_fill = component_third_wiseman_fill_sides[component_index]
            fractal_side = component_fractal_position_sides[component_index]

            if first_setup is not None:
                combined_first_wiseman_setup_side.iloc[bar_index] = int(first_setup.iloc[bar_index])
            if first_ignored is not None:
                combined_first_wiseman_ignored_reason.iloc[bar_index] = str(first_ignored.iloc[bar_index])
            if first_reversal is not None:
                combined_first_wiseman_reversal_side.iloc[bar_index] = int(first_reversal.iloc[bar_index])
            if second_fill is not None:
                combined_second_wiseman_fill_side.iloc[bar_index] = int(second_fill.iloc[bar_index])
            if third_fill is not None:
                combined_third_wiseman_fill_side.iloc[bar_index] = int(third_fill.iloc[bar_index])
            if fractal_side is not None:
                combined_fractal_position_side.iloc[bar_index] = int(fractal_side.iloc[bar_index])

        for strategy in self.strategies:
            component_signals.append(strategy.generate_signals(data).reindex(data.index).fillna(0).astype("float64"))
            contracts = getattr(strategy, "signal_contracts", None)
            component_contracts.append(
                contracts.reindex(data.index).fillna(0).astype("float64")
                if isinstance(contracts, pd.Series)
                else component_signals[-1]
            )
            fills = getattr(strategy, "signal_fill_prices", None)
            component_fills.append(fills.reindex(data.index) if isinstance(fills, pd.Series) else None)
            stop_losses = getattr(strategy, "signal_stop_loss_prices", None)
            component_stop_losses.append(stop_losses.reindex(data.index) if isinstance(stop_losses, pd.Series) else None)
            exit_reasons = getattr(strategy, "signal_exit_reason", None)
            component_exit_reasons.append(exit_reasons.reindex(data.index) if isinstance(exit_reasons, pd.Series) else None)
            component_first_wiseman_setup_sides.append(_aligned_side_series(strategy, "signal_first_wiseman_setup_side"))
            first_ignored_reason = getattr(strategy, "signal_first_wiseman_ignored_reason", None)
            component_first_wiseman_ignored_reasons.append(
                first_ignored_reason.reindex(data.index).fillna("").astype("object")
                if isinstance(first_ignored_reason, pd.Series)
                else None
            )
            component_first_wiseman_reversal_sides.append(_aligned_side_series(strategy, "signal_first_wiseman_reversal_side"))
            first_fill_prices = getattr(strategy, "signal_fill_prices_first", None)
            component_first_wiseman_fill_prices.append(
                first_fill_prices.reindex(data.index)
                if isinstance(first_fill_prices, pd.Series)
                else None
            )
            component_second_wiseman_fill_sides.append(_aligned_side_series(strategy, "signal_second_wiseman_fill_side"))
            component_third_wiseman_fill_sides.append(_aligned_side_series(strategy, "signal_third_wiseman_fill_side"))
            component_fractal_position_sides.append(_aligned_side_series(strategy, "signal_fractal_position_side"))

        wiseman_candidates = [
            i
            for i in range(len(self.strategies))
            if (
                isinstance(component_first_wiseman_fill_prices[i], pd.Series)
                and bool(component_first_wiseman_fill_prices[i].notna().any())
            )
            or (
                isinstance(component_first_wiseman_setup_sides[i], pd.Series)
                and bool(component_first_wiseman_setup_sides[i].ne(0).any())
            )
            or (
                isinstance(component_first_wiseman_ignored_reasons[i], pd.Series)
                and bool(component_first_wiseman_ignored_reasons[i].ne("").any())
            )
            or (
                isinstance(component_first_wiseman_reversal_sides[i], pd.Series)
                and bool(component_first_wiseman_reversal_sides[i].ne(0).any())
            )
        ]
        ntd_candidates = [
            i
            for i, fractal_side in enumerate(component_fractal_position_sides)
            if isinstance(fractal_side, pd.Series)
            and i not in wiseman_candidates
        ]
        if len(self.strategies) == 2 and len(wiseman_candidates) == 1 and len(ntd_candidates) == 1:
            return self._generate_wiseman_ntd_mode(
                data,
                component_contracts=component_contracts,
                component_fills=component_fills,
                component_stop_losses=component_stop_losses,
                component_exit_reasons=component_exit_reasons,
                component_first_wiseman_setup_sides=component_first_wiseman_setup_sides,
                component_first_wiseman_ignored_reasons=component_first_wiseman_ignored_reasons,
                component_first_wiseman_reversal_sides=component_first_wiseman_reversal_sides,
                component_first_wiseman_fill_prices=component_first_wiseman_fill_prices,
                component_second_wiseman_fill_sides=component_second_wiseman_fill_sides,
                component_third_wiseman_fill_sides=component_third_wiseman_fill_sides,
                component_fractal_position_sides=component_fractal_position_sides,
                wiseman_index=wiseman_candidates[0],
                ntd_index=ntd_candidates[0],
            )

        net_contracts = pd.Series(0.0, index=data.index, dtype="float64")
        fill_prices = pd.Series(np.nan, index=data.index, dtype="float64")
        exit_reason = pd.Series("", index=data.index, dtype="object")
        prior_effective_contracts = 0.0

        for i, _ in enumerate(data.index):
            raw_net_contracts = float(sum(series.iloc[i] for series in component_contracts))
            fill_component_indices = [
                component_index
                for component_index, series in enumerate(component_fills)
                if isinstance(series, pd.Series) and pd.notna(series.iloc[i])
            ]
            non_nan_fills = [float(component_fills[component_index].iloc[i]) for component_index in fill_component_indices]
            current_side = int(np.sign(prior_effective_contracts))
            pp_exit_component_indices: list[int] = []
            for component_index, series in enumerate(component_exit_reasons):
                if not isinstance(series, pd.Series):
                    continue
                reason_text = str(series.iloc[i]).strip()
                if reason_text == "":
                    continue
                fill_series = component_fills[component_index]
                has_fill = isinstance(fill_series, pd.Series) and pd.notna(fill_series.iloc[i])
                component_contracts_now = float(component_contracts[component_index].iloc[i])
                component_contracts_prev = float(component_contracts[component_index].iloc[i - 1]) if i > 0 else 0.0
                component_side_now = int(np.sign(component_contracts_now))
                component_side_prev = int(np.sign(component_contracts_prev))
                if (
                    reason_text in self._PROFIT_PROTECTION_REASONS
                    and current_side != 0
                    and has_fill
                    and component_side_prev == current_side
                    and component_side_now == 0
                ):
                    pp_exit_component_indices.append(component_index)
            raw_side = int(np.sign(raw_net_contracts))
            has_profit_protection_exit = bool(pp_exit_component_indices)
            contract_change_requested = not np.isclose(raw_net_contracts, prior_effective_contracts, atol=1e-12)
            has_execution_trigger = bool(non_nan_fills) or has_profit_protection_exit

            target_net_contracts = 0.0 if has_profit_protection_exit else raw_net_contracts
            allow_contract_change = (
                current_side == 0
                or (raw_side == current_side and abs(target_net_contracts) >= abs(prior_effective_contracts) - 1e-12)
                or has_profit_protection_exit
            )
            effective_net_contracts = (
                target_net_contracts
                if allow_contract_change and ((not contract_change_requested) or has_execution_trigger)
                else prior_effective_contracts
            )
            net_contracts.iloc[i] = effective_net_contracts

            if (
                allow_contract_change
                and not np.isclose(effective_net_contracts, prior_effective_contracts, atol=1e-12)
                and non_nan_fills
            ):
                label_component_index = fill_component_indices[-1]
                label_fill_series = component_fills[label_component_index]
                if isinstance(label_fill_series, pd.Series) and pd.notna(label_fill_series.iloc[i]):
                    fill_prices.iloc[i] = float(label_fill_series.iloc[i])
                else:
                    fill_prices.iloc[i] = non_nan_fills[-1]
                _copy_component_labels(label_component_index, i)
            else:
                _copy_component_labels(None, i)
            if has_profit_protection_exit:
                exit_reason.iloc[i] = str(component_exit_reasons[pp_exit_component_indices[0]].iloc[i])
            prior_effective_contracts = effective_net_contracts

        primary_wiseman_index = wiseman_candidates[0] if wiseman_candidates else None
        if primary_wiseman_index is not None:
            primary_setup = component_first_wiseman_setup_sides[primary_wiseman_index]
            primary_ignored_reason = component_first_wiseman_ignored_reasons[primary_wiseman_index]
            primary_reversal = component_first_wiseman_reversal_sides[primary_wiseman_index]
            if (
                isinstance(primary_setup, pd.Series)
                and bool(primary_setup.ne(0).any())
                and not bool(combined_first_wiseman_setup_side.ne(0).any())
            ):
                combined_first_wiseman_setup_side = primary_setup.copy()
            if isinstance(primary_ignored_reason, pd.Series) and bool(primary_ignored_reason.ne("").any()):
                combined_first_wiseman_ignored_reason = primary_ignored_reason.copy()
            if (
                isinstance(primary_reversal, pd.Series)
                and bool(primary_reversal.ne(0).any())
                and not bool(combined_first_wiseman_reversal_side.ne(0).any())
            ):
                combined_first_wiseman_reversal_side = primary_reversal.copy()

        signals = pd.Series(np.sign(net_contracts), index=data.index, dtype="int8")
        self.signal_contracts = net_contracts
        self.signal_fill_prices = fill_prices
        self.signal_stop_loss_prices = pd.Series(np.nan, index=data.index, dtype="float64")
        self.signal_exit_reason = exit_reason
        self.signal_first_wiseman_setup_side = combined_first_wiseman_setup_side
        self.signal_first_wiseman_ignored_reason = combined_first_wiseman_ignored_reason
        self.signal_first_wiseman_reversal_side = combined_first_wiseman_reversal_side
        self.signal_second_wiseman_fill_side = combined_second_wiseman_fill_side
        self.signal_third_wiseman_fill_side = combined_third_wiseman_fill_side
        self.signal_fractal_position_side = combined_fractal_position_side
        return signals
