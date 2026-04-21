from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtesting import BacktestConfig, BacktestEngine, parse_trade_size_equity_milestones, summarize_wiseman_markers
from backtesting.batch import run_batch_backtest
import backtesting.local_chart as chart_module
from backtesting.local_chart import (
    _ao_histogram_from_data,
    _bearish_first_wiseman_markers,
    _bullish_first_wiseman_markers,
    _execution_trade_path_lines,
    _first_wiseman_engine_markers,
    _second_wiseman_markers,
    _valid_third_wiseman_fractal_markers,
    _wiseman_fill_entry_markers,
    generate_batch_local_tradingview_chart,
    generate_local_tradingview_chart,
)
from backtesting.report import generate_backtest_clean_pdf_report, generate_backtest_pdf_report
from backtesting.tradingview import (
    generate_first_wiseman_bearish_pinescript,
    generate_first_wiseman_bullish_pinescript,
    generate_ut_bot_strategy_pinescript,
)
from backtesting.resample import infer_source_timeframe_label, resample_ohlcv
from backtesting.stats import infer_periods_per_year
from backtesting.strategy import (
    _annualized_volatility_scaled_return_threshold,
    _scaled_annualized_volatility_trigger,
    AlligatorAOStrategy,
    BWStrategy,
    CombinedStrategy,
    NTDStrategy,
    SmaCrossoverStrategy,
    WisemanStrategy,
)
from backtesting.fractals import detect_williams_fractals
from backtesting.engine import ExecutionEvent, Trade
from examples.run_backtest import write_signal_intent_flat_timestamps


class _SignalBarStrategy:
    execute_on_signal_bar = True

    def __init__(self, signals: pd.Series) -> None:
        self._signals = signals

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return self._signals.reindex(data.index).fillna(0).astype("int8")




class _SignalFillStrategy:
    execute_on_signal_bar = False

    def __init__(self, signals: pd.Series, fills: pd.Series) -> None:
        self._signals = signals
        self._fills = fills
        self.signal_fill_prices: pd.Series | None = None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        self.signal_fill_prices = self._fills.reindex(data.index)
        return self._signals.reindex(data.index).fillna(0).astype("int8")


class _SignalFillStopStrategy:
    execute_on_signal_bar = True

    def __init__(
        self,
        signals: pd.Series,
        fills: pd.Series,
        stop_losses: pd.Series,
        first_wiseman_reversal_side: pd.Series | None = None,
    ) -> None:
        self._signals = signals
        self._fills = fills
        self._stop_losses = stop_losses
        self._first_wiseman_reversal_side = first_wiseman_reversal_side
        self.signal_fill_prices: pd.Series | None = None
        self.signal_stop_loss_prices: pd.Series | None = None
        self.signal_first_wiseman_reversal_side: pd.Series | None = None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        self.signal_fill_prices = self._fills.reindex(data.index)
        self.signal_stop_loss_prices = self._stop_losses.reindex(data.index)
        self.signal_first_wiseman_reversal_side = (
            self._first_wiseman_reversal_side.reindex(data.index).fillna(0).astype("int8")
            if isinstance(self._first_wiseman_reversal_side, pd.Series)
            else pd.Series(0, index=data.index, dtype="int8")
        )
        return self._signals.reindex(data.index).fillna(0).astype("int8")


class _SignalContractsStrategy:
    execute_on_signal_bar = True

    def __init__(self, signals: pd.Series, contracts: pd.Series) -> None:
        self._signals = signals
        self._contracts = contracts
        self.signal_contracts: pd.Series | None = None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        self.signal_contracts = self._contracts.reindex(data.index).fillna(0.0).astype("float64")
        return self._signals.reindex(data.index).fillna(0).astype("int8")


class _SignalContractsExitReasonStrategy(_SignalContractsStrategy):
    def __init__(
        self,
        signals: pd.Series,
        contracts: pd.Series,
        *,
        exit_reasons: pd.Series | None = None,
        first_wiseman_setup_side: pd.Series | None = None,
        first_wiseman_ignored_reason: pd.Series | None = None,
        first_wiseman_reversal_side: pd.Series | None = None,
        first_wiseman_fill_prices: pd.Series | None = None,
        second_wiseman_fill_side: pd.Series | None = None,
        third_wiseman_fill_side: pd.Series | None = None,
        fractal_position_side: pd.Series | None = None,
        fill_prices: pd.Series | None = None,
        stop_loss_prices: pd.Series | None = None,
    ) -> None:
        super().__init__(signals, contracts)
        self._exit_reasons = exit_reasons
        self._first_wiseman_setup_side = first_wiseman_setup_side
        self._first_wiseman_ignored_reason = first_wiseman_ignored_reason
        self._first_wiseman_reversal_side = first_wiseman_reversal_side
        self._first_wiseman_fill_prices = first_wiseman_fill_prices
        self._second_wiseman_fill_side = second_wiseman_fill_side
        self._third_wiseman_fill_side = third_wiseman_fill_side
        self._fractal_position_side = fractal_position_side
        self._fill_prices = fill_prices
        self._stop_loss_prices = stop_loss_prices
        self.signal_exit_reason: pd.Series | None = None
        self.signal_first_wiseman_setup_side: pd.Series | None = None
        self.signal_first_wiseman_ignored_reason: pd.Series | None = None
        self.signal_first_wiseman_reversal_side: pd.Series | None = None
        self.signal_fill_prices_first: pd.Series | None = None
        self.signal_second_wiseman_fill_side: pd.Series | None = None
        self.signal_third_wiseman_fill_side: pd.Series | None = None
        self.signal_fractal_position_side: pd.Series | None = None
        self.signal_fill_prices: pd.Series | None = None
        self.signal_stop_loss_prices: pd.Series | None = None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = super().generate_signals(data)
        self.signal_fill_prices = (
            self._fill_prices.reindex(data.index)
            if isinstance(self._fill_prices, pd.Series)
            else pd.Series(np.nan, index=data.index, dtype="float64")
        )
        self.signal_stop_loss_prices = (
            self._stop_loss_prices.reindex(data.index)
            if isinstance(self._stop_loss_prices, pd.Series)
            else pd.Series(np.nan, index=data.index, dtype="float64")
        )
        self.signal_exit_reason = (
            self._exit_reasons.reindex(data.index).fillna("").astype("object")
            if isinstance(self._exit_reasons, pd.Series)
            else pd.Series("", index=data.index, dtype="object")
        )
        self.signal_first_wiseman_setup_side = (
            self._first_wiseman_setup_side.reindex(data.index).fillna(0).astype("int8")
            if isinstance(self._first_wiseman_setup_side, pd.Series)
            else pd.Series(0, index=data.index, dtype="int8")
        )
        self.signal_first_wiseman_ignored_reason = (
            self._first_wiseman_ignored_reason.reindex(data.index).fillna("").astype("object")
            if isinstance(self._first_wiseman_ignored_reason, pd.Series)
            else pd.Series("", index=data.index, dtype="object")
        )
        self.signal_first_wiseman_reversal_side = (
            self._first_wiseman_reversal_side.reindex(data.index).fillna(0).astype("int8")
            if isinstance(self._first_wiseman_reversal_side, pd.Series)
            else pd.Series(0, index=data.index, dtype="int8")
        )
        self.signal_fill_prices_first = (
            self._first_wiseman_fill_prices.reindex(data.index)
            if isinstance(self._first_wiseman_fill_prices, pd.Series)
            else pd.Series(np.nan, index=data.index, dtype="float64")
        )
        self.signal_second_wiseman_fill_side = (
            self._second_wiseman_fill_side.reindex(data.index).fillna(0).astype("int8")
            if isinstance(self._second_wiseman_fill_side, pd.Series)
            else pd.Series(0, index=data.index, dtype="int8")
        )
        self.signal_third_wiseman_fill_side = (
            self._third_wiseman_fill_side.reindex(data.index).fillna(0).astype("int8")
            if isinstance(self._third_wiseman_fill_side, pd.Series)
            else pd.Series(0, index=data.index, dtype="int8")
        )
        self.signal_fractal_position_side = (
            self._fractal_position_side.reindex(data.index).fillna(0).astype("int8")
            if isinstance(self._fractal_position_side, pd.Series)
            else pd.Series(0, index=data.index, dtype="int8")
        )
        return signals


class _IntrabarEventsStrategy:
    execute_on_signal_bar = True

    def __init__(self, intrabar_events: dict[int, list[dict[str, float | int | str]]]) -> None:
        self._intrabar_events = intrabar_events
        self.signal_intrabar_events: dict[int, list[dict[str, float | int | str]]] | None = None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        self.signal_intrabar_events = self._intrabar_events
        return pd.Series(0, index=data.index, dtype="int8")


class _SignalIntrabarEventsStrategy:
    execute_on_signal_bar = True

    def __init__(
        self,
        signals: pd.Series,
        intrabar_events: dict[int, list[dict[str, float | int | str]]],
        reversal_side: pd.Series | None = None,
    ) -> None:
        self._signals = signals
        self._intrabar_events = intrabar_events
        self._reversal_side = reversal_side
        self.signal_intrabar_events: dict[int, list[dict[str, float | int | str]]] | None = None
        self.signal_first_wiseman_reversal_side: pd.Series | None = None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        self.signal_intrabar_events = self._intrabar_events
        self.signal_first_wiseman_reversal_side = (
            self._reversal_side.reindex(data.index).fillna(0).astype("int8")
            if isinstance(self._reversal_side, pd.Series)
            else pd.Series(0, index=data.index, dtype="int8")
        )
        return self._signals.reindex(data.index).fillna(0).astype("int8")


def _sample_df(periods: int = 100, freq: str = "D") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=periods, freq=freq, tz="UTC")
    close = pd.Series(range(periods), index=idx, dtype="float64") + 100
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )


def test_backtest_runs_and_produces_stats() -> None:
    data = _sample_df()
    engine = BacktestEngine(BacktestConfig(initial_capital=10_000, fee_rate=0.0, slippage_rate=0.0))
    result = engine.run(data, SmaCrossoverStrategy(fast=5, slow=20))

    assert len(result.equity_curve) == len(data)
    assert "sharpe" in result.stats
    assert result.stats["total_trades"] >= 1
    assert "missing_bars" in result.data_quality


def test_order_types_and_sizing_modes() -> None:
    data = _sample_df(periods=60, freq="h")
    for order_type in ["market", "limit", "stop", "stop_limit"]:
        engine = BacktestEngine(
            BacktestConfig(
                initial_capital=10_000,
                fee_rate=0.0,
                slippage_rate=0.0,
                order_type=order_type,
                trade_size_mode="usd",
                trade_size_value=1_000,
                spread_rate=0.0001,
                borrow_rate_annual=0.02,
                funding_rate_per_period=0.00001,
                overnight_rate_annual=0.01,
            )
        )
        result = engine.run(data, SmaCrossoverStrategy(fast=5, slow=20))
        assert len(result.equity_curve) == len(data)


def test_max_loss_stops_out_long_position() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 95.0, 95.0],
            "high": [101.0, 101.0, 96.0, 96.0],
            "low": [99.0, 99.0, 85.0, 94.0],
            "close": [100.0, 100.0, 95.0, 95.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([1, 1, 1, 1], index=idx, dtype="int8")

    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="units",
            trade_size_value=10,
            max_loss=100,
        )
    )
    result = engine.run(data, _SignalBarStrategy(signals))

    assert any(event.event_type == "stop_out" for event in result.execution_events)
    stop_trade = next(t for t in result.trades if t.exit_time == idx[2])
    assert stop_trade.pnl == -100.0


def test_max_loss_stop_uses_gap_open_price_when_gapped() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 80.0, 80.0],
            "high": [101.0, 101.0, 90.0, 81.0],
            "low": [99.0, 99.0, 70.0, 79.0],
            "close": [100.0, 100.0, 80.0, 80.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([1, 1, 1, 1], index=idx, dtype="int8")
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="units",
            trade_size_value=10,
            max_loss=100,
        )
    )
    result = engine.run(data, _SignalBarStrategy(signals))

    stop_trade = next(t for t in result.trades if t.exit_time == idx[2])
    assert stop_trade.exit_price == 80.0
    assert stop_trade.pnl == -200.0


def test_max_loss_pct_of_equity_stops_out_long_position() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 95.0, 95.0],
            "high": [101.0, 101.0, 96.0, 96.0],
            "low": [99.0, 99.0, 89.0, 94.0],
            "close": [100.0, 100.0, 95.0, 95.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([1, 1, 1, 1], index=idx, dtype="int8")
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="units",
            trade_size_value=10,
            max_loss_pct_of_equity=0.01,
        )
    )
    result = engine.run(data, _SignalBarStrategy(signals))

    assert any(event.event_type == "stop_out" for event in result.execution_events)
    stop_trade = next(t for t in result.trades if t.exit_time == idx[2])
    assert stop_trade.exit_price == pytest.approx(90.0)
    assert stop_trade.pnl == pytest.approx(-100.0)


def test_max_loss_pct_of_equity_stops_out_short_position() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 105.0, 105.0],
            "high": [101.0, 101.0, 111.0, 106.0],
            "low": [99.0, 99.0, 104.0, 104.0],
            "close": [100.0, 100.0, 105.0, 105.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([-1, -1, -1, -1], index=idx, dtype="int8")
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="units",
            trade_size_value=10,
            max_loss_pct_of_equity=0.01,
        )
    )
    result = engine.run(data, _SignalBarStrategy(signals))

    assert any(event.event_type == "stop_out" for event in result.execution_events)
    stop_trade = next(t for t in result.trades if t.exit_time == idx[2])
    assert stop_trade.exit_price == pytest.approx(110.0)
    assert stop_trade.pnl == pytest.approx(-100.0)


def test_leveraged_position_is_liquidated_before_full_margin_loss() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 90.9, 100.0],
            "close": [100.0, 100.0, 91.0, 100.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([1, 1, 1, 0], index=idx, dtype="int8")
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="units",
            trade_size_value=1_000,
            max_leverage=10.0,
            leverage_stop_out_pct=0.01,
        )
    )
    result = engine.run(data, _SignalBarStrategy(signals))

    assert any(event.event_type == "liquidation" for event in result.execution_events)
    liq_trade = next(t for t in result.trades if t.exit_time == idx[2])
    assert liq_trade.exit_price == pytest.approx(91.0)
    assert liq_trade.pnl == pytest.approx(-9_000.0)


def test_leveraged_liquidation_uses_gap_open_price_when_gapped() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 80.0, 80.0],
            "high": [100.0, 100.0, 85.0, 80.0],
            "low": [100.0, 100.0, 79.0, 80.0],
            "close": [100.0, 100.0, 81.0, 80.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([1, 1, 1, 0], index=idx, dtype="int8")
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="units",
            trade_size_value=1_000,
            max_leverage=10.0,
            leverage_stop_out_pct=0.01,
        )
    )
    result = engine.run(data, _SignalBarStrategy(signals))

    liq_trade = next(t for t in result.trades if t.exit_time == idx[2])
    assert liq_trade.exit_price == pytest.approx(80.0)
    assert liq_trade.pnl == pytest.approx(-20_000.0)


def test_liquidation_fill_applies_execution_adjustments() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 90.9, 100.0],
            "close": [100.0, 100.0, 91.0, 100.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([1, 1, 1, 0], index=idx, dtype="int8")
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.001,
            spread_rate=0.002,
            trade_size_mode="units",
            trade_size_value=1_000,
            max_leverage=10.0,
            leverage_stop_out_pct=0.01,
        )
    )
    result = engine.run(data, _SignalBarStrategy(signals))

    liq_trade = next(t for t in result.trades if t.exit_time == idx[2])
    leverage = (liq_trade.entry_price * liq_trade.units) / 10_000
    adverse_move = (1.0 / leverage) - 0.01
    raw_liq = liq_trade.entry_price * (1.0 - adverse_move)
    # Liquidation is a sell for a long position, so execution adjustment should be adverse.
    assert liq_trade.exit_price == pytest.approx(raw_liq * (1 - 0.001 - 0.001))


def test_engine_raises_on_non_finite_capital_state() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [1e308, 1e308, 1e308, 1e308, 1e308, 1e308],
            "high": [1e308, 1e308, 1e308, 1e308, 1e308, 1e308],
            "low": [1e308, 1e308, 1e308, 1e308, 1e308, 1e308],
            "close": [1e308, 1e308, 1e308, 1e308, 1e308, 1e308],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    class _FlipSignals:
        execute_on_signal_bar = False

        def generate_signals(self, frame: pd.DataFrame) -> pd.Series:
            # Force repeated enter/exit transitions so fee math is exercised.
            return pd.Series([1, -1, 1, -1, 1, -1], index=frame.index, dtype="int8")

    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=1.0,
            slippage_rate=0.0,
            trade_size_mode="units",
            trade_size_value=1e308,
        )
    )

    with pytest.raises(ValueError, match="non-finite"):
        engine.run(data, _FlipSignals())


def test_short_leveraged_position_is_liquidated_before_full_margin_loss() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 109.1, 100.0],
            "low": [100.0, 100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 109.0, 100.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([-1, -1, -1, 0], index=idx, dtype="int8")
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="units",
            trade_size_value=1_000,
            max_leverage=10.0,
            leverage_stop_out_pct=0.01,
        )
    )
    result = engine.run(data, _SignalBarStrategy(signals))

    assert any(event.event_type == "liquidation" for event in result.execution_events)
    liq_trade = next(t for t in result.trades if t.exit_time == idx[2])
    assert liq_trade.exit_price == pytest.approx(109.0)
    assert liq_trade.pnl == pytest.approx(-9_000.0)


def test_signal_flip_fill_executes_before_same_bar_non_gap_short_liquidation() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 99.0, 100.0, 100.0],
            "high": [101.0, 101.0, 200.0, 101.0, 101.0],
            "low": [99.0, 99.0, 98.0, 99.0, 99.0],
            "close": [100.0, 100.0, 99.0, 100.0, 100.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([0, -1, 1, 1, 0], index=idx, dtype="int8")
    fills = pd.Series([np.nan, 100.0, 100.0, np.nan, np.nan], index=idx, dtype="float64")
    stop_losses = pd.Series([np.nan] * len(idx), index=idx, dtype="float64")

    result = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="percent_of_equity",
            trade_size_value=2.0,
            leverage_stop_out_pct=0.0,
        )
    ).run(
        data,
        _SignalFillStopStrategy(signals, fills, stop_losses),
    )

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert [event.event_type for event in bar_events[:2]] == ["exit", "entry"]
    assert [event.side for event in bar_events[:2]] == ["buy", "buy"]
    assert not any(event.event_type == "liquidation" for event in bar_events)
    assert result.positions.iloc[2] == 1


def test_signal_flip_fill_executes_before_same_bar_short_strategy_stop_loss() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 99.0, 100.0, 100.0],
            "high": [101.0, 101.0, 200.0, 101.0, 101.0],
            "low": [99.0, 99.0, 98.0, 99.0, 99.0],
            "close": [100.0, 100.0, 99.0, 100.0, 100.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([0, -1, 1, 1, 0], index=idx, dtype="int8")
    fills = pd.Series([np.nan, 100.0, 100.0, np.nan, np.nan], index=idx, dtype="float64")
    stop_losses = pd.Series([np.nan, 120.0, 120.0, np.nan, np.nan], index=idx, dtype="float64")

    result = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="percent_of_equity",
            trade_size_value=2.0,
            leverage_stop_out_pct=0.0,
        )
    ).run(
        data,
        _SignalFillStopStrategy(signals, fills, stop_losses),
    )

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert [event.event_type for event in bar_events[:2]] == ["exit", "entry"]
    assert [event.side for event in bar_events[:2]] == ["buy", "buy"]
    assert all("Strategy Stop Loss" not in str(event.strategy_reason) for event in bar_events)
    assert result.positions.iloc[2] == 1


def test_first_wiseman_reversal_fill_blocks_hold_guard_and_preempts_same_bar_short_stop() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 99.0, 100.0, 100.0],
            "high": [101.0, 101.0, 200.0, 101.0, 101.0],
            "low": [99.0, 99.0, 98.0, 99.0, 99.0],
            "close": [100.0, 100.0, 99.0, 100.0, 100.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([0, -1, 1, 1, 0], index=idx, dtype="int8")
    contracts = pd.Series([0.0, -1.0, 1.0, 1.0, 0.0], index=idx, dtype="float64")
    fill_prices = pd.Series([np.nan] * len(idx), index=idx, dtype="float64")
    first_fill_prices = pd.Series([np.nan, np.nan, 100.0, np.nan, np.nan], index=idx, dtype="float64")
    reversal_side = pd.Series([0, 0, 1, 0, 0], index=idx, dtype="int8")
    stop_losses = pd.Series([np.nan, 120.0, 120.0, np.nan, np.nan], index=idx, dtype="float64")

    strategy = _SignalContractsExitReasonStrategy(
        signals,
        contracts,
        fill_prices=fill_prices,
        first_wiseman_reversal_side=reversal_side,
        first_wiseman_fill_prices=first_fill_prices,
        stop_loss_prices=stop_losses,
    )
    result = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="percent_of_equity",
            trade_size_value=2.0,
            leverage_stop_out_pct=0.0,
        )
    ).run(data, strategy)

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert [event.event_type for event in bar_events[:2]] == ["exit", "entry"]
    assert all("Strategy Stop Loss" not in str(event.strategy_reason) for event in bar_events)
    assert result.positions.iloc[2] == 1


def test_first_wiseman_fill_without_reversal_marker_preempts_same_bar_short_stop() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 99.0, 100.0, 100.0],
            "high": [101.0, 101.0, 200.0, 101.0, 101.0],
            "low": [99.0, 99.0, 98.0, 99.0, 99.0],
            "close": [100.0, 100.0, 99.0, 100.0, 100.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([0, -1, 1, 1, 0], index=idx, dtype="int8")
    contracts = pd.Series([0.0, -1.0, 1.0, 1.0, 0.0], index=idx, dtype="float64")
    fill_prices = pd.Series([np.nan] * len(idx), index=idx, dtype="float64")
    first_fill_prices = pd.Series([np.nan, np.nan, 100.0, np.nan, np.nan], index=idx, dtype="float64")
    reversal_side = pd.Series([0, 0, 0, 0, 0], index=idx, dtype="int8")
    stop_losses = pd.Series([np.nan, 120.0, 120.0, np.nan, np.nan], index=idx, dtype="float64")

    strategy = _SignalContractsExitReasonStrategy(
        signals,
        contracts,
        fill_prices=fill_prices,
        first_wiseman_reversal_side=reversal_side,
        first_wiseman_fill_prices=first_fill_prices,
        stop_loss_prices=stop_losses,
    )
    result = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="percent_of_equity",
            trade_size_value=2.0,
        )
    ).run(data, strategy)

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert [event.event_type for event in bar_events[:2]] == ["exit", "entry"]
    assert all("Strategy Stop Loss" not in str(event.strategy_reason) for event in bar_events)
    assert result.positions.iloc[2] == 1


def test_opposite_flip_with_missing_primary_fill_series_still_executes_before_short_stop() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 99.0, 100.0, 100.0],
            "high": [101.0, 101.0, 200.0, 101.0, 101.0],
            "low": [99.0, 99.0, 98.0, 99.0, 99.0],
            "close": [100.0, 100.0, 99.0, 100.0, 100.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([0, -1, 1, 1, 0], index=idx, dtype="int8")
    contracts = pd.Series([0.0, -1.0, 1.0, 1.0, 0.0], index=idx, dtype="float64")
    fill_prices = pd.Series([np.nan] * len(idx), index=idx, dtype="float64")
    stop_losses = pd.Series([np.nan, 120.0, 120.0, np.nan, np.nan], index=idx, dtype="float64")

    strategy = _SignalContractsExitReasonStrategy(
        signals,
        contracts,
        fill_prices=fill_prices,
        stop_loss_prices=stop_losses,
    )
    result = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="percent_of_equity",
            trade_size_value=2.0,
        )
    ).run(data, strategy)

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert [event.event_type for event in bar_events[:2]] == ["exit", "entry"]
    assert all("Strategy Stop Loss" not in str(event.strategy_reason) for event in bar_events)
    assert result.positions.iloc[2] == 1


def test_first_wiseman_setup_side_can_force_flip_before_same_bar_short_stop_when_signal_is_stale() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 99.0, 100.0, 100.0],
            "high": [101.0, 101.0, 200.0, 101.0, 101.0],
            "low": [99.0, 99.0, 98.0, 99.0, 99.0],
            "close": [100.0, 100.0, 99.0, 100.0, 100.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    # Keep signal stale short on the conflict bar while first-W metadata carries bullish 1W fill.
    signals = pd.Series([0, -1, -1, 1, 0], index=idx, dtype="int8")
    contracts = pd.Series([0.0, -1.0, -1.0, 1.0, 0.0], index=idx, dtype="float64")
    fill_prices = pd.Series([np.nan] * len(idx), index=idx, dtype="float64")
    first_fill_prices = pd.Series([np.nan, np.nan, 100.0, np.nan, np.nan], index=idx, dtype="float64")
    first_setup_side = pd.Series([0, 0, 1, 0, 0], index=idx, dtype="int8")
    stop_losses = pd.Series([np.nan, 120.0, 120.0, np.nan, np.nan], index=idx, dtype="float64")

    strategy = _SignalContractsExitReasonStrategy(
        signals,
        contracts,
        fill_prices=fill_prices,
        first_wiseman_setup_side=first_setup_side,
        first_wiseman_fill_prices=first_fill_prices,
        stop_loss_prices=stop_losses,
    )
    result = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="percent_of_equity",
            trade_size_value=2.0,
        )
    ).run(data, strategy)

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert [event.event_type for event in bar_events[:2]] == ["exit", "entry"]
    assert all("Strategy Stop Loss" not in str(event.strategy_reason) for event in bar_events)
    assert result.positions.iloc[2] == 1


def test_gap_open_through_first_w_fill_still_preempts_short_stop_same_bar() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 110.0, 111.0, 112.0],
            "high": [101.0, 101.0, 130.0, 112.0, 113.0],
            "low": [99.0, 99.0, 109.0, 110.0, 111.0],
            "close": [100.0, 100.0, 111.0, 112.0, 112.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    # Conflict bar opens above bullish 1W fill level (already crossed at open),
    # then keeps rising into short stop region.
    signals = pd.Series([0, -1, -1, 1, 0], index=idx, dtype="int8")
    contracts = pd.Series([0.0, -1.0, -1.0, 1.0, 0.0], index=idx, dtype="float64")
    fill_prices = pd.Series([np.nan] * len(idx), index=idx, dtype="float64")
    first_fill_prices = pd.Series([np.nan, np.nan, 100.0, np.nan, np.nan], index=idx, dtype="float64")
    first_setup_side = pd.Series([0, 0, 1, 0, 0], index=idx, dtype="int8")
    stop_losses = pd.Series([np.nan, 120.0, 120.0, np.nan, np.nan], index=idx, dtype="float64")

    strategy = _SignalContractsExitReasonStrategy(
        signals,
        contracts,
        fill_prices=fill_prices,
        first_wiseman_setup_side=first_setup_side,
        first_wiseman_fill_prices=first_fill_prices,
        stop_loss_prices=stop_losses,
    )
    result = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="percent_of_equity",
            trade_size_value=2.0,
        )
    ).run(data, strategy)

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert [event.event_type for event in bar_events[:2]] == ["exit", "entry"]
    assert all("Strategy Stop Loss" not in str(event.strategy_reason) for event in bar_events)
    assert result.positions.iloc[2] == 1


def test_same_bar_closest_to_open_order_cancels_downstream_short_stop_and_liquidation() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 99.0, 100.0, 100.0],
            "high": [101.0, 101.0, 200.0, 101.0, 101.0],
            "low": [99.0, 99.0, 98.0, 99.0, 99.0],
            "close": [100.0, 100.0, 99.0, 100.0, 100.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([0, -1, 1, 1, 0], index=idx, dtype="int8")
    fills = pd.Series([np.nan, 100.0, 100.0, np.nan, np.nan], index=idx, dtype="float64")
    stop_losses = pd.Series([np.nan, 120.0, 120.0, np.nan, np.nan], index=idx, dtype="float64")

    result = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="percent_of_equity",
            trade_size_value=2.0,
            leverage_stop_out_pct=0.0,
        )
    ).run(
        data,
        _SignalFillStopStrategy(signals, fills, stop_losses),
    )

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert [event.event_type for event in bar_events[:2]] == ["exit", "entry"]
    assert all("Strategy Stop Loss" not in str(event.strategy_reason) for event in bar_events)
    assert not any(event.event_type == "liquidation" for event in bar_events)
    assert result.positions.iloc[2] == 1


def test_intrabar_events_execute_by_open_proximity_and_cancel_stale_short_stop() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 160.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([0, -1, 1, 1], index=idx, dtype="int8")
    intrabar_events = {
        2: [
            # Intentionally unsorted: stale short stop first, then closer bullish trigger.
            {"event_type": "exit", "side": 1, "price": 110.0, "contracts": 0.0, "reason": "Strategy Stop Loss Bearish 1W"},
            {"event_type": "entry", "side": 1, "price": 101.0, "contracts": 1.0, "reason": "Bullish 1W"},
        ]
    }

    result = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="percent_of_equity",
            trade_size_value=2.0,
            leverage_stop_out_pct=0.0,
        )
    ).run(data, _SignalIntrabarEventsStrategy(signals, intrabar_events))

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert [event.event_type for event in bar_events[:2]] == ["exit", "entry"]
    assert bar_events[0].price == pytest.approx(101.0)
    assert bar_events[1].price == pytest.approx(101.0)
    assert all(str(event.strategy_reason) != "Strategy Stop Loss Bearish 1W" for event in bar_events)
    assert not any(event.event_type == "liquidation" for event in bar_events)
    assert result.positions.iloc[2] == 1


def test_intrabar_closest_first_reversal_can_override_stale_desired_side() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 160.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    # Signal intent stays short on the conflict bar, but explicit intrabar 1W
    # reversal event should still execute by price path priority.
    signals = pd.Series([0, -1, -1, -1], index=idx, dtype="int8")
    intrabar_events = {
        2: [
            {"event_type": "exit", "side": 1, "price": 110.0, "contracts": 0.0, "reason": "Strategy Stop Loss Bearish 1W"},
            {"event_type": "entry", "side": 1, "price": 101.0, "contracts": 1.0, "reason": "Bullish 1W"},
        ]
    }

    result = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="percent_of_equity",
            trade_size_value=2.0,
            leverage_stop_out_pct=0.0,
        )
    ).run(data, _SignalIntrabarEventsStrategy(signals, intrabar_events))

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert [event.event_type for event in bar_events[:2]] == ["exit", "entry"]
    assert bar_events[0].price == pytest.approx(101.0)
    assert bar_events[1].price == pytest.approx(101.0)
    assert all(str(event.strategy_reason) != "Strategy Stop Loss Bearish 1W" for event in bar_events)
    assert result.positions.iloc[2] == 1


def test_sub_penny_prices_preserve_same_bar_flip_ordering_over_short_stop() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [0.00001200, 0.00001200, 0.00001234, 0.00001240, 0.00001245],
            "high": [0.00001210, 0.00001210, 0.00001280, 0.00001245, 0.00001250],
            "low": [0.00001190, 0.00001190, 0.00001220, 0.00001230, 0.00001240],
            "close": [0.00001200, 0.00001200, 0.00001234, 0.00001240, 0.00001245],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([0, -1, 1, 1, 0], index=idx, dtype="int8")
    fills = pd.Series([np.nan, 0.00001230, 0.00001235, np.nan, np.nan], index=idx, dtype="float64")
    stop_losses = pd.Series([np.nan, 0.00001260, 0.00001260, np.nan, np.nan], index=idx, dtype="float64")

    result = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="percent_of_equity",
            trade_size_value=1.0,
        )
    ).run(
        data,
        _SignalFillStopStrategy(signals, fills, stop_losses),
    )

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert [event.event_type for event in bar_events[:2]] == ["exit", "entry"]
    assert [event.side for event in bar_events[:2]] == ["buy", "buy"]
    assert all("Strategy Stop Loss" not in str(event.strategy_reason) for event in bar_events)
    assert result.positions.iloc[2] == 1


def test_near_equal_prices_do_not_break_same_bar_flip_priority() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0000000000, 100.0, 100.0],
            "high": [100.1, 100.1, 100.0000003000, 100.1, 100.1],
            "low": [99.9, 99.9, 99.9999999000, 99.9, 99.9],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([0, -1, 1, 1, 0], index=idx, dtype="int8")
    fills = pd.Series([np.nan, 100.0000001000, 100.0000001000, np.nan, np.nan], index=idx, dtype="float64")
    stop_losses = pd.Series([np.nan, 100.0000002000, 100.0000002000, np.nan, np.nan], index=idx, dtype="float64")

    result = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="percent_of_equity",
            trade_size_value=1.0,
        )
    ).run(
        data,
        _SignalFillStopStrategy(signals, fills, stop_losses),
    )

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert [event.event_type for event in bar_events[:2]] == ["exit", "entry"]
    assert all("Strategy Stop Loss" not in str(event.strategy_reason) for event in bar_events)
    assert result.positions.iloc[2] == 1


def test_equity_curve_accumulates_financing_drag_while_position_is_open() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0] * len(idx),
            "high": [100.0] * len(idx),
            "low": [100.0] * len(idx),
            "close": [100.0] * len(idx),
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([1, 1, 1, 1, 1, 1], index=idx, dtype="int8")

    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="units",
            trade_size_value=1,
            funding_rate_per_period=0.01,
        )
    )
    result = engine.run(data, _SignalBarStrategy(signals))

    # 1 unit * $100 notional * 1% funding charged once per loop iteration.
    # With execute_on_signal_bar=True and an always-long signal, there are
    # len(data)-2 financing debits before final flattening because financing is charged on carried positions from prior bars.
    expected_final = 10_000 - ((len(data) - 2) * 1.0)
    assert result.equity_curve.iloc[-1] == expected_final


def test_equity_cutoff_stops_backtest_and_closes_open_position() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 90.0, 80.0, 70.0, 60.0],
            "high": [101.0, 101.0, 91.0, 81.0, 71.0, 61.0],
            "low": [99.0, 99.0, 89.0, 79.0, 69.0, 59.0],
            "close": [100.0, 100.0, 90.0, 80.0, 70.0, 60.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([1, 1, 1, 1, 1, 1], index=idx, dtype="int8")
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="units",
            trade_size_value=100,
            equity_cutoff=8_000,
        )
    )

    result = engine.run(data, _SignalBarStrategy(signals))

    assert any(event.event_type == "equity_cutoff" for event in result.execution_events)
    assert any(event.event_type == "equity_cutoff_exit" for event in result.execution_events)
    cutoff_trade = next(t for t in result.trades if t.exit_time == idx[3])
    assert cutoff_trade.exit_price == 80.0
    assert result.equity_curve.iloc[3] == 8_000
    assert (result.equity_curve.iloc[4:] == 8_000).all()
    assert (result.positions.iloc[4:] == 0).all()


def test_equity_cutoff_stops_backtest_and_closes_open_short_position() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 110.0, 120.0, 130.0, 140.0],
            "high": [101.0, 101.0, 111.0, 121.0, 131.0, 141.0],
            "low": [99.0, 99.0, 109.0, 119.0, 129.0, 139.0],
            "close": [100.0, 100.0, 110.0, 120.0, 130.0, 140.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([-1, -1, -1, -1, -1, -1], index=idx, dtype="int8")
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="units",
            trade_size_value=100,
            equity_cutoff=8_000,
        )
    )

    result = engine.run(data, _SignalBarStrategy(signals))

    assert any(event.event_type == "equity_cutoff" for event in result.execution_events)
    assert any(event.event_type == "equity_cutoff_exit" for event in result.execution_events)
    cutoff_trade = next(t for t in result.trades if t.exit_time == idx[3])
    assert cutoff_trade.exit_price == 120.0
    assert result.equity_curve.iloc[3] == 8_000
    assert (result.equity_curve.iloc[4:] == 8_000).all()
    assert (result.positions.iloc[4:] == 0).all()


def test_same_bar_stop_out_triggers_before_equity_cutoff_and_liquidation() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [110.0, 110.0, 105.0],
            "high": [110.0, 110.0, 106.0],
            "low": [110.0, 105.0, 98.0],
            "close": [110.0, 105.0, 98.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([1, 1, 1], index=idx, dtype="int8")
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="percent_of_equity",
            trade_size_value=10.0,
            max_loss=9_090,
            equity_cutoff=500,
            max_leverage=10.0,
            leverage_stop_out_pct=0.0,
        )
    )

    result = engine.run(data, _SignalBarStrategy(signals))

    assert any(event.event_type == "stop_out" for event in result.execution_events)
    assert not any(event.event_type == "liquidation" for event in result.execution_events)
    assert not any(event.event_type == "equity_cutoff" for event in result.execution_events)
    stop_trade = next(t for t in result.trades if t.exit_time == idx[2])
    assert stop_trade.exit_price == pytest.approx(100.0, rel=1e-3)


def test_gap_below_liquidation_ignores_stop_out_and_equity_cutoff() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [110.0, 110.0, 98.5],
            "high": [110.0, 110.0, 99.0],
            "low": [110.0, 105.0, 98.0],
            "close": [110.0, 105.0, 98.5],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([1, 1, 1], index=idx, dtype="int8")
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="percent_of_equity",
            trade_size_value=10.0,
            max_loss=9_090,
            equity_cutoff=500,
            max_leverage=10.0,
            leverage_stop_out_pct=0.0,
        )
    )

    result = engine.run(data, _SignalBarStrategy(signals))

    assert any(event.event_type == "liquidation" for event in result.execution_events)
    assert not any(event.event_type == "stop_out" for event in result.execution_events)
    liq_trade = next(t for t in result.trades if t.exit_time == idx[2])
    assert liq_trade.exit_price == pytest.approx(98.5)


def test_backtest_stops_at_bankruptcy_floor() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 1.0, 1.0, 1.0],
            "high": [100.0, 100.0, 1.0, 1.0, 1.0],
            "low": [100.0, 100.0, 1.0, 1.0, 1.0],
            "close": [100.0, 100.0, 1.0, 1.0, 1.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([1, 1, 1, 1, 1], index=idx, dtype="int8")
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="units",
            trade_size_value=1_000,
        )
    )

    result = engine.run(data, _SignalBarStrategy(signals))

    assert result.equity_curve.min() == 0.0
    assert result.stats["max_drawdown"] == -1.0
    assert (result.equity_curve.loc[idx[2:]] == 0.0).all()
    assert (result.positions.loc[idx[2:]] == 0).all()


def test_final_forced_flatten_uses_close_not_order_type_fill() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 120.0],
            "high": [101.0, 101.0, 101.0, 121.0],
            "low": [99.0, 99.0, 99.0, 119.0],
            "close": [100.0, 100.0, 100.0, 80.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([1, 1, 1, 1], index=idx, dtype="int8")
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            order_type="market",
            trade_size_mode="units",
            trade_size_value=1,
            close_open_position_on_last_bar=True,
        )
    )

    result = engine.run(data, _SignalBarStrategy(signals))

    final_trade = result.trades[-1]
    assert final_trade.exit_time == idx[-1]
    assert final_trade.exit_price == pytest.approx(80.0)
    assert final_trade.pnl == pytest.approx(-20.0)


def test_final_forced_flatten_can_be_disabled() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 120.0],
            "high": [101.0, 101.0, 101.0, 121.0],
            "low": [99.0, 99.0, 99.0, 119.0],
            "close": [100.0, 100.0, 100.0, 80.0],
            "volume": [1_000] * len(idx),
        },
        index=idx,
    )

    signals = pd.Series([1, 1, 1, 1], index=idx, dtype="int8")
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            order_type="market",
            trade_size_mode="units",
            trade_size_value=1,
            close_open_position_on_last_bar=False,
        )
    )

    result = engine.run(data, _SignalBarStrategy(signals))

    assert result.trades == []
    assert result.positions.iloc[-1] == 1
    assert result.equity_curve.iloc[-1] == pytest.approx(9_980.0)
    assert not any(event.event_type == "exit" and event.time == idx[-1] for event in result.execution_events)

def test_hourly_period_inference_and_trade_ledger() -> None:
    data = _sample_df(periods=200, freq="h")
    periods = infer_periods_per_year(data.index)
    assert 8_700 <= periods <= 8_800

    engine = BacktestEngine(BacktestConfig(initial_capital=10_000, fee_rate=0.0, slippage_rate=0.0))
    result = engine.run(data, SmaCrossoverStrategy(fast=5, slow=20))
    trades_df = result.trades_dataframe()

    assert "entry_time" in trades_df.columns
    assert "exit_time" in trades_df.columns


def test_resample_ohlcv() -> None:
    data = _sample_df(periods=24, freq="h")
    data_4h = resample_ohlcv(data, "4h")

    assert len(data_4h) == 6
    assert set(["open", "high", "low", "close", "volume"]).issubset(data_4h.columns)


def test_resample_ohlcv_rejects_upsampling() -> None:
    data = _sample_df(periods=24, freq="h")

    with pytest.raises(ValueError, match="Cannot upsample OHLCV bars"):
        resample_ohlcv(data, "5m")


def test_infer_source_timeframe_label_detects_5m() -> None:
    data = _sample_df(periods=24, freq="5min")

    assert infer_source_timeframe_label(data.index) == "5m"


def test_local_chart_generation(tmp_path) -> None:
    data = _sample_df()
    engine = BacktestEngine(BacktestConfig(initial_capital=10_000, fee_rate=0.0, slippage_rate=0.0))
    result = engine.run(data, SmaCrossoverStrategy(fast=5, slow=20))

    out = tmp_path / "chart.html"
    path = generate_local_tradingview_chart(data, result.trades, str(out))

    html = out.read_text(encoding="utf-8")
    assert path.endswith("chart.html")
    assert "lightweight-charts" in html
    assert "setMarkers" in html
    assert "renderTradeEventLines" in html
    assert "renderTradePathLines" in html
    assert "lineStyle: 2" in html
    assert "Alligator Jaw (13, shift 8)" in html
    assert "AO Histogram (5,34) Log-Scaled %" in html
    assert "Williams AC Histogram (5,34,5) Log-Scaled %" in html


def test_chart_limits_fractal_markers_to_valid_third_wiseman_setups(tmp_path) -> None:
    idx = pd.date_range("2024-01-01", periods=9, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10, 11, 12, 13, 12, 11, 12, 11, 10],
            "high": [11, 12, 13, 15, 13, 12, 14, 12, 11],
            "low": [9, 10, 11, 12, 10, 9, 10, 9, 8],
            "close": [10.5, 11.5, 12.5, 13.5, 11.5, 10.5, 13, 10.5, 9.5],
            "volume": [1_000] * 9,
        },
        index=idx,
    )

    out = tmp_path / "dry_run_chart.html"
    generate_local_tradingview_chart(data, [], str(out), title="Indicator Dry-Run (No Trade Markers)")

    html = out.read_text(encoding="utf-8")
    assert "\"text\": \"F\"" not in html


def test_second_wiseman_markers_are_labeled_2w() -> None:
    data = _sample_df(periods=4, freq="h")
    fills = pd.Series(np.nan, index=data.index, dtype="float64")
    fills.iloc[1] = float(data["low"].iloc[1])
    fills.iloc[2] = float(data["high"].iloc[2])

    markers = _second_wiseman_markers(data, fills)

    assert len(markers) == 2
    assert {marker["text"] for marker in markers} == {"2W"}
    assert {marker["shape"] for marker in markers} == {"arrowDown", "arrowUp"}


def test_second_wiseman_markers_use_setup_side_when_provided() -> None:
    data = _sample_df(periods=4, freq="h")
    fills = pd.Series(np.nan, index=data.index, dtype="float64")
    setup_side = pd.Series(0, index=data.index, dtype="int8")
    setup_side.iloc[1] = -1
    setup_side.iloc[2] = 1

    markers = _second_wiseman_markers(data, fills, setup_side)

    assert len(markers) == 2
    by_time = {marker["time"]: marker for marker in markers}

    short_marker = by_time[int(data.index[1].timestamp())]
    assert short_marker["text"] == "2W"
    assert short_marker["position"] == "aboveBar"
    assert short_marker["shape"] == "arrowDown"
    assert short_marker["color"] == "#dc2626"

    long_marker = by_time[int(data.index[2].timestamp())]
    assert long_marker["text"] == "2W"
    assert long_marker["position"] == "belowBar"
    assert long_marker["shape"] == "arrowUp"
    assert long_marker["color"] == "#16a34a"

def test_wiseman_fill_entry_markers_use_se_le_labels() -> None:
    data = _sample_df(periods=4, freq="h")
    fills = pd.Series(np.nan, index=data.index, dtype="float64")
    fills.iloc[1] = float(data["low"].iloc[1])
    fills.iloc[2] = float(data["high"].iloc[2])

    markers = _wiseman_fill_entry_markers(data, fills)

    assert len(markers) == 2
    assert {marker["text"] for marker in markers} == {"SE", "LE"}


def test_wiseman_fill_entry_markers_use_fill_side_when_provided() -> None:
    data = _sample_df(periods=4, freq="h")
    fills = pd.Series(np.nan, index=data.index, dtype="float64")
    fills.iloc[1] = float(data["low"].iloc[1])
    fills.iloc[2] = float(data["high"].iloc[2])
    fill_side = pd.Series(0, index=data.index, dtype="int8")
    fill_side.iloc[1] = 1
    fill_side.iloc[2] = -1

    markers = _wiseman_fill_entry_markers(data, fills, fill_side)

    assert len(markers) == 2
    by_time = {marker["time"]: marker for marker in markers}

    long_marker = by_time[int(data.index[1].timestamp())]
    assert long_marker["text"] == "LE"
    assert long_marker["position"] == "belowBar"
    assert long_marker["shape"] == "arrowUp"

    short_marker = by_time[int(data.index[2].timestamp())]
    assert short_marker["text"] == "SE"
    assert short_marker["position"] == "aboveBar"
    assert short_marker["shape"] == "arrowDown"


def test_wiseman_fill_entry_markers_support_context_labels() -> None:
    data = _sample_df(periods=4, freq="h")
    fills = pd.Series(np.nan, index=data.index, dtype="float64")
    fills.iloc[1] = float(data["low"].iloc[1])
    fills.iloc[2] = float(data["high"].iloc[2])
    fill_side = pd.Series(0, index=data.index, dtype="int8")
    fill_side.iloc[1] = 1
    fill_side.iloc[2] = -1

    markers = _wiseman_fill_entry_markers(data, fills, fill_side, label="2W")

    assert len(markers) == 2
    by_time = {marker["time"]: marker for marker in markers}
    assert by_time[int(data.index[1].timestamp())]["text"] == "LE-2W"
    assert by_time[int(data.index[2].timestamp())]["text"] == "SE-2W"


def test_third_wiseman_markers_respect_engine_setup_side() -> None:
    data = _sample_df(periods=6, freq="h")
    third_setup_side = pd.Series(0, index=data.index, dtype="int8")
    third_setup_side.iloc[2] = -1
    third_setup_side.iloc[4] = 1

    markers = _valid_third_wiseman_fractal_markers(data, third_setup_side)

    assert markers == []


def test_chart_shows_ignored_first_wiseman_reason_markers(tmp_path) -> None:
    data = _sample_df(periods=6, freq="h")
    setup_side = pd.Series(0, index=data.index, dtype="int8")
    setup_side.iloc[2] = 1
    ignored_reason = pd.Series("", index=data.index, dtype="object")
    ignored_reason.iloc[2] = "no_breakout_until_end_of_data"

    out = tmp_path / "chart_with_ignored_1w.html"
    generate_local_tradingview_chart(
        data,
        [],
        str(out),
        first_setup_side=setup_side,
        first_ignored_reason=ignored_reason,
    )

    html = out.read_text(encoding="utf-8")
    assert "1W-N" in html


def test_chart_shows_alligator_filter_reason_marker(tmp_path) -> None:
    data = _sample_df(periods=6, freq="h")
    setup_side = pd.Series(0, index=data.index, dtype="int8")
    setup_side.iloc[2] = -1
    ignored_reason = pd.Series("", index=data.index, dtype="object")
    ignored_reason.iloc[2] = "gator_closed_canceled"

    out = tmp_path / "chart_with_gator_reason.html"
    generate_local_tradingview_chart(
        data,
        [],
        str(out),
        first_setup_side=setup_side,
        first_ignored_reason=ignored_reason,
    )

    html = out.read_text(encoding="utf-8")
    assert "1W-A" in html


def test_chart_shows_gator_open_percentile_filter_reason_marker(tmp_path) -> None:
    data = _sample_df(periods=6, freq="h")
    setup_side = pd.Series(0, index=data.index, dtype="int8")
    setup_side.iloc[2] = 1
    ignored_reason = pd.Series("", index=data.index, dtype="object")
    ignored_reason.iloc[2] = "gator_open_percentile_filter"

    out = tmp_path / "chart_with_gator_open_percentile_reason.html"
    generate_local_tradingview_chart(
        data,
        [],
        str(out),
        first_setup_side=setup_side,
        first_ignored_reason=ignored_reason,
    )

    html = out.read_text(encoding="utf-8")
    assert "1W-G" in html


def test_chart_shows_weaker_label(tmp_path) -> None:
    data = _sample_df(periods=8, freq="h")
    setup_side = pd.Series(0, index=data.index, dtype="int8")
    setup_side.iloc[3] = -1
    ignored_reason = pd.Series("", index=data.index, dtype="object")
    ignored_reason.iloc[3] = "weaker_than_active_setup"

    out = tmp_path / "chart_with_sup_wea_reason.html"
    generate_local_tradingview_chart(
        data,
        [],
        str(out),
        first_setup_side=setup_side,
        first_ignored_reason=ignored_reason,
    )

    html = out.read_text(encoding="utf-8")
    assert "1W-W" in html


def test_chart_shows_cooldown_label(tmp_path) -> None:
    data = _sample_df(periods=8, freq="h")
    setup_side = pd.Series(0, index=data.index, dtype="int8")
    setup_side.iloc[3] = 1
    ignored_reason = pd.Series("", index=data.index, dtype="object")
    ignored_reason.iloc[3] = "reversal_cooldown_active"

    out = tmp_path / "chart_with_cooldown_reason.html"
    generate_local_tradingview_chart(
        data,
        [],
        str(out),
        first_setup_side=setup_side,
        first_ignored_reason=ignored_reason,
    )

    html = out.read_text(encoding="utf-8")
    assert "1W-CD" in html


def test_chart_shows_ao_divergence_filtered_label(tmp_path) -> None:
    data = _sample_df(periods=8, freq="h")
    setup_side = pd.Series(0, index=data.index, dtype="int8")
    setup_side.iloc[3] = -1
    ignored_reason = pd.Series("", index=data.index, dtype="object")
    ignored_reason.iloc[3] = "ao_divergence_filter"

    out = tmp_path / "chart_with_divergence_reason.html"
    generate_local_tradingview_chart(
        data,
        [],
        str(out),
        first_setup_side=setup_side,
        first_ignored_reason=ignored_reason,
    )

    html = out.read_text(encoding="utf-8")
    assert "1W-D" in html

def test_engine_driven_first_wiseman_markers_place_reversal_on_engine_bar() -> None:
    data = _sample_df(periods=8, freq="h")
    setup_side = pd.Series(0, index=data.index, dtype="int8")
    ignored_reason = pd.Series("", index=data.index, dtype="object")
    reversal_side = pd.Series(0, index=data.index, dtype="int8")

    setup_side.iloc[2] = 1
    reversal_side.iloc[5] = -1

    markers = _first_wiseman_engine_markers(
        data,
        setup_side,
        ignored_reason,
        reversal_side,
    )

    assert len(markers) == 2
    by_time = {marker["time"]: marker for marker in markers}
    setup_marker = by_time[int(data.index[2].timestamp())]
    reverse_marker = by_time[int(data.index[5].timestamp())]
    assert setup_marker["text"] == "1W"
    assert reverse_marker["text"] == "1W-R"

def test_ao_histogram_points_preserve_full_timeline() -> None:
    data = _sample_df(periods=60, freq="h")
    points, _ = _ao_histogram_from_data(data)

    assert len(points) == len(data)
    assert all(point["time"] for point in points)
    assert any("value" not in point for point in points)


def test_alligator_ao_strategy_emits_valid_signals() -> None:
    data = _sample_df(periods=150, freq="h")
    signals = AlligatorAOStrategy().generate_signals(data)
    assert len(signals) == len(data)
    assert set(signals.unique()).issubset({-1, 0, 1})


def test_wiseman_reversal_levels_trigger_only_once() -> None:
    idx = pd.date_range("2024-01-01", periods=80, freq="h", tz="UTC")
    base = np.linspace(100, 140, len(idx))

    open_ = base.copy()
    close = base + 0.2
    high = np.maximum(open_, close) + 0.8
    low = np.minimum(open_, close) - 0.8

    # Seed a bearish 1W setup at bar 50, then trigger at 51 (short).
    open_[50], close[50], high[50], low[50] = 135.0, 132.0, 136.0, 131.0
    open_[51], close[51], high[51], low[51] = 131.5, 131.0, 133.0, 130.0
    open_[52], close[52], high[52], low[52] = 131.2, 131.0, 133.2, 130.0

    # First reversal (1W-R) at 55: cross back above original setup high.
    open_[55], close[55], high[55], low[55] = 133.0, 134.0, 137.0, 132.0

    for i in [56, 57, 58]:
        open_[i], close[i], high[i], low[i] = 133.0, 133.2, 134.0, 132.2

    # Cross back below the same setup low at 59, then above setup high again at 63.
    open_[59], close[59], high[59], low[59] = 131.0, 130.8, 132.0, 130.5
    for i in [60, 61, 62]:
        open_[i], close[i], high[i], low[i] = 131.0, 131.2, 132.0, 130.8
    open_[63], close[63], high[63], low[63] = 134.0, 135.0, 137.2, 133.0

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    signals = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001).generate_signals(data)

    assert signals.iloc[52] == -1
    assert signals.iloc[53] == -1
    assert signals.iloc[55] == 1
    # After the first valid reversal back to long, the original levels are disarmed.
    assert signals.iloc[59] == 1
    assert signals.iloc[63] == 1


def test_wiseman_reversal_waits_until_t_plus_3() -> None:
    idx = pd.date_range("2024-01-01", periods=80, freq="h", tz="UTC")
    base = np.linspace(100, 140, len(idx))

    open_ = base.copy()
    close = base + 0.2
    high = np.maximum(open_, close) + 0.8
    low = np.minimum(open_, close) - 0.8

    # Bearish 1W setup at t+0 (bar 50), trigger short at bar 51.
    open_[50], close[50], high[50], low[50] = 135.0, 132.0, 136.0, 131.0
    open_[51], close[51], high[51], low[51] = 131.5, 131.0, 133.0, 130.0

    # Entry occurs immediately on t+1 breakout (bar 51).

    # Cross again at t+3 (bar 53): reversal is now allowed.
    open_[53], close[53], high[53], low[53] = 133.0, 133.4, 136.4, 132.0

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    signals = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001).generate_signals(data)

    assert signals.iloc[51] == -1
    assert signals.iloc[53] == 1


def test_wiseman_wait_bars_to_close_fills_at_wait_close_if_limit_is_marketable() -> None:
    idx = pd.date_range("2024-01-01", periods=80, freq="h", tz="UTC")
    base = np.linspace(100, 140, len(idx))

    open_ = base.copy()
    close = base + 0.2
    high = np.maximum(open_, close) + 0.8
    low = np.minimum(open_, close) - 0.8

    # Bearish 1W setup at t+0 (bar 50), limit level at setup low (131.0).
    open_[50], close[50], high[50], low[50] = 135.0, 132.0, 136.0, 131.0

    # Breakouts during the waiting window should not enter.
    open_[51], close[51], high[51], low[51] = 132.0, 132.5, 133.0, 130.0

    # Wait-bar close (t+2) is above setup low. A sell limit at 131.0 is marketable
    # against the close and should fill immediately at the close.
    open_[52], close[52], high[52], low[52] = 132.0, 131.6, 132.4, 131.2

    # No new signal on t+3 because trade already entered at t+2 close.
    open_[53], close[53], high[53], low[53] = 131.0, 130.9, 131.8, 130.4

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001, first_wiseman_wait_bars_to_close=2)
    signals = strategy.generate_signals(data)
    fills = strategy.signal_fill_prices_first

    assert signals.iloc[51] == 0
    assert signals.iloc[52] == -1
    assert signals.iloc[53] == -1
    assert fills is not None
    assert fills.iloc[52] == pytest.approx(131.6)


def test_wiseman_wait_bars_to_close_places_resting_limit_when_wait_close_not_through_level() -> None:
    idx = pd.date_range("2024-01-01", periods=80, freq="h", tz="UTC")
    base = np.linspace(100, 140, len(idx))

    open_ = base.copy()
    close = base + 0.2
    high = np.maximum(open_, close) + 0.8
    low = np.minimum(open_, close) - 0.8

    # Bearish 1W setup at t+0 (bar 50), limit level at setup low (131.0).
    open_[50], close[50], high[50], low[50] = 135.0, 132.0, 136.0, 131.0

    # Breakouts during the waiting window should not enter.
    open_[51], close[51], high[51], low[51] = 132.0, 132.5, 133.0, 130.0
    # Wait-bar close (t+2) is below setup low, so placing a sell limit at 131.0
    # would be non-marketable and should rest.
    open_[52], close[52], high[52], low[52] = 131.0, 130.8, 131.4, 130.6
    # First subsequent bar touches setup low and should fill at limit.
    open_[53], close[53], high[53], low[53] = 131.8, 131.7, 132.0, 130.6

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001, first_wiseman_wait_bars_to_close=2)
    signals = strategy.generate_signals(data)
    fills = strategy.signal_fill_prices_first

    assert signals.iloc[51] == 0
    assert signals.iloc[52] == 0
    assert signals.iloc[53] == -1
    assert fills is not None
    assert fills.iloc[53] == pytest.approx(131.0)


def test_wiseman_wait_bars_to_close_bullish_fills_at_wait_close_if_limit_is_marketable() -> None:
    idx = pd.date_range("2024-01-01", periods=80, freq="h", tz="UTC")
    base = np.linspace(140, 100, len(idx))

    open_ = base.copy()
    close = base - 0.1
    high = base + 0.8
    low = base - 0.8

    # Bullish 1W setup at t+0 (bar 50), limit level at setup high (109.0).
    open_[50], close[50], high[50], low[50] = 105.0, 108.0, 109.0, 104.0
    low[49], low[51] = 106.0, 106.0

    # During waiting bars, setup should not trigger.
    open_[51], close[51], high[51], low[51] = 107.5, 108.0, 109.5, 106.0

    # Wait close (108.6) is below 109.0, so buy limit is marketable and should fill now.
    open_[52], close[52], high[52], low[52] = 108.1, 108.6, 108.9, 107.8

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001, first_wiseman_wait_bars_to_close=2)
    signals = strategy.generate_signals(data)
    fills = strategy.signal_fill_prices_first

    assert signals.iloc[51] == 0
    assert signals.iloc[52] == 1
    assert fills is not None
    assert fills.iloc[52] == pytest.approx(108.6)


def test_wiseman_wait_bars_to_close_bearish_resting_limit_requires_price_touch() -> None:
    idx = pd.date_range("2024-01-01", periods=80, freq="h", tz="UTC")
    base = np.linspace(100, 140, len(idx))

    open_ = base.copy()
    close = base + 0.2
    high = np.maximum(open_, close) + 0.8
    low = np.minimum(open_, close) - 0.8

    # Bearish 1W setup at t+0 (bar 50), limit level at setup low (131.0).
    open_[50], close[50], high[50], low[50] = 135.0, 132.0, 136.0, 131.0

    # During waiting bars, setup should not trigger.
    open_[51], close[51], high[51], low[51] = 132.0, 132.5, 133.0, 130.0

    # Wait close is below setup low, so a resting sell limit is placed at 131.0.
    open_[52], close[52], high[52], low[52] = 130.8, 130.6, 130.9, 130.2

    # Next bar still trades entirely below 131.0; order must remain unfilled.
    open_[53], close[53], high[53], low[53] = 130.7, 130.5, 130.8, 130.1

    # Following bar finally touches 131.0 and should fill at the limit.
    open_[54], close[54], high[54], low[54] = 131.3, 131.1, 131.4, 130.7

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001, first_wiseman_wait_bars_to_close=2)
    signals = strategy.generate_signals(data)
    fills = strategy.signal_fill_prices_first

    assert signals.iloc[52] == 0
    assert signals.iloc[53] == 0
    assert signals.iloc[54] == -1
    assert fills is not None
    assert fills.iloc[54] == pytest.approx(131.0)


def test_wiseman_entry_executes_on_t_plus_1_breakout_bar_and_marker_time() -> None:
    idx = pd.date_range("2024-01-01", periods=80, freq="h", tz="UTC")
    base = np.linspace(100, 140, len(idx))

    open_ = base.copy()
    close = base + 0.2
    high = np.maximum(open_, close) + 0.8
    low = np.minimum(open_, close) - 0.8

    # Bearish 1W setup at t+0 (bar 50), breakout/entry at t+1 (bar 51).
    open_[50], close[50], high[50], low[50] = 135.0, 132.0, 136.0, 131.0
    open_[51], close[51], high[51], low[51] = 131.5, 131.0, 133.0, 130.0

    # Force a later reversal so the initial short trade is closed and captured in trade history.
    open_[55], close[55], high[55], low[55] = 133.0, 134.0, 137.0, 132.0

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001)
    signals = strategy.generate_signals(data)
    fills = strategy.signal_fill_prices_first

    assert signals.iloc[51] == -1
    assert fills is not None
    assert fills.iloc[51] == pytest.approx(131.0)

    engine = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0, spread_rate=0.0))
    result = engine.run(data, strategy)

    first_short_trade = next((trade for trade in result.trades if trade.side == "short"), None)
    assert first_short_trade is not None
    assert first_short_trade.entry_time == idx[51]

    execution_markers = chart_module._trade_execution_markers(result.trades, data)
    short_entry_markers = [marker for marker in execution_markers if str(marker["text"]).startswith("SE-")]
    assert short_entry_markers
    assert int(short_entry_markers[0]["time"]) == int(idx[51].timestamp())


def test_wiseman_setup_requires_two_left_and_one_right_bars() -> None:
    idx = pd.date_range("2024-01-01", periods=40, freq="h", tz="UTC")
    base = np.linspace(100, 120, len(idx))

    open_ = base + 0.4
    close = base + 0.6
    high = base + 1.2
    low = base - 0.8

    # Candidate bearish pivot at bar 20 has only one lower high to its left
    # and only one lower high to its right, so it should be rejected.
    high[18] = 131.0
    high[19] = 129.0
    high[20] = 130.0
    high[21] = 128.5
    open_[20], close[20], low[20] = 129.5, 128.0, 126.8

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001)
    strategy.generate_signals(data)
    setup_side = strategy.signal_first_wiseman_setup_marker_side

    assert setup_side is not None
    assert int(setup_side.iloc[20]) == 0


def test_wiseman_reversal_trade_stops_out_at_setup_to_reversal_extreme() -> None:
    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 140, len(idx))

    open_ = base + 0.3
    close = base + 0.5
    high = base + 1.0
    low = base - 0.7

    # Bearish setup at bar 40, trigger short at 42, then reverse long at 45.
    setup_i = 40
    open_[setup_i], close[setup_i], high[setup_i], low[setup_i] = 130.0, 127.0, 132.0, 126.0
    high[setup_i - 1], high[setup_i + 1] = 129.0, 128.5
    low[42] = 125.0
    high[45] = 132.5

    expected_stop = float(low[setup_i : 46].min())
    low[46] = expected_stop + 0.2
    low[47] = expected_stop + 0.1
    low[48] = expected_stop - 0.1

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001)
    signals = strategy.generate_signals(data)
    fills = strategy.signal_fill_prices

    assert signals.iloc[42] == -1
    assert signals.iloc[45] == 1
    assert signals.iloc[46] == 1
    assert signals.iloc[47] == 1
    assert signals.iloc[48] == 0
    assert fills is not None
    assert fills.iloc[48] == expected_stop


def test_wiseman_reversal_stop_triggers_when_low_touches_stop_level() -> None:
    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 140, len(idx))

    open_ = base + 0.3
    close = base + 0.5
    high = base + 1.0
    low = base - 0.7

    # Bearish setup at bar 40, trigger short at 42, then reverse long at 45.
    setup_i = 40
    open_[setup_i], close[setup_i], high[setup_i], low[setup_i] = 130.0, 127.0, 132.0, 126.0
    high[setup_i - 1], high[setup_i + 1] = 129.0, 128.5
    low[42] = 125.0
    high[45] = 132.5

    expected_stop = float(low[setup_i : 46].min())
    low[46] = expected_stop + 0.2
    low[47] = expected_stop

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001)
    signals = strategy.generate_signals(data)
    fills = strategy.signal_fill_prices

    assert signals.iloc[42] == -1
    assert signals.iloc[45] == 1
    assert signals.iloc[46] == 1
    assert signals.iloc[47] == 0
    assert fills is not None
    assert fills.iloc[47] == expected_stop
def test_wiseman_bullish_reversal_trade_stops_out_at_setup_to_reversal_extreme() -> None:
    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(140, 100, len(idx))

    open_ = base - 0.3
    close = base - 0.5
    high = base + 0.7
    low = base - 1.0

    # Bullish setup at bar 40, trigger long at 42, then reverse short at 45.
    setup_i = 40
    open_[setup_i], close[setup_i], high[setup_i], low[setup_i] = 110.0, 113.0, 114.0, 108.0
    low[setup_i - 1], low[setup_i + 1] = 109.0, 109.5
    high[42] = 114.5
    low[45] = 107.5

    expected_stop = float(high[setup_i : 46].max())
    high[46] = expected_stop - 0.2
    high[47] = expected_stop - 0.1
    high[48] = expected_stop + 0.1

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001)
    signals = strategy.generate_signals(data)
    fills = strategy.signal_fill_prices

    assert signals.iloc[42] == 1
    assert signals.iloc[45] == -1
    assert signals.iloc[45] == -1
    assert signals.iloc[47] == -1
    assert signals.iloc[48] == 0
    assert fills is not None
    assert fills.iloc[48] == expected_stop


def test_wiseman_untriggered_setup_can_trigger_after_newer_setup(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: series + {13: 0.0, 8: 0.1, 5: 0.2}[period],
    )

    idx = pd.date_range("2024-01-01", periods=70, freq="h", tz="UTC")
    base = np.linspace(100, 135, len(idx))

    open_ = base + 0.4
    close = base + 0.6
    high = base + 1.2
    low = base - 0.8

    # Older bearish setup A: created first but not triggered initially.
    setup_a_i = 40
    open_[setup_a_i], close[setup_a_i], high[setup_a_i], low[setup_a_i] = 130.0, 128.0, 132.0, 124.0
    high[setup_a_i - 1], high[setup_a_i + 1] = 129.5, 129.0

    # Keep lows above setup A trigger level for a while.
    for i in [41, 42, 43, 44, 45, 46, 47, 48, 49]:
        low[i] = 126.5

    # Newer bearish setup B: triggered first.
    setup_b_i = 44
    open_[setup_b_i], close[setup_b_i], high[setup_b_i], low[setup_b_i] = 131.0, 129.5, 131.5, 127.2
    high[setup_b_i - 1], high[setup_b_i + 1] = 130.0, 130.0

    # Trigger setup B first, then setup A later.
    low[46] = 127.0
    low[50] = 123.8

    # Avoid invalidating setup A before it triggers.
    for i in range(len(idx)):
        if i < 50 and high[i] > 132.0:
            high[i] = 131.8

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001)
    signals = strategy.generate_signals(data)
    fills = strategy.signal_fill_prices

    assert fills is not None
    assert signals.iloc[50] == -1


def test_wiseman_reversal_uses_original_setup_level_unless_new_extreme(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: series + {13: 0.0, 8: 0.1, 5: 0.2}[period],
    )

    idx = pd.date_range("2024-01-01", periods=80, freq="h", tz="UTC")
    base = np.linspace(100, 140, len(idx))

    open_ = base.copy()
    close = base + 0.2
    high = np.maximum(open_, close) + 0.8
    low = np.minimum(open_, close) - 0.8

    # Initial bearish 1W (t+0): setup at 50, triggers short at 51.
    open_[50], close[50], high[50], low[50] = 135.0, 132.0, 136.0, 131.0
    open_[51], close[51], high[51], low[51] = 131.5, 131.0, 133.0, 130.0
    open_[52], close[52], high[52], low[52] = 131.2, 131.0, 133.2, 130.0

    # Later bearish 1W (t+11) with lower high should NOT replace reversal level.
    open_[61], close[61], high[61], low[61] = 134.0, 132.5, 134.8, 131.7
    high[60], high[62] = 133.8, 133.9

    # Crossing above later high alone should not reverse.
    open_[68], close[68], high[68], low[68] = 133.0, 133.2, 135.0, 131.8

    # Crossing above original t+0 high should reverse.
    open_[69], close[69], high[69], low[69] = 134.0, 134.2, 136.2, 132.5

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001)
    signals = strategy.generate_signals(data)

    assert signals.iloc[52] == -1
    assert signals.iloc[53] == -1
    assert signals.iloc[68] == -1


def test_wiseman_active_1w_stop_triggers_even_when_gator_is_closed() -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=80, freq="h", tz="UTC")
    base = np.linspace(140, 100, len(idx))

    open_ = base.copy()
    close = base - 0.1
    high = base + 0.8
    low = base - 0.8

    # Bullish 1W setup centered at bar 50, then trigger long at bar 51.
    open_[50], close[50], high[50], low[50] = 105.0, 108.0, 109.0, 104.0
    low[49], low[51] = 106.0, 106.0
    high[51], open_[51], close[51] = 109.5, 107.0, 108.0

    # Keep price above setup low, then break it at bar 60.
    for i in range(52, 60):
        high[i], low[i], open_[i], close[i] = 109.0, 106.8, 107.5, 108.0
    high[52] = 110.5
    high[60], low[60] = 108.0, 103.5

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    median_price = (data["high"] + data["low"]) / 2
    jaw = strategy_module._smma(median_price, 13).shift(8)
    teeth = strategy_module._smma(median_price, 8).shift(5)
    lips = strategy_module._smma(median_price, 5).shift(3)
    gator_range = pd.concat([jaw, teeth, lips], axis=1).max(axis=1) - pd.concat([jaw, teeth, lips], axis=1).min(axis=1)
    gator_slope = jaw.diff().abs() + teeth.diff().abs() + lips.diff().abs()
    range_baseline = gator_range.rolling(10, min_periods=1).median()
    slope_baseline = gator_slope.rolling(10, min_periods=1).median()
    gator_closed = (gator_range <= range_baseline) & (gator_slope <= slope_baseline)

    strategy = WisemanStrategy(gator_width_lookback=10, gator_width_mult=1.0)
    signals = strategy.generate_signals(data)

    assert bool(gator_closed.iloc[60])
    assert signals.iloc[52] == 1
    assert signals.iloc[59] == 1
    assert signals.iloc[60] == -1


def test_wiseman_does_not_emit_superseded_reason() -> None:
    data = _sample_df(periods=120, freq="h")
    strategy = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001)
    strategy.generate_signals(data)
    ignored = strategy.signal_first_wiseman_ignored_reason

    assert ignored is not None
    assert "superseded_by_more_extreme" not in set(ignored.astype(str))


def test_wiseman_reversal_cooldown_validation() -> None:
    strategy = WisemanStrategy(first_wiseman_reversal_cooldown=5)
    assert strategy.first_wiseman_reversal_cooldown == 5

    with pytest.raises(ValueError, match="first_wiseman_reversal_cooldown must be >= 0"):
        WisemanStrategy(first_wiseman_reversal_cooldown=-1)

def test_wiseman_executes_on_signal_bar_for_stop_triggers() -> None:
    assert WisemanStrategy.execute_on_signal_bar is True


def test_wiseman_super_ao_adds_three_contracts_and_reverses_full_size(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: series + {13: 0.0, 8: 5.0, 5: 10.0}[period],
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    # Bearish 1W setup centered at bar 40.
    open_[40], close[40], high[40], low[40] = 138.0, 135.0, 140.0, 134.0
    high[39], high[41] = 137.0, 136.5

    # Trigger 1W short at bar 42.
    low[42] = 133.8

    # Force three AO red bars while 1W short is active (bars 42-44).
    high[42], low[42] = 120.0, 118.0
    high[43], low[43] = 119.0, 116.0
    high[44], low[44] = 118.0, 115.0

    # Trigger Super AO add-on at bar 45, then reverse all 4 at bar 46.
    low[45] = 114.0
    high[46] = 140.5

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001)
    signals = strategy.generate_signals(data)
    contracts = strategy.signal_contracts

    assert signals.iloc[42] == -1
    assert contracts is not None
    assert contracts.iloc[42] == -1
    assert contracts.iloc[45] == -4
    assert signals.iloc[46] == 1
    assert contracts.iloc[46] == 4


def test_wiseman_second_wiseman_can_trigger_after_third_wiseman(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: series + {13: 0.0, 8: 5.0, 5: 10.0}[period],
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    # Bearish 1W setup centered at bar 40 and triggered at 42.
    open_[40], close[40], high[40], low[40] = 138.0, 135.0, 140.0, 134.0
    high[39], high[41] = 137.0, 136.5
    high[42], low[42] = 120.0, 118.0

    # Keep AO red three bars in a row, but delay 2W fill.
    high[43], low[43] = 119.0, 116.0
    high[44], low[44] = 118.0, 115.0

    # Force a valid 3W setup/fill before 2W fill.
    low[45] = 116.5  # above 2W trigger (115.0), so 2W does not fill yet
    low[46] = 115.5  # breaks 3W trigger (116.0) but still above 2W trigger
    low[47] = 114.5  # finally breaks 2W trigger

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    fractals = pd.DataFrame(
        {
            "up_fractal": False,
            "down_fractal": False,
            "up_fractal_price": np.nan,
            "down_fractal_price": np.nan,
        },
        index=idx,
    )
    fractals.iloc[43, fractals.columns.get_loc("down_fractal")] = True
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001)
    strategy.generate_signals(data)

    contracts = strategy.signal_contracts
    third_fill_side = strategy.signal_third_wiseman_fill_side
    second_fill_side = strategy.signal_second_wiseman_fill_side

    assert contracts is not None
    assert third_fill_side is not None
    assert second_fill_side is not None

    # 3W fills first (size becomes 6x), 2W then still valid (size becomes 9x with additive defaults 1+5+3).
    assert third_fill_side.iloc[46] == -1
    assert contracts.iloc[46] == -6
    assert second_fill_side.iloc[47] == -1
    assert contracts.iloc[47] == -9


def test_wiseman_contracts_can_be_configured(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: series + {13: 0.0, 8: 5.0, 5: 10.0}[period],
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    open_[40], close[40], high[40], low[40] = 138.0, 135.0, 140.0, 134.0
    high[39], high[41] = 137.0, 136.5
    high[42], low[42] = 120.0, 118.0
    high[43], low[43] = 119.0, 116.0
    high[44], low[44] = 118.0, 115.0
    low[45] = 116.5
    low[46] = 115.5
    low[47] = 114.5

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    fractals = pd.DataFrame(
        {
            "up_fractal": False,
            "down_fractal": False,
            "up_fractal_price": np.nan,
            "down_fractal_price": np.nan,
        },
        index=idx,
    )
    fractals.iloc[43, fractals.columns.get_loc("down_fractal")] = True
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        first_wiseman_contracts=2,
        second_wiseman_contracts=7,
        third_wiseman_contracts=11,
    )
    strategy.generate_signals(data)

    contracts = strategy.signal_contracts
    assert contracts is not None
    assert contracts.iloc[42] == -2
    assert contracts.iloc[46] == -13
    assert contracts.iloc[47] == -20


def test_wiseman_zero_first_contracts_disables_first_signal(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: series + {13: 0.0, 8: 5.0, 5: 10.0}[period],
    )

    idx = pd.date_range("2024-01-01", periods=80, freq="h", tz="UTC")
    base = np.linspace(100, 140, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    open_[50], close[50], high[50], low[50] = 135.0, 132.0, 136.0, 131.0
    high[49], high[51] = 134.0, 133.0
    low[52] = 130.8

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001, first_wiseman_contracts=0)
    signals = strategy.generate_signals(data)

    contracts = strategy.signal_contracts
    assert contracts is not None
    assert (contracts == 0).all()
    assert (signals == 0).all()


def test_wiseman_reversal_multiplier_scales_reversal_size(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: series + {13: 0.0, 8: 5.0, 5: 10.0}[period],
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    open_[40], close[40], high[40], low[40] = 138.0, 135.0, 140.0, 134.0
    high[39], high[41] = 137.0, 136.5
    low[42] = 133.8
    high[46] = 140.5

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    half = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001, reversal_contracts_mult=0.5)
    half.generate_signals(data)
    half_contracts = half.signal_contracts

    double = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001, reversal_contracts_mult=2.0)
    double.generate_signals(data)
    double_contracts = double.signal_contracts

    quarter = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001, reversal_contracts_mult=0.25)
    quarter.generate_signals(data)
    quarter_contracts = quarter.signal_contracts

    off = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001, reversal_contracts_mult=0.0)
    off_signals = off.generate_signals(data)
    off_contracts = off.signal_contracts

    assert half_contracts is not None
    assert half_contracts.iloc[42] == -1
    assert half_contracts.iloc[46] == 0.5

    assert double_contracts is not None
    assert double_contracts.iloc[42] == -1
    assert double_contracts.iloc[46] == 2

    assert quarter_contracts is not None
    assert quarter_contracts.iloc[42] == -1
    assert quarter_contracts.iloc[46] == 0.25

    assert off_contracts is not None
    assert off_signals.iloc[46] == 0
    assert off_contracts.iloc[46] == 0

def test_wiseman_third_wiseman_reanchors_to_most_recent_pending_fractal(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: series + {13: 0.0, 8: 5.0, 5: 10.0}[period],
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    # Bearish 1W setup centered at bar 40 and triggered at 42.
    open_[40], close[40], high[40], low[40] = 138.0, 135.0, 140.0, 134.0
    high[39], high[41] = 137.0, 136.5
    high[42], low[42] = 120.0, 118.0

    # Make 1W short remain active while 3W candidates appear.
    low[43] = 116.0  # first candidate 3W trigger
    low[45] = 115.5  # newer candidate 3W trigger (should replace first)
    low[46] = 116.2  # does not break first trigger before replacement
    low[48] = 115.7  # breaks old trigger, but not the newer replacement
    low[49] = 115.4  # finally breaks the replacement trigger

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    fractals = pd.DataFrame(
        {
            "up_fractal": False,
            "down_fractal": False,
            "up_fractal_price": np.nan,
            "down_fractal_price": np.nan,
        },
        index=idx,
    )
    fractals.iloc[43, fractals.columns.get_loc("down_fractal")] = True
    fractals.iloc[45, fractals.columns.get_loc("down_fractal")] = True
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001)
    strategy.generate_signals(data)

    third_setup_side = strategy.signal_third_wiseman_setup_side
    third_fill_side = strategy.signal_third_wiseman_fill_side

    assert third_setup_side is not None
    assert third_fill_side is not None

    # First pending 3W is canceled and replaced by the newer fractal.
    assert third_setup_side.iloc[43] == 0
    assert third_setup_side.iloc[45] == -1

    # Replacement trigger governs execution timing.
    assert third_fill_side.iloc[48] == 0
    assert third_fill_side.iloc[49] == -1


def test_wiseman_third_wiseman_stops_new_markers_after_third_fill(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: series + {13: 0.0, 8: 5.0, 5: 10.0}[period],
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    # Bearish 1W setup centered at bar 40 and triggered at 42.
    open_[40], close[40], high[40], low[40] = 138.0, 135.0, 140.0, 134.0
    high[39], high[41] = 137.0, 136.5
    high[42], low[42] = 120.0, 118.0

    # Initial 3W candidate then replacement before fill.
    low[43] = 116.0
    low[45] = 115.5
    low[46] = 116.2
    low[49] = 115.4  # fills replacement 3W

    # New candidate appears after 3W fill; should be ignored until next 1W trade.
    low[51] = 114.8

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    fractals = pd.DataFrame(
        {
            "up_fractal": False,
            "down_fractal": False,
            "up_fractal_price": np.nan,
            "down_fractal_price": np.nan,
        },
        index=idx,
    )
    fractals.iloc[43, fractals.columns.get_loc("down_fractal")] = True
    fractals.iloc[45, fractals.columns.get_loc("down_fractal")] = True
    fractals.iloc[51, fractals.columns.get_loc("down_fractal")] = True
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001)
    strategy.generate_signals(data)

    third_setup_side = strategy.signal_third_wiseman_setup_side
    third_fill_side = strategy.signal_third_wiseman_fill_side

    assert third_setup_side is not None
    assert third_fill_side is not None

    assert third_setup_side.iloc[43] == 0
    assert third_setup_side.iloc[45] == -1
    assert third_fill_side.iloc[49] == -1

    # No new 3W markers are allowed after 3W fill in the same 1W cycle.
    assert third_setup_side.iloc[51] == 0

def test_wiseman_emits_independent_first_and_second_fill_streams(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: series + {13: 0.0, 8: 5.0, 5: 10.0}[period],
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    open_[40], close[40], high[40], low[40] = 138.0, 135.0, 140.0, 134.0
    high[39], high[41] = 137.0, 136.5

    high[42], low[42] = 120.0, 118.0
    high[43], low[43] = 119.0, 116.0
    high[44], low[44] = 118.0, 115.0
    low[45] = 114.0

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001)
    strategy.generate_signals(data)

    assert strategy.signal_fill_prices_first is not None
    assert strategy.signal_fill_prices_second is not None
    first_fills = strategy.signal_fill_prices_first.dropna()
    second_fills = strategy.signal_fill_prices_second.dropna()
    assert float(first_fills.iloc[0]) == 134.0
    assert float(second_fills.iloc[0]) == 115.0


def test_wiseman_second_entry_is_blocked_when_original_entry_was_reversal(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: series + {13: 0.0, 8: 5.0, 5: 10.0}[period],
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    # Bearish 1W setup centered at bar 40, then trigger short at 42.
    open_[40], close[40], high[40], low[40] = 138.0, 135.0, 140.0, 134.0
    high[39], high[41] = 137.0, 136.5
    low[42] = 133.8

    # Reverse to long off the original setup level at bar 45.
    high[45] = 140.5

    # Keep long active and force three AO-green bars, then a would-be second entry trigger.
    high[45], low[45] = 140.5, 139.0
    high[46], low[46] = 141.5, 140.0
    high[47], low[47] = 142.5, 141.0
    high[48], low[48] = 143.5, 142.0

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001)
    signals = strategy.generate_signals(data)

    assert signals.iloc[42] == -1
    assert signals.iloc[45] == 1
    assert strategy.signal_contracts is not None
    assert not (strategy.signal_contracts.abs() == 4).any()
    assert strategy.signal_fill_prices_second is not None
    assert strategy.signal_fill_prices_second.dropna().empty


def test_wiseman_third_entry_ignores_first_invalid_fractal(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: series + {13: 0.0, 8: 5.0, 5: 10.0}[period],
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    # Bearish 1W setup centered at bar 40, then trigger short at bar 42.
    open_[40], close[40], high[40], low[40] = 138.0, 135.0, 140.0, 134.0
    high[39], high[41] = 137.0, 139.0
    low[41] = 138.0
    low[42] = 133.8

    # Keep the short active and create a later valid fractal trigger level.
    low[43] = 110.0
    low[46] = 109.5

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    fractals = pd.DataFrame(False, index=data.index, columns=["up_fractal", "down_fractal"])
    fractals["up_fractal_price"] = np.nan
    fractals["down_fractal_price"] = np.nan
    fractals.loc[idx[41], "down_fractal"] = True  # First down fractal after 1W; not below teeth.
    fractals.loc[idx[43], "down_fractal"] = True  # Later down fractal below teeth; should be used.

    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001)
    strategy.generate_signals(data)

    assert strategy.signal_fill_prices_third is not None
    third_fills = strategy.signal_fill_prices_third.dropna()
    assert len(third_fills) == 1
    assert float(third_fills.iloc[0]) == 110.0
    assert strategy.signal_third_wiseman_fill_side is not None
    third_fill_side = strategy.signal_third_wiseman_fill_side[third_fills.index]
    assert set(third_fill_side.astype(int).tolist()) == {-1}


def test_chart_third_wiseman_markers_ignore_first_invalid_fractal(monkeypatch) -> None:
    import backtesting.local_chart as chart_module

    monkeypatch.setattr(
        chart_module,
        "_smma",
        lambda series, period: series + {13: 0.0, 8: 5.0, 5: 10.0}[period],
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    open_[40], close[40], high[40], low[40] = 138.0, 135.0, 140.0, 134.0
    high[39], high[41] = 137.0, 139.0
    low[41] = 138.0
    low[42] = 133.8
    low[43] = 110.0

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    fractals = pd.DataFrame(False, index=data.index, columns=["up_fractal", "down_fractal"])
    fractals["up_fractal_price"] = np.nan
    fractals["down_fractal_price"] = np.nan
    fractals.loc[idx[41], "down_fractal"] = True
    fractals.loc[idx[43], "down_fractal"] = True
    fractals.loc[idx[43], "down_fractal_price"] = 109.0

    monkeypatch.setattr(chart_module, "detect_williams_fractals", lambda _data, tick_size=1.0: fractals)

    markers = chart_module._valid_third_wiseman_fractal_markers(data)

    assert markers == []


def test_chart_third_wiseman_markers_bullish_style(monkeypatch) -> None:
    import backtesting.local_chart as chart_module

    monkeypatch.setattr(
        chart_module,
        "_smma",
        lambda series, period: series + {13: 0.0, 8: -5.0, 5: -10.0}[period],
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(150, 100, len(idx))

    open_ = base + 0.5
    close = base + 0.3
    high = base + 1.0
    low = base - 1.0

    # Bullish 1W setup centered at bar 40, then trigger long at bar 42.
    open_[40], close[40], high[40], low[40] = 112.0, 115.0, 116.0, 110.0
    low[39], low[41] = 113.0, 111.0
    high[41] = 112.0
    high[42] = 116.5

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    fractals = pd.DataFrame(False, index=data.index, columns=["up_fractal", "down_fractal"])
    fractals["up_fractal_price"] = np.nan
    fractals["down_fractal_price"] = np.nan
    fractals.loc[idx[43], "up_fractal"] = True

    monkeypatch.setattr(chart_module, "detect_williams_fractals", lambda _data, tick_size=1.0: fractals)

    markers = chart_module._valid_third_wiseman_fractal_markers(data)

    assert markers == []

def test_execution_trade_path_lines_split_scale_in_legs() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    execution_events = [
        ExecutionEvent("entry", idx[1], "sell", 100.0, 1.0),
        ExecutionEvent("add", idx[2], "sell", 95.0, 3.0),
        ExecutionEvent("exit", idx[4], "buy", 90.0, 4.0),
    ]

    lines = _execution_trade_path_lines(execution_events, idx)

    assert len(lines) == 2
    values = sorted((line["points"][0]["value"], line["points"][1]["value"]) for line in lines)
    assert values == [(95.0, 90.0), (100.0, 90.0)]

def test_wiseman_teeth_profit_protection_closes_profitable_short(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: pd.Series({13: 90.0, 8: 100.0, 5: 110.0}[period], index=series.index),
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    # Bearish 1W setup at bar 40; entry triggers on bar 42 at setup low.
    open_[40], close[40], high[40], low[40] = 138.0, 135.0, 140.0, 134.0
    high[39], high[41] = 137.0, 136.5
    low[42] = 133.8

    # Keep trade profitable and trigger teeth-based protection after min bars in position.
    close[46], high[46], low[46] = 101.0, 103.0, 100.0

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        teeth_profit_protection_enabled=True,
        teeth_profit_protection_min_bars=4,
        teeth_profit_protection_min_unrealized_return=0.01,
        profit_protection_annualized_volatility_scaler=10.0,
    )
    signals = strategy.generate_signals(data)

    assert signals.iloc[44] == -1
    assert signals.iloc[45] == 0
    assert strategy.signal_fill_prices_first is not None
    assert strategy.signal_stop_loss_prices is not None
    assert strategy.signal_stop_loss_prices.iloc[42] == pytest.approx(140.0)
    assert strategy.signal_stop_loss_prices.iloc[44] == pytest.approx(140.0)
    assert strategy.signal_fill_prices_first.iloc[45] == pytest.approx(float(data["close"].iloc[45]))
    assert strategy.signal_exit_reason is not None
    assert str(strategy.signal_exit_reason.iloc[45]) == "Red Gator PP"


def test_wiseman_profit_protection_keeps_reversal_level_armed_until_trigger(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: pd.Series({13: 130.0, 8: 120.0, 5: 90.0}[period], index=series.index),
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(140, 100, len(idx))

    open_ = base - 0.3
    close = base - 0.5
    high = base + 0.7
    low = base - 1.0

    # Bullish 1W setup at bar 40 with long entry on bar 42 around setup high.
    setup_i = 40
    open_[setup_i], close[setup_i], high[setup_i], low[setup_i] = 110.0, 113.0, 114.0, 108.0
    low[setup_i - 1], low[setup_i + 1] = 109.0, 109.5
    high[42] = 114.5

    # Keep trade profitable into bar 45, then close below teeth to trigger protection.
    close[45], high[45], low[45] = 115.0, 116.0, 114.5

    # After profit-protection close, later cross original setup low to trigger reversal short.
    low[48] = 107.8

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        teeth_profit_protection_enabled=True,
        teeth_profit_protection_min_bars=4,
        teeth_profit_protection_min_unrealized_return=0.01,
        profit_protection_annualized_volatility_scaler=10.0,
    )
    signals = strategy.generate_signals(data)
    fills = strategy.signal_fill_prices_first

    assert signals.iloc[44] == 1
    assert signals.iloc[45] == 0
    assert signals.iloc[46] == 0
    assert signals.iloc[48] == -1
    assert fills is not None
    assert fills.iloc[45] == pytest.approx(float(data["close"].iloc[45]))
    assert fills.iloc[48] == pytest.approx(108.0)
    assert int(strategy.signal_first_wiseman_reversal_side.iloc[48]) == -1


def test_wiseman_zone_profit_protection_trails_long_and_exits_on_gap(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: pd.Series({13: 130.0, 8: 120.0, 5: 90.0}[period], index=series.index),
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(140, 100, len(idx))

    open_ = base - 0.3
    close = base - 0.5
    high = base + 0.7
    low = base - 1.0

    setup_i = 40
    open_[setup_i], close[setup_i], high[setup_i], low[setup_i] = 110.0, 113.0, 114.0, 108.0
    low[setup_i - 1], low[setup_i + 1] = 109.0, 109.5
    high[42] = 114.5

    low[46], high[46], close[46] = 116.0, 118.0, 117.5
    low[47], high[47], close[47] = 117.0, 119.0, 118.0
    open_[48], low[48], high[48], close[48] = 116.5, 116.0, 118.0, 116.8

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    zone_green = pd.Series(False, index=data.index, dtype=bool)
    zone_red = pd.Series(False, index=data.index, dtype=bool)
    zone_green.iloc[42:47] = True
    monkeypatch.setattr(
        strategy_module.WisemanStrategy,
        "_williams_zone_bars",
        lambda self, _data, _ao: (zone_green, zone_red),
    )

    strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        zone_profit_protection_enabled=True,
        zone_profit_protection_min_unrealized_return=0.01,
    )
    signals = strategy.generate_signals(data)

    assert signals.iloc[47] == 1
    assert signals.iloc[48] == 0
    assert strategy.signal_fill_prices_first is not None
    assert strategy.signal_fill_prices_first.iloc[48] == pytest.approx(116.0)
    assert strategy.signal_exit_reason is not None
    assert str(strategy.signal_exit_reason.iloc[48]) == "Williams Zone PP"


def test_wiseman_profit_protection_can_cancel_resting_reversal_level(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: pd.Series({13: 130.0, 8: 120.0, 5: 90.0}[period], index=series.index),
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(140, 100, len(idx))

    open_ = base - 0.3
    close = base - 0.5
    high = base + 0.7
    low = base - 1.0

    setup_i = 40
    open_[setup_i], close[setup_i], high[setup_i], low[setup_i] = 110.0, 113.0, 114.0, 108.0
    low[setup_i - 1], low[setup_i + 1] = 109.0, 109.5
    high[42] = 114.5

    close[45], high[45], low[45] = 115.0, 116.0, 114.5
    low[48] = 107.8

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        teeth_profit_protection_enabled=True,
        teeth_profit_protection_min_bars=4,
        teeth_profit_protection_min_unrealized_return=0.01,
        profit_protection_annualized_volatility_scaler=10.0,
        cancel_reversal_on_first_wiseman_exit=True,
    )
    signals = strategy.generate_signals(data)

    assert signals.iloc[44] == 1
    assert signals.iloc[45] == 0
    assert int(strategy.signal_first_wiseman_reversal_side.iloc[48]) == 0



def test_wiseman_wait_close_same_bar_conflict_before_t_plus_3_records_1w_c_without_reversal(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: pd.Series({13: 90.0, 8: 100.0, 5: 110.0}[period], index=series.index),
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    setup_i = 40
    open_[setup_i], close[setup_i], high[setup_i], low[setup_i] = 138.0, 135.0, 140.0, 134.0
    high[setup_i - 1], high[setup_i + 1] = 137.0, 136.5

    # t+2 close activates short entry in wait-to-close mode, but same bar also breaks setup high.
    close[42] = 134.8
    low[42] = 133.7
    high[42] = 140.6

    # t+3+ remains above setup high; no reversal should ever appear because stop happened before t+3.
    high[43] = 141.0
    high[44] = 141.4

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        first_wiseman_wait_bars_to_close=2,
    )
    signals = strategy.generate_signals(data)
    reversals = strategy.signal_first_wiseman_reversal_side
    ignored = strategy.signal_first_wiseman_ignored_reason
    fills = strategy.signal_fill_prices_first

    assert int(signals.iloc[42]) == 0
    assert reversals is not None
    assert int(reversals.iloc[42]) == 0
    assert int(reversals.iloc[43]) == 0
    assert int(reversals.iloc[44]) == 0
    assert strategy.signal_exit_reason is not None
    assert str(strategy.signal_exit_reason.iloc[42]) == "1W Reversal Stop"
    assert ignored is not None
    assert str(ignored.iloc[setup_i]) == "same_bar_stop_before_reversal_window"
    assert fills is not None
    assert np.isclose(float(fills.iloc[42]), float(close[42]))


def test_wiseman_wait_close_same_bar_conflict_on_t_plus_3_reverses_same_candle(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: pd.Series({13: 90.0, 8: 100.0, 5: 110.0}[period], index=series.index),
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    setup_i = 40
    open_[setup_i], close[setup_i], high[setup_i], low[setup_i] = 138.0, 135.0, 140.0, 134.0
    high[setup_i - 1], high[setup_i + 1] = 137.0, 136.5

    # At t+3, wait-to-close entry activates and same bar breaks setup high.
    close[43] = 134.8
    low[43] = 133.8
    high[43] = 140.8

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        first_wiseman_wait_bars_to_close=3,
    )
    signals = strategy.generate_signals(data)
    reversals = strategy.signal_first_wiseman_reversal_side
    ignored = strategy.signal_first_wiseman_ignored_reason

    assert reversals is not None
    assert int(reversals.iloc[43]) == 1
    assert int(signals.iloc[43]) == 1
    assert ignored is not None
    assert str(ignored.iloc[setup_i]) == ""


def test_bullish_first_wiseman_breaking_setup_low_on_t_plus_2_stops_out_and_cancels_reversal(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: pd.Series({13: 130.0, 8: 120.0, 5: 90.0}[period], index=series.index),
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(140, 100, len(idx))

    open_ = base - 0.3
    close = base - 0.5
    high = base + 0.7
    low = base - 1.0

    setup_i = 40
    open_[setup_i], close[setup_i], high[setup_i], low[setup_i] = 110.0, 113.0, 114.0, 108.0
    low[setup_i - 1], low[setup_i + 1] = 109.0, 109.5

    # t+1: break setup high -> long entry.
    high[41] = 114.6
    # t+2: break setup low -> stop out and cancel future reversal.
    low[42] = 107.8
    # t+3+: continue below setup low; must NOT reverse short.
    low[43] = 107.0
    low[44] = 106.8

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001, gator_width_valid_factor=2.0)
    signals = strategy.generate_signals(data)
    reversals = strategy.signal_first_wiseman_reversal_side

    assert signals.iloc[41] == 1
    assert signals.iloc[42] == 0
    assert signals.iloc[43] == 0
    assert signals.iloc[44] == 0
    assert reversals is not None
    assert int(reversals.iloc[42]) == 0
    assert int(reversals.iloc[43]) == 0
    assert int(reversals.iloc[44]) == 0
    assert strategy.signal_exit_reason is not None
    assert str(strategy.signal_exit_reason.iloc[42]) == "1W Reversal Stop"


def test_bearish_first_wiseman_breaking_setup_high_on_t_plus_2_stops_out_and_cancels_reversal(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: pd.Series({13: 90.0, 8: 100.0, 5: 110.0}[period], index=series.index),
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    setup_i = 40
    open_[setup_i], close[setup_i], high[setup_i], low[setup_i] = 138.0, 135.0, 140.0, 134.0
    high[setup_i - 1], high[setup_i + 1] = 137.0, 136.5

    # t+1: break setup low -> short entry.
    low[41] = 133.7
    # t+2: break setup high -> stop out and cancel future reversal.
    high[42] = 140.6
    # t+3+: continue above setup high; must NOT reverse long.
    high[43] = 141.0
    high[44] = 141.4

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(gator_width_lookback=1, gator_width_mult=0.0001, gator_width_valid_factor=2.0)
    signals = strategy.generate_signals(data)
    reversals = strategy.signal_first_wiseman_reversal_side

    assert signals.iloc[41] == -1
    assert signals.iloc[42] == 0
    assert signals.iloc[43] == 0
    assert signals.iloc[44] == 0
    assert reversals is not None
    assert int(reversals.iloc[42]) == 0
    assert int(reversals.iloc[43]) == 0
    assert int(reversals.iloc[44]) == 0
    assert strategy.signal_exit_reason is not None
    assert str(strategy.signal_exit_reason.iloc[42]) == "1W Reversal Stop"


def test_bullish_t_plus_2_stop_ignores_opposite_close_min_unrealized_return_gate(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: pd.Series({13: 130.0, 8: 120.0, 5: 90.0}[period], index=series.index),
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(140, 100, len(idx))

    open_ = base - 0.3
    close = base - 0.5
    high = base + 0.7
    low = base - 1.0

    setup_i = 40
    open_[setup_i], close[setup_i], high[setup_i], low[setup_i] = 110.0, 113.0, 114.0, 108.0
    low[setup_i - 1], low[setup_i + 1] = 109.0, 109.5

    high[41] = 114.6
    low[42] = 107.8

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        first_wiseman_opposite_close_min_unrealized_return=0.50,
    )
    signals = strategy.generate_signals(data)

    assert signals.iloc[41] == 1
    assert signals.iloc[42] == 0
    assert strategy.signal_exit_reason is not None
    assert str(strategy.signal_exit_reason.iloc[42]) == "1W Reversal Stop"



def test_bearish_t_plus_2_stop_ignores_opposite_close_min_unrealized_return_gate(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: pd.Series({13: 90.0, 8: 100.0, 5: 110.0}[period], index=series.index),
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    setup_i = 40
    open_[setup_i], close[setup_i], high[setup_i], low[setup_i] = 138.0, 135.0, 140.0, 134.0
    high[setup_i - 1], high[setup_i + 1] = 137.0, 136.5

    low[41] = 133.7
    high[42] = 140.6

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        first_wiseman_opposite_close_min_unrealized_return=0.50,
    )
    signals = strategy.generate_signals(data)

    assert signals.iloc[41] == -1
    assert signals.iloc[42] == 0
    assert strategy.signal_exit_reason is not None
    assert str(strategy.signal_exit_reason.iloc[42]) == "1W Reversal Stop"


def test_bullish_active_level_touch_after_t_plus_3_reverses_even_when_opposite_close_gate_is_set(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: pd.Series({13: 130.0, 8: 120.0, 5: 90.0}[period], index=series.index),
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(140, 100, len(idx))

    open_ = base - 0.3
    close = base - 0.5
    high = base + 0.7
    low = base - 1.0

    setup_i = 40
    open_[setup_i], close[setup_i], high[setup_i], low[setup_i] = 110.0, 113.0, 114.0, 108.0
    low[setup_i - 1], low[setup_i + 1] = 109.0, 109.5

    # Enter long on t+1.
    high[41] = 114.6
    # Keep lows above stop into t+3 so position remains open.
    low[42] = 108.5
    low[43] = 108.4
    # At t+4, touch opposite active level after reversal eligibility window opens.
    low[44] = 107.8

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        first_wiseman_opposite_close_min_unrealized_return=0.50,
    )
    signals = strategy.generate_signals(data)
    reversals = strategy.signal_first_wiseman_reversal_side

    assert signals.iloc[41] == 1
    assert signals.iloc[44] == -1
    assert reversals is not None
    assert int(reversals.iloc[44]) == -1


def test_bearish_active_level_touch_after_t_plus_3_reverses_even_when_opposite_close_gate_is_set(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: pd.Series({13: 90.0, 8: 100.0, 5: 110.0}[period], index=series.index),
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    setup_i = 40
    open_[setup_i], close[setup_i], high[setup_i], low[setup_i] = 138.0, 135.0, 140.0, 134.0
    high[setup_i - 1], high[setup_i + 1] = 137.0, 136.5

    # Enter short on t+1.
    low[41] = 133.7
    # Keep highs below stop into t+3 so position remains open.
    high[42] = 139.5
    high[43] = 139.8
    # At t+4, touch opposite active level after reversal eligibility window opens.
    high[44] = 140.6

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        first_wiseman_opposite_close_min_unrealized_return=0.50,
    )
    signals = strategy.generate_signals(data)
    reversals = strategy.signal_first_wiseman_reversal_side

    assert signals.iloc[41] == -1
    assert signals.iloc[44] == 1
    assert reversals is not None
    assert int(reversals.iloc[44]) == 1


def test_wiseman_profit_protection_does_not_carry_armed_state_across_reversal(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: pd.Series({13: 90.0, 8: 100.0, 5: 110.0}[period], index=series.index),
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    # Bearish 1W setup centered at bar 40, trigger short at bar 42.
    open_[40], close[40], high[40], low[40] = 138.0, 135.0, 140.0, 134.0
    high[39], high[41] = 137.0, 136.5
    low[42] = 133.8

    # Make the short profitable enough to arm profit-protection.
    close[45], high[45], low[45] = 95.0, 97.0, 94.0

    # Reverse to long at bar 46 by crossing the setup high.
    high[46], low[46], close[46] = 140.5, 136.0, 130.0

    # On the next bar, close below teeth (100). This must NOT close the fresh long
    # unless protection has been re-armed for the new position.
    high[47], low[47], close[47] = 101.0, 98.0, 99.0

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        teeth_profit_protection_enabled=True,
        teeth_profit_protection_min_bars=4,
    )
    signals = strategy.generate_signals(data)
    fills = strategy.signal_fill_prices_first

    assert signals.iloc[43] == -1
    assert signals.iloc[46] == 1
    assert signals.iloc[47] == 1
    assert fills is not None
    assert np.isnan(fills.iloc[47])


def test_wiseman_reversal_after_profit_protection_uses_original_entry_contract_size(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: pd.Series({13: 130.0, 8: 120.0, 5: 90.0}[period], index=series.index),
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(140, 100, len(idx))

    open_ = base - 0.3
    close = base - 0.5
    high = base + 0.7
    low = base - 1.0

    # Bullish 1W setup at bar 40 with long entry on bar 42 around setup high.
    setup_i = 40
    open_[setup_i], close[setup_i], high[setup_i], low[setup_i] = 110.0, 113.0, 114.0, 108.0
    low[setup_i - 1], low[setup_i + 1] = 109.0, 109.5
    high[42] = 114.5

    # Keep trade profitable into bar 45, then close below teeth to trigger protection.
    close[45], high[45], low[45] = 115.0, 116.0, 114.5

    # After profit-protection close, later cross original setup low to trigger reversal short.
    low[48] = 107.8

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        first_wiseman_contracts=5,
        second_wiseman_contracts=0,
        third_wiseman_contracts=0,
        reversal_contracts_mult=1.0,
        teeth_profit_protection_enabled=True,
        teeth_profit_protection_min_bars=4,
        teeth_profit_protection_min_unrealized_return=0.01,
        profit_protection_annualized_volatility_scaler=10.0,
    )
    signals = strategy.generate_signals(data)
    contracts = strategy.signal_contracts

    assert signals.iloc[44] == 1
    assert contracts is not None
    assert contracts.iloc[44] == 5
    assert signals.iloc[45] == 0
    assert contracts.iloc[45] == 0
    assert signals.iloc[48] == -1
    assert contracts.iloc[48] == -5


def test_engine_records_profit_protection_exit_for_1w_reversal_trade(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: pd.Series({13: 130.0, 8: 120.0, 5: 90.0}[period], index=series.index),
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(140, 100, len(idx))

    open_ = base - 0.3
    close = base - 0.5
    high = base + 0.7
    low = base - 1.0

    # Bullish 1W setup at bar 40, long entry at bar 42, PP exit at bar 45.
    setup_i = 40
    open_[setup_i], close[setup_i], high[setup_i], low[setup_i] = 110.0, 113.0, 114.0, 108.0
    low[setup_i - 1], low[setup_i + 1] = 109.0, 109.5
    high[42] = 114.5
    close[45], high[45], low[45] = 115.0, 116.0, 114.5

    # After the flat PP exit, the original setup low triggers a bearish 1W-R at bar 48.
    low[48] = 107.8

    # Make the reversal short profitable enough to arm PP, then close back above teeth
    # so the reversal trade itself exits flat with an explicit strategy exit reason.
    close[49], high[49], low[49] = 103.0, 104.0, 102.0
    close[50], high[50], low[50] = 101.0, 102.0, 100.0
    close[51], high[51], low[51] = 121.0, 122.0, 120.5

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        first_wiseman_contracts=5,
        second_wiseman_contracts=0,
        third_wiseman_contracts=0,
        reversal_contracts_mult=1.0,
        teeth_profit_protection_enabled=True,
        teeth_profit_protection_min_bars=2,
    )

    result = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="units",
            trade_size_value=1.0,
        )
    ).run(data, strategy)

    exit_reason = strategy.signal_exit_reason
    reversals = strategy.signal_first_wiseman_reversal_side
    assert exit_reason is not None
    assert reversals is not None
    assert int(reversals.iloc[48]) == -1
    assert str(exit_reason.iloc[51]) == "Red Gator PP"

    trades_df = result.trades_dataframe()
    reversal_trade = trades_df.iloc[1]

    assert reversal_trade["entry_signal"] == "Bearish 1W-R"
    assert reversal_trade["exit_signal"] == "Strategy Profit Protection Red Gator"
    assert pd.isna(reversal_trade["signal_intent_flat_time"])


def test_wiseman_opposite_1w_close_requires_min_unrealized_return(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: pd.Series({13: 90.0, 8: 100.0, 5: 110.0}[period], index=series.index),
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    setup_i = 40
    open_[setup_i], close[setup_i], high[setup_i], low[setup_i] = 138.0, 135.0, 140.0, 134.0
    high[setup_i - 1], high[setup_i + 1] = 137.0, 136.5

    low[42] = 133.8

    # Opposite 1W close level gets crossed, but only ~2% favorable excursion from entry.
    high[45] = 140.2
    low[45] = 132.2
    close[45] = 139.8

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    baseline_strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
    )
    baseline_signals = baseline_strategy.generate_signals(data)

    gated_strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        first_wiseman_opposite_close_min_unrealized_return=0.05,
    )
    gated_signals = gated_strategy.generate_signals(data)

    assert baseline_signals.iloc[44] == -1
    assert baseline_signals.iloc[45] == 1
    assert gated_signals.iloc[44] == -1
    assert gated_signals.iloc[45] == 1

def test_wiseman_profit_protection_uses_favorable_intrabar_excursion(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: pd.Series({13: 90.0, 8: 100.0, 5: 110.0}[period], index=series.index),
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    # Bearish setup and short entry.
    open_[40], close[40], high[40], low[40] = 138.0, 135.0, 140.0, 134.0
    high[39], high[41] = 137.0, 136.5
    low[42] = 133.8

    # On bar 45, low breaches min-unrealized threshold but close does not.
    close[45], high[45], low[45] = 133.2, 134.0, 132.0

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        teeth_profit_protection_enabled=True,
        teeth_profit_protection_min_bars=4,
        teeth_profit_protection_min_unrealized_return=0.01,
        profit_protection_annualized_volatility_scaler=10.0,
    )
    signals = strategy.generate_signals(data)

    assert signals.iloc[44] == -1
    assert signals.iloc[45] == 0


def test_wiseman_profit_protection_entry_bar_credit_optional(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: pd.Series({13: 90.0, 8: 100.0, 5: 110.0}[period], index=series.index),
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    open_[40], close[40], high[40], low[40] = 138.0, 135.0, 140.0, 134.0
    high[39], high[41] = 137.0, 136.5

    # Short entry on bar 42. Only the entry bar reaches the min-unrealized threshold.
    low[42], close[42], high[42] = 120.0, 133.7, 136.0
    low[43], close[43], high[43] = 132.8, 133.6, 134.2
    low[44], close[44], high[44] = 132.8, 133.6, 134.2
    low[45], close[45], high[45] = 132.8, 133.6, 134.2

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy_requires_after_min_bars = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        teeth_profit_protection_enabled=True,
        teeth_profit_protection_min_bars=4,
        teeth_profit_protection_min_unrealized_return=0.04,
        profit_protection_annualized_volatility_scaler=4.0,
        teeth_profit_protection_credit_unrealized_before_min_bars=False,
    )
    signals_requires_after_min_bars = strategy_requires_after_min_bars.generate_signals(data)

    strategy_credit_anytime = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        teeth_profit_protection_enabled=True,
        teeth_profit_protection_min_bars=4,
        teeth_profit_protection_min_unrealized_return=0.04,
        profit_protection_annualized_volatility_scaler=4.0,
        teeth_profit_protection_credit_unrealized_before_min_bars=True,
    )
    signals_credit_anytime = strategy_credit_anytime.generate_signals(data)

    assert signals_requires_after_min_bars.iloc[45] == -1
    assert signals_credit_anytime.iloc[45] == 0

def test_wiseman_teeth_profit_protection_can_be_disabled(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: pd.Series({13: 90.0, 8: 100.0, 5: 110.0}[period], index=series.index),
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    open_[40], close[40], high[40], low[40] = 138.0, 135.0, 140.0, 134.0
    high[39], high[41] = 137.0, 136.5
    low[42] = 133.8
    close[46], high[46], low[46] = 101.0, 103.0, 100.0

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        teeth_profit_protection_enabled=False,
    )
    signals = strategy.generate_signals(data)

    assert signals.iloc[45] == -1


def test_wiseman_profit_protection_gator_requirement_flag_defaults_to_enabled() -> None:
    assert WisemanStrategy().teeth_profit_protection_require_gator_open is True
    assert WisemanStrategy(teeth_profit_protection_require_gator_open=False).teeth_profit_protection_require_gator_open is False
    assert WisemanStrategy().teeth_profit_protection_credit_unrealized_before_min_bars is False
    assert WisemanStrategy(teeth_profit_protection_credit_unrealized_before_min_bars=True).teeth_profit_protection_credit_unrealized_before_min_bars is True
    assert WisemanStrategy().profit_protection_volatility_lookback == 20
    assert WisemanStrategy().profit_protection_annualized_volatility_scaler == 1.0
    assert WisemanStrategy(lips_profit_protection_volatility_lookback=30).profit_protection_volatility_lookback == 30
    assert WisemanStrategy(
        profit_protection_volatility_lookback=40,
        lips_profit_protection_volatility_lookback=30,
    ).profit_protection_volatility_lookback == 40
    assert WisemanStrategy(profit_protection_annualized_volatility_scaler=0.85).profit_protection_annualized_volatility_scaler == 0.85
    assert WisemanStrategy().lips_profit_protection_enabled is False
    assert WisemanStrategy().lips_profit_protection_min_unrealized_return == 1.0
    assert WisemanStrategy().lips_profit_protection_arm_on_min_unrealized_return is False


def test_profit_protection_min_unrealized_return_uses_annualized_volatility_scale() -> None:
    assert _annualized_volatility_scaled_return_threshold(0.01, 0.85, 0.85) == pytest.approx(0.01)
    assert _annualized_volatility_scaled_return_threshold(0.01, 0.425, 0.85) == pytest.approx(0.005)
    assert _annualized_volatility_scaled_return_threshold(0.01, None, 0.85) == np.inf
    assert _annualized_volatility_scaled_return_threshold(0.0, 0.85, 0.85) == 0.0


def test_lips_volatility_trigger_uses_annualized_volatility_scaler() -> None:
    assert _scaled_annualized_volatility_trigger(0.1, 0.25) == pytest.approx(0.025)
    assert _scaled_annualized_volatility_trigger(0.1, 1.0) == pytest.approx(0.1)
    assert _scaled_annualized_volatility_trigger(0.0, 0.25) == 0.0


def test_wiseman_lips_profit_protection_closes_short_before_teeth(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    def _custom_smma(series: pd.Series, period: int) -> pd.Series:
        idx = series.index
        if period == 13:
            return pd.Series(100.0, index=idx)
        if period == 8:
            values = np.full(len(idx), 105.0)
            values[39:] = 130.0
            return pd.Series(values, index=idx)
        values = np.full(len(idx), 110.0)
        values[39:] = 120.0
        return pd.Series(values, index=idx)

    monkeypatch.setattr(strategy_module, "_smma", _custom_smma)

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    open_[40], close[40], high[40], low[40] = 138.0, 135.0, 140.0, 134.0
    high[39], high[41] = 137.0, 136.5
    low[42] = 133.8

    # Keep short open and profitable; bar 45 crosses lips (120) but not teeth (130).
    close[45], high[45], low[45] = 124.0, 125.0, 110.0

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    baseline_strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        teeth_profit_protection_enabled=True,
        teeth_profit_protection_min_bars=3,
        teeth_profit_protection_min_unrealized_return=0.0,
    )
    baseline_signals = baseline_strategy.generate_signals(data)

    lips_strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        teeth_profit_protection_enabled=True,
        teeth_profit_protection_min_bars=3,
        teeth_profit_protection_min_unrealized_return=0.0,
        lips_profit_protection_enabled=True,
        lips_profit_protection_volatility_trigger=0.0,
    )
    lips_signals = lips_strategy.generate_signals(data)

    assert baseline_signals.iloc[45] == -1
    assert lips_signals.iloc[45] == 0
    assert baseline_strategy.signal_exit_reason is not None
    assert lips_strategy.signal_exit_reason is not None
    assert str(lips_strategy.signal_exit_reason.iloc[44]) == "Green Gator PP"


def test_wiseman_lips_profit_protection_scales_volatility_trigger(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    def _custom_smma(series: pd.Series, period: int) -> pd.Series:
        idx = series.index
        if period == 13:
            return pd.Series(100.0, index=idx)
        if period == 8:
            values = np.full(len(idx), 105.0)
            values[39:] = 130.0
            return pd.Series(values, index=idx)
        values = np.full(len(idx), 110.0)
        values[39:] = 120.0
        return pd.Series(values, index=idx)

    monkeypatch.setattr(strategy_module, "_smma", _custom_smma)

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    open_[40], close[40], high[40], low[40] = 138.0, 135.0, 140.0, 134.0
    high[39], high[41] = 137.0, 136.5
    low[42] = 133.8
    close[45], high[45], low[45] = 124.0, 125.0, 110.0

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    absolute_trigger_strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        teeth_profit_protection_enabled=True,
        teeth_profit_protection_min_bars=3,
        teeth_profit_protection_min_unrealized_return=0.0,
        lips_profit_protection_enabled=True,
        lips_profit_protection_volatility_trigger=3.0,
        profit_protection_annualized_volatility_scaler=1.0,
    )
    absolute_trigger_signals = absolute_trigger_strategy.generate_signals(data)

    scaled_trigger_strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        teeth_profit_protection_enabled=True,
        teeth_profit_protection_min_bars=3,
        teeth_profit_protection_min_unrealized_return=0.0,
        lips_profit_protection_enabled=True,
        lips_profit_protection_volatility_trigger=3.0,
        profit_protection_annualized_volatility_scaler=0.25,
    )
    scaled_trigger_signals = scaled_trigger_strategy.generate_signals(data)

    assert absolute_trigger_signals.iloc[45] == -1
    assert scaled_trigger_signals.iloc[45] == 0
    assert scaled_trigger_strategy.signal_exit_reason is not None
    assert str(scaled_trigger_strategy.signal_exit_reason.iloc[44]) == "Green Gator PP"


def test_wiseman_lips_profit_protection_can_use_min_unrealized_gate(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    def _custom_smma(series: pd.Series, period: int) -> pd.Series:
        idx = series.index
        if period == 13:
            return pd.Series(100.0, index=idx)
        if period == 8:
            values = np.full(len(idx), 105.0)
            values[39:] = 130.0
            return pd.Series(values, index=idx)
        values = np.full(len(idx), 110.0)
        values[39:] = 120.0
        return pd.Series(values, index=idx)

    monkeypatch.setattr(strategy_module, "_smma", _custom_smma)

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    open_[40], close[40], high[40], low[40] = 138.0, 135.0, 140.0, 134.0
    high[39], high[41] = 137.0, 136.5
    low[42] = 133.8

    # Crosses lips (120) but not teeth (130), with mild profit just above lips min-unrealized gate.
    close[45], high[45], low[45] = 124.0, 125.0, 132.0

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    mult_only_strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        teeth_profit_protection_enabled=True,
        teeth_profit_protection_min_bars=3,
        teeth_profit_protection_min_unrealized_return=20.0,
        lips_profit_protection_enabled=True,
        lips_profit_protection_volatility_trigger=1.0,
        lips_profit_protection_profit_trigger_mult=5.0,
    )
    mult_only_signals = mult_only_strategy.generate_signals(data)

    min_unrealized_strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        teeth_profit_protection_enabled=True,
        teeth_profit_protection_min_bars=3,
        teeth_profit_protection_min_unrealized_return=20.0,
        lips_profit_protection_enabled=True,
        lips_profit_protection_volatility_trigger=1.0,
        lips_profit_protection_profit_trigger_mult=5.0,
        lips_profit_protection_min_unrealized_return=0.01,
        lips_profit_protection_arm_on_min_unrealized_return=True,
    )
    min_unrealized_signals = min_unrealized_strategy.generate_signals(data)

    assert mult_only_signals.iloc[45] == -1
    assert min_unrealized_signals.iloc[45] == -1


def test_wiseman_lips_profit_protection_can_trigger_without_teeth_mode(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    def _custom_smma(series: pd.Series, period: int) -> pd.Series:
        idx = series.index
        if period == 13:
            return pd.Series(100.0, index=idx)
        if period == 8:
            values = np.full(len(idx), 105.0)
            values[39:] = 130.0
            return pd.Series(values, index=idx)
        values = np.full(len(idx), 110.0)
        values[39:] = 120.0
        return pd.Series(values, index=idx)

    monkeypatch.setattr(strategy_module, "_smma", _custom_smma)

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    open_[40], close[40], high[40], low[40] = 138.0, 135.0, 140.0, 134.0
    high[39], high[41] = 137.0, 136.5
    low[42] = 133.8
    close[45], high[45], low[45] = 124.0, 125.0, 110.0

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        teeth_profit_protection_enabled=False,
        teeth_profit_protection_min_unrealized_return=0.0,
        lips_profit_protection_enabled=True,
        lips_profit_protection_volatility_trigger=0.0,
    )
    signals = strategy.generate_signals(data)

    assert signals.iloc[45] == 0
    assert strategy.signal_exit_reason is not None
    assert str(strategy.signal_exit_reason.iloc[44]) == "Green Gator PP"


def test_wiseman_lips_profit_protection_still_waits_for_base_arming(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    def _custom_smma(series: pd.Series, period: int) -> pd.Series:
        idx = series.index
        if period == 13:
            return pd.Series(100.0, index=idx)
        if period == 8:
            values = np.full(len(idx), 105.0)
            values[39:] = 130.0
            return pd.Series(values, index=idx)
        values = np.full(len(idx), 110.0)
        values[39:] = 120.0
        return pd.Series(values, index=idx)

    monkeypatch.setattr(strategy_module, "_smma", _custom_smma)

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    open_[40], close[40], high[40], low[40] = 138.0, 135.0, 140.0, 134.0
    high[39], high[41] = 137.0, 136.5
    low[42] = 133.8
    close[45], high[45], low[45] = 124.0, 125.0, 110.0

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        teeth_profit_protection_enabled=False,
        teeth_profit_protection_min_bars=100,
        teeth_profit_protection_min_unrealized_return=0.0,
        lips_profit_protection_enabled=True,
        lips_profit_protection_volatility_trigger=0.0,
    )
    signals = strategy.generate_signals(data)

    assert signals.iloc[45] == -1
    assert strategy.signal_exit_reason is not None
    assert str(strategy.signal_exit_reason.iloc[45]) == ""

def test_wiseman_rejects_invalid_gator_width_params() -> None:
    for kwargs in [
        {"gator_width_lookback": 0},
        {"gator_width_mult": 0.0},
        {"gator_width_valid_factor": 0.0},
        {"gator_direction_mode": 0},
        {"gator_direction_mode": 4},
    ]:
        try:
            WisemanStrategy(**kwargs)
            raise AssertionError("Expected ValueError for invalid gator width config")
        except ValueError:
            pass


def test_wiseman_rejects_invalid_contract_params() -> None:
    for kwargs in [
        {"first_wiseman_contracts": -1},
        {"second_wiseman_contracts": -1},
        {"third_wiseman_contracts": -1},
        {"reversal_contracts_mult": -0.1},
        {"first_wiseman_wait_bars_to_close": -1},
        {"first_wiseman_divergence_filter_bars": -1},
        {"first_wiseman_opposite_close_min_unrealized_return": -0.01},
        {"teeth_profit_protection_min_bars": 0},
        {"teeth_profit_protection_min_unrealized_return": -0.01},
        {"profit_protection_volatility_lookback": 1},
        {"profit_protection_annualized_volatility_scaler": 0.0},
        {"lips_profit_protection_volatility_trigger": -0.01},
        {"lips_profit_protection_profit_trigger_mult": -0.1},
        {"lips_profit_protection_volatility_lookback": 1},
        {"lips_profit_protection_recent_trade_lookback": 0},
        {"lips_profit_protection_min_unrealized_return": -0.01},
    ]:
        try:
            WisemanStrategy(**kwargs)
            raise AssertionError("Expected ValueError for invalid contract config")
        except ValueError:
            pass


def test_wiseman_strategy_emits_valid_signals() -> None:
    data = pd.read_csv("examples/sample_ohlcv.csv", parse_dates=["timestamp"]).set_index("timestamp")
    signals = WisemanStrategy().generate_signals(data)
    assert len(signals) == len(data)
    assert set(signals.unique()).issubset({-1, 0, 1})


def test_backtest_runs_with_wiseman_strategy() -> None:
    data = pd.read_csv("examples/sample_ohlcv.csv", parse_dates=["timestamp"]).set_index("timestamp")
    engine = BacktestEngine(BacktestConfig(initial_capital=10_000, fee_rate=0.0, slippage_rate=0.0))
    result = engine.run(data, WisemanStrategy())
    assert len(result.equity_curve) == len(data)
    assert "total_trades" in result.stats


def test_wiseman_null_market_produces_no_signals_or_trades() -> None:
    idx = pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC")
    flat_price = np.full(len(idx), 100.0)
    data = pd.DataFrame(
        {
            "open": flat_price,
            "high": flat_price,
            "low": flat_price,
            "close": flat_price,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy()
    signals = strategy.generate_signals(data)

    assert (signals == 0).all()

    engine = BacktestEngine(BacktestConfig(initial_capital=10_000, fee_rate=0.0, slippage_rate=0.0))
    result = engine.run(data, strategy)

    assert result.stats["total_trades"] == 0
    assert len(result.trades) == 0


def test_wiseman_random_walk_market_stays_signal_balanced() -> None:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=240, freq="h", tz="UTC")
    log_steps = rng.normal(0.0, 0.002, size=len(idx))
    close = 100.0 * np.exp(np.cumsum(log_steps))
    open_ = np.concatenate(([close[0]], close[:-1]))
    spread = np.maximum(close * 0.001, 0.01)
    data = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": rng.integers(800, 1_200, size=len(idx)),
        },
        index=idx,
    )

    strategy = WisemanStrategy()
    signals = strategy.generate_signals(data)

    assert set(signals.unique()).issubset({-1, 0, 1})
    assert (signals != 0).any()
    assert (signals == 1).any()
    assert (signals == -1).any()

    engine = BacktestEngine(BacktestConfig(initial_capital=10_000, fee_rate=0.0, slippage_rate=0.0))
    result = engine.run(data, strategy)

    assert result.stats["total_trades"] > 0


def test_batch_run_and_switchable_chart(tmp_path) -> None:
    engine = BacktestEngine(BacktestConfig(initial_capital=10_000, fee_rate=0.0, slippage_rate=0.0))
    data_by_asset = {
        "BTC": _sample_df(periods=80, freq="h"),
        "ETH": _sample_df(periods=80, freq="h"),
    }

    batch_result = run_batch_backtest(
        data_by_asset=data_by_asset,
        timeframes=["1h", "4h"],
        engine=engine,
        strategy_factory=lambda _asset, _tf: SmaCrossoverStrategy(fast=5, slow=20),
    )

    assert len(batch_result.run_results) == 4
    assert "asset" in batch_result.summary.columns
    assert "timeframe" in batch_result.summary.columns
    assert batch_result.aggregate_stats["total_runs"] == 4.0

    chart_path = tmp_path / "batch_chart.html"
    generate_batch_local_tradingview_chart(batch_result, str(chart_path))
    html = chart_path.read_text(encoding="utf-8")

    assert "runSelect" in html
    assert "chart-stack" in html
    assert "BTC|1h" in html
    assert "ETH|4h" in html


def test_batch_run_auto_timeframe_uses_source_cadence() -> None:
    engine = BacktestEngine(BacktestConfig(initial_capital=10_000, fee_rate=0.0, slippage_rate=0.0))
    data_by_asset = {
        "BTC": _sample_df(periods=120, freq="5min"),
    }

    batch_result = run_batch_backtest(
        data_by_asset=data_by_asset,
        timeframes=["auto"],
        engine=engine,
        strategy_factory=lambda _asset, _tf: SmaCrossoverStrategy(fast=5, slow=20),
    )

    assert "BTC|5m" in batch_result.run_results
    assert "BTC|5m" in batch_result.run_data
    assert len(batch_result.run_data["BTC|5m"]) == len(data_by_asset["BTC"])


def test_batch_run_auto_timeframe_requires_detectable_source_frequency() -> None:
    engine = BacktestEngine(BacktestConfig(initial_capital=10_000, fee_rate=0.0, slippage_rate=0.0))
    idx = pd.Index([0, 1, 2, 3, 4])
    df = pd.DataFrame(
        {
            "open": [1.0, 1.1, 1.2, 1.3, 1.4],
            "high": [1.1, 1.2, 1.3, 1.4, 1.5],
            "low": [0.9, 1.0, 1.1, 1.2, 1.3],
            "close": [1.0, 1.1, 1.2, 1.3, 1.4],
            "volume": [100, 100, 100, 100, 100],
        },
        index=idx,
    )

    with pytest.raises(ValueError, match="Unable to infer source timeframe"):
        run_batch_backtest(
            data_by_asset={"BTC": df},
            timeframes=["auto"],
            engine=engine,
            strategy_factory=lambda _asset, _tf: SmaCrossoverStrategy(fast=2, slow=3),
        )

def test_bearish_first_wiseman_pinescript_generation(tmp_path) -> None:
    out = tmp_path / "first_wiseman_bearish.pine"
    path = generate_first_wiseman_bearish_pinescript(str(out))

    pine = out.read_text(encoding="utf-8")
    assert path.endswith("first_wiseman_bearish.pine")
    assert "indicator(\"1st Wiseman Detector (Bearish)\"" in pine
    assert "gatorOpenUp = lips > teeth and teeth > jaw" in pine
    assert "aoGreen = ao > ao[1]" in pine
    assert "label.new(candidateIndex, candidateHigh" in pine


def test_bullish_first_wiseman_pinescript_generation(tmp_path) -> None:
    out = tmp_path / "first_wiseman_bullish.pine"
    path = generate_first_wiseman_bullish_pinescript(str(out))

    pine = out.read_text(encoding="utf-8")
    assert path.endswith("first_wiseman_bullish.pine")
    assert "indicator(\"1st Wiseman Detector (Bullish)\"" in pine
    assert "gatorOpenDown = lips < teeth and teeth < jaw" in pine
    assert "aoRed = ao < ao[1]" in pine
    assert "alertcondition(bearishReverseSignal" in pine


def test_wiseman_summary_counts_match_chart_markers() -> None:
    data = pd.read_csv("examples/sample_ohlcv.csv", parse_dates=["timestamp"]).set_index("timestamp")
    summary = summarize_wiseman_markers(data)

    assert summary["bearish_first_wiseman"] >= 0
    assert summary["bearish_reverse"] >= 0
    assert summary["bullish_first_wiseman"] >= 0
    assert summary["bullish_reverse"] >= 0
    assert summary["bearish_first_wiseman"] + summary["bullish_first_wiseman"] >= 1


def test_ut_bot_strategy_pinescript_generation(tmp_path) -> None:
    out = tmp_path / "ut_bot_strategy.pine"
    path = generate_ut_bot_strategy_pinescript(str(out))

    pine = out.read_text(encoding="utf-8")
    assert path.endswith("ut_bot_strategy.pine")
    assert "strategy(\"UT Bot Alerts Strategy\"" in pine
    assert "ta.atr(c)" in pine
    assert "strategy.entry(\"Long\", strategy.long)" in pine
    assert "strategy.entry(\"Short\", strategy.short)" in pine


def test_pdf_report_generation(tmp_path) -> None:
    data = _sample_df(periods=120, freq="h")
    engine = BacktestEngine(BacktestConfig(initial_capital=10_000, fee_rate=0.0, slippage_rate=0.0))
    result = engine.run(data, SmaCrossoverStrategy(fast=5, slow=20))

    out = tmp_path / "report.pdf"
    written = generate_backtest_pdf_report(result, out)

    assert written == out
    assert out.exists()
    assert out.stat().st_size > 0

    pdf_text = out.read_bytes().decode("latin-1", errors="ignore")
    assert "Total capital" in pdf_text
    assert "Max capital:" in pdf_text
    assert "Min capital:" in pdf_text
    assert "Date" in pdf_text
    assert "Trade lines: long=green, short=red" in pdf_text
    assert "Total fees paid" in pdf_text
    assert "Total interest/funding paid" in pdf_text
    assert "Total volume traded" in pdf_text
    assert "Max effective leverage used" in pdf_text
    assert "Total profit before fees" in pdf_text
    assert "Data quality / outlier report" in pdf_text
    assert "Recommendations" in pdf_text
    assert "Robustness Summary" in pdf_text
    assert "SQN" in pdf_text
    assert "Sign-Flip Stress Test" in pdf_text
    assert "Trade Return Distribution" in pdf_text
    assert "Top profit contributors" in pdf_text
    assert "Top loss contributors" in pdf_text
    assert "Equity Curve Without Top 5% Trades" in pdf_text
    assert "Entry/Exit Outcomes" in pdf_text
    assert "Net PnL" in pdf_text




def test_pdf_report_entry_exit_outcomes_include_full_canonical_labels(tmp_path) -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100, 101, 102, 103, 104, 105],
            "volume": [1_000] * 6,
        },
        index=idx,
    )

    signals = pd.Series([0, 1, 0, -1, 0, 0], index=idx, dtype="int8")
    strategy = _SignalBarStrategy(signals)
    strategy.signal_exit_reason = pd.Series(["", "", "Green Gator PP", "", "Red Gator PP", ""], index=idx, dtype="object")

    result = BacktestEngine().run(data, strategy)

    out = tmp_path / "canonical_labels_report.pdf"
    generate_backtest_pdf_report(result, out)

    pdf_text = out.read_bytes().decode("latin-1", errors="ignore")
    assert "Strategy Profit Protection Green Gator" in pdf_text
    assert "Strategy Profit Protection Red Gator" in pdf_text


def test_clean_pdf_report_generation(tmp_path) -> None:
    data = _sample_df(periods=100, freq="h")
    engine = BacktestEngine(BacktestConfig(initial_capital=10_000, fee_rate=0.0, slippage_rate=0.0))
    result = engine.run(data, SmaCrossoverStrategy(fast=5, slow=20))

    out = tmp_path / "clean_report.pdf"
    generate_backtest_clean_pdf_report(result, out)

    assert out.exists()
    assert out.stat().st_size > 0
    pdf_text = out.read_bytes().decode("latin-1", errors="ignore")
    assert "Trade Signal Diagnostics Report" in pdf_text
    assert "Signal quality monitor" in pdf_text
    assert "Avg giveback from peak" in pdf_text
    assert "Net profit" in pdf_text
    assert "Diagnostics Charts: Equity, Drawdown, Return Distribution" in pdf_text
    assert "Diagnostics Charts: Percent Histograms" in pdf_text or "Trades executed: 0" in pdf_text


def test_trades_dataframe_includes_peak_and_giveback_metrics() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 105.0, 110.0],
            "high": [101.0, 110.0, 120.0, 112.0],
            "low": [99.0, 99.0, 102.0, 108.0],
            "close": [100.0, 105.0, 110.0, 110.0],
            "volume": [1_000.0] * 4,
        },
        index=idx,
    )
    strategy = _SignalBarStrategy(pd.Series([0, 1, 1, 0], index=idx, dtype="int8"))
    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(data, strategy)
    trades = result.trades_dataframe()

    assert "peak_unrealized_return_pct" in trades.columns
    assert "giveback_from_peak_pct" in trades.columns
    assert "capture_ratio_vs_peak" in trades.columns
    assert "realized_price_return_pct" in trades.columns
    assert "peak_unrealized_pnl" in trades.columns
    assert "giveback_from_peak_pnl" in trades.columns
    assert "peak_unrealized_position_return_pct" in trades.columns
    assert "realized_position_return_pct" in trades.columns
    assert "giveback_position_return_pct" in trades.columns
    assert float(trades.iloc[0]["peak_unrealized_return_pct"]) == pytest.approx(0.20, abs=1e-6)
    assert float(trades.iloc[0]["giveback_from_peak_pct"]) == pytest.approx(0.10, abs=1e-6)
    assert float(trades.iloc[0]["capture_ratio_vs_peak"]) == pytest.approx(0.50, abs=1e-6)
    assert float(trades.iloc[0]["realized_price_return_pct"]) == pytest.approx(0.10, abs=1e-6)
    assert float(trades.iloc[0]["peak_unrealized_pnl"]) == pytest.approx(2000.0, abs=1e-6)
    assert float(trades.iloc[0]["giveback_from_peak_pnl"]) == pytest.approx(1000.0, abs=1e-6)
    assert float(trades.iloc[0]["peak_unrealized_position_return_pct"]) == pytest.approx(0.20, abs=1e-6)
    assert float(trades.iloc[0]["realized_position_return_pct"]) == pytest.approx(0.10, abs=1e-6)
    assert float(trades.iloc[0]["giveback_position_return_pct"]) == pytest.approx(0.10, abs=1e-6)

def test_pdf_report_generation_with_multi_asset_underlying_index(tmp_path) -> None:
    data = _sample_df(periods=80, freq="h")
    engine = BacktestEngine(BacktestConfig(initial_capital=10_000, fee_rate=0.0, slippage_rate=0.0))
    result = engine.run(data, SmaCrossoverStrategy(fast=5, slow=20))

    idx = data.index
    btc = pd.Series([40_000 + i * 10 for i in range(len(idx))], index=idx, dtype="float64")
    eth = pd.Series([2_000 + i * 2 for i in range(len(idx))], index=idx, dtype="float64")

    out = tmp_path / "report_with_underlying.pdf"
    generate_backtest_pdf_report(result, out, underlying_prices={"BTC": btc, "ETH": eth})

    pdf_text = out.read_bytes().decode("latin-1", errors="ignore")
    assert "Portfolio Equity vs Underlying" in pdf_text
    assert r"Underlying \(Norm=100\)" in pdf_text


def test_pdf_report_generation_includes_asset_level_stats_page(tmp_path) -> None:
    data = _sample_df(periods=120, freq="h")
    engine = BacktestEngine(BacktestConfig(initial_capital=10_000, fee_rate=0.0, slippage_rate=0.0))
    result_a = engine.run(data, SmaCrossoverStrategy(fast=5, slow=20))
    result_b = engine.run(data * [1.02, 1.02, 1.02, 1.02, 1.0], SmaCrossoverStrategy(fast=8, slow=30))

    out = tmp_path / "report_asset_level.pdf"
    generate_backtest_pdf_report(
        result_a,
        out,
        asset_level_results=[("BTCUSD", result_a), ("ETHUSD", result_b)],
    )

    pdf_text = out.read_bytes().decode("latin-1", errors="ignore")
    assert "Asset-Level Stats" in pdf_text
    assert "BTCUSD" in pdf_text
    assert "ETHUSD" in pdf_text
    assert "Starting capital" in pdf_text
    assert "Total final capital" in pdf_text
    assert "Total PnL" in pdf_text
    assert "Total volume" in pdf_text
    assert "CAGR" in pdf_text
    assert "Max drawdown" in pdf_text


def test_execute_on_signal_bar_respects_same_bar_flips() -> None:
    data = _sample_df(periods=6, freq="h")
    signals = pd.Series([0, 1, -1, 0, -1, -1], index=data.index, dtype="int8")

    engine = BacktestEngine(BacktestConfig(initial_capital=10_000, fee_rate=0.0, slippage_rate=0.0))
    result = engine.run(data, _SignalBarStrategy(signals))

    trades = result.trades_dataframe()

    assert len(trades) == 3
    assert list(trades["side"]) == ["long", "short", "short"]
    assert list(trades["entry_time"]) == [data.index[1], data.index[2], data.index[4]]
    assert list(trades["exit_time"]) == [data.index[2], data.index[3], data.index[5]]


def test_trade_holding_bars_work_with_duplicate_timestamps() -> None:
    idx = pd.to_datetime(
        [
            "2024-01-01 00:00:00+00:00",
            "2024-01-01 01:00:00+00:00",
            "2024-01-01 01:00:00+00:00",
            "2024-01-01 02:00:00+00:00",
        ]
    )
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0, 102.0],
            "high": [101.0, 101.0, 102.0, 103.0],
            "low": [99.0, 99.0, 100.0, 101.0],
            "close": [100.0, 100.0, 101.0, 102.0],
            "volume": [1_000, 1_000, 1_000, 1_000],
        },
        index=idx,
    )

    signals = pd.Series([0, 1, 0, 0], index=idx, dtype="int8")
    engine = BacktestEngine(BacktestConfig(initial_capital=10_000, fee_rate=0.0, slippage_rate=0.0))
    result = engine.run(data, _SignalBarStrategy(signals))

    assert len(result.trades) == 1
    assert result.trades[0].holding_bars == 1


def test_signal_fill_price_hint_overrides_bar_open_execution_price() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [1000.0, 995.0, 970.0, 960.0],
            "high": [1010.0, 1002.0, 975.0, 965.0],
            "low": [980.0, 975.0, 960.0, 950.0],
            "close": [990.0, 978.0, 965.0, 955.0],
            "volume": [1000, 1000, 1000, 1000],
        },
        index=idx,
    )

    # Pending bearish stop at 980 is known at t0 close and triggers on t1.
    # The engine should fill at the known stop level (980), not t2 open (970).
    signals = pd.Series([0, -1, -1, -1], index=idx, dtype="int8")
    fills = pd.Series([np.nan, 980.0, np.nan, np.nan], index=idx, dtype="float64")

    engine = BacktestEngine(BacktestConfig(initial_capital=10_000, fee_rate=0.0, slippage_rate=0.0))
    result = engine.run(data, _SignalFillStrategy(signals, fills))

    trades = result.trades_dataframe()
    assert len(trades) == 1
    assert trades.iloc[0]["side"] == "short"
    assert trades.iloc[0]["entry_time"] == idx[2]
    assert trades.iloc[0]["entry_price"] == 980.0


def test_signal_magnitude_scales_contracts_without_signal_contracts() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1000, 1000, 1000, 1000],
        },
        index=idx,
    )

    signals = pd.Series([0.0, -4.0, -4.0, 0.0], index=idx, dtype="float64")

    cases = [
        (BacktestConfig(initial_capital=10_000, fee_rate=0.0, slippage_rate=0.0, trade_size_mode="percent_of_equity", trade_size_value=0.1), 40.0),
        (BacktestConfig(initial_capital=10_000, fee_rate=0.0, slippage_rate=0.0, trade_size_mode="usd", trade_size_value=1_000), 40.0),
        (BacktestConfig(initial_capital=10_000, fee_rate=0.0, slippage_rate=0.0, trade_size_mode="units", trade_size_value=2), 8.0),
    ]

    for cfg, expected_units in cases:
        engine = BacktestEngine(cfg)
        result = engine.run(data, _SignalBarStrategy(signals))
        trades = result.trades_dataframe()

        assert len(trades) == 1
        assert trades.iloc[0]["side"] == "short"
        assert trades.iloc[0]["units"] == expected_units


def test_max_position_size_caps_growth_but_keeps_trading_at_cap_notional() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0],
            "volume": [1000, 1000, 1000, 1000, 1000],
        },
        index=idx,
    )

    signals = pd.Series([0, 1, 3, 3, 0], index=idx, dtype="int8")

    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="units",
            trade_size_value=2.0,
            max_position_size=400.0,
        )
    )
    result = engine.run(data, _SignalBarStrategy(signals))

    trades = result.trades_dataframe()

    assert len(trades) == 1
    assert trades.iloc[0]["units"] == 4.0
    add_events = [e for e in result.execution_events if e.event_type == "add"]
    assert len(add_events) == 1
    assert add_events[0].units == 2.0


def test_hybrid_min_usd_percent_uses_minimum_floor_when_percent_is_lower() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1000, 1000, 1000, 1000],
        },
        index=idx,
    )

    signals = pd.Series([0, 1, 1, 0], index=idx, dtype="int8")

    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="hybrid_min_usd_percent",
            trade_size_value=0.01,
            trade_size_min_usd=500.0,
        )
    )
    result = engine.run(data, _SignalBarStrategy(signals))
    trades = result.trades_dataframe()

    assert len(trades) == 1
    assert trades.iloc[0]["units"] == 5.0


def test_equity_milestone_usd_sizing_steps_up_only_after_thresholds() -> None:
    milestones = parse_trade_size_equity_milestones("11000:1500,12000:2000")

    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="equity_milestone_usd",
            trade_size_value=1_000.0,
            trade_size_equity_milestones=milestones,
        )
    )

    closes = pd.Series([100.0, 100.0], dtype="float64")
    assert engine._resolve_units(10_000.0, 100.0, None, 1, closes, 365) == pytest.approx(10.0)
    assert engine._resolve_units(11_500.0, 100.0, None, 1, closes, 365) == pytest.approx(15.0)
    assert engine._resolve_units(11_900.0, 100.0, None, 1, closes, 365) == pytest.approx(15.0)
    assert engine._resolve_units(12_500.0, 100.0, None, 1, closes, 365) == pytest.approx(20.0)


def test_parse_trade_size_equity_milestones_rejects_duplicate_thresholds() -> None:
    with pytest.raises(ValueError, match="unique"):
        parse_trade_size_equity_milestones("15000:1500,15000:2000")


def test_stop_loss_scaled_sizes_position_by_entry_to_stop_risk() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [950.0, 950.0, 1000.0, 1005.0, 1000.0],
            "high": [960.0, 960.0, 1010.0, 1015.0, 1010.0],
            "low": [940.0, 940.0, 990.0, 995.0, 990.0],
            "close": [950.0, 950.0, 1000.0, 1005.0, 1000.0],
            "volume": [1000, 1000, 1000, 1000, 1000],
        },
        index=idx,
    )
    signals = pd.Series([0, 0, 1, 1, 0], index=idx, dtype="int8")
    fills = pd.Series([np.nan, np.nan, 1000.0, np.nan, np.nan], index=idx, dtype="float64")
    stop_losses = pd.Series([np.nan, np.nan, 900.0, np.nan, np.nan], index=idx, dtype="float64")

    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="stop_loss_scaled",
            trade_size_value=0.01,
        )
    )
    result = engine.run(data, _SignalFillStopStrategy(signals, fills, stop_losses))
    trades = result.trades_dataframe()

    assert len(trades) == 1
    assert trades.iloc[0]["units"] == pytest.approx(1.0)


def test_stop_loss_scaled_carries_forward_active_stop_loss_while_position_is_open() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [950.0, 950.0, 1000.0, 1005.0, 1000.0],
            "high": [960.0, 960.0, 1010.0, 1015.0, 1010.0],
            "low": [940.0, 940.0, 990.0, 995.0, 990.0],
            "close": [950.0, 950.0, 1000.0, 1005.0, 1000.0],
            "volume": [1000, 1000, 1000, 1000, 1000],
        },
        index=idx,
    )
    signals = pd.Series([0, 0, 1, 1, 0], index=idx, dtype="int8")
    fills = pd.Series([np.nan, np.nan, 1000.0, np.nan, np.nan], index=idx, dtype="float64")
    stop_losses = pd.Series([np.nan, np.nan, 900.0, np.nan, np.nan], index=idx, dtype="float64")

    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="stop_loss_scaled",
            trade_size_value=0.01,
        )
    )

    result = engine.run(data, _SignalFillStopStrategy(signals, fills, stop_losses))

    assert len(result.trades) == 1


def test_engine_exits_open_position_when_strategy_stop_loss_is_touched() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0, 100.0, 96.0],
            "high": [101.0, 102.0, 103.0, 101.0, 97.0],
            "low": [99.0, 99.5, 100.0, 95.0, 94.0],
            "close": [100.0, 101.0, 100.5, 96.0, 95.0],
            "volume": [1, 1, 1, 1, 1],
        },
        index=idx,
    )
    signals = pd.Series([0, 1, 1, 0, 0], index=idx, dtype="int8")
    fills = pd.Series([np.nan, 100.0, np.nan, np.nan, np.nan], index=idx, dtype="float64")
    stop_losses = pd.Series([np.nan, 98.0, 98.0, 98.0, np.nan], index=idx, dtype="float64")

    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(
        data,
        _SignalFillStopStrategy(signals, fills, stop_losses),
    )

    trades = result.trades_dataframe()
    assert len(trades) == 1
    assert trades.iloc[0]["exit_signal"] == "Strategy Stop Loss Bullish 1W"
    assert trades.iloc[0]["exit_price"] == pytest.approx(98.0)
    assert result.positions.iloc[3] == 0
    assert result.positions.iloc[4] == 0


def test_engine_ignores_conflicting_duplicate_intrabar_entries_for_same_reason() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1, 1, 1, 1],
        },
        index=idx,
    )
    intrabar_events = {
        1: [
            {"event_type": "entry", "side": 1, "price": 100.0, "contracts": 1.0, "reason": "Bearish 1W"},
            {"event_type": "entry", "side": -1, "price": 100.0, "contracts": 1.0, "reason": "Bearish 1W"},
        ]
    }
    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(
        data,
        _IntrabarEventsStrategy(intrabar_events),
    )

    entry_events = [event for event in result.execution_events if event.event_type == "entry"]
    assert len(entry_events) == 1
    assert entry_events[0].side == "buy"
    assert result.positions.iloc[1] == 1


def test_engine_filters_opposite_intrabar_entries_for_single_direction_non_reversal_signal() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1, 1, 1, 1],
        },
        index=idx,
    )
    signals = pd.Series([0, 1, 1, 1], index=idx, dtype="int8")
    intrabar_events = {
        1: [
            {"event_type": "entry", "side": -1, "price": 100.0, "contracts": 1.0, "reason": "Bearish 1W"},
            {"event_type": "entry", "side": 1, "price": 100.0, "contracts": 1.0, "reason": "Bullish 1W"},
        ]
    }
    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(
        data,
        _SignalIntrabarEventsStrategy(signals, intrabar_events),
    )

    entry_events = [event for event in result.execution_events if event.event_type == "entry"]
    assert len(entry_events) == 1
    assert entry_events[0].side == "buy"
    assert result.positions.iloc[1] == 1


def test_engine_allows_mixed_intrabar_entries_when_reversal_marker_is_present() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1, 1, 1, 1],
        },
        index=idx,
    )
    signals = pd.Series([0, -1, -1, -1], index=idx, dtype="int8")
    reversal_side = pd.Series([0, -1, 0, 0], index=idx, dtype="int8")
    intrabar_events = {
        1: [
            {"event_type": "entry", "side": 1, "price": 100.0, "contracts": 1.0, "reason": "Bullish 1W"},
            {"event_type": "exit", "side": -1, "price": 99.5, "contracts": 0.0, "reason": "Strategy Stop Loss Bullish 1W"},
            {"event_type": "entry", "side": -1, "price": 99.5, "contracts": 1.0, "reason": "Bearish 1W-R"},
        ]
    }
    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(
        data,
        _SignalIntrabarEventsStrategy(signals, intrabar_events, reversal_side=reversal_side),
    )

    bar_events = [event for event in result.execution_events if event.time == idx[1]]
    assert [event.event_type for event in bar_events[:3]] == ["entry", "exit", "entry"]
    assert [event.side for event in bar_events[:3]] == ["buy", "sell", "sell"]
    assert result.positions.iloc[1] == -1


def test_engine_intrabar_reversal_executes_before_same_bar_non_gap_liquidation() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 99.0, 100.0],
            "high": [101.0, 101.0, 200.0, 101.0],
            "low": [99.0, 99.0, 98.0, 99.0],
            "close": [100.0, 100.0, 99.0, 100.0],
            "volume": [1, 1, 1, 1],
        },
        index=idx,
    )
    signals = pd.Series([0, -1, 0, 0], index=idx, dtype="int8")
    intrabar_events = {
        2: [
            {"event_type": "entry", "side": 1, "price": 110.0, "contracts": 1.0, "reason": "Bullish 1W"},
        ]
    }

    result = BacktestEngine(
        BacktestConfig(
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="percent_of_equity",
            trade_size_value=2.0,
        )
    ).run(
        data,
        _SignalIntrabarEventsStrategy(signals, intrabar_events),
    )

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert [event.event_type for event in bar_events[:2]] == ["exit", "entry"]
    assert [event.side for event in bar_events[:2]] == ["buy", "buy"]
    assert not any(event.event_type == "liquidation" for event in bar_events)
    assert result.positions.iloc[2] == 1


def test_engine_non_reversal_flat_bar_keeps_only_first_intrabar_entry_side() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1, 1, 1, 1],
        },
        index=idx,
    )
    signals = pd.Series([0, 0, 0, 0], index=idx, dtype="int8")
    intrabar_events = {
        1: [
            {"event_type": "entry", "side": 1, "price": 100.0, "contracts": 1.0, "reason": "Bullish 1W"},
            {"event_type": "exit", "side": -1, "price": 99.5, "contracts": 0.0, "reason": "Strategy Stop Loss Bullish 1W"},
            {"event_type": "entry", "side": -1, "price": 99.5, "contracts": 1.0, "reason": "Bearish 1W"},
        ]
    }
    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(
        data,
        _SignalIntrabarEventsStrategy(signals, intrabar_events),
    )

    bar_events = [event for event in result.execution_events if event.time == idx[1]]
    assert [event.event_type for event in bar_events[:2]] == ["entry", "exit"]
    assert [event.side for event in bar_events[:2]] == ["buy", "sell"]
    assert all(not (event.event_type == "entry" and event.side == "sell") for event in bar_events)
    assert result.positions.iloc[1] == 0


def test_engine_allows_same_bar_reversal_after_strategy_stop_loss() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 99.0, 98.0, 97.0],
            "high": [101.0, 101.0, 100.0, 99.0, 98.0],
            "low": [99.0, 99.0, 97.5, 96.0, 95.0],
            "close": [100.0, 100.0, 98.5, 97.0, 96.0],
            "volume": [1, 1, 1, 1, 1],
        },
        index=idx,
    )
    signals = pd.Series([0, 1, -1, -1, 0], index=idx, dtype="int8")
    fills = pd.Series([np.nan, 100.0, 98.0, np.nan, np.nan], index=idx, dtype="float64")
    stop_losses = pd.Series([np.nan, 98.0, np.nan, np.nan, np.nan], index=idx, dtype="float64")
    reversal_side = pd.Series([0, 0, -1, 0, 0], index=idx, dtype="int8")

    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(
        data,
        _SignalFillStopStrategy(signals, fills, stop_losses, first_wiseman_reversal_side=reversal_side),
    )

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert [event.event_type for event in bar_events[:2]] == ["exit", "entry"]
    assert [event.side for event in bar_events[:2]] == ["sell", "sell"]
    assert bar_events[1].price == pytest.approx(98.0)


def test_engine_does_not_reenter_opposite_after_strategy_stop_without_reversal_marker() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 99.0, 98.0, 97.0],
            "high": [101.0, 101.0, 100.0, 99.0, 98.0],
            "low": [99.0, 99.0, 97.5, 96.0, 95.0],
            "close": [100.0, 100.0, 98.5, 97.0, 96.0],
            "volume": [1, 1, 1, 1, 1],
        },
        index=idx,
    )
    signals = pd.Series([0, 1, -1, -1, 0], index=idx, dtype="int8")
    fills = pd.Series([np.nan, 100.0, 98.0, np.nan, np.nan], index=idx, dtype="float64")
    stop_losses = pd.Series([np.nan, 98.0, np.nan, np.nan, np.nan], index=idx, dtype="float64")

    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(
        data,
        _SignalFillStopStrategy(signals, fills, stop_losses),
    )

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert [event.event_type for event in bar_events] == ["exit"]
    assert [event.side for event in bar_events] == ["sell"]
    assert result.positions.iloc[2] == 0


def test_engine_flat_intent_does_not_open_opposite_intrabar_entry_even_with_reversal_marker() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 99.0, 98.0, 97.0],
            "high": [101.0, 101.0, 100.0, 99.0, 98.0],
            "low": [99.0, 99.0, 97.5, 96.0, 95.0],
            "close": [100.0, 100.0, 98.5, 97.0, 96.0],
            "volume": [1, 1, 1, 1, 1],
        },
        index=idx,
    )
    signals = pd.Series([0, 1, 0, 0, 0], index=idx, dtype="int8")
    fills = pd.Series([np.nan, 100.0, np.nan, np.nan, np.nan], index=idx, dtype="float64")
    reversal_side = pd.Series([0, 0, -1, 0, 0], index=idx, dtype="int8")
    intrabar_events = {
        2: [
            {"event_type": "exit", "side": 1, "price": 98.0, "contracts": 0.0, "reason": "Strategy Stop Loss Bullish 1W"},
            {"event_type": "entry", "side": -1, "price": 98.0, "contracts": 1.0, "reason": "Bearish 1W-R"},
        ]
    }
    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(
        data,
        _SignalIntrabarEventsStrategy(signals, intrabar_events, reversal_side=reversal_side),
    )

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert [event.event_type for event in bar_events] == ["exit"]
    assert [event.side for event in bar_events] == ["sell"]
    assert result.positions.iloc[2] == 0


def test_engine_flat_intent_blocks_same_side_intrabar_reentry() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 99.0, 98.0, 97.0],
            "high": [101.0, 101.0, 100.0, 99.0, 98.0],
            "low": [99.0, 99.0, 97.5, 96.0, 95.0],
            "close": [100.0, 100.0, 98.5, 97.0, 96.0],
            "volume": [1, 1, 1, 1, 1],
        },
        index=idx,
    )
    signals = pd.Series([0, 1, 0, 0, 0], index=idx, dtype="int8")
    intrabar_events = {
        2: [
            {"event_type": "exit", "side": 1, "price": 98.0, "contracts": 0.0, "reason": "Strategy Stop Loss Bullish 1W"},
            {"event_type": "entry", "side": 1, "price": 98.0, "contracts": 1.0, "reason": "Bullish 1W"},
        ]
    }
    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(
        data,
        _SignalIntrabarEventsStrategy(signals, intrabar_events),
    )

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert [event.event_type for event in bar_events] == ["exit"]
    assert [event.side for event in bar_events] == ["sell"]
    assert result.positions.iloc[2] == 0


def test_engine_intrabar_stop_reason_is_normalized_to_open_position_side() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 99.0, 98.0],
            "high": [101.0, 101.0, 100.0, 99.0],
            "low": [99.0, 99.0, 97.5, 97.0],
            "close": [100.0, 100.0, 98.5, 97.5],
            "volume": [1, 1, 1, 1],
        },
        index=idx,
    )
    signals = pd.Series([0, 1, 0, 0], index=idx, dtype="int8")
    intrabar_events = {
        2: [
            {"event_type": "exit", "side": -1, "price": 98.0, "contracts": 0.0, "reason": "Strategy Stop Loss Bearish 1W"},
        ]
    }

    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(
        data,
        _SignalIntrabarEventsStrategy(signals, intrabar_events),
    )

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert len(bar_events) == 1
    assert bar_events[0].event_type == "exit"
    assert bar_events[0].strategy_reason == "Strategy Stop Loss Bullish 1W"
    assert result.trades[0].exit_signal == "Strategy Stop Loss Bullish 1W"


def test_engine_ignores_short_stop_loss_price_that_is_already_below_market() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 99.0, 98.0],
            "high": [101.0, 100.5, 99.5, 98.5],
            "low": [99.0, 98.5, 97.5, 96.5],
            "close": [100.0, 99.0, 98.0, 97.0],
            "volume": [1, 1, 1, 1],
        },
        index=idx,
    )
    signals = pd.Series([0, -1, -1, 0], index=idx, dtype="int8")
    fills = pd.Series([np.nan, 100.0, np.nan, 97.0], index=idx, dtype="float64")
    stop_losses = pd.Series([np.nan, 95.0, 95.0, np.nan], index=idx, dtype="float64")

    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(
        data,
        _SignalFillStopStrategy(signals, fills, stop_losses),
    )

    assert result.positions.iloc[2] == -1
    assert result.trades[0].exit_time == idx[3]
    assert result.trades[0].exit_signal == "Signal Intent Flat from Bearish 1W"


def test_engine_flat_intent_allows_same_side_add_before_intrabar_exit() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 99.0, 98.0, 97.0],
            "high": [101.0, 101.0, 100.0, 99.0, 98.0],
            "low": [99.0, 99.0, 97.5, 96.0, 95.0],
            "close": [100.0, 100.0, 98.5, 97.0, 96.0],
            "volume": [1, 1, 1, 1, 1],
        },
        index=idx,
    )
    signals = pd.Series([0, 1, 0, 0, 0], index=idx, dtype="int8")
    intrabar_events = {
        2: [
            {"event_type": "entry", "side": 1, "price": 99.5, "contracts": 1.0, "reason": "Bullish Add-on"},
            {"event_type": "exit", "side": 1, "price": 98.0, "contracts": 0.0, "reason": "Strategy Stop Loss Bullish 1W"},
        ]
    }
    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(
        data,
        _SignalIntrabarEventsStrategy(signals, intrabar_events),
    )

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert [event.event_type for event in bar_events[:2]] == ["add", "exit"]
    assert [event.side for event in bar_events[:2]] == ["buy", "sell"]
    assert result.positions.iloc[2] == 0


def test_engine_flat_intent_allows_opposite_entry_then_same_bar_stop_sequence() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 102.0, 101.0],
            "low": [99.0, 99.0, 98.0, 99.0],
            "close": [100.0, 100.0, 99.5, 100.0],
            "volume": [1, 1, 1, 1],
        },
        index=idx,
    )
    signals = pd.Series([0, -1, 0, 0], index=idx, dtype="int8")
    intrabar_events = {
        2: [
            {"event_type": "entry", "side": 1, "price": 101.0, "contracts": 1.0, "reason": "Bullish 1W"},
            {"event_type": "exit", "side": -1, "price": 99.0, "contracts": 0.0, "reason": "Strategy Stop Loss Bullish 1W"},
        ]
    }

    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(
        data,
        _SignalIntrabarEventsStrategy(signals, intrabar_events),
    )

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert [str(event.strategy_reason) for event in bar_events] == [
        "Strategy Reversal to Bullish 1W",
        "Bullish 1W",
        "Strategy Stop Loss Bullish 1W",
    ]
    assert result.positions.iloc[2] == 0


def test_engine_flat_intent_allows_post_stop_same_bar_1w_reversal_entry() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 102.0, 101.0],
            "low": [99.0, 99.0, 98.0, 99.0],
            "close": [100.0, 100.0, 99.5, 100.0],
            "volume": [1, 1, 1, 1],
        },
        index=idx,
    )
    signals = pd.Series([0, -1, 0, 0], index=idx, dtype="int8")
    intrabar_events = {
        2: [
            {"event_type": "entry", "side": 1, "price": 101.0, "contracts": 1.0, "reason": "Bullish 1W"},
            {"event_type": "exit", "side": -1, "price": 99.0, "contracts": 0.0, "reason": "Strategy Stop Loss Bullish 1W"},
            {"event_type": "entry", "side": -1, "price": 99.0, "contracts": 1.0, "reason": "Bearish 1W-R"},
        ]
    }

    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(
        data,
        _SignalIntrabarEventsStrategy(signals, intrabar_events),
    )

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert [str(event.strategy_reason) for event in bar_events] == [
        "Strategy Reversal to Bullish 1W",
        "Bullish 1W",
        "Strategy Stop Loss Bullish 1W",
        "Bearish 1W-R",
    ]
    assert result.positions.iloc[2] == -1


def test_engine_flat_intent_blocks_opposite_add_on_entries() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1, 1, 1, 1],
        },
        index=idx,
    )
    signals = pd.Series([0, -1, 0, 0], index=idx, dtype="int8")
    intrabar_events = {
        2: [
            {"event_type": "entry", "side": -1, "price": 99.5, "contracts": 1.0, "reason": "Bearish Add-on Fractal"},
            {"event_type": "entry", "side": 1, "price": 100.5, "contracts": 1.0, "reason": "Bullish Add-on Fractal"},
            {"event_type": "exit", "side": -1, "price": 100.2, "contracts": 0.0, "reason": "Signal Intent Flat from Bearish 1W"},
        ]
    }

    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(
        data,
        _SignalIntrabarEventsStrategy(signals, intrabar_events),
    )

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert [event.event_type for event in bar_events] == ["add", "exit"]
    assert [event.side for event in bar_events] == ["sell", "buy"]
    assert result.positions.iloc[2] == 0


def test_engine_blocks_opposite_add_on_even_with_reversal_marker_when_signal_direction_is_unchanged() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1, 1, 1, 1],
        },
        index=idx,
    )
    signals = pd.Series([0, -1, -1, -1], index=idx, dtype="int8")
    reversal_side = pd.Series([0, 0, 1, 0], index=idx, dtype="int8")
    intrabar_events = {
        2: [
            {"event_type": "entry", "side": 1, "price": 100.5, "contracts": 1.0, "reason": "Bullish Add-on Fractal"},
        ]
    }

    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(
        data,
        _SignalIntrabarEventsStrategy(signals, intrabar_events, reversal_side=reversal_side),
    )

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert bar_events == []
    assert result.positions.iloc[2] == -1


def test_engine_blocks_wrong_direction_bearish_add_on_fractal_reason() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1, 1, 1, 1],
        },
        index=idx,
    )
    signals = pd.Series([0, -1, -1, -1], index=idx, dtype="int8")
    intrabar_events = {
        2: [
            {"event_type": "entry", "side": 1, "price": 100.4, "contracts": 1.0, "reason": "Bearish Add-on Fractal"},
        ]
    }

    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(
        data,
        _SignalIntrabarEventsStrategy(signals, intrabar_events),
    )

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert bar_events == []
    assert result.positions.iloc[2] == -1


def test_engine_blocks_opposite_1w_entry_when_signal_direction_is_unidirectional() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1, 1, 1, 1],
        },
        index=idx,
    )
    signals = pd.Series([0, -1, -1, -1], index=idx, dtype="int8")
    reversal_side = pd.Series([0, 0, 1, 0], index=idx, dtype="int8")
    intrabar_events = {
        2: [
            {"event_type": "entry", "side": 1, "price": 100.4, "contracts": 1.0, "reason": "Bullish 1W"},
        ]
    }

    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(
        data,
        _SignalIntrabarEventsStrategy(signals, intrabar_events, reversal_side=reversal_side),
    )

    bar_events = [event for event in result.execution_events if event.time == idx[2]]
    assert bar_events == []
    assert result.positions.iloc[2] == -1


def test_engine_ignores_add_on_fractal_entry_when_no_position_is_open() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0],
            "volume": [1, 1, 1],
        },
        index=idx,
    )
    signals = pd.Series([0, 0, 0], index=idx, dtype="int8")
    intrabar_events = {
        1: [
            {"event_type": "entry", "side": 1, "price": 100.5, "contracts": 1.0, "reason": "Bullish Add-on Fractal"},
        ]
    }

    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(
        data,
        _SignalIntrabarEventsStrategy(signals, intrabar_events),
    )

    assert [event for event in result.execution_events if event.time == idx[1]] == []
    assert result.positions.iloc[1] == 0


def test_engine_handles_many_same_side_intrabar_add_on_fractals_without_side_flip() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0] * len(idx),
            "high": [101.0] * len(idx),
            "low": [99.0] * len(idx),
            "close": [100.0] * len(idx),
            "volume": [1] * len(idx),
        },
        index=idx,
    )
    signals = pd.Series([0] + [-1] * (len(idx) - 1), index=idx, dtype="int8")
    intrabar_events: dict[int, list[dict[str, float | int | str]]] = {}
    for bar in range(2, len(idx) - 1):
        intrabar_events[bar] = [
            {"event_type": "entry", "side": -1, "price": 99.5, "contracts": 1.0, "reason": "Bearish Add-on Fractal"},
        ]

    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(
        data,
        _SignalIntrabarEventsStrategy(signals, intrabar_events),
    )

    assert all(float(pos) <= 0 for pos in result.positions.iloc[1:-1])
    add_events = [event for event in result.execution_events if event.event_type == "add"]
    assert len(add_events) == len(idx) - 3
    assert all(event.side == "sell" for event in add_events)
    assert not any(event.side == "buy" and event.event_type in {"entry", "add"} for event in result.execution_events)


def test_engine_handles_many_signal_contract_add_ons_without_instability() -> None:
    idx = pd.date_range("2024-01-01", periods=25, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0] * len(idx),
            "high": [101.0] * len(idx),
            "low": [99.0] * len(idx),
            "close": [100.0] * len(idx),
            "volume": [1] * len(idx),
        },
        index=idx,
    )
    signals = pd.Series([-1] * len(idx), index=idx, dtype="int8")
    contracts = pd.Series([float(i + 1) for i in range(len(idx))], index=idx, dtype="float64")
    fills = pd.Series([100.0] * len(idx), index=idx, dtype="float64")

    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(
        data,
        _SignalContractsExitReasonStrategy(signals, contracts, fill_prices=fills),
    )

    active_side = int(result.positions.iloc[1])
    assert active_side != 0
    assert all(int(pos) == active_side for pos in result.positions.iloc[1:])
    add_events = [event for event in result.execution_events if event.event_type == "add"]
    assert len(add_events) >= 20
    entry_add_sides = [event.side for event in result.execution_events if event.event_type in {"entry", "add"}]
    assert len(set(entry_add_sides)) == 1


def test_engine_prefers_signal_direction_when_contract_sign_conflicts() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0] * len(idx),
            "high": [101.0] * len(idx),
            "low": [99.0] * len(idx),
            "close": [100.0] * len(idx),
            "volume": [1] * len(idx),
        },
        index=idx,
    )
    signals = pd.Series([-1] * len(idx), index=idx, dtype="int8")
    contracts = pd.Series([-1.0, -2.0, -3.0, 3.0, -4.0, -5.0], index=idx, dtype="float64")

    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(
        data,
        _SignalContractsStrategy(signals, contracts),
    )

    entry_add_events = [event for event in result.execution_events if event.event_type in {"entry", "add"}]
    assert entry_add_events
    assert len({event.side for event in entry_add_events}) == 1
    assert result.positions.iloc[1] != 0
    assert all(int(pos) == int(result.positions.iloc[1]) for pos in result.positions.iloc[1:])


def test_bw_strategy_keeps_signal_and_contract_direction_aligned_with_many_add_ons() -> None:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=1500, freq="h", tz="UTC")
    returns = rng.normal(0.0, 0.002, len(idx))
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = close * (1.0 + rng.normal(0.0, 0.0005, len(idx)))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.001, 0.0005, len(idx))))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.001, 0.0005, len(idx))))
    data = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1000},
        index=idx,
    )

    strategy = BWStrategy(
        divergence_filter_bars=0,
        fractal_add_on_contracts=1,
        ntd_initial_fractal_enabled=True,
        red_teeth_profit_protection_enabled=False,
    )
    signals = strategy.generate_signals(data)
    contracts = strategy.signal_contracts
    assert contracts is not None

    signal_side = np.sign(signals.astype("float64"))
    contract_side = np.sign(contracts.astype("float64"))
    mismatch = (signal_side != 0) & (contract_side != 0) & (signal_side != contract_side)
    assert not bool(mismatch.any())


def test_stop_loss_scaled_skips_reentry_until_new_stop_loss_signal_exists() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 95.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 96.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 94.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 95.0, 100.0, 100.0],
            "volume": [1000, 1000, 1000, 1000, 1000, 1000],
        },
        index=idx,
    )
    signals = pd.Series([0, 0, 1, 1, 1, 0], index=idx, dtype="int8")
    fills = pd.Series([np.nan, np.nan, 100.0, np.nan, np.nan, np.nan], index=idx, dtype="float64")
    stop_losses = pd.Series([np.nan, np.nan, 90.0, np.nan, np.nan, np.nan], index=idx, dtype="float64")

    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="stop_loss_scaled",
            trade_size_value=0.01,
            max_loss=50.0,
        )
    )

    result = engine.run(data, _SignalFillStopStrategy(signals, fills, stop_losses))

    assert len(result.trades) == 1


def test_stop_loss_scaled_skips_entries_without_stop_loss_prices() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1000, 1000, 1000, 1000],
        },
        index=idx,
    )
    signals = pd.Series([0, 1, 1, 0], index=idx, dtype="int8")

    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="stop_loss_scaled",
            trade_size_value=0.01,
        )
    )

    result = engine.run(data, _SignalBarStrategy(signals))

    assert len(result.trades) == 0


def test_volatility_scale_excludes_current_bar_close_to_avoid_lookahead() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    base_closes = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=idx, dtype="float64")
    spiked_current_close = base_closes.copy()
    spiked_current_close.iloc[4] = 10_000.0

    engine = BacktestEngine(
        BacktestConfig(
            trade_size_mode="volatility_scaled",
            volatility_target_annual=0.10,
            volatility_lookback=4,
            volatility_min_scale=0.10,
            volatility_max_scale=5.0,
        )
    )

    bar_index = 4
    periods_per_year = 24 * 365

    baseline_scale = engine._volatility_scale(base_closes, bar_index=bar_index, periods_per_year=periods_per_year)
    spiked_scale = engine._volatility_scale(spiked_current_close, bar_index=bar_index, periods_per_year=periods_per_year)

    assert spiked_scale == pytest.approx(baseline_scale)


def test_volatility_scaled_uses_all_available_history_when_lookback_exceeds_data() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    closes = [100.0, 120.0, 80.0, 120.0, 80.0, 80.0]
    data = pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [1000] * len(closes),
        },
        index=idx,
    )

    signals = pd.Series([0, 0, 0, 1, 1, 0], index=idx, dtype="int8")

    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="volatility_scaled",
            trade_size_value=0.1,
            volatility_target_annual=0.10,
            volatility_lookback=100,
            volatility_min_scale=0.10,
            volatility_max_scale=5.0,
        )
    )
    result = engine.run(data, _SignalBarStrategy(signals))
    trades = result.trades_dataframe()

    assert len(trades) == 1
    assert trades.iloc[0]["units"] < (10_000 * 0.1 / 120.0)


def test_volatility_scaled_size_reduces_exposure_when_realized_vol_is_high() -> None:
    idx = pd.date_range("2024-01-01", periods=7, freq="h", tz="UTC")
    closes = [100.0, 100.0, 110.0, 90.0, 110.0, 90.0, 90.0]
    data = pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [1000] * len(closes),
        },
        index=idx,
    )

    signals = pd.Series([0, 0, 0, 0, 1, 1, 0], index=idx, dtype="int8")

    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="volatility_scaled",
            trade_size_value=0.1,
            volatility_target_annual=0.10,
            volatility_lookback=3,
            volatility_min_scale=0.10,
            volatility_max_scale=5.0,
        )
    )
    result = engine.run(data, _SignalBarStrategy(signals))
    trades = result.trades_dataframe()

    assert len(trades) == 1
    assert trades.iloc[0]["units"] < (10_000 * 0.1 / 90.0)


def test_partial_rebalance_reductions_are_recorded_as_closed_trades() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 90.0, 80.0, 80.0],
            "high": [100.0, 100.0, 90.0, 80.0, 80.0],
            "low": [100.0, 100.0, 90.0, 80.0, 80.0],
            "close": [100.0, 100.0, 90.0, 80.0, 80.0],
            "volume": [1000, 1000, 1000, 1000, 1000],
        },
        index=idx,
    )

    signals = pd.Series([0, -1, -1, -1, 0], index=idx, dtype="int8")
    contracts = pd.Series([0.0, 4.0, 1.0, 1.0, 0.0], index=idx, dtype="float64")

    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.0,
            slippage_rate=0.0,
            trade_size_mode="units",
            trade_size_value=1.0,
        )
    )
    result = engine.run(data, _SignalContractsStrategy(signals, contracts))

    trades = result.trades_dataframe()

    assert len(trades) == 2
    assert list(trades["units"]) == [3.0, 1.0]
    assert list(trades["entry_price"]) == [100.0, 100.0]
    assert list(trades["exit_price"]) == [90.0, 80.0]
    assert result.stats["total_trades"] == 2.0
    assert result.stats["avg_trade_pnl"] == 25.0


def test_trade_pnl_reconciles_entry_exit_fees_and_financing_costs() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 105.0, 110.0],
            "high": [100.0, 105.0, 110.0, 110.0],
            "low": [100.0, 100.0, 105.0, 110.0],
            "close": [100.0, 105.0, 110.0, 110.0],
            "volume": [1_000, 1_000, 1_000, 1_000],
        },
        index=idx,
    )
    signals = pd.Series([0, 1, 1, 0], index=idx, dtype="int8")

    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000,
            fee_rate=0.01,
            slippage_rate=0.0,
            overnight_rate_annual=0.365,
            trade_size_mode="units",
            trade_size_value=10.0,
        )
    )
    result = engine.run(data, _SignalBarStrategy(signals))
    trades = result.trades_dataframe()

    assert len(trades) == 1
    assert trades.iloc[0]["gross_pnl"] == pytest.approx(100.0)
    assert trades.iloc[0]["entry_fee"] == pytest.approx(10.0)
    assert trades.iloc[0]["exit_fee"] == pytest.approx(11.0)
    assert trades.iloc[0]["financing_cost"] == pytest.approx(2.15)
    assert trades.iloc[0]["pnl"] == pytest.approx(76.85)
    assert result.total_fees_paid == pytest.approx(21.0)
    assert result.total_financing_paid == pytest.approx(2.15)
    assert float(trades["pnl"].sum()) == pytest.approx(result.equity_curve.iloc[-1] - result.equity_curve.iloc[0])
    assert result.stats["avg_trade_pnl"] == pytest.approx(76.85)


def test_exposure_uses_position_history_not_nonzero_returns() -> None:
    equity = pd.Series([10_000.0, 10_000.0, 10_000.0], index=pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"))
    returns = equity.pct_change().fillna(0.0)
    positions = pd.Series([0.0, 1.0, 1.0], index=equity.index)

    from backtesting.stats import compute_performance_stats

    computed = compute_performance_stats(equity_curve=equity, returns=returns, trades=[], positions=positions)

    assert computed["exposure"] == pytest.approx(2 / 3)


def test_wiseman_closed_gator_blocks_new_entries() -> None:
    idx = pd.date_range("2024-01-01", periods=80, freq="h", tz="UTC")
    base = np.linspace(100, 140, len(idx))

    open_ = base.copy()
    close = base + 0.2
    high = np.maximum(open_, close) + 0.8
    low = np.minimum(open_, close) - 0.8

    open_[50], close[50], high[50], low[50] = 135.0, 132.0, 136.0, 131.0
    open_[51], close[51], high[51], low[51] = 131.5, 131.0, 133.0, 130.0

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    # Inflate the closed-threshold so the gator is treated as closed throughout.
    signals = WisemanStrategy(gator_width_lookback=10, gator_width_mult=100.0).generate_signals(data)

    assert (signals == 0).all()


def test_detect_williams_fractals_basic_and_tick_offsets() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0] * 10,
            "high": [9.0, 10.0, 11.0, 14.0, 12.0, 10.0, 9.0, 10.0, 11.0, 10.5],
            "low": [8.0, 7.5, 7.0, 7.2, 7.5, 8.5, 9.0, 8.0, 7.0, 7.5],
            "close": [10.0] * 10,
            "volume": [1000] * 10,
        },
        index=idx,
    )

    fractals = detect_williams_fractals(data, tick_size=0.25)

    assert bool(fractals.loc[idx[3], "up_fractal"])
    assert fractals.loc[idx[3], "up_fractal_price"] == 14.25


def test_detect_williams_fractals_rightmost_plateau_tiebreak() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0] * 10,
            "high": [9.0, 10.0, 13.0, 13.0, 13.0, 10.0, 9.0, 10.0, 11.0, 10.5],
            "low": [8.0, 7.5, 6.0, 6.0, 6.0, 8.5, 9.0, 8.0, 7.0, 7.5],
            "close": [10.0] * 10,
            "volume": [1000] * 10,
        },
        index=idx,
    )

    fractals = detect_williams_fractals(data, tick_size=0.5)

    # Adjacent equal highs/lows should select the farthest-right bar in the run.
    assert bool(fractals.loc[idx[4], "up_fractal"])
    assert bool(fractals.loc[idx[4], "down_fractal"])
    assert bool(fractals.loc[idx[2], "up_fractal"]) is False
    assert bool(fractals.loc[idx[3], "up_fractal"]) is False
    assert fractals.loc[idx[4], "up_fractal_price"] == 13.5
    assert fractals.loc[idx[4], "down_fractal_price"] == 5.5


def test_ntd_strategy_enters_from_sleeping_gator_adds_above_teeth_and_exits_via_profit_protection(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=12, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.1, 10.0, 10.1, 10.8, 10.9, 11.1, 11.4, 11.8, 10.2, 10.1],
            "high": [10.2, 10.3, 11.0, 10.4, 10.5, 11.2, 11.1, 11.5, 12.0, 12.2, 10.5, 10.2],
            "low": [9.9, 9.8, 10.0, 9.5, 10.0, 10.7, 10.8, 11.0, 11.2, 11.6, 9.9, 9.8],
            "close": [10.0, 10.1, 10.6, 10.0, 10.2, 11.1, 11.0, 11.4, 11.8, 12.0, 9.95, 10.0],
            "volume": [1000.0] * 12,
        },
        index=idx,
    )

    jaw = pd.Series([10.0] * len(idx), index=idx)
    teeth = pd.Series([10.0] * len(idx), index=idx)
    lips = pd.Series([10.0] * len(idx), index=idx)
    fractals = pd.DataFrame(False, index=idx, columns=["up_fractal", "down_fractal"])
    fractals["up_fractal_price"] = np.nan
    fractals["down_fractal_price"] = np.nan
    fractals.loc[idx[2], "up_fractal"] = True
    fractals.loc[idx[2], "up_fractal_price"] = 11.0
    fractals.loc[idx[3], "down_fractal"] = True
    fractals.loc[idx[3], "down_fractal_price"] = 9.5
    fractals.loc[idx[6], "up_fractal"] = True
    fractals.loc[idx[6], "up_fractal_price"] = 12.0

    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: pd.Series(0.0, index=idx, dtype="float64"))

    strategy = NTDStrategy(
        gator_width_lookback=2,
        gator_width_mult=1.0,
        require_gator_close_reset=True,
        ao_ac_near_zero_lookback=2,
        ao_ac_near_zero_factor=1.0,
        teeth_profit_protection_enabled=True,
        teeth_profit_protection_min_bars=1,
        teeth_profit_protection_min_unrealized_return=0.01,
        teeth_profit_protection_require_gator_open=False,
        lips_profit_protection_enabled=False,
        zone_profit_protection_enabled=False,
    )
    signals = strategy.generate_signals(data)

    assert strategy.signal_contracts is not None
    assert strategy.signal_fill_prices is not None
    assert strategy.signal_stop_loss_prices is not None
    assert strategy.signal_exit_reason is not None
    assert strategy.signal_contracts.iloc[5] == pytest.approx(1.0)
    assert strategy.signal_fill_prices.iloc[5] == pytest.approx(11.0)
    assert strategy.signal_stop_loss_prices.iloc[5] == pytest.approx(9.5)
    assert strategy.signal_stop_loss_prices.iloc[9] == pytest.approx(9.5)
    assert strategy.signal_contracts.iloc[9] == pytest.approx(2.0)
    assert strategy.signal_fill_prices.iloc[9] == pytest.approx(11.1)
    assert strategy.signal_exit_reason.iloc[10] == "Red Gator PP"
    assert int(signals.iloc[10]) == 0
    assert strategy.signal_contracts.iloc[10] == pytest.approx(0.0)
    assert strategy.signal_stop_loss_prices.iloc[10:].isna().all()


def test_combined_strategy_supports_wiseman_and_ntd_side_by_side() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    wiseman_signals = pd.Series([0, 1, 1, 0], index=idx, dtype="int8")
    wiseman_contracts = pd.Series([0.0, 1.0, 1.0, 0.0], index=idx, dtype="float64")
    wiseman_fills = pd.Series([np.nan, 1.0, np.nan, np.nan], index=idx, dtype="float64")
    ntd_signals = pd.Series([0, 0, 1, 1], index=idx, dtype="int8")
    ntd_contracts = pd.Series([0.0, 0.0, 2.0, 2.0], index=idx, dtype="float64")
    ntd_fills = pd.Series([np.nan, np.nan, 1.0, np.nan], index=idx, dtype="float64")

    combined = CombinedStrategy(
        [
            _SignalContractsExitReasonStrategy(wiseman_signals, wiseman_contracts, fill_prices=wiseman_fills),
            _SignalContractsExitReasonStrategy(ntd_signals, ntd_contracts, fill_prices=ntd_fills),
        ]
    )
    signals = combined.generate_signals(
        pd.DataFrame(
            {"open": [1, 1, 1, 1], "high": [1, 1, 1, 1], "low": [1, 1, 1, 1], "close": [1, 1, 1, 1]},
            index=idx,
        )
    )

    assert signals.tolist() == [0, 1, 1, 1]
    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 1.0, 3.0, 3.0]


def test_combined_strategy_blocks_contract_reductions_without_pp_or_opposite_wiseman() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {"open": [1, 1, 1, 1], "high": [1, 1, 1, 1], "low": [1, 1, 1, 1], "close": [1, 1, 1, 1]},
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 1.0, 0.0], index=idx, dtype="float64"),
        fill_prices=pd.Series([np.nan, 1.0, np.nan, np.nan], index=idx, dtype="float64"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 2.0, 2.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 1, 1], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, np.nan, 1.0, np.nan], index=idx, dtype="float64"),
    )
    combined = CombinedStrategy([wiseman, ntd])

    signals = combined.generate_signals(data)

    assert signals.tolist() == [0, 1, 1, 1]
    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 1.0, 3.0, 3.0]
    assert combined.signal_exit_reason is not None
    assert combined.signal_exit_reason.tolist() == ["", "", "", ""]

    result = BacktestEngine(BacktestConfig(close_open_position_on_last_bar=False)).run(data, combined)

    assert result.trades == []


def test_combined_strategy_allows_pp_exit_reductions() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {"open": [1, 1, 1, 1], "high": [1, 1, 1, 1], "low": [1, 1, 1, 1], "close": [1, 1, 1, 1]},
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 1.0, 0.0], index=idx, dtype="float64"),
        exit_reasons=pd.Series(["", "", "", "Red Gator PP"], index=idx, dtype="object"),
        fill_prices=pd.Series([np.nan, 1.0, np.nan, 1.0], index=idx, dtype="float64"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 2.0, 2.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 1, 1], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, np.nan, 1.0, np.nan], index=idx, dtype="float64"),
    )
    combined = CombinedStrategy([wiseman, ntd])

    signals = combined.generate_signals(data)

    assert signals.tolist() == [0, 1, 1, 0]
    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 1.0, 3.0, 0.0]
    assert combined.signal_exit_reason is not None
    assert combined.signal_exit_reason.iloc[3] == "Red Gator PP"


def test_combined_strategy_ignores_pp_reason_without_same_bar_fill() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {"open": [1, 1, 1, 1], "high": [1, 1, 1, 1], "low": [1, 1, 1, 1], "close": [1, 1, 1, 1]},
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 1.0, 1.0], index=idx, dtype="float64"),
        exit_reasons=pd.Series(["", "", "", "Red Gator PP"], index=idx, dtype="object"),
        fill_prices=pd.Series([np.nan, 1.0, np.nan, np.nan], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 1, 0, 0], index=idx, dtype="int8"),
        first_wiseman_fill_prices=pd.Series([np.nan, 1.0, np.nan, np.nan], index=idx, dtype="float64"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 2.0, 2.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 1, 1], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, np.nan, 1.0, np.nan], index=idx, dtype="float64"),
    )
    combined = CombinedStrategy([wiseman, ntd])

    signals = combined.generate_signals(data)

    assert signals.tolist() == [0, 1, 1, 1]
    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 1.0, 3.0, 3.0]
    assert combined.signal_exit_reason is not None
    assert combined.signal_exit_reason.iloc[3] == ""


def test_combined_strategy_allows_ntd_pp_exit_in_generic_combiner() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {"open": [1, 1, 1, 1], "high": [1, 1, 1, 1], "low": [1, 1, 1, 1], "close": [1, 1, 1, 1]},
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 1.0, 1.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 1, 0, 0], index=idx, dtype="int8"),
        first_wiseman_fill_prices=pd.Series([np.nan, 1.0, np.nan, np.nan], index=idx, dtype="float64"),
        fill_prices=pd.Series([np.nan, 1.0, np.nan, np.nan], index=idx, dtype="float64"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 1, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 2.0, 0.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 1, 0], index=idx, dtype="int8"),
        exit_reasons=pd.Series(["", "", "", "Red Gator PP"], index=idx, dtype="object"),
        fill_prices=pd.Series([np.nan, np.nan, 1.0, 1.0], index=idx, dtype="float64"),
    )
    combined = CombinedStrategy([wiseman, ntd])

    signals = combined.generate_signals(data)

    assert signals.tolist() == [0, 1, 1, 0]
    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 1.0, 3.0, 0.0]
    assert combined.signal_exit_reason is not None
    assert combined.signal_exit_reason.iloc[3] == "Red Gator PP"


def test_combined_strategy_allows_ntd_pp_exit_in_wiseman_ntd_mode() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {"open": [1, 1, 1, 1], "high": [1, 1, 1, 1], "low": [1, 1, 1, 1], "close": [1, 1, 1, 1]},
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 1.0, 1.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 1, 0, 0], index=idx, dtype="int8"),
        first_wiseman_fill_prices=pd.Series([np.nan, 1.0, np.nan, np.nan], index=idx, dtype="float64"),
        fill_prices=pd.Series([np.nan, 1.0, np.nan, np.nan], index=idx, dtype="float64"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 1, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 2.0, 0.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 1, 0], index=idx, dtype="int8"),
        exit_reasons=pd.Series(["", "", "", "Red Gator PP"], index=idx, dtype="object"),
        fill_prices=pd.Series([np.nan, np.nan, 1.0, 1.0], index=idx, dtype="float64"),
    )
    combined = CombinedStrategy([wiseman, ntd])

    signals = combined.generate_signals(data)

    assert signals.tolist() == [0, 1, 1, 0]
    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 1.0, 3.0, 0.0]
    assert combined.signal_exit_reason is not None
    assert combined.signal_exit_reason.iloc[3] == "Red Gator PP"


def test_combined_strategy_flattens_on_ntd_green_lips_pp_exit_in_wiseman_ntd_mode() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {"open": [1, 1, 1, 1], "high": [1, 1, 1, 1], "low": [1, 1, 1, 1], "close": [1, 1, 1, 1]},
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 1.0, 1.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 1, 0, 0], index=idx, dtype="int8"),
        first_wiseman_fill_prices=pd.Series([np.nan, 1.0, np.nan, np.nan], index=idx, dtype="float64"),
        fill_prices=pd.Series([np.nan, 1.0, np.nan, np.nan], index=idx, dtype="float64"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 1, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 2.0, 0.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 1, 0], index=idx, dtype="int8"),
        exit_reasons=pd.Series(["", "", "", "Green Gator Lips PP"], index=idx, dtype="object"),
        fill_prices=pd.Series([np.nan, np.nan, 1.0, 1.0], index=idx, dtype="float64"),
    )
    combined = CombinedStrategy([wiseman, ntd])

    signals = combined.generate_signals(data)

    assert signals.tolist() == [0, 1, 1, 0]
    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 1.0, 3.0, 0.0]
    assert combined.signal_exit_reason is not None
    assert combined.signal_exit_reason.iloc[3] == "Green Gator Lips PP"


def test_combined_strategy_flattens_on_component_flat_exit_reason_even_when_reason_is_non_pp() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {"open": [1, 1, 1, 1], "high": [1, 1, 1, 1], "low": [1, 1, 1, 1], "close": [1, 1, 1, 1]},
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 1.0, 1.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 1, 0, 0], index=idx, dtype="int8"),
        first_wiseman_fill_prices=pd.Series([np.nan, 1.0, np.nan, np.nan], index=idx, dtype="float64"),
        fill_prices=pd.Series([np.nan, 1.0, np.nan, np.nan], index=idx, dtype="float64"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 1, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 2.0, 0.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 1, 0], index=idx, dtype="int8"),
        exit_reasons=pd.Series(["", "", "", "State Reset"], index=idx, dtype="object"),
        fill_prices=pd.Series([np.nan, np.nan, 1.0, 1.0], index=idx, dtype="float64"),
    )
    combined = CombinedStrategy([wiseman, ntd])

    signals = combined.generate_signals(data)

    assert signals.tolist() == [0, 1, 1, 0]
    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 1.0, 3.0, 0.0]
    assert combined.signal_exit_reason is not None
    assert combined.signal_exit_reason.iloc[3] == "State Reset"


def test_combined_strategy_wiseman_pp_exit_flattens_unified_position_in_wiseman_ntd_mode() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {"open": [1, 1, 1, 1], "high": [1, 1, 1, 1], "low": [1, 1, 1, 1], "close": [1, 1, 1, 1]},
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 1.0, 0.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 1, 0, 0], index=idx, dtype="int8"),
        first_wiseman_fill_prices=pd.Series([np.nan, 1.0, np.nan, 1.0], index=idx, dtype="float64"),
        exit_reasons=pd.Series(["", "", "", "Red Gator PP"], index=idx, dtype="object"),
        fill_prices=pd.Series([np.nan, 1.0, np.nan, 1.0], index=idx, dtype="float64"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 2.0, 2.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 1, 1], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, np.nan, 1.0, np.nan], index=idx, dtype="float64"),
    )
    combined = CombinedStrategy([wiseman, ntd])

    signals = combined.generate_signals(data)

    assert signals.tolist() == [0, 1, 1, 0]
    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 1.0, 3.0, 0.0]
    assert combined.signal_exit_reason is not None
    assert combined.signal_exit_reason.iloc[3] == "Red Gator PP"


def test_combined_strategy_opposite_wiseman_signal_begins_new_trade_without_reversal_marker() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {"open": [1, 1, 1, 1], "high": [1, 1, 1, 1], "low": [1, 1, 1, 1], "close": [1, 1, 1, 1]},
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, -1], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 1.0, -1.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 1, -1, 0], index=idx, dtype="int8"),
        first_wiseman_fill_prices=pd.Series([np.nan, 1.0, np.nan, 1.0], index=idx, dtype="float64"),
        fill_prices=pd.Series([np.nan, 1.0, np.nan, 1.0], index=idx, dtype="float64"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 2.0, 2.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 1, 1], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, np.nan, 1.0, np.nan], index=idx, dtype="float64"),
    )
    combined = CombinedStrategy([wiseman, ntd])

    signals = combined.generate_signals(data)

    assert signals.tolist() == [0, 1, 1, -1]
    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 1.0, 3.0, -1.0]
    assert combined.signal_exit_reason is not None
    assert combined.signal_exit_reason.iloc[3] == ""


def test_combined_strategy_opposite_wiseman_fill_flips_position_and_closes_prior_side() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0, 103.0],
            "high": [100.0, 100.0, 101.0, 103.0],
            "low": [100.0, 100.0, 101.0, 103.0],
            "close": [100.0, 100.0, 101.0, 103.0],
        },
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, -1], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 1.0, -1.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 1, -1, 0], index=idx, dtype="int8"),
        first_wiseman_fill_prices=pd.Series([np.nan, 100.0, np.nan, 103.0], index=idx, dtype="float64"),
        fill_prices=pd.Series([np.nan, 100.0, np.nan, 103.0], index=idx, dtype="float64"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 2.0, 2.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 1, 1], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, np.nan, 101.0, np.nan], index=idx, dtype="float64"),
    )
    combined = CombinedStrategy([wiseman, ntd])

    signals = combined.generate_signals(data)

    assert signals.tolist() == [0, 1, 1, -1]
    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 1.0, 3.0, -1.0]
    assert combined.signal_exit_reason is not None
    assert combined.signal_exit_reason.iloc[3] == ""

    result = BacktestEngine(
        BacktestConfig(
            fee_rate=0.0,
            slippage_rate=0.0,
            spread_rate=0.0,
            trade_size_mode="units",
            trade_size_value=1.0,
            close_open_position_on_last_bar=False,
        )
    ).run(data, combined)

    trades = result.trades_dataframe()
    assert len(trades) == 1
    assert trades.iloc[0]["entry_signal"] == "Bullish 1W"
    assert trades.iloc[0]["exit_signal"] == "Signal Intent Flip to Bearish 1W"
    assert len(result.execution_events) == 4


def test_combined_strategy_opposite_wiseman_fill_uses_1w_fill_price_over_same_bar_ntd_fill() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0, 103.0],
            "high": [100.0, 100.0, 101.0, 104.0],
            "low": [100.0, 100.0, 101.0, 103.0],
            "close": [100.0, 100.0, 101.0, 103.0],
        },
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, -1], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 1.0, -1.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 1, -1, 0], index=idx, dtype="int8"),
        first_wiseman_fill_prices=pd.Series([np.nan, 100.0, np.nan, 103.0], index=idx, dtype="float64"),
        fill_prices=pd.Series([np.nan, 100.0, np.nan, 103.0], index=idx, dtype="float64"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 2.0, 3.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 1, 1], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, np.nan, 101.0, 104.0], index=idx, dtype="float64"),
    )
    combined = CombinedStrategy([wiseman, ntd])

    combined.generate_signals(data)

    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 1.0, 3.0, -1.0]
    assert combined.signal_fill_prices is not None
    assert combined.signal_fill_prices.iloc[3] == pytest.approx(103.0)
    assert combined.signal_exit_reason is not None
    assert combined.signal_exit_reason.iloc[3] == ""

    result = BacktestEngine(
        BacktestConfig(
            fee_rate=0.0,
            slippage_rate=0.0,
            spread_rate=0.0,
            trade_size_mode="units",
            trade_size_value=1.0,
            close_open_position_on_last_bar=False,
        )
    ).run(data, combined)

    assert result.execution_events[-1].strategy_reason == "Bearish 1W"
    assert result.execution_events[-1].price == pytest.approx(103.0)


def test_combined_strategy_opposite_wiseman_executes_even_when_setup_marker_is_only_on_prior_bar() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0, 103.0],
            "high": [100.0, 100.0, 101.0, 103.0],
            "low": [100.0, 100.0, 101.0, 103.0],
            "close": [100.0, 100.0, 101.0, 103.0],
        },
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 0, -1], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 0.0, -1.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 0, -1, 0], index=idx, dtype="int8"),
        first_wiseman_fill_prices=pd.Series([np.nan, np.nan, np.nan, 103.0], index=idx, dtype="float64"),
        fill_prices=pd.Series([np.nan, np.nan, np.nan, 103.0], index=idx, dtype="float64"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 3.0, 3.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 1, 1], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, np.nan, 101.0, np.nan], index=idx, dtype="float64"),
    )
    combined = CombinedStrategy([wiseman, ntd])

    signals = combined.generate_signals(data)

    assert signals.tolist() == [0, 0, 1, -1]
    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 0.0, 3.0, -1.0]
    assert combined.signal_fill_prices is not None
    assert combined.signal_fill_prices.iloc[3] == pytest.approx(103.0)
    assert combined.signal_exit_reason is not None
    assert combined.signal_exit_reason.iloc[3] == ""


def test_combined_strategy_ignores_wiseman_opposite_when_regime_is_ntd() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [100.0, 101.0, 102.0, 103.0, 104.0],
            "low": [100.0, 101.0, 102.0, 103.0, 104.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
        },
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 0, -1, -1], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 0.0, -1.0, -1.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 0, -1, 0, 0], index=idx, dtype="int8"),
        first_wiseman_fill_prices=pd.Series([np.nan, np.nan, np.nan, 103.0, np.nan], index=idx, dtype="float64"),
        fill_prices=pd.Series([np.nan, np.nan, np.nan, 103.0, np.nan], index=idx, dtype="float64"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 1.0, 1.0, 1.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 1, 1, 1, 1], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, 101.0, np.nan, np.nan, np.nan], index=idx, dtype="float64"),
    )
    combined = CombinedStrategy([wiseman, ntd])

    signals = combined.generate_signals(data)

    assert signals.tolist() == [0, 1, 1, 1, 1]
    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 1.0, 1.0, 1.0, 1.0]
    assert combined.signal_fill_prices is not None
    assert np.isnan(combined.signal_fill_prices.iloc[3])


def test_combined_strategy_allows_prior_bar_valid_wiseman_setup_for_opposite_flip() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {"open": [1, 1, 1, 1], "high": [1, 1, 1, 1], "low": [1, 1, 1, 1], "close": [1, 1, 1, 1]},
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, -1], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 1.0, -1.0], index=idx, dtype="float64"),
        # Opposite setup marker exists on prior bar, but not on the flip bar.
        first_wiseman_setup_side=pd.Series([0, 1, -1, 0], index=idx, dtype="int8"),
        first_wiseman_fill_prices=pd.Series([np.nan, 1.0, np.nan, 1.0], index=idx, dtype="float64"),
        fill_prices=pd.Series([np.nan, 1.0, np.nan, 1.0], index=idx, dtype="float64"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 2.0, 2.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 1, 1], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, np.nan, 1.0, np.nan], index=idx, dtype="float64"),
    )
    combined = CombinedStrategy([wiseman, ntd])

    signals = combined.generate_signals(data)

    assert signals.tolist() == [0, 1, 1, -1]
    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 1.0, 3.0, -1.0]


def test_combined_strategy_does_not_borrow_fractal_labels_from_non_filling_component() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    data = pd.DataFrame(
        {"open": [50.0, 50.0, 50.0], "high": [55.0, 55.0, 55.0], "low": [45.0, 45.0, 45.0], "close": [50.0, 50.0, 50.0]},
        index=idx,
    )
    entry_component = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 0.0], index=idx, dtype="float64"),
        fill_prices=pd.Series([np.nan, 50.0, np.nan], index=idx, dtype="float64"),
    )
    passive_fractal_component = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 0.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 1, 1], index=idx, dtype="int8"),
    )

    combined = CombinedStrategy([entry_component, passive_fractal_component])
    result = BacktestEngine(
        BacktestConfig(
            fee_rate=0.0,
            slippage_rate=0.0,
            spread_rate=0.0,
            trade_size_mode="units",
            trade_size_value=1.0,
        )
    ).run(data, combined)

    assert combined.signal_fractal_position_side is not None
    assert int(combined.signal_fractal_position_side.iloc[1]) == 0
    assert len(result.execution_events) >= 1
    assert result.execution_events[0].strategy_reason == "Bullish 1W"


def test_combined_strategy_uses_only_one_initial_entry_then_ntd_adds() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {"open": [100.0, 100.0, 101.0, 102.0], "high": [100.0, 100.0, 101.0, 102.0], "low": [100.0, 100.0, 101.0, 102.0], "close": [100.0, 100.0, 101.0, 102.0]},
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 4.0, 4.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 1, 0, 0], index=idx, dtype="int8"),
        first_wiseman_fill_prices=pd.Series([np.nan, 100.0, np.nan, np.nan], index=idx, dtype="float64"),
        second_wiseman_fill_side=pd.Series([0, 0, 1, 0], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, 100.0, 101.0, np.nan], index=idx, dtype="float64"),
        stop_loss_prices=pd.Series([np.nan, 99.0, 99.5, 100.0], index=idx, dtype="float64"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 1.0, 2.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 1, 1], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, np.nan, 101.0, 102.0], index=idx, dtype="float64"),
        stop_loss_prices=pd.Series([np.nan, np.nan, 99.75, 100.5], index=idx, dtype="float64"),
    )

    combined = CombinedStrategy([wiseman, ntd])
    signals = combined.generate_signals(data)

    assert signals.tolist() == [0, 1, 1, 1]
    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 1.0, 5.0, 6.0]
    assert combined.signal_fill_prices is not None
    assert combined.signal_fill_prices.tolist()[1:] == pytest.approx([100.0, 101.0, 102.0])
    assert combined.signal_second_wiseman_fill_side is not None
    assert combined.signal_second_wiseman_fill_side.iloc[2] == 1
    assert combined.signal_fractal_position_side is not None
    assert combined.signal_fractal_position_side.iloc[2] == 1
    assert combined.signal_stop_loss_prices is not None
    assert combined.signal_stop_loss_prices.iloc[3] == pytest.approx(100.0)


def test_combined_strategy_applies_ntd_add_on_fractals_after_wiseman_initial_even_without_ntd_initial(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=7, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [999.0, 1000.0, 1002.0, 1005.0, 1012.0, 1015.0, 1022.0],
            "high": [999.0, 1000.0, 1002.0, 1020.0, 1015.0, 1022.0, 1025.0],
            "low": [998.0, 999.0, 1001.0, 1004.0, 1010.0, 1014.0, 1020.0],
            "close": [999.0, 1000.0, 1002.0, 1008.0, 1013.0, 1018.0, 1023.0],
        },
        index=idx,
    )

    fractals = pd.DataFrame(False, index=idx, columns=["up_fractal", "down_fractal"])
    fractals["up_fractal_price"] = np.nan
    fractals["down_fractal_price"] = np.nan
    fractals.iloc[1, fractals.columns.get_loc("up_fractal")] = True
    fractals.iloc[3, fractals.columns.get_loc("up_fractal")] = True

    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)
    monkeypatch.setattr(
        strategy_module,
        "_alligator_lines",
        lambda _data: (
            pd.Series(900.0, index=_data.index),
            pd.Series(900.0, index=_data.index),
            pd.Series(900.0, index=_data.index),
        ),
    )

    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, 1, 1, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 1, 0, 0, 0, 0, 0], index=idx, dtype="int8"),
        first_wiseman_fill_prices=pd.Series([np.nan, 1000.0, np.nan, np.nan, np.nan, np.nan, np.nan], index=idx, dtype="float64"),
        fill_prices=pd.Series([np.nan, 1000.0, np.nan, np.nan, np.nan, np.nan, np.nan], index=idx, dtype="float64"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 0, 0, 0, 0, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 0, 0, 0, 0, 0], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], index=idx, dtype="float64"),
    )

    combined = CombinedStrategy([wiseman, ntd])
    combined.generate_signals(data)

    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0]
    assert combined.signal_fill_prices is not None
    assert combined.signal_fill_prices.iloc[4] == pytest.approx(1000.0)
    assert combined.signal_fill_prices.iloc[6] == pytest.approx(1020.0)
    assert combined.signal_fractal_position_side is not None
    assert combined.signal_fractal_position_side.iloc[4] == 1
    assert combined.signal_fractal_position_side.iloc[6] == 1


def test_combined_strategy_does_not_double_count_ntd_add_on_when_contract_and_synthetic_fill_coincide(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [999.0, 1000.0, 1000.0, 1004.0, 1003.0],
            "high": [999.0, 1000.0, 1000.0, 1006.0, 1005.0],
            "low": [998.0, 999.0, 999.0, 1003.0, 1002.0],
            "close": [999.0, 1000.0, 1000.0, 1005.0, 1004.0],
        },
        index=idx,
    )

    fractals = pd.DataFrame(False, index=idx, columns=["up_fractal", "down_fractal"])
    fractals["up_fractal_price"] = np.nan
    fractals["down_fractal_price"] = np.nan
    fractals.iloc[1, fractals.columns.get_loc("up_fractal")] = True

    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)
    monkeypatch.setattr(
        strategy_module,
        "_alligator_lines",
        lambda _data: (
            pd.Series(900.0, index=_data.index),
            pd.Series(900.0, index=_data.index),
            pd.Series(900.0, index=_data.index),
        ),
    )

    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, 1, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 1.0, 1.0, 0.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 1, 0, 0, 0], index=idx, dtype="int8"),
        first_wiseman_fill_prices=pd.Series([np.nan, 1000.0, np.nan, np.nan, 1003.0], index=idx, dtype="float64"),
        fill_prices=pd.Series([np.nan, 1000.0, np.nan, np.nan, 1003.0], index=idx, dtype="float64"),
        exit_reasons=pd.Series(["", "", "", "", "Red Gator PP"], index=idx, dtype="object"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 0, 1, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 0.0, 1.0, 0.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 0, 1, 0], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, np.nan, np.nan, 1004.0, 1003.0], index=idx, dtype="float64"),
        exit_reasons=pd.Series(["", "", "", "", "Red Gator PP"], index=idx, dtype="object"),
    )

    combined = CombinedStrategy([wiseman, ntd])
    combined.generate_signals(data)

    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 1.0, 1.0, 2.0, 0.0]
    assert combined.signal_fill_prices is not None
    assert combined.signal_fill_prices.iloc[3] == pytest.approx(1004.0)


def test_combined_strategy_clamps_single_ntd_add_on_per_bar_even_if_component_contracts_jump() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0, 102.0, 103.0, 101.0],
            "high": [100.0, 100.0, 101.0, 102.0, 103.0, 101.0],
            "low": [100.0, 100.0, 101.0, 102.0, 103.0, 101.0],
            "close": [100.0, 100.0, 101.0, 102.0, 103.0, 101.0],
        },
        index=idx,
    )

    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, 1, 1, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 1.0, 1.0, 1.0, 0.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 1, 0, 0, 0, 0], index=idx, dtype="int8"),
        first_wiseman_fill_prices=pd.Series([np.nan, 100.0, np.nan, np.nan, np.nan, 101.0], index=idx, dtype="float64"),
        fill_prices=pd.Series([np.nan, 100.0, np.nan, np.nan, np.nan, 101.0], index=idx, dtype="float64"),
        exit_reasons=pd.Series(["", "", "", "", "", "Red Gator PP"], index=idx, dtype="object"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 1, 1, 1, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 1.0, 3.0, 3.0, 0.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 1, 1, 1, 0], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, np.nan, 101.0, 102.0, np.nan, 101.0], index=idx, dtype="float64"),
        exit_reasons=pd.Series(["", "", "", "", "", "Red Gator PP"], index=idx, dtype="object"),
    )

    combined = CombinedStrategy([wiseman, ntd])
    combined.generate_signals(data)

    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 1.0, 2.0, 3.0, 3.0, 0.0]
    assert combined.signal_fractal_position_side is not None
    assert combined.signal_fractal_position_side.iloc[3] == 1


def test_combined_strategy_counts_ntd_initial_fractal_when_wiseman_wins_initial_tiebreak() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    data = pd.DataFrame(
        {"open": [100.0, 100.0, 101.0], "high": [100.0, 100.0, 101.0], "low": [100.0, 100.0, 101.0], "close": [100.0, 100.0, 101.0]},
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 1.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 1, 0], index=idx, dtype="int8"),
        first_wiseman_fill_prices=pd.Series([np.nan, 100.0, np.nan], index=idx, dtype="float64"),
        fill_prices=pd.Series([np.nan, 100.0, np.nan], index=idx, dtype="float64"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 1.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 1, 1], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, 100.0, np.nan], index=idx, dtype="float64"),
    )

    combined = CombinedStrategy([wiseman, ntd])
    combined.generate_signals(data)

    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 2.0, 2.0]
    assert combined.signal_fractal_position_side is not None
    assert combined.signal_fractal_position_side.iloc[1] == 1


def test_combined_strategy_ignores_third_wiseman_fill_markers_when_third_is_disabled() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {"open": [100.0, 100.0, 101.0, 102.0], "high": [100.0, 100.0, 101.0, 102.0], "low": [100.0, 100.0, 101.0, 102.0], "close": [100.0, 100.0, 101.0, 102.0]},
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 2.0, 2.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 1, 0, 0], index=idx, dtype="int8"),
        first_wiseman_fill_prices=pd.Series([np.nan, 100.0, np.nan, np.nan], index=idx, dtype="float64"),
        third_wiseman_fill_side=pd.Series([0, 0, 1, 0], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, 100.0, 101.0, np.nan], index=idx, dtype="float64"),
    )
    wiseman.third_wiseman_contracts = 0
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 0, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 0.0, 0.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 0, 0], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, np.nan, np.nan, np.nan], index=idx, dtype="float64"),
    )

    combined = CombinedStrategy([wiseman, ntd])
    combined.generate_signals(data)

    assert combined.signal_third_wiseman_fill_side is not None
    assert combined.signal_third_wiseman_fill_side.tolist() == [0, 0, 0, 0]


def test_combined_strategy_prefers_fractal_marker_over_phantom_third_wiseman_marker() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {"open": [100.0, 100.0, 101.0, 102.0], "high": [100.0, 100.0, 101.0, 102.0], "low": [100.0, 100.0, 101.0, 102.0], "close": [100.0, 100.0, 101.0, 102.0]},
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 2.0, 2.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 1, 0, 0], index=idx, dtype="int8"),
        first_wiseman_fill_prices=pd.Series([np.nan, 100.0, np.nan, np.nan], index=idx, dtype="float64"),
        third_wiseman_fill_side=pd.Series([0, 0, 1, 0], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, 100.0, 101.0, np.nan], index=idx, dtype="float64"),
    )
    wiseman.third_wiseman_contracts = 0
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 1.0, 2.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 1, 1], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, np.nan, 101.0, 102.0], index=idx, dtype="float64"),
    )

    combined = CombinedStrategy([wiseman, ntd])
    combined.generate_signals(data)

    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert combined.signal_third_wiseman_fill_side is not None
    assert combined.signal_third_wiseman_fill_side.iloc[2] == 0
    assert combined.signal_fractal_position_side is not None
    assert combined.signal_fractal_position_side.iloc[2] == 1


def test_combined_strategy_opposite_ntd_initial_fractal_closes_and_reopens_opposite_trade() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    data = pd.DataFrame(
        {"open": [100.0, 100.0, 101.0, 99.0, 98.0], "high": [100.0, 100.0, 101.0, 99.0, 98.0], "low": [100.0, 100.0, 101.0, 99.0, 98.0], "close": [100.0, 100.0, 101.0, 99.0, 98.0]},
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 1.0, 1.0, 1.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 1, 0, 0, 0], index=idx, dtype="int8"),
        first_wiseman_fill_prices=pd.Series([np.nan, 100.0, np.nan, np.nan, np.nan], index=idx, dtype="float64"),
        fill_prices=pd.Series([np.nan, 100.0, np.nan, np.nan, np.nan], index=idx, dtype="float64"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 1, -1, -1], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 2.0, -2.0, -2.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 1, -1, -1], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, np.nan, 101.0, 99.0, np.nan], index=idx, dtype="float64"),
        stop_loss_prices=pd.Series([np.nan, np.nan, 99.5, 101.5, 101.0], index=idx, dtype="float64"),
    )

    combined = CombinedStrategy([wiseman, ntd])
    signals = combined.generate_signals(data)

    assert signals.tolist() == [0, 1, 1, -1, -1]
    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 1.0, 3.0, -1.0, -1.0]
    assert combined.signal_fill_prices is not None
    assert combined.signal_fill_prices.iloc[3] == pytest.approx(99.0)
    assert combined.signal_exit_reason is not None
    assert combined.signal_exit_reason.iloc[3] == ""
    assert combined.signal_fractal_position_side is not None
    assert combined.signal_fractal_position_side.iloc[3] == -1


def test_combined_strategy_wiseman_reversal_flips_entire_combined_position() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    data = pd.DataFrame(
        {"open": [100.0, 100.0, 101.0, 99.0, 98.0], "high": [100.0, 100.0, 101.0, 99.0, 98.0], "low": [100.0, 100.0, 101.0, 99.0, 98.0], "close": [100.0, 100.0, 101.0, 99.0, 98.0]},
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, -1, -1], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 1.0, -1.0, -1.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 1, 0, -1, 0], index=idx, dtype="int8"),
        first_wiseman_reversal_side=pd.Series([0, 0, 0, -1, 0], index=idx, dtype="int8"),
        first_wiseman_fill_prices=pd.Series([np.nan, 100.0, np.nan, 99.0, np.nan], index=idx, dtype="float64"),
        fill_prices=pd.Series([np.nan, 100.0, np.nan, 99.0, np.nan], index=idx, dtype="float64"),
        stop_loss_prices=pd.Series([np.nan, 99.0, 99.5, 100.5, 100.0], index=idx, dtype="float64"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 1, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 2.0, 2.0, 2.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 1, 1, 1], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, np.nan, 101.0, np.nan, np.nan], index=idx, dtype="float64"),
        stop_loss_prices=pd.Series([np.nan, np.nan, 99.75, 99.5, 99.0], index=idx, dtype="float64"),
    )

    combined = CombinedStrategy([wiseman, ntd])
    signals = combined.generate_signals(data)

    assert signals.tolist() == [0, 1, 1, -1, -1]
    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 1.0, 3.0, -3.0, -3.0]
    assert combined.signal_fill_prices is not None
    assert combined.signal_fill_prices.iloc[3] == pytest.approx(99.0)
    assert combined.signal_first_wiseman_reversal_side is not None
    assert combined.signal_first_wiseman_reversal_side.iloc[3] == -1


def test_combined_strategy_allows_ntd_add_ons_after_1w_reversal_entry(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0, 99.0, 98.0, 97.0],
            "high": [100.0, 100.0, 101.0, 99.0, 98.0, 97.0],
            "low": [100.0, 100.0, 98.0, 99.0, 98.0, 97.0],
            "close": [100.0, 100.0, 100.0, 99.0, 98.0, 97.0],
        },
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, -1, -1, -1], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 1.0, -1.0, -1.0, -1.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 1, 0, -1, 0, 0], index=idx, dtype="int8"),
        first_wiseman_reversal_side=pd.Series([0, 0, 0, -1, 0, 0], index=idx, dtype="int8"),
        first_wiseman_fill_prices=pd.Series([np.nan, 100.0, np.nan, 99.0, np.nan, np.nan], index=idx, dtype="float64"),
        fill_prices=pd.Series([np.nan, 100.0, np.nan, 99.0, np.nan, np.nan], index=idx, dtype="float64"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 0, 0, 0, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=idx, dtype="float64"),
        fill_prices=pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], index=idx, dtype="float64"),
    )

    fractals = pd.DataFrame(
        {
            "up_fractal": [False, False, False, False, False, False],
            "down_fractal": [False, False, True, False, False, False],
        },
        index=idx,
    )
    teeth = pd.Series([100.0, 100.0, 99.0, 99.0, 98.0, 97.0], index=idx, dtype="float64")
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (teeth, teeth, teeth))

    combined = CombinedStrategy([wiseman, ntd])
    signals = combined.generate_signals(data)

    assert signals.tolist() == [0, 1, 1, -1, -1, -1]
    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 1.0, 1.0, -1.0, -1.0, -2.0]
    assert combined.signal_fill_prices is not None
    assert combined.signal_fill_prices.iloc[5] == pytest.approx(98.0)
    assert combined.signal_fractal_position_side is not None
    assert combined.signal_fractal_position_side.iloc[5] == -1


def test_combined_strategy_ignores_stale_pp_from_prior_component_trade() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    data = pd.DataFrame(
        {"open": [100.0, 100.0, 101.0, 99.0, 98.0, 97.0], "high": [100.0, 100.0, 101.0, 99.0, 98.0, 97.0], "low": [100.0, 100.0, 101.0, 99.0, 98.0, 97.0], "close": [100.0, 100.0, 101.0, 99.0, 98.0, 97.0]},
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, -1, -1, -1], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 1.0, -1.0, -1.0, -1.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 1, 0, -1, 0, 0], index=idx, dtype="int8"),
        first_wiseman_reversal_side=pd.Series([0, 0, 0, -1, 0, 0], index=idx, dtype="int8"),
        first_wiseman_fill_prices=pd.Series([np.nan, 100.0, np.nan, 99.0, np.nan, np.nan], index=idx, dtype="float64"),
        fill_prices=pd.Series([np.nan, 100.0, np.nan, 99.0, np.nan, np.nan], index=idx, dtype="float64"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 1, 1, 0, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 2.0, 2.0, 0.0, 0.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 1, 1, 0, 0], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, np.nan, 101.0, np.nan, 98.0, np.nan], index=idx, dtype="float64"),
        exit_reasons=pd.Series(["", "", "", "", "Red Gator PP", ""], index=idx, dtype="object"),
    )

    combined = CombinedStrategy([wiseman, ntd])
    signals = combined.generate_signals(data)

    assert signals.tolist() == [0, 1, 1, -1, -1, -1]
    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 1.0, 3.0, -3.0, -3.0, -3.0]
    assert combined.signal_exit_reason is not None
    assert combined.signal_exit_reason.iloc[4] == ""


def test_combined_strategy_respects_wiseman_reversal_stop_exit() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    data = pd.DataFrame(
        {"open": [100.0, 100.0, 101.0, 98.0, 97.0], "high": [100.0, 100.0, 101.0, 98.0, 97.0], "low": [100.0, 100.0, 101.0, 98.0, 97.0], "close": [100.0, 100.0, 101.0, 98.0, 97.0]},
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, 0, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 1.0, 0.0, 0.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 1, 0, 0, 0], index=idx, dtype="int8"),
        first_wiseman_reversal_side=pd.Series([0, 0, 0, 0, 0], index=idx, dtype="int8"),
        first_wiseman_fill_prices=pd.Series([np.nan, 100.0, np.nan, np.nan, np.nan], index=idx, dtype="float64"),
        fill_prices=pd.Series([np.nan, 100.0, np.nan, 98.0, np.nan], index=idx, dtype="float64"),
        stop_loss_prices=pd.Series([np.nan, 99.0, 99.5, np.nan, np.nan], index=idx, dtype="float64"),
        exit_reasons=pd.Series(["", "", "", "1W Reversal Stop", ""], index=idx, dtype="object"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 1, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 2.0, 2.0, 2.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 1, 1, 1], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, np.nan, 101.0, np.nan, np.nan], index=idx, dtype="float64"),
    )

    combined = CombinedStrategy([wiseman, ntd])
    signals = combined.generate_signals(data)

    assert signals.tolist() == [0, 1, 1, 0, 0]
    assert combined.signal_contracts is not None
    assert combined.signal_contracts.tolist() == [0.0, 1.0, 3.0, 0.0, 0.0]
    assert combined.signal_fill_prices is not None
    assert combined.signal_fill_prices.iloc[3] == pytest.approx(98.0)
    assert combined.signal_exit_reason is not None
    assert combined.signal_exit_reason.iloc[3] == "1W Reversal Stop"



def test_combined_strategy_copies_wiseman_ignored_markers() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {"open": [100.0, 100.0, 100.0, 100.0], "high": [101.0, 102.0, 103.0, 104.0], "low": [99.0, 98.0, 97.0, 96.0], "close": [100.0, 100.0, 100.0, 100.0]},
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 0, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 0.0, 0.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 1, -1, 0], index=idx, dtype="int8"),
        first_wiseman_ignored_reason=pd.Series(["", "ao_divergence_filter", "invalidation_before_trigger", ""], index=idx, dtype="object"),
    )
    ntd = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 0, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 0.0, 0.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 1, 1], index=idx, dtype="int8"),
    )

    combined = CombinedStrategy([wiseman, ntd])
    combined.generate_signals(data)

    assert combined.signal_first_wiseman_setup_side is not None
    assert combined.signal_first_wiseman_setup_side.tolist() == [0, 1, -1, 0]
    assert combined.signal_first_wiseman_ignored_reason is not None
    assert combined.signal_first_wiseman_ignored_reason.tolist() == ["", "ao_divergence_filter", "invalidation_before_trigger", ""]


def test_combined_strategy_generic_path_keeps_wiseman_ignored_markers() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {"open": [100.0, 100.0, 100.0, 100.0], "high": [101.0, 102.0, 103.0, 104.0], "low": [99.0, 98.0, 97.0, 96.0], "close": [100.0, 100.0, 100.0, 100.0]},
        index=idx,
    )
    wiseman = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 0, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 0.0, 0.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 1, -1, 0], index=idx, dtype="int8"),
        first_wiseman_ignored_reason=pd.Series(["", "ao_divergence_filter", "invalidation_before_trigger", ""], index=idx, dtype="object"),
    )
    ntd_a = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 0, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 0.0, 0.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, 1, 1], index=idx, dtype="int8"),
    )
    ntd_b = _SignalContractsExitReasonStrategy(
        pd.Series([0, 0, 0, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 0.0, 0.0, 0.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 0, -1, -1], index=idx, dtype="int8"),
    )

    combined = CombinedStrategy([wiseman, ntd_a, ntd_b])
    combined.generate_signals(data)

    assert combined.signal_first_wiseman_setup_side is not None
    assert combined.signal_first_wiseman_setup_side.tolist() == [0, 1, -1, 0]
    assert combined.signal_first_wiseman_ignored_reason is not None
    assert combined.signal_first_wiseman_ignored_reason.tolist() == ["", "ao_divergence_filter", "invalidation_before_trigger", ""]


def test_ntd_reversal_stop_remains_at_opposite_fractal_price_until_new_fractal_confirms(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=9, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.1, 10.0, 10.1, 10.8, 10.7, 9.8, 9.7],
            "high": [10.2, 10.3, 11.0, 10.4, 10.5, 11.2, 10.9, 9.9, 9.8],
            "low": [9.9, 9.8, 10.0, 9.5, 10.0, 10.7, 10.4, 9.4, 9.2],
            "close": [10.0, 10.1, 10.6, 10.0, 10.2, 11.1, 10.6, 9.6, 9.4],
            "volume": [1000.0] * 9,
        },
        index=idx,
    )

    jaw = pd.Series([10.0] * len(idx), index=idx)
    teeth = pd.Series([10.0] * len(idx), index=idx)
    lips = pd.Series([10.0] * len(idx), index=idx)
    fractals = pd.DataFrame(False, index=idx, columns=["up_fractal", "down_fractal"])
    fractals["up_fractal_price"] = np.nan
    fractals["down_fractal_price"] = np.nan
    fractals.loc[idx[2], "up_fractal"] = True
    fractals.loc[idx[2], "up_fractal_price"] = 11.0
    fractals.loc[idx[3], "down_fractal"] = True
    fractals.loc[idx[3], "down_fractal_price"] = 9.5

    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: pd.Series(0.0, index=idx, dtype="float64"))

    strategy = NTDStrategy(
        gator_width_lookback=2,
        gator_width_mult=1.0,
        require_gator_close_reset=True,
        ao_ac_near_zero_lookback=2,
        ao_ac_near_zero_factor=1.0,
        teeth_profit_protection_enabled=False,
        lips_profit_protection_enabled=False,
        zone_profit_protection_enabled=False,
    )
    signals = strategy.generate_signals(data)

    assert strategy.signal_fill_prices is not None
    assert strategy.signal_stop_loss_prices is not None
    assert strategy.signal_exit_reason is not None
    assert strategy.signal_fill_prices.iloc[5] == pytest.approx(11.0)
    assert strategy.signal_stop_loss_prices.iloc[5] == pytest.approx(9.5)
    assert strategy.signal_stop_loss_prices.iloc[6] == pytest.approx(9.5)
    assert int(signals.iloc[7]) == -1
    assert strategy.signal_fill_prices.iloc[7] == pytest.approx(9.5)
    assert strategy.signal_stop_loss_prices.iloc[7] == pytest.approx(11.0)
    assert strategy.signal_stop_loss_prices.iloc[8] == pytest.approx(11.0)
    assert strategy.signal_exit_reason.eq("").all()


def test_ntd_reversal_stop_advances_to_most_recent_opposite_fractal_above_gator(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=12, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.1, 10.0, 10.1, 10.8, 10.7, 9.8, 9.7, 9.6, 9.5, 9.4],
            "high": [10.2, 10.3, 11.0, 10.4, 10.5, 11.2, 10.9, 9.9, 10.6, 9.8, 9.7, 9.6],
            "low": [9.9, 9.8, 10.0, 9.5, 10.0, 10.7, 10.4, 9.4, 9.3, 9.2, 9.1, 9.0],
            "close": [10.0, 10.1, 10.6, 10.0, 10.2, 11.1, 10.6, 9.6, 9.5, 9.4, 9.3, 9.2],
            "volume": [1000.0] * 12,
        },
        index=idx,
    )

    jaw = pd.Series([10.0] * len(idx), index=idx)
    teeth = pd.Series([10.0] * len(idx), index=idx)
    lips = pd.Series([10.0] * len(idx), index=idx)
    fractals = pd.DataFrame(False, index=idx, columns=["up_fractal", "down_fractal"])
    fractals["up_fractal_price"] = np.nan
    fractals["down_fractal_price"] = np.nan
    fractals.loc[idx[2], "up_fractal"] = True
    fractals.loc[idx[2], "up_fractal_price"] = 11.0
    fractals.loc[idx[3], "down_fractal"] = True
    fractals.loc[idx[3], "down_fractal_price"] = 9.5
    fractals.loc[idx[8], "up_fractal"] = True
    fractals.loc[idx[8], "up_fractal_price"] = 10.6

    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: pd.Series(0.0, index=idx, dtype="float64"))

    strategy = NTDStrategy(
        gator_width_lookback=2,
        gator_width_mult=1.0,
        require_gator_close_reset=True,
        ao_ac_near_zero_lookback=2,
        ao_ac_near_zero_factor=1.0,
        teeth_profit_protection_enabled=False,
        lips_profit_protection_enabled=False,
        zone_profit_protection_enabled=False,
    )
    signals = strategy.generate_signals(data)

    assert strategy.signal_stop_loss_prices is not None
    assert int(signals.iloc[7]) == -1
    assert strategy.signal_stop_loss_prices.iloc[7] == pytest.approx(11.0)
    assert strategy.signal_stop_loss_prices.iloc[9] == pytest.approx(11.0)
    assert strategy.signal_stop_loss_prices.iloc[10] == pytest.approx(10.6)
    assert strategy.signal_stop_loss_prices.iloc[11] == pytest.approx(10.6)


def test_ntd_short_stop_and_reverse_fills_at_fractal_high_not_breaker_bar_high(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [1000.0] * 10,
            "high": [1001.0, 1002.0, 999.0, 1000.0, 996.0, 999.0, 999.0, 999.0, 1010.0, 1001.0],
            "low": [999.0, 998.0, 995.0, 998.0, 994.0, 996.0, 996.0, 996.0, 994.0, 999.0],
            "close": [1000.0, 1000.0, 997.0, 999.0, 995.0, 998.0, 998.0, 998.0, 1008.0, 1000.0],
            "volume": [1000.0] * 10,
        },
        index=idx,
    )

    jaw = pd.Series([999.0] * len(idx), index=idx)
    teeth = pd.Series([999.0] * len(idx), index=idx)
    lips = pd.Series([999.0] * len(idx), index=idx)
    fractals = pd.DataFrame(False, index=idx, columns=["up_fractal", "down_fractal"])
    fractals["up_fractal_price"] = np.nan
    fractals["down_fractal_price"] = np.nan
    fractals.loc[idx[2], "down_fractal"] = True  # short trigger level = low(995)
    fractals.loc[idx[2], "down_fractal_price"] = 995.0
    fractals.loc[idx[3], "up_fractal"] = True  # stop level = high(1000)
    fractals.loc[idx[3], "up_fractal_price"] = 1000.0

    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: pd.Series(0.0, index=idx, dtype="float64"))

    strategy = NTDStrategy(
        gator_width_lookback=2,
        gator_width_mult=1.0,
        require_gator_close_reset=True,
        ao_ac_near_zero_lookback=2,
        ao_ac_near_zero_factor=1.0,
        teeth_profit_protection_enabled=False,
        lips_profit_protection_enabled=False,
        zone_profit_protection_enabled=False,
    )
    signals = strategy.generate_signals(data)

    assert strategy.signal_fill_prices is not None
    assert int(signals.iloc[8]) == 1
    assert strategy.signal_fill_prices.iloc[8] == pytest.approx(1000.0)


def test_ntd_long_stop_and_reverse_fills_at_fractal_low_not_breaker_bar_low(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [1000.0] * 10,
            "high": [1001.0, 1002.0, 1005.0, 1002.0, 1006.0, 1004.0, 1004.0, 1004.0, 1006.0, 1001.0],
            "low": [999.0, 998.0, 1001.0, 1000.0, 1004.0, 1001.0, 1001.0, 1001.0, 990.0, 999.0],
            "close": [1000.0, 1000.0, 1003.0, 1001.0, 1005.0, 1003.0, 1003.0, 1003.0, 992.0, 1000.0],
            "volume": [1000.0] * 10,
        },
        index=idx,
    )

    jaw = pd.Series([1001.0] * len(idx), index=idx)
    teeth = pd.Series([1001.0] * len(idx), index=idx)
    lips = pd.Series([1001.0] * len(idx), index=idx)
    fractals = pd.DataFrame(False, index=idx, columns=["up_fractal", "down_fractal"])
    fractals["up_fractal_price"] = np.nan
    fractals["down_fractal_price"] = np.nan
    fractals.loc[idx[2], "up_fractal"] = True  # long trigger level = high(1005)
    fractals.loc[idx[2], "up_fractal_price"] = 1005.0
    fractals.loc[idx[3], "down_fractal"] = True  # stop level = low(1000)
    fractals.loc[idx[3], "down_fractal_price"] = 1000.0

    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: pd.Series(0.0, index=idx, dtype="float64"))

    strategy = NTDStrategy(
        gator_width_lookback=2,
        gator_width_mult=1.0,
        require_gator_close_reset=True,
        ao_ac_near_zero_lookback=2,
        ao_ac_near_zero_factor=1.0,
        teeth_profit_protection_enabled=False,
        lips_profit_protection_enabled=False,
        zone_profit_protection_enabled=False,
    )
    signals = strategy.generate_signals(data)

    assert strategy.signal_fill_prices is not None
    assert int(signals.iloc[8]) == -1
    assert strategy.signal_fill_prices.iloc[8] == pytest.approx(1000.0)


def test_ntd_initial_entry_requires_sleeping_gator_and_near_zero_ao_ac(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=9, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.1, 10.0, 10.1, 10.8, 10.9, 10.8, 10.7],
            "high": [10.2, 10.3, 11.0, 10.4, 10.5, 11.2, 11.0, 10.9, 10.8],
            "low": [9.9, 9.8, 10.0, 9.5, 10.0, 10.7, 10.8, 10.7, 10.6],
            "close": [10.0, 10.1, 10.6, 10.0, 10.2, 11.1, 10.9, 10.8, 10.7],
            "volume": [1000.0] * 9,
        },
        index=idx,
    )

    jaw = pd.Series([10.0] * len(idx), index=idx)
    teeth = pd.Series([10.0] * len(idx), index=idx)
    lips = pd.Series([10.0] * len(idx), index=idx)
    fractals = pd.DataFrame(False, index=idx, columns=["up_fractal", "down_fractal"])
    fractals["up_fractal_price"] = np.nan
    fractals["down_fractal_price"] = np.nan
    fractals.loc[idx[2], "up_fractal"] = True
    fractals.loc[idx[2], "up_fractal_price"] = 11.0
    fractals.loc[idx[3], "down_fractal"] = True
    fractals.loc[idx[3], "down_fractal_price"] = 9.5

    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)
    quiet_ao = pd.Series(0.0, index=idx, dtype="float64")
    noisy_ao = pd.Series([0.0, 0.0, 0.6, 0.5, 0.55, 0.5, 0.45, 0.4, 0.35], index=idx, dtype="float64")

    quiet = NTDStrategy(
        gator_width_lookback=2,
        gator_width_mult=0.0001,
        require_gator_close_reset=True,
        ao_ac_near_zero_lookback=2,
        ao_ac_near_zero_factor=1.0,
        teeth_profit_protection_enabled=False,
        lips_profit_protection_enabled=False,
        zone_profit_protection_enabled=False,
    )
    noisy = NTDStrategy(
        gator_width_lookback=2,
        gator_width_mult=0.0001,
        require_gator_close_reset=True,
        ao_ac_near_zero_lookback=2,
        ao_ac_near_zero_factor=0.25,
        teeth_profit_protection_enabled=False,
        lips_profit_protection_enabled=False,
        zone_profit_protection_enabled=False,
    )

    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: quiet_ao)
    quiet_signals = quiet.generate_signals(data)
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: noisy_ao)
    noisy_signals = noisy.generate_signals(data)

    assert int(quiet_signals.iloc[5]) == 1
    assert quiet.signal_fill_prices is not None
    assert quiet.signal_fill_prices.iloc[5] == pytest.approx(11.0)
    assert noisy_signals.max() == 0


def test_ntd_generate_signals_ignores_removed_pending_entry_path(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.1, 10.1, 10.2, 10.3],
            "high": [10.1, 10.2, 11.0, 10.4, 10.5, 10.6],
            "low": [9.9, 9.8, 10.0, 9.6, 9.7, 9.8],
            "close": [10.0, 10.1, 10.6, 10.0, 10.1, 10.2],
            "volume": [1000.0] * 6,
        },
        index=idx,
    )

    jaw = pd.Series([10.0] * len(idx), index=idx)
    teeth = pd.Series([10.0] * len(idx), index=idx)
    lips = pd.Series([10.0] * len(idx), index=idx)
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, False, True, False, False, False],
            "down_fractal": [False, False, False, True, False, False],
            "up_fractal_price": [np.nan, np.nan, 11.0, np.nan, np.nan, np.nan],
            "down_fractal_price": [np.nan, np.nan, np.nan, 9.6, np.nan, np.nan],
        },
        index=idx,
    )

    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: pd.Series(0.0, index=idx, dtype="float64"))

    strategy = NTDStrategy(
        gator_width_lookback=2,
        gator_width_mult=1.0,
        require_gator_close_reset=True,
        ao_ac_near_zero_lookback=2,
        ao_ac_near_zero_factor=1.0,
        teeth_profit_protection_enabled=False,
        lips_profit_protection_enabled=False,
        zone_profit_protection_enabled=False,
    )

    signals = strategy.generate_signals(data)

    assert signals.index.equals(idx)
    assert strategy.signal_fill_prices is not None
    assert strategy.signal_stop_loss_prices is not None


def test_ntd_profit_protection_defaults_to_disabled() -> None:
    strategy = NTDStrategy()

    assert strategy.teeth_profit_protection_enabled is False
    assert strategy.teeth_profit_protection_credit_unrealized_before_min_bars is False
    assert strategy.lips_profit_protection_enabled is False
    assert strategy.zone_profit_protection_enabled is False


def test_ntd_teeth_profit_protection_can_credit_unrealized_before_min_bars(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.1, 10.0, 10.1, 10.8, 10.9, 10.95],
            "high": [10.2, 10.3, 11.0, 10.4, 10.5, 11.2, 11.0, 10.95],
            "low": [9.9, 9.8, 10.0, 9.5, 10.0, 10.4, 10.7, 10.8],
            "close": [10.0, 10.1, 10.6, 10.0, 10.2, 11.1, 10.95, 10.85],
            "volume": [1000.0] * 8,
        },
        index=idx,
    )

    jaw = pd.Series([10.0] * len(idx), index=idx)
    teeth = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0, 10.7, 10.9, 10.9], index=idx)
    lips = pd.Series([10.0] * len(idx), index=idx)
    fractals = pd.DataFrame(False, index=idx, columns=["up_fractal", "down_fractal"])
    fractals["up_fractal_price"] = np.nan
    fractals["down_fractal_price"] = np.nan
    fractals.loc[idx[2], "up_fractal"] = True
    fractals.loc[idx[2], "up_fractal_price"] = 11.0

    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: pd.Series(0.0, index=idx, dtype="float64"))

    requires_after_min_bars = NTDStrategy(
        gator_width_lookback=2,
        gator_width_mult=1.0,
        require_gator_close_reset=False,
        ao_ac_near_zero_lookback=2,
        ao_ac_near_zero_factor=1.0,
        teeth_profit_protection_enabled=True,
        teeth_profit_protection_min_bars=1,
        teeth_profit_protection_min_unrealized_return=0.015,
        profit_protection_annualized_volatility_scaler=4.0,
        teeth_profit_protection_credit_unrealized_before_min_bars=False,
        teeth_profit_protection_require_gator_open=False,
    )
    signals_requires_after_min_bars = requires_after_min_bars.generate_signals(data)

    credit_anytime = NTDStrategy(
        gator_width_lookback=2,
        gator_width_mult=1.0,
        require_gator_close_reset=False,
        ao_ac_near_zero_lookback=2,
        ao_ac_near_zero_factor=1.0,
        teeth_profit_protection_enabled=True,
        teeth_profit_protection_min_bars=1,
        teeth_profit_protection_min_unrealized_return=0.015,
        profit_protection_annualized_volatility_scaler=4.0,
        teeth_profit_protection_credit_unrealized_before_min_bars=True,
        teeth_profit_protection_require_gator_open=False,
    )
    signals_credit_anytime = credit_anytime.generate_signals(data)

    assert signals_requires_after_min_bars.iloc[7] == 1
    assert signals_credit_anytime.iloc[7] == 0


def test_ntd_zone_profit_protection_waits_for_break_after_activation_bar(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=11, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.1, 10.2, 10.3, 11.0, 11.05, 11.1, 11.15, 11.2, 11.25, 11.0],
            "high": [10.2, 11.0, 10.4, 10.5, 11.3, 11.25, 11.3, 11.35, 11.4, 11.45, 11.1],
            "low": [9.8, 10.0, 10.0, 10.1, 10.9, 10.8, 10.75, 10.7, 10.65, 10.6, 10.5],
            "close": [10.1, 10.6, 10.3, 10.4, 11.2, 11.15, 11.2, 11.25, 11.3, 11.35, 10.8],
            "volume": [1000.0] * 11,
        },
        index=idx,
    )

    jaw = pd.Series([10.0] * 11, index=idx)
    teeth = pd.Series([10.0] * 11, index=idx)
    lips = pd.Series([10.0] * 11, index=idx)
    zone_green = pd.Series([False, False, False, False, False, True, True, True, True, True, False], index=idx)
    zone_red = pd.Series(False, index=idx)
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, True, False, False, False, False, False, False, False, False, False],
            "down_fractal": [False] * 11,
            "up_fractal_price": [np.nan, 12.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "down_fractal_price": [np.nan] * 11,
        },
        index=idx,
    )

    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)
    monkeypatch.setattr(NTDStrategy, "_williams_zone_bars", lambda self, _data, _ao: (zone_green, zone_red))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: pd.Series(0.0, index=idx, dtype="float64"))

    strategy = NTDStrategy(
        gator_width_lookback=2,
        gator_width_mult=1.0,
        require_gator_close_reset=True,
        ao_ac_near_zero_lookback=2,
        ao_ac_near_zero_factor=1.0,
        teeth_profit_protection_enabled=False,
        lips_profit_protection_enabled=False,
        zone_profit_protection_enabled=True,
        zone_profit_protection_min_unrealized_return=0.0,
    )
    signals = strategy.generate_signals(data)

    assert strategy.signal_fill_prices is not None
    assert strategy.signal_exit_reason is not None
    assert int(signals.iloc[4]) == 1
    assert strategy.signal_exit_reason.iloc[9] == ""
    assert int(signals.iloc[9]) == 1
    assert strategy.signal_exit_reason.iloc[10] == "Williams Zone PP"
    assert int(signals.iloc[10]) == 0


def test_ntd_lips_profit_protection_waits_for_teeth_arming(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.1, 10.2, 10.3, 11.0, 11.2, 11.3, 11.1],
            "high": [10.2, 11.0, 10.4, 10.5, 11.4, 11.6, 11.5, 11.2],
            "low": [9.8, 10.0, 10.0, 10.1, 10.9, 11.0, 10.8, 10.7],
            "close": [10.1, 10.6, 10.3, 10.4, 11.2, 11.4, 10.95, 10.9],
            "volume": [1000.0] * 8,
        },
        index=idx,
    )

    jaw = pd.Series([9.8] * 8, index=idx)
    teeth = pd.Series([10.0, 10.0, 10.0, 10.0, 10.4, 10.6, 11.05, 11.0], index=idx)
    lips = pd.Series([10.2, 10.2, 10.2, 10.2, 10.8, 11.0, 11.1, 11.05], index=idx)
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, True, False, False, False, False, False, False],
            "down_fractal": [False] * 8,
            "up_fractal_price": [np.nan, 11.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "down_fractal_price": [np.nan] * 8,
        },
        index=idx,
    )

    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: pd.Series(0.0, index=idx, dtype="float64"))

    strategy = NTDStrategy(
        gator_width_lookback=2,
        gator_width_mult=10.0,
        require_gator_close_reset=False,
        teeth_profit_protection_enabled=True,
        teeth_profit_protection_min_bars=3,
        teeth_profit_protection_min_unrealized_return=2.0,
        teeth_profit_protection_require_gator_open=False,
        lips_profit_protection_enabled=True,
        lips_profit_protection_volatility_trigger=1.0,
        lips_profit_protection_profit_trigger_mult=100.0,
        lips_profit_protection_min_unrealized_return=2.0,
        lips_profit_protection_arm_on_min_unrealized_return=True,
        zone_profit_protection_enabled=False,
    )
    signals = strategy.generate_signals(data)

    assert strategy.signal_fill_prices is not None
    assert strategy.signal_exit_reason is not None
    assert int(signals.iloc[4]) == 1
    assert int(signals.iloc[6]) == 1
    assert np.isnan(strategy.signal_fill_prices.iloc[6])
    assert strategy.signal_exit_reason.iloc[6] == ""




def test_wiseman_filtered_setup_does_not_emit_one_bar_flat_signal() -> None:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")

    price = 100 + np.cumsum(rng.normal(0.0, 0.8, len(idx)))
    open_ = price + rng.normal(0.0, 0.2, len(idx))
    close = price + rng.normal(0.0, 0.2, len(idx))
    high = np.maximum(open_, close) + np.abs(rng.normal(0.4, 0.2, len(idx)))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.4, 0.2, len(idx)))

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    strategy = WisemanStrategy(
        first_wiseman_divergence_filter_bars=10,
        first_wiseman_reversal_cooldown=10,
        teeth_profit_protection_enabled=True,
        lips_profit_protection_enabled=True,
    )
    signals = strategy.generate_signals(data)

    exit_reason = strategy.signal_exit_reason
    reversal_side = strategy.signal_first_wiseman_reversal_side
    assert exit_reason is not None
    assert reversal_side is not None

    phantom_flat_indices: list[int] = []
    for i in range(1, len(signals)):
        if (
            int(signals.iloc[i - 1]) != 0
            and int(signals.iloc[i]) == 0
            and str(exit_reason.iloc[i]).strip() == ""
            and int(reversal_side.iloc[i]) == 0
        ):
            phantom_flat_indices.append(i)

    assert phantom_flat_indices == []
def test_wiseman_1w_divergence_filter_blocks_setup_without_recent_divergence(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    monkeypatch.setattr(
        strategy_module,
        "_smma",
        lambda series, period: pd.Series({13: 90.0, 8: 100.0, 5: 110.0}[period], index=series.index),
    )

    idx = pd.date_range("2024-01-01", periods=90, freq="h", tz="UTC")
    base = np.linspace(100, 150, len(idx))

    open_ = base + 0.5
    close = base + 0.7
    high = base + 1.0
    low = base - 1.0

    setup_i = 40
    open_[setup_i], close[setup_i], high[setup_i], low[setup_i] = 138.0, 135.0, 140.0, 134.0
    high[setup_i - 1], high[setup_i + 1] = 137.0, 136.5

    low[42] = 133.8

    data = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    baseline_strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
    )
    baseline_signals = baseline_strategy.generate_signals(data)

    filtered_strategy = WisemanStrategy(
        gator_width_lookback=1,
        gator_width_mult=0.0001,
        gator_width_valid_factor=2.0,
        first_wiseman_divergence_filter_bars=1,
    )
    filtered_signals = filtered_strategy.generate_signals(data)

    assert baseline_signals.iloc[44] == -1
    assert filtered_signals.iloc[44] == 0

    ignored = filtered_strategy.signal_first_wiseman_ignored_reason
    assert ignored is not None
    assert ignored.iloc[setup_i] == "ao_divergence_filter"


def test_infer_exit_signal_labels_1w_reverse_flips() -> None:
    engine = BacktestEngine()
    strategy = _SignalBarStrategy(pd.Series(dtype="int8"))
    strategy.signal_first_wiseman_setup_side = pd.Series([0, -1, 0], dtype="int8")
    strategy.signal_first_wiseman_reversal_side = pd.Series([0, 0, 0], dtype="int8")
    strategy.signal_second_wiseman_fill_side = pd.Series([0, 0, 0], dtype="int8")
    strategy.signal_third_wiseman_fill_side = pd.Series([0, 0, 0], dtype="int8")

    label = engine._infer_exit_signal(strategy, signal_index=1, prior_position=1, desired_position=-1)

    assert label == "Signal Intent Flip to Bearish 1W"


def test_engine_uses_strategy_exit_reason_labels() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100, 101, 102, 103, 104, 105],
            "volume": [1_000] * 6,
        },
        index=idx,
    )

    signals = pd.Series([0, 1, 0, -1, 0, 0], index=idx, dtype="int8")
    strategy = _SignalBarStrategy(signals)
    strategy.signal_exit_reason = pd.Series(["", "", "Green Gator PP", "", "Red Gator PP", ""], index=idx, dtype="object")

    result = BacktestEngine().run(data, strategy)
    trades_df = result.trades_dataframe()

    assert len(trades_df) == 2
    assert trades_df["exit_signal"].tolist() == ["Strategy Profit Protection Green Gator", "Strategy Profit Protection Red Gator"]


def test_infer_entry_signal_labels_1w_reversal_entries() -> None:
    engine = BacktestEngine()
    strategy = _SignalBarStrategy(pd.Series(dtype="int8"))
    strategy.signal_first_wiseman_setup_side = pd.Series([0, 0, 0], dtype="int8")
    strategy.signal_first_wiseman_reversal_side = pd.Series([0, 1, 0], dtype="int8")
    strategy.signal_second_wiseman_fill_side = pd.Series([0, 0, 0], dtype="int8")
    strategy.signal_third_wiseman_fill_side = pd.Series([0, 0, 0], dtype="int8")

    label = engine._infer_entry_signal(strategy, signal_index=1, desired_position=1)

    assert label == "Bullish 1W-R"


def test_infer_exit_signal_prefers_explicit_1w_reversal_label() -> None:
    engine = BacktestEngine()
    strategy = _SignalBarStrategy(pd.Series(dtype="int8"))
    strategy.signal_first_wiseman_setup_side = pd.Series([0, -1, 0], dtype="int8")
    strategy.signal_first_wiseman_reversal_side = pd.Series([0, -1, 0], dtype="int8")
    strategy.signal_second_wiseman_fill_side = pd.Series([0, 0, 0], dtype="int8")
    strategy.signal_third_wiseman_fill_side = pd.Series([0, 0, 0], dtype="int8")

    label = engine._infer_exit_signal(strategy, signal_index=1, prior_position=1, desired_position=-1)

    assert label == "Strategy Reversal to Bearish 1W"


def test_infer_entry_signal_uses_descriptive_fallback_labels() -> None:
    engine = BacktestEngine()
    strategy = _SignalBarStrategy(pd.Series(dtype="int8"))

    bullish = engine._infer_entry_signal(strategy, signal_index=0, desired_position=1)
    bearish = engine._infer_entry_signal(strategy, signal_index=0, desired_position=-1)

    assert bullish == "Bullish 1W"
    assert bearish == "Bearish 1W"


def test_infer_entry_and_exit_signal_use_fractal_labels_when_context_is_present() -> None:
    engine = BacktestEngine()
    idx = pd.RangeIndex(3)
    strategy = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 0.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 1, 1], index=idx, dtype="int8"),
    )
    strategy.generate_signals(pd.DataFrame({"open": [1, 1, 1], "high": [1, 1, 1], "low": [1, 1, 1], "close": [1, 1, 1]}, index=idx))

    entry_label = engine._infer_entry_signal(strategy, signal_index=1, desired_position=1)
    exit_label = engine._infer_exit_signal(strategy, signal_index=2, prior_position=1, desired_position=0)

    assert entry_label == "Bullish Fractal"
    assert exit_label == "Signal Intent Flat from Bullish Fractal"


def test_infer_exit_signal_prefers_fractal_label_when_flattening_fractal_trade() -> None:
    engine = BacktestEngine()
    strategy = _SignalBarStrategy(pd.Series(dtype="int8"))
    strategy.signal_first_wiseman_setup_side = pd.Series([0, 1, 1], dtype="int8")
    strategy.signal_first_wiseman_ignored_reason = pd.Series(["", "", ""], dtype="object")
    strategy.signal_first_wiseman_reversal_side = pd.Series([0, 0, 0], dtype="int8")
    strategy.signal_second_wiseman_fill_side = pd.Series([0, 0, 0], dtype="int8")
    strategy.signal_third_wiseman_fill_side = pd.Series([0, 0, 0], dtype="int8")
    strategy.signal_fractal_position_side = pd.Series([0, 1, 1], dtype="int8")

    label = engine._infer_exit_signal(strategy, signal_index=2, prior_position=1, desired_position=0)

    assert label == "Signal Intent Flat from Bullish Fractal"


def test_infer_entry_signal_prefers_fractal_when_1w_setup_has_no_first_fill() -> None:
    engine = BacktestEngine()
    idx = pd.RangeIndex(3)
    strategy = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 0.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 1, 0], index=idx, dtype="int8"),
        first_wiseman_ignored_reason=pd.Series(["", "", ""], index=idx, dtype="object"),
        first_wiseman_fill_prices=pd.Series([np.nan, np.nan, np.nan], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 1, 0], index=idx, dtype="int8"),
    )
    strategy.generate_signals(pd.DataFrame({"open": [1, 1, 1], "high": [1, 1, 1], "low": [1, 1, 1], "close": [1, 1, 1]}, index=idx))

    label = engine._infer_entry_signal(strategy, signal_index=1, desired_position=1)

    assert label == "Bullish Fractal"


def test_infer_entry_signal_uses_add_on_fractal_label_when_present() -> None:
    engine = BacktestEngine()
    strategy = _SignalBarStrategy(pd.Series(dtype="int8"))
    strategy.signal_first_wiseman_setup_side = pd.Series([0, 0], dtype="int8")
    strategy.signal_first_wiseman_ignored_reason = pd.Series(["", ""], dtype="object")
    strategy.signal_first_wiseman_reversal_side = pd.Series([0, 0], dtype="int8")
    strategy.signal_second_wiseman_fill_side = pd.Series([0, 0], dtype="int8")
    strategy.signal_third_wiseman_fill_side = pd.Series([0, 0], dtype="int8")
    strategy.signal_add_on_fractal_fill_side = pd.Series([0, 1], dtype="int8")
    strategy.signal_fractal_position_side = pd.Series([0, 1], dtype="int8")

    label = engine._infer_entry_signal(strategy, signal_index=1, desired_position=1)

    assert label == "Bullish Add-on Fractal"


def test_infer_signal_ignores_filtered_1w_markers_for_labeling() -> None:
    engine = BacktestEngine()
    strategy = _SignalBarStrategy(pd.Series(dtype="int8"))
    strategy.signal_first_wiseman_setup_side = pd.Series([0, 1], dtype="int8")
    strategy.signal_first_wiseman_ignored_reason = pd.Series(["", "gator_closed_canceled"], dtype="object")
    strategy.signal_first_wiseman_reversal_side = pd.Series([0, 0], dtype="int8")
    strategy.signal_second_wiseman_fill_side = pd.Series([0, 0], dtype="int8")
    strategy.signal_third_wiseman_fill_side = pd.Series([0, 0], dtype="int8")
    strategy.signal_fractal_position_side = pd.Series([0, 1], dtype="int8")

    label = engine._infer_entry_signal(strategy, signal_index=1, desired_position=1)

    assert label == "Bullish Fractal"


def test_infer_exit_signal_uses_reduce_reason_when_position_scales_down() -> None:
    engine = BacktestEngine()
    strategy = _SignalBarStrategy(pd.Series(dtype="int8"))
    strategy.signal_first_wiseman_setup_side = pd.Series([0], dtype="int8")
    strategy.signal_first_wiseman_reversal_side = pd.Series([0], dtype="int8")
    strategy.signal_second_wiseman_fill_side = pd.Series([0], dtype="int8")
    strategy.signal_third_wiseman_fill_side = pd.Series([0], dtype="int8")

    label = engine._infer_exit_signal(strategy, signal_index=0, prior_position=1, desired_position=1)

    assert label == "Signal Intent Reduce from Bullish 1W"


def test_engine_records_explicit_reduce_reason_for_partial_fractal_scale_out() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1_000] * 4,
        },
        index=idx,
    )
    strategy = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 2.0, 1.0, 1.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 1, 1, 1], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, 101.0, 102.0, np.nan], index=idx, dtype="float64"),
    )

    result = BacktestEngine(
        BacktestConfig(
            fee_rate=0.0,
            slippage_rate=0.0,
            spread_rate=0.0,
            trade_size_mode="units",
            trade_size_value=1.0,
            close_open_position_on_last_bar=False,
        )
    ).run(data, strategy)

    assert len(result.trades) == 1
    assert result.trades[0].exit_signal == "Signal Intent Reduce from Bullish Fractal"


def test_percent_of_equity_sizing_does_not_rebalance_without_new_signal() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 120.0, 140.0, 160.0],
            "high": [101.0, 121.0, 141.0, 161.0, 181.0],
            "low": [99.0, 99.0, 119.0, 139.0, 159.0],
            "close": [100.0, 120.0, 140.0, 160.0, 180.0],
            "volume": [1_000] * 5,
        },
        index=idx,
    )
    strategy = _SignalBarStrategy(pd.Series([0, 1, 1, 1, 1], index=idx, dtype="int8"))

    result = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            spread_rate=0.0,
            trade_size_mode="percent_of_equity",
            trade_size_value=1.0,
            close_open_position_on_last_bar=False,
        )
    ).run(data, strategy)

    assert result.trades == []
    assert [event.event_type for event in result.execution_events] == ["entry"]
    assert all(event.event_type != "reduce" for event in result.execution_events)


def test_same_side_contract_change_without_fill_does_not_reduce_position() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1_000] * 4,
        },
        index=idx,
    )
    strategy = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, 1], index=idx, dtype="int8"),
        pd.Series([0.0, 2.0, 1.0, 1.0], index=idx, dtype="float64"),
    )

    result = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            spread_rate=0.0,
            trade_size_mode="units",
            trade_size_value=1.0,
            close_open_position_on_last_bar=False,
        )
    ).run(data, strategy)

    assert result.trades == []
    assert [event.event_type for event in result.execution_events] == ["entry"]
    assert all(event.event_type != "reduce" for event in result.execution_events)


def test_same_side_contract_increase_with_fill_does_not_turn_into_volatility_scaled_reduction() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0] * 6,
            "high": [100.0] * 6,
            "low": [100.0] * 6,
            "close": [100.0, 300.0, 100.0, 300.0, 100.0, 100.0],
            "volume": [1_000] * 6,
        },
        index=idx,
    )
    strategy = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, 1, 1, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 1.0, 2.0, 2.0, 0.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 1, 1, 1, 1, 0], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, 100.0, np.nan, 100.0, np.nan, 100.0], index=idx, dtype="float64"),
    )

    result = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            spread_rate=0.0,
            trade_size_mode="volatility_scaled",
            trade_size_value=0.1,
            volatility_target_annual=0.1,
            volatility_lookback=3,
            volatility_min_scale=0.01,
            volatility_max_scale=5.0,
            close_open_position_on_last_bar=False,
        )
    ).run(data, strategy)

    assert len(result.trades) == 1
    assert [event.event_type for event in result.execution_events] == ["entry", "exit"]
    assert all(event.event_type != "reduce" for event in result.execution_events)


def test_signal_contracts_direction_overrides_stale_signal_direction_for_execution_intent() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0],
            "volume": [1_000] * 5,
        },
        index=idx,
    )
    # Signals stay long, but contracts deterministically flip short at bar 2.
    strategy = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 1, 1, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, -2.0, -2.0, 0.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 1, -1, -1, 0], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, 100.0, 100.0, np.nan, 100.0], index=idx, dtype="float64"),
    )

    result = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            spread_rate=0.0,
            trade_size_mode="units",
            trade_size_value=1.0,
            close_open_position_on_last_bar=False,
        )
    ).run(data, strategy)

    events_at_flip_bar = [event for event in result.execution_events if event.time == idx[2]]
    assert [event.event_type for event in events_at_flip_bar[:2]] == ["exit", "entry"]
    assert [event.side for event in events_at_flip_bar[:2]] == ["sell", "sell"]
    assert result.positions.iloc[2] == -1


def test_explicit_flat_signal_overrides_stale_nonzero_contract_series() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0],
            "volume": [1_000] * 5,
        },
        index=idx,
    )
    strategy = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 0, 0, 0], index=idx, dtype="int8"),
        # stale non-zero contracts persist despite explicit flat signal
        pd.Series([0.0, 1.0, 1.0, 1.0, 0.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 1, 0, 0, 0], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, 100.0, 100.0, np.nan, np.nan], index=idx, dtype="float64"),
    )

    result = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            spread_rate=0.0,
            trade_size_mode="units",
            trade_size_value=1.0,
            close_open_position_on_last_bar=False,
        )
    ).run(data, strategy)

    assert result.positions.iloc[2] == 0
    events_at_flat_bar = [event for event in result.execution_events if event.time == idx[2]]
    assert [event.event_type for event in events_at_flat_bar] == ["exit"]


def test_flat_intent_without_fill_hint_does_not_close_position() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1_000] * 4,
        },
        index=idx,
    )
    strategy = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 0, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 0.0, 0.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 1, 0, 0], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, 100.0, np.nan, np.nan], index=idx, dtype="float64"),
    )

    result = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            spread_rate=0.0,
            trade_size_mode="units",
            trade_size_value=1.0,
            close_open_position_on_last_bar=False,
        )
    ).run(data, strategy)

    assert result.trades == []
    assert [event.event_type for event in result.execution_events] == ["entry"]


def test_flat_exit_uses_fractal_context_from_open_entry_when_exit_bar_has_stale_1w_marker() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1_000] * 4,
        },
        index=idx,
    )
    strategy = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 0, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 0.0, 0.0], index=idx, dtype="float64"),
        first_wiseman_setup_side=pd.Series([0, 0, 1, 0], index=idx, dtype="int8"),
        first_wiseman_ignored_reason=pd.Series(["", "", "", ""], index=idx, dtype="object"),
        first_wiseman_fill_prices=pd.Series([np.nan, np.nan, np.nan, np.nan], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 1, 0, 0], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, 100.0, 100.0, np.nan], index=idx, dtype="float64"),
    )

    result = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            spread_rate=0.0,
            trade_size_mode="units",
            trade_size_value=1.0,
            close_open_position_on_last_bar=False,
        )
    ).run(data, strategy)

    assert len(result.trades) == 1
    assert result.trades[0].entry_signal == "Bullish Fractal"
    assert result.trades[0].exit_signal == "Signal Intent Flat from Bullish Fractal"


def test_explicit_stop_entry_fill_gaps_to_bar_open_when_trigger_is_outside_bar_range() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [95.0, 120.0, 121.0],
            "high": [96.0, 125.0, 122.0],
            "low": [94.0, 119.0, 120.0],
            "close": [95.0, 123.0, 121.0],
            "volume": [1_000] * 3,
        },
        index=idx,
    )
    strategy = _SignalContractsExitReasonStrategy(
        pd.Series([0, 1, 0], index=idx, dtype="int8"),
        pd.Series([0.0, 1.0, 0.0], index=idx, dtype="float64"),
        fractal_position_side=pd.Series([0, 1, 1], index=idx, dtype="int8"),
        fill_prices=pd.Series([np.nan, 100.0, np.nan], index=idx, dtype="float64"),
    )

    result = BacktestEngine(
        BacktestConfig(
            fee_rate=0.0,
            slippage_rate=0.0,
            spread_rate=0.0,
            order_type="stop",
            trade_size_mode="units",
            trade_size_value=1.0,
        )
    ).run(data, strategy)

    assert len(result.trades) == 1
    assert result.trades[0].entry_price == pytest.approx(120.0)
    assert result.execution_events[0].price == pytest.approx(120.0)




def test_infer_exit_signal_normalizes_strategy_reason_labels() -> None:
    engine = BacktestEngine()
    strategy = _SignalBarStrategy(pd.Series(dtype="int8"))
    strategy.signal_exit_reason = pd.Series(
        ["", "1W Reversal Stop", "Red Gator PP", "Green Gator PP", "Peak Drawdown PP"],
        dtype="object",
    )

    stop_label = engine._infer_exit_signal(strategy, signal_index=1, prior_position=1, desired_position=0)
    red_label = engine._infer_exit_signal(strategy, signal_index=2, prior_position=1, desired_position=0)
    green_label = engine._infer_exit_signal(strategy, signal_index=3, prior_position=-1, desired_position=0)
    peak_label = engine._infer_exit_signal(strategy, signal_index=4, prior_position=1, desired_position=0)

    assert stop_label == "Strategy Stop Loss Bullish 1W"
    assert red_label == "Strategy Profit Protection Red Gator"
    assert green_label == "Strategy Profit Protection Green Gator"
    assert peak_label == "Strategy Profit Protection Peak Drawdown"
def test_infer_exit_signal_uses_descriptive_fallback_labels() -> None:
    engine = BacktestEngine()
    strategy = _SignalBarStrategy(pd.Series(dtype="int8"))

    close_long = engine._infer_exit_signal(strategy, signal_index=0, prior_position=1, desired_position=0)
    close_short = engine._infer_exit_signal(strategy, signal_index=0, prior_position=-1, desired_position=0)
    reverse_to_bullish = engine._infer_exit_signal(strategy, signal_index=0, prior_position=-1, desired_position=1)

    assert close_long == "Signal Intent Flat from Bullish 1W"
    assert close_short == "Signal Intent Flat from Bearish 1W"
    assert reverse_to_bullish == "Signal Intent Flip to Bullish 1W"


def test_signal_intent_flat_timestamp_is_recorded_on_trade() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [1_000.0] * 4,
        },
        index=idx,
    )
    signals = pd.Series([0, 1, 0, 0], index=idx, dtype="int8")

    engine = BacktestEngine(BacktestConfig(initial_capital=10_000, fee_rate=0.0, slippage_rate=0.0, trade_size_mode="units", trade_size_value=1.0))
    result = engine.run(data, _SignalBarStrategy(signals))

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_signal == "Signal Intent Flat from Bullish 1W"
    assert trade.signal_intent_flat_time == idx[2]

    trades_df = result.trades_dataframe()
    assert trades_df.loc[0, "signal_intent_flat_time"] == idx[2]



def test_write_signal_intent_flat_timestamps_file(tmp_path) -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [1_000.0] * 4,
        },
        index=idx,
    )
    signals = pd.Series([0, 1, 0, 0], index=idx, dtype="int8")

    engine = BacktestEngine(BacktestConfig(initial_capital=10_000, fee_rate=0.0, slippage_rate=0.0, trade_size_mode="units", trade_size_value=1.0))
    result = engine.run(data, _SignalBarStrategy(signals))

    out_path = write_signal_intent_flat_timestamps(result, tmp_path)

    assert out_path == tmp_path / "signal_intent_flat_timestamps.txt"
    content = out_path.read_text(encoding="utf-8")
    assert idx[2].isoformat() in content
    assert "Signal Intent Flat from Bullish 1W" in content


def test_finalize_exit_signal_maps_reversal_stop_to_reversal_entry_label() -> None:
    engine = BacktestEngine()

    assert engine._finalize_exit_signal_label("Strategy Stop Loss Bullish 1W", "Bullish 1W-R") == "Strategy Stop Loss Bullish 1W-R"
    assert engine._finalize_exit_signal_label("Strategy Stop Loss Bearish 1W", "Bearish 1W-R") == "Strategy Stop Loss Bearish 1W-R"
    assert engine._finalize_exit_signal_label("Strategy Reversal to Bullish 1W", "Bullish 1W-R") == "Strategy Reversal to Bullish 1W"


def test_trade_execution_markers_include_compact_strategy_suffixes() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1_000] * 4,
        },
        index=idx,
    )
    trades = [
        Trade(
            side="long",
            entry_time=idx[1],
            exit_time=idx[2],
            entry_price=102.0,
            exit_price=102.5,
            units=1.0,
            pnl=0.5,
            return_pct=0.0049,
            holding_bars=1,
            entry_signal="Bullish Fractal",
            exit_signal="Strategy Profit Protection Green Gator",
        ),
        Trade(
            side="short",
            entry_time=idx[2],
            exit_time=idx[3],
            entry_price=102.5,
            exit_price=103.0,
            units=1.0,
            pnl=-0.5,
            return_pct=-0.0049,
            holding_bars=1,
            entry_signal="Bearish 3W",
            exit_signal="NTD Entry Stop",
        ),
    ]

    markers = chart_module._trade_execution_markers(trades, data)

    assert [str(marker["text"]) for marker in markers] == ["LE-F", "LX-G/SE", "SX-S"]
    assert [str(marker["position"]) for marker in markers] == ["belowBar", "aboveBar", "belowBar"]


def test_execution_event_markers_prevent_duplicate_entry_markers_from_partial_trade_slices() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1_000] * 4,
        },
        index=idx,
    )
    trades = [
        Trade(
            side="long",
            entry_time=idx[1],
            exit_time=idx[2],
            entry_price=101.0,
            exit_price=102.0,
            units=1.0,
            pnl=1.0,
            return_pct=1.0 / 101.0,
            holding_bars=1,
            entry_signal="Bullish Fractal",
            exit_signal="Signal Intent Reduce from Bullish Fractal",
        ),
        Trade(
            side="long",
            entry_time=idx[1],
            exit_time=idx[3],
            entry_price=101.0,
            exit_price=103.0,
            units=1.0,
            pnl=2.0,
            return_pct=2.0 / 101.0,
            holding_bars=2,
            entry_signal="Bullish Fractal",
            exit_signal="Signal Intent Flat from Bullish Fractal",
        ),
    ]
    execution_events = [
        ExecutionEvent("entry", idx[1], "buy", 101.0, 2.0, strategy_reason="Bullish Fractal"),
        ExecutionEvent("reduce", idx[2], "sell", 102.0, 1.0, strategy_reason="Signal Intent Reduce from Bullish Fractal"),
        ExecutionEvent("exit", idx[3], "sell", 103.0, 1.0, strategy_reason="Signal Intent Flat from Bullish Fractal"),
    ]

    trade_markers = chart_module._trade_execution_markers(trades, data)
    execution_markers = chart_module._execution_event_markers(execution_events, data)

    trade_entry_texts = [str(marker["text"]) for marker in trade_markers if str(marker["text"]).startswith("LE-")]
    execution_entry_texts = [str(marker["text"]) for marker in execution_markers if str(marker["text"]).startswith("LE-")]

    assert trade_entry_texts == ["LE-F/LE-F"]
    assert execution_entry_texts == ["LE-F"]


def test_local_chart_prefers_execution_event_markers_over_trade_slices(tmp_path) -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [1_000] * 4,
        },
        index=idx,
    )
    trades = [
        Trade(
            side="long",
            entry_time=idx[1],
            exit_time=idx[2],
            entry_price=101.0,
            exit_price=102.0,
            units=1.0,
            pnl=1.0,
            return_pct=1.0 / 101.0,
            holding_bars=1,
            entry_signal="Bullish Fractal",
            exit_signal="Signal Intent Reduce from Bullish Fractal",
        ),
        Trade(
            side="long",
            entry_time=idx[1],
            exit_time=idx[3],
            entry_price=101.0,
            exit_price=103.0,
            units=1.0,
            pnl=2.0,
            return_pct=2.0 / 101.0,
            holding_bars=2,
            entry_signal="Bullish Fractal",
            exit_signal="Signal Intent Flat from Bullish Fractal",
        ),
    ]
    execution_events = [
        ExecutionEvent("entry", idx[1], "buy", 101.0, 2.0, strategy_reason="Bullish Fractal"),
        ExecutionEvent("reduce", idx[2], "sell", 102.0, 1.0, strategy_reason="Signal Intent Reduce from Bullish Fractal"),
        ExecutionEvent("exit", idx[3], "sell", 103.0, 1.0, strategy_reason="Signal Intent Flat from Bullish Fractal"),
    ]

    out = tmp_path / "execution_markers_chart.html"
    generate_local_tradingview_chart(data, trades, str(out), execution_events=execution_events)
    html = out.read_text(encoding="utf-8")

    assert "LE-F/LE-F" not in html
    assert '"text": "LE-F"' in html


def test_bw_strategy_executes_1w_entry_reversal_and_short_stop_exit(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=9, freq="W", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.2, 8.2, 10.0, 10.1, 10.0, 9.8, 9.9, 10.0],
            "high": [10.5, 10.7, 11.0, 11.2, 10.8, 10.6, 10.4, 11.3, 10.4],
            "low": [9.0, 10.0, 8.0, 9.0, 9.5, 9.4, 7.9, 9.6, 9.7],
            "close": [10.1, 10.3, 10.4, 10.8, 10.2, 10.1, 8.4, 11.0, 10.1],
        },
        index=idx,
    )
    jaw = pd.Series(12.0, index=idx)
    teeth = pd.Series(10.2, index=idx)
    lips = pd.Series(10.0, index=idx)
    ao = pd.Series([0.2, 1.5, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4], index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)

    strategy = BWStrategy(divergence_filter_bars=4)
    result = BacktestEngine().run(data, strategy)

    reasons = [str(event.strategy_reason) for event in result.execution_events]
    assert reasons[:3] == ["Bullish 1W", "Strategy Reversal to Bearish 1W", "Bearish 1W-R"]


def test_bw_strategy_executes_bearish_1w_reversal_and_long_stop_exit(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=8, freq="W", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [9.8, 10.2, 11.8, 10.8, 10.6, 9.8, 9.4, 9.6],
            "high": [10.0, 11.0, 12.0, 11.5, 11.2, 12.1, 10.1, 10.0],
            "low": [9.3, 9.5, 9.0, 8.9, 9.2, 9.4, 8.8, 9.0],
            "close": [9.9, 10.5, 9.2, 9.4, 9.6, 10.5, 9.0, 9.4],
        },
        index=idx,
    )
    jaw = pd.Series(10.0, index=idx)
    teeth = pd.Series(10.2, index=idx)
    lips = pd.Series(12.0, index=idx)
    ao = pd.Series([2.0, 1.5, 1.6, 1.4, 1.3, 1.2, 1.1, 1.0], index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)

    strategy = BWStrategy(divergence_filter_bars=4)
    result = BacktestEngine().run(data, strategy)

    reasons = [str(event.strategy_reason) for event in result.execution_events]
    assert reasons[:3] == ["Bearish 1W", "Strategy Reversal to Bullish 1W", "Bullish 1W-R"]


def test_bw_long_position_flips_on_triggered_bearish_1w(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=7, freq="W", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.2, 8.5, 10.4, 11.6, 11.2, 10.8],
            "high": [10.4, 10.8, 11.0, 11.2, 12.0, 11.8, 11.0],
            "low": [9.2, 10.0, 8.0, 9.0, 9.5, 9.4, 9.8],
            "close": [10.1, 10.4, 10.6, 10.8, 9.8, 10.4, 10.1],
        },
        index=idx,
    )
    jaw = pd.Series([12.0, 12.0, 12.0, 12.0, 10.0, 10.0, 10.0], index=idx)
    teeth = pd.Series([11.0, 11.0, 11.0, 11.0, 10.5, 10.5, 10.5], index=idx)
    lips = pd.Series([10.0, 10.0, 10.0, 10.0, 12.0, 12.0, 12.0], index=idx)
    ao = pd.Series([1.0, 2.0, 1.5, 1.0, 1.2, 1.1, 1.0], index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)

    strategy = BWStrategy(divergence_filter_bars=0)
    result = BacktestEngine().run(data, strategy)

    reasons = [str(event.strategy_reason) for event in result.execution_events]
    assert reasons[:3] == ["Bullish 1W", "Signal Intent Flip to Bearish 1W", "Bearish 1W"]




def test_bw_reversal_short_flattens_and_flips_on_new_bullish_1w_trigger(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=9, freq="W", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.2, 8.2, 10.0, 10.1, 10.0, 9.8, 9.5, 9.9],
            "high": [10.5, 10.7, 11.0, 11.2, 10.8, 10.6, 10.4, 10.2, 10.7],
            "low": [9.0, 10.0, 8.0, 9.0, 9.5, 9.4, 7.9, 7.6, 9.4],
            "close": [10.1, 10.3, 10.4, 10.8, 10.2, 10.1, 8.4, 9.9, 10.5],
        },
        index=idx,
    )
    jaw = pd.Series([12.0, 12.0, 12.0, 12.0, 10.0, 10.0, 10.0, 12.0, 12.0], index=idx)
    teeth = pd.Series([11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0], index=idx)
    lips = pd.Series([10.0, 10.0, 10.0, 10.0, 12.0, 12.0, 12.0, 10.0, 10.0], index=idx)
    ao = pd.Series([0.2, 1.5, 1.0, 0.9, 1.2, 1.1, 0.6, 0.5, 0.4], index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)

    strategy = BWStrategy(divergence_filter_bars=0)
    result = BacktestEngine().run(data, strategy)

    reasons = [str(event.strategy_reason) for event in result.execution_events]
    assert reasons[:5] == [
        "Bullish 1W",
        "Strategy Reversal to Bearish 1W",
        "Bearish 1W-R",
        "Strategy Exit Reason: 1W-R Flattened by Opposite 1W",
        "Bullish 1W",
    ]


def test_bw_short_position_flips_on_triggered_bullish_1w(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=7, freq="W", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [9.5, 10.0, 11.6, 10.0, 8.6, 9.0, 9.3],
            "high": [10.0, 11.0, 12.0, 11.4, 10.0, 10.2, 10.3],
            "low": [9.0, 9.6, 9.0, 8.9, 8.0, 8.2, 8.4],
            "close": [9.6, 10.3, 9.2, 9.4, 9.5, 9.8, 9.9],
        },
        index=idx,
    )
    jaw = pd.Series([10.0, 10.0, 10.0, 10.0, 12.0, 12.0, 12.0], index=idx)
    teeth = pd.Series([10.4, 10.4, 10.4, 10.4, 11.0, 11.0, 11.0], index=idx)
    lips = pd.Series([12.0, 12.0, 12.0, 12.0, 10.0, 10.0, 10.0], index=idx)
    ao = pd.Series([2.0, 1.5, 1.6, 1.3, 1.0, 1.1, 1.2], index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)

    strategy = BWStrategy(divergence_filter_bars=0)
    result = BacktestEngine().run(data, strategy)

    reasons = [str(event.strategy_reason) for event in result.execution_events]
    assert reasons[:3] == ["Bearish 1W", "Signal Intent Flip to Bullish 1W", "Bullish 1W"]


def test_bw_strategy_requires_1w_setup_relative_to_red_teeth(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=4, freq="W", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.1, 10.2, 10.3],
            "high": [10.4, 10.5, 12.8, 11.2],
            "low": [9.8, 9.7, 10.0, 9.4],
            "close": [10.1, 10.2, 10.6, 9.9],
        },
        index=idx,
    )
    jaw = pd.Series([12.0, 12.0, 12.0, 10.0], index=idx)
    teeth = pd.Series([11.0, 11.0, 11.0, 11.0], index=idx)
    lips = pd.Series([10.0, 10.0, 10.0, 12.0], index=idx)
    ao = pd.Series([1.0, 0.8, 0.6, 0.9], index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)

    strategy = BWStrategy(divergence_filter_bars=0)
    strategy.generate_signals(data)
    setup_side = strategy.signal_first_wiseman_setup_marker_side
    assert setup_side is not None

    # Bar 2 fails bullish teeth gate (high above teeth), bar 3 fails bearish teeth gate (low below teeth).
    assert int(setup_side.iloc[2]) == 0
    assert int(setup_side.iloc[3]) == 0


def test_bw_strategy_marks_ignored_reasons_for_filtered_1w_candidates(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=7, freq="W", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.1, 10.2, 10.3, 10.2, 10.3, 10.2],
            "high": [10.4, 10.5, 12.8, 10.8, 10.7, 10.6, 10.5],
            "low": [9.8, 9.7, 9.6, 9.4, 9.2, 9.0, 8.9],
            "close": [10.1, 10.2, 10.6, 10.4, 10.5, 10.4, 10.3],
        },
        index=idx,
    )
    jaw = pd.Series([12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0], index=idx)
    teeth = pd.Series([11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0], index=idx)
    lips = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], index=idx)
    ao = pd.Series([1.0, 0.9, 0.8, 0.7, 0.6, 0.3, 0.2], index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)

    strategy = BWStrategy(divergence_filter_bars=2)
    strategy.generate_signals(data)
    setup_side = strategy.signal_first_wiseman_setup_marker_side
    ignored = strategy.signal_first_wiseman_ignored_reason
    assert setup_side is not None
    assert ignored is not None

    assert int(setup_side.iloc[2]) == 1
    assert str(ignored.iloc[2]) == "gator_closed_canceled"
    assert int(setup_side.iloc[4]) == 1
    assert str(ignored.iloc[4]) == "ao_divergence_filter"


def test_bw_strategy_marks_gator_open_percentile_filter_as_1w_g(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=5, freq="W", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [9.5, 9.4, 9.2, 9.3, 9.4],
            "high": [9.8, 9.7, 9.5, 9.6, 9.7],
            "low": [9.1, 9.0, 8.8, 9.1, 9.2],
            "close": [9.6, 9.5, 9.4, 9.4, 9.5],
        },
        index=idx,
    )
    jaw = pd.Series([12.0, 12.0, 12.0, 12.0, 12.0], index=idx)
    teeth = pd.Series([11.0, 11.0, 11.0, 11.0, 11.0], index=idx)
    lips = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0], index=idx)
    ao = pd.Series([1.0, 0.9, 0.8, 0.7, 0.6], index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        gator_open_filter_lookback=3,
        gator_open_filter_min_percentile=50.0,
    )
    strategy.generate_signals(data)
    setup_side = strategy.signal_first_wiseman_setup_marker_side
    ignored = strategy.signal_first_wiseman_ignored_reason
    assert setup_side is not None
    assert ignored is not None

    assert int(setup_side.iloc[2]) == 1
    assert str(ignored.iloc[2]) == "gator_open_percentile_filter"


def test_bw_can_close_on_opposite_1w_d_with_min_unrealized_gate(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=6, freq="W", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [11.0, 11.1, 11.4, 11.0, 8.3, 8.5],
            "high": [11.3, 11.4, 11.8, 11.2, 8.8, 9.1],
            "low": [10.8, 10.9, 11.1, 10.7, 8.0, 8.2],
            "close": [11.1, 11.2, 11.2, 10.9, 8.6, 8.8],
        },
        index=idx,
    )
    jaw = pd.Series([8.0, 8.0, 8.0, 8.0, 11.0, 11.0], index=idx)
    teeth = pd.Series([9.0, 9.0, 9.0, 9.0, 10.0, 10.0], index=idx)
    lips = pd.Series([10.0, 10.0, 10.0, 10.0, 9.0, 9.0], index=idx)
    ao = pd.Series([0.5, 0.2, 0.3, 0.2, 0.1, 0.2], index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)

    strategy = BWStrategy(
        divergence_filter_bars=2,
        red_teeth_profit_protection_enabled=False,
        green_lips_profit_protection_enabled=False,
        allow_close_on_1w_d=True,
        allow_close_on_1w_d_min_unrealized_return=0.10,
    )
    signals = strategy.generate_signals(data)
    fill_prices = strategy.signal_fill_prices
    exit_reason = strategy.signal_exit_reason

    assert int(signals.iloc[3]) == -1
    assert int(signals.iloc[4]) == -1
    assert int(signals.iloc[5]) == 0
    assert fill_prices is not None and float(fill_prices.iloc[5]) == pytest.approx(8.8)
    assert exit_reason is not None and str(exit_reason.iloc[5]) == "1W-D Opposite Close"


def test_bw_can_close_on_opposite_1w_a_with_min_unrealized_gate(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=6, freq="W", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [11.0, 11.1, 11.4, 11.0, 8.3, 8.5],
            "high": [11.3, 11.4, 11.8, 11.2, 11.5, 12.0],
            "low": [10.8, 10.9, 11.1, 10.7, 8.0, 8.1],
            "close": [11.1, 11.2, 11.2, 10.9, 8.6, 11.4],
        },
        index=idx,
    )
    jaw = pd.Series([8.0, 8.0, 8.0, 8.0, 8.0, 8.0], index=idx)
    teeth = pd.Series([9.0, 9.0, 9.0, 9.0, 9.0, 9.0], index=idx)
    lips = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0, 10.0], index=idx)
    ao = pd.Series([0.5, 0.2, 0.3, 0.2, 0.1, 0.2], index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)

    strategy = BWStrategy(
        divergence_filter_bars=2,
        red_teeth_profit_protection_enabled=False,
        green_lips_profit_protection_enabled=False,
        allow_close_on_1w_a=True,
        allow_close_on_1w_a_min_unrealized_return=0.10,
    )
    signals = strategy.generate_signals(data)
    fill_prices = strategy.signal_fill_prices
    exit_reason = strategy.signal_exit_reason

    assert int(signals.iloc[3]) == -1
    assert int(signals.iloc[4]) == -1
    assert int(signals.iloc[5]) == 0
    assert fill_prices is not None and float(fill_prices.iloc[5]) == pytest.approx(11.5)
    assert exit_reason is not None and str(exit_reason.iloc[5]) == "1W-A Opposite Close"


def test_bw_strategy_marks_invalidation_before_trigger_for_canceled_pending_setup(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=5, freq="W", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.1, 9.6, 9.8, 9.7],
            "high": [10.5, 10.6, 10.4, 10.0, 10.5],
            "low": [9.7, 9.8, 9.4, 9.3, 9.2],
            "close": [10.1, 10.2, 10.0, 9.9, 10.1],
        },
        index=idx,
    )
    jaw = pd.Series([12.0, 12.0, 12.0, 12.0, 12.0], index=idx)
    teeth = pd.Series([11.0, 11.0, 11.0, 11.0, 11.0], index=idx)
    lips = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0], index=idx)
    ao = pd.Series([1.0, 0.9, 0.8, 0.7, 0.6], index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)

    strategy = BWStrategy(divergence_filter_bars=0)
    strategy.generate_signals(data)
    ignored = strategy.signal_first_wiseman_ignored_reason
    assert ignored is not None

    assert str(ignored.iloc[2]) == "invalidation_before_trigger"


def test_bw_strategy_marks_all_valid_1w_setups_for_charting(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=6, freq="W", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.1, 9.8, 9.9, 9.6, 9.7],
            "high": [10.6, 10.7, 10.8, 10.7, 10.6, 10.7],
            "low": [9.9, 9.8, 9.5, 9.6, 9.0, 9.1],
            "close": [10.1, 10.2, 10.1, 10.0, 10.0, 9.9],
        },
        index=idx,
    )
    jaw = pd.Series(12.0, index=idx)
    teeth = pd.Series(11.0, index=idx)
    lips = pd.Series(10.0, index=idx)
    ao = pd.Series([1.0, 0.9, 0.7, 0.6, 0.4, 0.3], index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)

    strategy = BWStrategy(divergence_filter_bars=0)
    strategy.generate_signals(data)
    setup_side = strategy.signal_first_wiseman_setup_marker_side
    assert setup_side is not None

    # Both bars are valid bullish 1W setups, even though the earlier setup is still pending.
    assert int(setup_side.iloc[2]) == 1
    assert int(setup_side.iloc[4]) == 1


def test_bw_reversal_marker_side_aligns_with_reversal_execution_bar(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=9, freq="W", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.2, 8.2, 10.0, 10.1, 10.0, 9.8, 9.5, 9.9],
            "high": [10.5, 10.7, 11.0, 11.2, 10.8, 10.6, 10.4, 10.2, 10.7],
            "low": [9.0, 10.0, 8.0, 9.0, 9.5, 9.4, 7.9, 7.6, 9.4],
            "close": [10.1, 10.3, 10.4, 10.8, 10.2, 10.1, 8.4, 9.9, 10.5],
        },
        index=idx,
    )
    jaw = pd.Series([12.0, 12.0, 12.0, 12.0, 10.0, 10.0, 10.0, 12.0, 12.0], index=idx)
    teeth = pd.Series([11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0], index=idx)
    lips = pd.Series([10.0, 10.0, 10.0, 10.0, 12.0, 12.0, 12.0, 10.0, 10.0], index=idx)
    ao = pd.Series([0.2, 1.5, 1.0, 0.9, 1.2, 1.1, 0.6, 0.5, 0.4], index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)

    strategy = BWStrategy(divergence_filter_bars=0)
    result = BacktestEngine().run(data, strategy)
    reversal_side = strategy.signal_first_wiseman_reversal_side
    assert reversal_side is not None

    short_reversal_events = [event for event in result.execution_events if str(event.strategy_reason) == "Bearish 1W-R"]
    assert short_reversal_events
    reversal_time = short_reversal_events[0].time
    assert int(reversal_side.loc[reversal_time]) == -1
    assert int((reversal_side != 0).sum()) == 1


def test_bw_latest_pending_opposite_setup_replaces_earlier_one(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=8, freq="W", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.1, 9.6, 10.0, 11.4, 11.3, 10.7, 10.6],
            "high": [10.4, 10.5, 10.8, 11.0, 12.0, 12.2, 11.1, 10.9],
            "low": [9.8, 9.7, 9.5, 9.7, 10.0, 9.8, 9.9, 9.8],
            "close": [10.1, 10.2, 10.1, 10.8, 10.2, 10.1, 10.5, 10.4],
        },
        index=idx,
    )
    jaw = pd.Series([12.0, 12.0, 12.0, 12.0, 10.0, 10.0, 10.0, 10.0], index=idx)
    teeth = pd.Series([11.0, 11.0, 11.0, 11.0, 10.5, 10.5, 10.5, 10.5], index=idx)
    lips = pd.Series([10.0, 10.0, 10.0, 10.0, 12.0, 12.0, 12.0, 12.0], index=idx)
    ao = pd.Series([0.2, 0.1, 0.0, -0.1, 0.2, 0.3, 0.1, 0.0], index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)

    strategy = BWStrategy(divergence_filter_bars=0)
    result = BacktestEngine().run(data, strategy)

    reasons = [str(event.strategy_reason) for event in result.execution_events]
    # Long entry, then opposite-side flip uses the latest bearish setup (earlier pending setup is canceled).
    assert reasons[:3] == ["Bullish 1W", "Signal Intent Flip to Bearish 1W", "Bearish 1W"]
    se_events = [event for event in result.execution_events if str(event.strategy_reason) == "Bearish 1W"]
    assert se_events
    # If the earlier pending setup (bar 4 low=10.0) were used, the flip would trigger one bar earlier.
    assert se_events[0].time == idx[7]


def test_bw_opposite_side_trade_deterministically_flattens_prior_1w_r_position(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=9, freq="W", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.2, 8.2, 10.0, 10.1, 10.0, 9.8, 9.5, 9.9],
            "high": [10.5, 10.7, 11.0, 11.2, 10.8, 10.6, 10.4, 10.2, 10.7],
            "low": [9.0, 10.0, 8.0, 9.0, 9.5, 9.4, 7.9, 7.6, 9.4],
            "close": [10.1, 10.3, 10.4, 10.8, 10.2, 10.1, 8.4, 9.9, 10.5],
        },
        index=idx,
    )
    jaw = pd.Series([12.0, 12.0, 12.0, 12.0, 10.0, 10.0, 10.0, 12.0, 12.0], index=idx)
    teeth = pd.Series([11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0], index=idx)
    lips = pd.Series([10.0, 10.0, 10.0, 10.0, 12.0, 12.0, 12.0, 10.0, 10.0], index=idx)
    ao = pd.Series([0.2, 1.5, 1.0, 0.9, 1.2, 1.1, 0.6, 0.5, 0.4], index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)

    result = BacktestEngine().run(data, BWStrategy(divergence_filter_bars=0))
    events = [(event.time, str(event.strategy_reason)) for event in result.execution_events]

    # The long opened from earlier 1W/1W-R flow is flattened and reversed deterministically on the same bar.
    bar_events = [reason for time, reason in events if time == idx[8]]
    assert bar_events == ["Strategy Exit Reason: 1W-R Flattened by Opposite 1W", "Bullish 1W", "Engine End of Backtest"]


def test_bw_ntd_initial_fractal_long_entry_uses_fractal_high_stop_buy(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [99.9, 100.0, 100.1, 100.0, 100.0, 100.05, 100.1, 101.0],
            "high": [100.1, 100.2, 101.0, 100.2, 100.1, 100.1, 101.2, 101.5],
            "low": [99.8, 99.7, 100.0, 99.8, 99.9, 99.7, 100.0, 100.9],
            "close": [100.0, 100.1, 100.2, 100.0, 100.0, 100.0, 101.1, 101.2],
        },
        index=idx,
    )

    jaw = pd.Series(100.0, index=idx)
    teeth = pd.Series(100.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series(np.nan, index=idx)
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, False, True, False, False, False, False, False],
            "down_fractal": [False, True, False, False, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=True,
        ntd_sleeping_gator_lookback=2,
        ntd_sleeping_gator_tightness_mult=1.0,
        ntd_ranging_lookback=3,
        ntd_ranging_max_span_pct=0.05,
    )
    signals = strategy.generate_signals(data)
    fill_prices = strategy.signal_fill_prices
    assert int(signals.iloc[6]) == 1
    assert fill_prices is not None
    assert float(fill_prices.iloc[6]) == pytest.approx(101.0)


def test_bw_ntd_initial_fractal_stop_uses_lower_low_between_fractal_and_entry(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-02-01", periods=8, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0, 100.0, 99.8, 100.1, 100.2],
            "high": [100.2, 100.1, 101.0, 100.2, 100.1, 100.0, 101.1, 101.3],
            "low": [99.8, 99.6, 100.0, 99.9, 99.8, 99.2, 100.0, 100.1],
            "close": [100.0, 100.0, 100.2, 100.0, 100.0, 99.9, 101.0, 101.2],
        },
        index=idx,
    )
    jaw = pd.Series(100.0, index=idx)
    teeth = pd.Series(100.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series(np.nan, index=idx)
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, False, True, False, False, False, False, False],
            "down_fractal": [False, True, False, False, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=True,
        ntd_sleeping_gator_lookback=2,
        ntd_sleeping_gator_tightness_mult=1.0,
        ntd_ranging_lookback=3,
        ntd_ranging_max_span_pct=0.05,
    )
    strategy.generate_signals(data)
    stop_loss = strategy.signal_stop_loss_prices
    assert stop_loss is not None
    assert float(stop_loss.iloc[6]) == pytest.approx(99.2)


def test_bw_ntd_initial_fractal_short_entry_uses_fractal_low_stop_sell(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-03-01", periods=8, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.1, 100.0, 100.0, 100.0, 100.0, 99.95, 99.9, 99.7],
            "high": [100.3, 100.3, 100.2, 100.1, 100.1, 100.0, 100.0, 99.9],
            "low": [99.9, 100.0, 99.0, 99.8, 99.9, 99.9, 98.8, 98.7],
            "close": [100.0, 100.1, 99.2, 100.0, 100.0, 99.95, 98.9, 98.8],
        },
        index=idx,
    )

    jaw = pd.Series(100.0, index=idx)
    teeth = pd.Series(100.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series(np.nan, index=idx)
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, True, False, False, False, False, False, False],
            "down_fractal": [False, False, True, False, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=True,
        ntd_sleeping_gator_lookback=2,
        ntd_sleeping_gator_tightness_mult=1.0,
        ntd_ranging_lookback=3,
        ntd_ranging_max_span_pct=0.05,
    )
    signals = strategy.generate_signals(data)
    fill_prices = strategy.signal_fill_prices
    assert int(signals.iloc[6]) == -1
    assert fill_prices is not None
    assert float(fill_prices.iloc[6]) == pytest.approx(99.0)


def test_bw_ntd_initial_fractal_short_stop_uses_higher_high_between_fractal_and_entry(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-04-01", periods=8, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.25, 99.95, 99.8],
            "high": [100.2, 101.0, 100.1, 100.1, 100.2, 101.4, 100.0, 99.9],
            "low": [99.8, 100.5, 99.0, 99.8, 99.9, 100.2, 98.9, 98.7],
            "close": [100.0, 100.7, 99.2, 100.0, 100.0, 100.25, 99.0, 98.8],
        },
        index=idx,
    )
    jaw = pd.Series(100.0, index=idx)
    teeth = pd.Series(100.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series(np.nan, index=idx)
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, True, False, False, False, False, False, False],
            "down_fractal": [False, False, True, False, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=True,
        ntd_sleeping_gator_lookback=2,
        ntd_sleeping_gator_tightness_mult=1.0,
        ntd_ranging_lookback=3,
        ntd_ranging_max_span_pct=0.05,
    )
    strategy.generate_signals(data)
    stop_loss = strategy.signal_stop_loss_prices
    assert stop_loss is not None
    assert float(stop_loss.iloc[6]) == pytest.approx(101.4)


def test_bw_ntd_entry_can_trigger_on_same_bar_as_stop_loss(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-05-01", periods=9, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.1, 100.0, 100.0, 100.0, 100.0, 99.8, 99.7],
            "high": [101.0, 100.5, 105.0, 100.4, 100.2, 100.3, 105.5, 100.2, 99.9],
            "low": [99.5, 99.2, 100.0, 99.8, 99.0, 99.4, 100.0, 98.5, 98.8],
            "close": [100.0, 99.8, 100.2, 100.0, 99.7, 99.9, 105.0, 98.9, 99.0],
        },
        index=idx,
    )
    jaw = pd.Series(100.0, index=idx)
    teeth = pd.Series(100.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series(np.nan, index=idx)
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, False, True, False, False, False, False, False, False],
            "down_fractal": [False, True, False, False, True, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=True,
        ntd_sleeping_gator_lookback=2,
        ntd_sleeping_gator_tightness_mult=1.0,
        ntd_ranging_lookback=3,
        ntd_ranging_max_span_pct=0.2,
    )
    signals = strategy.generate_signals(data)
    fill_prices = strategy.signal_fill_prices
    assert fill_prices is not None
    assert int(signals.iloc[6]) == 1
    assert float(fill_prices.iloc[6]) == pytest.approx(105.0)
    assert int(signals.iloc[7]) == -1
    assert float(fill_prices.iloc[7]) == pytest.approx(99.0)


def test_bw_ntd_fractal_position_flattens_when_fractal_stop_is_breached(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-05-20", periods=9, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.1, 100.0, 100.0, 100.0, 100.2, 99.6, 99.8],
            "high": [100.6, 100.5, 105.0, 100.4, 100.3, 100.4, 105.2, 100.0, 100.1],
            "low": [99.6, 99.3, 100.0, 99.8, 99.7, 99.6, 100.1, 98.8, 99.0],
            "close": [100.0, 99.9, 100.2, 100.0, 100.0, 100.1, 105.0, 99.4, 99.7],
        },
        index=idx,
    )
    jaw = pd.Series(100.0, index=idx)
    teeth = pd.Series(100.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series(np.nan, index=idx)
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, False, True, False, False, False, False, False, False],
            "down_fractal": [False, True, False, False, False, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=True,
        ntd_sleeping_gator_lookback=2,
        ntd_sleeping_gator_tightness_mult=1.0,
        ntd_ranging_lookback=3,
        ntd_ranging_max_span_pct=0.2,
    )
    signals = strategy.generate_signals(data)
    contracts = strategy.signal_contracts
    fill_prices = strategy.signal_fill_prices
    assert contracts is not None
    assert fill_prices is not None

    # Long fractal fills on bar 6, then bar 7 breaches the active fractal stop.
    # The position should flatten at the stop level even without an opposing setup.
    assert int(signals.iloc[6]) == 1
    assert float(fill_prices.iloc[6]) == pytest.approx(105.0)
    assert int(signals.iloc[7]) == 0
    assert float(contracts.iloc[7]) == pytest.approx(0.0)
    assert float(fill_prices.iloc[7]) == pytest.approx(98.8)


def test_bw_ntd_opposite_fractal_reverses_open_position_with_fractal_labels(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-06-01", periods=10, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [103.0, 103.0, 103.1, 103.0, 103.0, 103.2, 103.2, 102.8, 102.5, 102.4],
            "high": [103.6, 103.4, 105.0, 103.5, 103.3, 103.4, 105.2, 103.0, 102.8, 102.7],
            "low": [102.6, 101.0, 102.7, 102.8, 102.0, 102.6, 103.0, 101.8, 102.1, 102.0],
            "close": [103.1, 102.9, 103.3, 103.1, 102.9, 103.2, 104.9, 102.0, 102.4, 102.3],
        },
        index=idx,
    )
    jaw = pd.Series(103.0, index=idx)
    teeth = pd.Series(103.0, index=idx)
    lips = pd.Series(103.0, index=idx)
    ao = pd.Series(np.nan, index=idx)
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, False, True, False, False, False, False, False, False, False],
            "down_fractal": [False, True, False, False, True, False, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=True,
        ntd_sleeping_gator_lookback=2,
        ntd_sleeping_gator_tightness_mult=1.0,
        ntd_ranging_lookback=3,
        ntd_ranging_max_span_pct=0.2,
    )
    result = BacktestEngine().run(data, strategy)
    reasons = [str(event.strategy_reason) for event in result.execution_events]
    assert reasons[:3] == [
        "Bullish Fractal",
        "Signal Intent Flip to Bearish Fractal",
        "Bearish Fractal",
    ]


def test_bw_ntd_fractal_entry_reverses_on_valid_opposite_1w_signal(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-08-01", periods=9, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.2, 100.1, 100.0, 100.1, 100.1, 102.0, 103.0, 102.5, 101.0],
            "high": [100.4, 100.3, 102.0, 100.2, 100.3, 102.2, 104.5, 103.0, 101.2],
            "low": [99.8, 99.6, 99.7, 99.9, 100.0, 101.8, 102.0, 101.0, 100.6],
            "close": [100.1, 100.0, 101.8, 100.0, 100.2, 102.1, 102.1, 101.2, 100.8],
        },
        index=idx,
    )
    jaw = pd.Series(95.0, index=idx)
    teeth = pd.Series(96.0, index=idx)
    lips = pd.Series(97.0, index=idx)
    ao = pd.Series([0.0, 0.0, 0.0, 0.05, 0.1, 0.2, 0.6, 0.55, 0.5], index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, False, True, False, False, False, False, False, False],
            "down_fractal": [False, True, False, False, False, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=True,
        ntd_sleeping_gator_lookback=2,
        ntd_sleeping_gator_tightness_mult=1.0,
        ntd_ranging_lookback=2,
        ntd_ranging_max_span_pct=0.2,
    )
    signals = strategy.generate_signals(data)
    fill_prices = strategy.signal_fill_prices

    assert fill_prices is not None
    assert float(fill_prices.iloc[5]) == pytest.approx(102.0)
    assert int(signals.iloc[7]) == -1
    assert float(fill_prices.iloc[7]) == pytest.approx(102.0)


def test_bw_1w_stop_out_uses_setup_level_as_fill_price(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-09-01", periods=6, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.5, 100.2, 99.6],
            "high": [100.4, 100.3, 101.0, 101.2, 100.4, 100.0],
            "low": [99.7, 99.5, 99.0, 99.8, 98.8, 99.1],
            "close": [100.1, 100.0, 100.8, 101.0, 99.2, 99.5],
        },
        index=idx,
    )
    jaw = pd.Series(105.0, index=idx)
    teeth = pd.Series(104.0, index=idx)
    lips = pd.Series(103.0, index=idx)
    ao = pd.Series([0.0, 1.0, 0.5, 0.4, 0.3, 0.2], index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)},
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(divergence_filter_bars=0, ntd_initial_fractal_enabled=False)
    signals = strategy.generate_signals(data)
    fill_prices = strategy.signal_fill_prices
    contracts = strategy.signal_contracts

    assert fill_prices is not None
    assert contracts is not None
    assert int(signals.iloc[3]) == 1
    assert float(fill_prices.iloc[3]) == pytest.approx(101.0)
    assert int(signals.iloc[4]) == 0
    assert float(fill_prices.iloc[4]) == pytest.approx(99.0)
    assert float(contracts.iloc[4]) == pytest.approx(0.0)


def test_bw_1w_long_entry_then_stop_can_both_execute_on_same_bar(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-09-01", periods=5, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.2, 100.0],
            "high": [100.4, 100.3, 101.0, 101.3, 100.2],
            "low": [99.7, 99.5, 99.0, 98.8, 99.6],
            "close": [100.1, 100.0, 100.8, 99.2, 99.9],
        },
        index=idx,
    )
    jaw = pd.Series(102.0, index=idx)
    teeth = pd.Series(101.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series([0.0, 1.0, 0.5, 0.4, 0.3], index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)},
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(divergence_filter_bars=0, ntd_initial_fractal_enabled=False, red_teeth_profit_protection_enabled=False)
    signals = strategy.generate_signals(data)
    assert strategy.signal_fill_prices is not None
    assert strategy.signal_contracts is not None

    # Entry is triggered first at setup high, then same-bar stop-out at setup low.
    assert int(signals.iloc[3]) == 0
    assert float(strategy.signal_fill_prices.iloc[3]) == pytest.approx(99.0)
    assert float(strategy.signal_contracts.iloc[3]) == pytest.approx(0.0)


def test_bw_1w_short_entry_then_stop_can_both_execute_on_same_bar(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-09-01", periods=5, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 99.8, 100.0],
            "high": [100.2, 100.4, 101.2, 101.4, 100.2],
            "low": [99.6, 99.5, 99.6, 99.1, 99.7],
            "close": [100.0, 100.1, 99.7, 100.9, 100.0],
        },
        index=idx,
    )
    jaw = pd.Series(98.0, index=idx)
    teeth = pd.Series(99.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series([0.0, -0.1, 0.3, 0.4, 0.2], index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)},
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(divergence_filter_bars=0, ntd_initial_fractal_enabled=False, red_teeth_profit_protection_enabled=False)
    signals = strategy.generate_signals(data)
    assert strategy.signal_fill_prices is not None
    assert strategy.signal_contracts is not None

    # Entry is triggered first at setup low, then same-bar stop-out at setup high.
    assert int(signals.iloc[3]) == 0
    assert float(strategy.signal_fill_prices.iloc[3]) == pytest.approx(101.2)
    assert float(strategy.signal_contracts.iloc[3]) == pytest.approx(0.0)


def test_bw_same_bar_1w_entry_and_stop_generate_two_execution_events(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-09-01", periods=5, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.2, 100.0],
            "high": [100.4, 100.3, 101.0, 101.3, 100.2],
            "low": [99.7, 99.5, 99.0, 98.8, 99.6],
            "close": [100.1, 100.0, 100.8, 99.2, 99.9],
        },
        index=idx,
    )
    jaw = pd.Series(102.0, index=idx)
    teeth = pd.Series(101.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series([0.0, 1.0, 0.5, 0.4, 0.3], index=idx, dtype="float64")
    fractals = pd.DataFrame({"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)}, index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(divergence_filter_bars=0, ntd_initial_fractal_enabled=False, red_teeth_profit_protection_enabled=False)
    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(data, strategy)

    same_bar_events = [event for event in result.execution_events if event.time == idx[3]]
    assert [str(event.strategy_reason) for event in same_bar_events] == [
        "Bullish 1W",
        "Strategy Stop Loss Bullish 1W",
    ]
    assert len(result.trades) >= 1
    assert result.trades[0].entry_time == idx[3]
    assert result.trades[0].exit_time == idx[3]
    assert result.trades[0].pnl < 0


def test_bw_same_bar_1w_entry_and_stop_not_ignored_with_stop_loss_scaled_sizing(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-09-01", periods=5, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.2, 100.0],
            "high": [100.4, 100.3, 101.0, 101.3, 100.2],
            "low": [99.7, 99.5, 99.0, 98.8, 99.6],
            "close": [100.1, 100.0, 100.8, 99.2, 99.9],
        },
        index=idx,
    )
    jaw = pd.Series(102.0, index=idx)
    teeth = pd.Series(101.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series([0.0, 1.0, 0.5, 0.4, 0.3], index=idx, dtype="float64")
    fractals = pd.DataFrame({"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)}, index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(divergence_filter_bars=0, ntd_initial_fractal_enabled=False, red_teeth_profit_protection_enabled=False)
    config = BacktestConfig(fee_rate=0.0, slippage_rate=0.0, trade_size_mode="stop_loss_scaled", trade_size_value=0.01)
    result = BacktestEngine(config).run(data, strategy)

    same_bar_events = [event for event in result.execution_events if event.time == idx[3]]
    assert [str(event.strategy_reason) for event in same_bar_events] == [
        "Bullish 1W",
        "Strategy Stop Loss Bullish 1W",
    ]
    assert len(result.trades) >= 1
    assert result.trades[0].entry_time == idx[3]
    assert result.trades[0].exit_time == idx[3]


def test_bw_same_bar_1w_entry_stop_and_reversal_generate_three_execution_events(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-09-01", periods=7, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [102.0, 102.2, 102.1, 101.8, 101.7, 101.9, 101.0],
            "high": [103.0, 102.8, 102.5, 102.2, 102.3, 103.2, 101.2],
            "low": [101.0, 101.2, 100.0, 101.1, 100.6, 99.8, 98.5],
            "close": [102.4, 102.0, 102.3, 101.9, 101.8, 100.4, 99.0],
        },
        index=idx,
    )
    jaw = pd.Series(104.0, index=idx)
    teeth = pd.Series(102.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series([0.0, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1], index=idx, dtype="float64")
    fractals = pd.DataFrame({"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)}, index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(divergence_filter_bars=0, ntd_initial_fractal_enabled=False, red_teeth_profit_protection_enabled=False)
    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(data, strategy)

    same_bar_events = [event for event in result.execution_events if event.time == idx[5]]
    assert [str(event.strategy_reason) for event in same_bar_events[:3]] == [
        "Bullish 1W",
        "Strategy Stop Loss Bullish 1W",
        "Bearish 1W-R",
    ]
    assert len(result.trades) >= 2
    assert result.trades[0].entry_time == idx[5]
    assert result.trades[0].exit_time == idx[5]
    assert result.trades[0].pnl < 0
    assert result.trades[1].entry_time == idx[5]
    assert result.trades[1].side == "short"


def test_bw_opposite_same_bar_1w_entry_and_stop_executes_new_trade_instead_of_ignoring(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=6, freq="W", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.2, 8.5, 10.4, 11.6, 11.2],
            "high": [10.4, 10.8, 11.0, 11.2, 12.0, 12.1],
            "low": [9.2, 10.0, 8.0, 9.0, 9.5, 9.4],
            "close": [10.1, 10.4, 10.6, 10.8, 9.8, 10.4],
        },
        index=idx,
    )
    jaw = pd.Series([12.0, 12.0, 12.0, 12.0, 10.0, 10.0], index=idx)
    teeth = pd.Series([11.0, 11.0, 11.0, 11.0, 10.5, 10.5], index=idx)
    lips = pd.Series([10.0, 10.0, 10.0, 10.0, 12.0, 12.0], index=idx)
    ao = pd.Series([1.0, 2.0, 1.5, 1.0, 1.2, 1.1], index=idx)
    fractals = pd.DataFrame({"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)}, index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(divergence_filter_bars=0, ntd_initial_fractal_enabled=False, red_teeth_profit_protection_enabled=False)
    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(data, strategy)

    same_bar_events = [event for event in result.execution_events if event.time == idx[5]]
    assert [str(event.strategy_reason) for event in same_bar_events] == [
        "Strategy Reversal to Bearish 1W",
        "Bearish 1W",
        "Strategy Stop Loss Bearish 1W",
    ]
    assert len(result.trades) >= 2
    assert result.trades[0].side == "long"
    assert result.trades[0].exit_time == idx[5]
    assert result.trades[1].side == "short"
    assert result.trades[1].entry_time == idx[5]
    assert result.trades[1].exit_time == idx[5]


def test_bw_opposite_same_bar_1w_entry_and_stop_executes_new_long_trade_instead_of_ignoring(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=6, freq="W", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [9.5, 10.0, 11.6, 10.0, 8.6, 9.0],
            "high": [10.0, 11.0, 12.0, 11.4, 10.0, 10.1],
            "low": [9.0, 9.6, 9.0, 8.9, 8.0, 7.9],
            "close": [9.6, 10.3, 9.2, 9.4, 9.5, 9.8],
        },
        index=idx,
    )
    jaw = pd.Series([10.0, 10.0, 10.0, 10.0, 12.0, 12.0], index=idx)
    teeth = pd.Series([10.4, 10.4, 10.4, 10.4, 11.0, 11.0], index=idx)
    lips = pd.Series([12.0, 12.0, 12.0, 12.0, 10.0, 10.0], index=idx)
    ao = pd.Series([2.0, 1.5, 1.6, 1.3, 1.0, 1.1], index=idx)
    fractals = pd.DataFrame({"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)}, index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(divergence_filter_bars=0, ntd_initial_fractal_enabled=False, red_teeth_profit_protection_enabled=False)
    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(data, strategy)

    same_bar_events = [event for event in result.execution_events if event.time == idx[5]]
    assert [str(event.strategy_reason) for event in same_bar_events] == [
        "Strategy Reversal to Bullish 1W",
        "Bullish 1W",
        "Strategy Stop Loss Bullish 1W",
    ]
    assert len(result.trades) >= 2
    assert result.trades[0].side == "short"
    assert result.trades[0].exit_time == idx[5]


    assert result.trades[1].side == "long"
    assert result.trades[1].entry_time == idx[5]
    assert result.trades[1].exit_time == idx[5]


def test_bw_opposite_1w_short_entry_then_stop_can_both_execute_on_same_bar(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=6, freq="W", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.2, 8.5, 10.4, 11.6, 11.2],
            "high": [10.4, 10.8, 11.0, 11.2, 12.0, 12.1],
            "low": [9.2, 10.0, 8.0, 9.0, 9.5, 9.4],
            "close": [10.1, 10.4, 10.6, 10.8, 9.8, 10.4],
        },
        index=idx,
    )
    jaw = pd.Series([12.0, 12.0, 12.0, 12.0, 10.0, 10.0], index=idx)
    teeth = pd.Series([11.0, 11.0, 11.0, 11.0, 10.5, 10.5], index=idx)
    lips = pd.Series([10.0, 10.0, 10.0, 10.0, 12.0, 12.0], index=idx)
    ao = pd.Series([1.0, 2.0, 1.5, 1.0, 1.2, 1.1], index=idx)
    fractals = pd.DataFrame({"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)}, index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(divergence_filter_bars=0, ntd_initial_fractal_enabled=False, red_teeth_profit_protection_enabled=False)
    signals = strategy.generate_signals(data)
    assert strategy.signal_fill_prices is not None
    assert strategy.signal_contracts is not None

    # Bar 5 triggers opposite bearish 1W entry then hits that setup's stop in same bar.
    assert int(signals.iloc[5]) == 0
    assert float(strategy.signal_fill_prices.iloc[5]) == pytest.approx(12.0)
    assert float(strategy.signal_contracts.iloc[5]) == pytest.approx(0.0)


def test_bw_opposite_1w_long_entry_then_stop_can_both_execute_on_same_bar(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=6, freq="W", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [9.5, 10.0, 11.6, 10.0, 8.6, 9.0],
            "high": [10.0, 11.0, 12.0, 11.4, 10.0, 10.1],
            "low": [9.0, 9.6, 9.0, 8.9, 8.0, 7.9],
            "close": [9.6, 10.3, 9.2, 9.4, 9.5, 9.8],
        },
        index=idx,
    )
    jaw = pd.Series([10.0, 10.0, 10.0, 10.0, 12.0, 12.0], index=idx)
    teeth = pd.Series([10.4, 10.4, 10.4, 10.4, 11.0, 11.0], index=idx)
    lips = pd.Series([12.0, 12.0, 12.0, 12.0, 10.0, 10.0], index=idx)
    ao = pd.Series([2.0, 1.5, 1.6, 1.3, 1.0, 1.1], index=idx)
    fractals = pd.DataFrame({"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)}, index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(divergence_filter_bars=0, ntd_initial_fractal_enabled=False, red_teeth_profit_protection_enabled=False)
    signals = strategy.generate_signals(data)
    assert strategy.signal_fill_prices is not None
    assert strategy.signal_contracts is not None

    # Bar 5 triggers opposite bullish 1W entry then hits that setup's stop in same bar.
    assert int(signals.iloc[5]) == 0
    assert float(strategy.signal_fill_prices.iloc[5]) == pytest.approx(8.0)
    assert float(strategy.signal_contracts.iloc[5]) == pytest.approx(0.0)


def test_bw_opposite_same_bar_short_entry_stop_does_not_emit_random_1w_entry_next_bar(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=7, freq="W", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.2, 8.5, 10.4, 11.6, 11.2, 10.3],
            "high": [10.4, 10.8, 11.0, 11.2, 12.0, 12.1, 10.6],
            "low": [9.2, 10.0, 8.0, 9.0, 9.5, 9.4, 9.9],
            "close": [10.1, 10.4, 10.6, 10.8, 9.8, 10.4, 10.2],
        },
        index=idx,
    )
    jaw = pd.Series([12.0, 12.0, 12.0, 12.0, 10.0, 10.0, 10.0], index=idx)
    teeth = pd.Series([11.0, 11.0, 11.0, 11.0, 10.5, 10.5, 10.5], index=idx)
    lips = pd.Series([10.0, 10.0, 10.0, 10.0, 12.0, 12.0, 12.0], index=idx)
    ao = pd.Series([1.0, 2.0, 1.5, 1.0, 1.2, 1.1, 1.0], index=idx)
    fractals = pd.DataFrame({"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)}, index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(divergence_filter_bars=0, ntd_initial_fractal_enabled=False, red_teeth_profit_protection_enabled=False)
    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(data, strategy)

    next_bar_events = [event for event in result.execution_events if event.time == idx[6]]
    assert next_bar_events == []
    assert strategy.signal_contracts is not None
    assert float(strategy.signal_contracts.iloc[6]) == pytest.approx(0.0)


def test_bw_opposite_same_bar_long_entry_stop_does_not_emit_random_1w_entry_next_bar(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=7, freq="W", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [9.5, 10.0, 11.6, 10.0, 8.6, 9.0, 9.7],
            "high": [10.0, 11.0, 12.0, 11.4, 10.0, 10.1, 10.2],
            "low": [9.0, 9.6, 9.0, 8.9, 8.0, 7.9, 9.1],
            "close": [9.6, 10.3, 9.2, 9.4, 9.5, 9.8, 9.6],
        },
        index=idx,
    )
    jaw = pd.Series([10.0, 10.0, 10.0, 10.0, 12.0, 12.0, 12.0], index=idx)
    teeth = pd.Series([10.4, 10.4, 10.4, 10.4, 11.0, 11.0, 11.0], index=idx)
    lips = pd.Series([12.0, 12.0, 12.0, 12.0, 10.0, 10.0, 10.0], index=idx)
    ao = pd.Series([2.0, 1.5, 1.6, 1.3, 1.0, 1.1, 1.2], index=idx)
    fractals = pd.DataFrame({"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)}, index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(divergence_filter_bars=0, ntd_initial_fractal_enabled=False, red_teeth_profit_protection_enabled=False)
    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(data, strategy)

    next_bar_events = [event for event in result.execution_events if event.time == idx[6]]
    assert next_bar_events == []
    assert strategy.signal_contracts is not None
    assert float(strategy.signal_contracts.iloc[6]) == pytest.approx(0.0)


def test_bw_red_teeth_profit_protection_exits_long_on_close_below_teeth(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-09-01", periods=6, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.5, 100.4, 100.0],
            "high": [100.4, 100.3, 101.0, 101.2, 103.0, 102.0],
            "low": [99.7, 99.5, 99.0, 99.8, 99.2, 99.6],
            "close": [100.1, 100.0, 100.8, 101.0, 101.1, 99.8],
        },
        index=idx,
    )
    jaw = pd.Series(102.0, index=idx)
    teeth = pd.Series(101.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series([0.0, 1.0, 0.5, 0.4, 0.3, 0.2], index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)},
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=False,
        red_teeth_profit_protection_enabled=True,
        red_teeth_profit_protection_min_bars=1,
        red_teeth_profit_protection_min_unrealized_return=0.01,
        red_teeth_profit_protection_volatility_lookback=2,
        red_teeth_profit_protection_annualized_volatility_scaler=1e9,
    )
    strategy.generate_signals(data)

    assert strategy.signal_fill_prices is not None
    assert strategy.signal_contracts is not None
    assert strategy.signal_stop_loss_prices is not None
    assert strategy.signal_exit_reason is not None
    assert float(strategy.signal_fill_prices.iloc[5]) == pytest.approx(float(data["close"].iloc[5]))
    assert float(strategy.signal_stop_loss_prices.iloc[5]) == pytest.approx(float(data["close"].iloc[5]))
    assert float(strategy.signal_contracts.iloc[5]) == pytest.approx(0.0)
    assert str(strategy.signal_exit_reason.iloc[5]) == "Red Gator Teeth PP"


def test_bw_profit_protection_min_unrealized_return_defaults_are_split_by_signal() -> None:
    strategy = BWStrategy()
    assert strategy.red_teeth_profit_protection_min_unrealized_return == pytest.approx(1.0)
    assert strategy.green_lips_profit_protection_min_unrealized_return == pytest.approx(1.1)
    assert strategy.zones_profit_protection_min_unrealized_return == pytest.approx(1.0)
    assert strategy.zones_profit_protection_min_same_color_bars == 5
    assert strategy.peak_drawdown_exit_enabled is False
    assert strategy.peak_drawdown_exit_pct == pytest.approx(0.01)
    assert strategy.peak_drawdown_exit_volatility_lookback == 20
    assert strategy.peak_drawdown_exit_annualized_volatility_scaler == pytest.approx(1.0)

    disabled_scaling = BWStrategy(
        red_teeth_profit_protection_annualized_volatility_scaler=0.0,
        green_lips_profit_protection_annualized_volatility_scaler=0.0,
        zones_profit_protection_annualized_volatility_scaler=0.0,
        peak_drawdown_exit_annualized_volatility_scaler=0.0,
    )
    assert disabled_scaling.red_teeth_profit_protection_annualized_volatility_scaler == pytest.approx(0.0)
    assert disabled_scaling.green_lips_profit_protection_annualized_volatility_scaler == pytest.approx(0.0)
    assert disabled_scaling.zones_profit_protection_annualized_volatility_scaler == pytest.approx(0.0)
    assert disabled_scaling.peak_drawdown_exit_annualized_volatility_scaler == pytest.approx(0.0)
    assert strategy.sigma_move_profit_protection_enabled is False
    assert strategy.sigma_move_profit_protection_lookback == 20
    assert strategy.sigma_move_profit_protection_sigma == pytest.approx(2.0)


def test_bw_peak_drawdown_exit_parameters_validate() -> None:
    with pytest.raises(ValueError, match="peak_drawdown_exit_pct"):
        BWStrategy(peak_drawdown_exit_pct=-0.01)
    with pytest.raises(ValueError, match="peak_drawdown_exit_volatility_lookback"):
        BWStrategy(peak_drawdown_exit_volatility_lookback=1)
    BWStrategy(peak_drawdown_exit_annualized_volatility_scaler=0.0)
    with pytest.raises(ValueError, match="peak_drawdown_exit_annualized_volatility_scaler"):
        BWStrategy(peak_drawdown_exit_annualized_volatility_scaler=-0.1)
    with pytest.raises(ValueError, match="sigma_move_profit_protection_lookback"):
        BWStrategy(sigma_move_profit_protection_lookback=1)
    with pytest.raises(ValueError, match="sigma_move_profit_protection_sigma"):
        BWStrategy(sigma_move_profit_protection_sigma=0.0)
    with pytest.raises(ValueError, match="close_on_underlying_gain_pct"):
        BWStrategy(close_on_underlying_gain_pct=-0.01)


def test_bw_sigma_move_profit_protection_exits_at_bar_close_after_intrabar_sigma_touch(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-08-15", periods=10, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [101.2, 101.4, 101.1, 101.0, 100.9, 100.8, 101.0, 100.7, 101.3, 101.4],
            "high": [101.8, 103.0, 101.6, 101.3, 101.2, 101.1, 101.8, 101.0, 102.0, 102.1],
            "low": [100.9, 101.0, 100.5, 100.8, 100.7, 100.5, 100.4, 95.0, 101.0, 101.2],
            "close": [101.3, 102.8, 100.8, 101.0, 100.9, 100.7, 101.6, 100.8, 101.8, 101.9],
        },
        index=idx,
    )
    jaw = pd.Series(104.0, index=idx)
    teeth = pd.Series(102.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series([1.5, 1.4, 1.2, 1.1, 1.0, 0.8, 0.6, 0.55, 0.5, 0.45], index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, True, False, False, False, False, False, False, False, False],
            "down_fractal": [False, False, True, False, False, False, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=True,
        ntd_sleeping_gator_lookback=2,
        ntd_sleeping_gator_tightness_mult=1.0,
        ntd_ranging_lookback=3,
        ntd_ranging_max_span_pct=0.2,
        red_teeth_profit_protection_enabled=False,
        green_lips_profit_protection_enabled=False,
        zones_profit_protection_enabled=False,
        peak_drawdown_exit_enabled=False,
        sigma_move_profit_protection_enabled=True,
        sigma_move_profit_protection_lookback=3,
        sigma_move_profit_protection_sigma=1.0,
    )
    signals = strategy.generate_signals(data)

    assert strategy.signal_fill_prices is not None
    assert strategy.signal_stop_loss_prices is not None
    assert strategy.signal_contracts is not None
    assert strategy.signal_exit_reason is not None
    assert int(signals.iloc[5]) == -1
    assert str(strategy.signal_exit_reason.iloc[6]) == "Sigma Move PP"
    assert float(strategy.signal_fill_prices.iloc[6]) == pytest.approx(float(data["close"].iloc[6]))
    assert float(strategy.signal_stop_loss_prices.iloc[6]) == pytest.approx(float(data["close"].iloc[6]))
    assert float(strategy.signal_contracts.iloc[6]) == pytest.approx(0.0)


def test_bw_underlying_gain_target_exits_intrabar_at_target_price(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-08-15", periods=10, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [101.2, 101.4, 101.1, 101.0, 100.9, 100.8, 99.2, 99.0, 99.3, 99.4],
            "high": [101.8, 103.0, 101.6, 101.3, 101.2, 101.1, 99.5, 99.6, 99.8, 100.0],
            "low": [100.9, 101.0, 100.5, 100.8, 100.7, 100.5, 98.5, 98.9, 99.0, 99.2],
            "close": [101.3, 102.8, 100.8, 101.0, 100.9, 100.7, 99.0, 99.2, 99.5, 99.7],
        },
        index=idx,
    )
    jaw = pd.Series(104.0, index=idx)
    teeth = pd.Series(102.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series([1.5, 1.4, 1.2, 1.1, 1.0, 0.8, 0.6, 0.55, 0.5, 0.45], index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, True, False, False, False, False, False, False, False, False],
            "down_fractal": [False, False, True, False, False, False, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=True,
        ntd_sleeping_gator_lookback=2,
        ntd_sleeping_gator_tightness_mult=1.0,
        ntd_ranging_lookback=3,
        ntd_ranging_max_span_pct=0.2,
        red_teeth_profit_protection_enabled=False,
        green_lips_profit_protection_enabled=False,
        zones_profit_protection_enabled=False,
        peak_drawdown_exit_enabled=False,
        sigma_move_profit_protection_enabled=False,
        close_on_underlying_gain_pct=0.01,
    )
    signals = strategy.generate_signals(data)

    assert strategy.signal_fill_prices is not None
    assert strategy.signal_stop_loss_prices is not None
    assert strategy.signal_contracts is not None
    assert strategy.signal_exit_reason is not None
    assert int(signals.iloc[5]) == -1
    assert str(strategy.signal_exit_reason.iloc[6]) == "Underlying Gain Target"
    entry_fill = float(strategy.signal_fill_prices.iloc[5])
    assert np.isfinite(entry_fill)
    target_price = entry_fill / 1.01
    expected_fill = (
        target_price
        if float(data["open"].iloc[6]) >= target_price
        else float(data["open"].iloc[6])
    )
    assert float(strategy.signal_fill_prices.iloc[6]) == pytest.approx(expected_fill)
    assert float(strategy.signal_stop_loss_prices.iloc[6]) == pytest.approx(expected_fill)
    assert float(strategy.signal_contracts.iloc[6]) == pytest.approx(0.0)


def test_bw_zones_profit_protection_min_same_color_bars_must_be_positive() -> None:
    with pytest.raises(ValueError, match="zones_profit_protection_min_same_color_bars"):
        BWStrategy(zones_profit_protection_min_same_color_bars=0)


def test_bw_zones_profit_protection_uses_resting_stop_order_not_same_bar_exit(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-09-01", periods=8, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.5, 100.6, 100.8, 100.9, 100.7],
            "high": [100.4, 100.3, 101.0, 101.2, 101.4, 101.5, 101.2, 100.9],
            "low": [99.7, 99.5, 99.0, 99.8, 100.2, 100.6, 100.5, 100.3],
            "close": [100.1, 100.0, 100.8, 101.0, 101.2, 101.3, 100.9, 100.4],
        },
        index=idx,
    )
    jaw = pd.Series(107.0, index=idx, dtype="float64")
    teeth = pd.Series(106.0, index=idx, dtype="float64")
    lips = pd.Series(105.0, index=idx, dtype="float64")
    ao = pd.Series([0.0, 1.0, 0.5, 0.4, 0.6, 0.8, 1.0, 1.2], index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)},
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=False,
        red_teeth_profit_protection_enabled=False,
        green_lips_profit_protection_enabled=False,
        zones_profit_protection_enabled=True,
        zones_profit_protection_min_bars=1,
        zones_profit_protection_min_unrealized_return=0.0,
        zones_profit_protection_volatility_lookback=2,
        zones_profit_protection_annualized_volatility_scaler=1e9,
        zones_profit_protection_min_same_color_bars=1,
    )
    signals = strategy.generate_signals(data)

    assert strategy.signal_exit_reason is not None
    assert strategy.signal_contracts is not None
    # Entry fills long on bar 3.
    assert int(signals.iloc[3]) == 1
    # First armed stop is placed from bar 4 low, but cannot execute until a later bar.
    assert str(strategy.signal_exit_reason.iloc[4]) != "Williams Zones PP"
    assert int(signals.iloc[4]) == 1
    # Bar 6 breaks the prior resting stop and exits.
    assert str(strategy.signal_exit_reason.iloc[6]) == "Williams Zones PP"
    assert int(signals.iloc[6]) == 0
    assert float(strategy.signal_contracts.iloc[6]) == pytest.approx(0.0)


def test_bw_red_teeth_profit_protection_exits_short_on_close_above_teeth(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-09-01", periods=6, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 99.8, 99.1, 99.5],
            "high": [100.2, 100.4, 101.2, 100.3, 100.8, 100.0],
            "low": [99.6, 99.5, 99.6, 99.2, 97.5, 99.0],
            "close": [100.0, 100.1, 99.7, 98.8, 99.0, 100.4],
        },
        index=idx,
    )
    jaw = pd.Series(98.0, index=idx)
    teeth = pd.Series(99.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series([0.0, -0.1, 0.3, 0.4, 0.2, 0.1], index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)},
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=False,
        red_teeth_profit_protection_enabled=True,
        red_teeth_profit_protection_min_bars=1,
        red_teeth_profit_protection_min_unrealized_return=0.01,
        red_teeth_profit_protection_volatility_lookback=2,
        red_teeth_profit_protection_annualized_volatility_scaler=1e9,
    )
    strategy.generate_signals(data)

    assert strategy.signal_fill_prices is not None
    assert strategy.signal_contracts is not None
    assert strategy.signal_stop_loss_prices is not None
    assert strategy.signal_exit_reason is not None
    assert float(strategy.signal_fill_prices.iloc[5]) == pytest.approx(float(data["close"].iloc[5]))
    assert float(strategy.signal_stop_loss_prices.iloc[5]) == pytest.approx(float(data["close"].iloc[5]))
    assert float(strategy.signal_contracts.iloc[5]) == pytest.approx(0.0)
    assert str(strategy.signal_exit_reason.iloc[5]) == "Red Gator Teeth PP"


def test_bw_green_lips_profit_protection_exits_long_on_close_below_lips(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-09-01", periods=6, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.5, 100.4, 100.0],
            "high": [100.4, 100.3, 101.0, 101.2, 103.0, 102.0],
            "low": [99.7, 99.5, 99.0, 99.8, 99.2, 99.6],
            "close": [100.1, 100.0, 100.8, 101.0, 101.1, 99.8],
        },
        index=idx,
    )
    jaw = pd.Series(102.0, index=idx)
    teeth = pd.Series(101.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series([0.0, 1.0, 0.5, 0.4, 0.3, 0.2], index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)},
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=False,
        red_teeth_profit_protection_enabled=False,
        green_lips_profit_protection_enabled=True,
        green_lips_profit_protection_min_bars=1,
        green_lips_profit_protection_min_unrealized_return=0.01,
        green_lips_profit_protection_volatility_lookback=2,
        green_lips_profit_protection_annualized_volatility_scaler=1e9,
    )
    strategy.generate_signals(data)

    assert strategy.signal_fill_prices is not None
    assert strategy.signal_contracts is not None
    assert strategy.signal_stop_loss_prices is not None
    assert strategy.signal_exit_reason is not None
    assert float(strategy.signal_fill_prices.iloc[5]) == pytest.approx(float(data["close"].iloc[5]))
    assert float(strategy.signal_stop_loss_prices.iloc[5]) == pytest.approx(float(data["close"].iloc[5]))
    assert float(strategy.signal_contracts.iloc[5]) == pytest.approx(0.0)
    assert str(strategy.signal_exit_reason.iloc[5]) == "Green Gator Lips PP"


def test_bw_green_lips_profit_protection_exits_short_on_close_above_lips(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-09-01", periods=6, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 99.8, 99.1, 99.5],
            "high": [100.2, 100.4, 101.2, 100.3, 100.8, 100.8],
            "low": [99.6, 99.5, 99.6, 99.2, 97.5, 99.0],
            "close": [100.0, 100.1, 99.7, 98.8, 99.0, 100.4],
        },
        index=idx,
    )
    jaw = pd.Series(98.0, index=idx)
    teeth = pd.Series(99.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series([0.0, -0.1, 0.3, 0.4, 0.2, 0.1], index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)},
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=False,
        red_teeth_profit_protection_enabled=False,
        green_lips_profit_protection_enabled=True,
        green_lips_profit_protection_min_bars=1,
        green_lips_profit_protection_min_unrealized_return=0.01,
        green_lips_profit_protection_volatility_lookback=2,
        green_lips_profit_protection_annualized_volatility_scaler=1e9,
    )
    strategy.generate_signals(data)

    assert strategy.signal_fill_prices is not None
    assert strategy.signal_contracts is not None
    assert strategy.signal_stop_loss_prices is not None
    assert strategy.signal_exit_reason is not None
    assert float(strategy.signal_fill_prices.iloc[5]) == pytest.approx(float(data["close"].iloc[5]))
    assert float(strategy.signal_stop_loss_prices.iloc[5]) == pytest.approx(float(data["close"].iloc[5]))
    assert float(strategy.signal_contracts.iloc[5]) == pytest.approx(0.0)
    assert str(strategy.signal_exit_reason.iloc[5]) == "Green Gator Lips PP"


def test_bw_profit_protection_min_unrealized_latch_flags_default_off() -> None:
    strategy = BWStrategy()
    assert strategy.red_teeth_latch_min_unrealized_return is False
    assert strategy.green_lips_latch_min_unrealized_return is False

    strategy_with_latches = BWStrategy(
        red_teeth_latch_min_unrealized_return=True,
        green_lips_latch_min_unrealized_return=True,
    )
    assert strategy_with_latches.red_teeth_latch_min_unrealized_return is True
    assert strategy_with_latches.green_lips_latch_min_unrealized_return is True


def test_bw_red_teeth_profit_protection_flatten_takes_priority_over_opposite_1w_flip(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-09-01", periods=7, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.5, 101.0, 99.8, 99.6],
            "high": [100.4, 100.3, 101.0, 101.2, 102.0, 102.0, 99.9],
            "low": [99.7, 99.5, 99.0, 99.8, 99.4, 98.5, 99.1],
            "close": [100.1, 100.0, 100.8, 101.0, 100.2, 99.0, 99.4],
        },
        index=idx,
    )
    jaw = pd.Series([102.0, 102.0, 102.0, 102.0, 98.0, 98.0, 98.0], index=idx, dtype="float64")
    teeth = pd.Series([101.0, 101.0, 101.0, 101.0, 99.9, 99.5, 99.5], index=idx, dtype="float64")
    lips = pd.Series([100.0, 100.0, 100.0, 100.0, 100.5, 100.2, 100.1], index=idx, dtype="float64")
    ao = pd.Series([0.0, 1.0, 0.5, 0.4, 0.6, 0.5, 0.4], index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)},
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=False,
        red_teeth_profit_protection_enabled=True,
        red_teeth_profit_protection_min_bars=1,
        red_teeth_profit_protection_min_unrealized_return=0.01,
        red_teeth_profit_protection_volatility_lookback=2,
        red_teeth_profit_protection_annualized_volatility_scaler=1e9,
    )
    signals = strategy.generate_signals(data)

    assert strategy.signal_contracts is not None
    assert strategy.signal_exit_reason is not None
    assert int(signals.iloc[3]) == 1
    assert str(strategy.signal_exit_reason.iloc[5]) == "Red Gator Teeth PP"
    assert float(strategy.signal_contracts.iloc[5]) == pytest.approx(0.0)
    # If opposite 1W logic were allowed to run before PP flatten, this bar could flip short.
    assert int(signals.iloc[5]) == 0
    assert float(strategy.signal_contracts.iloc[5]) == pytest.approx(0.0)


def test_bw_red_teeth_profit_protection_requires_fresh_close_cross(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-09-01", periods=7, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.5, 100.4, 100.3, 100.1],
            "high": [100.4, 100.3, 101.0, 101.2, 102.0, 101.8, 100.4],
            "low": [99.7, 99.5, 99.0, 99.8, 99.3, 99.2, 99.4],
            # Bar 3 and bar 4 both close below teeth; only bar 3 is the crossing bar.
            "close": [100.1, 100.0, 100.8, 100.0, 99.9, 100.2, 100.0],
        },
        index=idx,
    )
    jaw = pd.Series(102.0, index=idx)
    teeth = pd.Series(101.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series([0.0, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1], index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)},
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=False,
        red_teeth_profit_protection_enabled=True,
        red_teeth_profit_protection_min_bars=1,
        red_teeth_profit_protection_min_unrealized_return=0.01,
        red_teeth_profit_protection_volatility_lookback=2,
        red_teeth_profit_protection_annualized_volatility_scaler=1e9,
    )
    strategy.generate_signals(data)

    assert strategy.signal_exit_reason is not None
    # No fresh cross after the strict min-bars gate is satisfied, so red-teeth PP should not fire.
    assert str(strategy.signal_exit_reason.iloc[5]) != "Red Gator Teeth PP"


def test_bw_red_teeth_profit_protection_min_unrealized_gate_is_not_latched(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-09-01", periods=7, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 101.0, 100.2, 100.0, 99.8],
            # Bar 4 reaches >1% favorable excursion, but bar 6 (the close-cross bar) does not.
            "high": [100.3, 100.2, 101.2, 101.6, 101.6, 100.4, 100.8],
            "low": [99.7, 99.8, 99.2, 100.6, 99.9, 99.7, 99.4],
            "close": [100.0, 100.1, 100.9, 101.1, 100.6, 100.4, 99.9],
        },
        index=idx,
    )
    jaw = pd.Series(102.0, index=idx)
    teeth = pd.Series([101.0, 101.0, 101.0, 100.2, 100.0, 100.0, 100.0], index=idx, dtype="float64")
    lips = pd.Series(99.8, index=idx)
    ao = pd.Series([0.0, 1.0, 0.6, 0.4, 0.3, 0.2, 0.1], index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)},
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=False,
        red_teeth_profit_protection_enabled=True,
        red_teeth_profit_protection_min_bars=1,
        red_teeth_profit_protection_min_unrealized_return=0.01,
        red_teeth_profit_protection_volatility_lookback=2,
        red_teeth_profit_protection_annualized_volatility_scaler=1e9,
    )
    signals = strategy.generate_signals(data)

    assert strategy.signal_contracts is not None
    assert strategy.signal_exit_reason is not None
    # Position stays open on the cross bar because favorable excursion there is < min-unrealized threshold.
    assert int(signals.iloc[6]) == 1
    assert float(strategy.signal_contracts.iloc[6]) > 0.0
    assert str(strategy.signal_exit_reason.iloc[6]) != "Red Gator Teeth PP"


def test_bw_red_teeth_profit_protection_gator_direction_alignment_is_latched_until_flat(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-09-01", periods=6, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.5, 100.4, 100.0],
            "high": [100.4, 100.3, 101.0, 101.2, 103.0, 102.0],
            "low": [99.7, 99.5, 99.0, 99.8, 99.2, 99.6],
            "close": [100.1, 100.0, 100.8, 101.0, 101.1, 99.8],
        },
        index=idx,
    )
    jaw = pd.Series([102.0, 102.0, 102.0, 99.0, 99.0, 99.0], index=idx, dtype="float64")
    teeth = pd.Series(101.0, index=idx)
    lips = pd.Series([100.0, 100.0, 100.0, 103.0, 100.0, 100.0], index=idx, dtype="float64")
    ao = pd.Series([0.0, 1.0, 0.5, 0.4, 0.3, 0.2], index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)},
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=False,
        red_teeth_profit_protection_enabled=True,
        red_teeth_profit_protection_min_bars=1,
        red_teeth_profit_protection_min_unrealized_return=0.01,
        red_teeth_profit_protection_volatility_lookback=2,
        red_teeth_profit_protection_annualized_volatility_scaler=1e9,
        red_teeth_profit_protection_require_gator_direction_alignment=True,
    )
    signals = strategy.generate_signals(data)

    assert strategy.signal_contracts is not None
    assert strategy.signal_exit_reason is not None
    assert int(signals.iloc[5]) == 0
    assert float(strategy.signal_contracts.iloc[5]) == pytest.approx(0.0)
    assert str(strategy.signal_exit_reason.iloc[5]) == "Red Gator Teeth PP"


def test_bw_ntd_cancels_older_pending_fractal_in_favor_of_newest(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-07-01", periods=9, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 105.0, 101.0, 110.0, 101.0, 106.0, 106.0, 106.0],
            "low": [99.5, 99.0, 100.0, 99.8, 99.5, 99.5, 99.8, 99.8, 99.8],
            "close": [100.0, 100.0, 100.1, 100.0, 100.2, 100.0, 100.1, 100.1, 100.1],
        },
        index=idx,
    )
    jaw = pd.Series(100.0, index=idx)
    teeth = pd.Series(100.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series(np.nan, index=idx)
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, False, True, False, True, False, False, False, False],
            "down_fractal": [False, True, False, False, False, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=True,
        ntd_sleeping_gator_lookback=2,
        ntd_sleeping_gator_tightness_mult=1.0,
        ntd_ranging_lookback=3,
        ntd_ranging_max_span_pct=0.5,
    )
    signals = strategy.generate_signals(data)
    assert int((signals != 0).sum()) == 0


def test_bw_ntd_and_1w_setup_can_both_register_on_same_bar(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-08-01", periods=8, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [101.0, 101.0, 101.0, 101.0, 101.0, 101.2, 101.0, 101.1],
            "high": [101.8, 101.6, 103.0, 101.5, 101.4, 101.6, 103.2, 103.0],
            "low": [100.8, 100.6, 101.0, 100.9, 100.8, 100.9, 100.6, 100.8],
            "close": [101.1, 101.0, 101.2, 101.1, 101.0, 101.3, 101.4, 101.2],
        },
        index=idx,
    )
    jaw = pd.Series(104.0, index=idx)
    teeth = pd.Series(102.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series([2.0, 1.8, 1.7, 1.6, 1.4, 1.2, 1.0, 0.9], index=idx)
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, False, True, False, False, False, False, False],
            "down_fractal": [False, True, False, False, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=True,
        ntd_sleeping_gator_lookback=2,
        ntd_sleeping_gator_tightness_mult=1.0,
        ntd_ranging_lookback=3,
        ntd_ranging_max_span_pct=0.2,
    )
    signals = strategy.generate_signals(data)
    setup_side = strategy.signal_first_wiseman_setup_side
    assert setup_side is not None
    assert int(signals.iloc[6]) == 1
    assert int(setup_side.iloc[6]) == 1


def test_bw_same_bar_short_fractal_and_bullish_1w_setup_can_later_flip_long(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-08-15", periods=10, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [101.2, 101.4, 101.1, 101.0, 100.9, 100.8, 101.0, 100.7, 101.3, 101.4],
            "high": [101.8, 103.0, 101.6, 101.3, 101.2, 101.1, 101.8, 101.0, 102.0, 102.1],
            "low": [100.9, 101.0, 100.5, 100.8, 100.7, 100.5, 100.4, 100.6, 101.0, 101.2],
            "close": [101.3, 102.8, 100.8, 101.0, 100.9, 100.7, 101.6, 100.8, 101.8, 101.9],
        },
        index=idx,
    )
    jaw = pd.Series(104.0, index=idx)
    teeth = pd.Series(102.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series([1.5, 1.4, 1.2, 1.1, 1.0, 0.8, 0.6, 0.55, 0.5, 0.45], index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, True, False, False, False, False, False, False, False, False],
            "down_fractal": [False, False, True, False, False, False, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=True,
        ntd_sleeping_gator_lookback=2,
        ntd_sleeping_gator_tightness_mult=1.0,
        ntd_ranging_lookback=3,
        ntd_ranging_max_span_pct=0.2,
    )
    signals = strategy.generate_signals(data)
    fill_prices = strategy.signal_fill_prices
    assert fill_prices is not None

    assert int(signals.iloc[6]) == -1
    assert int(signals.iloc[8]) == 1
    assert float(fill_prices.iloc[8]) == pytest.approx(101.8)


def test_bw_same_bar_long_fractal_and_bearish_1w_setup_can_later_flip_short(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-08-20", periods=10, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [102.4, 100.6, 102.2, 102.1, 102.0, 102.2, 102.8, 102.7, 101.9, 101.8],
            "high": [102.8, 100.8, 103.0, 102.6, 102.5, 102.7, 104.0, 103.0, 102.0, 101.9],
            "low": [101.8, 100.0, 102.0, 101.9, 101.8, 102.0, 102.2, 102.1, 101.2, 101.0],
            "close": [102.2, 100.2, 102.6, 102.2, 102.1, 102.5, 102.4, 102.3, 101.3, 101.1],
        },
        index=idx,
    )
    jaw = pd.Series(100.0, index=idx)
    teeth = pd.Series(102.0, index=idx)
    lips = pd.Series(104.0, index=idx)
    ao = pd.Series([0.6, 0.8, 1.0, 1.1, 1.2, 1.3, 1.6, 1.5, 1.4, 1.3], index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, False, True, False, False, False, False, False, False, False],
            "down_fractal": [False, True, False, False, False, False, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=True,
        ntd_sleeping_gator_lookback=2,
        ntd_sleeping_gator_tightness_mult=1.0,
        ntd_ranging_lookback=3,
        ntd_ranging_max_span_pct=0.2,
    )
    signals = strategy.generate_signals(data)
    fill_prices = strategy.signal_fill_prices
    assert fill_prices is not None

    assert int(signals.iloc[6]) == 1
    assert int(signals.iloc[7]) == -1
    assert float(fill_prices.iloc[7]) == pytest.approx(102.2)


def test_bw_same_bar_opposite_1w_setup_can_override_later_opposite_fractal_when_it_triggers(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-09-01", periods=9, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [101.8, 101.9, 102.0, 101.9, 101.8, 101.9, 101.7, 102.2, 102.3],
            "high": [102.4, 102.5, 103.0, 102.4, 102.3, 102.2, 102.8, 103.1, 103.0],
            "low": [101.4, 101.5, 101.8, 101.5, 101.0, 101.0, 100.8, 101.9, 102.0],
            "close": [101.9, 102.0, 102.2, 101.9, 101.7, 101.8, 102.0, 102.9, 102.8],
        },
        index=idx,
    )
    jaw = pd.Series(104.0, index=idx)
    teeth = pd.Series(102.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, 1.4, 1.2, 1.3, 1.2], index=idx)
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, False, True, False, False, False, False, False, False],
            "down_fractal": [False, True, False, True, False, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=True,
        ntd_sleeping_gator_lookback=2,
        ntd_sleeping_gator_tightness_mult=1.0,
        ntd_ranging_lookback=3,
        ntd_ranging_max_span_pct=0.2,
    )
    result = BacktestEngine().run(data, strategy)
    reasons = [str(event.strategy_reason) for event in result.execution_events]
    assert reasons[:3] == [
        "Bearish Fractal",
        "Signal Intent Flip to Bullish 1W",
        "Bullish 1W",
    ]


def test_bw_same_bar_bullish_fractal_and_bearish_1w_setup_emits_flip_when_setup_later_triggers(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-10-01", periods=9, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [102.2, 102.1, 102.0, 102.1, 102.2, 102.1, 102.3, 101.8, 101.7],
            "high": [102.7, 102.6, 103.2, 102.8, 102.7, 102.6, 103.3, 102.0, 101.9],
            "low": [101.8, 101.7, 102.1, 101.8, 101.6, 101.7, 101.9, 101.0, 100.9],
            "close": [102.1, 102.0, 102.2, 102.0, 102.1, 102.0, 102.1, 101.1, 101.0],
        },
        index=idx,
    )
    jaw = pd.Series(100.0, index=idx)
    teeth = pd.Series(102.0, index=idx)
    lips = pd.Series(104.0, index=idx)
    ao = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, 1.4, 1.6, 1.5, 1.4], index=idx)
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, False, True, True, False, False, False, False, False],
            "down_fractal": [False, True, False, False, False, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=True,
        ntd_sleeping_gator_lookback=2,
        ntd_sleeping_gator_tightness_mult=1.0,
        ntd_ranging_lookback=3,
        ntd_ranging_max_span_pct=0.2,
    )
    result = BacktestEngine().run(data, strategy)
    reasons = [str(event.strategy_reason) for event in result.execution_events]
    assert reasons[0] == "Bullish Fractal"
    assert "Signal Intent Flip to Bearish 1W" in reasons
    assert "Bearish 1W" in reasons


def test_bw_bearish_weaker_same_side_setup_does_not_replace_pending_trigger(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2025-01-01", periods=8, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.2, 100.4, 100.6, 100.8, 104.0, 101.0, 103.8, 101.2],
            "high": [100.5, 100.7, 100.9, 101.1, 105.0, 103.0, 104.5, 101.6],
            "low": [99.9, 100.1, 100.2, 100.4, 100.0, 100.6, 99.6, 100.8],
            "close": [100.3, 100.5, 100.7, 100.9, 101.5, 100.8, 101.0, 101.0],
        },
        index=idx,
    )
    jaw = pd.Series(100.0, index=idx)
    teeth = pd.Series(101.0, index=idx)
    lips = pd.Series(102.0, index=idx)
    ao = pd.Series([0.1, 0.2, 0.3, 0.4, 1.0, 0.7, 1.2, 1.0], index=idx, dtype="float64")
    fractals = pd.DataFrame({"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)}, index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(divergence_filter_bars=0, ntd_initial_fractal_enabled=False)
    signals = strategy.generate_signals(data)
    fill_prices = strategy.signal_fill_prices
    ignored = strategy.signal_first_wiseman_ignored_reason
    assert fill_prices is not None
    assert ignored is not None

    assert int(signals.iloc[6]) == -1
    assert float(fill_prices.iloc[6]) == pytest.approx(100.0)
    assert str(ignored.iloc[6]) == "weaker_than_active_setup"


def test_bw_bullish_weaker_same_side_setup_does_not_replace_pending_trigger(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2025-02-01", periods=8, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.8, 100.6, 100.4, 100.2, 96.0, 99.0, 96.2, 99.4],
            "high": [101.1, 100.9, 100.7, 100.5, 100.0, 99.5, 100.4, 99.8],
            "low": [100.5, 100.3, 100.1, 99.9, 95.0, 96.0, 95.4, 99.0],
            "close": [100.7, 100.5, 100.3, 100.1, 99.0, 99.2, 99.8, 99.6],
        },
        index=idx,
    )
    jaw = pd.Series(102.0, index=idx)
    teeth = pd.Series(101.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series([1.0, 0.9, 0.8, 0.7, -1.0, -0.7, -1.2, -1.0], index=idx, dtype="float64")
    fractals = pd.DataFrame({"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)}, index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(divergence_filter_bars=0, ntd_initial_fractal_enabled=False)
    signals = strategy.generate_signals(data)
    fill_prices = strategy.signal_fill_prices
    ignored = strategy.signal_first_wiseman_ignored_reason
    assert fill_prices is not None
    assert ignored is not None

    assert int(signals.iloc[6]) == 1
    assert float(fill_prices.iloc[6]) == pytest.approx(100.0)
    assert str(ignored.iloc[6]) == "weaker_than_active_setup"


def test_bw_long_tracks_highest_high_across_multiple_bearish_opposite_1w_setups(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2025-03-01", periods=11, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 99.8, 96.0, 99.0, 109.0, 103.2, 108.8, 102.2, 107.8, 104.5, 104.4],
            "high": [100.4, 100.2, 100.0, 101.0, 110.0, 103.5, 109.0, 102.5, 108.0, 105.0, 104.8],
            "low": [99.6, 99.4, 95.0, 98.8, 104.0, 104.2, 105.0, 104.3, 105.5, 103.9, 104.0],
            "close": [99.9, 99.7, 99.0, 100.5, 106.0, 103.1, 106.2, 102.0, 106.1, 104.2, 104.2],
        },
        index=idx,
    )
    jaw = pd.Series([102.0, 102.0, 102.0, 102.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0], index=idx)
    teeth = pd.Series([101.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0], index=idx)
    lips = pd.Series([100.0, 100.0, 100.0, 100.0, 102.0, 102.0, 102.0, 102.0, 102.0, 102.0, 102.0], index=idx)
    ao = pd.Series([0.0, -0.3, -1.0, -0.8, 1.0, 0.6, 1.2, 0.5, 1.3, 0.4, 0.3], index=idx, dtype="float64")
    fractals = pd.DataFrame({"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)}, index=idx)
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(divergence_filter_bars=0, ntd_initial_fractal_enabled=False)
    signals = strategy.generate_signals(data)
    fill_prices = strategy.signal_fill_prices
    ignored = strategy.signal_first_wiseman_ignored_reason
    assert fill_prices is not None
    assert ignored is not None

    assert int(signals.iloc[3]) == 1
    assert int(signals.iloc[9]) == -1
    assert float(fill_prices.iloc[9]) == pytest.approx(104.0)
    assert str(ignored.iloc[6]) == "weaker_than_active_setup"
    assert str(ignored.iloc[8]) == "weaker_than_active_setup"


def test_bw_ntd_entry_allows_waking_gator_condition(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-11-01", periods=8, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.1, 100.0, 100.0, 100.0, 100.1, 100.2],
            "high": [100.3, 100.2, 101.0, 100.2, 100.1, 100.1, 101.2, 101.3],
            "low": [99.8, 99.7, 100.0, 99.8, 99.9, 99.8, 100.0, 100.1],
            "close": [100.0, 100.0, 100.2, 100.0, 100.0, 100.0, 101.1, 101.2],
        },
        index=idx,
    )
    jaw = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.1, 100.2], index=idx)
    teeth = pd.Series([100.0, 100.0, 100.2, 100.0, 100.0, 100.0, 100.1, 100.2], index=idx)
    lips = pd.Series([100.0, 100.0, 100.4, 100.0, 100.0, 100.0, 100.1, 100.2], index=idx)
    ao = pd.Series(np.nan, index=idx)
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, False, True, False, False, False, False, False],
            "down_fractal": [False, True, False, False, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=True,
        ntd_sleeping_gator_lookback=2,
        ntd_sleeping_gator_tightness_mult=1.0,
        ntd_ranging_lookback=3,
        ntd_ranging_max_span_pct=0.2,
    )
    signals = strategy.generate_signals(data)
    assert int(signals.iloc[6]) == 1


def test_bw_fractal_position_does_not_emit_1w_reversal_labels_without_1w_setup(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-12-01", periods=9, freq="W", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.2, 8.2, 10.0, 10.1, 10.0, 9.8, 9.9, 10.0],
            "high": [10.5, 10.7, 11.0, 11.2, 10.8, 10.6, 10.4, 11.3, 10.4],
            "low": [9.0, 10.0, 8.0, 9.0, 9.5, 9.4, 7.9, 9.6, 9.7],
            "close": [10.1, 10.3, 10.4, 10.8, 10.2, 10.1, 8.4, 11.0, 10.1],
        },
        index=idx,
    )
    jaw = pd.Series(12.0, index=idx)
    teeth = pd.Series(10.2, index=idx)
    lips = pd.Series(10.0, index=idx)
    ao = pd.Series(np.nan, index=idx)
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, False, False, False, True, False, False, False, False],
            "down_fractal": [False, False, True, False, False, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    result = BacktestEngine().run(
        data,
        BWStrategy(
            divergence_filter_bars=4,
            ntd_initial_fractal_enabled=True,
            ntd_sleeping_gator_lookback=2,
            ntd_sleeping_gator_tightness_mult=1.0,
            ntd_ranging_lookback=3,
            ntd_ranging_max_span_pct=1.0,
        ),
    )
    reasons = [str(event.strategy_reason) for event in result.execution_events]
    assert any("Fractal" in reason for reason in reasons)
    assert not any("1W-R" in reason for reason in reasons)


def test_bw_fractal_short_stop_advances_to_latest_opposite_fractal_or_interim_high(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=12, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.1, 10.0, 10.1, 10.8, 10.7, 9.8, 9.7, 9.6, 9.5, 9.4],
            "high": [10.2, 10.3, 11.0, 10.4, 10.5, 11.2, 10.9, 9.9, 10.6, 9.8, 9.7, 9.6],
            "low": [9.9, 9.8, 10.0, 9.5, 10.0, 10.7, 10.4, 9.4, 9.3, 9.2, 9.1, 9.0],
            "close": [10.0, 10.1, 10.6, 10.0, 10.2, 11.1, 10.6, 9.6, 9.5, 9.4, 9.3, 9.2],
        },
        index=idx,
    )
    jaw = pd.Series(10.0, index=idx)
    teeth = pd.Series(10.0, index=idx)
    lips = pd.Series(10.0, index=idx)
    ao = pd.Series(np.nan, index=idx)
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, False, True, False, False, False, False, False, True, False, False, False],
            "down_fractal": [False, False, False, True, False, False, False, False, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=True,
        ntd_sleeping_gator_lookback=2,
        ntd_sleeping_gator_tightness_mult=1.0,
        ntd_ranging_lookback=3,
        ntd_ranging_max_span_pct=0.2,
    )
    signals = strategy.generate_signals(data)

    assert strategy.signal_stop_loss_prices is not None
    assert int(signals.iloc[7]) == -1
    assert strategy.signal_stop_loss_prices.iloc[7] == pytest.approx(11.2)
    assert strategy.signal_stop_loss_prices.iloc[9] == pytest.approx(11.2)
    assert strategy.signal_stop_loss_prices.iloc[10] == pytest.approx(10.6)
    assert strategy.signal_stop_loss_prices.iloc[11] == pytest.approx(10.6)


def test_bw_fractal_short_stop_requires_opposite_fractal_above_teeth(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-02-01", periods=8, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.1, 10.0, 10.0, 9.9, 9.8, 9.7],
            "high": [10.3, 10.4, 10.3, 10.2, 10.1, 10.0, 9.9, 9.8],
            "low": [9.9, 9.8, 9.5, 9.6, 9.4, 9.3, 9.2, 9.1],
            "close": [10.0, 10.1, 9.8, 9.9, 9.7, 9.6, 9.5, 9.4],
        },
        index=idx,
    )
    jaw = pd.Series(10.7, index=idx)
    teeth = pd.Series(10.5, index=idx)
    lips = pd.Series(10.3, index=idx)
    ao = pd.Series(np.nan, index=idx)
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, True, False, False, False, False, False, False],
            "down_fractal": [False, False, True, False, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=True,
        ntd_sleeping_gator_lookback=2,
        ntd_sleeping_gator_tightness_mult=1.0,
        ntd_ranging_lookback=3,
        ntd_ranging_max_span_pct=0.2,
    )
    signals = strategy.generate_signals(data)

    assert strategy.signal_fill_prices is not None
    assert strategy.signal_stop_loss_prices is not None
    # Up fractal at bar 1 is below teeth (10.5), so there is no valid opposite
    # fractal anchor for a short stop and the short entry must not be armed/filled.
    assert int(signals.iloc[5]) == 0
    assert pd.isna(strategy.signal_fill_prices.iloc[5])
    assert pd.isna(strategy.signal_stop_loss_prices.iloc[5])


def test_bw_fractal_short_stop_updates_on_new_above_teeth_fractal_even_when_gator_is_open(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-03-01", periods=12, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.1, 10.0, 10.1, 10.8, 10.7, 9.8, 9.7, 9.6, 9.5, 9.4],
            "high": [10.2, 10.3, 11.0, 10.4, 10.5, 11.2, 10.9, 9.9, 10.6, 9.8, 9.7, 9.6],
            "low": [9.9, 9.8, 10.0, 9.5, 10.0, 10.7, 10.4, 9.4, 9.3, 9.2, 9.1, 9.0],
            "close": [10.0, 10.1, 10.6, 10.0, 10.2, 11.1, 10.6, 9.6, 9.5, 9.4, 9.3, 9.2],
        },
        index=idx,
    )
    jaw = pd.Series([10.0] * 7 + [9.2, 9.1, 9.0, 8.9, 8.8], index=idx)
    teeth = pd.Series([10.0] * 7 + [10.0, 10.0, 10.0, 10.0, 10.0], index=idx)
    lips = pd.Series([10.0] * 7 + [10.9, 11.0, 11.1, 11.2, 11.3], index=idx)
    ao = pd.Series(np.nan, index=idx)
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, False, True, False, False, False, False, False, True, False, False, False],
            "down_fractal": [False, False, False, True, False, False, False, False, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=True,
        ntd_sleeping_gator_lookback=2,
        ntd_sleeping_gator_tightness_mult=1.0,
        ntd_ranging_lookback=3,
        ntd_ranging_max_span_pct=0.2,
    )
    signals = strategy.generate_signals(data)

    assert strategy.signal_stop_loss_prices is not None
    assert int(signals.iloc[7]) == -1
    assert strategy.signal_stop_loss_prices.iloc[7] == pytest.approx(11.2)
    # A newer qualifying up fractal above teeth is confirmed later and should update
    # the short stop even though gator lines are now widely open.
    assert strategy.signal_stop_loss_prices.iloc[10] == pytest.approx(10.6)


def test_bw_fractal_short_position_reverses_when_updated_fractal_stop_is_hit(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-04-01", periods=12, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.1, 10.0, 10.1, 10.8, 10.7, 9.8, 9.7, 9.6, 9.5, 9.6],
            "high": [10.2, 10.3, 11.0, 10.4, 10.5, 11.2, 10.9, 9.9, 10.6, 9.8, 9.7, 10.7],
            "low": [9.9, 9.8, 10.0, 9.5, 10.0, 10.7, 10.4, 9.4, 9.3, 9.2, 9.1, 9.2],
            "close": [10.0, 10.1, 10.6, 10.0, 10.2, 11.1, 10.6, 9.6, 9.5, 9.4, 9.3, 10.3],
        },
        index=idx,
    )
    jaw = pd.Series([10.0] * len(idx), index=idx)
    teeth = pd.Series([10.0] * len(idx), index=idx)
    lips = pd.Series([10.0] * len(idx), index=idx)
    ao = pd.Series(np.nan, index=idx)
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, False, True, False, False, False, False, False, True, False, False, False],
            "down_fractal": [False, False, False, True, False, False, False, False, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=True,
        ntd_sleeping_gator_lookback=2,
        ntd_sleeping_gator_tightness_mult=1.0,
        ntd_ranging_lookback=3,
        ntd_ranging_max_span_pct=0.2,
    )
    signals = strategy.generate_signals(data)

    assert strategy.signal_fill_prices is not None
    assert strategy.signal_stop_loss_prices is not None
    assert int(signals.iloc[7]) == -1
    # Stop is hit at 10.6 and should no longer flatten the strategy.
    assert np.isfinite(float(strategy.signal_fill_prices.iloc[11]))
    assert int(signals.iloc[11]) != 0
    assert strategy.signal_stop_loss_prices.iloc[10] == pytest.approx(10.6)


def test_bw_1w_reversal_does_not_stop_out_on_entry_bar_when_reversal_stop_equals_bar_extreme(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-06-01", periods=7, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [101.0, 101.2, 101.1, 101.2, 101.1, 101.0, 100.6],
            "high": [102.0, 102.1, 103.5, 102.5, 104.0, 110.0, 109.0],
            "low": [100.6, 100.7, 100.0, 100.8, 100.9, 99.8, 99.7],
            "close": [101.1, 101.3, 104.6, 101.5, 101.4, 100.2, 100.1],
        },
        index=idx,
    )
    jaw = pd.Series(104.0, index=idx)
    teeth = pd.Series(102.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series([1.2, 1.1, 0.9, 0.8, 0.7, 0.6, 0.5], index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)},
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(divergence_filter_bars=0, ntd_initial_fractal_enabled=False)
    signals = strategy.generate_signals(data)

    assert strategy.signal_fill_prices is not None
    assert strategy.signal_first_wiseman_reversal_side is not None
    assert int(strategy.signal_first_wiseman_reversal_side.iloc[5]) == -1
    assert int(signals.iloc[5]) == -1
    assert strategy.signal_fill_prices.iloc[5] == pytest.approx(100.0)
    # The reversal stop equals the entry-bar extreme (110.0), but should only be
    # enforceable from the following bar onward.
    assert int(signals.iloc[6]) == -1


def test_bw_uses_configured_contract_sizes_for_1w_and_ntd_initial_entries(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.5, 100.2, 99.6, 99.8, 100.0],
            "high": [100.4, 100.3, 101.0, 101.2, 100.4, 100.0, 100.2, 100.3],
            "low": [99.7, 99.5, 99.0, 99.8, 98.8, 99.1, 99.2, 99.4],
            "close": [100.1, 100.0, 100.8, 101.0, 99.2, 99.5, 99.9, 100.1],
        },
        index=idx,
    )
    jaw = pd.Series(102.0, index=idx)
    teeth = pd.Series(101.0, index=idx)
    lips = pd.Series(100.0, index=idx)
    ao = pd.Series([0.0, 1.0, 0.5, 0.4, 0.3, 0.2, np.nan, np.nan], index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, False, False, False, False, False, True, False],
            "down_fractal": [False, False, False, False, False, True, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    first_w = BWStrategy(divergence_filter_bars=0, first_wiseman_contracts=2, ntd_initial_fractal_enabled=False)
    first_w_signals = first_w.generate_signals(data)
    assert first_w.signal_contracts is not None
    assert int(first_w_signals.iloc[3]) == 1
    assert float(first_w.signal_contracts.iloc[3]) == pytest.approx(2.0)

    ntd_idx = pd.date_range("2024-11-01", periods=8, freq="D", tz="UTC")
    ntd_data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.1, 100.0, 100.0, 100.0, 100.1, 100.2],
            "high": [100.3, 100.2, 101.0, 100.2, 100.1, 100.1, 101.2, 101.3],
            "low": [99.8, 99.7, 100.0, 99.8, 99.9, 99.8, 100.0, 100.1],
            "close": [100.0, 100.0, 100.2, 100.0, 100.0, 100.0, 101.1, 101.2],
        },
        index=ntd_idx,
    )
    ntd_jaw = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.1, 100.2], index=ntd_idx)
    ntd_teeth = pd.Series([100.0, 100.0, 100.2, 100.0, 100.0, 100.0, 100.1, 100.2], index=ntd_idx)
    ntd_lips = pd.Series([100.0, 100.0, 100.4, 100.0, 100.0, 100.0, 100.1, 100.2], index=ntd_idx)
    ntd_ao = pd.Series(np.nan, index=ntd_idx)
    ntd_fractals = pd.DataFrame(
        {
            "up_fractal": [False, False, True, False, False, False, False, False],
            "down_fractal": [False, True, False, False, False, False, False, False],
        },
        index=ntd_idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (ntd_jaw, ntd_teeth, ntd_lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ntd_ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: ntd_fractals)

    ntd = BWStrategy(
        divergence_filter_bars=0,
        first_wiseman_contracts=1,
        ntd_initial_fractal_enabled=True,
        ntd_initial_fractal_contracts=4,
        ntd_sleeping_gator_lookback=2,
        ntd_sleeping_gator_tightness_mult=1.0,
        ntd_ranging_lookback=3,
        ntd_ranging_max_span_pct=0.2,
    )
    ntd_signals = ntd.generate_signals(ntd_data)
    assert ntd.signal_contracts is not None
    assert int(ntd_signals.iloc[6]) == 1
    assert float(ntd.signal_contracts.iloc[6]) == pytest.approx(4.0)



def test_bw_ntd_initial_fractal_contracts_zero_disables_initial_fractal_entries(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-03-01", periods=8, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0] * len(idx),
            "high": [100.1, 100.2, 101.0, 100.2, 100.1, 100.2, 101.1, 100.9],
            "low": [99.9, 99.8, 100.0, 99.9, 99.8, 99.9, 100.5, 100.4],
            "close": [100.0, 100.0, 100.6, 100.1, 100.0, 100.1, 100.9, 100.7],
        },
        index=idx,
    )
    jaw = pd.Series([100.0] * len(idx), index=idx)
    teeth = pd.Series([100.0, 100.0, 100.2, 100.0, 100.0, 100.0, 100.1, 100.1], index=idx)
    lips = pd.Series([100.0, 100.0, 100.4, 100.0, 100.0, 100.0, 100.2, 100.2], index=idx)
    ao = pd.Series(np.nan, index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, False, True, False, False, False, False, False],
            "down_fractal": [False, True, False, False, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        first_wiseman_contracts=0,
        ntd_initial_fractal_enabled=True,
        ntd_initial_fractal_contracts=0,
        ntd_sleeping_gator_lookback=2,
        ntd_sleeping_gator_tightness_mult=1.0,
        ntd_ranging_lookback=3,
        ntd_ranging_max_span_pct=0.2,
        red_teeth_profit_protection_enabled=False,
        green_lips_profit_protection_enabled=False,
    )
    signals = strategy.generate_signals(data)

    assert strategy.signal_contracts is not None
    assert strategy.signal_fill_prices is not None
    assert strategy.signal_fractal_position_side is not None
    assert signals.eq(0).all()
    assert strategy.signal_contracts.eq(0).all()
    assert strategy.signal_fill_prices.isna().all()
    assert strategy.signal_fractal_position_side.eq(0).all()

def test_bw_fractal_add_on_uses_most_recent_qualifying_fractal_and_contract_size(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-02-01", periods=10, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.5, 100.8, 101.0, 101.2, 101.4, 101.6, 101.8],
            "high": [100.2, 100.3, 101.1, 101.4, 102.0, 101.7, 102.1, 101.9, 102.8, 102.2],
            "low": [99.8, 99.7, 99.0, 100.1, 100.5, 100.8, 101.0, 101.2, 101.4, 101.6],
            "close": [100.0, 100.0, 100.9, 101.2, 101.1, 101.3, 101.5, 101.7, 102.3, 101.9],
        },
        index=idx,
    )
    jaw = pd.Series([102.0, 102.0, 102.0, 102.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0], index=idx)
    teeth = pd.Series([101.0, 101.0, 101.0, 101.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0], index=idx)
    lips = pd.Series([100.0, 100.0, 100.0, 100.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0], index=idx)
    ao = pd.Series([0.0, 1.0, 0.5, 0.4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, False, False, False, True, False, True, False, False, False],
            "down_fractal": [False] * len(idx),
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        first_wiseman_contracts=2,
        fractal_add_on_contracts=3,
        ntd_initial_fractal_enabled=False,
    )
    signals = strategy.generate_signals(data)

    assert strategy.signal_contracts is not None
    assert strategy.signal_add_on_fractal_fill_side is not None
    assert int(signals.iloc[3]) == 1
    # Most recent qualifying fractal is bar 6 (price 102.1), not bar 4 (price 102.0).
    assert strategy.signal_fill_prices is not None
    assert strategy.signal_fill_prices.iloc[9] == pytest.approx(102.1)
    assert int(strategy.signal_add_on_fractal_fill_side.iloc[9]) == 1
    assert float(strategy.signal_contracts.iloc[9]) == pytest.approx(5.0)


def test_bw_ntd_initial_fractal_entry_allows_follow_on_fractal_add_on(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-01-01", periods=12, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0] * 12,
            "high": [100.2, 100.2, 101.0, 100.3, 100.2, 100.3, 101.2, 100.8, 101.1, 101.0, 101.4, 101.5],
            "low": [99.8, 99.7, 100.0, 99.9, 99.8, 99.9, 100.4, 100.2, 100.5, 100.5, 100.9, 101.0],
            "close": [100.0, 100.0, 100.5, 100.1, 100.0, 100.1, 101.0, 100.6, 100.9, 100.8, 101.3, 101.4],
        },
        index=idx,
    )
    jaw = pd.Series([100.0] * len(idx), index=idx)
    teeth = pd.Series([100.0, 100.0, 100.2, 100.0, 100.0, 100.0, 100.1, 100.1, 100.2, 100.2, 100.2, 100.2], index=idx)
    lips = pd.Series([100.0, 100.0, 100.4, 100.0, 100.0, 100.0, 100.2, 100.2, 100.3, 100.3, 100.3, 100.3], index=idx)
    ao = pd.Series(np.nan, index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, False, True, False, False, False, False, True, False, False, False, False],
            "down_fractal": [False, True, False, False, False, False, False, False, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        first_wiseman_contracts=1,
        ntd_initial_fractal_enabled=True,
        ntd_initial_fractal_contracts=2,
        fractal_add_on_contracts=1,
        ntd_sleeping_gator_lookback=2,
        ntd_sleeping_gator_tightness_mult=1.0,
        ntd_ranging_lookback=3,
        ntd_ranging_max_span_pct=0.2,
        red_teeth_profit_protection_enabled=False,
        green_lips_profit_protection_enabled=False,
    )
    signals = strategy.generate_signals(data)

    assert strategy.signal_contracts is not None
    assert strategy.signal_fill_prices is not None
    assert strategy.signal_add_on_fractal_fill_side is not None
    assert int(signals.iloc[6]) == 1
    assert float(strategy.signal_contracts.iloc[6]) == pytest.approx(2.0)
    assert int(strategy.signal_add_on_fractal_fill_side.iloc[9]) == 1
    assert strategy.signal_fill_prices.iloc[9] == pytest.approx(100.8)
    assert float(strategy.signal_contracts.iloc[9]) == pytest.approx(3.0)


def test_bw_pending_opposite_1w_does_not_block_same_bar_long_add_on_and_activation_flattens_add_on(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-02-01", periods=10, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.5, 100.8, 101.0, 101.2, 101.4, 101.6, 101.8],
            "high": [100.2, 100.3, 101.1, 101.4, 102.0, 101.7, 101.8, 101.9, 102.8, 102.6],
            "low": [99.8, 99.7, 99.0, 100.1, 100.5, 100.8, 101.0, 101.2, 101.4, 101.3],
            "close": [100.0, 100.0, 100.9, 101.2, 101.1, 101.3, 101.5, 101.7, 101.2, 101.9],
        },
        index=idx,
    )
    jaw = pd.Series([102.0, 102.0, 102.0, 102.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0], index=idx)
    teeth = pd.Series([101.0, 101.0, 101.0, 101.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0], index=idx)
    lips = pd.Series([100.0, 100.0, 100.0, 100.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0], index=idx)
    ao = pd.Series([0.0, 1.0, 0.5, 0.4, np.nan, np.nan, np.nan, 0.1, 0.2, np.nan], index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {
            "up_fractal": [False, False, False, False, True, False, False, False, False, False],
            "down_fractal": [False] * len(idx),
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        first_wiseman_contracts=2,
        fractal_add_on_contracts=3,
        ntd_initial_fractal_enabled=False,
        red_teeth_profit_protection_enabled=False,
    )
    signals = strategy.generate_signals(data)

    assert strategy.signal_contracts is not None
    assert strategy.signal_add_on_fractal_fill_side is not None
    assert strategy.signal_fill_prices is not None
    assert strategy.signal_first_wiseman_setup_side is not None
    assert int(signals.iloc[3]) == 1
    # Add-on fills on bar 8 even though a bearish 1W setup also forms on bar 8.
    assert strategy.signal_fill_prices.iloc[8] == pytest.approx(102.0)
    assert int(strategy.signal_add_on_fractal_fill_side.iloc[8]) == 1
    assert int(strategy.signal_first_wiseman_setup_side.iloc[8]) == -1
    assert float(strategy.signal_contracts.iloc[8]) == pytest.approx(5.0)
    # Next bar activates the pending bearish 1W and should flatten prior add-on exposure.
    assert int(signals.iloc[9]) == -1
    assert float(strategy.signal_contracts.iloc[9]) == pytest.approx(-2.0)
    assert int(strategy.signal_add_on_fractal_fill_side.iloc[9]) == 0


def test_bw_pending_opposite_1w_does_not_block_same_bar_short_add_on_and_activation_flattens_add_on(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-03-01", periods=10, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 99.5, 99.2, 99.0, 98.8, 98.6, 98.4, 98.2],
            "high": [100.2, 100.3, 101.0, 99.9, 99.5, 99.2, 99.0, 98.8, 98.6, 98.7],
            "low": [99.8, 99.7, 98.9, 98.6, 98.0, 98.3, 98.2, 98.1, 97.2, 97.4],
            "close": [100.0, 100.0, 99.1, 98.8, 98.9, 98.7, 98.5, 98.3, 98.8, 98.1],
        },
        index=idx,
    )
    jaw = pd.Series([98.0, 98.0, 98.0, 98.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0], index=idx)
    teeth = pd.Series([99.0, 99.0, 99.0, 99.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0], index=idx)
    lips = pd.Series([100.0, 100.0, 100.0, 100.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0], index=idx)
    ao = pd.Series([0.0, -1.0, -0.5, -0.4, np.nan, np.nan, np.nan, -0.1, -0.2, np.nan], index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {
            "up_fractal": [False] * len(idx),
            "down_fractal": [False, False, False, False, True, False, False, False, False, False],
        },
        index=idx,
    )
    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        first_wiseman_contracts=2,
        fractal_add_on_contracts=3,
        ntd_initial_fractal_enabled=False,
        red_teeth_profit_protection_enabled=False,
    )
    signals = strategy.generate_signals(data)

    assert strategy.signal_contracts is not None
    assert strategy.signal_add_on_fractal_fill_side is not None
    assert strategy.signal_fill_prices is not None
    assert strategy.signal_first_wiseman_setup_side is not None
    assert int(signals.iloc[3]) == -1
    # Add-on fills on bar 8 even though a bullish 1W setup also forms on bar 8.
    assert strategy.signal_fill_prices.iloc[8] == pytest.approx(98.0)
    assert int(strategy.signal_add_on_fractal_fill_side.iloc[8]) == -1
    assert int(strategy.signal_first_wiseman_setup_side.iloc[8]) == 1
    assert float(strategy.signal_contracts.iloc[8]) == pytest.approx(-5.0)
    # Next bar activates the pending bullish 1W and should flatten prior add-on exposure.
    assert int(signals.iloc[9]) == 1
    assert float(strategy.signal_contracts.iloc[9]) == pytest.approx(2.0)
    assert int(strategy.signal_add_on_fractal_fill_side.iloc[9]) == 0


def test_bw_ntd_short_stop_and_reverse_fills_at_fractal_high_not_breaker_bar_high(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-06-01", periods=9, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [1000.0] * 9,
            "high": [1002.0, 1005.0, 999.0, 1000.0, 995.0, 1000.0, 1010.0, 1001.0, 1001.0],
            "low": [998.0, 1001.0, 990.0, 997.0, 988.0, 989.0, 999.0, 999.0, 999.0],
            "close": [1000.0, 1004.0, 992.0, 999.0, 990.0, 990.0, 1008.0, 1000.0, 1000.0],
        },
        index=idx,
    )
    jaw = pd.Series(999.0, index=idx)
    teeth = pd.Series(999.0, index=idx)
    lips = pd.Series(999.0, index=idx)
    ao = pd.Series(0.0, index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)},
        index=idx,
    )
    fractals.loc[idx[1], "up_fractal"] = True    # initial short stop anchor (1005)
    fractals.loc[idx[2], "down_fractal"] = True  # short trigger (990)
    fractals.loc[idx[3], "up_fractal"] = True    # reversal long trigger (1000)

    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=True,
        ntd_ranging_lookback=1,
        ntd_ranging_max_span_pct=1.0,
    )
    signals = strategy.generate_signals(data)

    assert strategy.signal_fill_prices is not None
    assert strategy.signal_stop_loss_prices is not None
    # Reverse must happen on the same stop bar at the exact stop price.
    assert strategy.signal_fill_prices.iloc[5] == pytest.approx(1000.0)




def test_bw_ntd_reversal_executes_on_stop_bar_at_stop_price_in_engine(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-06-01", periods=9, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [1000.0] * 9,
            "high": [1002.0, 1005.0, 999.0, 1000.0, 995.0, 1000.0, 1010.0, 1001.0, 1001.0],
            "low": [998.0, 1001.0, 990.0, 997.0, 988.0, 989.0, 999.0, 999.0, 999.0],
            "close": [1000.0, 1004.0, 992.0, 999.0, 990.0, 990.0, 1008.0, 1000.0, 1000.0],
        },
        index=idx,
    )
    jaw = pd.Series(999.0, index=idx)
    teeth = pd.Series(999.0, index=idx)
    lips = pd.Series(999.0, index=idx)
    ao = pd.Series(0.0, index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)},
        index=idx,
    )
    fractals.loc[idx[1], "up_fractal"] = True
    fractals.loc[idx[2], "down_fractal"] = True
    fractals.loc[idx[3], "up_fractal"] = True

    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=True,
        ntd_ranging_lookback=1,
        ntd_ranging_max_span_pct=1.0,
    )
    result = BacktestEngine(BacktestConfig(initial_capital=10_000)).run(data, strategy)

    matching_entries = [
        event
        for event in result.execution_events
        if event.event_type == "entry" and event.time == idx[5] and event.strategy_reason == "Bullish Fractal"
    ]
    assert matching_entries, "expected bullish fractal reversal entry on stop bar"
    assert matching_entries[0].price == pytest.approx(1000.0)

    next_bar_entries = [
        event
        for event in result.execution_events
        if event.event_type == "entry" and event.time == idx[6] and event.strategy_reason == "Bullish Fractal"
    ]
    assert not next_bar_entries


def test_engine_allows_add_on_after_ntd_style_stop_reversal_entry() -> None:
    idx = pd.date_range("2024-06-01", periods=6, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 101.0, 101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 99.0, 99.0, 100.0, 101.0, 102.0],
            "close": [100.0, 100.0, 100.0, 101.5, 102.5, 103.5],
            "volume": [1, 1, 1, 1, 1, 1],
        },
        index=idx,
    )
    # Simulate an NTD-style same-bar stop/reversal into long at bar 2,
    # then a long add-on at bar 4 (contracts 1 -> 2).
    signals = pd.Series([0, 0, 1, 1, 1, 0], index=idx, dtype="int8")
    contracts = pd.Series([0.0, 0.0, 1.0, 1.0, 2.0, 0.0], index=idx, dtype="float64")
    fills = pd.Series([np.nan, np.nan, 100.0, np.nan, 103.0, np.nan], index=idx, dtype="float64")
    stop_losses = pd.Series([np.nan, np.nan, 98.0, 98.0, 99.0, np.nan], index=idx, dtype="float64")
    reversal_side = pd.Series([0, 0, 1, 0, 0, 0], index=idx, dtype="int8")

    result = BacktestEngine(BacktestConfig(fee_rate=0.0, slippage_rate=0.0)).run(
        data,
        _SignalContractsExitReasonStrategy(
            signals,
            contracts,
            fill_prices=fills,
            stop_loss_prices=stop_losses,
            first_wiseman_reversal_side=reversal_side,
            fractal_position_side=pd.Series([0, 0, 1, 1, 1, 0], index=idx, dtype="int8"),
        ),
    )

    add_events = [event for event in result.execution_events if event.event_type == "add"]
    assert len(add_events) == 1
    assert add_events[0].time == idx[4]
    assert add_events[0].side == "buy"
    assert add_events[0].price == pytest.approx(103.0)


def test_bw_ntd_long_stop_and_reverse_fills_at_fractal_low_not_breaker_bar_low(monkeypatch) -> None:
    import backtesting.strategy as strategy_module

    idx = pd.date_range("2024-06-01", periods=9, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [1000.0] * 9,
            "high": [1002.0, 999.0, 1005.0, 1002.0, 1008.0, 1004.0, 1001.0, 1001.0, 1001.0],
            "low": [998.0, 995.0, 1001.0, 1000.0, 1004.0, 1001.0, 990.0, 999.0, 999.0],
            "close": [1000.0, 996.0, 1004.0, 1001.0, 1007.0, 1003.0, 992.0, 1000.0, 1000.0],
        },
        index=idx,
    )
    jaw = pd.Series(1001.0, index=idx)
    teeth = pd.Series(1001.0, index=idx)
    lips = pd.Series(1001.0, index=idx)
    ao = pd.Series(0.0, index=idx, dtype="float64")
    fractals = pd.DataFrame(
        {"up_fractal": [False] * len(idx), "down_fractal": [False] * len(idx)},
        index=idx,
    )
    fractals.loc[idx[1], "down_fractal"] = True  # initial long stop anchor (995)
    fractals.loc[idx[2], "up_fractal"] = True    # long trigger (1005)
    fractals.loc[idx[3], "down_fractal"] = True  # reversal short trigger (1000)

    monkeypatch.setattr(strategy_module, "_alligator_lines", lambda _data: (jaw, teeth, lips))
    monkeypatch.setattr(strategy_module, "_williams_ao", lambda _data: ao)
    monkeypatch.setattr(strategy_module, "detect_williams_fractals", lambda _data: fractals)

    strategy = BWStrategy(
        divergence_filter_bars=0,
        ntd_initial_fractal_enabled=True,
        ntd_ranging_lookback=1,
        ntd_ranging_max_span_pct=1.0,
    )
    signals = strategy.generate_signals(data)

    assert strategy.signal_fill_prices is not None
    assert strategy.signal_stop_loss_prices is not None
    assert int(signals.iloc[6]) == -1
    # Reverse must happen on the same stop bar at the exact stop price.
    assert strategy.signal_fill_prices.iloc[6] == pytest.approx(1000.0)
