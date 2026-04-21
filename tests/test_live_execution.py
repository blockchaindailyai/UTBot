from __future__ import annotations

import pandas as pd
import pytest

from backtesting.live_execution import ExecutionSignal, LiveBar, PaperTradingEngine


@pytest.fixture
def base_bar() -> LiveBar:
    return LiveBar(
        symbol="BTC_USDT",
        timestamp=pd.Timestamp("2024-01-01T00:00:00Z"),
        open=100.0,
        high=105.0,
        low=95.0,
        close=102.0,
        volume=10.0,
    )


def test_market_entry_signal_fills_on_next_bar_open(base_bar: LiveBar) -> None:
    engine = PaperTradingEngine(symbol="BTC_USDT", initial_cash=10_000.0, fee_rate=0.0, slippage_rate=0.0)
    signal = ExecutionSignal(
        symbol="BTC_USDT",
        timestamp=base_bar.timestamp,
        action="enter",
        side="buy",
        order_type="market",
        quantity=2.0,
        stop_loss_price=95.0,
        take_profit_price=110.0,
    )

    engine.submit_signal(signal)
    fills = engine.on_bar(base_bar)
    snapshot = engine.snapshot()

    assert len(fills) == 1
    assert fills[0].reason == "market"
    assert fills[0].price == 100.0
    assert snapshot.position.quantity == 2.0
    assert snapshot.position.average_price == 100.0
    assert snapshot.position.stop_loss_price == 95.0
    assert snapshot.position.take_profit_price == 110.0
    assert snapshot.unrealized_pnl == pytest.approx(4.0)
    assert snapshot.equity == pytest.approx(10_004.0)


def test_market_order_honors_explicit_signal_fill_price(base_bar: LiveBar) -> None:
    engine = PaperTradingEngine(symbol="BTC_USDT", initial_cash=10_000.0, fee_rate=0.0, slippage_rate=0.0)
    signal = ExecutionSignal(
        symbol="BTC_USDT",
        timestamp=base_bar.timestamp,
        action="enter",
        side="buy",
        order_type="market",
        quantity=1.0,
        metadata={"market_fill_price": 102.0},
    )

    engine.submit_signal(signal)
    fills = engine.on_bar(base_bar)

    assert len(fills) == 1
    assert fills[0].reason == "market"
    assert fills[0].price == pytest.approx(102.0)
    assert engine.snapshot().position.average_price == pytest.approx(102.0)


def test_limit_order_waits_until_price_trades_through_limit() -> None:
    engine = PaperTradingEngine(symbol="BTC_USDT", fee_rate=0.0, slippage_rate=0.0)
    signal = ExecutionSignal(
        symbol="BTC_USDT",
        timestamp=pd.Timestamp("2024-01-01T00:00:00Z"),
        action="enter",
        side="buy",
        order_type="limit",
        quantity=1.0,
        limit_price=98.0,
    )
    engine.submit_signal(signal)

    no_fill_bar = LiveBar(
        symbol="BTC_USDT",
        timestamp=pd.Timestamp("2024-01-01T00:01:00Z"),
        open=101.0,
        high=103.0,
        low=99.0,
        close=102.0,
    )
    trigger_bar = LiveBar(
        symbol="BTC_USDT",
        timestamp=pd.Timestamp("2024-01-01T00:02:00Z"),
        open=100.0,
        high=101.0,
        low=97.5,
        close=99.0,
    )

    assert engine.on_bar(no_fill_bar) == []
    fills = engine.on_bar(trigger_bar)

    assert len(fills) == 1
    assert fills[0].price == 98.0
    assert engine.snapshot().position.quantity == 1.0
    assert engine.snapshot().position.average_price == 98.0


def test_reverse_signal_closes_open_position_and_opens_new_side() -> None:
    engine = PaperTradingEngine(symbol="BTC_USDT", fee_rate=0.0, slippage_rate=0.0)
    engine.submit_signal(
        ExecutionSignal(
            symbol="BTC_USDT",
            timestamp=pd.Timestamp("2024-01-01T00:00:00Z"),
            action="enter",
            side="buy",
            order_type="market",
            quantity=1.0,
        )
    )
    engine.on_bar(
        LiveBar(
            symbol="BTC_USDT",
            timestamp=pd.Timestamp("2024-01-01T00:01:00Z"),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
        )
    )

    engine.submit_signal(
        ExecutionSignal(
            symbol="BTC_USDT",
            timestamp=pd.Timestamp("2024-01-01T00:02:00Z"),
            action="reverse",
            side="sell",
            order_type="market",
            quantity=2.0,
        )
    )
    fills = engine.on_bar(
        LiveBar(
            symbol="BTC_USDT",
            timestamp=pd.Timestamp("2024-01-01T00:03:00Z"),
            open=104.0,
            high=105.0,
            low=103.0,
            close=104.0,
        )
    )
    snapshot = engine.snapshot()

    assert [fill.reason for fill in fills] == ["market", "market"]
    assert fills[0].realized_pnl == pytest.approx(4.0)
    assert snapshot.position.quantity == -2.0
    assert snapshot.position.average_price == 104.0
    assert snapshot.realized_pnl == pytest.approx(4.0)
    assert snapshot.unrealized_pnl == pytest.approx(0.0)


def test_market_exit_signal_can_fill_on_bar_close_when_requested() -> None:
    engine = PaperTradingEngine(symbol="BTC_USDT", fee_rate=0.0, slippage_rate=0.0)
    engine.submit_signal(
        ExecutionSignal(
            symbol="BTC_USDT",
            timestamp=pd.Timestamp("2024-01-01T00:00:00Z"),
            action="enter",
            side="buy",
            order_type="market",
            quantity=1.0,
        )
    )
    engine.on_bar(
        LiveBar(
            symbol="BTC_USDT",
            timestamp=pd.Timestamp("2024-01-01T00:01:00Z"),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
        )
    )

    engine.submit_signal(
        ExecutionSignal(
            symbol="BTC_USDT",
            timestamp=pd.Timestamp("2024-01-01T00:02:00Z"),
            action="exit",
            order_type="market",
            quantity=1.0,
            metadata={"fill_on_close": True, "exit_reason": "Strategy Profit Protection Red Gator"},
        )
    )
    fills = engine.on_bar(
        LiveBar(
            symbol="BTC_USDT",
            timestamp=pd.Timestamp("2024-01-01T00:02:00Z"),
            open=103.0,
            high=104.0,
            low=102.0,
            close=103.5,
        )
    )

    assert len(fills) == 1
    assert fills[0].price == 103.5
    assert fills[0].strategy_reason == "Strategy Profit Protection Red Gator"
    assert engine.snapshot().position.quantity == 0.0


def test_duplicate_signal_id_submission_is_idempotent_for_non_cancel_actions(base_bar: LiveBar) -> None:
    engine = PaperTradingEngine(symbol="BTC_USDT", initial_cash=10_000.0, fee_rate=0.0, slippage_rate=0.0)
    signal = ExecutionSignal(
        symbol="BTC_USDT",
        timestamp=base_bar.timestamp,
        signal_id="sig-1",
        action="enter",
        side="buy",
        order_type="market",
        quantity=1.0,
    )

    first_orders = engine.submit_signal(signal)
    second_orders = engine.submit_signal(signal)
    fills = engine.on_bar(base_bar)

    assert len(first_orders) == 1
    assert second_orders == []
    assert len(fills) == 1
    assert engine.snapshot().position.quantity == pytest.approx(1.0)


def test_duplicate_scale_signal_id_does_not_create_dual_threads_of_position_growth() -> None:
    engine = PaperTradingEngine(symbol="BTC_USDT", fee_rate=0.0, slippage_rate=0.0)
    engine.submit_signal(
        ExecutionSignal(
            symbol="BTC_USDT",
            timestamp=pd.Timestamp("2024-01-01T00:00:00Z"),
            signal_id="entry-1",
            action="enter",
            side="buy",
            order_type="market",
            quantity=1.0,
        )
    )
    engine.on_bar(
        LiveBar(
            symbol="BTC_USDT",
            timestamp=pd.Timestamp("2024-01-01T00:00:00Z"),
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
        )
    )

    scale = ExecutionSignal(
        symbol="BTC_USDT",
        timestamp=pd.Timestamp("2024-01-01T01:00:00Z"),
        signal_id="scale-1",
        action="scale",
        side="buy",
        order_type="market",
        quantity=1.0,
    )
    engine.submit_signal(scale)
    engine.submit_signal(scale)
    fills = engine.on_bar(
        LiveBar(
            symbol="BTC_USDT",
            timestamp=pd.Timestamp("2024-01-01T01:00:00Z"),
            open=101.0,
            high=101.0,
            low=101.0,
            close=101.0,
        )
    )

    assert len(fills) == 1
    assert engine.snapshot().position.quantity == pytest.approx(2.0)


def test_protective_stop_is_triggered_from_open_position() -> None:
    engine = PaperTradingEngine(symbol="BTC_USDT", fee_rate=0.0, slippage_rate=0.0)
    engine.submit_signal(
        ExecutionSignal(
            symbol="BTC_USDT",
            timestamp=pd.Timestamp("2024-01-01T00:00:00Z"),
            action="enter",
            side="buy",
            order_type="market",
            quantity=1.0,
            stop_loss_price=96.0,
        )
    )
    engine.on_bar(
        LiveBar(
            symbol="BTC_USDT",
            timestamp=pd.Timestamp("2024-01-01T00:01:00Z"),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
        )
    )

    stop_fills = engine.on_bar(
        LiveBar(
            symbol="BTC_USDT",
            timestamp=pd.Timestamp("2024-01-01T00:02:00Z"),
            open=98.0,
            high=99.0,
            low=95.0,
            close=96.0,
        )
    )
    snapshot = engine.snapshot()

    assert len(stop_fills) == 1
    assert stop_fills[0].reason == "protective_stop"
    assert stop_fills[0].price == 96.0
    assert stop_fills[0].realized_pnl == pytest.approx(-4.0)
    assert snapshot.position.quantity == 0.0
    assert snapshot.realized_pnl == pytest.approx(-4.0)
    assert snapshot.position.stop_loss_price is None
    assert len(snapshot.fills) == 2


def test_max_loss_pct_of_equity_triggers_paper_stop_for_long_position() -> None:
    engine = PaperTradingEngine(
        symbol="BTC_USDT",
        initial_cash=10_000.0,
        fee_rate=0.0,
        slippage_rate=0.0,
        max_loss_pct_of_equity=0.01,
    )
    engine.submit_signal(
        ExecutionSignal(
            symbol="BTC_USDT",
            timestamp=pd.Timestamp("2024-01-01T00:00:00Z"),
            action="enter",
            side="buy",
            order_type="market",
            quantity=10.0,
        )
    )
    engine.on_bar(
        LiveBar(
            symbol="BTC_USDT",
            timestamp=pd.Timestamp("2024-01-01T00:01:00Z"),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
        )
    )

    stop_fills = engine.on_bar(
        LiveBar(
            symbol="BTC_USDT",
            timestamp=pd.Timestamp("2024-01-01T00:02:00Z"),
            open=95.0,
            high=96.0,
            low=89.0,
            close=90.0,
        )
    )

    assert len(stop_fills) == 1
    assert stop_fills[0].reason == "max_loss_stop"
    assert stop_fills[0].price == pytest.approx(90.0)
    assert stop_fills[0].realized_pnl == pytest.approx(-100.0)
    assert engine.snapshot().position.quantity == 0.0


def test_stop_limit_order_turns_into_resting_limit_after_trigger() -> None:
    engine = PaperTradingEngine(symbol="BTC_USDT", fee_rate=0.0, slippage_rate=0.0)
    engine.submit_signal(
        ExecutionSignal(
            symbol="BTC_USDT",
            timestamp=pd.Timestamp("2024-01-01T00:00:00Z"),
            action="enter",
            side="buy",
            order_type="stop_limit",
            quantity=1.0,
            stop_price=105.0,
            limit_price=103.0,
        )
    )

    trigger_only_bar = LiveBar(
        symbol="BTC_USDT",
        timestamp=pd.Timestamp("2024-01-01T00:01:00Z"),
        open=104.0,
        high=106.0,
        low=104.0,
        close=105.0,
    )
    fill_bar = LiveBar(
        symbol="BTC_USDT",
        timestamp=pd.Timestamp("2024-01-01T00:02:00Z"),
        open=103.0,
        high=104.0,
        low=102.5,
        close=103.5,
    )

    assert engine.on_bar(trigger_only_bar) == []
    snapshot_after_trigger = engine.snapshot()
    assert len(snapshot_after_trigger.open_orders) == 1
    assert snapshot_after_trigger.open_orders[0].order_type == "limit"
    assert snapshot_after_trigger.open_orders[0].triggered_at == trigger_only_bar.timestamp

    fills = engine.on_bar(fill_bar)
    assert len(fills) == 1
    assert fills[0].reason == "limit"
    assert fills[0].price == 103.0
    assert engine.snapshot().position.quantity == 1.0
