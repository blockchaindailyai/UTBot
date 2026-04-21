from __future__ import annotations

import importlib.util
import json
import sys
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "run_paper_trading.py"
SPEC = importlib.util.spec_from_file_location("run_paper_trading", MODULE_PATH)
run_paper_trading = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_paper_trading
assert SPEC.loader is not None
SPEC.loader.exec_module(run_paper_trading)


@dataclass
class _FakeCandle:
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class _SequenceClient:
    def __init__(self, snapshots: list[list[_FakeCandle]]) -> None:
        self._snapshots = snapshots
        self._calls = 0

    def fetch_kline(self, symbol: str, interval: str, limit: int = 400):
        idx = min(self._calls, len(self._snapshots) - 1)
        self._calls += 1
        return self._snapshots[idx]


class _TrackingSequenceClient(_SequenceClient):
    def __init__(self, snapshots: list[list[_FakeCandle]]) -> None:
        super().__init__(snapshots)
        self.limits: list[int] = []

    def fetch_kline(self, symbol: str, interval: str, limit: int = 400):
        self.limits.append(limit)
        return super().fetch_kline(symbol, interval, limit=limit)


class _FailThenRecoverClient(_SequenceClient):
    def __init__(self, snapshots: list[list[_FakeCandle]], failures_before_success: int) -> None:
        super().__init__(snapshots)
        self.failures_before_success = failures_before_success
        self.failures = 0

    def fetch_kline(self, symbol: str, interval: str, limit: int = 400):
        if self.failures < self.failures_before_success:
            self.failures += 1
            raise urllib.error.URLError("simulated reset")
        return super().fetch_kline(symbol, interval, limit=limit)


class _RealtimeSignalStrategy:
    execute_on_signal_bar = True

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=data.index, dtype="int8")
        if len(data) >= 3:
            signal.iloc[-1] = 1
        return signal


class _StaleLongStrategy:
    execute_on_signal_bar = True

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=data.index, dtype="int8")
        if len(data) >= 2:
            signal.iloc[-2:] = 1
        return signal


class _EnterThenHoldLongStrategy:
    execute_on_signal_bar = True

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=data.index, dtype="int8")
        if len(data) >= 3:
            signal.iloc[-1] = 1
        if len(data) >= 4:
            signal.iloc[-2:] = 1
        return signal


class _DelayedFlatExitStrategy:
    execute_on_signal_bar = True

    def __init__(self) -> None:
        self.signal_exit_reason: pd.Series | None = None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=data.index, dtype="int8")
        exit_reason = pd.Series("", index=data.index, dtype="object")
        if len(data) >= 3:
            signal.iloc[2] = 1
        if len(data) >= 4:
            signal.iloc[3] = 1
        if len(data) >= 5:
            signal.iloc[3:] = 0
            exit_reason.iloc[3] = "Red Gator PP"
        self.signal_exit_reason = exit_reason
        return signal


class _PreviewThenCloseExitStrategy:
    execute_on_signal_bar = True

    def __init__(self) -> None:
        self.signal_exit_reason: pd.Series | None = None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=data.index, dtype="int8")
        exit_reason = pd.Series("", index=data.index, dtype="object")
        if len(data) >= 3:
            signal.iloc[2] = 1
        if len(data) >= 4:
            signal.iloc[2] = 1
            signal.iloc[3] = 0
            exit_reason.iloc[3] = "Red Gator PP"
        self.signal_exit_reason = exit_reason
        return signal


class _PreviewThenCloseExitStrategyBWRedTeeth:
    execute_on_signal_bar = True

    def __init__(self) -> None:
        self.signal_exit_reason: pd.Series | None = None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=data.index, dtype="int8")
        exit_reason = pd.Series("", index=data.index, dtype="object")
        if len(data) >= 3:
            signal.iloc[2] = 1
        if len(data) >= 4:
            signal.iloc[2] = 1
            signal.iloc[3] = 0
            exit_reason.iloc[3] = "Red Gator Teeth PP"
        self.signal_exit_reason = exit_reason
        return signal


class _RepaintedHoldThenConfirmedExitStrategy:
    execute_on_signal_bar = True

    def __init__(self) -> None:
        self.signal_exit_reason: pd.Series | None = None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=data.index, dtype="int8")
        exit_reason = pd.Series("", index=data.index, dtype="object")
        if len(data) >= 3:
            signal.iloc[2] = 1
        if len(data) >= 4:
            signal.iloc[2] = 1
            signal.iloc[3] = 1
        if len(data) >= 5:
            signal.iloc[2] = 1
            signal.iloc[3] = 0
            signal.iloc[4] = 0
            exit_reason.iloc[4] = "Red Gator PP"
        self.signal_exit_reason = exit_reason
        return signal


class _IntrabarEventOnlyStrategy:
    execute_on_signal_bar = True

    def __init__(self, events_by_bar_index: dict[int, list[dict[str, float | int | str]]]) -> None:
        self._events_by_bar_index = events_by_bar_index
        self.signal_intrabar_events: dict[int, list[dict[str, float | int | str]]] | None = None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        self.signal_intrabar_events = self._events_by_bar_index
        return pd.Series(0, index=data.index, dtype="int8")


def test_dashboard_chart_includes_gator_profit_protection_fallback_markers_without_real_fill() -> None:
    args = run_paper_trading.build_parser().parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--interval",
            "Min60",
            "--capital",
            "10000",
            "--order-type",
            "market",
            "--slippage",
            "0",
            "--size-mode",
            "units",
            "--size-value",
            "1",
        ]
    )
    session = run_paper_trading.RealTimePaperTradingSession(
        symbol="BTC_USDT",
        interval="Min60",
        strategy=_DelayedFlatExitStrategy(),
        config=run_paper_trading._build_backtest_config(args),
        warmup_bars=10,
    )

    priming = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 102, 103, 101, 102.5),
            _candle("2024-01-01T03:00:00Z", 103, 104, 102, 103.5),
        ]
    )
    session.update_market_snapshot(priming, pd.Timestamp("2024-01-01T04:30:00Z"))
    session.prime(priming, pd.Timestamp("2024-01-01T04:30:00Z"))

    closing = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 102, 103, 101, 102.5),
            _candle("2024-01-01T03:00:00Z", 103, 104, 102, 103.5),
            _candle("2024-01-01T04:00:00Z", 104, 105, 103, 104.5),
        ]
    )
    session.update_market_snapshot(closing, pd.Timestamp("2024-01-01T05:30:00Z"))
    processed = session.process_market_data(closing, pd.Timestamp("2024-01-01T05:30:00Z"))
    payload = session.artifacts_payload()

    assert processed[0]["fills"] == 0
    assert session.snapshot_summary()["fill_count"] == 0
    fallback_markers = [
        marker
        for marker in payload["price_chart"]["markers"]
        if marker.get("markerGroup") == "gator_profit_protection_fallback"
    ]
    assert {(marker["time"], marker["text"]) for marker in fallback_markers} == {
        (int(pd.Timestamp("2024-01-01T02:00:00Z").timestamp()), "LE-1W"),
        (int(pd.Timestamp("2024-01-01T03:00:00Z").timestamp()), "LX-R"),
    }
    assert [line["label"] for line in payload["price_chart"]["trade_event_lines"] if line["label"] in {"LE", "LX"}] == ["LE", "LX"]


def test_bw_intrabar_event_executes_immediately_on_live_snapshot() -> None:
    args = run_paper_trading.build_parser().parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--interval",
            "Min60",
            "--capital",
            "10000",
            "--order-type",
            "market",
            "--size-mode",
            "units",
            "--size-value",
            "1",
        ]
    )
    session = run_paper_trading.RealTimePaperTradingSession(
        symbol="BTC_USDT",
        interval="Min60",
        strategy=run_paper_trading.BWStrategy(first_wiseman_contracts=1),
        config=run_paper_trading._build_backtest_config(args),
        warmup_bars=10,
    )
    session.strategy = _IntrabarEventOnlyStrategy(
        {
            2: [
                {
                    "event_type": "entry",
                    "side": 1,
                    "price": 101.25,
                    "contracts": 1.0,
                    "reason": "Bullish 1W",
                    "stop_loss_price": 99.5,
                }
            ]
        }
    )

    initial = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.0),
        ]
    )
    now = pd.Timestamp("2024-01-01T02:30:00Z")
    session.prime(initial, now)

    live_with_active = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.0),
            _candle("2024-01-01T02:00:00Z", 101, 102, 100, 101.4),
        ]
    )
    processed = session.process_market_data(live_with_active, now)

    assert len(processed) == 1
    assert processed[0]["intrabar"] is True
    assert processed[0]["fills"] == 1
    summary = session.snapshot_summary()
    assert summary["fill_count"] == 1
    assert summary["position_quantity"] == pytest.approx(1.0)


def test_bw_intrabar_event_is_not_double_executed_when_live_bar_updates() -> None:
    args = run_paper_trading.build_parser().parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--interval",
            "Min60",
            "--capital",
            "10000",
            "--order-type",
            "market",
            "--size-mode",
            "units",
            "--size-value",
            "1",
        ]
    )
    session = run_paper_trading.RealTimePaperTradingSession(
        symbol="BTC_USDT",
        interval="Min60",
        strategy=run_paper_trading.BWStrategy(first_wiseman_contracts=1),
        config=run_paper_trading._build_backtest_config(args),
        warmup_bars=10,
    )
    session.strategy = _IntrabarEventOnlyStrategy(
        {
            2: [
                {
                    "event_type": "entry",
                    "side": 1,
                    "price": 101.25,
                    "contracts": 1.0,
                    "reason": "Bullish 1W",
                }
            ]
        }
    )

    base = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.0),
        ]
    )
    session.prime(base, pd.Timestamp("2024-01-01T02:30:00Z"))

    first_live = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.0),
            _candle("2024-01-01T02:00:00Z", 101, 102, 100, 101.4),
        ]
    )
    session.process_market_data(first_live, pd.Timestamp("2024-01-01T02:30:00Z"))
    assert session.snapshot_summary()["fill_count"] == 1

    updated_live = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.0),
            _candle("2024-01-01T02:00:00Z", 101, 103, 100, 102.0),
        ]
    )
    processed = session.process_market_data(updated_live, pd.Timestamp("2024-01-01T02:40:00Z"))
    assert len(processed) == 1
    assert processed[0]["fills"] == 0
    assert session.snapshot_summary()["fill_count"] == 1


def test_realtime_session_closed_bar_exit_is_not_suppressed_by_intrabar_preview() -> None:
    args = run_paper_trading.build_parser().parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--interval",
            "Min60",
            "--capital",
            "10000",
            "--order-type",
            "market",
            "--size-mode",
            "units",
            "--size-value",
            "1",
        ]
    )
    session = run_paper_trading.RealTimePaperTradingSession(
        symbol="BTC_USDT",
        interval="Min60",
        strategy=_PreviewThenCloseExitStrategy(),
        config=run_paper_trading._build_backtest_config(args),
        warmup_bars=10,
    )

    initial = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
        ]
    )
    session.update_market_snapshot(initial, pd.Timestamp("2024-01-01T02:30:00Z"))
    session.prime(initial, pd.Timestamp("2024-01-01T02:30:00Z"))

    closed_third = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 102, 103, 101, 102.5),
        ]
    )
    session.update_market_snapshot(closed_third, pd.Timestamp("2024-01-01T03:30:00Z"))
    opened = session.process_market_data(closed_third, pd.Timestamp("2024-01-01T03:30:00Z"))
    assert opened[0]["fills"] == 1
    assert session.snapshot_summary()["position_quantity"] == 1.0

    preview_fourth = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 102, 103, 101, 102.5),
            _candle("2024-01-01T03:00:00Z", 104, 106, 103, 105.0),
        ]
    )
    session.update_market_snapshot(preview_fourth, pd.Timestamp("2024-01-01T03:45:00Z"))
    preview_processed = session.process_market_data(preview_fourth, pd.Timestamp("2024-01-01T03:45:00Z"))
    assert all(item["fills"] == 0 for item in preview_processed)
    assert session.snapshot_summary()["position_quantity"] == 1.0

    closed_fourth = preview_fourth.copy()
    session.update_market_snapshot(closed_fourth, pd.Timestamp("2024-01-01T04:30:00Z"))
    closed_processed = session.process_market_data(closed_fourth, pd.Timestamp("2024-01-01T04:30:00Z"))

    assert len(closed_processed) == 1
    assert closed_processed[0]["fills"] == 1
    assert session.snapshot_summary()["position_quantity"] == 0.0
    fills = session.fills_dataframe()
    assert fills["reason"].tolist() == ["1W", "Red PP"]


def test_realtime_session_bw_red_teeth_exit_waits_for_closed_bar_confirmation() -> None:
    args = run_paper_trading.build_parser().parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--interval",
            "Min60",
            "--capital",
            "10000",
            "--order-type",
            "market",
            "--size-mode",
            "units",
            "--size-value",
            "1",
        ]
    )
    session = run_paper_trading.RealTimePaperTradingSession(
        symbol="BTC_USDT",
        interval="Min60",
        strategy=_PreviewThenCloseExitStrategyBWRedTeeth(),
        config=run_paper_trading._build_backtest_config(args),
        warmup_bars=10,
    )

    initial = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
        ]
    )
    session.update_market_snapshot(initial, pd.Timestamp("2024-01-01T02:30:00Z"))
    session.prime(initial, pd.Timestamp("2024-01-01T02:30:00Z"))

    closed_third = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 102, 103, 101, 102.5),
        ]
    )
    session.update_market_snapshot(closed_third, pd.Timestamp("2024-01-01T03:30:00Z"))
    opened = session.process_market_data(closed_third, pd.Timestamp("2024-01-01T03:30:00Z"))
    assert opened[0]["fills"] == 1
    assert session.snapshot_summary()["position_quantity"] == 1.0

    preview_fourth = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 102, 103, 101, 102.5),
            _candle("2024-01-01T03:00:00Z", 104, 106, 103, 105.0),
        ]
    )
    session.update_market_snapshot(preview_fourth, pd.Timestamp("2024-01-01T03:45:00Z"))
    preview_processed = session.process_market_data(preview_fourth, pd.Timestamp("2024-01-01T03:45:00Z"))
    assert all(item["fills"] == 0 for item in preview_processed)
    assert session.snapshot_summary()["position_quantity"] == 1.0

    closed_fourth = preview_fourth.copy()
    session.update_market_snapshot(closed_fourth, pd.Timestamp("2024-01-01T04:30:00Z"))
    closed_processed = session.process_market_data(closed_fourth, pd.Timestamp("2024-01-01T04:30:00Z"))

    assert len(closed_processed) == 1
    assert closed_processed[0]["fills"] == 1
    assert session.snapshot_summary()["position_quantity"] == 0.0
    fills = session.fills_dataframe()
    assert fills["reason"].tolist() == ["1W", "Red Gator Teeth PP"]


def test_realtime_session_ignores_repainted_historical_exit_in_favor_of_latest_confirmed_close() -> None:
    args = run_paper_trading.build_parser().parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--interval",
            "Min60",
            "--capital",
            "10000",
            "--order-type",
            "market",
            "--size-mode",
            "units",
            "--size-value",
            "1",
        ]
    )
    session = run_paper_trading.RealTimePaperTradingSession(
        symbol="BTC_USDT",
        interval="Min60",
        strategy=_RepaintedHoldThenConfirmedExitStrategy(),
        config=run_paper_trading._build_backtest_config(args),
        warmup_bars=10,
    )

    initial = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
        ]
    )
    session.update_market_snapshot(initial, pd.Timestamp("2024-01-01T02:30:00Z"))
    session.prime(initial, pd.Timestamp("2024-01-01T02:30:00Z"))

    third = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 102, 103, 101, 102.5),
        ]
    )
    session.update_market_snapshot(third, pd.Timestamp("2024-01-01T03:30:00Z"))
    opened = session.process_market_data(third, pd.Timestamp("2024-01-01T03:30:00Z"))
    assert opened[0]["fills"] == 1
    assert session.snapshot_summary()["position_quantity"] == 1.0

    fourth = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 102, 103, 101, 102.5),
            _candle("2024-01-01T03:00:00Z", 103, 104, 102, 103.5),
        ]
    )
    session.update_market_snapshot(fourth, pd.Timestamp("2024-01-01T04:30:00Z"))
    held = session.process_market_data(fourth, pd.Timestamp("2024-01-01T04:30:00Z"))
    assert held[0]["fills"] == 0
    assert session.snapshot_summary()["position_quantity"] == 1.0

    fifth = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 102, 103, 101, 102.5),
            _candle("2024-01-01T03:00:00Z", 103, 104, 102, 103.5),
            _candle("2024-01-01T04:00:00Z", 104, 106, 103, 105.5),
        ]
    )
    session.update_market_snapshot(fifth, pd.Timestamp("2024-01-01T05:30:00Z"))
    closed = session.process_market_data(fifth, pd.Timestamp("2024-01-01T05:30:00Z"))
    fills = session.fills_dataframe()

    assert closed[0]["fills"] == 1
    assert session.snapshot_summary()["position_quantity"] == 0.0
    assert fills["reason"].tolist() == ["1W", "Red PP"]
    assert fills["execution_reason"].tolist() == ["market", "market"]


class _IntrabarStopEntryStrategy:
    execute_on_signal_bar = True

    def __init__(self, trigger_price: float = 103.0) -> None:
        self.trigger_price = trigger_price
        self.signal_fill_prices: pd.Series | None = None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=data.index, dtype="int8")
        fill_prices = pd.Series(float("nan"), index=data.index, dtype="float64")
        if len(data) >= 3:
            signal.iloc[-1] = 1
            fill_prices.iloc[-1] = self.trigger_price
        self.signal_fill_prices = fill_prices
        return signal


class _ExplicitCloseFillExitStrategy:
    execute_on_signal_bar = True

    def __init__(self) -> None:
        self.signal_exit_reason: pd.Series | None = None
        self.signal_fill_prices: pd.Series | None = None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=data.index, dtype="int8")
        fills = pd.Series(float("nan"), index=data.index, dtype="float64")
        exit_reason = pd.Series("", index=data.index, dtype="object")
        if len(data) >= 3:
            signal.iloc[2] = 1
        if len(data) >= 4:
            signal.iloc[3] = 1
        if len(data) >= 5:
            signal.iloc[4] = 0
            fills.iloc[4] = float(data["close"].iloc[4])
            exit_reason.iloc[4] = "Red Gator PP"
        self.signal_fill_prices = fills
        self.signal_exit_reason = exit_reason
        return signal


class _ExplicitMarketFillEntryStrategy:
    execute_on_signal_bar = True

    def __init__(self, trigger_price: float = 103.0) -> None:
        self.trigger_price = trigger_price
        self.signal_fill_prices: pd.Series | None = None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=data.index, dtype="int8")
        fills = pd.Series(float("nan"), index=data.index, dtype="float64")
        if len(data) >= 3:
            signal.iloc[2] = 1
            fills.iloc[2] = self.trigger_price
        self.signal_fill_prices = fills
        return signal


def _candle(ts: str, open_: float, high: float, low: float, close: float) -> _FakeCandle:
    return _FakeCandle(time=int(pd.Timestamp(ts).timestamp()), open=open_, high=high, low=low, close=close, volume=1000.0)


def test_paper_trading_parser_accepts_realtime_strategy_and_execution_flags() -> None:
    parser = run_paper_trading.build_parser()
    args = parser.parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--interval",
            "Min60",
            "--capital",
            "25000",
            "--fee",
            "0.001",
            "--slippage",
            "0.0005",
            "--spread",
            "0.0004",
            "--order-type",
            "stop_limit",
            "--size-mode",
            "volatility_scaled",
            "--size-value",
            "0.25",
            "--size-min-usd",
            "500",
            "--volatility-target-annualized",
            "0.12",
            "--volatility-lookback",
            "30",
            "--max-position-size",
            "5000",
            "--max-leverage",
            "2",
            "--max-loss-pct-of-equity",
            "0.02",
            "--equity-cutoff",
            "7000",
            "--strategy",
            "wiseman",
            "--wiseman-1w-contracts",
            "1",
            "--wiseman-2w-contracts",
            "0",
            "--wiseman-3w-contracts",
            "0",
            "--wiseman-reversal-contracts-mult",
            "1.0",
            "--gator-width-valid-factor",
            "1.0",
            "--gator-width-mult",
            "0.5",
            "--gator-width-lookback",
            "180",
            "--wiseman-profit-protection-teeth-exit",
            "--wiseman-profit-protection-min-bars",
            "15",
            "--wiseman-profit-protection-min-unrealized-return",
            "0.01",
            "--no-wiseman-profit-protection-require-gator-open",
            "--wiseman-profit-protection-volatility-lookback",
            "40",
            "--wiseman-profit-protection-annualized-volatility-scaler",
            "0.85",
            "--wiseman-cancel-reversal-on-first-exit",
            "--wiseman-gator-direction-mode",
            "1",
            "--wiseman-reversal-cooldown",
            "1",
            "--wiseman-profit-protection-lips-exit",
            "--wiseman-profit-protection-lips-volatility-trigger",
            "0.04",
            "--wiseman-profit-protection-lips-profit-trigger-mult",
            "5",
            "--wiseman-profit-protection-lips-volatility-lookback",
            "30",
            "--wiseman-profit-protection-lips-recent-trade-lookback",
            "10",
            "--wiseman-profit-protection-lips-min-unrealized-return",
            "0.03",
            "--wiseman-profit-protection-lips-arm-on-min-unrealized-return",
            "--wiseman-profit-protection-zones-exit",
            "--wiseman-profit-protection-zones-min-unrealized-return",
            "0.025",
            "--1W-divergence-filter",
            "240",
        ]
    )

    assert args.symbol == "BTC_USDT"
    assert args.interval == "Min60"
    assert args.order_type == "stop_limit"
    assert args.size_mode == "volatility_scaled"
    assert args.volatility_target_annual == 0.12
    assert args.max_position_size == 5000.0
    assert args.max_leverage == 2.0
    assert args.max_loss_pct_of_equity == pytest.approx(0.02)
    assert args.equity_cutoff == 7000.0
    assert args.wiseman_1w_contracts == 1
    assert args.wiseman_2w_contracts == 0
    assert args.wiseman_3w_contracts == 0
    assert args.wiseman_1w_divergence_filter_bars == 240
    strategy = run_paper_trading._build_strategy(args)
    assert isinstance(strategy, run_paper_trading.WisemanStrategy)
    assert strategy.teeth_profit_protection_enabled is True
    assert strategy.lips_profit_protection_enabled is True
    assert strategy.zone_profit_protection_enabled is True
    assert strategy.profit_protection_volatility_lookback == 40
    assert strategy.profit_protection_annualized_volatility_scaler == pytest.approx(0.85)
    assert strategy.zone_profit_protection_min_unrealized_return == pytest.approx(0.025)

    config = run_paper_trading._build_backtest_config(args)
    assert config.max_loss_pct_of_equity == pytest.approx(0.02)


def test_paper_trading_parser_accepts_stop_loss_scaled_size_mode() -> None:
    parser = run_paper_trading.build_parser()
    args = parser.parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--size-mode",
            "stop_loss_scaled",
            "--size-value",
            "0.01",
        ]
    )

    assert args.size_mode == "stop_loss_scaled"
    assert args.size_value == pytest.approx(0.01)


def test_paper_trading_parser_accepts_equity_milestone_usd_size_mode() -> None:
    parser = run_paper_trading.build_parser()
    args = parser.parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--size-mode",
            "equity_milestone_usd",
            "--size-value",
            "1000",
            "--size-equity-milestones",
            "15000:1500,20000:2000",
        ]
    )

    assert args.size_mode == "equity_milestone_usd"
    assert args.size_equity_milestones == "15000:1500,20000:2000"

    config = run_paper_trading._build_backtest_config(args)
    assert config.trade_size_equity_milestones == ((15000.0, 1500.0), (20000.0, 2000.0))


def test_paper_trading_build_strategy_supports_ntd_and_combo() -> None:
    parser = run_paper_trading.build_parser()

    ntd_args = parser.parse_args(["--symbol", "BTC_USDT", "--strategy", "ntd"])
    ntd_strategy = run_paper_trading._build_strategy(ntd_args)
    assert isinstance(ntd_strategy, run_paper_trading.NTDStrategy)
    assert ntd_strategy.teeth_profit_protection_enabled is False
    assert ntd_strategy.lips_profit_protection_enabled is False
    assert ntd_strategy.zone_profit_protection_enabled is False

    combo_args = parser.parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--strategy",
            "wiseman,ntd",
            "--wiseman-profit-protection-credit-unrealized-before-min-bars",
        ]
    )
    combo_strategy = run_paper_trading._build_strategy(combo_args)
    assert isinstance(combo_strategy, run_paper_trading.CombinedStrategy)
    ntd_components = [strategy for strategy in combo_strategy.strategies if isinstance(strategy, run_paper_trading.NTDStrategy)]
    assert len(ntd_components) == 1
    assert ntd_components[0].require_gator_close_reset is True
    assert ntd_components[0].ao_ac_near_zero_lookback == 50
    assert ntd_components[0].teeth_profit_protection_credit_unrealized_before_min_bars is True
    assert ntd_components[0].ao_ac_near_zero_factor == pytest.approx(0.25)


def test_paper_trading_build_strategy_accepts_bw_contract_flags() -> None:
    parser = run_paper_trading.build_parser()
    args = parser.parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--strategy",
            "bw",
            "--bw-1w-gator-open-lookback",
            "100",
            "--bw-1w-gator-open-percentile",
            "50",
            "--bw-1w-contracts",
            "2",
            "--bw-only-trade-1w-r",
            "--bw-ntd-initial-fractal-enabled",
            "--bw-ntd-initial-fractal-contracts",
            "4",
            "--no-bw-profit-protection-red-teeth-exit",
            "--bw-profit-protection-red-teeth-min-bars",
            "6",
            "--bw-profit-protection-red-teeth-require-gator-direction-alignment",
            "--no-bw-profit-protection-green-lips-exit",
            "--bw-profit-protection-green-lips-min-bars",
            "7",
            "--bw-profit-protection-green-lips-require-gator-direction-alignment",
            "--bw-profit-protection-zones-exit",
            "--bw-profit-protection-zones-min-bars",
            "9",
            "--bw-profit-protection-zones-min-unrealized-return",
            "0.05",
            "--bw-profit-protection-zones-volatility-lookback",
            "180",
            "--bw-profit-protection-zones-annualized-volatility-scaler",
            "0.5",
            "--bw-profit-protection-zones-min-same-color-bars",
            "6",
            "--allow-close-on-1w-d",
            "--allow-close-on-1w-d-min-unrealized-return",
            "0.05",
            "--allow-close-on-1w-a",
            "--allow-close-on-1w-a-min-unrealized-return",
            "0.06",
            "--bw-profit-protection-sigma-move-exit",
            "--bw-profit-protection-sigma-move-lookback",
            "30",
            "--bw-profit-protection-sigma-move-sigma",
            "2.5",
            "--bw-close-on-underlying-gain-pct",
            "0.04",
        ]
    )

    strategy = run_paper_trading._build_strategy(args)
    assert isinstance(strategy, run_paper_trading.BWStrategy)
    assert strategy.first_wiseman_contracts == 2
    assert strategy.only_trade_1w_reversals is True
    assert strategy.gator_open_filter_lookback == 100
    assert strategy.gator_open_filter_min_percentile == pytest.approx(50.0)
    assert strategy.ntd_initial_fractal_contracts == 4
    assert strategy.red_teeth_profit_protection_enabled is False
    assert strategy.red_teeth_profit_protection_min_bars == 6
    assert strategy.red_teeth_profit_protection_require_gator_direction_alignment is True
    assert strategy.green_lips_profit_protection_enabled is False
    assert strategy.green_lips_profit_protection_min_bars == 7
    assert strategy.green_lips_profit_protection_require_gator_direction_alignment is True
    assert strategy.zones_profit_protection_enabled is True
    assert strategy.zones_profit_protection_min_bars == 9
    assert strategy.zones_profit_protection_min_unrealized_return == pytest.approx(0.05)
    assert strategy.zones_profit_protection_volatility_lookback == 180
    assert strategy.zones_profit_protection_annualized_volatility_scaler == pytest.approx(0.5)
    assert strategy.zones_profit_protection_min_same_color_bars == 6
    assert strategy.allow_close_on_1w_d is True
    assert strategy.allow_close_on_1w_d_min_unrealized_return == pytest.approx(0.05)
    assert strategy.allow_close_on_1w_a is True
    assert strategy.allow_close_on_1w_a_min_unrealized_return == pytest.approx(0.06)
    assert strategy.sigma_move_profit_protection_enabled is True
    assert strategy.sigma_move_profit_protection_lookback == 30
    assert strategy.sigma_move_profit_protection_sigma == pytest.approx(2.5)
    assert strategy.close_on_underlying_gain_pct == pytest.approx(0.04)


def test_realtime_session_only_trades_new_closed_bars() -> None:
    args = run_paper_trading.build_parser().parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--interval",
            "Min60",
            "--capital",
            "10000",
            "--order-type",
            "market",
            "--slippage",
            "0",
            "--size-mode",
            "units",
            "--size-value",
            "1",
            "--max-cycles",
            "1",
        ]
    )
    config = run_paper_trading._build_backtest_config(args)
    session = run_paper_trading.RealTimePaperTradingSession(
        symbol="BTC_USDT",
        interval="Min60",
        strategy=_RealtimeSignalStrategy(),
        config=config,
        warmup_bars=10,
    )

    initial = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
        ]
    )
    session.prime(initial, pd.Timestamp("2024-01-01T02:30:00Z"))

    assert session.snapshot_summary()["fill_count"] == 0
    assert session.snapshot_summary()["position_quantity"] == 0.0

    updated = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 102, 103, 101, 102.5),
        ]
    )
    processed = session.process_market_data(updated, pd.Timestamp("2024-01-01T03:30:00Z"))

    assert len(processed) == 1
    assert processed[0]["fills"] == 1
    assert session.snapshot_summary()["fill_count"] == 1
    assert session.snapshot_summary()["position_quantity"] == 1.0

    processed_again = session.process_market_data(updated, pd.Timestamp("2024-01-01T03:45:00Z"))
    assert processed_again == []
    assert session.snapshot_summary()["fill_count"] == 1


def test_realtime_session_ignores_stale_pre_start_signal_state() -> None:
    args = run_paper_trading.build_parser().parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--interval",
            "Min60",
            "--capital",
            "10000",
            "--order-type",
            "market",
            "--slippage",
            "0",
            "--size-mode",
            "units",
            "--size-value",
            "1",
        ]
    )
    session = run_paper_trading.RealTimePaperTradingSession(
        symbol="BTC_USDT",
        interval="Min60",
        strategy=_StaleLongStrategy(),
        config=run_paper_trading._build_backtest_config(args),
        warmup_bars=10,
    )

    initial = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
        ]
    )
    session.prime(initial, pd.Timestamp("2024-01-01T02:30:00Z"))

    updated = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 102, 103, 101, 102.5),
        ]
    )
    processed = session.process_market_data(updated, pd.Timestamp("2024-01-01T03:30:00Z"))

    assert len(processed) == 1
    assert processed[0]["fills"] == 0
    assert session.snapshot_summary()["fill_count"] == 0
    assert session.snapshot_summary()["position_quantity"] == 0.0


def test_realtime_session_does_not_micro_rebalance_without_new_signal() -> None:
    args = run_paper_trading.build_parser().parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--interval",
            "Min60",
            "--capital",
            "10000",
            "--order-type",
            "market",
            "--size-mode",
            "percent_of_equity",
            "--size-value",
            "0.5",
        ]
    )
    session = run_paper_trading.RealTimePaperTradingSession(
        symbol="BTC_USDT",
        interval="Min60",
        strategy=_EnterThenHoldLongStrategy(),
        config=run_paper_trading._build_backtest_config(args),
        warmup_bars=10,
    )

    initial = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
        ]
    )
    session.prime(initial, pd.Timestamp("2024-01-01T02:30:00Z"))

    third_bar = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 110, 112, 109, 111.0),
        ]
    )
    first_processed = session.process_market_data(third_bar, pd.Timestamp("2024-01-01T03:30:00Z"))
    assert first_processed[0]["fills"] == 1
    first_fill_count = session.snapshot_summary()["fill_count"]

    fourth_bar = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 110, 112, 109, 111.0),
            _candle("2024-01-01T03:00:00Z", 150, 151, 149, 150.0),
        ]
    )
    second_processed = session.process_market_data(fourth_bar, pd.Timestamp("2024-01-01T04:30:00Z"))

    assert len(second_processed) == 1
    assert second_processed[0]["fills"] == 0
    assert session.snapshot_summary()["fill_count"] == first_fill_count


def test_realtime_session_exits_when_latest_strategy_state_is_flat_but_position_is_still_open() -> None:
    args = run_paper_trading.build_parser().parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--interval",
            "Min60",
            "--capital",
            "10000",
            "--order-type",
            "market",
            "--size-mode",
            "units",
            "--size-value",
            "1",
        ]
    )
    session = run_paper_trading.RealTimePaperTradingSession(
        symbol="BTC_USDT",
        interval="Min60",
        strategy=_DelayedFlatExitStrategy(),
        config=run_paper_trading._build_backtest_config(args),
        warmup_bars=10,
    )

    initial = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
        ]
    )
    session.prime(initial, pd.Timestamp("2024-01-01T02:30:00Z"))

    third_bar = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 102, 103, 101, 102.5),
        ]
    )
    first_processed = session.process_market_data(third_bar, pd.Timestamp("2024-01-01T03:30:00Z"))
    assert first_processed[0]["fills"] == 1
    assert session.snapshot_summary()["position_quantity"] == 1.0

    fourth_bar = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 102, 103, 101, 102.5),
            _candle("2024-01-01T03:00:00Z", 103, 104, 102, 103.5),
        ]
    )
    second_processed = session.process_market_data(fourth_bar, pd.Timestamp("2024-01-01T04:30:00Z"))
    assert second_processed[0]["fills"] == 0
    assert session.snapshot_summary()["position_quantity"] == 1.0

    fifth_bar = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 102, 103, 101, 102.5),
            _candle("2024-01-01T03:00:00Z", 103, 104, 102, 103.5),
            _candle("2024-01-01T04:00:00Z", 104, 105, 103, 104.5),
        ]
    )
    third_processed = session.process_market_data(fifth_bar, pd.Timestamp("2024-01-01T05:30:00Z"))

    assert len(third_processed) == 1
    assert third_processed[0]["fills"] == 1
    assert session.snapshot_summary()["position_quantity"] == 0.0
    fills = session.fills_dataframe()
    assert len(fills) == 2
    assert fills["reason"].tolist() == ["1W", "Red PP"]

    payload = session.artifacts_payload()
    execution_markers = [
        marker for marker in payload["price_chart"]["markers"] if marker.get("markerGroup") == "execution"
    ]
    assert {str(marker["text"]) for marker in execution_markers} == {"LE-1W", "LX-Red PP"}


def test_realtime_session_places_and_triggers_intrabar_stop_orders_before_bar_close() -> None:
    args = run_paper_trading.build_parser().parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--interval",
            "Min60",
            "--capital",
            "10000",
            "--order-type",
            "stop",
            "--size-mode",
            "units",
            "--size-value",
            "1",
        ]
    )
    session = run_paper_trading.RealTimePaperTradingSession(
        symbol="BTC_USDT",
        interval="Min60",
        strategy=_IntrabarStopEntryStrategy(trigger_price=103.0),
        config=run_paper_trading._build_backtest_config(args),
        warmup_bars=10,
    )

    initial = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
        ]
    )
    session.update_market_snapshot(initial, pd.Timestamp("2024-01-01T02:05:00Z"))

    not_triggered = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 102, 102.5, 101, 102.2),
        ]
    )
    session.update_market_snapshot(not_triggered, pd.Timestamp("2024-01-01T02:15:00Z"))
    first_pass = session.process_market_data(not_triggered, pd.Timestamp("2024-01-01T02:15:00Z"))

    assert session.snapshot_summary()["fill_count"] == 0
    assert all(item["fills"] == 0 for item in first_pass)

    triggered = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 102, 104.0, 101, 103.5),
        ]
    )
    session.update_market_snapshot(triggered, pd.Timestamp("2024-01-01T02:25:00Z"))
    second_pass = session.process_market_data(triggered, pd.Timestamp("2024-01-01T02:25:00Z"))

    assert any(item["fills"] == 1 and item.get("intrabar") for item in second_pass)
    assert session.snapshot_summary()["fill_count"] == 1
    assert session.snapshot_summary()["position_quantity"] == 1.0


def test_realtime_session_fills_existing_stop_order_intrabar_after_closed_setup() -> None:
    args = run_paper_trading.build_parser().parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--interval",
            "Min60",
            "--capital",
            "10000",
            "--order-type",
            "stop",
            "--size-mode",
            "units",
            "--size-value",
            "1",
        ]
    )
    session = run_paper_trading.RealTimePaperTradingSession(
        symbol="BTC_USDT",
        interval="Min60",
        strategy=_IntrabarStopEntryStrategy(trigger_price=103.0),
        config=run_paper_trading._build_backtest_config(args),
        warmup_bars=10,
    )

    initial = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
        ]
    )
    session.update_market_snapshot(initial, pd.Timestamp("2024-01-01T02:30:00Z"))
    session.prime(initial, pd.Timestamp("2024-01-01T02:30:00Z"))

    closed_third = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 102, 102.5, 101, 102.2),
        ]
    )
    session.update_market_snapshot(closed_third, pd.Timestamp("2024-01-01T03:30:00Z"))
    closed_processed = session.process_market_data(closed_third, pd.Timestamp("2024-01-01T03:30:00Z"))

    assert closed_processed[0]["fills"] == 0
    assert session.snapshot_summary()["fill_count"] == 0
    assert len(session.engine.snapshot().open_orders) == 1

    triggered_fourth = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 102, 102.5, 101, 102.2),
            _candle("2024-01-01T03:00:00Z", 102.4, 104.0, 102.1, 103.5),
        ]
    )
    session.update_market_snapshot(triggered_fourth, pd.Timestamp("2024-01-01T03:45:00Z"))
    intrabar_processed = session.process_market_data(triggered_fourth, pd.Timestamp("2024-01-01T03:45:00Z"))

    assert any(item["fills"] == 1 and item.get("intrabar") for item in intrabar_processed)
    assert session.snapshot_summary()["fill_count"] == 1
    assert session.snapshot_summary()["position_quantity"] == 1.0
    fills = session.fills_dataframe()
    assert fills["execution_reason"].tolist() == ["stop"]
    assert fills["reason"].tolist() == ["1W"]


def test_realtime_session_updates_dashboard_mark_price_without_new_closed_bar() -> None:
    args = run_paper_trading.build_parser().parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--interval",
            "Min60",
            "--capital",
            "10000",
            "--order-type",
            "market",
            "--size-mode",
            "units",
            "--size-value",
            "1",
        ]
    )
    session = run_paper_trading.RealTimePaperTradingSession(
        symbol="BTC_USDT",
        interval="Min60",
        strategy=_RealtimeSignalStrategy(),
        config=run_paper_trading._build_backtest_config(args),
        warmup_bars=10,
    )

    initial = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
        ]
    )
    session.update_market_snapshot(initial, pd.Timestamp("2024-01-01T02:30:00Z"))
    session.prime(initial, pd.Timestamp("2024-01-01T02:30:00Z"))

    closed_third = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 102, 103, 101, 102.5),
        ]
    )
    session.update_market_snapshot(closed_third, pd.Timestamp("2024-01-01T03:30:00Z"))
    processed = session.process_market_data(closed_third, pd.Timestamp("2024-01-01T03:30:00Z"))
    assert processed[0]["fills"] == 1

    forming_fourth = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 102, 103, 101, 102.5),
            _candle("2024-01-01T03:00:00Z", 149, 151, 148, 150.0),
        ]
    )
    session.update_market_snapshot(forming_fourth, pd.Timestamp("2024-01-01T03:45:00Z"))
    processed_again = session.process_market_data(forming_fourth, pd.Timestamp("2024-01-01T03:45:00Z"))

    assert any(item.get("intrabar") for item in processed_again)
    summary = session.snapshot_summary()
    payload = session.artifacts_payload()
    assert summary["mark_price"] == 150.0
    assert payload["price_chart"]["count"] == len(session.latest_market_data)
    assert summary["unrealized_pnl"] >= 0.0
    assert payload["price_chart"]["candles"][-1]["time"] == int(pd.Timestamp("2024-01-01T03:00:00Z").timestamp())
    assert payload["price_chart"]["candles"][-1]["close"] == 150.0


def test_realtime_session_honors_explicit_market_exit_fill_at_bar_close() -> None:
    args = run_paper_trading.build_parser().parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--interval",
            "Min60",
            "--capital",
            "10000",
            "--order-type",
            "market",
            "--slippage",
            "0",
            "--size-mode",
            "units",
            "--size-value",
            "1",
        ]
    )
    session = run_paper_trading.RealTimePaperTradingSession(
        symbol="BTC_USDT",
        interval="Min60",
        strategy=_ExplicitCloseFillExitStrategy(),
        config=run_paper_trading._build_backtest_config(args),
        warmup_bars=10,
    )

    initial = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
        ]
    )
    session.update_market_snapshot(initial, pd.Timestamp("2024-01-01T02:30:00Z"))
    session.prime(initial, pd.Timestamp("2024-01-01T02:30:00Z"))

    third = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 102, 103, 101, 102.5),
        ]
    )
    session.update_market_snapshot(third, pd.Timestamp("2024-01-01T03:30:00Z"))
    first_processed = session.process_market_data(third, pd.Timestamp("2024-01-01T03:30:00Z"))
    assert first_processed[0]["fills"] == 1

    fourth = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 102, 103, 101, 102.5),
            _candle("2024-01-01T03:00:00Z", 103, 104, 102, 103.5),
        ]
    )
    session.update_market_snapshot(fourth, pd.Timestamp("2024-01-01T04:30:00Z"))
    second_processed = session.process_market_data(fourth, pd.Timestamp("2024-01-01T04:30:00Z"))
    assert second_processed[0]["fills"] == 0

    fifth = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 102, 103, 101, 102.5),
            _candle("2024-01-01T03:00:00Z", 103, 104, 102, 103.5),
            _candle("2024-01-01T04:00:00Z", 104, 106, 103, 105.5),
        ]
    )
    session.update_market_snapshot(fifth, pd.Timestamp("2024-01-01T05:30:00Z"))
    third_processed = session.process_market_data(fifth, pd.Timestamp("2024-01-01T05:30:00Z"))
    fills = session.fills_dataframe()

    assert third_processed[0]["fills"] == 1
    assert len(fills) == 2
    assert fills.iloc[-1]["reason"] == "Red PP"
    assert fills.iloc[-1]["execution_reason"] == "market"
    assert fills.iloc[-1]["price"] == pytest.approx(105.5)
    assert session.snapshot_summary()["position_quantity"] == 0.0


def test_realtime_session_honors_explicit_market_entry_fill_price() -> None:
    args = run_paper_trading.build_parser().parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--interval",
            "Min60",
            "--capital",
            "10000",
            "--order-type",
            "market",
            "--slippage",
            "0",
            "--size-mode",
            "units",
            "--size-value",
            "1",
        ]
    )
    session = run_paper_trading.RealTimePaperTradingSession(
        symbol="BTC_USDT",
        interval="Min60",
        strategy=_ExplicitMarketFillEntryStrategy(trigger_price=103.0),
        config=run_paper_trading._build_backtest_config(args),
        warmup_bars=10,
    )

    initial = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
        ]
    )
    session.update_market_snapshot(initial, pd.Timestamp("2024-01-01T02:30:00Z"))
    session.prime(initial, pd.Timestamp("2024-01-01T02:30:00Z"))

    third = run_paper_trading._candles_to_dataframe(
        [
            _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
            _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            _candle("2024-01-01T02:00:00Z", 102.0, 104.0, 101.0, 103.5),
        ]
    )
    session.update_market_snapshot(third, pd.Timestamp("2024-01-01T03:30:00Z"))
    processed = session.process_market_data(third, pd.Timestamp("2024-01-01T03:30:00Z"))

    assert processed[0]["fills"] == 1
    fills = session.fills_dataframe()
    assert len(fills) == 1
    assert fills.iloc[0]["execution_reason"] == "market"
    assert fills.iloc[0]["price"] == pytest.approx(103.0)
    assert session.snapshot_summary()["position_quantity"] == 1.0


def test_realtime_cli_writes_summary_and_fills_from_live_polls(tmp_path) -> None:
    out_dir = tmp_path / "paper_artifacts"
    client = _SequenceClient(
        [
            [
                _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
                _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            ],
            [
                _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
                _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
                _candle("2024-01-01T02:00:00Z", 102, 103, 101, 102.5),
            ],
        ]
    )
    times = iter(
        [
            pd.Timestamp("2024-01-01T02:30:00Z"),
            pd.Timestamp("2024-01-01T03:30:00Z"),
        ]
    )

    args = run_paper_trading.build_parser().parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--interval",
            "Min60",
            "--out",
            str(out_dir),
            "--summary-name",
            "summary.json",
            "--fills-name",
            "fills.csv",
            "--order-type",
            "market",
            "--size-mode",
            "units",
            "--size-value",
            "1",
            "--max-cycles",
            "2",
        ]
    )

    original_build_strategy = run_paper_trading._build_strategy
    run_paper_trading._build_strategy = lambda parsed_args: _RealtimeSignalStrategy()
    try:
        summary_path, fills_path, summary = run_paper_trading.run_from_args(
            args,
            client=client,
            now_provider=lambda: next(times),
            sleep_fn=lambda _seconds: None,
        )
    finally:
        run_paper_trading._build_strategy = original_build_strategy

    assert summary_path.exists()
    assert fills_path.exists()

    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fills = pd.read_csv(fills_path)

    assert written_summary["symbol"] == "BTC_USDT"
    assert written_summary["interval"] == "Min60"
    assert written_summary["fill_count"] == 1
    assert len(fills) == 1
    assert summary["fill_count"] == written_summary["fill_count"]

    status_path = out_dir / "paper_status.md"
    trades_path = out_dir / "paper_trades.md"
    dashboard_path = out_dir / "paper_dashboard.html"
    dashboard_data_path = out_dir / "paper_dashboard_data.json"
    dashboard_script_path = out_dir / "paper_dashboard_data.js"

    assert status_path.exists()
    assert trades_path.exists()
    assert dashboard_path.exists()
    assert dashboard_data_path.exists()
    assert dashboard_script_path.exists()

    status_text = status_path.read_text(encoding="utf-8")
    trades_text = trades_path.read_text(encoding="utf-8")
    dashboard_html = dashboard_path.read_text(encoding="utf-8")
    dashboard_data = json.loads(dashboard_data_path.read_text(encoding="utf-8"))
    dashboard_script = dashboard_script_path.read_text(encoding="utf-8")

    assert "## Current Position" in status_text
    assert "## Fill Ledger" in trades_text
    assert "setInterval(load, 1000)" in dashboard_html
    assert "paper_dashboard_data.js" in dashboard_html
    assert "Price / Execution Chart" in dashboard_html
    assert "Alligator" in dashboard_html
    assert "AO pane" in dashboard_html
    assert "AC pane" in dashboard_html
    assert dashboard_html.index("Price / Execution Chart") < dashboard_html.index("summary-cards")
    assert "collapseMarkers" in dashboard_html
    assert "window.__PAPER_DASHBOARD_DATA__ = " in dashboard_script
    assert dashboard_data["summary"]["position_quantity"] == 1.0
    assert dashboard_data["equity_curve"]
    assert dashboard_data["fills"][0]["side"] == "buy"
    assert dashboard_data["fills"][0]["reason"] == "1W"
    assert dashboard_data["fills"][0]["execution_reason"] == "market"
    assert dashboard_data["current_position"]["notional_value"] > 0
    assert dashboard_data["price_chart"]["candles"]
    assert set(dashboard_data["price_chart"]["alligator"]) == {"jaw", "teeth", "lips"}
    assert isinstance(dashboard_data["price_chart"]["ao"], list)
    assert isinstance(dashboard_data["price_chart"]["ac"], list)
    assert isinstance(dashboard_data["price_chart"]["markers"], list)
    assert dashboard_data["price_chart"]["trade_event_lines"]


def test_realtime_cli_uses_incremental_fetch_after_initial_warmup(tmp_path) -> None:
    out_dir = tmp_path / "paper_artifacts_incremental"
    client = _TrackingSequenceClient(
        [
            [
                _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
                _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            ],
            [
                _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
                _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
                _candle("2024-01-01T02:00:00Z", 102, 103, 101, 102.5),
            ],
            [
                _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
                _candle("2024-01-01T02:00:00Z", 102, 103, 101, 102.5),
                _candle("2024-01-01T03:00:00Z", 103, 104, 102, 103.5),
            ],
        ]
    )
    times = iter(
        [
            pd.Timestamp("2024-01-01T02:30:00Z"),
            pd.Timestamp("2024-01-01T03:30:00Z"),
            pd.Timestamp("2024-01-01T04:30:00Z"),
        ]
    )
    args = run_paper_trading.build_parser().parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--interval",
            "Min60",
            "--out",
            str(out_dir),
            "--order-type",
            "market",
            "--size-mode",
            "units",
            "--size-value",
            "1",
            "--warmup-bars",
            "50",
            "--max-cycles",
            "3",
        ]
    )

    original_build_strategy = run_paper_trading._build_strategy
    run_paper_trading._build_strategy = lambda parsed_args: _RealtimeSignalStrategy()
    try:
        run_paper_trading.run_from_args(
            args,
            client=client,
            now_provider=lambda: next(times),
            sleep_fn=lambda _seconds: None,
        )
    finally:
        run_paper_trading._build_strategy = original_build_strategy

    assert client.limits == [50, 8, 8]


def test_realtime_cli_survives_transient_fetch_failures_and_writes_artifacts(tmp_path) -> None:
    out_dir = tmp_path / "paper_artifacts_recovery"
    client = _FailThenRecoverClient(
        [
            [
                _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
                _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            ],
            [
                _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
                _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
                _candle("2024-01-01T02:00:00Z", 102, 103, 101, 102.5),
            ],
        ],
        failures_before_success=2,
    )
    times = iter(
        [
            pd.Timestamp("2024-01-01T02:30:00Z"),
            pd.Timestamp("2024-01-01T02:30:10Z"),
            pd.Timestamp("2024-01-01T02:30:20Z"),
            pd.Timestamp("2024-01-01T03:30:00Z"),
        ]
    )
    sleeps: list[float] = []
    args = run_paper_trading.build_parser().parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--interval",
            "Min60",
            "--out",
            str(out_dir),
            "--order-type",
            "market",
            "--size-mode",
            "units",
            "--size-value",
            "1",
            "--max-cycles",
            "2",
        ]
    )

    original_build_strategy = run_paper_trading._build_strategy
    run_paper_trading._build_strategy = lambda parsed_args: _RealtimeSignalStrategy()
    try:
        summary_path, fills_path, summary = run_paper_trading.run_from_args(
            args,
            client=client,
            now_provider=lambda: next(times),
            sleep_fn=lambda seconds: sleeps.append(seconds),
        )
    finally:
        run_paper_trading._build_strategy = original_build_strategy

    assert summary_path.exists()
    assert fills_path.exists()
    assert summary["fill_count"] == 1
    assert client.failures == 2
    assert sleeps[:2] == [5.0, 5.0]


def test_realtime_cli_survives_locked_dashboard_script_artifact(tmp_path) -> None:
    out_dir = tmp_path / "paper_artifacts_locked_script"
    client = _SequenceClient(
        [
            [
                _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
                _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            ],
            [
                _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
                _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
                _candle("2024-01-01T02:00:00Z", 102, 103, 101, 102.5),
            ],
        ]
    )
    times = iter(
        [
            pd.Timestamp("2024-01-01T02:30:00Z"),
            pd.Timestamp("2024-01-01T03:30:00Z"),
        ]
    )
    args = run_paper_trading.build_parser().parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--interval",
            "Min60",
            "--out",
            str(out_dir),
            "--order-type",
            "market",
            "--size-mode",
            "units",
            "--size-value",
            "1",
            "--max-cycles",
            "2",
        ]
    )

    original_build_strategy = run_paper_trading._build_strategy
    original_replace = Path.replace

    def _locked_replace(self: Path, target: Path):
        if Path(target).name == "paper_dashboard_data.js":
            raise PermissionError("simulated lock")
        return original_replace(self, target)

    run_paper_trading._build_strategy = lambda parsed_args: _RealtimeSignalStrategy()
    try:
        with patch.object(run_paper_trading.Path, "replace", _locked_replace):
            summary_path, fills_path, summary = run_paper_trading.run_from_args(
                args,
                client=client,
                now_provider=lambda: next(times),
                sleep_fn=lambda _seconds: None,
            )
    finally:
        run_paper_trading._build_strategy = original_build_strategy

    assert summary_path.exists()
    assert fills_path.exists()
    assert summary["fill_count"] == 1
    assert not (out_dir / "paper_dashboard_data.js").exists()


def test_realtime_cli_skips_rewriting_unchanged_artifacts(tmp_path) -> None:
    out_dir = tmp_path / "paper_artifacts_unchanged"
    client = _SequenceClient(
        [
            [
                _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
                _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            ],
            [
                _candle("2024-01-01T00:00:00Z", 100, 101, 99, 100.5),
                _candle("2024-01-01T01:00:00Z", 101, 102, 100, 101.5),
            ],
        ]
    )
    times = iter(
        [
            pd.Timestamp("2024-01-01T02:30:00Z"),
            pd.Timestamp("2024-01-01T02:30:00Z"),
        ]
    )
    args = run_paper_trading.build_parser().parse_args(
        [
            "--symbol",
            "BTC_USDT",
            "--interval",
            "Min60",
            "--out",
            str(out_dir),
            "--order-type",
            "market",
            "--size-mode",
            "units",
            "--size-value",
            "1",
            "--max-cycles",
            "2",
        ]
    )

    original_build_strategy = run_paper_trading._build_strategy
    original_replace = Path.replace
    replace_calls: list[str] = []

    def _tracking_replace(self: Path, target: Path):
        replace_calls.append(Path(target).name)
        return original_replace(self, target)

    run_paper_trading._build_strategy = lambda parsed_args: _RealtimeSignalStrategy()
    try:
        with patch.object(run_paper_trading.Path, "replace", _tracking_replace):
            run_paper_trading.run_from_args(
                args,
                client=client,
                now_provider=lambda: next(times),
                sleep_fn=lambda _seconds: None,
            )
    finally:
        run_paper_trading._build_strategy = original_build_strategy

    expected_files = {
        "paper_dashboard.html",
        "paper_dashboard_data.js",
        "paper_dashboard_data.json",
        "paper_fills.csv",
        "paper_status.md",
        "paper_summary.json",
        "paper_trades.md",
    }
    assert set(replace_calls) >= expected_files
