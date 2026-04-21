from __future__ import annotations

import importlib.util
import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from backtesting.engine import BacktestConfig, BacktestEngine
from backtesting.local_chart import (
    _ac_histogram_from_data,
    _alligator_series_from_data,
    _ao_histogram_from_data,
    _candles_from_data,
    _combine_markers,
    _execution_event_lines,
    _execution_trade_path_lines,
    _first_wiseman_engine_markers,
    _first_wiseman_ignored_markers,
    _second_wiseman_markers,
    _valid_third_wiseman_fractal_markers,
    _williams_zones_colors,
    _wiseman_fill_entry_markers,
    _wiseman_markers,
)
from backtesting.strategy import BWStrategy, WisemanStrategy

MODULE_PATH = Path(__file__).resolve().parents[1] / "LiveData" / "live_kcex_chart.py"
PAPER_TRADING_MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "run_paper_trading.py"


class _FakeKCEXClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def infer_top_assets(self, desired: int = 10) -> list[str]:
        return ["BTC_USDT", "ETH_USDT"][:desired]

    def infer_timeframes(self) -> list[str]:
        return ["Min1", "Min60"]

    def fetch_kline(self, symbol: str, interval: str, limit: int = 400):
        self.calls.append((symbol, interval, limit))
        base = 1_704_067_200
        return [
            live_kcex_chart.Candle(time=base, open=100.0, high=101.0, low=99.0, close=100.5),
            live_kcex_chart.Candle(time=base + 60, open=100.5, high=102.0, low=100.0, close=101.5),
        ]


def _tag_markers(markers, marker_group):
    return [{**marker, "markerGroup": marker_group} for marker in markers]


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


class _GreenGatorCloseOnBarStrategy:
    execute_on_signal_bar = True

    def __init__(self) -> None:
        self.signal_exit_reason: pd.Series | None = None
        self.signal_fill_prices: pd.Series | None = None

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=data.index, dtype="int8")
        exit_reason = pd.Series("", index=data.index, dtype="object")
        fill_prices = pd.Series(float("nan"), index=data.index, dtype="float64")
        if len(data) >= 2:
            signal.iloc[1] = -1
            fill_prices.iloc[1] = 102.0
        if len(data) >= 3:
            signal.iloc[2] = 0
            fill_prices.iloc[2] = float(data["close"].iloc[2])
            exit_reason.iloc[2] = "Green Gator PP"
        self.signal_fill_prices = fill_prices
        self.signal_exit_reason = exit_reason
        return signal


SPEC = importlib.util.spec_from_file_location("live_kcex_chart", MODULE_PATH)
live_kcex_chart = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = live_kcex_chart
assert SPEC.loader is not None
SPEC.loader.exec_module(live_kcex_chart)

PAPER_SPEC = importlib.util.spec_from_file_location("run_paper_trading", PAPER_TRADING_MODULE_PATH)
run_paper_trading = importlib.util.module_from_spec(PAPER_SPEC)
sys.modules[PAPER_SPEC.name] = run_paper_trading
assert PAPER_SPEC.loader is not None
PAPER_SPEC.loader.exec_module(run_paper_trading)


def test_candles_to_dataframe_sorts_and_uses_utc_index() -> None:
    candles = [
        {"time": 1704067320, "open": 2.0, "high": 3.0, "low": 1.5, "close": 2.5},
        {"time": 1704067200, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5},
    ]

    frame = live_kcex_chart._candles_to_dataframe(candles)

    assert list(frame.columns) == ["open", "high", "low", "close"]
    assert str(frame.index.tz) == "UTC"
    assert list(frame.index.astype("int64")) == sorted(frame.index.astype("int64").tolist())
    assert float(frame.iloc[0]["open"]) == 1.0
    assert float(frame.iloc[1]["close"]) == 2.5


def test_live_chart_payload_matches_backtest_chart_rendering_components() -> None:
    data = pd.read_csv("examples/sample_ohlcv.csv", parse_dates=["timestamp"]).set_index("timestamp")
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data.index = data.index.tz_convert("UTC")

    candles = [
        {
            "time": int(ts.timestamp()),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
        }
        for ts, row in data[["open", "high", "low", "close"]].iterrows()
    ]

    strategy = WisemanStrategy()
    payload = live_kcex_chart._build_live_chart_payload("BTC_USDT", "Min60", candles, strategy)

    replay_strategy, replay_execution_events, _ = live_kcex_chart._replay_strategy_execution(data, strategy)
    ao_histogram, ao_colors = _ao_histogram_from_data(data)
    ac_histogram, ac_colors = _ac_histogram_from_data(data)
    zone_colors = _williams_zones_colors(ao_colors, ac_colors)
    engine_first_markers = _first_wiseman_engine_markers(
        data,
        replay_strategy.signal_first_wiseman_setup_side,
        replay_strategy.signal_first_wiseman_ignored_reason,
        replay_strategy.signal_first_wiseman_reversal_side,
    )
    if engine_first_markers:
        wiseman_markers = []
    else:
        raw_markers = _wiseman_markers(data)
        wiseman_markers = [*raw_markers["bearish"], *raw_markers["bullish"]]

    expected_markers = _combine_markers(
        wiseman_markers,
        engine_first_markers,
        _first_wiseman_ignored_markers(data, replay_strategy.signal_first_wiseman_setup_side, replay_strategy.signal_first_wiseman_ignored_reason),
        _tag_markers(_second_wiseman_markers(data, replay_strategy.signal_fill_prices_second, replay_strategy.signal_second_wiseman_setup_side), "second_wiseman"),
        _tag_markers(_valid_third_wiseman_fractal_markers(data, replay_strategy.signal_third_wiseman_setup_side), "third_wiseman"),
        _tag_markers(_wiseman_fill_entry_markers(data, replay_strategy.signal_fill_prices_second, replay_strategy.signal_second_wiseman_fill_side, label="2W"), "second_wiseman_entry"),
        _tag_markers(_wiseman_fill_entry_markers(data, replay_strategy.signal_fill_prices_third, replay_strategy.signal_third_wiseman_fill_side, label="3W"), "third_wiseman_entry"),
        _tag_markers(live_kcex_chart._execution_event_markers(replay_execution_events, data), "execution"),
    )

    assert payload["symbol"] == "BTC_USDT"
    assert payload["interval"] == "Min60"
    assert payload["count"] == len(candles)
    assert payload["candles"] == _candles_from_data(data, zone_colors=zone_colors)
    assert payload["alligator"] == _alligator_series_from_data(data)
    assert payload["ao"] == ao_histogram
    assert payload["ac"] == ac_histogram
    assert payload["markers"] == expected_markers
    assert payload["trade_event_lines"] == _execution_event_lines(replay_execution_events, data.index)
    assert payload["trade_path_lines"] == _execution_trade_path_lines(replay_execution_events, data.index)
    marker_labels = {str(marker["text"]) for marker in payload["markers"]}
    assert "1W-R" in marker_labels
    assert marker_labels & {"1W-I", "1W-C", "1W-N", "1W-G", "1W-W", "2W"}
    assert any(label.startswith(("LE-", "SE-", "LX-", "SX-")) for label in marker_labels)
    marker_groups = {str(marker.get("markerGroup", "")) for marker in payload["markers"]}
    assert {"second_wiseman", "second_wiseman_entry", "third_wiseman_entry", "execution"} <= marker_groups


def test_active_live_data_store_refreshes_only_active_selection() -> None:
    client = _FakeKCEXClient()
    store = live_kcex_chart.ActiveLiveDataStore(client)

    first_snapshot = store.refresh_active_market_snapshot()
    assert client.calls == [("BTC_USDT", "Min1", 400)]
    assert first_snapshot["symbol"] == "BTC_USDT"
    assert first_snapshot["interval"] == "Min1"
    assert first_snapshot["market_revision"] == 1

    selection = store.set_active_selection("ETH_USDT", "Min60")
    assert selection.symbol == "ETH_USDT"
    assert selection.interval == "Min60"
    assert client.calls[-1] == ("ETH_USDT", "Min60", 400)

    current_snapshot = store.snapshot()
    assert current_snapshot["symbol"] == "ETH_USDT"
    assert current_snapshot["interval"] == "Min60"
    assert current_snapshot["selection_generation"] == 1
    assert current_snapshot["market_revision"] == 2


def test_live_chart_server_uses_active_pair_snapshot_flow() -> None:
    assert live_kcex_chart.ActiveLiveDataStore.refresh_market_forever.__defaults__ == (1.0,)
    assert live_kcex_chart.ActiveLiveDataStore.refresh_strategy_forever.__defaults__ == (0.25,)
    html = (MODULE_PATH.parent / "index.html").read_text(encoding="utf-8")
    direct_html = (MODULE_PATH.parent / "index.direct.html").read_text(encoding="utf-8")
    assert "/api/select" in html
    assert "/api/chart-state" in html
    assert "setInterval(() => refreshChart(), 1000);" in html
    assert "setInterval(refreshChart, 3000);" in direct_html


def test_live_chart_strategy_payload_respects_custom_wiseman_configuration() -> None:
    data = pd.read_csv("examples/sample_ohlcv.csv", parse_dates=["timestamp"]).set_index("timestamp")
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data.index = data.index.tz_convert("UTC")

    strategy = WisemanStrategy(first_wiseman_divergence_filter_bars=240)
    payload = live_kcex_chart._build_strategy_payload(data, strategy)
    _, replay_execution_events, _ = live_kcex_chart._replay_strategy_execution(data, strategy)

    expected_strategy = WisemanStrategy(first_wiseman_divergence_filter_bars=240)
    BacktestEngine(BacktestConfig(close_open_position_on_last_bar=False)).run(data, expected_strategy)
    expected_markers = _combine_markers(
        _first_wiseman_engine_markers(
            data,
            expected_strategy.signal_first_wiseman_setup_side,
            expected_strategy.signal_first_wiseman_ignored_reason,
            expected_strategy.signal_first_wiseman_reversal_side,
        ),
        _first_wiseman_ignored_markers(
            data,
            expected_strategy.signal_first_wiseman_setup_side,
            expected_strategy.signal_first_wiseman_ignored_reason,
        ),
        _tag_markers(
            _second_wiseman_markers(data, expected_strategy.signal_fill_prices_second, expected_strategy.signal_second_wiseman_setup_side),
            "second_wiseman",
        ),
        _tag_markers(
            _valid_third_wiseman_fractal_markers(data, expected_strategy.signal_third_wiseman_setup_side),
            "third_wiseman",
        ),
        _tag_markers(
            _wiseman_fill_entry_markers(data, expected_strategy.signal_fill_prices_second, expected_strategy.signal_second_wiseman_fill_side, label="2W"),
            "second_wiseman_entry",
        ),
        _tag_markers(
            _wiseman_fill_entry_markers(data, expected_strategy.signal_fill_prices_third, expected_strategy.signal_third_wiseman_fill_side, label="3W"),
            "third_wiseman_entry",
        ),
        _tag_markers(live_kcex_chart._execution_event_markers(replay_execution_events, data), "execution"),
    )

    assert payload["markers"] == expected_markers
    marker_labels = {str(marker["text"]) for marker in payload["markers"]}
    assert "1W-D" in marker_labels


def test_kcex_client_retries_transient_url_errors() -> None:
    client = live_kcex_chart.KCEXClient(
        base_url="https://example.invalid",
        timeout=1,
        request_retries=3,
        retry_backoff_seconds=0.0,
        retry_backoff_max_seconds=0.0,
    )
    responses = [
        urllib.error.URLError("reset"),
        urllib.error.URLError("reset"),
        '{"data":[{"time":[1704067200],"open":[100],"high":[101],"low":[99],"close":[100.5]}]}',
    ]

    class _Response:
        def __init__(self, payload: str) -> None:
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return self.payload.encode("utf-8")

    def _fake_urlopen(_req, timeout, context=None):
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return _Response(item)

    with patch.object(live_kcex_chart.urllib.request, "urlopen", side_effect=_fake_urlopen):
        candles = client.fetch_kline("BTC_USDT", "Min1", limit=5)

    assert len(candles) == 1
    assert candles[0].close == 100.5


def test_live_chart_strategy_payload_includes_gator_profit_protection_fallback_markers() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
        },
        index=idx,
    )

    payload = live_kcex_chart._build_strategy_payload(data, _DelayedFlatExitStrategy())

    fallback_markers = [
        marker
        for marker in payload["markers"]
        if marker.get("markerGroup") == "gator_profit_protection_fallback"
    ]
    assert {(marker["time"], marker["text"]) for marker in fallback_markers} == {
        (int(idx[2].timestamp()), "LE-1W"),
        (int(idx[3].timestamp()), "LX-R"),
    }
    assert [line["label"] for line in payload["trade_event_lines"] if line["label"] in {"LE", "LX"}] == ["LE", "LX"]


def test_live_chart_strategy_payload_skips_profit_protection_fallback_for_bw_strategy() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
        },
        index=idx,
    )

    payload = live_kcex_chart._build_strategy_payload(data, BWStrategy())

    assert not [
        marker
        for marker in payload["markers"]
        if marker.get("markerGroup") == "gator_profit_protection_fallback"
    ]
    assert not [
        line
        for line in payload["trade_event_lines"]
        if line.get("label") in {"LE", "SE", "LX", "SX"}
    ]


def test_live_chart_green_gator_profit_protection_exit_executes_on_signal_bar_close() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 101.0, 104.0],
            "high": [101.0, 102.0, 106.0],
            "low": [99.0, 100.0, 98.0],
            "close": [100.5, 101.5, 99.0],
        },
        index=idx,
    )

    _strategy, execution_events, completed_trades = live_kcex_chart._replay_strategy_execution(
        data,
        _GreenGatorCloseOnBarStrategy(),
    )

    assert len(completed_trades) == 1
    assert len(execution_events) == 2
    assert execution_events[0].event_type == "entry"
    assert execution_events[0].price == 101.9796
    assert execution_events[1].event_type == "exit"
    assert execution_events[1].time == idx[2]
    assert execution_events[1].price == 99.0198


def test_live_chart_build_strategy_matches_paper_trading_profit_protection_flags() -> None:
    args = run_paper_trading.build_parser().parse_args([
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
        "--1W-divergence-filter",
        "240",
    ])

    live_strategy = live_kcex_chart._build_strategy(args)
    paper_strategy = run_paper_trading._build_strategy(args)
    live_wiseman = next(
        strategy
        for strategy in live_strategy.strategies
        if isinstance(strategy, run_paper_trading.WisemanStrategy)
    )

    assert live_wiseman.teeth_profit_protection_enabled is True
    assert live_wiseman.teeth_profit_protection_min_bars == 15
    assert live_wiseman.teeth_profit_protection_min_unrealized_return == 0.01
    assert live_wiseman.teeth_profit_protection_require_gator_open is False
    assert live_wiseman.profit_protection_volatility_lookback == 40
    assert live_wiseman.profit_protection_annualized_volatility_scaler == 0.85
    assert live_wiseman.lips_profit_protection_enabled is True
    assert live_wiseman.lips_profit_protection_volatility_trigger == 0.04
    assert live_wiseman.lips_profit_protection_profit_trigger_mult == 5.0
    assert live_wiseman.lips_profit_protection_volatility_lookback == 30
    assert live_wiseman.lips_profit_protection_recent_trade_lookback == 10
    assert live_wiseman.lips_profit_protection_min_unrealized_return == 0.03
    assert live_wiseman.lips_profit_protection_arm_on_min_unrealized_return is True
    assert live_wiseman.first_wiseman_divergence_filter_bars == 240

    for attr in [
        "teeth_profit_protection_enabled",
        "teeth_profit_protection_min_bars",
        "teeth_profit_protection_min_unrealized_return",
        "teeth_profit_protection_require_gator_open",
        "profit_protection_volatility_lookback",
        "profit_protection_annualized_volatility_scaler",
        "lips_profit_protection_enabled",
        "lips_profit_protection_volatility_trigger",
        "lips_profit_protection_profit_trigger_mult",
        "lips_profit_protection_volatility_lookback",
        "lips_profit_protection_recent_trade_lookback",
        "lips_profit_protection_min_unrealized_return",
        "lips_profit_protection_arm_on_min_unrealized_return",
        "first_wiseman_divergence_filter_bars",
    ]:
        assert getattr(live_wiseman, attr) == getattr(paper_strategy, attr)


def test_live_chart_strategy_payload_tracks_delayed_flat_exit_like_paper_engine() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
        },
        index=idx,
    )

    payload = live_kcex_chart._build_strategy_payload(data, _DelayedFlatExitStrategy())

    marker_labels = {str(marker["text"]) for marker in payload["markers"]}
    execution_markers = [marker for marker in payload["markers"] if marker.get("markerGroup") == "execution"]

    assert {"LE-1W", "LX-R"} <= marker_labels
    assert {str(marker["text"]) for marker in execution_markers} == {"LE-1W", "LX-R"}
    assert payload["trade_event_lines"]
    assert payload["trade_path_lines"]


def test_execution_event_markers_include_compact_strategy_suffixes() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
        },
        index=idx,
    )
    events = [
        live_kcex_chart.ExecutionEvent(
            event_type="entry",
            time=idx[0],
            side="buy",
            price=100.5,
            units=1.0,
            strategy_reason="Bullish Fractal",
        ),
        live_kcex_chart.ExecutionEvent(
            event_type="exit",
            time=idx[1],
            side="sell",
            price=101.5,
            units=1.0,
            strategy_reason="Strategy Profit Protection Green Gator",
        ),
        live_kcex_chart.ExecutionEvent(
            event_type="entry",
            time=idx[1],
            side="sell",
            price=101.5,
            units=1.0,
            strategy_reason="Bearish 3W",
        ),
        live_kcex_chart.ExecutionEvent(
            event_type="exit",
            time=idx[2],
            side="buy",
            price=102.5,
            units=1.0,
            strategy_reason="NTD Entry Stop",
        ),
    ]

    markers = live_kcex_chart._execution_event_markers(events, data)

    assert [str(marker["text"]) for marker in markers] == ["LE-F", "LX-G/SE", "SX-S"]
    assert [str(marker["position"]) for marker in markers] == ["belowBar", "aboveBar", "belowBar"]


def test_ntd_markers_label_add_ons_for_live_and_paper_charts() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
        },
        index=idx,
    )
    fill_prices = pd.Series([101.0, 102.0, float("nan")], index=idx, dtype="float64")
    fractal_side = pd.Series([1, 1, 0], index=idx, dtype="int8")
    contracts = pd.Series([1.0, 2.0, 2.0], index=idx, dtype="float64")

    live_markers = live_kcex_chart._ntd_fill_entry_markers(data, fill_prices, fractal_side, contracts)
    paper_markers = run_paper_trading._ntd_fill_entry_markers(data, fill_prices, fractal_side, contracts)

    assert [str(marker["text"]) for marker in live_markers] == ["NTD-E", "NTD-A"]
    assert [str(marker["text"]) for marker in paper_markers] == ["NTD-E", "NTD-A"]


def test_build_signal_scale_replaces_existing_orders_to_prevent_dual_side_books() -> None:
    ts = pd.Timestamp("2026-01-01T00:00:00Z")
    signal = live_kcex_chart._build_signal(
        symbol="BTC_USDT",
        timestamp=ts,
        target_qty=3.0,
        current_qty=1.0,
        projected_qty=1.0,
        order_request={"order_type": "stop", "limit_price": None, "stop_price": 101.0},
        signal_fill=101.0,
    )
    assert signal is not None
    assert signal.action == "scale"
    assert signal.side == "buy"
    assert signal.quantity == 2.0
    assert signal.cancel_existing_orders is True


def test_build_signal_reduce_cancels_stale_add_orders_before_submitting_opposite_side() -> None:
    ts = pd.Timestamp("2026-01-01T00:00:00Z")
    signal = live_kcex_chart._build_signal(
        symbol="BTC_USDT",
        timestamp=ts,
        target_qty=2.0,
        current_qty=5.0,
        projected_qty=7.0,
        order_request={"order_type": "stop", "limit_price": None, "stop_price": 99.0},
        signal_fill=99.0,
    )
    assert signal is not None
    assert signal.action == "exit"
    assert signal.side is None
    assert signal.quantity == 3.0
    assert signal.cancel_existing_orders is True
