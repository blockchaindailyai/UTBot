from __future__ import annotations

import argparse
import copy
import json
import ssl
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import error as urllib_error

import numpy as np
import pandas as pd

try:
    import certifi
except ImportError:  # pragma: no cover - optional dependency
    certifi = None

from backtesting import ExecutionSignal, LiveBar, PaperTradingEngine
from backtesting.engine import BacktestConfig, BacktestEngine, ExecutionEvent
from backtesting.local_chart import (
    _ac_histogram_from_data,
    _alligator_series_from_data,
    _ao_histogram_from_data,
    _candles_from_data,
    _combine_markers,
    _compact_trade_reason,
    _execution_event_lines,
    _execution_trade_path_lines,
    _first_wiseman_engine_markers,
    _first_wiseman_ignored_markers,
    _gator_profit_protection_fallback_overlays,
    _second_wiseman_markers,
    _valid_third_wiseman_fractal_markers,
    _williams_zones_colors,
    _wiseman_fill_entry_markers,
    _wiseman_markers,
)
from backtesting.stats import infer_periods_per_year
from backtesting.strategy import BWStrategy, CombinedStrategy, NTDStrategy, Strategy, WisemanStrategy

KCEX_BASE = "https://www.kcex.com"

COMMON_TIMEFRAMES = ["Min1", "Min3", "Min5", "Min15", "Min30", "Min60", "Hour4", "Hour12", "Day1"]

INTERVAL_SECONDS = {
    "Min1": 60,
    "Min3": 180,
    "Min5": 300,
    "Min15": 900,
    "Min30": 1800,
    "Min60": 3600,
    "Hour4": 14400,
    "Hour12": 43200,
    "Day1": 86400,
}

FALLBACK_ASSETS = [
    "BTC_USDT",
    "ETH_USDT",
    "SOL_USDT",
    "XRP_USDT",
    "DOGE_USDT",
    "BNB_USDT",
    "ADA_USDT",
    "TRX_USDT",
    "AVAX_USDT",
    "LINK_USDT",
]


@dataclass
class Candle:
    time: int
    open: float
    high: float
    low: float
    close: float


@dataclass
class SelectionSnapshot:
    symbol: str
    interval: str
    generation: int


def normalize_epoch_seconds(value: Any) -> int:
    ts = int(float(value))
    if ts > 10_000_000_000:  # milliseconds input
        ts //= 1000
    return ts


class KCEXClient:
    def __init__(
        self,
        base_url: str = KCEX_BASE,
        timeout: int = 15,
        request_retries: int = 5,
        retry_backoff_seconds: float = 0.75,
        retry_backoff_max_seconds: float = 8.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.request_retries = max(1, int(request_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.retry_backoff_max_seconds = max(self.retry_backoff_seconds, float(retry_backoff_max_seconds))
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        self.ssl_context = self._build_ssl_context()

    @staticmethod
    def _build_ssl_context() -> ssl.SSLContext:
        if certifi is not None:
            return ssl.create_default_context(cafile=certifi.where())
        return ssl.create_default_context()

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = f"?{urllib.parse.urlencode(params)}" if params else ""
        url = f"{self.base_url}{path}{query}"
        req = urllib.request.Request(url, headers=self.headers)
        last_error: BaseException | None = None
        for attempt in range(1, self.request_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout, context=self.ssl_context) as response:
                    payload = response.read().decode("utf-8")
                return json.loads(payload)
            except urllib_error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                last_error = exc
                if not retryable or attempt >= self.request_retries:
                    break
            except (
                urllib_error.URLError,
                ConnectionResetError,
                TimeoutError,
                ssl.SSLError,
                OSError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                if attempt >= self.request_retries:
                    break
            if self.retry_backoff_seconds > 0:
                backoff = min(self.retry_backoff_seconds * (2 ** (attempt - 1)), self.retry_backoff_max_seconds)
                time.sleep(backoff)
        raise RuntimeError(f"KCEX request failed after {self.request_retries} attempts for {url}") from last_error

    def _parse_kline_rows(self, data: Any, limit: int) -> list[Candle]:
        payload = data.get("data") if isinstance(data, dict) and "data" in data else data

        blocks: list[dict[str, Any]] = []
        if isinstance(payload, list):
            if payload and isinstance(payload[0], dict) and "time" in payload[0]:
                blocks = [p for p in payload if isinstance(p, dict)]
        elif isinstance(payload, dict) and "time" in payload:
            blocks = [payload]

        rows: list[Candle] = []
        for block in blocks:
            times = block.get("time") or []
            opens = block.get("open") or []
            highs = block.get("high") or []
            lows = block.get("low") or []
            closes = block.get("close") or []
            count = min(len(times), len(opens), len(highs), len(lows), len(closes))
            for idx in range(count):
                try:
                    o = float(opens[idx])
                    h = float(highs[idx])
                    l = float(lows[idx])
                    c = float(closes[idx])
                    t = normalize_epoch_seconds(times[idx])
                except (TypeError, ValueError):
                    continue
                rows.append(Candle(time=t, open=o, high=h, low=l, close=c))

        rows.sort(key=lambda c: c.time)

        dedup: dict[int, Candle] = {}
        for row in rows:
            dedup[row.time] = row
        final_rows = sorted(dedup.values(), key=lambda c: c.time)
        return final_rows[-limit:]

    def fetch_kline(self, symbol: str, interval: str, limit: int = 400) -> list[Candle]:
        end = int(time.time())
        span_per_bar = INTERVAL_SECONDS.get(interval, 3600)
        start = end - (limit * span_per_bar)
        data = self._get_json(
            f"/fapi/v1/contract/kline/{symbol}",
            {"interval": interval, "start": start, "end": end},
        )
        return self._parse_kline_rows(data, limit=limit)

    def infer_top_assets(self, desired: int = 10) -> list[str]:
        candidates: list[tuple[str, float]] = []
        for path in ["/fapi/v1/contract/ticker", "/fapi/v1/contract/symbols", "/fapi/v1/contract/detail"]:
            try:
                raw = self._get_json(path)
            except Exception:
                continue
            items: list[dict[str, Any]] = []
            if isinstance(raw, dict) and isinstance(raw.get("data"), list):
                items = [x for x in raw["data"] if isinstance(x, dict)]
            elif isinstance(raw, list):
                items = [x for x in raw if isinstance(x, dict)]
            for item in items:
                symbol = str(item.get("symbol") or item.get("contractCode") or item.get("pair") or "").upper().strip()
                if not symbol or "USDT" not in symbol:
                    continue
                if "_" not in symbol and symbol.endswith("USDT"):
                    symbol = symbol.replace("USDT", "_USDT")
                score = 0.0
                for field in ["amount24", "quoteVolume", "turnover24h", "volValue24", "volume", "vol"]:
                    try:
                        score = max(score, float(item.get(field, 0) or 0))
                    except (TypeError, ValueError):
                        pass
                candidates.append((symbol, score))

        if not candidates:
            return FALLBACK_ASSETS[:desired]

        ranked: list[str] = []
        seen: set[str] = set()
        for symbol, _score in sorted(candidates, key=lambda x: x[1], reverse=True):
            if symbol in seen:
                continue
            seen.add(symbol)
            ranked.append(symbol)
            if len(ranked) >= desired:
                break
        return ranked or FALLBACK_ASSETS[:desired]

    def infer_timeframes(self) -> list[str]:
        valid: list[str] = []
        for tf in COMMON_TIMEFRAMES:
            try:
                if self.fetch_kline("BTC_USDT", tf, limit=3):
                    valid.append(tf)
            except Exception:
                continue
        return valid or COMMON_TIMEFRAMES


def _candles_to_dataframe(candles: list[dict[str, float | int]]) -> pd.DataFrame:
    frame = pd.DataFrame(candles)
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    frame = frame.set_index("time").sort_index()
    return frame[["open", "high", "low", "close"]].astype("float64")


def _closed_bars(data: pd.DataFrame, interval: str, now_ts: int | None = None) -> pd.DataFrame:
    if data.empty:
        return data
    if now_ts is None:
        now_ts = int(time.time())
    interval_seconds = INTERVAL_SECONDS.get(interval, 3600)
    cutoff = pd.Timestamp(now_ts - interval_seconds, unit="s", tz="UTC")
    return data[data.index <= cutoff]


def _tag_markers(markers: list[dict[str, str | int | float]], marker_group: str) -> list[dict[str, str | int | float]]:
    return [{**marker, "markerGroup": marker_group} for marker in markers]


def _ntd_fill_entry_markers(
    data: pd.DataFrame,
    fill_prices: pd.Series | None,
    fractal_position_side: pd.Series | None,
    contracts: pd.Series | None = None,
) -> list[dict[str, str | int | float]]:
    if data.empty or not isinstance(fill_prices, pd.Series) or not isinstance(fractal_position_side, pd.Series):
        return []
    fills = fill_prices.reindex(data.index)
    sides = fractal_position_side.reindex(data.index).fillna(0).astype("int8")
    aligned_contracts = contracts.reindex(data.index).fillna(0.0).astype("float64") if isinstance(contracts, pd.Series) else None
    markers: list[dict[str, str | int | float]] = []
    for i, ts in enumerate(data.index):
        side = int(sides.iloc[i])
        fill = fills.iloc[i]
        if side == 0 or pd.isna(fill):
            continue
        is_add_on = False
        if aligned_contracts is not None:
            current_contracts = float(aligned_contracts.iloc[i])
            previous_contracts = float(aligned_contracts.iloc[i - 1]) if i > 0 else 0.0
            is_add_on = (
                int(np.sign(previous_contracts)) == side
                and abs(current_contracts) > (abs(previous_contracts) + 1e-12)
            )
        row = data.iloc[i]
        row_low = float(row["low"])
        row_high = float(row["high"])
        row_close = float(row["close"])
        marker_offset = max((row_high - row_low) * 0.35, abs(row_close) * 0.001, 1e-8)
        bullish = side > 0
        markers.append(
            {
                "time": int(ts.timestamp()),
                "position": "belowBar" if bullish else "aboveBar",
                "price": row_low - marker_offset if bullish else row_high + marker_offset,
                "color": "#0ea5e9" if bullish else "#f97316",
                "shape": "arrowUp" if bullish else "arrowDown",
                "text": "NTD-A" if is_add_on else "NTD-E",
            }
        )
    return markers


def _strategy_has_third_wiseman_enabled(strategy: Strategy) -> bool:
    if isinstance(strategy, WisemanStrategy):
        return bool(strategy.third_wiseman_contracts > 0)
    if isinstance(strategy, CombinedStrategy):
        return any(_strategy_has_third_wiseman_enabled(component) for component in strategy.strategies)
    return False


def _strategy_includes_bw(strategy: Strategy) -> bool:
    if isinstance(strategy, BWStrategy):
        return True
    if isinstance(strategy, CombinedStrategy):
        return any(_strategy_includes_bw(component) for component in strategy.strategies)
    return False


def _resolve_contracts(signal_value: float, signal_contracts: pd.Series | None, signal_index: int) -> float:
    if int(np.sign(signal_value)) == 0:
        return 0.0
    if signal_contracts is not None:
        raw = abs(float(signal_contracts.iloc[signal_index]))
    else:
        raw = abs(float(signal_value))
    return raw if raw > 0 else 1.0


def _resolve_order_request(*, config: BacktestConfig, side: int, prev_close: float, signal_fill: float | None) -> dict[str, float | str | None]:
    order_type = config.order_type.lower()
    if order_type == "market":
        return {"order_type": "market", "limit_price": None, "stop_price": None}

    if signal_fill is not None and np.isfinite(signal_fill) and signal_fill > 0:
        if order_type == "limit":
            return {"order_type": "limit", "limit_price": float(signal_fill), "stop_price": None}
        if order_type == "stop":
            return {"order_type": "stop", "limit_price": None, "stop_price": float(signal_fill)}
        stop_price = float(signal_fill)
    else:
        if order_type == "limit":
            limit_price = prev_close * (1 - config.limit_offset_pct if side > 0 else 1 + config.limit_offset_pct)
            return {"order_type": "limit", "limit_price": float(limit_price), "stop_price": None}
        stop_price = prev_close * (1 + config.stop_offset_pct if side > 0 else 1 - config.stop_offset_pct)
        if order_type == "stop":
            return {"order_type": "stop", "limit_price": None, "stop_price": float(stop_price)}

    limit_price = stop_price * (1 + config.stop_limit_offset_pct if side > 0 else 1 - config.stop_limit_offset_pct)
    return {"order_type": "stop_limit", "limit_price": float(limit_price), "stop_price": float(stop_price)}


def _signed_open_order_projection(engine: PaperTradingEngine) -> float:
    projected_qty = engine.position.quantity
    for order in engine.open_orders:
        signed_qty = order.quantity if order.side == "buy" else -order.quantity
        if order.reduce_only:
            if projected_qty > 0:
                projected_qty = max(0.0, projected_qty + signed_qty)
            elif projected_qty < 0:
                projected_qty = min(0.0, projected_qty + signed_qty)
        else:
            projected_qty += signed_qty
    return float(projected_qty)


def _build_signal(
    *,
    symbol: str,
    timestamp: pd.Timestamp,
    target_qty: float,
    current_qty: float,
    projected_qty: float,
    order_request: dict[str, float | str | None],
    signal_fill: float | None,
) -> ExecutionSignal | None:
    if np.isclose(target_qty, projected_qty, atol=1e-12):
        return None

    current_side = int(np.sign(current_qty))
    target_side = int(np.sign(target_qty))
    target_abs = abs(float(target_qty))
    current_abs = abs(float(current_qty))
    signal_metadata = (
        {"market_fill_price": float(signal_fill)}
        if str(order_request["order_type"]) == "market" and signal_fill is not None and np.isfinite(signal_fill) and signal_fill > 0
        else {}
    )

    if target_side == 0:
        if current_side == 0:
            return None
        return ExecutionSignal(
            symbol=symbol,
            timestamp=timestamp,
            action="exit",
            order_type=str(order_request["order_type"]),
            quantity=current_abs,
            limit_price=order_request["limit_price"],
            stop_price=order_request["stop_price"],
            cancel_existing_orders=True,
            metadata=signal_metadata,
        )

    side_label = "buy" if target_side > 0 else "sell"
    if current_side == 0:
        return ExecutionSignal(
            symbol=symbol,
            timestamp=timestamp,
            action="enter",
            side=side_label,
            order_type=str(order_request["order_type"]),
            quantity=target_abs,
            limit_price=order_request["limit_price"],
            stop_price=order_request["stop_price"],
            cancel_existing_orders=True,
            metadata=signal_metadata,
        )

    if current_side != target_side:
        return ExecutionSignal(
            symbol=symbol,
            timestamp=timestamp,
            action="reverse",
            side=side_label,
            order_type=str(order_request["order_type"]),
            quantity=target_abs,
            limit_price=order_request["limit_price"],
            stop_price=order_request["stop_price"],
            cancel_existing_orders=True,
            metadata=signal_metadata,
        )

    if target_abs > current_abs:
        return ExecutionSignal(
            symbol=symbol,
            timestamp=timestamp,
            action="scale",
            side=side_label,
            order_type=str(order_request["order_type"]),
            quantity=target_abs - current_abs,
            limit_price=order_request["limit_price"],
            stop_price=order_request["stop_price"],
            cancel_existing_orders=True,
            metadata=signal_metadata,
        )

    return ExecutionSignal(
        symbol=symbol,
        timestamp=timestamp,
        action="exit",
        order_type=str(order_request["order_type"]),
        quantity=current_abs - target_abs,
        limit_price=order_request["limit_price"],
        stop_price=order_request["stop_price"],
        cancel_existing_orders=True,
        metadata=signal_metadata,
    )


def _strategy_reasons_for_signal(
    strategy: Strategy,
    engine: BacktestEngine,
    *,
    signal_index: int,
    current_qty: float,
    target_qty: float,
) -> tuple[str | None, str | None]:
    current_side = int(np.sign(current_qty))
    target_side = int(np.sign(target_qty))

    entry_reason: str | None = None
    exit_reason: str | None = None

    if target_side != 0:
        entry_reason = str(engine._infer_entry_signal(strategy, signal_index=signal_index, desired_position=target_side))

    if current_side != 0 and (target_side == 0 or target_side != current_side):
        exit_reason = str(
            engine._infer_exit_signal(
                strategy,
                signal_index=signal_index,
                prior_position=current_side,
                desired_position=target_side,
            )
        )

    if current_side != 0 and target_side == current_side and abs(target_qty) < abs(current_qty):
        exit_reason = exit_reason or "Reduce"

    return entry_reason, exit_reason


def _resolved_signal_contracts(signals: pd.Series, signal_contracts: pd.Series | None) -> pd.Series:
    return pd.Series(
        [_resolve_contracts(float(signals.iloc[i]), signal_contracts, i) for i in range(len(signals))],
        index=signals.index,
        dtype="float64",
    )


def _latest_changed_signal_index(
    *,
    signals: pd.Series,
    resolved_contracts: pd.Series,
    previous_signals: pd.Series,
    previous_contracts: pd.Series,
    max_signal_index: int,
) -> int | None:
    aligned_previous_signals = previous_signals.reindex(signals.index).fillna(0.0)
    aligned_previous_contracts = previous_contracts.reindex(signals.index).fillna(0.0)
    previous_index = set(previous_signals.index)

    for i in range(max_signal_index, -1, -1):
        current_side = int(np.sign(float(signals.iloc[i])))
        previous_side = int(np.sign(float(aligned_previous_signals.iloc[i])))
        current_contracts = float(resolved_contracts.iloc[i]) if current_side != 0 else 0.0
        previous_contracts_value = float(aligned_previous_contracts.iloc[i]) if previous_side != 0 else 0.0
        ts = signals.index[i]
        if ts not in previous_index:
            prior_current_side = int(np.sign(float(signals.iloc[i - 1]))) if i > 0 else 0
            prior_current_contracts = float(resolved_contracts.iloc[i - 1]) if i > 0 and prior_current_side != 0 else 0.0
            if current_side != prior_current_side or abs(current_contracts - prior_current_contracts) > 1e-12:
                return i
            continue
        if current_side != previous_side or abs(current_contracts - previous_contracts_value) > 1e-12:
            return i
    return None


def _fills_dataframe(engine: PaperTradingEngine) -> pd.DataFrame:
    snapshot = engine.snapshot()
    return pd.DataFrame(
        [
            {
                "timestamp": fill.timestamp,
                "side": fill.side,
                "quantity": fill.quantity,
                "price": fill.price,
                "reason": fill.strategy_reason or fill.reason,
                "execution_reason": fill.reason,
                "realized_pnl": fill.realized_pnl,
            }
            for fill in snapshot.fills
        ]
    )


@dataclass
class _ExecutionMarkerTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str


def _completed_trades_from_fills(fills: pd.DataFrame) -> list[_ExecutionMarkerTrade]:
    if fills.empty:
        return []

    fills_sorted = fills.copy().sort_values("timestamp").reset_index(drop=True)
    open_qty = 0.0
    entry_time: pd.Timestamp | None = None
    trades: list[_ExecutionMarkerTrade] = []

    for row in fills_sorted.itertuples(index=False):
        timestamp = pd.Timestamp(row.timestamp)
        signed_qty = float(row.quantity) if row.side == "buy" else -float(row.quantity)
        if np.isclose(open_qty, 0.0, atol=1e-12):
            open_qty = signed_qty
            entry_time = timestamp
            continue

        same_direction = np.sign(open_qty) == np.sign(signed_qty)
        if same_direction:
            open_qty += signed_qty
            continue

        closed_qty = min(abs(open_qty), abs(signed_qty))
        if closed_qty > 0 and entry_time is not None:
            trades.append(
                _ExecutionMarkerTrade(
                    entry_time=entry_time,
                    exit_time=timestamp,
                    side="long" if open_qty > 0 else "short",
                )
            )

        leftover = abs(signed_qty) - closed_qty
        if leftover > 1e-12:
            open_qty = np.sign(signed_qty) * leftover
            entry_time = timestamp
        else:
            remaining = abs(open_qty) - closed_qty
            if remaining > 1e-12:
                open_qty = np.sign(open_qty) * remaining
            else:
                open_qty = 0.0
                entry_time = None

    return trades


def _execution_event_markers(
    execution_events: list[ExecutionEvent],
    data: pd.DataFrame,
) -> list[dict[str, str | int | float]]:
    marker_buckets: dict[tuple[int, str], dict[str, str | int | float]] = {}
    for event in execution_events:
        event_type = str(event.event_type)
        if event_type not in {"entry", "add", "reduce", "exit"}:
            continue
        ts = pd.Timestamp(event.time)
        if ts not in data.index:
            continue
        row = data.loc[ts]
        row_low = float(row["low"])
        row_high = float(row["high"])
        row_close = float(row["close"])
        marker_offset = max((row_high - row_low) * 0.35, abs(row_close) * 0.001, 1e-8)
        side = str(event.side)
        is_buy = side == "buy"
        if event_type in {"entry", "add"}:
            prefix = "LE" if is_buy else "SE"
        else:
            prefix = "SX" if is_buy else "LX"
        compact_reason = _compact_trade_reason(getattr(event, "strategy_reason", None))
        label = f"{prefix}-{compact_reason}" if compact_reason else prefix
        position = "belowBar" if prefix in {"LE", "SX"} else "aboveBar"
        price = row_low - marker_offset if position == "belowBar" else row_high + marker_offset
        color = "#16a34a" if position == "belowBar" else "#dc2626"
        shape = "arrowUp" if position == "belowBar" else "arrowDown"
        key = (int(ts.timestamp()), position)
        existing = marker_buckets.get(key)
        if existing is None:
            marker_buckets[key] = {
                "time": key[0],
                "position": position,
                "price": price,
                "color": color,
                "shape": shape,
                "text": label,
            }
            continue
        existing_text = str(existing.get("text", ""))
        if label not in existing_text.split("/"):
            existing["text"] = f"{existing_text}/{label}" if existing_text else label
    return sorted(marker_buckets.values(), key=lambda marker: (int(marker["time"]), str(marker["position"])))


def _paper_execution_events(fills: pd.DataFrame) -> list[ExecutionEvent]:
    if fills.empty:
        return []

    fills_sorted = fills.copy()
    fills_sorted["timestamp"] = pd.to_datetime(fills_sorted["timestamp"], utc=True)
    fills_sorted = fills_sorted.sort_values("timestamp").reset_index(drop=True)

    events: list[ExecutionEvent] = []
    open_qty = 0.0

    for row in fills_sorted.itertuples(index=False):
        timestamp = pd.Timestamp(row.timestamp)
        side = str(row.side)
        signed_qty = float(row.quantity) if side == "buy" else -float(row.quantity)
        next_qty = open_qty + signed_qty
        fill_price = float(row.price)
        strategy_reason = str(getattr(row, "reason", "") or "").strip() or None

        if np.isclose(open_qty, 0.0, atol=1e-12):
            if not np.isclose(next_qty, 0.0, atol=1e-12):
                events.append(
                    ExecutionEvent(
                        event_type="entry",
                        time=timestamp,
                        side=side,
                        price=fill_price,
                        units=abs(signed_qty),
                        strategy_reason=strategy_reason,
                    )
                )
            open_qty = next_qty
            continue

        if np.sign(open_qty) == np.sign(next_qty) and abs(next_qty) > abs(open_qty):
            events.append(
                ExecutionEvent(
                    event_type="add",
                    time=timestamp,
                    side=side,
                    price=fill_price,
                    units=abs(signed_qty),
                    strategy_reason=strategy_reason,
                )
            )
            open_qty = next_qty
            continue

        if np.sign(open_qty) == np.sign(next_qty) and abs(next_qty) < abs(open_qty):
            event_type = "exit" if np.isclose(next_qty, 0.0, atol=1e-12) else "reduce"
            events.append(
                ExecutionEvent(
                    event_type=event_type,
                    time=timestamp,
                    side=side,
                    price=fill_price,
                    units=abs(signed_qty),
                    strategy_reason=strategy_reason,
                )
            )
            open_qty = next_qty
            continue

        if np.sign(open_qty) != np.sign(next_qty) and not np.isclose(next_qty, 0.0, atol=1e-12):
            exit_units = abs(open_qty)
            entry_units = abs(next_qty)
            if exit_units > 1e-12:
                events.append(
                    ExecutionEvent(
                        event_type="exit",
                        time=timestamp,
                        side=side,
                        price=fill_price,
                        units=exit_units,
                        strategy_reason=strategy_reason,
                    )
                )
            if entry_units > 1e-12:
                events.append(
                    ExecutionEvent(
                        event_type="entry",
                        time=timestamp,
                        side=side,
                        price=fill_price,
                        units=entry_units,
                        strategy_reason=strategy_reason,
                    )
                )
            open_qty = next_qty
            continue

        event_type = "exit" if np.isclose(next_qty, 0.0, atol=1e-12) else "reduce"
        events.append(
            ExecutionEvent(
                event_type=event_type,
                time=timestamp,
                side=side,
                price=fill_price,
                units=abs(signed_qty),
                strategy_reason=strategy_reason,
            )
        )
        open_qty = next_qty

    return events


def _replay_strategy_execution(
    data: pd.DataFrame,
    strategy_template: Strategy | None,
    *,
    include_intrabar_preview: bool = False,
) -> tuple[Strategy, list[ExecutionEvent], list[_ExecutionMarkerTrade]]:
    strategy = (
        copy.deepcopy(strategy_template)
        if strategy_template is not None
        else CombinedStrategy([WisemanStrategy(), NTDStrategy()])
    )
    config = BacktestConfig(close_open_position_on_last_bar=False)
    sizing_engine = BacktestEngine(config)
    execution_engine = PaperTradingEngine(
        symbol="LIVE_CHART",
        initial_cash=config.initial_capital,
        fee_rate=config.fee_rate,
        slippage_rate=config.slippage_rate,
        spread_rate=config.spread_rate,
    )
    periods_per_year = infer_periods_per_year(data.index, default=252)

    initial_history = data.iloc[:1].copy()
    raw_initial_signals = strategy.generate_signals(initial_history)
    initial_signals = raw_initial_signals.reindex(initial_history.index).fillna(0)
    initial_signal_contracts = getattr(strategy, "signal_contracts", None)
    if isinstance(initial_signal_contracts, pd.Series):
        initial_signal_contracts = initial_signal_contracts.reindex(initial_history.index).fillna(0.0)
    else:
        initial_signal_contracts = None
    last_strategy_signals = initial_signals.astype("float64").copy()
    last_strategy_contracts = _resolved_signal_contracts(initial_signals, initial_signal_contracts)

    for bar_index in range(1, len(data)):
        history = data.iloc[: bar_index + 1].copy()
        raw_signals = strategy.generate_signals(history)
        signals = raw_signals.reindex(history.index).fillna(0)
        signal_fill_prices = getattr(strategy, "signal_fill_prices", None)
        signal_contracts = getattr(strategy, "signal_contracts", None)
        if isinstance(signal_fill_prices, pd.Series):
            signal_fill_prices = signal_fill_prices.reindex(history.index)
        else:
            signal_fill_prices = None
        if isinstance(signal_contracts, pd.Series):
            signal_contracts = signal_contracts.reindex(history.index).fillna(0.0)
        else:
            signal_contracts = None
        resolved_contracts = _resolved_signal_contracts(signals, signal_contracts)

        latest_signal_index = bar_index if getattr(strategy, "execute_on_signal_bar", False) else bar_index - 1
        if latest_signal_index < 0:
            last_strategy_signals = signals.astype("float64").copy()
            last_strategy_contracts = resolved_contracts.astype("float64").copy()
            continue

        changed_signal_index = _latest_changed_signal_index(
            signals=signals,
            resolved_contracts=resolved_contracts,
            previous_signals=last_strategy_signals,
            previous_contracts=last_strategy_contracts,
            max_signal_index=latest_signal_index,
        )
        signal_index = changed_signal_index if changed_signal_index is not None else latest_signal_index
        signal_value = float(signals.iloc[signal_index])
        desired_position = int(np.sign(signal_value))
        desired_contracts = float(resolved_contracts.iloc[signal_index]) if desired_position != 0 else 0.0
        should_emit_signal = changed_signal_index is not None
        persist_bar_record = not (include_intrabar_preview and bar_index == (len(data) - 1))
        if not persist_bar_record and changed_signal_index is not None and changed_signal_index != latest_signal_index:
            changed_signal_index = None
            signal_index = latest_signal_index
            signal_value = float(signals.iloc[signal_index])
            desired_position = int(np.sign(signal_value))
            desired_contracts = float(resolved_contracts.iloc[signal_index]) if desired_position != 0 else 0.0
            should_emit_signal = False

        signal_fill = None
        if signal_fill_prices is not None:
            raw_fill = signal_fill_prices.iloc[signal_index]
            if pd.notna(raw_fill):
                signal_fill = float(raw_fill)

        snapshot = execution_engine.snapshot()
        current_bar = history.iloc[bar_index]
        target_units = execution_engine.position.quantity
        if should_emit_signal:
            sizing_price = signal_fill if signal_fill is not None else float(current_bar["open"])
            target_units = 0.0
            if desired_position != 0:
                base_units = sizing_engine._resolve_units(
                    capital=snapshot.equity,
                    fill_price=sizing_price,
                    stop_loss_price=None,
                    bar_index=bar_index,
                    closes=history["close"],
                    periods_per_year=periods_per_year,
                )
                target_units = base_units * desired_contracts * desired_position
                if config.max_leverage is not None and config.max_leverage > 0 and sizing_price > 0:
                    max_units = (snapshot.equity * config.max_leverage) / sizing_price
                    target_units = float(np.clip(target_units, -max_units, max_units))
                if config.max_position_size is not None and config.max_position_size > 0 and sizing_price > 0:
                    max_units_notional = config.max_position_size / sizing_price
                    target_units = float(np.clip(target_units, -max_units_notional, max_units_notional))

        reference_side = desired_position if desired_position != 0 else (-1 if execution_engine.position.quantity > 0 else 1)
        prev_close = float(history["close"].iloc[max(0, bar_index - 1)])
        order_request = _resolve_order_request(config=config, side=reference_side, prev_close=prev_close, signal_fill=signal_fill)
        signal = _build_signal(
            symbol="LIVE_CHART",
            timestamp=history.index[bar_index],
            target_qty=target_units,
            current_qty=execution_engine.position.quantity,
            projected_qty=_signed_open_order_projection(execution_engine),
            order_request=order_request,
            signal_fill=signal_fill,
        )
        if signal is not None:
            entry_reason, exit_reason = _strategy_reasons_for_signal(
                strategy,
                sizing_engine,
                signal_index=signal_index,
                current_qty=execution_engine.position.quantity,
                target_qty=target_units,
            )
            profit_protection_exit_reasons = {
                "Strategy Profit Protection Green Gator",
                "Strategy Profit Protection Red Gator",
                "Green PP",
                "Red PP",
            }
            signal.metadata = {
                **signal.metadata,
                "entry_reason": entry_reason,
                "exit_reason": exit_reason,
                "fill_on_close": (
                    persist_bar_record
                    and signal.action == "exit"
                    and exit_reason in profit_protection_exit_reasons
                ),
                "signal_bar_time": history.index[signal_index].isoformat(),
            }
            close_confirmed_profit_protection_exit = (
                signal.action == "exit"
                and exit_reason in profit_protection_exit_reasons
            )
            if persist_bar_record or not close_confirmed_profit_protection_exit:
                execution_engine.submit_signal(signal)

        last_strategy_signals = signals.astype("float64").copy()
        last_strategy_contracts = resolved_contracts.astype("float64").copy()
        execution_engine.on_bar(LiveBar.from_series("LIVE_CHART", history.index[bar_index], current_bar))

    fills = _fills_dataframe(execution_engine)
    return strategy, _paper_execution_events(fills), _completed_trades_from_fills(fills)


def _empty_market_payload(symbol: str, interval: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "interval": interval,
        "candles": [],
        "count": 0,
        "alligator": {"jaw": [], "teeth": [], "lips": []},
        "ao": [],
        "ac": [],
        "market_revision": 0,
    }


def _empty_strategy_payload() -> dict[str, Any]:
    return {
        "markers": [],
        "trade_event_lines": [],
        "trade_path_lines": [],
        "strategy_revision": 0,
        "strategy_bar_time": None,
    }


def _build_market_payload(symbol: str, interval: str, candles: list[dict[str, float | int]]) -> dict[str, Any]:
    data = _candles_to_dataframe(candles)
    if data.empty:
        return _empty_market_payload(symbol, interval)

    ao_histogram, ao_colors = _ao_histogram_from_data(data)
    ac_histogram, ac_colors = _ac_histogram_from_data(data)
    zone_colors = _williams_zones_colors(ao_colors, ac_colors)

    return {
        "symbol": symbol,
        "interval": interval,
        "candles": _candles_from_data(data, zone_colors=zone_colors),
        "count": len(candles),
        "alligator": _alligator_series_from_data(data),
        "ao": ao_histogram,
        "ac": ac_histogram,
        "market_revision": 0,
    }


def _build_strategy_payload(
    data: pd.DataFrame,
    strategy_template: Strategy | None = None,
    *,
    include_intrabar_preview: bool = False,
) -> dict[str, Any]:
    if data.empty:
        return _empty_strategy_payload()

    strategy, execution_events, completed_trades = _replay_strategy_execution(
        data,
        strategy_template,
        include_intrabar_preview=include_intrabar_preview,
    )
    third_wiseman_enabled = _strategy_has_third_wiseman_enabled(strategy)

    engine_first_markers = _first_wiseman_engine_markers(
        data,
        getattr(strategy, "signal_first_wiseman_setup_side", None),
        getattr(strategy, "signal_first_wiseman_ignored_reason", None),
        getattr(strategy, "signal_first_wiseman_reversal_side", None),
    )
    if engine_first_markers:
        wiseman_markers = []
    else:
        raw_markers = _wiseman_markers(data)
        wiseman_markers = [*raw_markers["bearish"], *raw_markers["bullish"]]
    fallback_overlays = (
        {
            "markers": [],
            "trade_event_lines": [],
            "trade_path_lines": [],
        }
        if _strategy_includes_bw(strategy)
        else _gator_profit_protection_fallback_overlays(
            data,
            strategy,
            completed_trades,
        )
    )

    combined_markers = _combine_markers(
        wiseman_markers,
        engine_first_markers,
        _first_wiseman_ignored_markers(
            data,
            getattr(strategy, "signal_first_wiseman_setup_side", None),
            getattr(strategy, "signal_first_wiseman_ignored_reason", None),
        ),
        _tag_markers(
            _second_wiseman_markers(
                data,
                getattr(strategy, "signal_fill_prices_second", None),
                getattr(strategy, "signal_second_wiseman_setup_side", None),
            ),
            "second_wiseman",
        ),
        _tag_markers(
            (
                _valid_third_wiseman_fractal_markers(
                    data,
                    getattr(strategy, "signal_third_wiseman_setup_side", None),
                )
                if third_wiseman_enabled
                else []
            ),
            "third_wiseman",
        ),
        _tag_markers(
            _wiseman_fill_entry_markers(
                data,
                getattr(strategy, "signal_fill_prices_second", None),
                getattr(strategy, "signal_second_wiseman_fill_side", None),
                label="2W" if _strategy_includes_bw(strategy) else None,
            ),
            "second_wiseman_entry",
        ),
        _tag_markers(
            (
                _wiseman_fill_entry_markers(
                    data,
                    getattr(strategy, "signal_fill_prices_third", None),
                    getattr(strategy, "signal_third_wiseman_fill_side", None),
                    label="3W" if _strategy_includes_bw(strategy) else None,
                )
                if third_wiseman_enabled
                else []
            ),
            "third_wiseman_entry",
        ),
        _tag_markers(
            _ntd_fill_entry_markers(
                data,
                getattr(strategy, "signal_fill_prices", None),
                getattr(strategy, "signal_fractal_position_side", None),
                getattr(strategy, "signal_contracts", None),
            ),
            "ntd_entry",
        ),
        _tag_markers(_execution_event_markers(execution_events, data), "execution"),
        _tag_markers(
            list(fallback_overlays["markers"]),
            "gator_profit_protection_fallback",
        ),
    )

    return {
        "markers": combined_markers,
        "trade_event_lines": [
            *_execution_event_lines(execution_events, data.index),
            *list(fallback_overlays["trade_event_lines"]),
        ],
        "trade_path_lines": [
            *_execution_trade_path_lines(execution_events, data.index),
            *list(fallback_overlays["trade_path_lines"]),
        ],
        "strategy_revision": 0,
        "strategy_bar_time": int(data.index[-1].timestamp()),
    }


def _build_live_chart_payload(
    symbol: str,
    interval: str,
    candles: list[dict[str, float | int]],
    strategy_template: Strategy | None = None,
) -> dict[str, Any]:
    market_payload = _build_market_payload(symbol, interval, candles)
    data = _candles_to_dataframe(candles)
    strategy_payload = _build_strategy_payload(data, strategy_template)
    return {**market_payload, **strategy_payload}


class ActiveLiveDataStore:
    def __init__(self, client: KCEXClient, strategy_template: Strategy | None = None) -> None:
        self.client = client
        self.strategy_template = (
            copy.deepcopy(strategy_template)
            if strategy_template is not None
            else CombinedStrategy([WisemanStrategy(), NTDStrategy()])
        )
        self.assets = client.infer_top_assets(10)
        self.timeframes = client.infer_timeframes()
        self.lock = threading.Lock()
        self.update_event = threading.Event()
        self.active_symbol = self.assets[0]
        self.active_interval = self.timeframes[0]
        self.selection_generation = 0
        self.market_revision = 0
        self.strategy_revision = 0
        self.market_payload = _empty_market_payload(self.active_symbol, self.active_interval)
        self.strategy_payload = _empty_strategy_payload()
        self.raw_candles: list[dict[str, float | int]] = []
        self.last_strategy_bar_time: int | None = None

    def _normalize(self, candles: list[Candle]) -> list[dict[str, float | int]]:
        return [{"time": c.time, "open": c.open, "high": c.high, "low": c.low, "close": c.close} for c in candles]

    def active_selection(self) -> SelectionSnapshot:
        with self.lock:
            return SelectionSnapshot(self.active_symbol, self.active_interval, self.selection_generation)

    def set_active_selection(self, symbol: str, interval: str) -> SelectionSnapshot:
        with self.lock:
            normalized_symbol = symbol if symbol in self.assets else self.assets[0]
            normalized_interval = interval if interval in self.timeframes else self.timeframes[0]
            if normalized_symbol == self.active_symbol and normalized_interval == self.active_interval:
                return SelectionSnapshot(self.active_symbol, self.active_interval, self.selection_generation)

            self.active_symbol = normalized_symbol
            self.active_interval = normalized_interval
            self.selection_generation += 1
            self.market_payload = _empty_market_payload(self.active_symbol, self.active_interval)
            self.strategy_payload = _empty_strategy_payload()
            self.raw_candles = []
            self.last_strategy_bar_time = None
            selection = SelectionSnapshot(self.active_symbol, self.active_interval, self.selection_generation)

        self.update_event.set()
        self.refresh_active_market_snapshot()
        return selection

    def refresh_active_market_snapshot(self) -> dict[str, Any]:
        selection = self.active_selection()
        candles = self.client.fetch_kline(selection.symbol, selection.interval, limit=400)
        normalized = self._normalize(candles)
        market_payload = _build_market_payload(selection.symbol, selection.interval, normalized)
        with self.lock:
            if selection.generation != self.selection_generation:
                return copy.deepcopy(self.market_payload)
            self.raw_candles = normalized
            self.market_revision += 1
            market_payload["market_revision"] = self.market_revision
            self.market_payload = market_payload
            current_payload = copy.deepcopy(self.market_payload)
        self.update_event.set()
        return current_payload

    def refresh_market_forever(self, poll_seconds: float = 1.0) -> None:
        while True:
            try:
                self.refresh_active_market_snapshot()
            except Exception:
                pass
            self.update_event.wait(timeout=poll_seconds)
            self.update_event.clear()

    def refresh_strategy_forever(self, poll_seconds: float = 0.25) -> None:
        while True:
            selection = self.active_selection()
            with self.lock:
                candles = list(self.raw_candles)
                last_strategy_bar_time = self.last_strategy_bar_time

            if candles:
                data = _candles_to_dataframe(candles)
                closed_data = _closed_bars(data, selection.interval)
                if not closed_data.empty:
                    closed_bar_time = int(closed_data.index[-1].timestamp())
                    if last_strategy_bar_time != closed_bar_time:
                        try:
                            strategy_payload = _build_strategy_payload(
                                closed_data,
                                self.strategy_template,
                                include_intrabar_preview=False,
                            )
                        except Exception:
                            strategy_payload = None
                        if strategy_payload is not None:
                            with self.lock:
                                if selection.generation == self.selection_generation:
                                    self.strategy_revision += 1
                                    strategy_payload["strategy_revision"] = self.strategy_revision
                                    self.strategy_payload = strategy_payload
                                    self.last_strategy_bar_time = closed_bar_time
            self.update_event.wait(timeout=poll_seconds)
            self.update_event.clear()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            payload = {**copy.deepcopy(self.market_payload), **copy.deepcopy(self.strategy_payload)}
            payload["symbol"] = self.active_symbol
            payload["interval"] = self.active_interval
            payload["selection_generation"] = self.selection_generation
            payload["generated_at"] = int(time.time())
            return payload


class Handler(BaseHTTPRequestHandler):
    store: ActiveLiveDataStore
    static_root: Path

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path in ("/", "/index.html"):
            body = (self.static_root / "index.html").read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return

        if parsed.path == "/api/markets":
            self._send_json({"assets": self.store.assets})
            return

        if parsed.path == "/api/timeframes":
            self._send_json({"timeframes": self.store.timeframes})
            return

        if parsed.path == "/api/select":
            params = urllib.parse.parse_qs(parsed.query)
            symbol = (params.get("symbol", [self.store.active_selection().symbol])[0] or "").upper()
            interval = params.get("interval", [self.store.active_selection().interval])[0]
            selection = self.store.set_active_selection(symbol, interval)
            self._send_json({
                "symbol": selection.symbol,
                "interval": selection.interval,
                "selection_generation": selection.generation,
            })
            return

        if parsed.path in ("/api/klines", "/api/chart-state"):
            params = urllib.parse.parse_qs(parsed.query)
            current_selection = self.store.active_selection()
            symbol = (params.get("symbol", [current_selection.symbol])[0] or "").upper()
            interval = params.get("interval", [current_selection.interval])[0]
            if symbol != current_selection.symbol or interval != current_selection.interval:
                self.store.set_active_selection(symbol, interval)
            try:
                payload = self.store.snapshot()
            except Exception as exc:
                self._send_json({"error": "Failed to read live chart state", "detail": str(exc)}, status=HTTPStatus.BAD_GATEWAY)
                return
            self._send_json(payload)
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)


def _build_strategy(args: argparse.Namespace) -> CombinedStrategy:
    wiseman = WisemanStrategy(
        gator_width_lookback=args.gator_width_lookback,
        gator_width_mult=args.gator_width_mult,
        gator_width_valid_factor=args.gator_width_valid_factor,
        gator_direction_mode=args.wiseman_gator_direction_mode,
        first_wiseman_contracts=args.wiseman_1w_contracts,
        second_wiseman_contracts=args.wiseman_2w_contracts,
        third_wiseman_contracts=args.wiseman_3w_contracts,
        reversal_contracts_mult=args.wiseman_reversal_contracts_mult,
        first_wiseman_wait_bars_to_close=args.wiseman_1w_wait_bars_to_close,
        first_wiseman_divergence_filter_bars=args.wiseman_1w_divergence_filter_bars,
        first_wiseman_opposite_close_min_unrealized_return=args.wiseman_1w_opposite_close_min_unrealized_return,
        first_wiseman_reversal_cooldown=args.wiseman_reversal_cooldown,
        cancel_reversal_on_first_wiseman_exit=args.wiseman_cancel_reversal_on_first_exit,
        teeth_profit_protection_enabled=args.wiseman_profit_protection_teeth_exit,
        teeth_profit_protection_min_bars=args.wiseman_profit_protection_min_bars,
        teeth_profit_protection_min_unrealized_return=args.wiseman_profit_protection_min_unrealized_return,
        teeth_profit_protection_credit_unrealized_before_min_bars=args.wiseman_profit_protection_credit_unrealized_before_min_bars,
        teeth_profit_protection_require_gator_open=args.wiseman_profit_protection_require_gator_open,
        profit_protection_volatility_lookback=args.wiseman_profit_protection_volatility_lookback,
                profit_protection_annualized_volatility_scaler=args.wiseman_profit_protection_annualized_volatility_scaler,
        lips_profit_protection_enabled=args.wiseman_profit_protection_lips_exit,
        lips_profit_protection_volatility_trigger=args.wiseman_profit_protection_lips_volatility_trigger,
        lips_profit_protection_profit_trigger_mult=args.wiseman_profit_protection_lips_profit_trigger_mult,
        lips_profit_protection_volatility_lookback=args.wiseman_profit_protection_lips_volatility_lookback,
        lips_profit_protection_recent_trade_lookback=args.wiseman_profit_protection_lips_recent_trade_lookback,
        lips_profit_protection_min_unrealized_return=args.wiseman_profit_protection_lips_min_unrealized_return,
        lips_profit_protection_arm_on_min_unrealized_return=args.wiseman_profit_protection_lips_arm_on_min_unrealized_return,
    )
    ntd = NTDStrategy(
        gator_width_lookback=args.gator_width_lookback,
        gator_width_mult=args.gator_width_mult,
        ao_ac_near_zero_lookback=args.ntd_ao_ac_near_zero_lookback,
        ao_ac_near_zero_factor=args.ntd_ao_ac_near_zero_factor,
        require_gator_close_reset=args.ntd_require_gator_close_reset,
        teeth_profit_protection_enabled=args.wiseman_profit_protection_teeth_exit,
        teeth_profit_protection_min_bars=args.wiseman_profit_protection_min_bars,
        teeth_profit_protection_min_unrealized_return=args.wiseman_profit_protection_min_unrealized_return,
        teeth_profit_protection_credit_unrealized_before_min_bars=args.wiseman_profit_protection_credit_unrealized_before_min_bars,
        teeth_profit_protection_require_gator_open=args.wiseman_profit_protection_require_gator_open,
        profit_protection_volatility_lookback=args.wiseman_profit_protection_volatility_lookback,
        profit_protection_annualized_volatility_scaler=args.wiseman_profit_protection_annualized_volatility_scaler,
        lips_profit_protection_enabled=args.wiseman_profit_protection_lips_exit,
        lips_profit_protection_volatility_trigger=args.wiseman_profit_protection_lips_volatility_trigger,
        lips_profit_protection_profit_trigger_mult=args.wiseman_profit_protection_lips_profit_trigger_mult,
        lips_profit_protection_volatility_lookback=args.wiseman_profit_protection_lips_volatility_lookback,
        lips_profit_protection_recent_trade_lookback=args.wiseman_profit_protection_lips_recent_trade_lookback,
        lips_profit_protection_min_unrealized_return=args.wiseman_profit_protection_lips_min_unrealized_return,
        lips_profit_protection_arm_on_min_unrealized_return=args.wiseman_profit_protection_lips_arm_on_min_unrealized_return,
    )
    return CombinedStrategy([wiseman, ntd])


def run_server(port: int, strategy_template: Strategy | None = None) -> None:
    client = KCEXClient()
    store = ActiveLiveDataStore(client, strategy_template)
    try:
        store.refresh_active_market_snapshot()
    except Exception:
        pass

    market_updater = threading.Thread(target=store.refresh_market_forever, kwargs={"poll_seconds": 1.0}, daemon=True)
    market_updater.start()
    strategy_updater = threading.Thread(target=store.refresh_strategy_forever, kwargs={"poll_seconds": 0.25}, daemon=True)
    strategy_updater.start()

    Handler.store = store
    Handler.static_root = Path(__file__).resolve().parent

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Live KCEX chart server running on http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live TradingView-style KCEX chart")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--gator-width-lookback", type=int, default=50)
    parser.add_argument("--gator-width-mult", type=float, default=1.0)
    parser.add_argument("--gator-width-valid-factor", type=float, default=1.0)
    parser.add_argument("--ntd-ao-ac-near-zero-lookback", type=int, default=50)
    parser.add_argument("--ntd-ao-ac-near-zero-factor", type=float, default=0.25)
    parser.add_argument("--ntd-require-gator-close-reset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wiseman-gator-direction-mode", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--wiseman-1w-contracts", type=int, default=1)
    parser.add_argument("--wiseman-2w-contracts", type=int, default=3)
    parser.add_argument("--wiseman-3w-contracts", type=int, default=5)
    parser.add_argument("--wiseman-reversal-contracts-mult", type=float, default=1.0)
    parser.add_argument("--1W-wait-bars-to-close", dest="wiseman_1w_wait_bars_to_close", type=int, default=0)
    parser.add_argument("--1W-divergence-filter", dest="wiseman_1w_divergence_filter_bars", type=int, default=0)
    parser.add_argument("--wiseman-1w-opposite-close-min-unrealized-return", type=float, default=0.0)
    parser.add_argument("--wiseman-reversal-cooldown", type=int, default=0)
    parser.add_argument("--wiseman-cancel-reversal-on-first-exit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wiseman-profit-protection-teeth-exit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wiseman-profit-protection-min-bars", type=int, default=3)
    parser.add_argument("--wiseman-profit-protection-min-unrealized-return", type=float, default=1.0)
    parser.add_argument(
        "--wiseman-profit-protection-credit-unrealized-before-min-bars",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--wiseman-profit-protection-require-gator-open", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wiseman-profit-protection-volatility-lookback", type=int, default=None)
    parser.add_argument("--wiseman-profit-protection-annualized-volatility-scaler", type=float, default=1.0)
    parser.add_argument("--wiseman-profit-protection-lips-exit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wiseman-profit-protection-lips-volatility-trigger", type=float, default=0.02)
    parser.add_argument("--wiseman-profit-protection-lips-profit-trigger-mult", type=float, default=2.0)
    parser.add_argument("--wiseman-profit-protection-lips-volatility-lookback", type=int, default=20)
    parser.add_argument("--wiseman-profit-protection-lips-recent-trade-lookback", type=int, default=5)
    parser.add_argument("--wiseman-profit-protection-lips-min-unrealized-return", type=float, default=1.0)
    parser.add_argument("--wiseman-profit-protection-lips-arm-on-min-unrealized-return", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    run_server(args.port, _build_strategy(args))
