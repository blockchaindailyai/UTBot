from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import math
import pandas as pd

OrderType = Literal["market", "limit", "stop", "stop_limit"]
SignalAction = Literal["enter", "exit", "reverse", "scale", "cancel"]
OrderSide = Literal["buy", "sell"]
PositionSide = Literal["long", "short", "flat"]


@dataclass(slots=True)
class LiveBar:
    symbol: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @classmethod
    def from_series(cls, symbol: str, timestamp: pd.Timestamp, row: pd.Series) -> "LiveBar":
        return cls(
            symbol=symbol,
            timestamp=timestamp,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0.0) or 0.0),
        )


@dataclass(slots=True)
class ExecutionSignal:
    symbol: str
    timestamp: pd.Timestamp
    action: SignalAction
    side: OrderSide | None = None
    order_type: OrderType = "market"
    quantity: float = 0.0
    limit_price: float | None = None
    stop_price: float | None = None
    reduce_only: bool = False
    cancel_existing_orders: bool = False
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    signal_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PaperOrder:
    order_id: int
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    created_at: pd.Timestamp
    limit_price: float | None = None
    stop_price: float | None = None
    reduce_only: bool = False
    signal_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: Literal["open", "filled", "cancelled"] = "open"
    triggered_at: pd.Timestamp | None = None
    filled_at: pd.Timestamp | None = None
    fill_price: float | None = None


@dataclass(slots=True)
class PaperFill:
    order_id: int
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fee: float
    timestamp: pd.Timestamp
    reason: str
    strategy_reason: str | None = None
    signal_action: str | None = None
    realized_pnl: float = 0.0


@dataclass(slots=True)
class PaperPosition:
    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    entry_equity: float | None = None
    opened_at: pd.Timestamp | None = None
    last_updated_at: pd.Timestamp | None = None
    stop_loss_price: float | None = None
    take_profit_price: float | None = None

    @property
    def side(self) -> PositionSide:
        if self.quantity > 0:
            return "long"
        if self.quantity < 0:
            return "short"
        return "flat"

    @property
    def is_open(self) -> bool:
        return not math.isclose(self.quantity, 0.0, abs_tol=1e-12)


@dataclass(slots=True)
class PaperTradingSnapshot:
    timestamp: pd.Timestamp | None
    cash: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    fees_paid: float
    open_orders: list[PaperOrder]
    fills: list[PaperFill]
    position: PaperPosition
    mark_price: float | None


class PaperTradingEngine:
    """Paper broker/execution engine for live signal-driven trading.

    The engine keeps a single net position per symbol and supports market, limit,
    stop, and stop-limit orders, along with signal-driven scale/reverse/exit
    workflows and basic stop-loss / take-profit monitoring.
    """

    def __init__(
        self,
        *,
        initial_cash: float = 10_000.0,
        fee_rate: float = 0.0005,
        slippage_rate: float = 0.0002,
        spread_rate: float = 0.0,
        max_loss: float | None = None,
        max_loss_pct_of_equity: float | None = None,
        symbol: str = "PAPER",
    ) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if fee_rate < 0:
            raise ValueError("fee_rate must be >= 0")
        if slippage_rate < 0:
            raise ValueError("slippage_rate must be >= 0")
        if spread_rate < 0:
            raise ValueError("spread_rate must be >= 0")
        if max_loss is not None and max_loss < 0:
            raise ValueError("max_loss must be >= 0")
        if max_loss_pct_of_equity is not None and max_loss_pct_of_equity < 0:
            raise ValueError("max_loss_pct_of_equity must be >= 0")

        self.symbol = symbol
        self.initial_cash = float(initial_cash)
        self.fee_rate = float(fee_rate)
        self.slippage_rate = float(slippage_rate)
        self.spread_rate = float(spread_rate)
        self.max_loss = None if max_loss is None else float(max_loss)
        self.max_loss_pct_of_equity = None if max_loss_pct_of_equity is None else float(max_loss_pct_of_equity)

        self.cash = float(initial_cash)
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self.position = PaperPosition(symbol=symbol)
        self.open_orders: list[PaperOrder] = []
        self.fills: list[PaperFill] = []
        self._mark_price: float | None = None
        self._next_order_id = 1
        self._last_timestamp: pd.Timestamp | None = None
        self._submitted_signal_keys: set[tuple[str, str, pd.Timestamp, str | None, float, str, float | None, float | None]] = set()

    def submit_signal(self, signal: ExecutionSignal) -> list[PaperOrder]:
        if signal.symbol != self.symbol:
            raise ValueError(f"signal symbol {signal.symbol!r} does not match engine symbol {self.symbol!r}")
        if signal.action != "cancel" and signal.quantity < 0:
            raise ValueError("signal quantity must be >= 0")
        if signal.signal_id is not None and signal.action != "cancel":
            signal_key = (
                str(signal.signal_id),
                str(signal.action),
                pd.Timestamp(signal.timestamp),
                signal.side,
                float(signal.quantity),
                str(signal.order_type),
                float(signal.limit_price) if signal.limit_price is not None else None,
                float(signal.stop_price) if signal.stop_price is not None else None,
            )
            if signal_key in self._submitted_signal_keys:
                return []
            self._submitted_signal_keys.add(signal_key)
        if signal.cancel_existing_orders:
            self.cancel_open_orders()

        if signal.action == "cancel":
            self.cancel_open_orders(signal_id=signal.signal_id)
            return []

        orders: list[PaperOrder] = []
        if signal.action == "exit":
            quantity = abs(self.position.quantity) if signal.quantity <= 0 else signal.quantity
            if quantity <= 0:
                return []
            exit_side: OrderSide = "sell" if self.position.quantity > 0 else "buy"
            orders.append(
                self._create_order(
                    side=exit_side,
                    order_type=signal.order_type,
                    quantity=quantity,
                    created_at=signal.timestamp,
                    limit_price=signal.limit_price,
                    stop_price=signal.stop_price,
                    reduce_only=True,
                    signal_id=signal.signal_id,
                    metadata={**signal.metadata, "signal_action": signal.action},
                )
            )
        else:
            if signal.side is None:
                raise ValueError("enter/reverse/scale signals require a side")
            if signal.quantity <= 0:
                raise ValueError("enter/reverse/scale signals require quantity > 0")
            if signal.action == "reverse" and self.position.is_open:
                reverse_exit_side: OrderSide = "sell" if self.position.quantity > 0 else "buy"
                orders.append(
                    self._create_order(
                        side=reverse_exit_side,
                        order_type="market",
                        quantity=abs(self.position.quantity),
                        created_at=signal.timestamp,
                        reduce_only=True,
                        signal_id=signal.signal_id,
                        metadata={**signal.metadata, "signal_action": "reverse_exit"},
                    )
                )
            orders.append(
                self._create_order(
                    side=signal.side,
                    order_type=signal.order_type,
                    quantity=signal.quantity,
                    created_at=signal.timestamp,
                    limit_price=signal.limit_price,
                    stop_price=signal.stop_price,
                    reduce_only=signal.reduce_only,
                    signal_id=signal.signal_id,
                    metadata={
                        **signal.metadata,
                        "signal_action": signal.action,
                        "stop_loss_price": signal.stop_loss_price,
                        "take_profit_price": signal.take_profit_price,
                    },
                )
            )

        self.open_orders.extend(orders)
        return orders

    def cancel_open_orders(self, signal_id: str | None = None) -> int:
        cancelled = 0
        for order in self.open_orders:
            if order.status != "open":
                continue
            if signal_id is not None and order.signal_id != signal_id:
                continue
            order.status = "cancelled"
            cancelled += 1
        self.open_orders = [order for order in self.open_orders if order.status == "open"]
        return cancelled

    def on_bar(self, bar: LiveBar) -> list[PaperFill]:
        if bar.symbol != self.symbol:
            raise ValueError(f"bar symbol {bar.symbol!r} does not match engine symbol {self.symbol!r}")
        self._last_timestamp = bar.timestamp
        self._mark_price = float(bar.close)

        bar_fills: list[PaperFill] = []
        bar_fills.extend(self._process_protective_exits(bar))

        remaining_orders: list[PaperOrder] = []
        for order in self.open_orders:
            fill_price, reason, triggered_at = self._evaluate_order_fill(order, bar)
            if fill_price is None or reason is None:
                remaining_orders.append(order)
                continue
            if triggered_at is not None:
                order.triggered_at = triggered_at
            fill = self._fill_order(order, fill_price=fill_price, timestamp=bar.timestamp, reason=reason)
            bar_fills.append(fill)
        self.open_orders = [order for order in remaining_orders if order.status == "open"]
        self.fills.extend(bar_fills)
        return bar_fills

    def mark_to_market(self, *, timestamp: pd.Timestamp, price: float) -> None:
        self._last_timestamp = timestamp
        self._mark_price = float(price)

    def snapshot(self) -> PaperTradingSnapshot:
        unrealized = self.unrealized_pnl(mark_price=self._mark_price)
        return PaperTradingSnapshot(
            timestamp=self._last_timestamp,
            cash=self.cash,
            equity=self.cash + unrealized,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=unrealized,
            fees_paid=self.fees_paid,
            open_orders=[order for order in self.open_orders if order.status == "open"],
            fills=list(self.fills),
            position=PaperPosition(
                symbol=self.position.symbol,
                quantity=self.position.quantity,
                average_price=self.position.average_price,
                entry_equity=self.position.entry_equity,
                opened_at=self.position.opened_at,
                last_updated_at=self.position.last_updated_at,
                stop_loss_price=self.position.stop_loss_price,
                take_profit_price=self.position.take_profit_price,
            ),
            mark_price=self._mark_price,
        )

    def unrealized_pnl(self, mark_price: float | None = None) -> float:
        if not self.position.is_open:
            return 0.0
        price = self._mark_price if mark_price is None else mark_price
        if price is None:
            return 0.0
        if self.position.quantity > 0:
            return (float(price) - self.position.average_price) * self.position.quantity
        return (self.position.average_price - float(price)) * abs(self.position.quantity)

    def _create_order(
        self,
        *,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        created_at: pd.Timestamp,
        limit_price: float | None = None,
        stop_price: float | None = None,
        reduce_only: bool = False,
        signal_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PaperOrder:
        if quantity <= 0:
            raise ValueError("order quantity must be positive")
        if order_type in {"limit", "stop_limit"} and (limit_price is None or limit_price <= 0):
            raise ValueError(f"{order_type} orders require limit_price > 0")
        if order_type in {"stop", "stop_limit"} and (stop_price is None or stop_price <= 0):
            raise ValueError(f"{order_type} orders require stop_price > 0")
        order = PaperOrder(
            order_id=self._next_order_id,
            symbol=self.symbol,
            side=side,
            order_type=order_type,
            quantity=float(quantity),
            created_at=created_at,
            limit_price=limit_price,
            stop_price=stop_price,
            reduce_only=reduce_only,
            signal_id=signal_id,
            metadata=metadata or {},
        )
        self._next_order_id += 1
        return order

    def _process_protective_exits(self, bar: LiveBar) -> list[PaperFill]:
        if not self.position.is_open:
            return []

        stop_candidates: list[tuple[float, str]] = []
        take_profit_candidates: list[tuple[float, str]] = []
        engine_stop_price = self._max_loss_stop_price()
        if self.position.quantity > 0:
            if self.position.stop_loss_price is not None and bar.low <= self.position.stop_loss_price:
                stop_candidates.append((self.position.stop_loss_price, "protective_stop"))
            if engine_stop_price is not None and bar.low <= engine_stop_price:
                stop_candidates.append((engine_stop_price, "max_loss_stop"))
            if self.position.take_profit_price is not None and bar.high >= self.position.take_profit_price:
                take_profit_candidates.append((self.position.take_profit_price, "take_profit"))
            exit_side: OrderSide = "sell"
        else:
            if self.position.stop_loss_price is not None and bar.high >= self.position.stop_loss_price:
                stop_candidates.append((self.position.stop_loss_price, "protective_stop"))
            if engine_stop_price is not None and bar.high >= engine_stop_price:
                stop_candidates.append((engine_stop_price, "max_loss_stop"))
            if self.position.take_profit_price is not None and bar.low <= self.position.take_profit_price:
                take_profit_candidates.append((self.position.take_profit_price, "take_profit"))
            exit_side = "buy"

        protective_price: float | None = None
        reason: str | None = None
        if stop_candidates:
            if self.position.quantity > 0:
                protective_price, reason = max(stop_candidates, key=lambda item: item[0])
            else:
                protective_price, reason = min(stop_candidates, key=lambda item: item[0])
        elif take_profit_candidates:
            protective_price, reason = take_profit_candidates[0]

        if protective_price is None or reason is None:
            return []

        if reason == "protective_stop":
            fill_price = min(bar.open, protective_price) if self.position.quantity > 0 else max(bar.open, protective_price)
        elif reason == "max_loss_stop":
            fill_price = min(bar.open, protective_price) if self.position.quantity > 0 else max(bar.open, protective_price)
        else:
            fill_price = max(bar.open, protective_price) if self.position.quantity > 0 else min(bar.open, protective_price)

        order = self._create_order(
            side=exit_side,
            order_type="market",
            quantity=abs(self.position.quantity),
            created_at=bar.timestamp,
            reduce_only=True,
            metadata={"signal_action": reason},
        )
        fill = self._fill_order(order, fill_price=fill_price, timestamp=bar.timestamp, reason=reason)
        return [fill]

    def _effective_max_loss(self) -> float | None:
        thresholds: list[float] = []
        if self.max_loss is not None and self.max_loss > 0:
            thresholds.append(self.max_loss)
        if (
            self.max_loss_pct_of_equity is not None
            and self.max_loss_pct_of_equity > 0
            and self.position.entry_equity is not None
            and self.position.entry_equity > 0
        ):
            thresholds.append(self.position.entry_equity * self.max_loss_pct_of_equity)
        if not thresholds:
            return None
        return min(thresholds)

    def _max_loss_stop_price(self) -> float | None:
        if not self.position.is_open or self.position.average_price <= 0:
            return None
        max_loss = self._effective_max_loss()
        if max_loss is None or max_loss <= 0:
            return None
        units = abs(self.position.quantity)
        if units <= 0:
            return None
        loss_per_unit = max_loss / units
        if self.position.quantity > 0:
            stop_price = self.position.average_price - loss_per_unit
            return stop_price if stop_price > 0 else None
        return self.position.average_price + loss_per_unit

    def _evaluate_order_fill(self, order: PaperOrder, bar: LiveBar) -> tuple[float | None, str | None, pd.Timestamp | None]:
        if order.status != "open":
            return None, None, None

        if order.order_type == "market":
            explicit_fill = order.metadata.get("market_fill_price")
            if explicit_fill is not None:
                try:
                    explicit_fill_price = float(explicit_fill)
                except (TypeError, ValueError):
                    explicit_fill_price = None
                if explicit_fill_price is not None and explicit_fill_price > 0:
                    return self._apply_execution_adjustment(explicit_fill_price, order.side), "market", bar.timestamp
            fill_on_close = bool(order.metadata.get("fill_on_close", False))
            raw_price = bar.close if fill_on_close else bar.open
            return self._apply_execution_adjustment(raw_price, order.side), "market", bar.timestamp

        if order.order_type == "limit":
            if order.side == "buy" and bar.low <= float(order.limit_price):
                raw_price = min(float(order.limit_price), bar.open)
                return self._apply_execution_adjustment(raw_price, order.side), "limit", bar.timestamp
            if order.side == "sell" and bar.high >= float(order.limit_price):
                raw_price = max(float(order.limit_price), bar.open)
                return self._apply_execution_adjustment(raw_price, order.side), "limit", bar.timestamp
            return None, None, None

        if order.order_type == "stop":
            if order.side == "buy" and bar.high >= float(order.stop_price):
                raw_price = max(float(order.stop_price), bar.open)
                return self._apply_execution_adjustment(raw_price, order.side), "stop", bar.timestamp
            if order.side == "sell" and bar.low <= float(order.stop_price):
                raw_price = min(float(order.stop_price), bar.open)
                return self._apply_execution_adjustment(raw_price, order.side), "stop", bar.timestamp
            return None, None, None

        if order.order_type == "stop_limit":
            stop_price = float(order.stop_price)
            limit_price = float(order.limit_price)
            if order.side == "buy" and bar.high >= stop_price:
                if bar.open >= stop_price:
                    stop_trigger_price = bar.open
                else:
                    stop_trigger_price = stop_price
                if bar.low <= limit_price <= bar.high or bar.open <= limit_price:
                    raw_price = min(limit_price, stop_trigger_price, bar.open)
                    return self._apply_execution_adjustment(raw_price, order.side), "stop_limit", bar.timestamp
                order.order_type = "limit"
                order.triggered_at = bar.timestamp
                return None, None, bar.timestamp
            if order.side == "sell" and bar.low <= stop_price:
                if bar.open <= stop_price:
                    stop_trigger_price = bar.open
                else:
                    stop_trigger_price = stop_price
                if bar.low <= limit_price <= bar.high or bar.open >= limit_price:
                    raw_price = max(limit_price, stop_trigger_price, bar.open)
                    return self._apply_execution_adjustment(raw_price, order.side), "stop_limit", bar.timestamp
                order.order_type = "limit"
                order.triggered_at = bar.timestamp
                return None, None, bar.timestamp
        return None, None, None

    def _fill_order(self, order: PaperOrder, *, fill_price: float, timestamp: pd.Timestamp, reason: str) -> PaperFill:
        order.status = "filled"
        order.filled_at = timestamp
        order.fill_price = float(fill_price)

        signed_quantity = order.quantity if order.side == "buy" else -order.quantity
        fee = abs(fill_price * order.quantity) * self.fee_rate
        realized = self._apply_fill_to_position(signed_quantity=signed_quantity, fill_price=fill_price, timestamp=timestamp, reduce_only=order.reduce_only, metadata=order.metadata)
        self.cash -= fee
        self.fees_paid += fee
        signal_action = str(order.metadata.get("signal_action", "") or "").strip() or None
        entry_reason = str(order.metadata.get("entry_reason", "") or "").strip() or None
        exit_reason = str(order.metadata.get("exit_reason", "") or "").strip() or None
        strategy_reason = entry_reason
        if signal_action in {"exit", "reverse_exit"} or order.reduce_only:
            strategy_reason = exit_reason or strategy_reason
        if strategy_reason is None and signal_action == "protective_stop":
            strategy_reason = "Protective Stop"
        if strategy_reason is None and signal_action == "take_profit":
            strategy_reason = "Take Profit"

        return PaperFill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            fee=fee,
            timestamp=timestamp,
            reason=reason,
            strategy_reason=strategy_reason,
            signal_action=signal_action,
            realized_pnl=realized,
        )

    def _apply_fill_to_position(
        self,
        *,
        signed_quantity: float,
        fill_price: float,
        timestamp: pd.Timestamp,
        reduce_only: bool,
        metadata: dict[str, Any],
    ) -> float:
        current_qty = self.position.quantity
        current_avg = self.position.average_price
        remaining_qty = current_qty
        realized = 0.0

        if reduce_only and current_qty == 0:
            return 0.0

        if current_qty == 0 or math.copysign(1.0, current_qty) == math.copysign(1.0, signed_quantity):
            new_qty = current_qty + signed_quantity
            if reduce_only and abs(new_qty) > abs(current_qty):
                new_qty = current_qty
            if not math.isclose(new_qty, current_qty, abs_tol=1e-12):
                total_abs = abs(current_qty) + abs(signed_quantity)
                weighted_notional = (abs(current_qty) * current_avg) + (abs(signed_quantity) * fill_price)
                self.position.average_price = weighted_notional / total_abs if total_abs > 0 else 0.0
                self.position.quantity = new_qty
                self.position.entry_equity = self.cash + self.unrealized_pnl(fill_price)
                if self.position.opened_at is None:
                    self.position.opened_at = timestamp
            remaining_qty = self.position.quantity
        else:
            close_qty = min(abs(current_qty), abs(signed_quantity))
            if current_qty > 0:
                realized = (fill_price - current_avg) * close_qty
            else:
                realized = (current_avg - fill_price) * close_qty
            self.realized_pnl += realized
            self.cash += realized

            leftover = abs(signed_quantity) - close_qty
            if leftover > 0 and not reduce_only:
                self.position.quantity = math.copysign(leftover, signed_quantity)
                self.position.average_price = fill_price
                self.position.entry_equity = self.cash + self.unrealized_pnl(fill_price)
                self.position.opened_at = timestamp
            else:
                self.position.quantity = math.copysign(max(abs(current_qty) - close_qty, 0.0), current_qty)
                if math.isclose(self.position.quantity, 0.0, abs_tol=1e-12):
                    self.position.quantity = 0.0
                    self.position.average_price = 0.0
                    self.position.entry_equity = None
                    self.position.opened_at = None
                    self.position.stop_loss_price = None
                    self.position.take_profit_price = None
            remaining_qty = self.position.quantity

        if not math.isclose(remaining_qty, 0.0, abs_tol=1e-12):
            stop_loss = metadata.get("stop_loss_price")
            take_profit = metadata.get("take_profit_price")
            if stop_loss is not None:
                self.position.stop_loss_price = float(stop_loss)
            if take_profit is not None:
                self.position.take_profit_price = float(take_profit)
        self.position.last_updated_at = timestamp
        return realized

    def _apply_execution_adjustment(self, raw_price: float, side: OrderSide) -> float:
        spread_component = self.spread_rate / 2.0
        if side == "buy":
            return float(raw_price) * (1.0 + self.slippage_rate + spread_component)
        return float(raw_price) * (1.0 - self.slippage_rate - spread_component)
