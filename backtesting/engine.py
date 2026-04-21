from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .stats import compute_performance_stats
from .strategy import Strategy


@dataclass(slots=True)
class BacktestConfig:
    initial_capital: float = 10_000.0
    fee_rate: float = 0.0
    slippage_rate: float = 0.0


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

        signals = strategy.generate_signals(data).reindex(data.index).fillna(0).astype("int8")
        close = data["close"].astype("float64")

        capital = float(self.config.initial_capital)
        equity_values = [capital]
        positions = [0]

        units = 0.0
        entry_price = 0.0
        entry_time: pd.Timestamp | None = None
        entry_index = 0
        trades: list[Trade] = []

        for i in range(1, len(data)):
            signal_prev = int(signals.iloc[i - 1])
            signal_now = int(signals.iloc[i])
            price = float(close.iloc[i])
            ts = data.index[i]

            if units == 0 and signal_prev != 0:
                fill = price * (1.0 + self.config.slippage_rate * signal_prev)
                fee = capital * self.config.fee_rate
                deployable = max(capital - fee, 0.0)
                units = (deployable / fill) * signal_prev
                entry_price = fill
                entry_time = ts
                entry_index = i
                capital -= fee

            if units != 0 and signal_now != signal_prev:
                fill = price * (1.0 - self.config.slippage_rate * (1 if units > 0 else -1))
                gross = (fill - entry_price) * units
                fee = abs(fill * units) * self.config.fee_rate
                pnl = gross - fee
                capital += pnl
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

            mark_to_market = capital if units == 0 else capital + (price - entry_price) * units
            equity_values.append(float(mark_to_market))
            positions.append(0 if units == 0 else (1 if units > 0 else -1))

        equity = pd.Series(equity_values, index=data.index, dtype="float64")
        returns = equity.pct_change().fillna(0.0)
        positions_series = pd.Series(positions, index=data.index, dtype="int8")
        stats = compute_performance_stats(
            equity_curve=equity,
            returns=returns,
            trades=trades,
            periods_per_year=252,
            positions=positions_series,
        )
        stats["final_equity"] = float(equity.iloc[-1])
        stats["total_trades"] = float(len(trades))

        return BacktestResult(
            equity_curve=equity,
            returns=returns,
            positions=positions_series,
            trades=trades,
            stats=stats,
        )
