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
    size: float = 1.0
    contracts: float = 1.0
    position_size_mode: str = "equity_percent"
    position_size_value: float = 1.0
    volatility_target_annual: float = 0.20
    volatility_lookback: int = 20
    volatility_min_scale: float = 0.25
    volatility_max_scale: float = 3.0
    execute_on_signal_bar: bool = False


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

        signals = strategy.generate_signals(data).reindex(data.index).fillna(0).astype("int8")
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

        for i in range(1, len(data)):
            signal_now = int(signals.iloc[i])
            price = float(close.iloc[i])
            ts = data.index[i]
            desired_signal = signal_now if self.config.execute_on_signal_bar else int(signals.iloc[i - 1])
            current_signal = 0 if units == 0 else (1 if units > 0 else -1)

            if units != 0 and desired_signal != current_signal:
                fill = price * (1.0 - self.config.slippage_rate * current_signal)
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
                entry_price = fill
                entry_time = ts
                entry_index = i
                capital -= fee

            mark_to_market = capital if units == 0 else capital + (price - entry_price) * units
            equity_values.append(float(mark_to_market))
            positions.append(0 if units == 0 else (1 if units > 0 else -1))

        if units != 0:
            final_price = float(close.iloc[-1])
            final_ts = data.index[-1]
            side = 1 if units > 0 else -1
            fill = final_price * (1.0 - self.config.slippage_rate * side)
            gross = (fill - entry_price) * units
            fee = abs(fill * units) * self.config.fee_rate
            pnl = gross - fee
            capital += pnl
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
