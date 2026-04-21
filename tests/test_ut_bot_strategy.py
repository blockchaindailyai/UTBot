from __future__ import annotations

import pandas as pd

from backtesting.engine import BacktestConfig, BacktestEngine
from backtesting.strategy import UTBotStrategy, compute_ut_bot_components
from examples import run_backtest


def test_ut_bot_strategy_generates_directional_signals() -> None:
    idx = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
    closes = pd.Series(
        [100 + i * 0.8 for i in range(20)] + [116 - i * 1.2 for i in range(20)],
        index=idx,
        dtype="float64",
    )
    data = pd.DataFrame(
        {
            "open": closes.shift(1).fillna(closes.iloc[0]),
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
        },
        index=idx,
    )

    strategy = UTBotStrategy(key_value=1.0, atr_period=10)
    signals = strategy.generate_signals(data)

    assert (signals == 1).any()
    assert (signals == -1).any()


def test_engine_accepts_size_and_contracts_for_ut_bot() -> None:
    idx = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    closes = pd.Series(
        [100 + i * 0.5 for i in range(30)] + [115 - i * 0.7 for i in range(30)],
        index=idx,
        dtype="float64",
    )
    data = pd.DataFrame(
        {
            "open": closes.shift(1).fillna(closes.iloc[0]),
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
        },
        index=idx,
    )
    strategy = UTBotStrategy()
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000.0,
            size=0.5,
            contracts=2.0,
            execute_on_signal_bar=True,
        )
    )

    result = engine.run(data, strategy)
    assert len(result.positions) == len(data)


def test_run_backtest_parser_accepts_ut_bot_aliases() -> None:
    parser = run_backtest.build_parser()
    args = parser.parse_args(["--csv", "examples/sample_ohlcv.csv", "--strategy", "ut-bot"])
    assert args.strategy == "ut-bot"


def test_compute_ut_bot_components_align_with_signal_index() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    closes = pd.Series([100 + i for i in range(15)] + [115 - i for i in range(15)], index=idx, dtype="float64")
    data = pd.DataFrame(
        {
            "open": closes.shift(1).fillna(closes.iloc[0]),
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
        },
        index=idx,
    )

    trailing_stop, buy_signal, sell_signal, position_state = compute_ut_bot_components(data)
    assert trailing_stop.index.equals(data.index)
    assert buy_signal.index.equals(data.index)
    assert sell_signal.index.equals(data.index)
    assert position_state.index.equals(data.index)
