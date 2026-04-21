from __future__ import annotations

import argparse
import json
from pathlib import Path

from backtesting import (
    BacktestConfig,
    BacktestEngine,
    BuyAndHoldStrategy,
    MovingAverageCrossStrategy,
    UTBotStrategy,
    filter_ohlcv_by_date,
    generate_backtest_pdf_report,
    generate_local_tradingview_chart,
    load_ohlcv_csv,
)
from backtesting.tradingview import generate_ut_bot_strategy_pinescript


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Barebones backtest runner")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--fee", type=float, default=0.0)
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument(
        "--strategy",
        type=str,
        default="ma_cross",
        help="Strategy name: buy_hold, ma_cross, ut_bot (also accepts utbot, ut-bot)",
    )
    parser.add_argument("--fast", type=int, default=20)
    parser.add_argument("--slow", type=int, default=50)
    parser.add_argument("--ut-key-value", type=float, default=1.0)
    parser.add_argument("--ut-atr-period", type=int, default=10)
    parser.add_argument(
        "--size-mode",
        type=str,
        default="equity_percent",
        choices=["static_usd", "equity_percent", "volatility_scaled"],
    )
    parser.add_argument("--size-value", type=float, default=1.0)
    parser.add_argument("--volatility-target-annual", type=float, default=0.20)
    parser.add_argument("--volatility-lookback", type=int, default=20)
    parser.add_argument("--volatility-min-scale", type=float, default=0.25)
    parser.add_argument("--volatility-max-scale", type=float, default=3.0)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument(
        "--signal-timeframe",
        type=str,
        default=None,
        help=(
            "Optional higher timeframe for signal generation (for example 1D). "
            "When set, signals are recalculated on progressively formed higher-timeframe bars "
            "while execution still happens on the source bars."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_ohlcv_csv(args.csv)
    data = filter_ohlcv_by_date(data, start=args.start, end=args.end)

    strategy_name = args.strategy.strip().lower().replace("-", "_")
    if strategy_name == "buy_hold":
        strategy = BuyAndHoldStrategy()
    elif strategy_name == "ma_cross":
        strategy = MovingAverageCrossStrategy(args.fast, args.slow)
    elif strategy_name in {"ut_bot", "utbot"}:
        strategy = UTBotStrategy(key_value=args.ut_key_value, atr_period=args.ut_atr_period)
    else:
        raise ValueError("Unsupported strategy. Use one of: buy_hold, ma_cross, ut_bot")
    print(f"Selected strategy: {strategy_name}")
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=args.capital,
            fee_rate=args.fee,
            slippage_rate=args.slippage,
            position_size_mode=args.size_mode,
            position_size_value=args.size_value,
            volatility_target_annual=args.volatility_target_annual,
            volatility_lookback=args.volatility_lookback,
            volatility_min_scale=args.volatility_min_scale,
            volatility_max_scale=args.volatility_max_scale,
            execute_on_signal_bar=strategy_name in {"ut_bot", "utbot"},
            signal_timeframe=args.signal_timeframe,
        )
    )
    result = engine.run(data, strategy)
    print(f"Generated trades: {len(result.trades)}")

    (out_dir / "stats.json").write_text(json.dumps(result.stats, indent=2), encoding="utf-8")
    result.trades_dataframe().to_csv(out_dir / "trades.csv", index=False)
    generate_local_tradingview_chart(data=data, result=result, output_path=out_dir / "chart.html")
    generate_backtest_pdf_report(result=result, output_path=out_dir / "report.pdf")
    ut_bot_strategy_path = out_dir / "ut_bot_strategy.pine"
    generate_ut_bot_strategy_pinescript(output_path=str(ut_bot_strategy_path))
    print(f"TradingView UT Bot strategy script written to: {ut_bot_strategy_path}")
    print("Open the .pine file, copy its contents, and paste into TradingView Pine Editor, then click Add to chart.")


if __name__ == "__main__":
    main()
