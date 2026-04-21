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
    parser.add_argument("--strategy", choices=["buy_hold", "ma_cross", "ut_bot"], default="ma_cross")
    parser.add_argument("--fast", type=int, default=20)
    parser.add_argument("--slow", type=int, default=50)
    parser.add_argument("--ut-key-value", type=float, default=1.0)
    parser.add_argument("--ut-atr-period", type=int, default=10)
    parser.add_argument("--size", type=float, default=1.0)
    parser.add_argument("--contracts", type=float, default=1.0)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_ohlcv_csv(args.csv)
    data = filter_ohlcv_by_date(data, start=args.start, end=args.end)

    if args.strategy == "buy_hold":
        strategy = BuyAndHoldStrategy()
    elif args.strategy == "ma_cross":
        strategy = MovingAverageCrossStrategy(args.fast, args.slow)
    else:
        strategy = UTBotStrategy(key_value=args.ut_key_value, atr_period=args.ut_atr_period)
    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=args.capital,
            fee_rate=args.fee,
            slippage_rate=args.slippage,
            size=args.size,
            contracts=args.contracts,
            execute_on_signal_bar=args.strategy == "ut_bot",
        )
    )
    result = engine.run(data, strategy)

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
