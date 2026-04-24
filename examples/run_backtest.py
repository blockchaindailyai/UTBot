from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

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
from backtesting.resample import normalize_timeframe, resample_ohlcv
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
        "--ut-ma-filter",
        action="store_true",
        help="Enable MA trade filtering (buy signals only above MA, sell signals only below MA).",
    )
    parser.add_argument(
        "--ut-ma-period",
        type=int,
        default=60,
        help="Moving-average period used by --ut-ma-filter.",
    )
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
    parser.add_argument(
        "--signal-timeframe-progressive",
        action="store_true",
        help=(
            "Enable progressive intrabar simulation for signal-timeframe bars "
            "(default behavior)."
        ),
    )
    parser.add_argument(
        "--signal-timeframe-closed-only",
        action="store_true",
        help=(
            "Disable progressive intrabar simulation and only update signal-timeframe "
            "signals on higher-timeframe bar close."
        ),
    )
    parser.add_argument(
        "--max-intrabar-evaluations-per-signal-bar",
        type=int,
        default=24,
        help=(
            "Max number of intrabar signal evaluations within each signal-timeframe bar. "
            "Lower values are faster on dense source data (like 5m) and signals are "
            "forward-filled between evaluation points."
        ),
    )
    parser.add_argument(
        "--signal-timeframe-history-bars",
        type=int,
        default=None,
        help=(
            "Optional cap for how many higher-timeframe bars are passed to strategy "
            "during each intrabar re-evaluation. Lower values are faster but may "
            "change indicator behavior."
        ),
    )
    return parser


def _collect_set_cli_flags(parser: argparse.ArgumentParser, args: argparse.Namespace) -> dict[str, object]:
    defaults = {
        action.dest: action.default
        for action in parser._actions
        if action.dest != "help"
    }

    set_flags: dict[str, object] = {}
    for name, value in vars(args).items():
        default = defaults.get(name)
        if name == "csv":
            set_flags[name] = value
            continue
        if value == default or value is None or value is False:
            continue
        set_flags[name] = value
    return set_flags


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
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
        strategy = UTBotStrategy(
            key_value=args.ut_key_value,
            atr_period=args.ut_atr_period,
            ma_filter_enabled=args.ut_ma_filter,
            ma_period=args.ut_ma_period,
        )
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
            signal_timeframe_progressive=not args.signal_timeframe_closed_only,
            max_intrabar_evaluations_per_signal_bar=args.max_intrabar_evaluations_per_signal_bar,
            signal_timeframe_history_bars=args.signal_timeframe_history_bars,
        )
    )
    result = engine.run(data, strategy)
    print(f"Generated trades: {len(result.trades)}")

    (out_dir / "stats.json").write_text(json.dumps(result.stats, indent=2), encoding="utf-8")
    result.trades_dataframe().to_csv(out_dir / "trades.csv", index=False)
    chart_data = data
    if args.signal_timeframe:
        normalized_signal_timeframe = normalize_timeframe(args.signal_timeframe)
        source_freq = data.index.to_series().diff().dropna().median()
        try:
            target_freq = pd.to_timedelta(normalized_signal_timeframe)
        except (TypeError, ValueError):
            target_freq = None
        if target_freq is not None and pd.notna(source_freq) and target_freq > source_freq:
            chart_data = resample_ohlcv(data, args.signal_timeframe)

    generate_local_tradingview_chart(data=chart_data, result=result, output_path=out_dir / "chart.html")
    cli_flags = _collect_set_cli_flags(parser, args)
    generate_backtest_pdf_report(
        result=result,
        output_path=out_dir / "report.pdf",
        cli_flags=cli_flags,
    )
    ut_bot_strategy_path = out_dir / "ut_bot_strategy.pine"
    generate_ut_bot_strategy_pinescript(output_path=str(ut_bot_strategy_path))
    print(f"TradingView UT Bot strategy script written to: {ut_bot_strategy_path}")
    print("Open the .pine file, copy its contents, and paste into TradingView Pine Editor, then click Add to chart.")


if __name__ == "__main__":
    main()
