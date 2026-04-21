from __future__ import annotations

import argparse
import json
from pathlib import Path

from backtesting import (
    AlligatorAOStrategy,
    BacktestConfig,
    BacktestEngine,
    generate_batch_local_tradingview_chart,
    load_ohlcv_csv,
    parse_trade_size_equity_milestones,
    run_batch_backtest,
)


def _parse_assets(arg: str) -> dict[str, str]:
    pairs = [p.strip() for p in arg.split(",") if p.strip()]
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid asset mapping '{pair}'. Use ASSET=path.csv")
        asset, path = pair.split("=", 1)
        out[asset.strip()] = path.strip()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True, help="Comma-separated ASSET=csv_path list")
    parser.add_argument("--timeframes", default="auto", help="Comma-separated intervals like 5m,1h,4h,1d or auto for source cadence")
    parser.add_argument("--capital", type=float, default=10_000)
    parser.add_argument("--fee", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument("--spread", type=float, default=0.0)
    parser.add_argument("--order-type", default="market", choices=["market", "limit", "stop", "stop_limit"])
    parser.add_argument("--size-mode", default="percent_of_equity", choices=["percent_of_equity", "usd", "units", "hybrid_min_usd_percent", "volatility_scaled", "stop_loss_scaled", "equity_milestone_usd"])
    parser.add_argument("--size-value", type=float, default=1.0)
    parser.add_argument("--size-min-usd", type=float, default=0.0)
    parser.add_argument("--size-equity-milestones", default="", help="Comma-separated EQUITY:USD step pairs for equity_milestone_usd sizing, e.g. 15000:1500,20000:2000")
    parser.add_argument("--volatility-target-annual", type=float, default=0.15)
    parser.add_argument("--volatility-lookback", type=int, default=20)
    parser.add_argument("--volatility-min-scale", type=float, default=0.25)
    parser.add_argument("--volatility-max-scale", type=float, default=3.0)
    parser.add_argument("--max-leverage", type=float, default=None)
    parser.add_argument("--max-position-size", type=float, default=None, help="Maximum absolute position notional in quote currency (e.g. USD)")
    parser.add_argument("--leverage-stop-out", type=float, default=0.0)
    parser.add_argument("--borrow-annual", type=float, default=0.0)
    parser.add_argument("--funding-per-period", type=float, default=0.0)
    parser.add_argument("--overnight-annual", type=float, default=0.0)
    parser.add_argument("--max-loss", type=float, default=None)
    parser.add_argument("--max-loss-pct-of-equity", type=float, default=None)
    parser.add_argument("--out", default="artifacts_batch")
    args = parser.parse_args()

    assets_map = _parse_assets(args.assets)
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()] if args.timeframes else []
    data_by_asset = {asset: load_ohlcv_csv(path) for asset, path in assets_map.items()}

    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=args.capital,
            fee_rate=args.fee,
            slippage_rate=args.slippage,
            spread_rate=args.spread,
            order_type=args.order_type,
            trade_size_mode=args.size_mode,
            trade_size_value=args.size_value,
            trade_size_min_usd=args.size_min_usd,
            trade_size_equity_milestones=parse_trade_size_equity_milestones(args.size_equity_milestones),
            volatility_target_annual=args.volatility_target_annual,
            volatility_lookback=args.volatility_lookback,
            volatility_min_scale=args.volatility_min_scale,
            volatility_max_scale=args.volatility_max_scale,
            max_leverage=args.max_leverage,
            max_position_size=args.max_position_size,
            leverage_stop_out_pct=args.leverage_stop_out,
            borrow_rate_annual=args.borrow_annual,
            funding_rate_per_period=args.funding_per_period,
            overnight_rate_annual=args.overnight_annual,
            max_loss=args.max_loss,
            max_loss_pct_of_equity=args.max_loss_pct_of_equity,
        )
    )

    def strategy_factory(_: str, __: str) -> AlligatorAOStrategy:
        return AlligatorAOStrategy()

    batch_result = run_batch_backtest(
        data_by_asset=data_by_asset,
        timeframes=timeframes,
        engine=engine,
        strategy_factory=strategy_factory,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "batch_summary.csv"
    batch_result.summary.to_csv(summary_path, index=False)

    aggregate_stats_path = out_dir / "aggregate_stats.json"
    aggregate_stats_path.write_text(json.dumps(batch_result.aggregate_stats, indent=2), encoding="utf-8")

    aggregate_equity_path = out_dir / "aggregate_equity.csv"
    batch_result.aggregate_equity_curve.rename("equity").to_csv(aggregate_equity_path, index_label="timestamp")

    chart_path = out_dir / "batch_tradingview_chart.html"
    generate_batch_local_tradingview_chart(batch_result, str(chart_path))

    print("Batch backtest complete")
    print(f"Runs: {len(batch_result.run_results)}")
    print(f"Summary written to: {summary_path}")
    print(f"Aggregate stats written to: {aggregate_stats_path}")
    print(f"Aggregate equity written to: {aggregate_equity_path}")
    print(f"Batch chart written to: {chart_path}")


if __name__ == "__main__":
    main()
