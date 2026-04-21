from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from backtesting import (
    AlligatorAOStrategy,
    BacktestConfig,
    BacktestEngine,
    CombinedStrategy,
    NTDStrategy,
    BWStrategy,
    WisemanStrategy,
    compute_trade_diagnostics,
    generate_monte_carlo_pdf_report,
    load_ohlcv_csv,
    parse_trade_size_equity_milestones,
    resample_ohlcv,
    run_return_bootstrap_monte_carlo,
    filter_ohlcv_by_date,
    infer_source_timeframe_label,
)


def _count_nonzero(series: pd.Series | None) -> int:
    if not isinstance(series, pd.Series):
        return 0
    return int((series.fillna(0) != 0).sum())


def _execution_event_diagnostics(baseline) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in getattr(baseline, "execution_events", []):
        event_type = str(getattr(event, "event_type", "unknown"))
        counts[event_type] = counts.get(event_type, 0) + 1
    counts["closed_trades"] = len(getattr(baseline, "trades", []))
    return counts


def _backtest_end_reason(event_counts: dict[str, int], final_equity: float) -> str:
    if event_counts.get("equity_cutoff", 0) > 0:
        return "equity_cutoff"
    if event_counts.get("liquidation", 0) > 0 and final_equity <= 0:
        return "liquidation_bankruptcy"
    return "end_of_data"


def _wiseman_signal_diagnostics(strategy: WisemanStrategy) -> dict[str, int]:
    setup_series = getattr(strategy, "signal_first_wiseman_setup_side", None)
    ignored_reason_series = getattr(strategy, "signal_first_wiseman_ignored_reason", None)
    diagnostics = {
        "first_wiseman_setups": _count_nonzero(setup_series),
        "first_wiseman_price_events": _count_nonzero(getattr(strategy, "signal_fill_prices_first", None)),
        "first_wiseman_reversals": _count_nonzero(getattr(strategy, "signal_first_wiseman_reversal_side", None)),
        "second_wiseman_fills": _count_nonzero(getattr(strategy, "signal_second_wiseman_fill_side", None)),
        "third_wiseman_fills": _count_nonzero(getattr(strategy, "signal_third_wiseman_fill_side", None)),
    }
    if isinstance(ignored_reason_series, pd.Series):
        reasons = ignored_reason_series.fillna("").astype(str).str.strip()
        diagnostics["first_wiseman_ignored"] = int((reasons != "").sum())
        for reason, count in reasons[reasons != ""].value_counts().items():
            diagnostics[f"ignored_reason_{reason}"] = int(count)
    return diagnostics


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _parse_strategy_selection(selection: str) -> tuple[str, ...]:
    allowed = {"alligator_ao", "wiseman", "ntd", "bw"}
    names = tuple(dict.fromkeys(part.strip().lower() for part in selection.split(",") if part.strip()))
    if not names:
        raise ValueError("--strategy must include at least one strategy name")
    invalid = [name for name in names if name not in allowed]
    if invalid:
        raise ValueError(f"Unsupported strategy name(s): {', '.join(invalid)}")
    if ("alligator_ao" in names or "bw" in names) and len(names) > 1:
        raise ValueError("alligator_ao and bw cannot be combined with other strategies")
    return names


def _build_strategy(
    args: argparse.Namespace,
    strategy_names: tuple[str, ...],
) -> AlligatorAOStrategy | WisemanStrategy | NTDStrategy | BWStrategy | CombinedStrategy:
    def _build_wiseman_strategy() -> WisemanStrategy:
        return WisemanStrategy(
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
            zone_profit_protection_enabled=args.wiseman_profit_protection_zones_exit,
            zone_profit_protection_min_unrealized_return=args.wiseman_profit_protection_zones_min_unrealized_return,
        )

    def _build_ntd_strategy() -> NTDStrategy:
        return NTDStrategy(
            gator_width_lookback=args.gator_width_lookback,
            gator_width_mult=args.gator_width_mult,
            require_gator_close_reset=(
                args.ntd_require_gator_close_reset
                if args.ntd_require_gator_close_reset is not None
                else "wiseman" not in strategy_names
            ),
            ao_ac_near_zero_lookback=args.ntd_ao_ac_near_zero_lookback,
            ao_ac_near_zero_factor=args.ntd_ao_ac_near_zero_factor,
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
            zone_profit_protection_enabled=args.wiseman_profit_protection_zones_exit,
            zone_profit_protection_min_unrealized_return=args.wiseman_profit_protection_zones_min_unrealized_return,
        )

    if strategy_names == ("alligator_ao",):
        return AlligatorAOStrategy()
    if strategy_names == ("bw",):
        return BWStrategy(
            divergence_filter_bars=args.bw_1w_divergence_filter_bars,
            gator_open_filter_lookback=args.bw_1w_gator_open_lookback,
            gator_open_filter_min_percentile=args.bw_1w_gator_open_percentile,
            ntd_initial_fractal_enabled=args.bw_ntd_initial_fractal_enabled,
            fractal_add_on_contracts=args.bw_fractal_add_on_contracts,
            ntd_sleeping_gator_lookback=args.bw_ntd_sleeping_gator_lookback,
            ntd_sleeping_gator_tightness_mult=args.bw_ntd_sleeping_gator_tightness_mult,
            ntd_ranging_lookback=args.bw_ntd_ranging_lookback,
            ntd_ranging_max_span_pct=args.bw_ntd_ranging_max_span_pct,
            peak_drawdown_exit_enabled=args.bw_peak_drawdown_exit,
            peak_drawdown_exit_pct=args.bw_peak_drawdown_exit_pct,
            peak_drawdown_exit_volatility_lookback=args.bw_peak_drawdown_exit_volatility_lookback,
            peak_drawdown_exit_annualized_volatility_scaler=(
                args.bw_peak_drawdown_exit_annualized_volatility_scaler
            ),
            sigma_move_profit_protection_enabled=args.bw_profit_protection_sigma_move_exit,
            sigma_move_profit_protection_lookback=args.bw_profit_protection_sigma_move_lookback,
            sigma_move_profit_protection_sigma=args.bw_profit_protection_sigma_move_sigma,
            close_on_underlying_gain_pct=args.bw_close_on_underlying_gain_pct,
        )
    strategies = []
    if "wiseman" in strategy_names:
        strategies.append(_build_wiseman_strategy())
    if "ntd" in strategy_names:
        strategies.append(_build_ntd_strategy())
    return strategies[0] if len(strategies) == 1 else CombinedStrategy(strategies)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run strategy backtest then Monte Carlo return bootstraps.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--strategy", default="alligator_ao", help="Strategy name(s): alligator_ao, wiseman, ntd, bw, or comma-separated wiseman,ntd")
    parser.add_argument("--bw-1w-divergence-filter", "--1W-divergence-filter-bw", dest="bw_1w_divergence_filter_bars", type=int, default=0)
    parser.add_argument("--bw-1w-gator-open-lookback", type=int, default=0)
    parser.add_argument("--bw-1w-gator-open-percentile", type=float, default=50.0)
    parser.add_argument("--bw-ntd-initial-fractal-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bw-fractal-add-on-contracts", "--Fractal-add-on-contracts", dest="bw_fractal_add_on_contracts", type=int, default=0)
    parser.add_argument("--bw-ntd-sleeping-gator-lookback", type=int, default=50)
    parser.add_argument("--bw-ntd-sleeping-gator-tightness-mult", type=float, default=0.75)
    parser.add_argument("--bw-ntd-ranging-lookback", type=int, default=20)
    parser.add_argument("--bw-ntd-ranging-max-span-pct", type=float, default=0.025)
    parser.add_argument("--bw-peak-drawdown-exit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bw-peak-drawdown-exit-pct", type=float, default=0.01)
    parser.add_argument("--bw-peak-drawdown-exit-volatility-lookback", type=int, default=20)
    parser.add_argument("--bw-peak-drawdown-exit-annualized-volatility-scaler", type=float, default=1.0)
    parser.add_argument("--bw-profit-protection-sigma-move-exit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bw-profit-protection-sigma-move-lookback", type=int, default=20)
    parser.add_argument("--bw-profit-protection-sigma-move-sigma", type=float, default=2.0)
    parser.add_argument("--bw-close-on-underlying-gain-pct", type=float, default=0.0)
    parser.add_argument("--capital", type=float, default=10_000)
    parser.add_argument("--fee", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument("--spread", type=float, default=0.0)
    parser.add_argument("--order-type", default="market", choices=["market", "limit", "stop", "stop_limit"])
    parser.add_argument(
        "--size-mode",
        default="percent_of_equity",
        choices=["percent_of_equity", "usd", "units", "hybrid_min_usd_percent", "volatility_scaled", "stop_loss_scaled", "equity_milestone_usd"],
    )
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
    parser.add_argument("--equity-cutoff", type=float, default=None, help="Stop each simulated leg when equity falls to this value")

    parser.add_argument("--simulations", type=int, default=1000)
    parser.add_argument("--horizon-bars", type=int, default=None)
    parser.add_argument(
        "--allow-horizon-extension",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow Monte Carlo horizon to exceed baseline return bars",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--out", default="artifacts_monte_carlo")
    parser.add_argument("--timeframe", default=None, help="Optional output bar timeframe (e.g. 5m, 1h, 1d)")
    parser.add_argument("--start", default=None, help="Inclusive start date/time (e.g. 2024-01-01)")
    parser.add_argument("--end", default=None, help="Inclusive end date/time (e.g. 2025-12-12)")

    parser.add_argument("--gator-width-lookback", type=int, default=50)
    parser.add_argument("--gator-width-mult", type=float, default=1.0)
    parser.add_argument("--gator-width-valid-factor", type=float, default=1.0)
    parser.add_argument("--ntd-ao-ac-near-zero-lookback", type=int, default=50)
    parser.add_argument("--ntd-ao-ac-near-zero-factor", type=float, default=0.25)
    parser.add_argument("--ntd-require-gator-close-reset", action=argparse.BooleanOptionalAction, default=None)
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
    parser.add_argument("--wiseman-profit-protection-zones-exit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wiseman-profit-protection-zones-min-unrealized-return", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    strategy_names = _parse_strategy_selection(args.strategy)

    data = load_ohlcv_csv(args.csv)
    source_timeframe = infer_source_timeframe_label(data.index)
    effective_timeframe = args.timeframe or (None if source_timeframe == "unknown" else source_timeframe)

    if args.timeframe:
        if source_timeframe == "unknown" or effective_timeframe != source_timeframe:
            data = resample_ohlcv(data, effective_timeframe)

    data = filter_ohlcv_by_date(data, start=args.start, end=args.end)
    if len(data) < 2:
        raise ValueError("Filtered dataset has fewer than 2 bars; widen --start/--end range")
    def _build_strategy_for_run() -> AlligatorAOStrategy | WisemanStrategy | NTDStrategy | BWStrategy | CombinedStrategy:
        return _build_strategy(args, strategy_names)

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
            equity_cutoff=args.equity_cutoff,
        )
    )
    pilot_bars = min(len(data), 1000)
    if pilot_bars >= 200:
        pilot_slice = data.tail(pilot_bars).copy()
        pilot_started = time.perf_counter()
        engine.run(pilot_slice, _build_strategy_for_run())
        pilot_elapsed = time.perf_counter() - pilot_started
        estimated_baseline_seconds = pilot_elapsed * (len(data) / pilot_bars)
        print(
            "Baseline backtest runtime estimate: "
            f"~{_format_eta(estimated_baseline_seconds)} "
            f"(pilot {pilot_bars:,} bars in {pilot_elapsed:.2f}s, total bars {len(data):,})"
        )
    else:
        print(f"Baseline backtest runtime estimate: dataset is small ({len(data):,} bars), expected to finish quickly.")

    strategy = _build_strategy_for_run()
    baseline = engine.run(data, strategy)
    baseline_trade_count = len(baseline.trades)
    baseline_bars = len(baseline.returns)
    baseline_trade_diagnostics = compute_trade_diagnostics(
        trades_df=baseline.trades_dataframe(),
        initial_capital=args.capital,
        total_fees_paid=baseline.total_fees_paid,
        execution_events=baseline.execution_events,
        slippage_rate=args.slippage,
    )

    requested_horizon = int(args.horizon_bars) if args.horizon_bars is not None else baseline_bars
    if requested_horizon > baseline_bars and not args.allow_horizon_extension:
        raise ValueError(
            "Requested --horizon-bars exceeds available baseline return bars; "
            "either reduce --horizon-bars or pass --allow-horizon-extension. "
            f"requested={requested_horizon}, available={baseline_bars}"
        )
    baseline_event_counts = _execution_event_diagnostics(baseline)
    wiseman_diagnostics = _wiseman_signal_diagnostics(strategy) if args.strategy == "wiseman" else {}

    sample_simulations = min(64, args.simulations)
    if sample_simulations > 0 and requested_horizon > 0:
        estimate_started = time.perf_counter()
        run_return_bootstrap_monte_carlo(
            returns=baseline.returns,
            initial_capital=args.capital,
            simulations=sample_simulations,
            horizon_bars=requested_horizon,
            seed=args.seed,
            block_size=args.block_size,
            threads=args.threads,
            equity_cutoff=args.equity_cutoff,
        )
        estimate_elapsed = time.perf_counter() - estimate_started
        estimated_mc_seconds = estimate_elapsed * (args.simulations / sample_simulations)
        print(
            "Monte Carlo runtime estimate: "
            f"~{_format_eta(estimated_mc_seconds)} "
            f"(pilot {sample_simulations:,} sim in {estimate_elapsed:.2f}s, target {args.simulations:,})"
        )

    mc = run_return_bootstrap_monte_carlo(
        returns=baseline.returns,
        initial_capital=args.capital,
        simulations=args.simulations,
        horizon_bars=args.horizon_bars,
        seed=args.seed,
        block_size=args.block_size,
        threads=args.threads,
        equity_cutoff=args.equity_cutoff,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mc.summary["baseline_total_trades"] = float(baseline_trade_count)
    mc.summary["baseline_bars"] = float(baseline_bars)
    if baseline_bars > 0:
        mc.summary["horizon_vs_baseline_bars_multiple"] = float(mc.horizon_bars / baseline_bars)
        mc.summary["projected_trade_count_at_baseline_cadence"] = float(baseline_trade_count * (mc.horizon_bars / baseline_bars))
    mc.summary["source_timeframe"] = source_timeframe
    mc.summary["effective_timeframe"] = effective_timeframe or source_timeframe
    mc.summary["baseline_end_reason"] = _backtest_end_reason(baseline_event_counts, float(baseline.equity_curve.iloc[-1]))
    mc.summary.update({f"baseline_event_{k}": v for k, v in baseline_event_counts.items()})
    if wiseman_diagnostics:
        mc.summary.update({f"wiseman_{k}": v for k, v in wiseman_diagnostics.items()})

    (out_dir / "monte_carlo_summary.json").write_text(json.dumps(mc.summary, indent=2), encoding="utf-8")
    mc.equity_paths.to_csv(out_dir / "monte_carlo_equity_paths.csv", index=False)
    cli_flags = vars(args).copy()
    cli_flags["source_timeframe"] = source_timeframe
    cli_flags["effective_timeframe"] = effective_timeframe or source_timeframe

    generate_monte_carlo_pdf_report(
        mc,
        out_dir / "monte_carlo_report.pdf",
        cli_flags=cli_flags,
        csv_source=args.csv,
        baseline_trade_count=baseline_trade_count,
        trade_diagnostics=baseline_trade_diagnostics,
    )

    print("Monte Carlo complete")
    print(f"Detected source timeframe: {source_timeframe}")
    if effective_timeframe:
        print(f"Effective timeframe used: {effective_timeframe}")
    if args.start or args.end:
        range_start = data.index[0].isoformat()
        range_end = data.index[-1].isoformat()
        print(f"Date range used: {range_start} -> {range_end}")
    print(f"Baseline final equity: {baseline.equity_curve.iloc[-1]:,.2f}")
    print(
        "Baseline execution diagnostics "
        f"(closed-trades/entries/exits/adds/reduces/liquidations/equity-cutoff): "
        f"{baseline_event_counts.get('closed_trades', 0)} / "
        f"{baseline_event_counts.get('entry', 0)} / "
        f"{baseline_event_counts.get('exit', 0)} / "
        f"{baseline_event_counts.get('add', 0)} / "
        f"{baseline_event_counts.get('reduce', 0)} / "
        f"{baseline_event_counts.get('liquidation', 0)} / "
        f"{baseline_event_counts.get('equity_cutoff', 0)}"
    )
    print(f"Baseline run termination reason: {mc.summary['baseline_end_reason']}")
    if wiseman_diagnostics:
        print(
            "Wiseman signal diagnostics "
            f"(setups/1W-price-events/reversals/2W/3W/ignored): "
            f"{wiseman_diagnostics['first_wiseman_setups']} / "
            f"{wiseman_diagnostics['first_wiseman_price_events']} / "
            f"{wiseman_diagnostics['first_wiseman_reversals']} / "
            f"{wiseman_diagnostics['second_wiseman_fills']} / "
            f"{wiseman_diagnostics['third_wiseman_fills']} / "
            f"{wiseman_diagnostics.get('first_wiseman_ignored', 0)}"
        )
        ignored_reason_lines = [
            (k.replace("ignored_reason_", ""), v)
            for k, v in wiseman_diagnostics.items()
            if k.startswith("ignored_reason_")
        ]
        for reason, count in sorted(ignored_reason_lines, key=lambda item: item[1], reverse=True):
            print(f"  ignored[{reason}]: {count}")
    print(f"P50 final equity: {mc.summary['final_equity_p50']:,.2f}")
    print(f"P5-P95 final equity: {mc.summary['final_equity_p5']:,.2f} to {mc.summary['final_equity_p95']:,.2f}")
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
