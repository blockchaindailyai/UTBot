from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from backtesting import (
    AlligatorAOStrategy,
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    BWStrategy,
    CombinedStrategy,
    NTDStrategy,
    WisemanStrategy,
    compute_performance_stats,
    generate_first_wiseman_bearish_pinescript,
    generate_first_wiseman_bullish_pinescript,
    generate_local_tradingview_chart,
    generate_backtest_pdf_report,
    generate_backtest_clean_pdf_report,
    generate_trade_marker_pinescript,
    filter_ohlcv_by_date,
    load_ohlcv_csv,
    parse_trade_size_equity_milestones,
)
from backtesting.stats import infer_periods_per_year


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _runtime_estimate_message(
    *,
    engine: BacktestEngine,
    data: pd.DataFrame,
    build_strategy: Callable[[], object],
    csv_path: str,
) -> str:
    pilot_bars = min(len(data), 1000)
    if pilot_bars < 200:
        return f"[{csv_path}] Backtest runtime estimate: dataset is small ({len(data):,} bars), expected to finish quickly."

    pilot_slice = data.tail(pilot_bars).copy()
    pilot_started = time.perf_counter()
    try:
        engine.run(pilot_slice, build_strategy())
    except Exception as exc:
        return (
            f"[{csv_path}] Backtest runtime estimate skipped: pilot slice could not be evaluated "
            f"({exc}). Continuing with full run."
        )

    pilot_elapsed = time.perf_counter() - pilot_started
    estimated_full_seconds = pilot_elapsed * (len(data) / pilot_bars)
    return (
        f"[{csv_path}] Backtest runtime estimate: "
        f"~{_format_eta(estimated_full_seconds)} "
        f"(pilot {pilot_bars:,} bars in {pilot_elapsed:.2f}s, total bars {len(data):,})"
    )


def _strategy_includes_bw(strategy: object) -> bool:
    if isinstance(strategy, BWStrategy):
        return True
    if isinstance(strategy, CombinedStrategy):
        return any(_strategy_includes_bw(component) for component in strategy.strategies)
    return False


def _parse_asset_allocations(entries: list[str]) -> dict[str, float]:
    allocation_map: dict[str, float] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid --asset-size value '{entry}'. Use KEY=SIZE")
        key, value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --asset-size value '{entry}'. KEY cannot be empty")
        try:
            allocation_map[key] = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid --asset-size value '{entry}'. SIZE must be numeric") from exc
    return allocation_map


def _resolve_asset_allocation_for_csv(csv_path: str, allocation_map: dict[str, float]) -> float | None:
    csv = Path(csv_path)
    candidates = [csv_path, str(csv), str(csv.resolve()), csv.name, csv.stem]
    for key in candidates:
        if key in allocation_map:
            return allocation_map[key]
    return None


def _build_per_csv_allocations(csv_paths: list[str], allocation_map: dict[str, float]) -> dict[str, float]:
    if not csv_paths:
        return {}

    per_csv_allocation: dict[str, float] = {}
    for csv_path in csv_paths:
        allocation = _resolve_asset_allocation_for_csv(csv_path, allocation_map)
        if allocation is None:
            per_csv_allocation[csv_path] = 1.0
            continue
        if allocation < 0:
            raise ValueError(f"--asset-size must be non-negative for '{csv_path}'")
        per_csv_allocation[csv_path] = float(allocation)
    return per_csv_allocation


def _safe_asset_slug(csv_path: str, used: set[str]) -> str:
    base = Path(csv_path).stem or "asset"
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in base)
    slug = slug.strip("_") or "asset"
    candidate = slug
    counter = 2
    while candidate in used:
        candidate = f"{slug}_{counter}"
        counter += 1
    used.add(candidate)
    return candidate


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




def _artifact_path(out_dir: Path, base_name: str, suffix: str, asset_slug: str | None = None) -> Path:
    if asset_slug is None:
        return out_dir / f"{base_name}{suffix}"
    return out_dir / f"{base_name}-{asset_slug}{suffix}"


def write_signal_intent_flat_timestamps(result, output_path: Path) -> Path:
    lines = ["Signal Intent Flat timestamps\n"]
    lines.append("================================\n")

    found = 0
    for trade in result.trades:
        ts = getattr(trade, "signal_intent_flat_time", None)
        if ts is None:
            continue
        found += 1
        lines.append(
            f"{ts.isoformat()} | exit_signal={trade.exit_signal} | side={trade.side} | "
            f"entry_time={trade.entry_time.isoformat()} | exit_time={trade.exit_time.isoformat()}\n"
        )

    if found == 0:
        lines.append("No 'Signal Intent Flat from ...' timestamps detected.\n")

    destination = output_path / "signal_intent_flat_timestamps.txt" if output_path.exists() and output_path.is_dir() else output_path
    destination.write_text("".join(lines), encoding="utf-8")
    return destination




def _build_trade_execution_log(result: BacktestResult) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    active_trade_id = 0
    active_position_units = 0.0

    for idx, event in enumerate(result.execution_events, start=1):
        event_type = str(event.event_type).strip().lower()
        if event_type == "entry":
            active_trade_id += 1
            active_position_units = 0.0
        trade_id = active_trade_id if active_trade_id > 0 else None
        equity_usd_raw = getattr(event, "capital_snapshot", None)
        equity_usd = float(equity_usd_raw) if equity_usd_raw is not None else None
        event_usd_amount = abs(float(event.price) * float(event.units))

        if event_type in {"entry", "add"}:
            active_position_units += float(event.units)
        elif event_type in {"reduce"}:
            active_position_units = max(0.0, active_position_units - float(event.units))
        elif event_type in {"exit", "equity_cutoff_exit", "liquidation", "stop_out"}:
            active_position_units = 0.0

        total_position_usd = abs(float(event.price) * active_position_units)
        event_leverage = (event_usd_amount / equity_usd) if equity_usd and equity_usd > 0 else None
        total_position_leverage = (total_position_usd / equity_usd) if equity_usd and equity_usd > 0 else None

        rows.append(
            {
                "sequence": idx,
                "trade_id": trade_id,
                "event_type": event.event_type,
                "time": event.time,
                "side": event.side,
                "price": event.price,
                "units": event.units,
                "usd_amount": event_usd_amount,
                "equity_usd": equity_usd,
                "event_leverage": event_leverage,
                "total_position_units": active_position_units,
                "total_position_usd": total_position_usd,
                "total_position_leverage": total_position_leverage,
                "reasoning": event.strategy_reason,
                "sizing_mode": getattr(event, "sizing_mode", None),
                "capital_snapshot": getattr(event, "capital_snapshot", None),
                "base_notional": getattr(event, "base_notional", None),
                "multiplier": getattr(event, "volatility_scale", None),
                "realized_volatility_annual": getattr(event, "realized_vol_annual", None),
                "scaled_notional": getattr(event, "scaled_notional", None),
            }
        )

        if event_type in {"exit", "equity_cutoff_exit", "liquidation", "stop_out"}:
            active_trade_id = 0

    return pd.DataFrame(rows)


def _is_stop_like_exit(value: object) -> bool:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return False
    stop_terms = ("stop", "stop out", "liquidation", "protective stop")
    return any(term in normalized for term in stop_terms)


def _build_signal_reason_monitor(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if trades.empty:
        empty_entry = pd.DataFrame(
            columns=[
                "entry_signal",
                "trades",
                "win_rate",
                "stop_exit_rate",
                "early_stop_rate",
                "net_pnl",
                "avg_return_pct",
                "avg_holding_bars",
                "avg_peak_unrealized_return_pct",
                "avg_giveback_from_peak_pct",
                "avg_capture_ratio_vs_peak",
            ]
        )
        empty_exit = pd.DataFrame(
            columns=[
                "exit_signal",
                "trades",
                "win_rate",
                "avg_return_pct",
                "avg_holding_bars",
                "net_pnl",
                "avg_giveback_from_peak_pct",
            ]
        )
        empty_pair = pd.DataFrame(
            columns=[
                "entry_signal",
                "exit_signal",
                "trades",
                "win_rate",
                "avg_holding_bars",
                "net_pnl",
                "avg_giveback_from_peak_pct",
            ]
        )
        return {"by_entry": empty_entry, "by_exit": empty_exit, "entry_exit_pairs": empty_pair}

    frame = trades.copy()
    frame["entry_signal"] = frame.get("entry_signal", pd.Series(index=frame.index, dtype="object")).fillna("Unknown").astype(str)
    frame["exit_signal"] = frame.get("exit_signal", pd.Series(index=frame.index, dtype="object")).fillna("Unknown").astype(str)
    frame["pnl"] = pd.to_numeric(frame.get("pnl", 0.0), errors="coerce").fillna(0.0)
    frame["return_pct"] = pd.to_numeric(frame.get("return_pct", 0.0), errors="coerce").fillna(0.0)
    frame["holding_bars"] = pd.to_numeric(frame.get("holding_bars", 0.0), errors="coerce").fillna(0.0)
    frame["is_win"] = frame["pnl"] > 0
    frame["is_stop_exit"] = frame["exit_signal"].map(_is_stop_like_exit)
    frame["is_early_stop"] = frame["is_stop_exit"] & (frame["holding_bars"] <= 2)
    frame["peak_unrealized_return_pct"] = pd.to_numeric(
        frame.get("peak_unrealized_return_pct", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    frame["giveback_from_peak_pct"] = pd.to_numeric(
        frame.get("giveback_from_peak_pct", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    frame["capture_ratio_vs_peak"] = pd.to_numeric(
        frame.get("capture_ratio_vs_peak", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)

    by_entry = (
        frame.groupby("entry_signal", dropna=False)
        .agg(
            trades=("pnl", "size"),
            win_rate=("is_win", "mean"),
            stop_exit_rate=("is_stop_exit", "mean"),
            early_stop_rate=("is_early_stop", "mean"),
            net_pnl=("pnl", "sum"),
            avg_return_pct=("return_pct", "mean"),
            avg_holding_bars=("holding_bars", "mean"),
            avg_peak_unrealized_return_pct=("peak_unrealized_return_pct", "mean"),
            avg_giveback_from_peak_pct=("giveback_from_peak_pct", "mean"),
            avg_capture_ratio_vs_peak=("capture_ratio_vs_peak", "mean"),
        )
        .reset_index()
        .sort_values(["stop_exit_rate", "trades"], ascending=[False, False])
    )

    by_exit = (
        frame.groupby("exit_signal", dropna=False)
        .agg(
            trades=("pnl", "size"),
            win_rate=("is_win", "mean"),
            avg_return_pct=("return_pct", "mean"),
            avg_holding_bars=("holding_bars", "mean"),
            net_pnl=("pnl", "sum"),
            avg_giveback_from_peak_pct=("giveback_from_peak_pct", "mean"),
        )
        .reset_index()
        .sort_values(["trades", "net_pnl"], ascending=[False, True])
    )

    entry_exit_pairs = (
        frame.groupby(["entry_signal", "exit_signal"], dropna=False)
        .agg(
            trades=("pnl", "size"),
            win_rate=("is_win", "mean"),
            avg_holding_bars=("holding_bars", "mean"),
            net_pnl=("pnl", "sum"),
            avg_giveback_from_peak_pct=("giveback_from_peak_pct", "mean"),
        )
        .reset_index()
        .sort_values(["trades", "net_pnl"], ascending=[False, True])
    )
    return {"by_entry": by_entry, "by_exit": by_exit, "entry_exit_pairs": entry_exit_pairs}


def _run_csv_paths_in_threads(
    csv_paths: list[str],
    worker: Callable[[str], tuple[str, BacktestResult, dict[str, object]]],
) -> list[tuple[str, BacktestResult, dict[str, object]]]:
    if len(csv_paths) <= 1:
        return [worker(csv_paths[0])]

    with ThreadPoolExecutor(max_workers=len(csv_paths)) as pool:
        futures = {csv_path: pool.submit(worker, csv_path) for csv_path in csv_paths}
        return [futures[csv_path].result() for csv_path in csv_paths]


def _run_csv_paths_serially(
    csv_paths: list[str],
    worker: Callable[[str], tuple[str, BacktestResult, dict[str, object]]],
) -> list[tuple[str, BacktestResult, dict[str, object]]]:
    return [worker(csv_path) for csv_path in csv_paths]


def _build_consolidated_result(results: list[BacktestResult], initial_capital: float, slippage_rate: float) -> BacktestResult:
    if not results:
        raise ValueError("No run results to consolidate")

    master_index = pd.Index(sorted(set().union(*(result.equity_curve.index for result in results))))
    if len(master_index) < 2:
        raise ValueError("Consolidated result requires at least two timestamps")

    aligned_equities: list[pd.Series] = []
    aligned_positions: list[pd.Series] = []
    for idx, result in enumerate(results):
        equity = result.equity_curve.astype(float).sort_index()
        aligned_equity = equity.reindex(master_index).ffill()
        aligned_equity = aligned_equity.fillna(float(equity.iloc[0]))
        aligned_equities.append(aligned_equity.rename(idx))

        positions = result.positions.astype(float).sort_index()
        aligned_position = positions.reindex(master_index).ffill().fillna(0.0)
        aligned_positions.append(aligned_position.rename(idx))

    equity_df = pd.concat(aligned_equities, axis=1)
    consolidated_equity = equity_df.sum(axis=1)
    consolidated_returns = consolidated_equity.pct_change().fillna(0.0)

    positions_df = pd.concat(aligned_positions, axis=1)
    consolidated_positions = positions_df.mean(axis=1)

    consolidated_trades = [trade for result in results for trade in result.trades]
    periods_per_year = infer_periods_per_year(consolidated_equity.index)
    consolidated_stats = compute_performance_stats(
        equity_curve=consolidated_equity,
        returns=consolidated_returns,
        trades=consolidated_trades,
        periods_per_year=periods_per_year,
        positions=consolidated_positions,
    )
    total_fees_nominal = float(sum(r.total_fees_paid for r in results))
    total_financing_nominal = float(sum(r.total_financing_paid for r in results))
    total_profit_before_fees_nominal = float(sum(r.total_profit_before_fees for r in results))
    total_initial_capital = float(sum(float(r.equity_curve.iloc[0]) for r in results))

    consolidated_stats["slippage_rate"] = float(slippage_rate)
    consolidated_stats["total_runs"] = float(len(results))
    consolidated_stats["initial_capital_total"] = total_initial_capital
    consolidated_stats["consolidated_initial_capital"] = float(consolidated_equity.iloc[0])
    consolidated_stats["requested_consolidated_initial_capital"] = float(initial_capital)
    consolidated_stats["total_fees_paid_nominal_sum"] = total_fees_nominal
    consolidated_stats["total_financing_paid_nominal_sum"] = total_financing_nominal
    consolidated_stats["total_profit_before_fees_nominal_sum"] = total_profit_before_fees_nominal

    quality_rows = pd.DataFrame([r.data_quality for r in results]).fillna(0.0)
    consolidated_quality = {
        "is_datetime_index": bool(quality_rows.get("is_datetime_index", pd.Series([True])).all()),
        "timezone_aware": bool(quality_rows.get("timezone_aware", pd.Series([True])).all()),
        "duplicate_timestamps": float(quality_rows.get("duplicate_timestamps", pd.Series([0.0])).sum()),
        "missing_bars": float(quality_rows.get("missing_bars", pd.Series([0.0])).sum()),
        "outlier_bars": float(quality_rows.get("outlier_bars", pd.Series([0.0])).sum()),
    }

    return BacktestResult(
        equity_curve=consolidated_equity,
        returns=consolidated_returns,
        positions=consolidated_positions,
        trades=consolidated_trades,
        stats=consolidated_stats,
        data_quality=consolidated_quality,
        execution_events=[event for result in results for event in result.execution_events],
        total_fees_paid=total_fees_nominal,
        total_financing_paid=total_financing_nominal,
        total_profit_before_fees=total_profit_before_fees_nominal,
    )
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True, help="One or more OHLCV CSV files")
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
    parser.add_argument("--equity-cutoff", type=float, default=None)
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--start", default=None, help="Inclusive start date/time (e.g. 2024-01-01)")
    parser.add_argument("--end", default=None, help="Inclusive end date/time (e.g. 2025-12-12)")
    parser.add_argument("--strategy", default="alligator_ao", help="Strategy name(s): alligator_ao, wiseman, ntd, bw, or comma-separated wiseman,ntd")
    parser.add_argument("--bw-1w-divergence-filter", "--1W-divergence-filter-bw", dest="bw_1w_divergence_filter_bars", type=int, default=0)
    parser.add_argument("--bw-1w-gator-open-lookback", type=int, default=0)
    parser.add_argument("--bw-1w-gator-open-percentile", type=float, default=50.0)
    parser.add_argument("--bw-1w-contracts", type=int, default=1)
    parser.add_argument("--bw-only-trade-1w-r", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bw-ntd-initial-fractal-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bw-ntd-initial-fractal-contracts", type=int, default=1)
    parser.add_argument("--bw-fractal-add-ons-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bw-fractal-add-on-contracts", "--Fractal-add-on-contracts", dest="bw_fractal_add_on_contracts", type=int, default=0)
    if "--bw-fractal-add-ons-enabled" not in parser._option_string_actions:
        parser.add_argument(
            "--bw-fractal-add-ons-enabled",
            dest="bw_fractal_add_ons_enabled",
            action="store_true",
            default=None,
            help="Enable/disable BW fractal add-on entries without changing --bw-fractal-add-on-contracts.",
        )
    if "--no-bw-fractal-add-ons-enabled" not in parser._option_string_actions:
        parser.add_argument(
            "--no-bw-fractal-add-ons-enabled",
            dest="bw_fractal_add_ons_enabled",
            action="store_false",
        )
    parser.add_argument("--bw-ntd-sleeping-gator-lookback", type=int, default=50)
    parser.add_argument("--bw-ntd-sleeping-gator-tightness-mult", type=float, default=0.75)
    parser.add_argument("--bw-ntd-ranging-lookback", type=int, default=20)
    parser.add_argument("--bw-ntd-ranging-max-span-pct", type=float, default=0.025)
    parser.add_argument("--bw-profit-protection-red-teeth-exit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bw-profit-protection-red-teeth-min-bars", type=int, default=3)
    parser.add_argument("--bw-profit-protection-red-teeth-min-unrealized-return", type=float, default=1.0)
    parser.add_argument("--bw-profit-protection-red-teeth-volatility-lookback", type=int, default=20)
    parser.add_argument("--bw-profit-protection-red-teeth-annualized-volatility-scaler", type=float, default=1.0)
    parser.add_argument(
        "--bw-profit-protection-red-teeth-require-gator-direction-alignment",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--bw-profit-protection-green-lips-exit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bw-profit-protection-green-lips-min-bars", type=int, default=3)
    parser.add_argument("--bw-profit-protection-green-lips-min-unrealized-return", type=float, default=1.1)
    parser.add_argument("--bw-profit-protection-green-lips-volatility-lookback", type=int, default=20)
    parser.add_argument("--bw-profit-protection-green-lips-annualized-volatility-scaler", type=float, default=1.0)
    parser.add_argument(
        "--bw-profit-protection-green-lips-require-gator-direction-alignment",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--bw-profit-protection-red-teeth-latch-min-unrealized-return",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--bw-profit-protection-green-lips-latch-min-unrealized-return",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--bw-profit-protection-zones-exit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bw-profit-protection-zones-min-bars", type=int, default=3)
    parser.add_argument("--bw-profit-protection-zones-min-unrealized-return", type=float, default=1.0)
    parser.add_argument("--bw-profit-protection-zones-volatility-lookback", type=int, default=20)
    parser.add_argument("--bw-profit-protection-zones-annualized-volatility-scaler", type=float, default=1.0)
    parser.add_argument("--bw-profit-protection-zones-min-same-color-bars", type=int, default=5)
    parser.add_argument("--bw-peak-drawdown-exit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bw-peak-drawdown-exit-pct", type=float, default=0.01)
    parser.add_argument("--bw-peak-drawdown-exit-volatility-lookback", type=int, default=20)
    parser.add_argument("--bw-peak-drawdown-exit-annualized-volatility-scaler", type=float, default=1.0)
    parser.add_argument("--bw-profit-protection-sigma-move-exit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bw-profit-protection-sigma-move-lookback", type=int, default=20)
    parser.add_argument("--bw-profit-protection-sigma-move-sigma", type=float, default=2.0)
    parser.add_argument(
        "--bw-close-on-underlying-gain-pct",
        type=float,
        default=0.0,
        help="Close BW positions as soon as underlying price gain reaches this decimal threshold (e.g. 0.03 = 3%). Disabled at 0.",
    )
    parser.add_argument("--allow-close-on-1w-d", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-close-on-1w-d-min-unrealized-return", type=float, default=0.0)
    parser.add_argument("--allow-close-on-1w-a", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-close-on-1w-a-min-unrealized-return", type=float, default=0.0)
    parser.add_argument("--chart-max-bars", type=int, default=None, help="Optional cap for bars embedded in local TradingView HTML")
    parser.add_argument(
        "--asset-size",
        action="append",
        default=[],
        help="Per-asset capital multiplier as KEY=MULTIPLIER. In multi-CSV mode, each run starts with --capital * MULTIPLIER. Unspecified assets keep the full --capital baseline (1.0x).",
    )
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
    parser = build_parser()
    args = parser.parse_args()
    strategy_names = _parse_strategy_selection(args.strategy)
    if args.chart_max_bars is not None and args.chart_max_bars < 2:
        raise ValueError("--chart-max-bars must be at least 2")
    asset_allocation_map = _parse_asset_allocations(args.asset_size)
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

    def _build_strategy() -> AlligatorAOStrategy | WisemanStrategy | NTDStrategy | BWStrategy | CombinedStrategy:
        if strategy_names == ("alligator_ao",):
            return AlligatorAOStrategy()
        if strategy_names == ("bw",):
            bw_fractal_add_ons_enabled = (
                args.bw_fractal_add_ons_enabled
                if args.bw_fractal_add_ons_enabled is not None
                else True
            )
            bw_fractal_add_on_contracts = (
                args.bw_fractal_add_on_contracts if bw_fractal_add_ons_enabled else 0
            )
            return BWStrategy(
                divergence_filter_bars=args.bw_1w_divergence_filter_bars,
                gator_open_filter_lookback=args.bw_1w_gator_open_lookback,
                gator_open_filter_min_percentile=args.bw_1w_gator_open_percentile,
                first_wiseman_contracts=args.bw_1w_contracts,
                only_trade_1w_reversals=args.bw_only_trade_1w_r,
                ntd_initial_fractal_enabled=args.bw_ntd_initial_fractal_enabled,
                ntd_initial_fractal_contracts=args.bw_ntd_initial_fractal_contracts,
                fractal_add_on_contracts=bw_fractal_add_on_contracts,
                ntd_sleeping_gator_lookback=args.bw_ntd_sleeping_gator_lookback,
                ntd_sleeping_gator_tightness_mult=args.bw_ntd_sleeping_gator_tightness_mult,
                ntd_ranging_lookback=args.bw_ntd_ranging_lookback,
                ntd_ranging_max_span_pct=args.bw_ntd_ranging_max_span_pct,
                red_teeth_profit_protection_enabled=args.bw_profit_protection_red_teeth_exit,
                red_teeth_profit_protection_min_bars=args.bw_profit_protection_red_teeth_min_bars,
                red_teeth_profit_protection_min_unrealized_return=args.bw_profit_protection_red_teeth_min_unrealized_return,
                red_teeth_profit_protection_volatility_lookback=args.bw_profit_protection_red_teeth_volatility_lookback,
                red_teeth_profit_protection_annualized_volatility_scaler=args.bw_profit_protection_red_teeth_annualized_volatility_scaler,
                red_teeth_profit_protection_require_gator_direction_alignment=(
                    args.bw_profit_protection_red_teeth_require_gator_direction_alignment
                ),
                green_lips_profit_protection_enabled=args.bw_profit_protection_green_lips_exit,
                green_lips_profit_protection_min_bars=args.bw_profit_protection_green_lips_min_bars,
                green_lips_profit_protection_min_unrealized_return=args.bw_profit_protection_green_lips_min_unrealized_return,
                green_lips_profit_protection_volatility_lookback=args.bw_profit_protection_green_lips_volatility_lookback,
                green_lips_profit_protection_annualized_volatility_scaler=(
                    args.bw_profit_protection_green_lips_annualized_volatility_scaler
                ),
                green_lips_profit_protection_require_gator_direction_alignment=(
                    args.bw_profit_protection_green_lips_require_gator_direction_alignment
                ),
                red_teeth_latch_min_unrealized_return=(
                    args.bw_profit_protection_red_teeth_latch_min_unrealized_return
                ),
                green_lips_latch_min_unrealized_return=(
                    args.bw_profit_protection_green_lips_latch_min_unrealized_return
                ),
                zones_profit_protection_enabled=args.bw_profit_protection_zones_exit,
                zones_profit_protection_min_bars=args.bw_profit_protection_zones_min_bars,
                zones_profit_protection_min_unrealized_return=args.bw_profit_protection_zones_min_unrealized_return,
                zones_profit_protection_volatility_lookback=args.bw_profit_protection_zones_volatility_lookback,
                zones_profit_protection_annualized_volatility_scaler=(
                    args.bw_profit_protection_zones_annualized_volatility_scaler
                ),
                zones_profit_protection_min_same_color_bars=args.bw_profit_protection_zones_min_same_color_bars,
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
                allow_close_on_1w_d=args.allow_close_on_1w_d,
                allow_close_on_1w_d_min_unrealized_return=args.allow_close_on_1w_d_min_unrealized_return,
                allow_close_on_1w_a=args.allow_close_on_1w_a,
                allow_close_on_1w_a_min_unrealized_return=args.allow_close_on_1w_a_min_unrealized_return,
            )
        strategies = []
        if "wiseman" in strategy_names:
            strategies.append(_build_wiseman_strategy())
        if "ntd" in strategy_names:
            strategies.append(_build_ntd_strategy())
        return strategies[0] if len(strategies) == 1 else CombinedStrategy(strategies)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    multi_asset = len(args.csv) > 1
    run_summaries: list[dict[str, object]] = []
    run_results: list[tuple[str, BacktestResult]] = []

    per_csv_allocation = _build_per_csv_allocations(args.csv, asset_allocation_map)

    slug_by_csv: dict[str, str | None] = {}
    if multi_asset:
        used_slugs: set[str] = set()
        for csv_path in args.csv:
            slug_by_csv[csv_path] = _safe_asset_slug(csv_path, used_slugs)
    else:
        slug_by_csv[args.csv[0]] = None

    def _run_single_csv(csv_path: str) -> tuple[str, BacktestResult, dict[str, object]]:
        data = load_ohlcv_csv(csv_path)
        data = filter_ohlcv_by_date(data, start=args.start, end=args.end)
        if len(data) < 2:
            raise ValueError(f"Filtered dataset has fewer than 2 bars for '{csv_path}'; widen --start/--end range")

        chart_data = data.tail(args.chart_max_bars).copy() if args.chart_max_bars else data
        asset_allocation = per_csv_allocation.get(csv_path, 0.0)
        asset_initial_capital = args.capital * asset_allocation if multi_asset else args.capital
        engine = BacktestEngine(
            BacktestConfig(
                initial_capital=asset_initial_capital,
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
        print(
            _runtime_estimate_message(
                engine=engine,
                data=data,
                build_strategy=_build_strategy,
                csv_path=csv_path,
            )
        )

        strategy = _build_strategy()
        result = engine.run(data, strategy)
        asset_slug = slug_by_csv[csv_path]

        pine_path = _artifact_path(out_dir, "trade_markers", ".pine", asset_slug)
        generate_trade_marker_pinescript(result.trades, str(pine_path))

        wiseman_pine_path = _artifact_path(out_dir, "first_wiseman_bearish", ".pine", asset_slug)
        generate_first_wiseman_bearish_pinescript(str(wiseman_pine_path))

        bullish_wiseman_pine_path = _artifact_path(out_dir, "first_wiseman_bullish", ".pine", asset_slug)
        generate_first_wiseman_bullish_pinescript(str(bullish_wiseman_pine_path))

        local_chart_path = _artifact_path(out_dir, "tradingview_local_chart", ".html", asset_slug)
        generate_local_tradingview_chart(
            chart_data,
            result.trades,
            str(local_chart_path),
            execution_events=result.execution_events,
            second_fill_prices=getattr(strategy, "signal_fill_prices_second", None),
            third_fill_prices=getattr(strategy, "signal_fill_prices_third", None),
            second_setup_side=getattr(strategy, "signal_second_wiseman_setup_side", None),
            second_fill_side=getattr(strategy, "signal_second_wiseman_fill_side", None),
            third_fill_side=getattr(strategy, "signal_third_wiseman_fill_side", None),
            third_setup_side=getattr(strategy, "signal_third_wiseman_setup_side", None),
            first_setup_side=getattr(
                strategy,
                "signal_first_wiseman_setup_marker_side",
                getattr(strategy, "signal_first_wiseman_setup_side", None),
            ),
            first_ignored_reason=getattr(strategy, "signal_first_wiseman_ignored_reason", None),
            first_reversal_side=getattr(strategy, "signal_first_wiseman_reversal_side", None),
            disambiguate_bw_add_on_markers=_strategy_includes_bw(strategy),
        )

        stats_path = _artifact_path(out_dir, "stats", ".json", asset_slug)
        stats_path.write_text(json.dumps(result.stats, indent=2), encoding="utf-8")

        quality_path = _artifact_path(out_dir, "data_quality", ".json", asset_slug)
        quality_path.write_text(json.dumps(result.data_quality, indent=2), encoding="utf-8")

        trades_path = _artifact_path(out_dir, "trades", ".csv", asset_slug)
        result.trades_dataframe().to_csv(trades_path, index=False)

        trade_execution_log_path = _artifact_path(out_dir, "trade_execution_log", ".csv", asset_slug)
        _build_trade_execution_log(result).to_csv(trade_execution_log_path, index=False)
        signal_reason_monitor_frames = _build_signal_reason_monitor(result.trades_dataframe())
        signal_reason_monitor_entry_path = _artifact_path(out_dir, "signal_reason_monitor_by_entry", ".csv", asset_slug)
        signal_reason_monitor_exit_path = _artifact_path(out_dir, "signal_reason_monitor_by_exit", ".csv", asset_slug)
        signal_reason_monitor_pair_path = _artifact_path(out_dir, "signal_reason_monitor_entry_exit_pairs", ".csv", asset_slug)
        signal_reason_monitor_frames["by_entry"].to_csv(signal_reason_monitor_entry_path, index=False)
        signal_reason_monitor_frames["by_exit"].to_csv(signal_reason_monitor_exit_path, index=False)
        signal_reason_monitor_frames["entry_exit_pairs"].to_csv(signal_reason_monitor_pair_path, index=False)

        report_path = _artifact_path(out_dir, "report", ".pdf", asset_slug)
        clean_report_path = _artifact_path(out_dir, "signal_diagnostics_report", ".pdf", asset_slug)
        cli_flags = vars(args).copy()
        cli_flags["csv"] = csv_path
        cli_flags["size_value"] = args.size_value
        cli_flags["asset_allocation"] = asset_allocation
        cli_flags["initial_capital"] = asset_initial_capital
        generate_backtest_pdf_report(result, report_path, cli_flags=cli_flags)
        generate_backtest_clean_pdf_report(result, clean_report_path)
        signal_intent_path = write_signal_intent_flat_timestamps(result, _artifact_path(out_dir, "signal_intent_flat_timestamps", ".txt", asset_slug))

        print(f"[{csv_path}] Backtest complete")
        if args.start or args.end:
            range_start = data.index[0].isoformat()
            range_end = data.index[-1].isoformat()
            print(f"[{csv_path}] Date range used: {range_start} -> {range_end}")
        print(f"[{csv_path}] Capital multiplier used: {asset_allocation:.4f} ({asset_initial_capital:,.2f} initial capital)")
        print(f"[{csv_path}] Position size value used: {args.size_value}")
        print(f"[{csv_path}] Trades: {len(result.trades)}")
        print(f"[{csv_path}] Stats written to: {stats_path}")
        print(f"[{csv_path}] Data quality written to: {quality_path}")
        print(f"[{csv_path}] Trade history written to: {trades_path}")
        print(f"[{csv_path}] Trade execution log written to: {trade_execution_log_path}")
        print(f"[{csv_path}] Signal reason monitor (entry) written to: {signal_reason_monitor_entry_path}")
        print(f"[{csv_path}] Signal reason monitor (exit) written to: {signal_reason_monitor_exit_path}")
        print(f"[{csv_path}] Signal reason monitor (entry/exit pairs) written to: {signal_reason_monitor_pair_path}")
        print(f"[{csv_path}] TradingView Pine written to: {pine_path}")
        print(f"[{csv_path}] TradingView bearish 1st Wiseman Pine written to: {wiseman_pine_path}")
        print(f"[{csv_path}] TradingView bullish 1st Wiseman Pine written to: {bullish_wiseman_pine_path}")
        print(f"[{csv_path}] Local TradingView chart written to: {local_chart_path}")
        if len(chart_data) != len(data):
            print(f"[{csv_path}] Chart bars embedded: {len(chart_data):,} of {len(data):,} (latest window)")
        print(f"[{csv_path}] PDF report written to: {report_path}")
        print(f"[{csv_path}] Clean PDF report written to: {clean_report_path}")
        print(f"[{csv_path}] Signal Intent Flat timestamps written to: {signal_intent_path}")

        summary = {
            "csv": csv_path,
            "asset": asset_slug or Path(csv_path).stem,
            "allocation": asset_allocation,
            "initial_capital": asset_initial_capital,
            "trade_size_value": args.size_value,
            "trades": len(result.trades),
            "net_profit": result.stats.get("net_profit"),
            "return_pct": result.stats.get("return_pct"),
            "max_drawdown_pct": result.stats.get("max_drawdown_pct"),
        }
        return csv_path, result, summary

    # Multi-asset CLI runs are intentionally serialized so each asset's result is
    # fully deterministic and cannot be affected by concurrent indicator/strategy
    # execution in other worker threads.
    for csv_path, result, summary in _run_csv_paths_serially(args.csv, _run_single_csv):
        run_results.append((csv_path, result))
        run_summaries.append(summary)

    if multi_asset:
        summary_path = out_dir / "multi_asset_summary.csv"
        pd.DataFrame(run_summaries).to_csv(summary_path, index=False)
        print(f"Multi-asset summary written to: {summary_path}")

        consolidated_initial_capital = float(sum(args.capital * per_csv_allocation[csv_path] for csv_path in args.csv))
        consolidated_result = _build_consolidated_result([result for _, result in run_results], initial_capital=consolidated_initial_capital, slippage_rate=args.slippage)
        consolidated_stats_path = out_dir / "consolidated_stats.json"
        consolidated_stats_path.write_text(json.dumps(consolidated_result.stats, indent=2), encoding="utf-8")

        consolidated_trades_path = out_dir / "consolidated_trades.csv"
        consolidated_trade_frames = []
        consolidated_execution_frames = []
        for csv_path, result in run_results:
            asset_name = Path(csv_path).stem
            frame = result.trades_dataframe()
            frame.insert(0, "asset", asset_name)
            consolidated_trade_frames.append(frame)

            execution_frame = _build_trade_execution_log(result)
            execution_frame.insert(0, "asset", asset_name)
            consolidated_execution_frames.append(execution_frame)
        pd.concat(consolidated_trade_frames, ignore_index=True).to_csv(consolidated_trades_path, index=False)

        consolidated_trade_execution_log_path = out_dir / "consolidated_trade_execution_log.csv"
        pd.concat(consolidated_execution_frames, ignore_index=True).to_csv(consolidated_trade_execution_log_path, index=False)
        consolidated_reason_monitor_frames = _build_signal_reason_monitor(
            pd.concat(consolidated_trade_frames, ignore_index=True)
        )
        consolidated_reason_entry_path = out_dir / "consolidated_signal_reason_monitor_by_entry.csv"
        consolidated_reason_exit_path = out_dir / "consolidated_signal_reason_monitor_by_exit.csv"
        consolidated_reason_pairs_path = out_dir / "consolidated_signal_reason_monitor_entry_exit_pairs.csv"
        consolidated_reason_monitor_frames["by_entry"].to_csv(consolidated_reason_entry_path, index=False)
        consolidated_reason_monitor_frames["by_exit"].to_csv(consolidated_reason_exit_path, index=False)
        consolidated_reason_monitor_frames["entry_exit_pairs"].to_csv(consolidated_reason_pairs_path, index=False)

        consolidated_report_path = out_dir / "consolidated_report.pdf"
        consolidated_clean_report_path = out_dir / "consolidated_signal_diagnostics_report.pdf"
        consolidated_flags = vars(args).copy()
        consolidated_flags["csv"] = args.csv
        generate_backtest_pdf_report(
            consolidated_result,
            consolidated_report_path,
            cli_flags=consolidated_flags,
            asset_level_results=[(Path(csv_path).stem, result) for csv_path, result in run_results],
        )
        generate_backtest_clean_pdf_report(consolidated_result, consolidated_clean_report_path)

        print(f"Consolidated stats written to: {consolidated_stats_path}")
        print(f"Consolidated trades written to: {consolidated_trades_path}")
        print(f"Consolidated trade execution log written to: {consolidated_trade_execution_log_path}")
        print(f"Consolidated signal reason monitor (entry) written to: {consolidated_reason_entry_path}")
        print(f"Consolidated signal reason monitor (exit) written to: {consolidated_reason_exit_path}")
        print(f"Consolidated signal reason monitor (entry/exit pairs) written to: {consolidated_reason_pairs_path}")
        print(f"Consolidated PDF report written to: {consolidated_report_path}")
        print(f"Consolidated clean PDF report written to: {consolidated_clean_report_path}")


if __name__ == "__main__":
    main()
