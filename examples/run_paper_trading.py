from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from LiveData.live_kcex_chart import KCEXClient
from backtesting import (
    AlligatorAOStrategy,
    BacktestConfig,
    BacktestEngine,
    BWStrategy,
    CombinedStrategy,
    ExecutionSignal,
    LiveBar,
    NTDStrategy,
    PaperTradingEngine,
    WisemanStrategy,
    compute_performance_stats,
    parse_trade_size_equity_milestones,
)
from backtesting.engine import ExecutionEvent
from backtesting.local_chart import (
    _ac_histogram_from_data,
    _alligator_series_from_data,
    _ao_histogram_from_data,
    _candles_from_data,
    _combine_markers,
    _execution_event_markers,
    _execution_event_lines,
    _execution_trade_path_lines,
    _first_wiseman_engine_markers,
    _first_wiseman_ignored_markers,
    _gator_profit_protection_fallback_overlays,
    _second_wiseman_markers,
    _valid_third_wiseman_fractal_markers,
    _williams_zones_colors,
    _wiseman_fill_entry_markers,
    _wiseman_markers,
)
from backtesting.stats import infer_periods_per_year

_INTERVAL_SECONDS = {
    "Min1": 60,
    "Min3": 180,
    "Min5": 300,
    "Min15": 900,
    "Min30": 1800,
    "Min60": 3600,
    "Hour4": 14400,
    "Hour12": 43200,
    "Day1": 86400,
}


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


class SupportsFetchKline(Protocol):
    def fetch_kline(self, symbol: str, interval: str, limit: int = 400) -> list[Any]: ...


class _SignalStrategy(Protocol):
    execute_on_signal_bar: bool

    def generate_signals(self, data: pd.DataFrame) -> pd.Series: ...


@dataclass(slots=True)
class _PaperTradeMetricsRow:
    pnl: float
    holding_bars: int


@dataclass(slots=True)
class _CompletedTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float
    holding_bars: int
    entry_reason: str
    exit_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "side": self.side,
            "quantity": float(self.quantity),
            "entry_price": float(self.entry_price),
            "exit_price": float(self.exit_price),
            "pnl": float(self.pnl),
            "return_pct": float(self.return_pct),
            "holding_bars": int(self.holding_bars),
            "entry_reason": self.entry_reason,
            "exit_reason": self.exit_reason,
        }


@dataclass(slots=True)
class PaperTradingArtifacts:
    summary_path: Path
    fills_path: Path
    dashboard_path: Path
    dashboard_data_path: Path
    dashboard_script_path: Path
    status_path: Path
    trades_path: Path


@dataclass(slots=True)
class _ArtifactWriteState:
    pending_generation: int = 0
    completed_generation: int = 0
    report_generation: int = 0
    stop_requested: bool = False
    error: BaseException | None = None
    latest_artifacts: PaperTradingArtifacts | None = None
    latest_payload: dict[str, Any] | None = None
    artifact_content_digests: dict[str, str] = field(default_factory=dict)
    locked_artifact_paths: set[str] = field(default_factory=set)


def _atomic_write_text(path: Path, content: str) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(content, encoding="utf-8")
    _replace_temp_file(temp_path, path)


def _atomic_write_frame_csv(path: Path, frame: pd.DataFrame) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temp_path, index=False)
    _replace_temp_file(temp_path, path)


def _replace_temp_file(
    temp_path: Path,
    path: Path,
    *,
    retries: int = 10,
    retry_delay_seconds: float = 0.2,
) -> None:
    last_error: PermissionError | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            temp_path.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt >= max(1, retries):
                break
            time.sleep(max(0.0, retry_delay_seconds))
    raise last_error if last_error is not None else RuntimeError(f"Failed to replace {path} with {temp_path}")


def _best_effort_atomic_write_text(path: Path, content: str) -> bool:
    try:
        _atomic_write_text(path, content)
    except PermissionError as exc:
        print(f"[paper-trading] Skipping locked artifact write for {path}: {exc}", file=sys.stderr, flush=True)
        return False
    return True


def _best_effort_atomic_write_frame_csv(path: Path, frame: pd.DataFrame) -> bool:
    try:
        _atomic_write_frame_csv(path, frame)
    except PermissionError as exc:
        print(f"[paper-trading] Skipping locked artifact write for {path}: {exc}", file=sys.stderr, flush=True)
        return False
    return True


def _content_digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _log_locked_artifact_once(state: _ArtifactWriteState, path: Path, exc: PermissionError) -> None:
    path_key = str(path)
    if path_key in state.locked_artifact_paths:
        return
    state.locked_artifact_paths.add(path_key)
    print(f"[paper-trading] Skipping locked artifact write for {path}: {exc}", file=sys.stderr, flush=True)


def _clear_locked_artifact_notice(state: _ArtifactWriteState, path: Path) -> None:
    path_key = str(path)
    if path_key not in state.locked_artifact_paths:
        return
    state.locked_artifact_paths.discard(path_key)
    print(f"[paper-trading] Artifact write recovered for {path}", file=sys.stderr, flush=True)


def _write_cached_text_artifact(state: _ArtifactWriteState, path: Path, content: str) -> bool:
    path_key = str(path)
    digest = _content_digest(content)
    if state.artifact_content_digests.get(path_key) == digest and path.exists():
        _clear_locked_artifact_notice(state, path)
        return False
    try:
        _atomic_write_text(path, content)
    except PermissionError as exc:
        _log_locked_artifact_once(state, path, exc)
        return False
    state.artifact_content_digests[path_key] = digest
    _clear_locked_artifact_notice(state, path)
    return True


def _write_cached_frame_csv_artifact(state: _ArtifactWriteState, path: Path, frame: pd.DataFrame) -> bool:
    return _write_cached_text_artifact(state, path, frame.to_csv(index=False))


def _tag_markers(markers: list[dict[str, str | int | float]], marker_group: str) -> list[dict[str, str | int | float]]:
    return [{**marker, "markerGroup": marker_group} for marker in markers]


def _ntd_fill_entry_markers(
    data: pd.DataFrame,
    fill_prices: pd.Series | None,
    fractal_position_side: pd.Series | None,
    contracts: pd.Series | None = None,
) -> list[dict[str, str | int | float]]:
    if data.empty or not isinstance(fill_prices, pd.Series) or not isinstance(fractal_position_side, pd.Series):
        return []
    fills = fill_prices.reindex(data.index)
    sides = fractal_position_side.reindex(data.index).fillna(0).astype("int8")
    aligned_contracts = contracts.reindex(data.index).fillna(0.0).astype("float64") if isinstance(contracts, pd.Series) else None
    markers: list[dict[str, str | int | float]] = []
    for i, ts in enumerate(data.index):
        side = int(sides.iloc[i])
        fill = fills.iloc[i]
        if side == 0 or pd.isna(fill):
            continue
        is_add_on = False
        if aligned_contracts is not None:
            current_contracts = float(aligned_contracts.iloc[i])
            previous_contracts = float(aligned_contracts.iloc[i - 1]) if i > 0 else 0.0
            is_add_on = (
                int(np.sign(previous_contracts)) == side
                and abs(current_contracts) > (abs(previous_contracts) + 1e-12)
            )
        row = data.iloc[i]
        row_low = float(row["low"])
        row_high = float(row["high"])
        row_close = float(row["close"])
        marker_offset = max((row_high - row_low) * 0.35, abs(row_close) * 0.001, 1e-8)
        bullish = side > 0
        markers.append(
            {
                "time": int(ts.timestamp()),
                "position": "belowBar" if bullish else "aboveBar",
                "price": row_low - marker_offset if bullish else row_high + marker_offset,
                "color": "#0ea5e9" if bullish else "#f97316",
                "shape": "arrowUp" if bullish else "arrowDown",
                "text": "NTD-A" if is_add_on else "NTD-E",
            }
        )
    return markers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the paper-trading engine in real time from live KCEX candles")
    parser.add_argument("--symbol", default="BTC_USDT")
    parser.add_argument("--interval", default="Min60", choices=sorted(_INTERVAL_SECONDS))
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--warmup-bars", type=int, default=400)
    parser.add_argument("--max-cycles", type=int, default=None, help="Optional safety cap for polling cycles")
    parser.add_argument("--fetch-failure-sleep-seconds", type=float, default=5.0)
    parser.add_argument(
        "--fetch-max-consecutive-failures",
        type=int,
        default=0,
        help="Abort after this many consecutive fetch failures; 0 keeps retrying forever.",
    )
    parser.add_argument("--capital", type=float, default=10_000)
    parser.add_argument("--fee", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument("--spread", type=float, default=0.0)
    parser.add_argument("--order-type", default="market", choices=["market", "limit", "stop", "stop_limit"])
    parser.add_argument("--limit-offset-pct", type=float, default=0.001)
    parser.add_argument("--stop-offset-pct", type=float, default=0.001)
    parser.add_argument("--stop-limit-offset-pct", type=float, default=0.0005)
    parser.add_argument("--size-mode", default="percent_of_equity", choices=["percent_of_equity", "usd", "units", "hybrid_min_usd_percent", "volatility_scaled", "stop_loss_scaled", "equity_milestone_usd"])
    parser.add_argument("--size-value", type=float, default=1.0)
    parser.add_argument("--size-min-usd", type=float, default=0.0)
    parser.add_argument("--size-equity-milestones", default="", help="Comma-separated EQUITY:USD step pairs for equity_milestone_usd sizing, e.g. 15000:1500,20000:2000")
    parser.add_argument("--volatility-target-annual", dest="volatility_target_annual", type=float, default=0.15)
    parser.add_argument("--volatility-target-annualized", dest="volatility_target_annual", type=float)
    parser.add_argument("--volatility-lookback", type=int, default=20)
    parser.add_argument("--volatility-min-scale", type=float, default=0.25)
    parser.add_argument("--volatility-max-scale", type=float, default=3.0)
    parser.add_argument("--max-leverage", type=float, default=None)
    parser.add_argument("--max-position-size", type=float, default=None)
    parser.add_argument("--leverage-stop-out", type=float, default=0.0)
    parser.add_argument("--borrow-annual", type=float, default=0.0)
    parser.add_argument("--funding-per-period", type=float, default=0.0)
    parser.add_argument("--overnight-annual", type=float, default=0.0)
    parser.add_argument("--max-loss", type=float, default=None)
    parser.add_argument("--max-loss-pct-of-equity", type=float, default=None)
    parser.add_argument("--equity-cutoff", type=float, default=None)
    parser.add_argument("--strategy", default="wiseman", help="Strategy name(s): alligator_ao, wiseman, ntd, bw, or comma-separated wiseman,ntd")
    parser.add_argument("--bw-1w-divergence-filter", "--1W-divergence-filter-bw", dest="bw_1w_divergence_filter_bars", type=int, default=0)
    parser.add_argument("--bw-1w-gator-open-lookback", type=int, default=0)
    parser.add_argument("--bw-1w-gator-open-percentile", type=float, default=50.0)
    parser.add_argument("--bw-1w-contracts", type=int, default=1)
    parser.add_argument("--bw-only-trade-1w-r", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bw-ntd-initial-fractal-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bw-ntd-initial-fractal-contracts", type=int, default=1)
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
    parser.add_argument("--bw-close-on-underlying-gain-pct", type=float, default=0.0)
    parser.add_argument("--allow-close-on-1w-d", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-close-on-1w-d-min-unrealized-return", type=float, default=0.0)
    parser.add_argument("--allow-close-on-1w-a", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-close-on-1w-a-min-unrealized-return", type=float, default=0.0)
    parser.add_argument("--out", default="artifacts_paper")
    parser.add_argument("--summary-name", default="paper_summary.json")
    parser.add_argument("--fills-name", default="paper_fills.csv")
    parser.add_argument("--status-name", default="paper_status.md")
    parser.add_argument("--trades-name", default="paper_trades.md")
    parser.add_argument("--dashboard-name", default="paper_dashboard.html")
    parser.add_argument("--dashboard-data-name", default="paper_dashboard_data.json")
    parser.add_argument("--dashboard-script-name", default="paper_dashboard_data.js")
    parser.add_argument("--gator-width-lookback", type=int, default=50)
    parser.add_argument("--gator-width-mult", type=float, default=1.0)
    parser.add_argument("--gator-width-valid-factor", type=float, default=1.0)
    parser.add_argument("--ntd-ao-ac-near-zero-lookback", type=int, default=50)
    parser.add_argument("--ntd-ao-ac-near-zero-factor", type=float, default=0.25)
    parser.add_argument("--ntd-require-gator-close-reset", action=argparse.BooleanOptionalAction, default=True)
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


def _build_strategy(args: argparse.Namespace) -> AlligatorAOStrategy | WisemanStrategy | NTDStrategy | BWStrategy | CombinedStrategy:
    strategy_names = _parse_strategy_selection(args.strategy)

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
            # Paper trading keeps both profit-protection paths active at all times
            # so live signals/chart overlays stay aligned with the requested close logic.
            teeth_profit_protection_enabled=True,
            teeth_profit_protection_min_bars=args.wiseman_profit_protection_min_bars,
            teeth_profit_protection_min_unrealized_return=args.wiseman_profit_protection_min_unrealized_return,
            teeth_profit_protection_credit_unrealized_before_min_bars=args.wiseman_profit_protection_credit_unrealized_before_min_bars,
            teeth_profit_protection_require_gator_open=args.wiseman_profit_protection_require_gator_open,
            profit_protection_volatility_lookback=args.wiseman_profit_protection_volatility_lookback,
                profit_protection_annualized_volatility_scaler=args.wiseman_profit_protection_annualized_volatility_scaler,
            lips_profit_protection_enabled=True,
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
            require_gator_close_reset=args.ntd_require_gator_close_reset,
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
            first_wiseman_contracts=args.bw_1w_contracts,
            only_trade_1w_reversals=args.bw_only_trade_1w_r,
            ntd_initial_fractal_enabled=args.bw_ntd_initial_fractal_enabled,
            ntd_initial_fractal_contracts=args.bw_ntd_initial_fractal_contracts,
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


def _build_backtest_config(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        initial_capital=float(args.capital),
        fee_rate=float(args.fee),
        slippage_rate=float(args.slippage),
        spread_rate=float(args.spread),
        order_type=str(args.order_type),
        limit_offset_pct=float(args.limit_offset_pct),
        stop_offset_pct=float(args.stop_offset_pct),
        stop_limit_offset_pct=float(args.stop_limit_offset_pct),
        trade_size_mode=str(args.size_mode),
        trade_size_value=float(args.size_value),
        trade_size_min_usd=float(args.size_min_usd),
        trade_size_equity_milestones=parse_trade_size_equity_milestones(args.size_equity_milestones),
        volatility_target_annual=float(args.volatility_target_annual),
        volatility_lookback=int(args.volatility_lookback),
        volatility_min_scale=float(args.volatility_min_scale),
        volatility_max_scale=float(args.volatility_max_scale),
        max_leverage=args.max_leverage,
        max_position_size=args.max_position_size,
        leverage_stop_out_pct=float(args.leverage_stop_out),
        borrow_rate_annual=float(args.borrow_annual),
        funding_rate_per_period=float(args.funding_per_period),
        overnight_rate_annual=float(args.overnight_annual),
        max_loss=args.max_loss,
        max_loss_pct_of_equity=args.max_loss_pct_of_equity,
        equity_cutoff=args.equity_cutoff,
        close_open_position_on_last_bar=False,
    )


def _candles_to_dataframe(candles: list[Any]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for candle in candles:
        if isinstance(candle, dict):
            raw = candle
        else:
            raw = {
                "time": getattr(candle, "time"),
                "open": getattr(candle, "open"),
                "high": getattr(candle, "high"),
                "low": getattr(candle, "low"),
                "close": getattr(candle, "close"),
                "volume": getattr(candle, "volume", 0.0),
            }
        rows.append(
            {
                "timestamp": pd.to_datetime(int(raw["time"]), unit="s", utc=True),
                "open": float(raw["open"]),
                "high": float(raw["high"]),
                "low": float(raw["low"]),
                "close": float(raw["close"]),
                "volume": float(raw.get("volume", 0.0) or 0.0),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").set_index("timestamp")
    return frame[["open", "high", "low", "close", "volume"]]


def _closed_bars(data: pd.DataFrame, interval_seconds: int, now: pd.Timestamp) -> pd.DataFrame:
    if data.empty:
        return data
    cutoff = now - pd.Timedelta(seconds=interval_seconds)
    return data[data.index <= cutoff]


def _resolve_contracts(signal_value: float, signal_contracts: pd.Series | None, signal_index: int) -> float:
    if int(np.sign(signal_value)) == 0:
        return 0.0
    if signal_contracts is not None:
        raw = abs(float(signal_contracts.iloc[signal_index]))
    else:
        raw = abs(float(signal_value))
    return raw if raw > 0 else 1.0


def _resolve_order_request(*, config: BacktestConfig, side: int, prev_close: float, signal_fill: float | None) -> dict[str, float | str | None]:
    order_type = config.order_type.lower()
    if order_type == "market":
        return {"order_type": "market", "limit_price": None, "stop_price": None}

    if signal_fill is not None and np.isfinite(signal_fill) and signal_fill > 0:
        if order_type == "limit":
            return {"order_type": "limit", "limit_price": float(signal_fill), "stop_price": None}
        if order_type == "stop":
            return {"order_type": "stop", "limit_price": None, "stop_price": float(signal_fill)}
        stop_price = float(signal_fill)
    else:
        if order_type == "limit":
            limit_price = prev_close * (1 - config.limit_offset_pct if side > 0 else 1 + config.limit_offset_pct)
            return {"order_type": "limit", "limit_price": float(limit_price), "stop_price": None}
        stop_price = prev_close * (1 + config.stop_offset_pct if side > 0 else 1 - config.stop_offset_pct)
        if order_type == "stop":
            return {"order_type": "stop", "limit_price": None, "stop_price": float(stop_price)}

    limit_price = stop_price * (1 + config.stop_limit_offset_pct if side > 0 else 1 - config.stop_limit_offset_pct)
    return {"order_type": "stop_limit", "limit_price": float(limit_price), "stop_price": float(stop_price)}


def _signed_open_order_projection(engine: PaperTradingEngine) -> float:
    projected_qty = engine.position.quantity
    for order in engine.open_orders:
        signed_qty = order.quantity if order.side == "buy" else -order.quantity
        if order.reduce_only:
            if projected_qty > 0:
                projected_qty = max(0.0, projected_qty + signed_qty)
            elif projected_qty < 0:
                projected_qty = min(0.0, projected_qty + signed_qty)
        else:
            projected_qty += signed_qty
    return float(projected_qty)


def _build_signal(
    *,
    symbol: str,
    timestamp: pd.Timestamp,
    target_qty: float,
    current_qty: float,
    projected_qty: float,
    order_request: dict[str, float | str | None],
    signal_fill: float | None,
) -> ExecutionSignal | None:
    if np.isclose(target_qty, projected_qty, atol=1e-12):
        return None

    current_side = int(np.sign(current_qty))
    target_side = int(np.sign(target_qty))
    target_abs = abs(float(target_qty))
    current_abs = abs(float(current_qty))
    signal_metadata = (
        {"market_fill_price": float(signal_fill)}
        if str(order_request["order_type"]) == "market" and signal_fill is not None and np.isfinite(signal_fill) and signal_fill > 0
        else {}
    )

    if target_side == 0:
        if current_side == 0:
            return None
        return ExecutionSignal(
            symbol=symbol,
            timestamp=timestamp,
            action="exit",
            order_type=str(order_request["order_type"]),
            quantity=current_abs,
            limit_price=order_request["limit_price"],
            stop_price=order_request["stop_price"],
            cancel_existing_orders=True,
            metadata=signal_metadata,
        )

    side_label = "buy" if target_side > 0 else "sell"
    if current_side == 0:
        return ExecutionSignal(
            symbol=symbol,
            timestamp=timestamp,
            action="enter",
            side=side_label,
            order_type=str(order_request["order_type"]),
            quantity=target_abs,
            limit_price=order_request["limit_price"],
            stop_price=order_request["stop_price"],
            cancel_existing_orders=True,
            metadata=signal_metadata,
        )

    if current_side != target_side:
        return ExecutionSignal(
            symbol=symbol,
            timestamp=timestamp,
            action="reverse",
            side=side_label,
            order_type=str(order_request["order_type"]),
            quantity=target_abs,
            limit_price=order_request["limit_price"],
            stop_price=order_request["stop_price"],
            cancel_existing_orders=True,
            metadata=signal_metadata,
        )

    delta = target_abs - current_abs
    if np.isclose(delta, 0.0, atol=1e-12):
        return None
    if delta > 0:
        return ExecutionSignal(
            symbol=symbol,
            timestamp=timestamp,
            action="scale",
            side=side_label,
            order_type=str(order_request["order_type"]),
            quantity=float(delta),
            limit_price=order_request["limit_price"],
            stop_price=order_request["stop_price"],
            cancel_existing_orders=True,
            metadata=signal_metadata,
        )

    return ExecutionSignal(
        symbol=symbol,
        timestamp=timestamp,
        action="exit",
        order_type=str(order_request["order_type"]),
        quantity=float(abs(delta)),
        limit_price=order_request["limit_price"],
        stop_price=order_request["stop_price"],
        cancel_existing_orders=True,
        metadata=signal_metadata,
    )


def _compact_strategy_reason(reason: str | None) -> str | None:
    if reason is None:
        return None

    normalized = str(reason).strip()
    if normalized == "":
        return None

    direct_mapping = {
        "Bullish 1W": "1W",
        "Bearish 1W": "1W",
        "Bullish 1W-R": "1W-R",
        "Bearish 1W-R": "1W-R",
        "Bullish 2W": "2W",
        "Bearish 2W": "2W",
        "Bullish 3W": None,
        "Bearish 3W": None,
        "Strategy Profit Protection Green Gator": "Green PP",
        "Strategy Profit Protection Red Gator": "Red PP",
        "Green Gator PP": "Green PP",
        "Red Gator PP": "Red PP",
        "Green Gator Lips PP": "Green PP",
        "Red Gator Teeth PP": "Red PP",
        "Green Gator": "Green PP",
        "Red Gator": "Red PP",
        "Strategy Stop Loss Bullish 1W": "1W Stop",
        "Strategy Stop Loss Bearish 1W": "1W Stop",
        "1W Reversal Stop": "1W Stop",
        "Strategy Reversal to Bullish 1W": "1W-R",
        "Strategy Reversal to Bearish 1W": "1W-R",
        "Bullish 1W Reversal": "1W-R",
        "Bearish 1W Reversal": "1W-R",
        "protective_stop": "Protective Stop",
        "take_profit": "Take Profit",
    }
    if normalized in direct_mapping:
        return direct_mapping[normalized]

    if normalized.startswith("Signal Intent Flat from "):
        return normalized.removeprefix("Signal Intent Flat from ")
    if normalized.startswith("Signal Intent Flip to "):
        return normalized.removeprefix("Signal Intent Flip to ")
    if normalized.startswith("Strategy Exit Reason: "):
        return normalized.removeprefix("Strategy Exit Reason: ")
    if normalized.startswith("Bullish ") or normalized.startswith("Bearish "):
        parts = normalized.split(" ", maxsplit=1)
        if len(parts) == 2:
            return parts[1]
    return normalized


def _strategy_has_third_wiseman_enabled(strategy: _SignalStrategy) -> bool:
    if isinstance(strategy, WisemanStrategy):
        return bool(strategy.third_wiseman_contracts > 0)
    if isinstance(strategy, CombinedStrategy):
        return any(_strategy_has_third_wiseman_enabled(component) for component in strategy.strategies)
    return False


def _strategy_includes_bw(strategy: _SignalStrategy) -> bool:
    if isinstance(strategy, BWStrategy):
        return True
    if isinstance(strategy, CombinedStrategy):
        return any(_strategy_includes_bw(component) for component in strategy.strategies)
    return False


def _strategy_reasons_for_signal(
    strategy: _SignalStrategy,
    engine: BacktestEngine,
    *,
    signal_index: int,
    current_qty: float,
    target_qty: float,
) -> tuple[str | None, str | None]:
    current_side = int(np.sign(current_qty))
    target_side = int(np.sign(target_qty))

    entry_reason: str | None = None
    exit_reason: str | None = None

    if target_side != 0:
        entry_reason = _compact_strategy_reason(engine._infer_entry_signal(strategy, signal_index=signal_index, desired_position=target_side))

    if current_side != 0 and (target_side == 0 or target_side != current_side):
        exit_reason = _compact_strategy_reason(
            engine._infer_exit_signal(
                strategy,
                signal_index=signal_index,
                prior_position=current_side,
                desired_position=target_side,
            )
        )

    if current_side != 0 and target_side == current_side and abs(target_qty) < abs(current_qty):
        exit_reason = exit_reason or "Reduce"

    return entry_reason, exit_reason


def _format_number(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    numeric = float(value)
    return f"{numeric:,.{digits}f}"


def _format_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:,.{digits}f}%"


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], empty_message: str) -> str:
    if not rows:
        return empty_message
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body_lines = []
    for row in rows:
        body_lines.append("| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |")
    return "\n".join([header, separator, *body_lines])


def _build_dashboard_html(data_script_filename: str) -> str:
    safe_data_script_filename = json.dumps(data_script_filename)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Paper Trading Dashboard</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0f172a;
      --panel: #111827;
      --panel-border: #1f2937;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --good: #22c55e;
      --bad: #ef4444;
      --warn: #f59e0b;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Arial, sans-serif; background: var(--bg); color: var(--text); }}
    header {{ padding: 20px 24px 8px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .subtle {{ color: var(--muted); font-size: 14px; }}
    .layout {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 16px; padding: 0 24px 24px; }}
    .card {{ background: var(--panel); border: 1px solid var(--panel-border); border-radius: 12px; padding: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.25); }}
    .span-12 {{ grid-column: span 12; }}
    .span-8 {{ grid-column: span 8; }}
    .span-6 {{ grid-column: span 6; }}
    .span-4 {{ grid-column: span 4; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
    .stat {{ border: 1px solid #1e293b; border-radius: 10px; padding: 12px; background: rgba(15, 23, 42, 0.65); }}
    .stat .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .stat .value {{ margin-top: 6px; font-size: 22px; font-weight: bold; }}
    .stat .value.good {{ color: var(--good); }}
    .stat .value.bad {{ color: var(--bad); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ text-align: left; padding: 8px 6px; border-bottom: 1px solid #1e293b; }}
    th {{ color: var(--muted); font-weight: 600; }}
    .chart {{ width: 100%; height: 260px; border-radius: 10px; background: linear-gradient(180deg, rgba(30,41,59,0.45), rgba(15,23,42,0.9)); border: 1px solid #1e293b; }}
    .chart-toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
    }}
    .chart-toolbar .subtle {{ flex: 1 1 220px; }}
    .indicators {{ display: flex; flex-wrap: wrap; gap: 10px; }}
    .indicator-toggle {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
      color: #93c5fd;
    }}
    .indicator-toggle input[type="checkbox"] {{ accent-color: #22c55e; }}
    .chart-stack {{
      display: grid;
      grid-template-rows: minmax(360px, 1fr) 140px 140px;
      height: 700px;
      width: 100%;
      background: #0b1220;
      border: 1px solid #1e293b;
      border-radius: 10px;
      overflow: hidden;
    }}
    .chart-pane {{
      position: relative;
      border-bottom: 1px solid #1e293b;
    }}
    .chart-pane:last-child {{ border-bottom: none; }}
    .pane-label {{
      position: absolute;
      top: 8px;
      left: 10px;
      z-index: 5;
      font-size: 12px;
      color: var(--muted);
      background: rgba(15, 23, 42, 0.75);
      padding: 2px 6px;
      border-radius: 4px;
      pointer-events: none;
    }}
    .chart-host {{
      position: absolute;
      inset: 0;
    }}
    .tag {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; background: #1e293b; color: var(--text); }}
    .tag.good {{ background: rgba(34,197,94,0.18); color: #86efac; }}
    .tag.bad {{ background: rgba(239,68,68,0.18); color: #fca5a5; }}
    .tag.flat {{ background: rgba(148,163,184,0.18); color: #cbd5e1; }}
    @media (max-width: 1100px) {{
      .span-8, .span-6, .span-4 {{ grid-column: span 12; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Paper Trading Dashboard</h1>
    <div class=\"subtle\" id=\"header-meta\">Loading live paper-trading state…</div>
  </header>
  <main class=\"layout\">
    <section class=\"card span-12\">
      <h2>Price / Execution Chart</h2>
      <div class=\"chart-toolbar\">
        <div class=\"indicators\">
          <label class=\"indicator-toggle\"><input id=\"showAlligator\" type=\"checkbox\" checked /> Alligator</label>
          <label class=\"indicator-toggle\"><input id=\"showAO\" type=\"checkbox\" checked /> AO pane</label>
          <label class=\"indicator-toggle\"><input id=\"showAC\" type=\"checkbox\" checked /> AC pane</label>
          <label class=\"indicator-toggle\"><input id=\"showZones\" type=\"checkbox\" checked /> Zones</label>
          <label class=\"indicator-toggle\"><input id=\"showWisemanSignals\" type=\"checkbox\" checked /> 1W / 1W-R</label>
          <label class=\"indicator-toggle\"><input id=\"showSecondWiseman\" type=\"checkbox\" checked /> 2W</label>
          <label class=\"indicator-toggle\"><input id=\"showThirdWiseman\" type=\"checkbox\" checked /> 3W</label>
        </div>
        <div class=\"subtle\" id=\"price-chart-status\">Loading chart…</div>
      </div>
      <div class=\"chart-stack\" id=\"price-chart-stack\">
        <div class=\"chart-pane\" id=\"price-pane\">
          <div class=\"pane-label\">Price</div>
          <div class=\"chart-host\" id=\"price-chart\"></div>
        </div>
        <div class=\"chart-pane\" id=\"ao-pane\">
          <div class=\"pane-label\">AO</div>
          <div class=\"chart-host\" id=\"ao-chart\"></div>
        </div>
        <div class=\"chart-pane\" id=\"ac-pane\">
          <div class=\"pane-label\">AC</div>
          <div class=\"chart-host\" id=\"ac-chart\"></div>
        </div>
      </div>
    </section>
    <section class=\"card span-12\">
      <div class=\"stats-grid\" id=\"summary-cards\"></div>
    </section>
    <section class=\"card span-8\">
      <h2>Equity Curve</h2>
      <svg class=\"chart\" id=\"equity-chart\" viewBox=\"0 0 900 260\" preserveAspectRatio=\"none\"></svg>
    </section>
    <section class=\"card span-4\">
      <h2>Current Position</h2>
      <div id=\"current-position\"></div>
    </section>
    <section class=\"card span-8\">
      <h2>Drawdown</h2>
      <svg class=\"chart\" id=\"drawdown-chart\" viewBox=\"0 0 900 260\" preserveAspectRatio=\"none\"></svg>
    </section>
    <section class=\"card span-4\">
      <h2>Open Orders</h2>
      <div id=\"open-orders\"></div>
    </section>
    <section class=\"card span-6\">
      <h2>Completed Trades</h2>
      <div id=\"completed-trades\"></div>
    </section>
    <section class=\"card span-6\">
      <h2>Recent Fills</h2>
      <div id=\"recent-fills\"></div>
    </section>
    <section class=\"card span-12\">
      <h2>Position / Equity Timeline</h2>
      <div id=\"timeline-table\"></div>
    </section>
  </main>
  <script src=\"https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js\"></script>
  <script>
    const DATA_SCRIPT_URL = {safe_data_script_filename};

    const fmtNumber = (value, digits = 2) => Number(value ?? 0).toLocaleString(undefined, {{ minimumFractionDigits: digits, maximumFractionDigits: digits }});
    const fmtPct = (value, digits = 2) => `${{fmtNumber((Number(value ?? 0) * 100), digits)}}%`;
    const escapeHtml = (value) => String(value ?? '')
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#39;');

    const renderTable = (containerId, rows, columns, emptyMessage) => {{
      const container = document.getElementById(containerId);
      if (!rows || rows.length === 0) {{
        container.innerHTML = `<div class=\"subtle\">${{escapeHtml(emptyMessage)}}</div>`;
        return;
      }}
      const head = columns.map((col) => `<th>${{escapeHtml(col.label)}}</th>`).join('');
      const body = rows.map((row) => `<tr>${{columns.map((col) => `<td>${{escapeHtml(col.render ? col.render(row[col.key], row) : row[col.key])}}</td>`).join('')}}</tr>`).join('');
      container.innerHTML = `<table><thead><tr>${{head}}</tr></thead><tbody>${{body}}</tbody></table>`;
    }};

    const renderLineChart = (svgId, values, color, formatter) => {{
      const svg = document.getElementById(svgId);
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const width = 900;
      const height = 260;
      const padding = 20;
      const numeric = (values || []).map((point) => Number(point.value));
      if (numeric.length === 0) {{
        const empty = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        empty.setAttribute('x', '24');
        empty.setAttribute('y', '42');
        empty.setAttribute('fill', '#94a3b8');
        empty.textContent = 'No data yet';
        svg.appendChild(empty);
        return;
      }}
      let min = Math.min(...numeric);
      let max = Math.max(...numeric);
      if (Math.abs(max - min) < 1e-9) {{
        max += 1;
        min -= 1;
      }}
      const makeY = (value) => height - padding - ((value - min) / (max - min)) * (height - padding * 2);
      const makeX = (index) => padding + (index / Math.max(1, numeric.length - 1)) * (width - padding * 2);

      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', color);
      path.setAttribute('stroke-width', '3');
      path.setAttribute('stroke-linejoin', 'round');
      path.setAttribute('stroke-linecap', 'round');
      path.setAttribute('d', values.map((point, index) => `${{index === 0 ? 'M' : 'L'}}${{makeX(index)}},${{makeY(Number(point.value))}}`).join(' '));
      svg.appendChild(path);

      const latest = values[values.length - 1];
      const latestY = makeY(Number(latest.value));
      const latestX = makeX(values.length - 1);
      const marker = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      marker.setAttribute('cx', String(latestX));
      marker.setAttribute('cy', String(latestY));
      marker.setAttribute('r', '4');
      marker.setAttribute('fill', color);
      svg.appendChild(marker);

      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', '24');
      label.setAttribute('y', '28');
      label.setAttribute('fill', '#e5e7eb');
      label.textContent = `Latest: ${{formatter(Number(latest.value))}}`;
      svg.appendChild(label);

      const minLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      minLabel.setAttribute('x', '24');
      minLabel.setAttribute('y', String(height - 12));
      minLabel.setAttribute('fill', '#94a3b8');
      minLabel.textContent = `Min: ${{formatter(min)}}`;
      svg.appendChild(minLabel);

      const maxLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      maxLabel.setAttribute('x', '740');
      maxLabel.setAttribute('y', '28');
      maxLabel.setAttribute('fill', '#94a3b8');
      maxLabel.textContent = `Max: ${{formatter(max)}}`;
      svg.appendChild(maxLabel);
    }};

    const showAlligatorEl = document.getElementById('showAlligator');
    const showAOEl = document.getElementById('showAO');
    const showACEl = document.getElementById('showAC');
    const showZonesEl = document.getElementById('showZones');
    const showWisemanSignalsEl = document.getElementById('showWisemanSignals');
    const showSecondWisemanEl = document.getElementById('showSecondWiseman');
    const showThirdWisemanEl = document.getElementById('showThirdWiseman');
    const priceChartStatusEl = document.getElementById('price-chart-status');
    const pricePaneEl = document.getElementById('price-pane');
    const aoPaneEl = document.getElementById('ao-pane');
    const acPaneEl = document.getElementById('ac-pane');
    const priceChartHost = document.getElementById('price-chart');
    const aoChartHost = document.getElementById('ao-chart');
    const acChartHost = document.getElementById('ac-chart');
    const baseChartOptions = {{
      layout: {{ background: {{ color: '#0f172a' }}, textColor: '#cbd5e1' }},
      grid: {{ vertLines: {{ color: '#1e293b' }}, horzLines: {{ color: '#1e293b' }} }},
      rightPriceScale: {{ borderColor: '#334155', minimumWidth: 72 }},
      timeScale: {{ borderColor: '#334155', timeVisible: true, secondsVisible: false }},
      crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
      handleScroll: {{ mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false }},
      handleScale: {{ mouseWheel: true, pinch: true, axisPressedMouseMove: false }},
    }};
    const priceChart = LightweightCharts.createChart(priceChartHost, baseChartOptions);
    const aoChart = LightweightCharts.createChart(aoChartHost, {{
      ...baseChartOptions,
      rightPriceScale: {{ borderColor: '#334155', minimumWidth: 72, scaleMargins: {{ top: 0.15, bottom: 0.15 }} }},
      leftPriceScale: {{ visible: false }},
      timeScale: {{ visible: false, borderColor: '#334155', timeVisible: true, secondsVisible: false }},
    }});
    const acChart = LightweightCharts.createChart(acChartHost, {{
      ...baseChartOptions,
      rightPriceScale: {{ borderColor: '#334155', minimumWidth: 72, scaleMargins: {{ top: 0.15, bottom: 0.15 }} }},
      leftPriceScale: {{ visible: false }},
    }});
    const priceCandleSeries = priceChart.addCandlestickSeries({{
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    }});
    const jawSeries = priceChart.addLineSeries({{ color: '#2563eb', lineWidth: 2, priceLineVisible: false, lastValueVisible: false }});
    const teethSeries = priceChart.addLineSeries({{ color: '#ef4444', lineWidth: 2, priceLineVisible: false, lastValueVisible: false }});
    const lipsSeries = priceChart.addLineSeries({{ color: '#22c55e', lineWidth: 2, priceLineVisible: false, lastValueVisible: false }});
    const aoSeries = aoChart.addHistogramSeries({{ base: 0, priceLineVisible: false, lastValueVisible: true }});
    const acSeries = acChart.addHistogramSeries({{ base: 0, priceLineVisible: false, lastValueVisible: true }});
    let paperTradeEventLineSeries = [];
    let paperTradePathLineSeries = [];
    let sharedLogicalRange = null;
    let syncGuard = false;
    let chartRangeInitialized = false;

    const cloneLogicalRange = (range) => range ? {{ from: Number(range.from), to: Number(range.to) }} : null;
    const logicalRangesEqual = (left, right) => Boolean(left && right)
      && Math.abs(Number(left.from) - Number(right.from)) < 1e-4
      && Math.abs(Number(left.to) - Number(right.to)) < 1e-4;
    const readVisibleRange = (chart) => {{
      const timeScale = chart.timeScale();
      return typeof timeScale.getVisibleLogicalRange === 'function'
        ? cloneLogicalRange(timeScale.getVisibleLogicalRange())
        : null;
    }};
    const applyVisibleRange = (chart, range) => {{
      if (!range) return;
      const timeScale = chart.timeScale();
      if (typeof timeScale.setVisibleLogicalRange !== 'function') return;
      const currentRange = readVisibleRange(chart);
      if (logicalRangesEqual(currentRange, range)) return;
      timeScale.setVisibleLogicalRange(range);
    }};
    const restoreRange = (range) => {{
      if (!range) return;
      sharedLogicalRange = cloneLogicalRange(range);
      syncGuard = true;
      applyVisibleRange(priceChart, sharedLogicalRange);
      applyVisibleRange(aoChart, sharedLogicalRange);
      applyVisibleRange(acChart, sharedLogicalRange);
      syncGuard = false;
    }};
    const syncVisibleRange = (sourceChart, targetCharts) => {{
      const timeScale = sourceChart.timeScale();
      if (typeof timeScale.subscribeVisibleLogicalRangeChange !== 'function') return;
      timeScale.subscribeVisibleLogicalRangeChange((range) => {{
        const normalizedRange = cloneLogicalRange(range);
        if (syncGuard || !normalizedRange || logicalRangesEqual(sharedLogicalRange, normalizedRange)) return;
        sharedLogicalRange = normalizedRange;
        syncGuard = true;
        targetCharts.forEach((chart) => applyVisibleRange(chart, normalizedRange));
        syncGuard = false;
      }});
    }};
    syncVisibleRange(priceChart, [aoChart, acChart]);

    const renderPaperTradeEventLines = (eventLines) => {{
      paperTradeEventLineSeries.forEach((series) => priceChart.removeSeries(series));
      paperTradeEventLineSeries = [];
      (eventLines || []).forEach((eventLine) => {{
        const series = priceChart.addLineSeries({{
          color: eventLine.color || '#94a3b8',
          lineWidth: 2,
          lineStyle: LightweightCharts.LineStyle.Dashed,
          lastValueVisible: false,
          priceLineVisible: false,
          crosshairMarkerVisible: false,
          title: eventLine.label || 'Execution',
        }});
        series.setData(eventLine.points || []);
        paperTradeEventLineSeries.push(series);
      }});
    }};

    const renderPaperTradePathLines = (tradePaths) => {{
      paperTradePathLineSeries.forEach((series) => priceChart.removeSeries(series));
      paperTradePathLineSeries = [];
      (tradePaths || []).forEach((tradePath) => {{
        const series = priceChart.addLineSeries({{
          color: tradePath.color || '#94a3b8',
          lineWidth: 2,
          lastValueVisible: false,
          priceLineVisible: false,
          crosshairMarkerVisible: false,
          title: tradePath.label || 'Trade Path',
        }});
        series.setData(tradePath.points || []);
        paperTradePathLineSeries.push(series);
      }});
    }};

    const filterMarkers = (markers) => (Array.isArray(markers) ? markers : []).filter((marker) => {{
      const text = String(marker?.text || '');
      const group = String(marker?.markerGroup || '');
      const hasThirdWisemanLabel = text.includes('3W');
      if (!showWisemanSignalsEl.checked && (text === '1W' || text === '1W-R' || text.startsWith('1W-'))) return false;
      if (!showSecondWisemanEl.checked && (group === 'second_wiseman' || group === 'second_wiseman_entry' || text === '2W')) return false;
      if (!showThirdWisemanEl.checked && (group === 'third_wiseman' || group === 'third_wiseman_entry' || hasThirdWisemanLabel)) return false;
      return true;
    }});
    const collapseMarkers = (markers) => {{
      const buckets = new Map();
      (Array.isArray(markers) ? markers : []).forEach((marker) => {{
        const key = `${{marker.time}}::${{marker.position}}`;
        const existing = buckets.get(key);
        if (!existing) {{
          buckets.set(key, {{ ...marker, textParts: [String(marker.text || '')] }});
          return;
        }}
        const nextText = String(marker.text || '');
        if (nextText && !existing.textParts.includes(nextText)) existing.textParts.push(nextText);
      }});
      return Array.from(buckets.values())
        .map((marker) => {{
          const {{ textParts, ...rest }} = marker;
          return {{ ...rest, text: textParts.filter(Boolean).join('/') }};
        }})
        .sort((left, right) => Number(left.time) - Number(right.time));
    }};
    const applyPaneVisibility = () => {{
      aoPaneEl.style.display = showAOEl.checked ? 'block' : 'none';
      acPaneEl.style.display = showACEl.checked ? 'block' : 'none';
      resizePriceCharts();
    }};
    const renderPriceChart = (payload) => {{
      const priceChartPayload = payload.price_chart || {{}};
      const preservedRange = chartRangeInitialized ? readVisibleRange(priceChart) : null;
      const candleRows = Array.isArray(priceChartPayload.candles) ? priceChartPayload.candles : [];
      const plainCandles = candleRows.map((candle) => {{
        const base = {{
          time: candle.time,
          open: candle.open,
          high: candle.high,
          low: candle.low,
          close: candle.close,
        }};
        return showZonesEl.checked ? {{ ...base, color: candle.color, borderColor: candle.borderColor, wickColor: candle.wickColor }} : base;
      }});
      priceCandleSeries.setData(plainCandles);
      jawSeries.setData(showAlligatorEl.checked ? (priceChartPayload.alligator?.jaw || []) : []);
      teethSeries.setData(showAlligatorEl.checked ? (priceChartPayload.alligator?.teeth || []) : []);
      lipsSeries.setData(showAlligatorEl.checked ? (priceChartPayload.alligator?.lips || []) : []);
      aoSeries.setData(showAOEl.checked ? (priceChartPayload.ao || []) : []);
      acSeries.setData(showACEl.checked ? (priceChartPayload.ac || []) : []);
      priceCandleSeries.setMarkers(collapseMarkers(filterMarkers(priceChartPayload.markers || [])));
      renderPaperTradeEventLines(priceChartPayload.trade_event_lines || []);
      renderPaperTradePathLines(priceChartPayload.trade_path_lines || []);
      if (candleRows.length > 0) {{
        if (!chartRangeInitialized) {{
          priceChart.timeScale().fitContent();
          restoreRange(readVisibleRange(priceChart));
          chartRangeInitialized = true;
        }} else {{
          restoreRange(preservedRange);
        }}
        priceChartStatusEl.textContent = `${{priceChartPayload.symbol || payload.summary?.symbol || ''}} ${{priceChartPayload.interval || payload.summary?.interval || ''}} • ${{candleRows.length}} candles • indicators live`;
      }} else {{
        priceChartStatusEl.textContent = 'No chart data yet';
      }}
    }};

    const renderSummaryCards = (payload) => {{
      const stats = payload.stats || {{}};
      const summary = payload.summary || {{}};
      const cards = [
        {{ label: 'Equity', value: fmtNumber(summary.equity), klass: (summary.equity ?? 0) >= (summary.initial_cash ?? 0) ? 'good' : 'bad' }},
        {{ label: 'Net PnL', value: fmtNumber((summary.equity ?? 0) - (summary.initial_cash ?? 0)), klass: ((summary.equity ?? 0) - (summary.initial_cash ?? 0)) >= 0 ? 'good' : 'bad' }},
        {{ label: 'Max Drawdown', value: fmtPct(stats.max_drawdown ?? 0), klass: 'bad' }},
        {{ label: 'Win Rate', value: fmtPct(stats.win_rate ?? 0), klass: (stats.win_rate ?? 0) >= 0.5 ? 'good' : '' }},
        {{ label: 'Sharpe', value: fmtNumber(stats.sharpe ?? 0), klass: (stats.sharpe ?? 0) >= 0 ? 'good' : 'bad' }},
        {{ label: 'Trades', value: fmtNumber(stats.total_trades ?? 0, 0), klass: '' }},
        {{ label: 'Open Orders', value: fmtNumber(summary.open_order_count ?? 0, 0), klass: '' }},
        {{ label: 'Fees Paid', value: fmtNumber(summary.fees_paid ?? 0), klass: 'bad' }},
      ];
      document.getElementById('summary-cards').innerHTML = cards.map((card) => `
        <div class=\"stat\">
          <div class=\"label\">${{escapeHtml(card.label)}}</div>
          <div class=\"value ${{escapeHtml(card.klass)}}\">${{escapeHtml(card.value)}}</div>
        </div>`).join('');
    }};

    const renderPosition = (payload) => {{
      const position = payload.current_position || {{}};
      const quantity = Number(position.quantity ?? 0);
      const hasOpenPosition = Math.abs(quantity) > 1e-12;
      const quantityText = hasOpenPosition
        ? `${{fmtNumber(quantity, 6)}} ($${{fmtNumber(position.notional_value ?? 0)}})`
        : fmtNumber(0, 6);
      const pnlText = hasOpenPosition
        ? `${{fmtNumber(payload.summary?.unrealized_pnl ?? 0)}} (${{fmtPct(position.unrealized_return_pct ?? 0)}})`
        : fmtNumber(payload.summary?.unrealized_pnl ?? 0);
      document.getElementById('current-position').innerHTML = `
        <p>Quantity: <strong>${{escapeHtml(quantityText)}}</strong></p>
        <p>Average price: <strong>${{escapeHtml(fmtNumber(position.average_price ?? 0))}}</strong></p>
        <p>Mark price: <strong>${{escapeHtml(fmtNumber(payload.summary?.mark_price ?? 0))}}</strong></p>
        <p>Unrealized PnL: <strong>${{escapeHtml(pnlText)}}</strong></p>
        <p>Opened at: <strong>${{escapeHtml(position.opened_at ?? 'n/a')}}</strong></p>
        <p>Updated at: <strong>${{escapeHtml(position.last_updated_at ?? 'n/a')}}</strong></p>`;
    }};

    const renderOpenOrders = (payload) => renderTable(
      'open-orders',
      payload.open_orders || [],
      [
        {{ label: 'ID', key: 'order_id' }},
        {{ label: 'Side', key: 'side' }},
        {{ label: 'Type', key: 'order_type' }},
        {{ label: 'Qty', key: 'quantity', render: (value) => fmtNumber(value, 6) }},
        {{ label: 'Limit', key: 'limit_price', render: (value) => value == null ? '—' : fmtNumber(value) }},
        {{ label: 'Stop', key: 'stop_price', render: (value) => value == null ? '—' : fmtNumber(value) }},
      ],
      'No working orders.'
    );

    const renderTrades = (payload) => renderTable(
      'completed-trades',
      (payload.completed_trades || []).slice(-12).reverse(),
      [
        {{ label: 'Entry', key: 'entry_time' }},
        {{ label: 'Exit', key: 'exit_time' }},
        {{ label: 'Side', key: 'side' }},
        {{ label: 'Qty', key: 'quantity', render: (value) => fmtNumber(value, 6) }},
        {{ label: 'Entry Reason', key: 'entry_reason' }},
        {{ label: 'Exit Reason', key: 'exit_reason' }},
        {{ label: 'PnL', key: 'pnl', render: (value) => fmtNumber(value) }},
        {{ label: 'Return', key: 'return_pct', render: (value) => fmtPct(value) }},
      ],
      'No completed trades yet.'
    );

    const renderFills = (payload) => renderTable(
      'recent-fills',
      (payload.fills || []).slice(-12).reverse(),
      [
        {{ label: 'Time', key: 'timestamp' }},
        {{ label: 'Side', key: 'side' }},
        {{ label: 'Qty', key: 'quantity', render: (value) => fmtNumber(value, 6) }},
        {{ label: 'Price', key: 'price', render: (value) => fmtNumber(value) }},
        {{ label: 'PnL', key: 'realized_pnl', render: (value) => fmtNumber(value) }},
        {{ label: 'Reason', key: 'reason' }},
        {{ label: 'Execution', key: 'execution_reason' }},
      ],
      'No fills yet.'
    );

    const renderTimeline = (payload) => renderTable(
      'timeline-table',
      (payload.equity_curve || []).slice(-25).reverse(),
      [
        {{ label: 'Bar Close', key: 'timestamp' }},
        {{ label: 'Equity', key: 'equity', render: (value) => fmtNumber(value) }},
        {{ label: 'Cash', key: 'cash', render: (value) => fmtNumber(value) }},
        {{ label: 'Realized', key: 'realized_pnl', render: (value) => fmtNumber(value) }},
        {{ label: 'Unrealized', key: 'unrealized_pnl', render: (value) => fmtNumber(value) }},
        {{ label: 'Position Qty', key: 'position_quantity', render: (value) => fmtNumber(value, 6) }},
        {{ label: 'Mark', key: 'mark_price', render: (value) => fmtNumber(value) }},
      ],
      'No timeline records yet.'
    );

    const render = (payload) => {{
      renderSummaryCards(payload);
      renderPosition(payload);
      renderOpenOrders(payload);
      renderTrades(payload);
      renderFills(payload);
      renderTimeline(payload);
      renderPriceChart(payload);
      renderLineChart('equity-chart', (payload.equity_curve || []).map((point) => ({{ value: point.equity }})), '#38bdf8', (value) => fmtNumber(value));
      renderLineChart('drawdown-chart', (payload.drawdown_curve || []).map((point) => ({{ value: point.drawdown }})), '#ef4444', (value) => fmtPct(value));
      const summary = payload.summary || {{}};
      document.getElementById('header-meta').textContent = `${{summary.symbol || ''}} • ${{summary.interval || ''}} • last updated ${{payload.updated_at || 'n/a'}} • last processed bar ${{summary.last_processed_bar_time || 'n/a'}}`;
    }};

    const load = async () => {{
      try {{
        const script = document.createElement('script');
        script.src = `${{DATA_SCRIPT_URL}}?t=${{Date.now()}}`;
        script.async = true;
        await new Promise((resolve, reject) => {{
          script.onload = resolve;
          script.onerror = () => reject(new Error('Failed to load local dashboard data script'));
          document.head.appendChild(script);
        }});
        if (!window.__PAPER_DASHBOARD_DATA__) {{
          throw new Error('Dashboard data script did not define window.__PAPER_DASHBOARD_DATA__');
        }}
        render(window.__PAPER_DASHBOARD_DATA__);
        script.remove();
      }} catch (error) {{
        document.getElementById('header-meta').textContent = `Dashboard refresh failed: ${{error}}`;
      }}
    }};

    const resizePriceCharts = () => {{
      const priceRect = pricePaneEl.getBoundingClientRect();
      const aoRect = aoPaneEl.getBoundingClientRect();
      const acRect = acPaneEl.getBoundingClientRect();
      priceChart.resize(priceRect.width, Math.max(240, priceRect.height));
      aoChart.resize(aoRect.width || priceRect.width, Math.max(60, aoRect.height || 1));
      acChart.resize(acRect.width || priceRect.width, Math.max(60, acRect.height || 1));
    }};

    load();
    [showAlligatorEl, showAOEl, showACEl, showZonesEl, showWisemanSignalsEl, showSecondWisemanEl, showThirdWisemanEl]
      .forEach((el) => el.addEventListener('change', () => {{
        applyPaneVisibility();
        if (window.__PAPER_DASHBOARD_DATA__) render(window.__PAPER_DASHBOARD_DATA__);
      }}));
    applyPaneVisibility();
    window.setInterval(load, 1000);
    window.addEventListener('resize', resizePriceCharts);
    resizePriceCharts();
  </script>
</body>
</html>
"""


def _build_completed_trades(fills: pd.DataFrame, timeline: pd.DataFrame) -> list[_CompletedTrade]:
    if fills.empty:
        return []

    fills_sorted = fills.copy().sort_values("timestamp").reset_index(drop=True)
    timeline_positions: dict[pd.Timestamp, int] = {}
    if not timeline.empty:
        for idx, ts in enumerate(pd.to_datetime(timeline["timestamp"], utc=True)):
            timeline_positions[pd.Timestamp(ts)] = idx

    open_qty = 0.0
    average_entry = 0.0
    entry_time: pd.Timestamp | None = None
    entry_reasons: list[str] = []
    trades: list[_CompletedTrade] = []

    for row in fills_sorted.itertuples(index=False):
        timestamp = pd.Timestamp(row.timestamp)
        signed_qty = float(row.quantity) if row.side == "buy" else -float(row.quantity)
        strategy_reason = str(getattr(row, "reason", "") or "").strip()
        if np.isclose(open_qty, 0.0, atol=1e-12):
            open_qty = signed_qty
            average_entry = float(row.price)
            entry_time = timestamp
            entry_reasons = [strategy_reason] if strategy_reason else []
            continue

        same_direction = np.sign(open_qty) == np.sign(signed_qty)
        if same_direction:
            total_abs = abs(open_qty) + abs(signed_qty)
            average_entry = ((abs(open_qty) * average_entry) + (abs(signed_qty) * float(row.price))) / total_abs if total_abs > 0 else 0.0
            open_qty += signed_qty
            if entry_time is None:
                entry_time = timestamp
            if strategy_reason and strategy_reason not in entry_reasons:
                entry_reasons.append(strategy_reason)
            continue

        closed_qty = min(abs(open_qty), abs(signed_qty))
        if closed_qty > 0 and entry_time is not None:
            entry_idx = timeline_positions.get(entry_time)
            exit_idx = timeline_positions.get(timestamp)
            if entry_idx is not None and exit_idx is not None:
                holding_bars = max(1, exit_idx - entry_idx)
            else:
                holding_bars = 1
            side = "long" if open_qty > 0 else "short"
            return_pct = float(row.realized_pnl) / (average_entry * closed_qty) if average_entry > 0 and closed_qty > 0 else 0.0
            trades.append(
                _CompletedTrade(
                    entry_time=entry_time,
                    exit_time=timestamp,
                    side=side,
                    quantity=closed_qty,
                    entry_price=average_entry,
                    exit_price=float(row.price),
                    pnl=float(row.realized_pnl),
                    return_pct=return_pct,
                    holding_bars=holding_bars,
                    entry_reason=", ".join(entry_reasons) if entry_reasons else "Unspecified",
                    exit_reason=str(strategy_reason or "Unspecified"),
                )
            )

        leftover = abs(signed_qty) - closed_qty
        if leftover > 1e-12:
            open_qty = np.sign(signed_qty) * leftover
            average_entry = float(row.price)
            entry_time = timestamp
            entry_reasons = [strategy_reason] if strategy_reason else []
        else:
            remaining = abs(open_qty) - closed_qty
            if remaining > 1e-12:
                open_qty = np.sign(open_qty) * remaining
            else:
                open_qty = 0.0
                average_entry = 0.0
                entry_time = None
                entry_reasons = []

    return trades


def _build_equity_curve_frame(bar_records: list[dict[str, Any]]) -> pd.DataFrame:
    if not bar_records:
        return pd.DataFrame(columns=["timestamp", "cash", "equity", "realized_pnl", "unrealized_pnl", "position_quantity", "mark_price"])
    frame = pd.DataFrame(bar_records)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return frame


def _paper_execution_events(fills: pd.DataFrame) -> list[ExecutionEvent]:
    if fills.empty:
        return []

    fills_sorted = fills.copy()
    fills_sorted["timestamp"] = pd.to_datetime(fills_sorted["timestamp"], utc=True)
    fills_sorted = fills_sorted.sort_values("timestamp").reset_index(drop=True)

    events: list[ExecutionEvent] = []
    open_qty = 0.0

    for row in fills_sorted.itertuples(index=False):
        timestamp = pd.Timestamp(row.timestamp)
        side = str(row.side)
        signed_qty = float(row.quantity) if side == "buy" else -float(row.quantity)
        next_qty = open_qty + signed_qty
        fill_price = float(row.price)
        strategy_reason = str(getattr(row, "reason", "") or "").strip() or None

        if np.isclose(open_qty, 0.0, atol=1e-12):
            if not np.isclose(next_qty, 0.0, atol=1e-12):
                events.append(
                    ExecutionEvent(
                        event_type="entry",
                        time=timestamp,
                        side=side,
                        price=fill_price,
                        units=abs(signed_qty),
                        strategy_reason=strategy_reason,
                    )
                )
            open_qty = next_qty
            continue

        if np.sign(open_qty) == np.sign(next_qty) and abs(next_qty) > abs(open_qty):
            events.append(
                ExecutionEvent(
                    event_type="add",
                    time=timestamp,
                    side=side,
                    price=fill_price,
                    units=abs(signed_qty),
                    strategy_reason=strategy_reason,
                )
            )
            open_qty = next_qty
            continue

        if np.sign(open_qty) == np.sign(next_qty) and abs(next_qty) < abs(open_qty):
            event_type = "exit" if np.isclose(next_qty, 0.0, atol=1e-12) else "reduce"
            events.append(
                ExecutionEvent(
                    event_type=event_type,
                    time=timestamp,
                    side=side,
                    price=fill_price,
                    units=abs(signed_qty),
                    strategy_reason=strategy_reason,
                )
            )
            open_qty = next_qty
            continue

        if np.sign(open_qty) != np.sign(next_qty) and not np.isclose(next_qty, 0.0, atol=1e-12):
            exit_units = abs(open_qty)
            entry_units = abs(next_qty)
            if exit_units > 1e-12:
                events.append(
                    ExecutionEvent(
                        event_type="exit",
                        time=timestamp,
                        side=side,
                        price=fill_price,
                        units=exit_units,
                        strategy_reason=strategy_reason,
                    )
                )
            if entry_units > 1e-12:
                events.append(
                    ExecutionEvent(
                        event_type="entry",
                        time=timestamp,
                        side=side,
                        price=fill_price,
                        units=entry_units,
                        strategy_reason=strategy_reason,
                    )
                )
            open_qty = next_qty
            continue

        event_type = "exit" if np.isclose(next_qty, 0.0, atol=1e-12) else "reduce"
        events.append(
            ExecutionEvent(
                event_type=event_type,
                time=timestamp,
                side=side,
                price=fill_price,
                units=abs(signed_qty),
                strategy_reason=strategy_reason,
            )
        )
        open_qty = next_qty

    return events


def _compute_live_stats(summary: dict[str, Any], equity_curve: pd.DataFrame, completed_trades: list[_CompletedTrade]) -> dict[str, float]:
    if equity_curve.empty:
        return {
            "periods_per_year": 0.0,
            "total_return": 0.0,
            "cagr": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "total_trades": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_trade_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "avg_holding_bars": 0.0,
            "exposure": 0.0,
            "peak_equity": float(summary.get("initial_cash", 0.0)),
            "min_equity": float(summary.get("initial_cash", 0.0)),
            "net_pnl": 0.0,
        }

    indexed = equity_curve.set_index("timestamp")
    equity_series = indexed["equity"].astype(float)
    returns = equity_series.pct_change().fillna(0.0)
    periods_per_year = infer_periods_per_year(equity_series.index, default=252)
    trade_rows = [_PaperTradeMetricsRow(pnl=trade.pnl, holding_bars=trade.holding_bars) for trade in completed_trades]
    stats = compute_performance_stats(equity_series, returns, trade_rows, periods_per_year=periods_per_year)
    stats["peak_equity"] = float(equity_series.max())
    stats["min_equity"] = float(equity_series.min())
    stats["net_pnl"] = float(equity_series.iloc[-1] - equity_series.iloc[0])
    return stats


def _serialize_fills_dataframe(fills: pd.DataFrame) -> list[dict[str, Any]]:
    if fills.empty:
        return []
    output: list[dict[str, Any]] = []
    for row in fills.sort_values("timestamp").itertuples(index=False):
        output.append(
            {
                "timestamp": pd.Timestamp(row.timestamp).isoformat(),
                "symbol": row.symbol,
                "side": row.side,
                "quantity": float(row.quantity),
                "price": float(row.price),
                "fee": float(row.fee),
                "reason": row.reason,
                "execution_reason": getattr(row, "execution_reason", None),
                "signal_action": getattr(row, "signal_action", None),
                "realized_pnl": float(row.realized_pnl),
            }
        )
    return output


def _write_status_markdown(summary: dict[str, Any], stats: dict[str, float], payload: dict[str, Any], output_path: Path) -> None:
    current_position = payload["current_position"]
    open_orders = payload["open_orders"]
    status_text = "\n".join(
        [
            f"# Paper Trading Status — {summary['symbol']} {summary['interval']}",
            "",
            f"_Updated: {payload['updated_at']}_",
            "",
            "## Account Snapshot",
            "",
            f"- Strategy: `{summary['strategy']}`",
            f"- Started at: `{summary['started_at']}`",
            f"- Last processed bar: `{summary['last_processed_bar_time']}`",
            f"- Cash: `{_format_number(summary['cash'])}`",
            f"- Equity: `{_format_number(summary['equity'])}`",
            f"- Realized PnL: `{_format_number(summary['realized_pnl'])}`",
            f"- Unrealized PnL: `{_format_number(summary['unrealized_pnl'])}`",
            f"- Fees paid: `{_format_number(summary['fees_paid'])}`",
            f"- Mark price: `{_format_number(summary['mark_price'])}`",
            "",
            "## Current Position",
            "",
            f"- Quantity: `{_format_number(current_position['quantity'], 6)} (${_format_number(current_position['notional_value'])})`",
            f"- Average price: `{_format_number(current_position['average_price'])}`",
            f"- Mark price: `{_format_number(summary['mark_price'])}`",
            f"- Unrealized PnL: `{_format_number(summary['unrealized_pnl'])} ({_format_pct(current_position['unrealized_return_pct'])})`",
            f"- Opened at: `{current_position['opened_at']}`",
            f"- Last updated at: `{current_position['last_updated_at']}`",
            f"- Stop loss: `{_format_number(current_position['stop_loss_price']) if current_position['stop_loss_price'] is not None else 'n/a'}`",
            f"- Take profit: `{_format_number(current_position['take_profit_price']) if current_position['take_profit_price'] is not None else 'n/a'}`",
            "",
            "## Live Stats",
            "",
            f"- Net PnL: `{_format_number(stats['net_pnl'])}`",
            f"- Total return: `{_format_pct(stats['total_return'])}`",
            f"- CAGR: `{_format_pct(stats['cagr'])}`",
            f"- Max drawdown: `{_format_pct(stats['max_drawdown'])}`",
            f"- Sharpe: `{_format_number(stats['sharpe'])}`",
            f"- Sortino: `{_format_number(stats['sortino'])}`",
            f"- Win rate: `{_format_pct(stats['win_rate'])}`",
            f"- Profit factor: `{_format_number(stats['profit_factor'])}`",
            f"- Expectancy: `{_format_number(stats['expectancy'])}`",
            f"- Total trades: `{_format_number(stats['total_trades'], 0)}`",
            f"- Peak equity: `{_format_number(stats['peak_equity'])}`",
            f"- Min equity: `{_format_number(stats['min_equity'])}`",
            "",
            "## Open Orders",
            "",
            _markdown_table(
                open_orders,
                [
                    ("Order ID", "order_id"),
                    ("Side", "side"),
                    ("Type", "order_type"),
                    ("Quantity", "quantity"),
                    ("Limit", "limit_price"),
                    ("Stop", "stop_price"),
                    ("Created", "created_at"),
                ],
                "No working orders.",
            ),
            "",
            "## Recent Timeline",
            "",
            _markdown_table(
                list(reversed(payload["equity_curve"][-15:])),
                [
                    ("Bar", "timestamp"),
                    ("Equity", "equity"),
                    ("Cash", "cash"),
                    ("Realized", "realized_pnl"),
                    ("Unrealized", "unrealized_pnl"),
                    ("Position Qty", "position_quantity"),
                ],
                "No timeline rows yet.",
            ),
            "",
        ]
    )
    _best_effort_atomic_write_text(output_path, status_text)


def _write_trades_markdown(payload: dict[str, Any], output_path: Path) -> None:
    trades_text = "\n".join(
        [
            f"# Paper Trades — {payload['summary']['symbol']} {payload['summary']['interval']}",
            "",
            f"_Updated: {payload['updated_at']}_",
            "",
            "## Completed Trades",
            "",
            _markdown_table(
                payload["completed_trades"],
                [
                    ("Entry", "entry_time"),
                    ("Exit", "exit_time"),
                    ("Side", "side"),
                    ("Qty", "quantity"),
                    ("Entry Price", "entry_price"),
                    ("Exit Price", "exit_price"),
                    ("Entry Reason", "entry_reason"),
                    ("Exit Reason", "exit_reason"),
                    ("PnL", "pnl"),
                    ("Return", "return_pct"),
                ],
                "No completed trades yet.",
            ),
            "",
            "## Fill Ledger",
            "",
            _markdown_table(
                payload["fills"],
                [
                    ("Time", "timestamp"),
                    ("Side", "side"),
                    ("Qty", "quantity"),
                    ("Price", "price"),
                    ("Fee", "fee"),
                    ("Realized PnL", "realized_pnl"),
                    ("Reason", "reason"),
                    ("Execution", "execution_reason"),
                ],
                "No fills yet.",
            ),
            "",
        ]
    )
    _best_effort_atomic_write_text(output_path, trades_text)


class RealTimePaperTradingSession:
    def __init__(
        self,
        *,
        symbol: str,
        interval: str,
        strategy: _SignalStrategy,
        config: BacktestConfig,
        warmup_bars: int = 400,
    ) -> None:
        if interval not in _INTERVAL_SECONDS:
            raise ValueError(f"Unsupported interval: {interval}")
        self.symbol = symbol
        self.interval = interval
        self.interval_seconds = _INTERVAL_SECONDS[interval]
        self.strategy = strategy
        self.config = config
        self.warmup_bars = max(50, int(warmup_bars))
        self.engine = PaperTradingEngine(
            symbol=symbol,
            initial_cash=config.initial_capital,
            fee_rate=config.fee_rate,
            slippage_rate=config.slippage_rate,
            spread_rate=config.spread_rate,
            max_loss=config.max_loss,
            max_loss_pct_of_equity=config.max_loss_pct_of_equity,
        )
        self.sizing_engine = BacktestEngine(config)
        self.history = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        self.latest_market_data = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        self.last_processed_bar_time: pd.Timestamp | None = None
        self.last_live_bar_signature: tuple[str, float, float, float, float] | None = None
        self.started_at: pd.Timestamp | None = None
        self.bar_records: list[dict[str, Any]] = []
        self._last_strategy_signals: pd.Series | None = None
        self._last_strategy_contracts: pd.Series | None = None
        self._last_committed_strategy_signals: pd.Series | None = None
        self._last_committed_strategy_contracts: pd.Series | None = None
        self._startup_ignore_signal_bar_before_or_equal: pd.Timestamp | None = None
        self._startup_baseline_side: int = 0
        self._startup_baseline_contracts: float = 0.0
        self._startup_gate_active: bool = True
        self._lock = threading.RLock()
        self._history_revision = 0
        self._latest_market_signature: tuple[int, str, float, float, float, float] | None = None
        self._cached_price_chart_overlay_key: tuple[int, int] | None = None
        self._cached_price_chart_overlay_payload: dict[str, Any] | None = None
        self._executed_intrabar_event_keys: set[tuple[str, int, str]] = set()
        self._active_signal_base_units: float | None = None

    def _append_bar_record(self, timestamp: pd.Timestamp, mark_price: float | None) -> None:
        snapshot = self.engine.snapshot()
        self.bar_records.append(
            {
                "timestamp": timestamp.isoformat(),
                "cash": float(snapshot.cash),
                "equity": float(snapshot.equity),
                "realized_pnl": float(snapshot.realized_pnl),
                "unrealized_pnl": float(snapshot.unrealized_pnl if mark_price is None else self.engine.unrealized_pnl(mark_price)),
                "position_quantity": float(snapshot.position.quantity),
                "mark_price": None if mark_price is None else float(mark_price),
            }
        )
        deduped = {
            row["timestamp"]: row
            for row in sorted(self.bar_records, key=lambda item: item["timestamp"])
        }
        self.bar_records = list(deduped.values())

    def _resolved_signal_contracts(
        self,
        signals: pd.Series,
        signal_contracts: pd.Series | None,
    ) -> pd.Series:
        return pd.Series(
            [_resolve_contracts(float(signals.iloc[i]), signal_contracts, i) for i in range(len(signals))],
            index=signals.index,
            dtype="float64",
        )

    def _remember_strategy_snapshot(
        self,
        signals: pd.Series,
        resolved_contracts: pd.Series,
        *,
        committed: bool,
    ) -> None:
        self._last_strategy_signals = signals.astype("float64").copy()
        self._last_strategy_contracts = resolved_contracts.astype("float64").copy()
        if committed:
            self._last_committed_strategy_signals = self._last_strategy_signals.copy()
            self._last_committed_strategy_contracts = self._last_strategy_contracts.copy()

    def _latest_changed_signal_index(
        self,
        *,
        signals: pd.Series,
        resolved_contracts: pd.Series,
        exit_reasons: pd.Series | None,
        max_signal_index: int,
        committed_only: bool,
    ) -> int | None:
        if committed_only:
            previous_signals = self._last_committed_strategy_signals
            previous_contracts = self._last_committed_strategy_contracts
        else:
            previous_signals = self._last_strategy_signals
            previous_contracts = self._last_strategy_contracts
        if previous_signals is None or previous_contracts is None:
            return None

        aligned_previous_signals = previous_signals.reindex(signals.index).fillna(0.0)
        aligned_previous_contracts = previous_contracts.reindex(signals.index).fillna(0.0)
        previous_index = set(previous_signals.index)
        latest_changed_index: int | None = None

        for i in range(max_signal_index, -1, -1):
            current_side = int(np.sign(float(signals.iloc[i])))
            previous_side = int(np.sign(float(aligned_previous_signals.iloc[i])))
            current_contracts = float(resolved_contracts.iloc[i]) if current_side != 0 else 0.0
            previous_contracts_value = float(aligned_previous_contracts.iloc[i]) if previous_side != 0 else 0.0
            ts = signals.index[i]
            changed = False
            if ts not in previous_index:
                prior_previous_side = int(np.sign(float(aligned_previous_signals.iloc[i - 1]))) if i > 0 else 0
                prior_previous_contracts = (
                    float(aligned_previous_contracts.iloc[i - 1]) if i > 0 and prior_previous_side != 0 else 0.0
                )
                changed = current_side != prior_previous_side or abs(current_contracts - prior_previous_contracts) > 1e-12
            else:
                changed = current_side != previous_side or abs(current_contracts - previous_contracts_value) > 1e-12

            if not changed:
                continue

            if latest_changed_index is None:
                latest_changed_index = i

            if isinstance(exit_reasons, pd.Series) and i < len(exit_reasons):
                reason_raw = exit_reasons.iloc[i]
                if pd.notna(reason_raw) and str(reason_raw).strip():
                    return i

        return latest_changed_index

    def _evaluate_strategy_history(
        self,
        history: pd.DataFrame,
    ) -> tuple[
        pd.Series,
        pd.Series | None,
        pd.Series | None,
        pd.Series,
        dict[int, list[dict[str, float | int | str]]],
    ]:
        raw_signals = self.strategy.generate_signals(history)
        signals = raw_signals.reindex(history.index).fillna(0)
        signal_fill_prices = getattr(self.strategy, "signal_fill_prices", None)
        signal_contracts = getattr(self.strategy, "signal_contracts", None)
        signal_intrabar_events = getattr(self.strategy, "signal_intrabar_events", None)
        if isinstance(signal_fill_prices, pd.Series):
            signal_fill_prices = signal_fill_prices.reindex(history.index)
        else:
            signal_fill_prices = None
        if isinstance(signal_contracts, pd.Series):
            signal_contracts = signal_contracts.reindex(history.index).fillna(0.0)
        else:
            signal_contracts = None
        if not isinstance(signal_intrabar_events, dict):
            signal_intrabar_events = {}
        resolved_contracts = self._resolved_signal_contracts(signals, signal_contracts)
        return signals, signal_fill_prices, signal_contracts, resolved_contracts, signal_intrabar_events

    def _execute_intrabar_events(
        self,
        *,
        bar_time: pd.Timestamp,
        bar_index: int,
        events: list[dict[str, float | int | str]],
    ) -> int:
        executed = 0
        for event_position, raw_event in enumerate(events):
            event_type = str(raw_event.get("event_type", "")).strip().lower()
            event_side = int(raw_event.get("side", 0) or 0)
            if event_type not in {"entry", "exit"}:
                continue
            if event_side not in {-1, 0, 1}:
                continue
            event_price = raw_event.get("price")
            event_contracts = raw_event.get("contracts")
            try:
                fill_price = float(event_price)
                contracts_value = float(event_contracts)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(fill_price) or fill_price <= 0:
                continue
            if not np.isfinite(contracts_value) or contracts_value <= 0:
                continue
            reason = str(raw_event.get("reason", "")).strip()
            stop_loss_raw = raw_event.get("stop_loss_price")
            stop_loss_price: float | None = None
            try:
                stop_loss_candidate = float(stop_loss_raw) if stop_loss_raw is not None else np.nan
                if np.isfinite(stop_loss_candidate) and stop_loss_candidate > 0:
                    stop_loss_price = stop_loss_candidate
            except (TypeError, ValueError):
                stop_loss_price = None
            event_fingerprint = f"{event_type}|{event_side}|{fill_price:.10f}|{contracts_value:.10f}|{reason}|{stop_loss_price}"
            event_key = (bar_time.isoformat(), event_position, event_fingerprint)
            if event_key in self._executed_intrabar_event_keys:
                continue
            self._executed_intrabar_event_keys.add(event_key)

            signal: ExecutionSignal | None = None
            if event_type == "exit":
                signal = ExecutionSignal(
                    symbol=self.symbol,
                    timestamp=bar_time,
                    action="exit",
                    order_type="market",
                    quantity=contracts_value,
                    metadata={
                        "market_fill_price": fill_price,
                        "fill_on_close": False,
                        "exit_reason": reason,
                        "intrabar_event": True,
                        "intrabar_bar_index": bar_index,
                    },
                )
            elif event_type == "entry" and event_side != 0:
                signal = ExecutionSignal(
                    symbol=self.symbol,
                    timestamp=bar_time,
                    action="enter",
                    side="buy" if event_side > 0 else "sell",
                    order_type="market",
                    quantity=contracts_value,
                    stop_loss_price=stop_loss_price,
                    metadata={
                        "market_fill_price": fill_price,
                        "fill_on_close": False,
                        "entry_reason": reason,
                        "intrabar_event": True,
                        "intrabar_bar_index": bar_index,
                    },
                )
            if signal is None:
                continue

            self.engine.submit_signal(signal)
            self.engine.on_bar(
                LiveBar(
                    symbol=self.symbol,
                    timestamp=bar_time,
                    open=fill_price,
                    high=fill_price,
                    low=fill_price,
                    close=fill_price,
                    volume=0.0,
                )
            )
            executed += 1
        return executed

    def prime(self, candles: pd.DataFrame, now: pd.Timestamp) -> None:
        with self._lock:
            closed = _closed_bars(candles, self.interval_seconds, now)
            if closed.empty:
                self.history = closed.copy()
                self.last_processed_bar_time = None
                self._executed_intrabar_event_keys.clear()
                self._active_signal_base_units = None
                self._history_revision += 1
                self._cached_price_chart_overlay_key = None
                self._cached_price_chart_overlay_payload = None
                return
            self.history = closed.tail(self.warmup_bars).copy()
            self.last_processed_bar_time = self.history.index[-1]
            self.started_at = now
            self._executed_intrabar_event_keys.clear()
            self._startup_ignore_signal_bar_before_or_equal = self.last_processed_bar_time
            self._history_revision += 1
            self._cached_price_chart_overlay_key = None
            self._cached_price_chart_overlay_payload = None
            signals, _, _, resolved_contracts, _ = self._evaluate_strategy_history(self.history)
            self._remember_strategy_snapshot(signals, resolved_contracts, committed=True)
            baseline_signal_index = len(self.history) - 1 if getattr(self.strategy, "execute_on_signal_bar", False) else max(0, len(self.history) - 2)
            baseline_signal_value = float(signals.iloc[baseline_signal_index]) if len(signals) > 0 else 0.0
            self._startup_baseline_side = int(np.sign(baseline_signal_value))
            self._startup_baseline_contracts = (
                float(resolved_contracts.iloc[baseline_signal_index])
                if self._startup_baseline_side != 0 and len(resolved_contracts) > 0
                else 0.0
            )
            self._startup_gate_active = True
            self._active_signal_base_units = None
            self._append_bar_record(self.last_processed_bar_time, float(self.history.iloc[-1]["close"]))

    def update_market_snapshot(self, candles: pd.DataFrame, now: pd.Timestamp) -> None:
        with self._lock:
            if candles.empty:
                self.latest_market_data = candles.copy()
                return

            latest = candles[~candles.index.duplicated(keep="last")].sort_index().tail(self.warmup_bars).copy()
            latest_signature = (
                len(latest),
                latest.index[-1].isoformat(),
                float(latest.iloc[-1]["open"]),
                float(latest.iloc[-1]["high"]),
                float(latest.iloc[-1]["low"]),
                float(latest.iloc[-1]["close"]),
            )
            if latest_signature != self._latest_market_signature:
                self._latest_market_signature = latest_signature
            self.latest_market_data = latest
            latest_timestamp = latest.index[-1]
            latest_close = float(latest.iloc[-1]["close"])
            self.engine.mark_to_market(timestamp=min(latest_timestamp, now), price=latest_close)

    def process_market_data(self, candles: pd.DataFrame, now: pd.Timestamp) -> list[dict[str, Any]]:
        with self._lock:
            closed = _closed_bars(candles, self.interval_seconds, now)
            active_rows = candles[candles.index > closed.index[-1]] if not candles.empty and not closed.empty else candles.iloc[0:0]
            if closed.empty and active_rows.empty:
                return []
            if self.history.empty and not closed.empty:
                self.prime(closed, now)
                if active_rows.empty:
                    return []

            processed: list[dict[str, Any]] = []
            history_changed = False
            if not closed.empty:
                new_rows = closed[closed.index > self.last_processed_bar_time] if self.last_processed_bar_time is not None else closed
                if not new_rows.empty:
                    periods_per_year = infer_periods_per_year(closed.index, default=252)
                    for ts, row in new_rows.iterrows():
                        self.history = pd.concat([self.history, row.to_frame().T])
                        self.history = self.history[~self.history.index.duplicated(keep="last")].sort_index().tail(self.warmup_bars)
                        history = self.history.copy()
                        signal_payload = self._process_single_bar(
                            history,
                            ts,
                            periods_per_year,
                            persist_bar_record=True,
                        )
                        processed.append(signal_payload)
                        self.last_processed_bar_time = ts
                        history_changed = True

            if history_changed:
                self._history_revision += 1
                self._cached_price_chart_overlay_key = None
                self._cached_price_chart_overlay_payload = None

            if not active_rows.empty:
                live_row = active_rows.iloc[-1]
                live_ts = active_rows.index[-1]
                live_signature = (
                    live_ts.isoformat(),
                    float(live_row["open"]),
                    float(live_row["high"]),
                    float(live_row["low"]),
                    float(live_row["close"]),
                )
                if live_signature != self.last_live_bar_signature:
                    processed.append(self._process_intrabar_snapshot(live_ts, live_row))
                    self.last_live_bar_signature = live_signature
            return processed

    def _process_single_bar(
        self,
        history: pd.DataFrame,
        bar_time: pd.Timestamp,
        periods_per_year: int,
        *,
        persist_bar_record: bool,
        intrabar_latest_signal_only: bool = False,
    ) -> dict[str, Any]:
        signals, signal_fill_prices, signal_contracts, resolved_contracts, signal_intrabar_events = self._evaluate_strategy_history(history)

        bar_index = len(history) - 1
        latest_signal_index = bar_index if getattr(self.strategy, "execute_on_signal_bar", False) else bar_index - 1
        if latest_signal_index < 0:
            if persist_bar_record:
                self._append_bar_record(bar_time, float(history.iloc[bar_index]["close"]))
            return {"timestamp": bar_time.isoformat(), "fills": 0, "position_quantity": self.engine.position.quantity}

        signal_index = latest_signal_index
        signal_value = float(signals.iloc[signal_index])
        desired_position = int(np.sign(signal_value))
        desired_contracts = float(resolved_contracts.iloc[signal_index]) if desired_position != 0 else 0.0
        should_emit_signal = True
        strategy_exit_reasons = getattr(self.strategy, "signal_exit_reason", None)
        latest_changed_signal_index = self._latest_changed_signal_index(
            signals=signals,
            resolved_contracts=resolved_contracts,
            exit_reasons=strategy_exit_reasons if isinstance(strategy_exit_reasons, pd.Series) else None,
            max_signal_index=signal_index,
            committed_only=persist_bar_record,
        )
        reason_signal_index = signal_index if latest_changed_signal_index is None else latest_changed_signal_index
        if latest_changed_signal_index is None:
            should_emit_signal = False
        elif latest_changed_signal_index != signal_index:
            should_emit_signal = desired_position != int(np.sign(self.engine.position.quantity))
        if (
            intrabar_latest_signal_only
            and signal_index != (len(history) - 1)
        ):
            should_emit_signal = False
        if (
            should_emit_signal
            and self._startup_ignore_signal_bar_before_or_equal is not None
            and history.index[signal_index] <= self._startup_ignore_signal_bar_before_or_equal
        ):
            should_emit_signal = False
        if should_emit_signal and self._startup_gate_active:
            same_startup_intent = (
                desired_position == self._startup_baseline_side
                and abs(desired_contracts - self._startup_baseline_contracts) <= 1e-12
            )
            if same_startup_intent:
                should_emit_signal = False
            else:
                self._startup_gate_active = False
        signal_fill = None
        if signal_fill_prices is not None:
            raw_fill = signal_fill_prices.iloc[signal_index]
            if pd.notna(raw_fill):
                signal_fill = float(raw_fill)

        snapshot = self.engine.snapshot()
        account_equity = snapshot.equity
        current_bar = history.iloc[bar_index]
        target_units = self.engine.position.quantity
        if should_emit_signal:
            sizing_price = signal_fill if signal_fill is not None else float(current_bar["open"])
            target_units = 0.0
            if desired_position != 0:
                current_side = int(np.sign(self.engine.position.quantity))
                if current_side == desired_position and self._active_signal_base_units is not None:
                    base_units = self._active_signal_base_units
                elif current_side == desired_position and desired_contracts > 0:
                    base_units = abs(float(self.engine.position.quantity)) / desired_contracts
                    self._active_signal_base_units = base_units
                else:
                    base_units = self.sizing_engine._resolve_units(
                        capital=account_equity,
                        fill_price=sizing_price,
                        stop_loss_price=None,
                        bar_index=bar_index,
                        closes=history["close"],
                        periods_per_year=periods_per_year,
                    )
                    self._active_signal_base_units = base_units
                target_units = base_units * desired_contracts * desired_position
                if self.config.max_leverage is not None and self.config.max_leverage > 0 and sizing_price > 0:
                    max_units = (account_equity * self.config.max_leverage) / sizing_price
                    target_units = float(np.clip(target_units, -max_units, max_units))
                if self.config.max_position_size is not None and self.config.max_position_size > 0 and sizing_price > 0:
                    max_units_notional = self.config.max_position_size / sizing_price
                    target_units = float(np.clip(target_units, -max_units_notional, max_units_notional))
            else:
                self._active_signal_base_units = None

        reference_side = desired_position if desired_position != 0 else (-1 if self.engine.position.quantity > 0 else 1)
        prev_close = float(history["close"].iloc[max(0, bar_index - 1)])
        order_request = _resolve_order_request(config=self.config, side=reference_side, prev_close=prev_close, signal_fill=signal_fill)
        signal = _build_signal(
            symbol=self.symbol,
            timestamp=bar_time,
            target_qty=target_units,
            current_qty=self.engine.position.quantity,
            projected_qty=_signed_open_order_projection(self.engine),
            order_request=order_request,
            signal_fill=signal_fill,
        )
        if signal is not None:
            entry_reason, exit_reason = _strategy_reasons_for_signal(
                self.strategy,
                self.sizing_engine,
                signal_index=reason_signal_index,
                current_qty=self.engine.position.quantity,
                target_qty=target_units,
            )
            profit_protection_exit_reasons = {
                "Strategy Profit Protection Green Gator",
                "Strategy Profit Protection Red Gator",
                "Green Gator Lips PP",
                "Red Gator Teeth PP",
                "Green PP",
                "Red PP",
            }
            signal.metadata = {
                **signal.metadata,
                "entry_reason": entry_reason,
                "exit_reason": exit_reason,
                "fill_on_close": (
                    persist_bar_record
                    and signal.action == "exit"
                    and exit_reason in profit_protection_exit_reasons
                ),
                "signal_bar_time": history.index[reason_signal_index].isoformat(),
            }
            close_confirmed_profit_protection_exit = (
                signal.action == "exit"
                and exit_reason in profit_protection_exit_reasons
            )
            if persist_bar_record or not close_confirmed_profit_protection_exit:
                self.engine.submit_signal(signal)
        self._remember_strategy_snapshot(signals, resolved_contracts, committed=persist_bar_record)

        fills_before = len(self.engine.fills)
        self._execute_intrabar_events(
            bar_time=bar_time,
            bar_index=bar_index,
            events=signal_intrabar_events.get(bar_index, []),
        )
        self.engine.on_bar(LiveBar.from_series(self.symbol, bar_time, current_bar))
        if np.isclose(self.engine.position.quantity, 0.0, atol=1e-12):
            self._active_signal_base_units = None
        fills_after = len(self.engine.fills)
        if persist_bar_record:
            self._append_bar_record(bar_time, float(current_bar["close"]))
        final_snapshot = self.engine.snapshot()
        return {
            "timestamp": bar_time.isoformat(),
            "signal_value": signal_value,
            "desired_position": desired_position,
            "fills": fills_after - fills_before,
            "position_quantity": final_snapshot.position.quantity,
            "equity": final_snapshot.equity,
        }

    def _process_intrabar_snapshot(
        self,
        bar_time: pd.Timestamp,
        bar_row: pd.Series,
    ) -> dict[str, Any]:
        intrabar_history = pd.concat([self.history, bar_row.to_frame().T])
        intrabar_history = intrabar_history[~intrabar_history.index.duplicated(keep="last")].sort_index().tail(self.warmup_bars)
        periods_per_year = infer_periods_per_year(intrabar_history.index, default=252)
        payload = self._process_single_bar(
            intrabar_history,
            bar_time,
            periods_per_year,
            persist_bar_record=False,
            intrabar_latest_signal_only=True,
        )
        payload["intrabar"] = True
        return payload

    def snapshot_summary(self) -> dict[str, Any]:
        with self._lock:
            snapshot = self.engine.snapshot()
            live_mark_price = snapshot.mark_price
            if not self.latest_market_data.empty:
                live_mark_price = float(self.latest_market_data.iloc[-1]["close"])
            live_unrealized = self.engine.unrealized_pnl(live_mark_price)
            live_equity = snapshot.cash + live_unrealized
            return {
                "symbol": self.symbol,
                "interval": self.interval,
                "strategy": self.strategy.__class__.__name__,
                "started_at": self.started_at.isoformat() if self.started_at is not None else None,
                "last_processed_bar_time": self.last_processed_bar_time.isoformat() if self.last_processed_bar_time is not None else None,
                "initial_cash": self.config.initial_capital,
                "cash": snapshot.cash,
                "equity": live_equity,
                "realized_pnl": snapshot.realized_pnl,
                "unrealized_pnl": live_unrealized,
                "fees_paid": snapshot.fees_paid,
                "position_quantity": snapshot.position.quantity,
                "position_side": snapshot.position.side,
                "position_average_price": snapshot.position.average_price,
                "mark_price": live_mark_price,
                "open_order_count": len(snapshot.open_orders),
                "fill_count": len(snapshot.fills),
                "config": {
                    "order_type": self.config.order_type,
                    "size_mode": self.config.trade_size_mode,
                    "size_value": self.config.trade_size_value,
                    "size_min_usd": self.config.trade_size_min_usd,
                    "volatility_target_annual": self.config.volatility_target_annual,
                    "volatility_lookback": self.config.volatility_lookback,
                    "max_leverage": self.config.max_leverage,
                    "max_position_size": self.config.max_position_size,
                    "max_loss": self.config.max_loss,
                    "max_loss_pct_of_equity": self.config.max_loss_pct_of_equity,
                    "equity_cutoff": self.config.equity_cutoff,
                },
            }

    def fills_dataframe(self) -> pd.DataFrame:
        with self._lock:
            snapshot = self.engine.snapshot()
            return pd.DataFrame(
                [
                    {
                        "timestamp": fill.timestamp,
                        "symbol": fill.symbol,
                        "side": fill.side,
                        "quantity": fill.quantity,
                        "price": fill.price,
                        "fee": fill.fee,
                        "reason": fill.strategy_reason or fill.reason,
                        "execution_reason": fill.reason,
                        "signal_action": fill.signal_action,
                        "realized_pnl": fill.realized_pnl,
                    }
                    for fill in snapshot.fills
                ],
                columns=[
                    "timestamp",
                    "symbol",
                    "side",
                    "quantity",
                    "price",
                    "fee",
                    "reason",
                    "execution_reason",
                    "signal_action",
                    "realized_pnl",
                ],
            )

    def fill_count(self) -> int:
        with self._lock:
            return len(self.engine.fills)

    def _build_price_chart_overlay_payload(
        self,
        market_chart_data: pd.DataFrame,
        strategy_chart_data: pd.DataFrame,
        execution_events: list[ExecutionEvent],
        completed_trades: list[_CompletedTrade],
    ) -> dict[str, list[dict[str, Any]]]:
        if market_chart_data.empty and strategy_chart_data.empty:
            return {
                "markers": [],
                "trade_event_lines": [],
                "trade_path_lines": [],
            }

        strategy_data = strategy_chart_data if not strategy_chart_data.empty else market_chart_data
        strategy = copy.deepcopy(self.strategy)
        strategy.generate_signals(strategy_data.copy())
        third_wiseman_enabled = _strategy_has_third_wiseman_enabled(strategy)

        engine_first_markers = _first_wiseman_engine_markers(
            strategy_data,
            getattr(
                strategy,
                "signal_first_wiseman_setup_marker_side",
                getattr(strategy, "signal_first_wiseman_setup_side", None),
            ),
            getattr(strategy, "signal_first_wiseman_ignored_reason", None),
            getattr(strategy, "signal_first_wiseman_reversal_side", None),
        )
        if engine_first_markers:
            wiseman_markers: list[dict[str, str | int | float]] = []
        else:
            raw_markers = _wiseman_markers(strategy_data)
            wiseman_markers = [*raw_markers["bearish"], *raw_markers["bullish"]]
        fallback_overlays = _gator_profit_protection_fallback_overlays(
            strategy_data,
            strategy,
            completed_trades,
        )

        markers = _combine_markers(
            wiseman_markers,
            engine_first_markers,
            _first_wiseman_ignored_markers(
                strategy_data,
                getattr(
                    strategy,
                    "signal_first_wiseman_setup_marker_side",
                    getattr(strategy, "signal_first_wiseman_setup_side", None),
                ),
                getattr(strategy, "signal_first_wiseman_ignored_reason", None),
            ),
            _tag_markers(
                _second_wiseman_markers(
                    strategy_data,
                    getattr(strategy, "signal_fill_prices_second", None),
                    getattr(strategy, "signal_second_wiseman_setup_side", None),
                ),
                "second_wiseman",
            ),
            _tag_markers(
                (
                    _valid_third_wiseman_fractal_markers(
                        strategy_data,
                        getattr(strategy, "signal_third_wiseman_setup_side", None),
                    )
                    if third_wiseman_enabled
                    else []
                ),
                "third_wiseman",
            ),
            _tag_markers(
                _wiseman_fill_entry_markers(
                    strategy_data,
                    getattr(strategy, "signal_fill_prices_second", None),
                    getattr(strategy, "signal_second_wiseman_fill_side", None),
                    label="2W" if _strategy_includes_bw(strategy) else None,
                ),
                "second_wiseman_entry",
            ),
            _tag_markers(
                (
                    _wiseman_fill_entry_markers(
                        strategy_data,
                        getattr(strategy, "signal_fill_prices_third", None),
                        getattr(strategy, "signal_third_wiseman_fill_side", None),
                        label="3W" if _strategy_includes_bw(strategy) else None,
                    )
                    if third_wiseman_enabled
                    else []
                ),
                "third_wiseman_entry",
            ),
            _tag_markers(
                _ntd_fill_entry_markers(
                    strategy_data,
                    getattr(strategy, "signal_fill_prices", None),
                    getattr(strategy, "signal_fractal_position_side", None),
                    getattr(strategy, "signal_contracts", None),
                ),
                "ntd_entry",
            ),
            _tag_markers(_execution_event_markers(execution_events, market_chart_data), "execution"),
            _tag_markers(
                list(fallback_overlays["markers"]),
                "gator_profit_protection_fallback",
            ),
        )
        return {
            "markers": markers,
            "trade_event_lines": [
                *_execution_event_lines(execution_events, market_chart_data.index),
                *list(fallback_overlays["trade_event_lines"]),
            ],
            "trade_path_lines": [
                *_execution_trade_path_lines(execution_events, market_chart_data.index),
                *list(fallback_overlays["trade_path_lines"]),
            ],
        }

    def _build_price_chart_payload(
        self,
        market_chart_data: pd.DataFrame,
        price_chart_overlay_payload: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        if market_chart_data.empty:
            return {
                "symbol": self.symbol,
                "interval": self.interval,
                "count": 0,
                "candles": [],
                "alligator": {"jaw": [], "teeth": [], "lips": []},
                "ao": [],
                "ac": [],
                "markers": list(price_chart_overlay_payload.get("markers", [])),
                "trade_event_lines": list(price_chart_overlay_payload.get("trade_event_lines", [])),
                "trade_path_lines": list(price_chart_overlay_payload.get("trade_path_lines", [])),
            }

        ao_histogram, ao_colors = _ao_histogram_from_data(market_chart_data)
        ac_histogram, ac_colors = _ac_histogram_from_data(market_chart_data)
        zone_colors = _williams_zones_colors(ao_colors, ac_colors)
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "count": len(market_chart_data),
            "candles": _candles_from_data(market_chart_data, zone_colors=zone_colors),
            "alligator": _alligator_series_from_data(market_chart_data),
            "ao": ao_histogram,
            "ac": ac_histogram,
            "markers": list(price_chart_overlay_payload.get("markers", [])),
            "trade_event_lines": list(price_chart_overlay_payload.get("trade_event_lines", [])),
            "trade_path_lines": list(price_chart_overlay_payload.get("trade_path_lines", [])),
        }

    def artifacts_payload(self) -> dict[str, Any]:
        with self._lock:
            snapshot = self.engine.snapshot()
            live_mark_price = snapshot.mark_price
            if not self.latest_market_data.empty:
                live_mark_price = float(self.latest_market_data.iloc[-1]["close"])
            live_unrealized = self.engine.unrealized_pnl(live_mark_price)
            live_equity = snapshot.cash + live_unrealized
            summary = {
                "symbol": self.symbol,
                "interval": self.interval,
                "strategy": self.strategy.__class__.__name__,
                "started_at": self.started_at.isoformat() if self.started_at is not None else None,
                "last_processed_bar_time": self.last_processed_bar_time.isoformat() if self.last_processed_bar_time is not None else None,
                "initial_cash": self.config.initial_capital,
                "cash": snapshot.cash,
                "equity": live_equity,
                "realized_pnl": snapshot.realized_pnl,
                "unrealized_pnl": live_unrealized,
                "fees_paid": snapshot.fees_paid,
                "position_quantity": snapshot.position.quantity,
                "position_side": snapshot.position.side,
                "position_average_price": snapshot.position.average_price,
                "mark_price": live_mark_price,
                "open_order_count": len(snapshot.open_orders),
                "fill_count": len(snapshot.fills),
                "config": {
                    "order_type": self.config.order_type,
                    "size_mode": self.config.trade_size_mode,
                    "size_value": self.config.trade_size_value,
                    "size_min_usd": self.config.trade_size_min_usd,
                    "volatility_target_annual": self.config.volatility_target_annual,
                    "volatility_lookback": self.config.volatility_lookback,
                    "max_leverage": self.config.max_leverage,
                    "max_position_size": self.config.max_position_size,
                    "equity_cutoff": self.config.equity_cutoff,
                },
            }
            fills = pd.DataFrame(
                [
                    {
                        "timestamp": fill.timestamp,
                        "symbol": fill.symbol,
                        "side": fill.side,
                        "quantity": fill.quantity,
                        "price": fill.price,
                        "fee": fill.fee,
                        "reason": fill.strategy_reason or fill.reason,
                        "execution_reason": fill.reason,
                        "signal_action": fill.signal_action,
                        "realized_pnl": fill.realized_pnl,
                    }
                    for fill in snapshot.fills
                ],
                columns=[
                    "timestamp",
                    "symbol",
                    "side",
                    "quantity",
                    "price",
                    "fee",
                    "reason",
                    "execution_reason",
                    "signal_action",
                    "realized_pnl",
                ],
            )
            equity_curve = _build_equity_curve_frame(list(self.bar_records))
            market_chart_data = (
                self.latest_market_data[["open", "high", "low", "close"]].copy()
                if not self.latest_market_data.empty
                else (
                    self.history[["open", "high", "low", "close"]].copy()
                    if not self.history.empty
                    else pd.DataFrame(columns=["open", "high", "low", "close"])
                )
            )
            strategy_chart_data = (
                self.history[["open", "high", "low", "close"]].copy()
                if not self.history.empty
                else market_chart_data.copy()
            )
            history_revision = self._history_revision
            cached_price_chart_overlay_key = self._cached_price_chart_overlay_key
            cached_price_chart_overlay_payload = (
                copy.deepcopy(self._cached_price_chart_overlay_payload)
                if self._cached_price_chart_overlay_payload is not None
                else None
            )
            candidate_updated_at: list[pd.Timestamp] = []
            if self.started_at is not None:
                candidate_updated_at.append(self.started_at)
            if self.last_processed_bar_time is not None:
                candidate_updated_at.append(self.last_processed_bar_time)
            if not self.latest_market_data.empty:
                candidate_updated_at.append(pd.Timestamp(self.latest_market_data.index[-1]))
            if snapshot.position.last_updated_at is not None:
                candidate_updated_at.append(snapshot.position.last_updated_at)
            if snapshot.fills:
                candidate_updated_at.append(pd.Timestamp(snapshot.fills[-1].timestamp))

        completed_trades = _build_completed_trades(fills, equity_curve)
        stats = _compute_live_stats(summary, equity_curve, completed_trades)
        execution_events = _paper_execution_events(fills)
        drawdown_curve: list[dict[str, Any]] = []
        if not equity_curve.empty:
            running_max = equity_curve["equity"].cummax()
            drawdowns = (equity_curve["equity"] / running_max) - 1.0
            drawdown_curve = [
                {"timestamp": ts.isoformat(), "drawdown": float(dd)}
                for ts, dd in zip(pd.to_datetime(equity_curve["timestamp"], utc=True), drawdowns.astype(float), strict=False)
            ]

        position_notional = abs(float(snapshot.position.quantity)) * float(snapshot.mark_price or 0.0)
        unrealized_return_pct: float | None = None
        if abs(float(snapshot.position.quantity)) > 1e-12 and snapshot.position.average_price > 0 and snapshot.mark_price is not None:
            if snapshot.position.quantity > 0:
                unrealized_return_pct = (float(snapshot.mark_price) - float(snapshot.position.average_price)) / float(snapshot.position.average_price)
            else:
                unrealized_return_pct = (float(snapshot.position.average_price) - float(snapshot.mark_price)) / float(snapshot.position.average_price)

        price_chart_overlay_cache_key = (history_revision, len(fills))
        if (
            cached_price_chart_overlay_key == price_chart_overlay_cache_key
            and cached_price_chart_overlay_payload is not None
        ):
            price_chart_overlay_payload = cached_price_chart_overlay_payload
        else:
            price_chart_overlay_payload = self._build_price_chart_overlay_payload(
                market_chart_data,
                strategy_chart_data,
                execution_events,
                completed_trades,
            )
            with self._lock:
                if self._history_revision == history_revision and len(self.engine.fills) == len(fills):
                    self._cached_price_chart_overlay_key = price_chart_overlay_cache_key
                    self._cached_price_chart_overlay_payload = copy.deepcopy(price_chart_overlay_payload)

        price_chart_payload = self._build_price_chart_payload(
            market_chart_data,
            price_chart_overlay_payload,
        )

        return {
            "updated_at": max(candidate_updated_at).isoformat() if candidate_updated_at else pd.Timestamp.now(tz="UTC").isoformat(),
            "summary": summary,
            "stats": stats,
            "current_position": {
                "side": snapshot.position.side,
                "quantity": float(snapshot.position.quantity),
                "notional_value": float(position_notional),
                "average_price": float(snapshot.position.average_price),
                "unrealized_return_pct": unrealized_return_pct,
                "opened_at": snapshot.position.opened_at.isoformat() if snapshot.position.opened_at is not None else None,
                "last_updated_at": snapshot.position.last_updated_at.isoformat() if snapshot.position.last_updated_at is not None else None,
                "stop_loss_price": snapshot.position.stop_loss_price,
                "take_profit_price": snapshot.position.take_profit_price,
            },
            "open_orders": [
                {
                    "order_id": order.order_id,
                    "side": order.side,
                    "order_type": order.order_type,
                    "quantity": float(order.quantity),
                    "limit_price": order.limit_price,
                    "stop_price": order.stop_price,
                    "reduce_only": bool(order.reduce_only),
                    "created_at": order.created_at.isoformat(),
                    "signal_id": order.signal_id,
                }
                for order in snapshot.open_orders
            ],
            "fills": _serialize_fills_dataframe(fills),
            "completed_trades": [trade.as_dict() for trade in completed_trades],
            "equity_curve": [
                {
                    "timestamp": pd.Timestamp(row.timestamp).isoformat(),
                    "cash": float(row.cash),
                    "equity": float(row.equity),
                    "realized_pnl": float(row.realized_pnl),
                    "unrealized_pnl": float(row.unrealized_pnl),
                    "position_quantity": float(row.position_quantity),
                    "mark_price": None if pd.isna(row.mark_price) else float(row.mark_price),
                }
                for row in equity_curve.itertuples(index=False)
            ],
            "drawdown_curve": drawdown_curve,
            "price_chart": price_chart_payload,
        }


def _artifact_paths(args: argparse.Namespace, out_dir: Path) -> PaperTradingArtifacts:
    return PaperTradingArtifacts(
        summary_path=out_dir / args.summary_name,
        fills_path=out_dir / args.fills_name,
        dashboard_path=out_dir / args.dashboard_name,
        dashboard_data_path=out_dir / args.dashboard_data_name,
        dashboard_script_path=out_dir / args.dashboard_script_name,
        status_path=out_dir / args.status_name,
        trades_path=out_dir / args.trades_name,
    )


def _merge_cached_candles(existing: pd.DataFrame, latest: pd.DataFrame, warmup_bars: int) -> pd.DataFrame:
    if existing.empty:
        return latest[~latest.index.duplicated(keep="last")].sort_index().tail(max(int(warmup_bars) + 2, 2)).copy()
    if latest.empty:
        return existing.copy()
    combined = pd.concat([existing, latest])
    return combined[~combined.index.duplicated(keep="last")].sort_index().tail(max(int(warmup_bars) + 2, 2)).copy()


def _write_live_artifacts(
    session: RealTimePaperTradingSession,
    state: _ArtifactWriteState,
    args: argparse.Namespace,
    out_dir: Path,
    *,
    include_reports: bool,
) -> tuple[PaperTradingArtifacts, dict[str, Any]]:
    artifacts = _artifact_paths(args, out_dir)
    payload = session.artifacts_payload()
    _write_cached_text_artifact(state, artifacts.summary_path, json.dumps(payload["summary"], indent=2))
    _write_cached_frame_csv_artifact(state, artifacts.fills_path, pd.DataFrame(payload["fills"]))
    _write_cached_text_artifact(state, artifacts.dashboard_data_path, json.dumps(payload, indent=2))
    _write_cached_text_artifact(
        state,
        artifacts.dashboard_script_path,
        "window.__PAPER_DASHBOARD_DATA__ = " + json.dumps(payload, indent=2) + ";\n",
    )
    if not artifacts.dashboard_path.exists():
        _write_cached_text_artifact(state, artifacts.dashboard_path, _build_dashboard_html(artifacts.dashboard_script_path.name))
    if include_reports:
        _write_status_markdown(payload["summary"], payload["stats"], payload, artifacts.status_path)
        _write_trades_markdown(payload, artifacts.trades_path)
    return artifacts, payload


def _artifact_writer_forever(
    state: _ArtifactWriteState,
    condition: threading.Condition,
    session: RealTimePaperTradingSession,
    args: argparse.Namespace,
    out_dir: Path,
) -> None:
    while True:
        with condition:
            while state.completed_generation >= state.pending_generation and not state.stop_requested:
                condition.wait()
            target_generation = state.pending_generation
            include_reports = state.report_generation > state.completed_generation or state.stop_requested
            if target_generation <= state.completed_generation and state.stop_requested:
                return
        try:
            artifacts, payload = _write_live_artifacts(
                session,
                state,
                args,
                out_dir,
                include_reports=include_reports,
            )
        except BaseException as exc:
            with condition:
                state.error = exc
                condition.notify_all()
            return
        with condition:
            state.latest_artifacts = artifacts
            state.latest_payload = payload
            state.completed_generation = max(state.completed_generation, target_generation)
            condition.notify_all()
            if state.stop_requested and state.completed_generation >= state.pending_generation:
                return


def run_from_args(
    args: argparse.Namespace,
    *,
    client: SupportsFetchKline | None = None,
    now_provider: callable | None = None,
    sleep_fn: callable | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    client = client or KCEXClient()
    now_provider = now_provider or (lambda: pd.Timestamp.now(tz="UTC"))
    sleep_fn = sleep_fn or time.sleep

    session = RealTimePaperTradingSession(
        symbol=args.symbol,
        interval=args.interval,
        strategy=_build_strategy(args),
        config=_build_backtest_config(args),
        warmup_bars=args.warmup_bars,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cycles = 0
    artifacts_state = _ArtifactWriteState()
    artifact_condition = threading.Condition()
    artifact_thread = threading.Thread(
        target=_artifact_writer_forever,
        args=(artifacts_state, artifact_condition, session, args, out_dir),
        daemon=True,
    )
    artifact_thread.start()

    cached_candles = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    fetch_limit = max(2, int(args.warmup_bars))
    consecutive_fetch_failures = 0
    try:
        while True:
            with artifact_condition:
                if artifacts_state.error is not None:
                    raise RuntimeError("Artifact writer thread failed") from artifacts_state.error
            now = now_provider()
            try:
                candles = _candles_to_dataframe(client.fetch_kline(args.symbol, args.interval, limit=fetch_limit))
                consecutive_fetch_failures = 0
            except Exception as exc:
                consecutive_fetch_failures += 1
                print(
                    (
                        f"[paper-trading] KCEX fetch failed for {args.symbol} {args.interval} "
                        f"(consecutive failures: {consecutive_fetch_failures}): {exc}"
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                if args.fetch_max_consecutive_failures > 0 and consecutive_fetch_failures >= args.fetch_max_consecutive_failures:
                    raise RuntimeError(
                        f"Aborting after {consecutive_fetch_failures} consecutive KCEX fetch failures"
                    ) from exc
                sleep_fn(max(float(args.fetch_failure_sleep_seconds), 0.0))
                continue
            cached_candles = _merge_cached_candles(cached_candles, candles, args.warmup_bars)
            fetch_limit = min(max(8, int(args.warmup_bars // 20) or 1), int(args.warmup_bars))
            last_processed_before = session.last_processed_bar_time
            fill_count_before = session.fill_count()
            session.update_market_snapshot(cached_candles, now)
            if session.last_processed_bar_time is None:
                session.prime(cached_candles, now)
            else:
                session.process_market_data(cached_candles, now)

            cycles += 1
            last_processed_after = session.last_processed_bar_time
            fill_count_after = session.fill_count()
            with artifact_condition:
                artifacts_state.pending_generation = cycles
                if (
                    cycles == 1
                    or last_processed_after != last_processed_before
                    or fill_count_after != fill_count_before
                ):
                    artifacts_state.report_generation = cycles
                artifact_condition.notify_all()
            if args.max_cycles is not None and cycles >= args.max_cycles:
                break
            sleep_fn(args.poll_seconds)
    finally:
        with artifact_condition:
            artifacts_state.pending_generation = max(artifacts_state.pending_generation, cycles)
            artifacts_state.report_generation = max(artifacts_state.report_generation, artifacts_state.pending_generation)
            artifacts_state.stop_requested = True
            artifact_condition.notify_all()
        artifact_thread.join()
        if artifacts_state.error is not None:
            raise RuntimeError("Artifact writer thread failed") from artifacts_state.error

    artifacts = artifacts_state.latest_artifacts
    payload = artifacts_state.latest_payload
    assert artifacts is not None and payload is not None
    print(json.dumps(payload["summary"], indent=2))
    print(f"Saved paper-trading summary to {artifacts.summary_path}")
    print(f"Saved paper-trading fills to {artifacts.fills_path}")
    print(f"Saved live dashboard page to {artifacts.dashboard_path}")
    print(f"Saved human-readable status page to {artifacts.status_path}")
    print(f"Saved human-readable trades page to {artifacts.trades_path}")
    return artifacts.summary_path, artifacts.fills_path, payload["summary"]


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_from_args(args)


if __name__ == "__main__":
    main()
