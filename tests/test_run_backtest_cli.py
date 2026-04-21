from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtesting import BacktestResult, ExecutionEvent, Trade

MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "run_backtest.py"
SPEC = importlib.util.spec_from_file_location("run_backtest", MODULE_PATH)
assert SPEC and SPEC.loader
run_backtest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_backtest)


class _EstimateEngineRaises:
    def run(self, data: pd.DataFrame, strategy: object) -> None:
        raise ValueError("stop_loss_scaled trade sizing requires strategy.signal_stop_loss_prices for entry signals")


class _EstimateEngineSucceeds:
    def run(self, data: pd.DataFrame, strategy: object) -> None:
        return None


def test_parse_asset_allocations_accepts_repeatable_entries() -> None:
    allocation_map = run_backtest._parse_asset_allocations(["BTC=0.5", "ETH.csv=0.25"])

    assert allocation_map == {"BTC": 0.5, "ETH.csv": 0.25}


@pytest.mark.parametrize("entry", ["BTC", "=1", "BTC=abc"])
def test_parse_asset_allocations_rejects_invalid_entries(entry: str) -> None:
    with pytest.raises(ValueError):
        run_backtest._parse_asset_allocations([entry])


def test_resolve_asset_allocation_matches_path_before_filename_or_stem() -> None:
    csv_path = str(Path("examples") / "BINANCE_XRPUSD60.csv")
    full_path = str(Path(csv_path).resolve())
    allocation_map = {
        "BINANCE_XRPUSD60": 0.4,
        "BINANCE_XRPUSD60.csv": 0.6,
        csv_path: 0.8,
        full_path: 1.0,
    }

    assert run_backtest._resolve_asset_allocation_for_csv(csv_path, allocation_map=allocation_map) == 0.8


def test_resolve_asset_allocation_falls_back_to_none() -> None:
    assert run_backtest._resolve_asset_allocation_for_csv("examples/UNKNOWN.csv", allocation_map={}) is None


def test_build_per_csv_allocations_defaults_each_asset_to_full_capital() -> None:
    csv_paths = ["examples/BTC.csv", "examples/ETH.csv", "examples/TRX.csv"]

    allocations = run_backtest._build_per_csv_allocations(csv_paths, allocation_map={})

    assert allocations == {csv_path: 1.0 for csv_path in csv_paths}


def test_build_per_csv_allocations_keeps_unspecified_assets_at_full_capital() -> None:
    csv_paths = ["examples/BTC.csv", "examples/ETH.csv"]

    allocations = run_backtest._build_per_csv_allocations(csv_paths, allocation_map={"BTC": 0.5})

    assert allocations == {
        "examples/BTC.csv": 0.5,
        "examples/ETH.csv": 1.0,
    }


def test_build_per_csv_allocations_accepts_multipliers_above_one() -> None:
    allocations = run_backtest._build_per_csv_allocations(["examples/BTC.csv"], allocation_map={"BTC": 1.5})

    assert allocations == {"examples/BTC.csv": 1.5}


def test_parse_strategy_selection_supports_ntd_and_wiseman_combo() -> None:
    assert run_backtest._parse_strategy_selection("ntd") == ("ntd",)
    assert run_backtest._parse_strategy_selection("wiseman,ntd") == ("wiseman", "ntd")
    assert run_backtest._parse_strategy_selection("bw") == ("bw",)


def test_parse_strategy_selection_rejects_invalid_combinations() -> None:
    with pytest.raises(ValueError):
        run_backtest._parse_strategy_selection("alligator_ao,ntd")


def test_run_backtest_ntd_combo_disables_gator_close_reset() -> None:
    strategy_names = run_backtest._parse_strategy_selection("wiseman,ntd")

    assert strategy_names == ("wiseman", "ntd")


def test_build_parser_accepts_ntd_cli_flags() -> None:
    parser = run_backtest.build_parser()

    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--strategy",
            "ntd",
            "--ntd-ao-ac-near-zero-lookback",
            "80",
            "--ntd-ao-ac-near-zero-factor",
            "0.8",
            "--ntd-require-gator-close-reset",
        ]
    )

    assert args.ntd_ao_ac_near_zero_lookback == 80
    assert args.ntd_ao_ac_near_zero_factor == pytest.approx(0.8)
    assert args.ntd_require_gator_close_reset is True


def test_build_parser_accepts_bw_contract_flags() -> None:
    parser = run_backtest.build_parser()

    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--strategy",
            "bw",
            "--bw-1w-gator-open-lookback",
            "100",
            "--bw-1w-gator-open-percentile",
            "50",
            "--bw-1w-contracts",
            "2",
            "--bw-only-trade-1w-r",
            "--bw-ntd-initial-fractal-contracts",
            "4",
            "--Fractal-add-on-contracts",
            "3",
            "--bw-fractal-add-ons-enabled",
            "--no-bw-profit-protection-red-teeth-exit",
            "--bw-profit-protection-red-teeth-min-bars",
            "5",
            "--bw-profit-protection-red-teeth-min-unrealized-return",
            "0.02",
            "--bw-profit-protection-red-teeth-require-gator-direction-alignment",
            "--no-bw-profit-protection-green-lips-exit",
            "--bw-profit-protection-green-lips-min-bars",
            "7",
            "--bw-profit-protection-green-lips-min-unrealized-return",
            "0.03",
            "--bw-profit-protection-green-lips-require-gator-direction-alignment",
            "--bw-profit-protection-red-teeth-latch-min-unrealized-return",
            "--bw-profit-protection-green-lips-latch-min-unrealized-return",
            "--bw-profit-protection-zones-exit",
            "--bw-profit-protection-zones-min-bars",
            "8",
            "--bw-profit-protection-zones-min-unrealized-return",
            "0.04",
            "--bw-profit-protection-zones-volatility-lookback",
            "180",
            "--bw-profit-protection-zones-annualized-volatility-scaler",
            "0.5",
            "--bw-profit-protection-zones-min-same-color-bars",
            "6",
            "--bw-peak-drawdown-exit",
            "--bw-peak-drawdown-exit-pct",
            "0.0125",
            "--bw-peak-drawdown-exit-volatility-lookback",
            "55",
            "--bw-peak-drawdown-exit-annualized-volatility-scaler",
            "0.8",
            "--bw-close-on-underlying-gain-pct",
            "0.025",
            "--allow-close-on-1w-d",
            "--allow-close-on-1w-d-min-unrealized-return",
            "0.05",
            "--allow-close-on-1w-a",
            "--allow-close-on-1w-a-min-unrealized-return",
            "0.06",
        ]
    )

    assert args.bw_1w_contracts == 2
    assert args.bw_only_trade_1w_r is True
    assert args.bw_1w_gator_open_lookback == 100
    assert args.bw_1w_gator_open_percentile == pytest.approx(50.0)
    assert args.bw_ntd_initial_fractal_contracts == 4
    assert args.bw_fractal_add_on_contracts == 3
    assert args.bw_fractal_add_ons_enabled is True
    assert args.bw_profit_protection_red_teeth_exit is False
    assert args.bw_profit_protection_red_teeth_min_bars == 5
    assert args.bw_profit_protection_red_teeth_min_unrealized_return == pytest.approx(0.02)
    assert args.bw_profit_protection_red_teeth_require_gator_direction_alignment is True
    assert args.bw_profit_protection_green_lips_exit is False
    assert args.bw_profit_protection_green_lips_min_bars == 7
    assert args.bw_profit_protection_green_lips_min_unrealized_return == pytest.approx(0.03)
    assert args.bw_profit_protection_green_lips_require_gator_direction_alignment is True
    assert args.bw_profit_protection_red_teeth_latch_min_unrealized_return is True
    assert args.bw_profit_protection_green_lips_latch_min_unrealized_return is True
    assert args.bw_profit_protection_zones_exit is True
    assert args.bw_profit_protection_zones_min_bars == 8
    assert args.bw_profit_protection_zones_min_unrealized_return == pytest.approx(0.04)
    assert args.bw_profit_protection_zones_volatility_lookback == 180
    assert args.bw_profit_protection_zones_annualized_volatility_scaler == pytest.approx(0.5)
    assert args.bw_profit_protection_zones_min_same_color_bars == 6
    assert args.bw_peak_drawdown_exit is True
    assert args.bw_peak_drawdown_exit_pct == pytest.approx(0.0125)
    assert args.bw_peak_drawdown_exit_volatility_lookback == 55
    assert args.bw_peak_drawdown_exit_annualized_volatility_scaler == pytest.approx(0.8)
    assert args.bw_close_on_underlying_gain_pct == pytest.approx(0.025)
    assert args.allow_close_on_1w_d is True
    assert args.allow_close_on_1w_d_min_unrealized_return == pytest.approx(0.05)
    assert args.allow_close_on_1w_a is True
    assert args.allow_close_on_1w_a_min_unrealized_return == pytest.approx(0.06)


def test_build_parser_can_disable_bw_fractal_add_ons() -> None:
    parser = run_backtest.build_parser()

    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--strategy",
            "bw",
            "--no-bw-fractal-add-ons-enabled",
            "--bw-fractal-add-on-contracts",
            "5",
        ]
    )

    assert args.bw_fractal_add_ons_enabled is False
    assert args.bw_fractal_add_on_contracts == 5


def test_build_parser_accepts_equity_milestone_usd_size_mode() -> None:
    parser = run_backtest.build_parser()

    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--size-mode",
            "equity_milestone_usd",
            "--size-value",
            "1000",
            "--size-equity-milestones",
            "15000:1500,20000:2000",
        ]
    )

    assert args.size_mode == "equity_milestone_usd"
    assert args.size_equity_milestones == "15000:1500,20000:2000"




def test_build_trade_execution_log_tracks_individual_trade_reasoning() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    result = BacktestResult(
        equity_curve=pd.Series([10_000.0, 10_050.0, 10_100.0], index=idx),
        returns=pd.Series([0.0, 0.005, 0.004975], index=idx),
        positions=pd.Series([0.0, 1.0, 0.0], index=idx),
        trades=[
            Trade(
                side="long",
                entry_time=idx[0],
                exit_time=idx[2],
                entry_price=100.0,
                exit_price=101.0,
                units=1.0,
                pnl=1.0,
                return_pct=0.01,
                holding_bars=2,
                entry_signal="Bullish 1W",
                exit_signal="Exit to Flat",
            )
        ],
        stats={},
        data_quality={},
        execution_events=[
            ExecutionEvent(
                event_type="entry",
                time=idx[0],
                side="buy",
                price=100.0,
                units=1.0,
                strategy_reason="Bullish 1W",
            ),
            ExecutionEvent(
                event_type="add",
                time=idx[1],
                side="buy",
                price=100.5,
                units=0.5,
                strategy_reason="Fractal Add-on",
            ),
            ExecutionEvent(
                event_type="exit",
                time=idx[2],
                side="sell",
                price=101.0,
                units=1.5,
                strategy_reason="Exit to Flat",
            ),
        ],
        total_fees_paid=0.0,
        total_financing_paid=0.0,
        total_profit_before_fees=1.0,
    )

    frame = run_backtest._build_trade_execution_log(result)

    assert list(frame["event_type"]) == ["entry", "add", "exit"]
    assert list(frame["trade_id"]) == [1, 1, 1]
    assert list(frame["reasoning"]) == ["Bullish 1W", "Fractal Add-on", "Exit to Flat"]
    assert list(frame["usd_amount"]) == [100.0, 50.25, 151.5]
    assert list(frame["total_position_units"]) == [1.0, 1.5, 0.0]
    assert list(frame["total_position_usd"]) == [100.0, 150.75, 0.0]
    assert list(frame["sizing_mode"]) == [None, None, None]


def test_artifact_path_uses_unique_suffix_for_multi_asset() -> None:
    out_dir = Path("artifacts")
    assert run_backtest._artifact_path(out_dir, "report", ".pdf", "BINANCE_ETHUSD1h") == out_dir / "report-BINANCE_ETHUSD1h.pdf"
    assert run_backtest._artifact_path(out_dir, "report", ".pdf", None) == out_dir / "report.pdf"


def test_build_signal_reason_monitor_computes_stop_and_early_stop_rates() -> None:
    trades = pd.DataFrame(
        [
            {
                "entry_signal": "1W Long",
                "exit_signal": "Protective Stop",
                "pnl": -50.0,
                "return_pct": -0.01,
                "holding_bars": 1,
            },
            {
                "entry_signal": "1W Long",
                "exit_signal": "Trend Exit",
                "pnl": 80.0,
                "return_pct": 0.02,
                "holding_bars": 6,
            },
            {
                "entry_signal": "AO Short",
                "exit_signal": "Stop Out",
                "pnl": -20.0,
                "return_pct": -0.005,
                "holding_bars": 4,
            },
        ]
    )

    monitor = run_backtest._build_signal_reason_monitor(trades)
    by_entry = monitor["by_entry"].set_index("entry_signal")
    by_exit = monitor["by_exit"].set_index("exit_signal")
    by_pairs = monitor["entry_exit_pairs"]

    assert by_entry.loc["1W Long", "trades"] == 2
    assert by_entry.loc["1W Long", "stop_exit_rate"] == pytest.approx(0.5)
    assert by_entry.loc["1W Long", "early_stop_rate"] == pytest.approx(0.5)
    assert "avg_giveback_from_peak_pct" in by_entry.columns
    assert "avg_capture_ratio_vs_peak" in by_entry.columns
    assert by_entry.loc["AO Short", "stop_exit_rate"] == pytest.approx(1.0)
    assert by_exit.loc["Protective Stop", "trades"] == 1
    assert set(by_pairs.columns) == {
        "entry_signal",
        "exit_signal",
        "trades",
        "win_rate",
        "avg_holding_bars",
        "net_pnl",
        "avg_giveback_from_peak_pct",
    }


def test_build_signal_reason_monitor_empty_input_returns_empty_frames() -> None:
    monitor = run_backtest._build_signal_reason_monitor(pd.DataFrame())

    assert list(monitor) == ["by_entry", "by_exit", "entry_exit_pairs"]
    assert monitor["by_entry"].empty
    assert monitor["by_exit"].empty
    assert monitor["entry_exit_pairs"].empty


def test_runtime_estimate_message_skips_failed_pilot_slice() -> None:
    data = pd.DataFrame(
        {
            "open": [100.0] * 250,
            "high": [101.0] * 250,
            "low": [99.0] * 250,
            "close": [100.0] * 250,
            "volume": [1000.0] * 250,
        },
        index=pd.date_range("2024-01-01", periods=250, freq="h", tz="UTC"),
    )

    message = run_backtest._runtime_estimate_message(
        engine=_EstimateEngineRaises(),
        data=data,
        build_strategy=lambda: object(),
        csv_path="examples/BINANCE_ETHUSD4h.csv",
    )

    assert "runtime estimate skipped" in message
    assert "Continuing with full run." in message
    assert "signal_stop_loss_prices" in message


def test_runtime_estimate_message_reports_small_datasets() -> None:
    data = pd.DataFrame(
        {
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.0] * 10,
            "volume": [1000.0] * 10,
        },
        index=pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC"),
    )

    message = run_backtest._runtime_estimate_message(
        engine=_EstimateEngineSucceeds(),
        data=data,
        build_strategy=lambda: object(),
        csv_path="examples/BINANCE_ETHUSD4h.csv",
    )

    assert "dataset is small" in message


def test_build_consolidated_result_aggregates_equity_and_trades() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    trade = Trade(
        side="long",
        entry_time=index[0],
        exit_time=index[1],
        entry_price=100.0,
        exit_price=101.0,
        units=1.0,
        pnl=1.0,
        return_pct=0.01,
        holding_bars=1,
    )
    result_a = BacktestResult(
        equity_curve=pd.Series([2000.0, 2010.0, 2020.0], index=index),
        returns=pd.Series([0.0, 0.005, 0.004975], index=index),
        positions=pd.Series([0.0, 1.0, 0.0], index=index),
        trades=[trade],
        stats={"slippage_rate": 0.0002},
        data_quality={"is_datetime_index": True, "timezone_aware": True, "duplicate_timestamps": 0.0, "missing_bars": 1.0, "outlier_bars": 0.0},
        execution_events=[],
        total_fees_paid=1.0,
        total_financing_paid=0.0,
        total_profit_before_fees=2.0,
    )
    result_b = BacktestResult(
        equity_curve=pd.Series([8000.0, 7960.0, 8016.0], index=index),
        returns=pd.Series([0.0, -0.005, 0.007035], index=index),
        positions=pd.Series([0.0, -1.0, 0.0], index=index),
        trades=[trade],
        stats={"slippage_rate": 0.0002},
        data_quality={"is_datetime_index": True, "timezone_aware": True, "duplicate_timestamps": 0.0, "missing_bars": 2.0, "outlier_bars": 1.0},
        execution_events=[],
        total_fees_paid=2.0,
        total_financing_paid=0.5,
        total_profit_before_fees=1.5,
    )

    consolidated = run_backtest._build_consolidated_result([result_a, result_b], initial_capital=10_000.0, slippage_rate=0.0002)

    assert consolidated.equity_curve.iloc[0] == 10000.0
    assert consolidated.equity_curve.iloc[-1] == pytest.approx(10036.0, abs=1e-6)
    assert len(consolidated.trades) == 2
    assert consolidated.data_quality["missing_bars"] == 3.0
    assert consolidated.stats["total_runs"] == 2.0
    assert consolidated.stats["initial_capital_total"] == 10000.0
    assert consolidated.stats["consolidated_initial_capital"] == 10000.0
    assert consolidated.stats["total_fees_paid_nominal_sum"] == 3.0
    assert consolidated.stats["total_financing_paid_nominal_sum"] == 0.5
    assert consolidated.stats["total_profit_before_fees_nominal_sum"] == 3.5
    assert consolidated.stats["requested_consolidated_initial_capital"] == 10000.0
    assert consolidated.total_fees_paid == 3.0
    assert consolidated.total_financing_paid == 0.5
    assert consolidated.total_profit_before_fees == 3.5
    assert len(consolidated.execution_events) == 0


def test_build_consolidated_result_handles_staggered_date_ranges() -> None:
    idx_a = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    idx_b = pd.date_range("2024-01-02", periods=3, freq="D", tz="UTC")
    result_a = BacktestResult(
        equity_curve=pd.Series([2000.0, 2010.0, 2020.0], index=idx_a),
        returns=pd.Series([0.0, 0.005, 0.004975], index=idx_a),
        positions=pd.Series([0.0, 1.0, 0.0], index=idx_a),
        trades=[],
        stats={},
        data_quality={"is_datetime_index": True, "timezone_aware": True, "duplicate_timestamps": 0.0, "missing_bars": 0.0, "outlier_bars": 0.0},
        execution_events=[],
        total_fees_paid=0.0,
        total_financing_paid=0.0,
        total_profit_before_fees=0.0,
    )
    result_b = BacktestResult(
        equity_curve=pd.Series([8000.0, 8100.0, 8200.0], index=idx_b),
        returns=pd.Series([0.0, 0.0125, 0.012346], index=idx_b),
        positions=pd.Series([0.0, -1.0, 0.0], index=idx_b),
        trades=[],
        stats={},
        data_quality={"is_datetime_index": True, "timezone_aware": True, "duplicate_timestamps": 0.0, "missing_bars": 0.0, "outlier_bars": 0.0},
        execution_events=[],
        total_fees_paid=0.0,
        total_financing_paid=0.0,
        total_profit_before_fees=0.0,
    )

    consolidated = run_backtest._build_consolidated_result([result_a, result_b], initial_capital=10_000.0, slippage_rate=0.0)

    assert consolidated.equity_curve.iloc[0] == 10_000.0
    assert consolidated.equity_curve.loc[pd.Timestamp("2024-01-02", tz="UTC")] == 10_010.0
    assert consolidated.equity_curve.iloc[-1] == 10_220.0


def test_run_csv_paths_in_threads_single_path() -> None:
    calls: list[str] = []

    def worker(csv_path: str) -> tuple[str, BacktestResult, dict[str, object]]:
        calls.append(csv_path)
        return csv_path, BacktestResult(
            equity_curve=pd.Series([1.0, 1.0], index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")),
            returns=pd.Series([0.0, 0.0], index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")),
            positions=pd.Series([0.0, 0.0], index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")),
            trades=[],
            stats={},
            data_quality={},
            execution_events=[],
            total_fees_paid=0.0,
            total_financing_paid=0.0,
            total_profit_before_fees=0.0,
        ), {"csv": csv_path}

    results = run_backtest._run_csv_paths_in_threads(["one.csv"], worker)

    assert calls == ["one.csv"]
    assert [csv for csv, _, _ in results] == ["one.csv"]


def test_run_csv_paths_serially_preserves_input_order() -> None:
    calls: list[str] = []

    def worker(csv_path: str) -> tuple[str, BacktestResult, dict[str, object]]:
        calls.append(csv_path)
        return csv_path, BacktestResult(
            equity_curve=pd.Series([1.0, 1.0], index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")),
            returns=pd.Series([0.0, 0.0], index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")),
            positions=pd.Series([0.0, 0.0], index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")),
            trades=[],
            stats={},
            data_quality={},
            execution_events=[],
            total_fees_paid=0.0,
            total_financing_paid=0.0,
            total_profit_before_fees=0.0,
        ), {"csv": csv_path}

    csv_paths = ["b.csv", "a.csv", "c.csv"]
    results = run_backtest._run_csv_paths_serially(csv_paths, worker)

    assert calls == csv_paths
    assert [csv for csv, _, _ in results] == csv_paths


def test_run_csv_paths_in_threads_preserves_input_order() -> None:
    def worker(csv_path: str) -> tuple[str, BacktestResult, dict[str, object]]:
        return csv_path, BacktestResult(
            equity_curve=pd.Series([1.0, 1.0], index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")),
            returns=pd.Series([0.0, 0.0], index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")),
            positions=pd.Series([0.0, 0.0], index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")),
            trades=[],
            stats={},
            data_quality={},
            execution_events=[],
            total_fees_paid=0.0,
            total_financing_paid=0.0,
            total_profit_before_fees=0.0,
        ), {"csv": csv_path}

    csv_paths = ["b.csv", "a.csv", "c.csv"]
    results = run_backtest._run_csv_paths_in_threads(csv_paths, worker)

    assert [csv for csv, _, _ in results] == csv_paths
