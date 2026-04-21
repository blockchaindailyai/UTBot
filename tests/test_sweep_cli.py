from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module(module_name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parameter_sweep_accepts_wiseman_profit_protection_alias() -> None:
    module = _load_module("run_wiseman_parameter_sweep", "examples/run_wiseman_parameter_sweep.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--wiseman-profit-protection-credit-unrealized-before-min-bars",
            "true,false",
        ]
    )

    assert args.profit_protection_credit_unrealized_before_min_bars == "true,false"


def test_parameter_sweep_accepts_1w_wait_bars_to_close() -> None:
    module = _load_module("run_wiseman_parameter_sweep", "examples/run_wiseman_parameter_sweep.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--1W-wait-bars-to-close",
            "3",
            "--wiseman-gator-direction-mode",
            "2",
        ]
    )

    assert args.wiseman_1w_wait_bars_to_close == 3
    assert args.wiseman_gator_direction_mode == "2"


def test_rolling_sweep_accepts_requested_wiseman_flags() -> None:
    module = _load_module("run_rolling_timeframe_sweep", "examples/run_rolling_timeframe_sweep.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--window-months",
            "6",
            "--jump-forward-months",
            "1",
            "--wiseman-profit-protection-credit-unrealized-before-min-bars",
            "--1W-wait-bars-to-close",
            "2",
            "--wiseman-gator-direction-mode",
            "3",
        ]
    )

    assert args.wiseman_profit_protection_credit_unrealized_before_min_bars is True
    assert args.wiseman_1w_wait_bars_to_close == 2
    assert args.wiseman_gator_direction_mode == 3


def test_monte_carlo_accepts_wiseman_gator_direction_mode() -> None:
    module = _load_module("run_monte_carlo", "examples/run_monte_carlo.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--strategy",
            "wiseman",
            "--wiseman-gator-direction-mode",
            "2",
        ]
    )

    assert args.wiseman_gator_direction_mode == 2


def test_monte_carlo_build_strategy_supports_ntd_and_combo() -> None:
    module = _load_module("run_monte_carlo", "examples/run_monte_carlo.py")
    parser = module.build_parser()

    ntd_args = parser.parse_args(["--csv", "examples/sample_ohlcv.csv", "--strategy", "ntd"])
    ntd_names = module._parse_strategy_selection(ntd_args.strategy)
    ntd_strategy = module._build_strategy(ntd_args, ntd_names)
    assert isinstance(ntd_strategy, module.NTDStrategy)
    assert ntd_strategy.require_gator_close_reset is True

    combo_args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--strategy",
            "wiseman,ntd",
        ]
    )
    combo_names = module._parse_strategy_selection(combo_args.strategy)
    combo_strategy = module._build_strategy(combo_args, combo_names)
    assert isinstance(combo_strategy, module.CombinedStrategy)
    ntd_components = [strategy for strategy in combo_strategy.strategies if isinstance(strategy, module.NTDStrategy)]
    assert len(ntd_components) == 1
    assert ntd_components[0].require_gator_close_reset is False


def test_rolling_sweep_build_strategy_supports_ntd_and_combo() -> None:
    module = _load_module("run_rolling_timeframe_sweep", "examples/run_rolling_timeframe_sweep.py")
    parser = module.build_parser()

    ntd_args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--window-months",
            "6",
            "--jump-forward-months",
            "1",
            "--strategy",
            "ntd",
        ]
    )
    ntd_strategy = module._build_strategy(ntd_args)
    assert isinstance(ntd_strategy, module.NTDStrategy)
    assert ntd_strategy.require_gator_close_reset is True

    combo_args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--window-months",
            "6",
            "--jump-forward-months",
            "1",
            "--strategy",
            "wiseman,ntd",
        ]
    )
    combo_strategy = module._build_strategy(combo_args)
    assert isinstance(combo_strategy, module.CombinedStrategy)
    ntd_components = [strategy for strategy in combo_strategy.strategies if isinstance(strategy, module.NTDStrategy)]
    assert len(ntd_components) == 1
    assert ntd_components[0].require_gator_close_reset is False


def test_monte_carlo_accepts_1w_opposite_close_min_unrealized_return_flag() -> None:
    module = _load_module("run_monte_carlo", "examples/run_monte_carlo.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--strategy",
            "wiseman",
            "--wiseman-1w-opposite-close-min-unrealized-return",
            "0.05",
        ]
    )

    assert args.wiseman_1w_opposite_close_min_unrealized_return == 0.05


def test_rolling_sweep_accepts_1w_opposite_close_min_unrealized_return_flag() -> None:
    module = _load_module("run_rolling_timeframe_sweep", "examples/run_rolling_timeframe_sweep.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--window-months",
            "6",
            "--jump-forward-months",
            "1",
            "--wiseman-1w-opposite-close-min-unrealized-return",
            "0.04",
        ]
    )

    assert args.wiseman_1w_opposite_close_min_unrealized_return == 0.04

def test_monte_carlo_accepts_hybrid_and_volatility_size_modes() -> None:
    module = _load_module("run_monte_carlo", "examples/run_monte_carlo.py")
    parser = module.build_parser()
    hybrid_args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--size-mode",
            "hybrid_min_usd_percent",
            "--size-min-usd",
            "250",
        ]
    )
    vol_args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--size-mode",
            "volatility_scaled",
            "--volatility-target-annual",
            "0.12",
            "--volatility-lookback",
            "30",
        ]
    )

    assert hybrid_args.size_mode == "hybrid_min_usd_percent"
    assert hybrid_args.size_min_usd == 250.0
    assert vol_args.size_mode == "volatility_scaled"
    assert vol_args.volatility_target_annual == 0.12
    assert vol_args.volatility_lookback == 30

    stop_loss_args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--size-mode",
            "stop_loss_scaled",
            "--size-value",
            "0.01",
        ]
    )
    assert stop_loss_args.size_mode == "stop_loss_scaled"
    assert stop_loss_args.size_value == 0.01

    milestone_args = parser.parse_args(
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

    assert milestone_args.size_mode == "equity_milestone_usd"
    assert milestone_args.size_equity_milestones == "15000:1500,20000:2000"


def test_parameter_sweep_accepts_hybrid_and_volatility_size_modes() -> None:
    module = _load_module("run_wiseman_parameter_sweep", "examples/run_wiseman_parameter_sweep.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--size-mode",
            "volatility_scaled",
            "--size-min-usd",
            "500",
            "--volatility-target-annual",
            "0.18",
            "--volatility-lookback",
            "15",
        ]
    )

    assert args.size_mode == "volatility_scaled"
    assert args.size_min_usd == 500.0
    assert args.volatility_target_annual == 0.18
    assert args.volatility_lookback == 15

    stop_loss_args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--size-mode",
            "stop_loss_scaled",
            "--size-value",
            "0.02",
        ]
    )
    assert stop_loss_args.size_mode == "stop_loss_scaled"

    milestone_args = parser.parse_args(
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

    assert milestone_args.size_mode == "equity_milestone_usd"
    assert milestone_args.size_equity_milestones == "15000:1500,20000:2000"
    assert stop_loss_args.size_value == 0.02


def test_rolling_sweep_accepts_hybrid_and_volatility_size_modes() -> None:
    module = _load_module("run_rolling_timeframe_sweep", "examples/run_rolling_timeframe_sweep.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--window-months",
            "6",
            "--jump-forward-months",
            "1",
            "--size-mode",
            "hybrid_min_usd_percent",
            "--size-min-usd",
            "100",
            "--volatility-max-scale",
            "2.5",
        ]
    )

    assert args.size_mode == "hybrid_min_usd_percent"
    assert args.size_min_usd == 100.0
    assert args.volatility_max_scale == 2.5

    stop_loss_args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--window-months",
            "6",
            "--jump-forward-months",
            "1",
            "--size-mode",
            "stop_loss_scaled",
            "--size-value",
            "0.015",
        ]
    )
    assert stop_loss_args.size_mode == "stop_loss_scaled"

    milestone_args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--window-months",
            "6",
            "--jump-forward-months",
            "1",
            "--size-mode",
            "equity_milestone_usd",
            "--size-value",
            "1000",
            "--size-equity-milestones",
            "15000:1500,20000:2000",
        ]
    )

    assert milestone_args.size_mode == "equity_milestone_usd"
    assert milestone_args.size_equity_milestones == "15000:1500,20000:2000"
    assert stop_loss_args.size_value == 0.015


def test_monte_carlo_accepts_max_position_size_flag() -> None:
    module = _load_module("run_monte_carlo", "examples/run_monte_carlo.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--max-position-size",
            "5",
        ]
    )

    assert args.max_position_size == 5.0


def test_rolling_sweep_accepts_reversal_cancel_flag() -> None:
    module = _load_module("run_rolling_timeframe_sweep", "examples/run_rolling_timeframe_sweep.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--window-months",
            "6",
            "--jump-forward-months",
            "1",
            "--wiseman-cancel-reversal-on-first-exit",
        ]
    )

    assert args.wiseman_cancel_reversal_on_first_exit is True


def test_parameter_sweep_accepts_reversal_cooldown_flag() -> None:
    module = _load_module("run_wiseman_parameter_sweep", "examples/run_wiseman_parameter_sweep.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--wiseman-reversal-cooldown",
            "5",
        ]
    )

    assert args.wiseman_reversal_cooldown == 5


def test_rolling_sweep_accepts_reversal_cooldown_flag() -> None:
    module = _load_module("run_rolling_timeframe_sweep", "examples/run_rolling_timeframe_sweep.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--window-months",
            "6",
            "--jump-forward-months",
            "1",
            "--wiseman-reversal-cooldown",
            "4",
        ]
    )

    assert args.wiseman_reversal_cooldown == 4


def test_rolling_sweep_report_includes_complete_window_results_section(tmp_path) -> None:
    import argparse

    import pandas as pd

    module = _load_module("run_rolling_timeframe_sweep", "examples/run_rolling_timeframe_sweep.py")

    rows = []
    for i in range(45):
        rows.append(
            {
                "window_id": i + 1,
                "start": f"2024-01-{(i % 28) + 1:02d}T00:00:00",
                "end": f"2024-02-{(i % 28) + 1:02d}T00:00:00",
                "planned_end": f"2024-02-{(i % 28) + 1:02d}T00:00:00",
                "bars": 200 + i,
                "trades": 10 + (i % 5),
                "total_return": 0.01 * (i - 10),
                "cagr": 0.05,
                "sharpe": 0.5 + (i * 0.01),
                "max_drawdown": -0.02 * (1 + (i % 4)),
                "final_equity": 10_000 + i * 25,
                "win_rate": 0.4 + ((i % 6) * 0.05),
                "liquidated": i % 17 == 0,
                "equity_cutoff_hit": i % 19 == 0,
                "window_cutoff": i % 23 == 0,
                "cluster": i % 3,
            }
        )
    df = pd.DataFrame(rows)

    summary = {
        "windows_total": len(df),
        "windows_liquidated": int(df["liquidated"].sum()),
        "windows_equity_cutoff_hit": int(df["equity_cutoff_hit"].sum()),
        "windows_time_cutoff": int(df["window_cutoff"].sum()),
        "mean_total_return": float(df["total_return"].mean()),
        "median_total_return": float(df["total_return"].median()),
        "mean_sharpe": float(df["sharpe"].mean()),
        "median_sharpe": float(df["sharpe"].median()),
        "expected_win_rate": float(df["win_rate"].mean()),
    }

    args = argparse.Namespace(
        csv="examples/sample_ohlcv.csv",
        start=None,
        end=None,
        window_months=6,
        jump_forward_months=1,
        threads=1,
        strategy="alligator_ao",
        capital=10_000,
        fee=0.0005,
        slippage=0.0002,
        spread=0.0,
        order_type="market",
        size_mode="percent_of_equity",
        size_value=1.0,
        size_min_usd=0.0,
        volatility_target_annual=0.15,
        volatility_lookback=20,
        volatility_min_scale=0.25,
        volatility_max_scale=3.0,
        max_leverage=None,
        max_position_size=None,
        leverage_stop_out=0.0,
        borrow_annual=0.0,
        funding_per_period=0.0,
        overnight_annual=0.0,
        max_loss=None,
        equity_cutoff=None,
    )
    data = pd.read_csv("examples/sample_ohlcv.csv", parse_dates=["timestamp"]).set_index("timestamp")

    out = tmp_path / "rolling_report.pdf"
    module._make_pdf_report(out, df, summary, args, data)

    pdf_text = out.read_bytes().decode("latin-1", errors="ignore")
    assert "Run configuration" in pdf_text
    assert "order_type" in pdf_text
    assert "funding_per_period" in pdf_text
    assert "Portfolio-level diagnostics" in pdf_text
    assert "positive_sharpe" in pdf_text
    assert "Complete rolling window sequence results" in pdf_text
    assert "45" in pdf_text


def test_monte_carlo_accepts_lips_profit_protection_flags() -> None:
    module = _load_module("run_monte_carlo", "examples/run_monte_carlo.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--strategy",
            "wiseman",
            "--wiseman-profit-protection-lips-exit",
            "--wiseman-profit-protection-lips-volatility-trigger",
            "0.03",
            "--wiseman-profit-protection-lips-profit-trigger-mult",
            "2.5",
            "--wiseman-profit-protection-lips-volatility-lookback",
            "30",
            "--wiseman-profit-protection-lips-recent-trade-lookback",
            "7",
            "--wiseman-profit-protection-lips-min-unrealized-return",
            "0.012",
            "--wiseman-profit-protection-lips-arm-on-min-unrealized-return",
        ]
    )

    assert args.wiseman_profit_protection_lips_exit is True
    assert args.wiseman_profit_protection_lips_volatility_trigger == 0.03
    assert args.wiseman_profit_protection_lips_profit_trigger_mult == 2.5
    assert args.wiseman_profit_protection_lips_volatility_lookback == 30
    assert args.wiseman_profit_protection_lips_recent_trade_lookback == 7
    assert args.wiseman_profit_protection_lips_min_unrealized_return == 0.012
    assert args.wiseman_profit_protection_lips_arm_on_min_unrealized_return is True


def test_monte_carlo_accepts_profit_protection_volatility_lookback_flag() -> None:
    module = _load_module("run_monte_carlo", "examples/run_monte_carlo.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--strategy",
            "wiseman",
            "--wiseman-profit-protection-volatility-lookback",
            "40",
        ]
    )

    assert args.wiseman_profit_protection_volatility_lookback == 40


def test_monte_carlo_accepts_profit_protection_annualized_volatility_scaler_flag() -> None:
    module = _load_module("run_monte_carlo", "examples/run_monte_carlo.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--strategy",
            "wiseman",
            "--wiseman-profit-protection-annualized-volatility-scaler",
            "0.85",
        ]
    )

    assert args.wiseman_profit_protection_annualized_volatility_scaler == 0.85


def test_monte_carlo_accepts_zone_profit_protection_flags() -> None:
    module = _load_module("run_monte_carlo", "examples/run_monte_carlo.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--strategy",
            "wiseman",
            "--wiseman-profit-protection-zones-exit",
            "--wiseman-profit-protection-zones-min-unrealized-return",
            "0.018",
        ]
    )

    assert args.wiseman_profit_protection_zones_exit is True
    assert args.wiseman_profit_protection_zones_min_unrealized_return == 0.018


def test_rolling_sweep_accepts_lips_profit_protection_flags() -> None:
    module = _load_module("run_rolling_timeframe_sweep", "examples/run_rolling_timeframe_sweep.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--window-months",
            "6",
            "--jump-forward-months",
            "1",
            "--wiseman-profit-protection-lips-exit",
            "--wiseman-profit-protection-lips-volatility-trigger",
            "0.025",
            "--wiseman-profit-protection-lips-profit-trigger-mult",
            "3.0",
            "--wiseman-profit-protection-lips-min-unrealized-return",
            "0.02",
            "--wiseman-profit-protection-lips-arm-on-min-unrealized-return",
        ]
    )

    assert args.wiseman_profit_protection_lips_exit is True
    assert args.wiseman_profit_protection_lips_volatility_trigger == 0.025
    assert args.wiseman_profit_protection_lips_profit_trigger_mult == 3.0
    assert args.wiseman_profit_protection_lips_min_unrealized_return == 0.02
    assert args.wiseman_profit_protection_lips_arm_on_min_unrealized_return is True


def test_rolling_sweep_accepts_profit_protection_volatility_lookback_flag() -> None:
    module = _load_module("run_rolling_timeframe_sweep", "examples/run_rolling_timeframe_sweep.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--window-months",
            "6",
            "--jump-forward-months",
            "1",
            "--wiseman-profit-protection-volatility-lookback",
            "35",
        ]
    )

    assert args.wiseman_profit_protection_volatility_lookback == 35


def test_rolling_sweep_accepts_profit_protection_annualized_volatility_scaler_flag() -> None:
    module = _load_module("run_rolling_timeframe_sweep", "examples/run_rolling_timeframe_sweep.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--window-months",
            "6",
            "--jump-forward-months",
            "1",
            "--wiseman-profit-protection-annualized-volatility-scaler",
            "0.9",
        ]
    )

    assert args.wiseman_profit_protection_annualized_volatility_scaler == 0.9


def test_rolling_sweep_accepts_zone_profit_protection_flags() -> None:
    module = _load_module("run_rolling_timeframe_sweep", "examples/run_rolling_timeframe_sweep.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--window-months",
            "6",
            "--jump-forward-months",
            "1",
            "--wiseman-profit-protection-zones-exit",
            "--wiseman-profit-protection-zones-min-unrealized-return",
            "0.017",
        ]
    )

    assert args.wiseman_profit_protection_zones_exit is True
    assert args.wiseman_profit_protection_zones_min_unrealized_return == 0.017


def test_parameter_sweep_accepts_zone_profit_protection_flags() -> None:
    module = _load_module("run_wiseman_parameter_sweep", "examples/run_wiseman_parameter_sweep.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--wiseman-profit-protection-zones-exit",
            "true,false",
            "--wiseman-profit-protection-zones-min-unrealized-return",
            "0.01,0.02",
        ]
    )

    assert args.wiseman_profit_protection_zones_exit == "true,false"
    assert args.wiseman_profit_protection_zones_min_unrealized_return == "0.01,0.02"


def test_parameter_sweep_accepts_profit_protection_volatility_lookback_flag() -> None:
    module = _load_module("run_wiseman_parameter_sweep", "examples/run_wiseman_parameter_sweep.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--wiseman-profit-protection-volatility-lookback",
            "20,40",
        ]
    )

    assert args.wiseman_profit_protection_volatility_lookback == "20,40"


def test_parameter_sweep_accepts_profit_protection_annualized_volatility_scaler_flag() -> None:
    module = _load_module("run_wiseman_parameter_sweep", "examples/run_wiseman_parameter_sweep.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--wiseman-profit-protection-annualized-volatility-scaler",
            "0.75,0.85,1.0",
        ]
    )

    assert args.wiseman_profit_protection_annualized_volatility_scaler == "0.75,0.85,1.0"


def test_monte_carlo_accepts_1w_divergence_filter_flag() -> None:
    module = _load_module("run_monte_carlo", "examples/run_monte_carlo.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--strategy",
            "wiseman",
            "--1W-divergence-filter",
            "180",
        ]
    )

    assert args.wiseman_1w_divergence_filter_bars == 180


def test_rolling_sweep_accepts_1w_divergence_filter_flag() -> None:
    module = _load_module("run_rolling_timeframe_sweep", "examples/run_rolling_timeframe_sweep.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--window-months",
            "6",
            "--jump-forward-months",
            "1",
            "--1W-divergence-filter",
            "180",
        ]
    )

    assert args.wiseman_1w_divergence_filter_bars == 180


def test_parameter_sweep_accepts_1w_divergence_filter_flag() -> None:
    module = _load_module("run_wiseman_parameter_sweep", "examples/run_wiseman_parameter_sweep.py")
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--csv",
            "examples/sample_ohlcv.csv",
            "--1W-divergence-filter",
            "180",
        ]
    )

    assert args.wiseman_1w_divergence_filter_bars == 180
