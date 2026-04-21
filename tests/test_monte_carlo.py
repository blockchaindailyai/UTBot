from __future__ import annotations

import pandas as pd
import pytest

from backtesting.monte_carlo import (
    MonteCarloResult,
    _compute_monte_carlo_analytics,
    generate_monte_carlo_pdf_report,
    run_return_bootstrap_monte_carlo,
)


def test_monte_carlo_bootstrap_outputs_summary_and_paths(tmp_path) -> None:
    returns = pd.Series([0.0, 0.01, -0.005, 0.02, -0.01, 0.004], dtype="float64")

    result = run_return_bootstrap_monte_carlo(
        returns=returns,
        initial_capital=10_000,
        simulations=200,
        horizon_bars=20,
        seed=7,
        block_size=2,
        threads=2,
    )

    assert result.equity_paths.shape == (200, 20)
    assert "final_equity_p50" in result.summary
    assert result.summary["final_equity_p95"] >= result.summary["final_equity_p5"]
    assert result.summary["threads_used"] >= 1

    pdf_path = tmp_path / "mc_report.pdf"
    generated = generate_monte_carlo_pdf_report(
        result,
        pdf_path,
        csv_source="examples/sample_ohlcv.csv",
        baseline_trade_count=17,
    )
    assert generated.exists()
    pdf_bytes = generated.read_bytes()
    assert pdf_bytes.startswith(b"%PDF-")
    assert b"CSV source: examples/sample_ohlcv.csv" in pdf_bytes
    assert b"Baseline trades: 17" in pdf_bytes
    assert b"Key conclusion:" not in pdf_bytes


def test_monte_carlo_reproducible_with_fixed_seed_and_threads() -> None:
    returns = pd.Series([0.01, -0.01, 0.015, 0.002, -0.003, 0.004], dtype="float64")
    r1 = run_return_bootstrap_monte_carlo(
        returns=returns,
        initial_capital=10_000,
        simulations=128,
        horizon_bars=30,
        seed=123,
        block_size=3,
        threads=4,
    )
    r2 = run_return_bootstrap_monte_carlo(
        returns=returns,
        initial_capital=10_000,
        simulations=128,
        horizon_bars=30,
        seed=123,
        block_size=3,
        threads=4,
    )

    assert r1.summary["final_equity_p50"] == r2.summary["final_equity_p50"]
    assert r1.equity_paths.equals(r2.equity_paths)


def test_monte_carlo_cagr_uses_inferred_periods_per_year() -> None:
    index = pd.date_range("2024-01-01", periods=24 * 30, freq="h")
    returns = pd.Series(0.001, index=index, dtype="float64")

    result = run_return_bootstrap_monte_carlo(
        returns=returns,
        initial_capital=10_000,
        simulations=32,
        horizon_bars=24 * 30,
        seed=5,
        block_size=1,
        threads=1,
    )

    assert result.summary["periods_per_year"] > 8_000
    assert result.summary["cagr_mean"] > 10.0


def test_monte_carlo_sharpe_scales_with_inferred_periods_per_year() -> None:
    values = [0.01, -0.005, 0.008, -0.003, 0.004, -0.002, 0.006]
    returns_daily = pd.Series(values, dtype="float64")
    returns_hourly = pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="h"), dtype="float64")

    daily = run_return_bootstrap_monte_carlo(
        returns=returns_daily,
        initial_capital=10_000,
        simulations=64,
        horizon_bars=40,
        seed=11,
        block_size=1,
        threads=1,
    )
    hourly = run_return_bootstrap_monte_carlo(
        returns=returns_hourly,
        initial_capital=10_000,
        simulations=64,
        horizon_bars=40,
        seed=11,
        block_size=1,
        threads=1,
    )

    assert hourly.summary["periods_per_year"] > daily.summary["periods_per_year"]
    assert hourly.summary["approx_sharpe"] > daily.summary["approx_sharpe"] * 3


def test_monte_carlo_baseline_alignment_uses_horizon() -> None:
    returns = pd.Series([-0.5, 1.0], dtype="float64")

    result = run_return_bootstrap_monte_carlo(
        returns=returns,
        initial_capital=10_000,
        simulations=128,
        horizon_bars=1,
        seed=3,
        block_size=1,
        threads=1,
    )

    assert result.baseline_final_equity == 5_000
    assert result.summary["probability_return_below_baseline"] == 0.0


def test_monte_carlo_baseline_alignment_wraps_for_longer_horizon() -> None:
    returns = pd.Series([0.1, -0.05], dtype="float64")

    result = run_return_bootstrap_monte_carlo(
        returns=returns,
        initial_capital=10_000,
        simulations=64,
        horizon_bars=5,
        seed=2,
        block_size=1,
        threads=1,
    )

    # Baseline path should be [ +10%, -5%, +10%, -5%, +10% ].
    expected = 10_000 * (1.1 * 0.95 * 1.1 * 0.95 * 1.1)
    assert result.baseline_final_equity == pytest.approx(expected)


def test_monte_carlo_path_sharpe_uses_true_path_returns() -> None:
    result = MonteCarloResult(
        initial_capital=100.0,
        baseline_final_equity=114.0,
        simulations=1,
        horizon_bars=3,
        seed=1,
        method="test",
        equity_paths=pd.DataFrame([[100.0, 120.0, 114.0]]),
        summary={"periods_per_year": 252.0, "approx_sharpe": 0.0, "expected_return": 0.0, "max_drawdown_p95_worst": 0.0, "probability_return_below_baseline": 0.0, "return_skew": 0.0},
    )

    analytics = _compute_monte_carlo_analytics(result)
    sharpe = float(analytics["path_sharpe"][0])

    expected_rets = pd.Series([0.2, -0.05], dtype="float64")
    expected_sharpe = (expected_rets.mean() / expected_rets.std(ddof=0)) * (252 ** 0.5)
    assert sharpe == pytest.approx(expected_sharpe)


def test_monte_carlo_negative_drift_has_low_profit_probability() -> None:
    # Loss-heavy sample should not produce optimistic outcomes.
    returns = pd.Series([-0.02, -0.01, -0.015, 0.005, -0.005, -0.01], dtype="float64")

    result = run_return_bootstrap_monte_carlo(
        returns=returns,
        initial_capital=10_000,
        simulations=2_000,
        horizon_bars=60,
        seed=99,
        block_size=1,
        threads=1,
    )

    assert result.summary["expected_return"] < 0
    assert result.summary["probability_profit"] < 0.1


def test_monte_carlo_zero_drift_stays_near_balanced_outcomes() -> None:
    # Symmetric sample should remain approximately balanced between profit/loss.
    returns = pd.Series([-0.02, -0.01, 0.01, 0.02], dtype="float64")

    result = run_return_bootstrap_monte_carlo(
        returns=returns,
        initial_capital=10_000,
        simulations=4_000,
        horizon_bars=60,
        seed=17,
        block_size=1,
        threads=1,
    )

    assert abs(result.summary["expected_return"]) < 0.03
    assert result.summary["probability_profit"] == pytest.approx(0.5, abs=0.06)


def test_monte_carlo_ruin_ends_path_at_zero() -> None:
    returns = pd.Series([-1.5, 0.2, 0.2], dtype="float64")

    result = run_return_bootstrap_monte_carlo(
        returns=returns,
        initial_capital=10_000,
        simulations=256,
        horizon_bars=5,
        seed=21,
        block_size=1,
        threads=1,
    )

    ruined_paths = result.equity_paths.eq(0.0).any(axis=1)
    assert ruined_paths.any()
    for _, row in result.equity_paths.loc[ruined_paths].iterrows():
        first_zero_idx = int((row == 0.0).to_numpy().argmax())
        assert (row.iloc[first_zero_idx:] == 0.0).all()


def test_monte_carlo_equity_cutoff_freezes_individual_legs() -> None:
    returns = pd.Series([-0.6, 0.3, 0.2], dtype="float64")

    result = run_return_bootstrap_monte_carlo(
        returns=returns,
        initial_capital=10_000,
        simulations=256,
        horizon_bars=6,
        seed=8,
        block_size=1,
        threads=1,
        equity_cutoff=5_000,
    )

    hit_cutoff = result.equity_paths.eq(5_000.0).any(axis=1)
    assert hit_cutoff.any()
    for _, row in result.equity_paths.loc[hit_cutoff].iterrows():
        first_idx = int((row == 5_000.0).to_numpy().argmax())
        assert (row.iloc[first_idx:] == 5_000.0).all()
