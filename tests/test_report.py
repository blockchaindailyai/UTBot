from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtesting.engine import BacktestResult
from backtesting.report import (
    _SimplePdf,
    _add_text,
    _build_robustness_diagnostics,
    _compute_r_multiples,
    _compute_trade_metrics_from_pnl,
    _detect_log_growth,
    _entry_exit_signal_rows,
    _percent_label,
    _sqn_interpretation,
    _value_label,
)


def _sample_result() -> BacktestResult:
    equity = pd.Series([10_000.0, 10_200.0, 10_100.0, 10_350.0], dtype="float64")
    returns = equity.pct_change().fillna(0.0)
    return BacktestResult(
        equity_curve=equity,
        returns=returns,
        positions=pd.Series([0, 1, 0, 1], dtype="int8"),
        trades=[],
        stats={"periods_per_year": 252.0, "cagr": 0.20, "max_drawdown": -0.10},
        data_quality={},
        execution_events=[],
        total_fees_paid=0.0,
        total_financing_paid=0.0,
        total_profit_before_fees=0.0,
    )


def test_compute_r_multiples_prefers_initial_risk() -> None:
    trades_df = pd.DataFrame(
        {
            "pnl": [100.0, -40.0, 60.0],
            "return_pct": [0.01, -0.004, 0.006],
            "initial_risk": [50.0, 20.0, 30.0],
        }
    )

    r_values = _compute_r_multiples(trades_df, initial_capital=10_000.0)

    assert r_values.tolist() == [2.0, -2.0, 2.0]


def test_robustness_diagnostics_contains_requested_sections() -> None:
    result = _sample_result()
    trades_df = pd.DataFrame(
        {
            "pnl": [120.0, -60.0, 80.0, -30.0, 140.0, -70.0, 90.0, -20.0, 110.0, -50.0],
            "return_pct": [0.012, -0.006, 0.008, -0.003, 0.014, -0.007, 0.009, -0.002, 0.011, -0.005],
        }
    )

    diagnostics = _build_robustness_diagnostics(result, trades_df)

    assert "sqn" in diagnostics
    assert diagnostics["sqn_interpretation"] in {"no edge", "weak", "tradable", "strong", "excellent"}
    assert diagnostics["removed_count"] == 1
    assert len(diagnostics["sign_flip_cagrs"]) == 100
    assert "first_half" in diagnostics and "second_half" in diagnostics
    assert diagnostics["classification"] in {"fragile", "moderate", "robust", "highly robust"}


def test_robustness_diagnostics_handles_tiny_cagr_baseline() -> None:
    result = _sample_result()
    result.stats["cagr"] = 1e-18
    trades_df = pd.DataFrame({"pnl": [0.1, -0.05, 0.08, -0.02], "return_pct": [0.01, -0.005, 0.008, -0.002]})

    diagnostics = _build_robustness_diagnostics(result, trades_df)

    assert pd.notna(diagnostics["cagr_drop_pct"])
    assert float(diagnostics["cagr_drop_pct"]) >= 0.0


def test_compute_trade_metrics_from_pnl_handles_very_large_notional_without_overflow() -> None:
    scale = 1e307
    pnl = pd.Series([scale, -0.5 * scale, scale], dtype="float64")

    metrics = _compute_trade_metrics_from_pnl(pnl, initial_capital=10.0 * scale, periods_per_year=252.0)

    assert pd.notna(metrics["max_drawdown"])
    assert metrics["profit_factor"] == 4.0


def test_sqn_interpretation_thresholds() -> None:
    assert _sqn_interpretation(1.5) == "no edge"
    assert _sqn_interpretation(1.8) == "weak"
    assert _sqn_interpretation(2.5) == "tradable"
    assert _sqn_interpretation(4.0) == "strong"
    assert _sqn_interpretation(5.1) == "excellent"


def test_large_value_labels_use_scientific_notation() -> None:
    assert _value_label(123.456) == "123.46"
    assert _value_label(12_345.0) == "12,345"
    assert _value_label(5.1847055e21) == "5.18e+21"


def test_percent_label_compacts_extreme_values() -> None:
    assert _percent_label(0.12345) == "12.35%"
    assert _percent_label(5.1847055e21) == "5.18e+23%"


def test_detect_log_growth_identifies_exponential_trend() -> None:
    assert _detect_log_growth([1.0, 2.0, 4.1, 8.0, 16.3, 32.4, 65.0, 128.0, 256.0])
    assert not _detect_log_growth([100.0, 101.0, 102.0, 101.5, 103.0, 102.5, 103.5, 104.0, 104.5])


def test_entry_exit_signal_rows_include_only_observed_combinations() -> None:
    trades_df = pd.DataFrame(
        {
            "entry_signal": ["Bearish 1W", "Bearish 1W", "Bullish 1W"],
            "exit_signal": ["Red Gator PP", "Red Gator PP", "Bearish 1W"],
            "pnl": [1000.0, -250.0, 400.0],
        }
    )

    rows = _entry_exit_signal_rows(trades_df)
    row_lookup = {(entry, exit_signal): (count, pnl) for entry, exit_signal, count, pnl in rows}

    assert row_lookup[("Bearish 1W", "Red Gator PP")] == (2, 750.0)
    assert row_lookup[("Bullish 1W", "Bearish 1W")] == (1, 400.0)
    assert ("Bullish 1W", "Green Gator PP") not in row_lookup


def test_simple_pdf_save_sanitizes_unicode_punctuation(tmp_path: Path) -> None:
    pdf = _SimplePdf()
    page = pdf.new_page()

    _add_text(page, 40, 560, "Runtime estimate: ~13s … ‘quoted’ — dash")

    out = tmp_path / "unicode-report.pdf"
    pdf.save(out)

    assert out.exists()
    assert out.stat().st_size > 0

