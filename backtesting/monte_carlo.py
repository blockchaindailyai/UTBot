from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import os
from textwrap import wrap

import numpy as np
import pandas as pd

from .stats import infer_periods_per_year


@dataclass(slots=True)
class MonteCarloResult:
    initial_capital: float
    baseline_final_equity: float
    simulations: int
    horizon_bars: int
    seed: int | None
    method: str
    equity_paths: pd.DataFrame
    summary: dict[str, float]


def _resolve_thread_count(requested_threads: int | None, simulations: int) -> int:
    if simulations <= 1:
        return 1
    if requested_threads is not None:
        return max(1, min(int(requested_threads), simulations))
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, simulations))


def _simulate_returns_chunk(
    sample: np.ndarray,
    horizon: int,
    block: int,
    chunk_sims: int,
    seed: int | None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sim_returns = np.zeros((chunk_sims, horizon), dtype="float64")

    if block == 1:
        idx = rng.integers(0, len(sample), size=(chunk_sims, horizon))
        sim_returns[:, :] = sample[idx]
        return sim_returns

    for i in range(chunk_sims):
        cursor = 0
        while cursor < horizon:
            start = int(rng.integers(0, len(sample)))
            end = min(start + block, len(sample))
            segment = sample[start:end]
            length = min(len(segment), horizon - cursor)
            sim_returns[i, cursor : cursor + length] = segment[:length]
            cursor += length
    return sim_returns


def _build_equity_paths_with_ruin(
    sim_returns: np.ndarray,
    initial_capital: float,
    equity_cutoff: float | None = None,
) -> np.ndarray:
    if sim_returns.size == 0:
        return np.zeros((0, 0), dtype="float64")
    sims, horizon = sim_returns.shape
    paths = np.zeros((sims, horizon), dtype="float64")
    cutoff = equity_cutoff if equity_cutoff is not None and equity_cutoff > 0 else None
    for i in range(sims):
        equity = float(initial_capital)
        finished = False
        for j in range(horizon):
            if finished:
                paths[i, j] = equity
                continue
            equity *= 1.0 + float(sim_returns[i, j])
            if equity <= 0.0:
                equity = 0.0
                finished = True
            elif cutoff is not None and equity <= cutoff:
                equity = cutoff
                finished = True
            paths[i, j] = equity
    return paths


def _max_drawdown_per_path(sim_returns: np.ndarray) -> np.ndarray:
    growth = np.cumprod(1.0 + sim_returns, axis=1)
    running_max = np.maximum.accumulate(growth, axis=1)
    drawdowns = (growth / running_max) - 1.0
    return drawdowns.min(axis=1)




def _longest_drawdown_duration(drawdown_row: np.ndarray) -> int:
    longest = 0
    current = 0
    for v in drawdown_row:
        if v < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _recovery_bars_after_max_drawdown(drawdown_row: np.ndarray) -> int:
    if drawdown_row.size == 0:
        return 0
    trough_idx = int(np.argmin(drawdown_row))
    if drawdown_row[trough_idx] >= 0:
        return 0
    recovered = np.where(drawdown_row[trough_idx:] >= -1e-12)[0]
    if recovered.size == 0:
        return int(drawdown_row.size - trough_idx - 1)
    return int(recovered[0])


def _path_drawdowns(equity_paths: np.ndarray) -> np.ndarray:
    running_max = np.maximum.accumulate(equity_paths, axis=1)
    ratio = np.divide(
        equity_paths,
        running_max,
        out=np.zeros_like(equity_paths, dtype="float64"),
        where=running_max > 0,
    )
    return ratio - 1.0


def _annualized_return(final_equity: np.ndarray, initial_capital: float, horizon_bars: int, periods_per_year: int = 252) -> np.ndarray:
    if horizon_bars <= 0:
        return np.zeros_like(final_equity)
    years = horizon_bars / periods_per_year
    if years <= 0:
        return np.zeros_like(final_equity)
    return np.power(np.maximum(final_equity / initial_capital, 1e-12), 1.0 / years) - 1.0


def _var_cvar(values: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha_pct = (1.0 - confidence) * 100.0
    var = float(np.percentile(values, alpha_pct))
    tail = values[values <= var]
    cvar = float(np.mean(tail)) if tail.size else var
    return var, cvar


def _baseline_returns_for_horizon(sample: np.ndarray, horizon: int) -> np.ndarray:
    if sample.size == 0 or horizon <= 0:
        return np.array([], dtype="float64")
    if horizon <= sample.size:
        return sample[:horizon]
    repeats = int(np.ceil(horizon / sample.size))
    return np.tile(sample, repeats)[:horizon]


def run_return_bootstrap_monte_carlo(
    returns: pd.Series,
    initial_capital: float,
    simulations: int = 1000,
    horizon_bars: int | None = None,
    seed: int | None = 42,
    block_size: int = 1,
    threads: int | None = None,
    equity_cutoff: float | None = None,
) -> MonteCarloResult:
    clean_returns = returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean_returns.empty:
        raise ValueError("Returns series is empty")
    if simulations <= 0:
        raise ValueError("simulations must be > 0")

    sample = clean_returns.to_numpy(copy=True)
    periods_per_year = infer_periods_per_year(clean_returns.index, default=252)
    horizon = int(horizon_bars or len(sample))
    horizon = max(1, horizon)
    block = max(1, int(block_size))

    worker_count = _resolve_thread_count(threads, simulations)

    if worker_count == 1:
        sim_returns = _simulate_returns_chunk(sample, horizon, block, simulations, seed)
    else:
        chunk_sizes = [simulations // worker_count] * worker_count
        for i in range(simulations % worker_count):
            chunk_sizes[i] += 1

        # Use SeedSequence to produce reproducible, independent child RNG streams.
        seed_sequence = np.random.SeedSequence(seed)
        child_seeds = seed_sequence.spawn(worker_count)

        chunks: list[np.ndarray] = []
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = []
            for i, size in enumerate(chunk_sizes):
                if size <= 0:
                    continue
                child_seed = int(child_seeds[i].generate_state(1, dtype=np.uint64)[0])
                futures.append(pool.submit(_simulate_returns_chunk, sample, horizon, block, size, child_seed))
            for f in futures:
                chunks.append(f.result())
        sim_returns = np.vstack(chunks) if chunks else np.zeros((0, horizon), dtype="float64")

    paths = _build_equity_paths_with_ruin(sim_returns, initial_capital=initial_capital, equity_cutoff=equity_cutoff)
    equity_paths = pd.DataFrame(paths)

    final_equity = equity_paths.iloc[:, -1].to_numpy() if not equity_paths.empty else np.array([initial_capital])
    total_return = (final_equity / initial_capital) - 1.0

    max_dds = _max_drawdown_per_path(sim_returns) if sim_returns.size else np.array([0.0], dtype="float64")
    drawdown_paths = _path_drawdowns(paths)
    dd_durations = np.asarray([_longest_drawdown_duration(row) for row in drawdown_paths], dtype="float64")
    dd_recovery_bars = np.asarray([_recovery_bars_after_max_drawdown(row) for row in drawdown_paths], dtype="float64")
    mean_r = float(np.mean(sim_returns)) if sim_returns.size else 0.0
    std_r = float(np.std(sim_returns, ddof=0)) if sim_returns.size else 0.0
    downside = np.minimum(sim_returns, 0.0) if sim_returns.size else np.array([0.0], dtype="float64")
    downside_dev = float(np.sqrt(np.mean(np.square(downside)))) if sim_returns.size else 0.0
    sharpe = 0.0 if std_r == 0 else (mean_r / std_r) * np.sqrt(periods_per_year)
    sortino = 0.0 if downside_dev == 0 else (mean_r / downside_dev) * np.sqrt(periods_per_year)
    cagr_dist = _annualized_return(final_equity, initial_capital, horizon, periods_per_year)
    avg_cagr = float(np.mean(cagr_dist)) if cagr_dist.size else 0.0
    med_cagr = float(np.median(cagr_dist)) if cagr_dist.size else 0.0
    calmar = avg_cagr / abs(float(np.mean(max_dds))) if float(np.mean(max_dds)) != 0 else 0.0
    mar = med_cagr / abs(float(np.median(max_dds))) if float(np.median(max_dds)) != 0 else 0.0

    baseline_horizon_returns = _baseline_returns_for_horizon(sample, horizon)
    baseline_final = float(initial_capital * np.cumprod(1.0 + baseline_horizon_returns)[-1])

    return_var_90, cvar_90 = _var_cvar(total_return, 0.90)
    return_var_95, cvar_95 = _var_cvar(total_return, 0.95)
    return_var_99, cvar_99 = _var_cvar(total_return, 0.99)

    min_equity = paths.min(axis=1) if paths.size else np.array([initial_capital], dtype="float64")
    ulcer_per_path = np.sqrt(np.mean(np.square(np.minimum(drawdown_paths, 0.0)), axis=1)) if drawdown_paths.size else np.array([0.0], dtype="float64")

    summary = {
        "threads_used": float(worker_count),
        "simulations": float(simulations),
        "horizon_bars": float(horizon),
        "periods_per_year": float(periods_per_year),
        "initial_capital": float(initial_capital),
        "final_equity_p5": float(np.percentile(final_equity, 5)),
        "final_equity_p1": float(np.percentile(final_equity, 1)),
        "final_equity_p25": float(np.percentile(final_equity, 25)),
        "final_equity_p50": float(np.percentile(final_equity, 50)),
        "final_equity_p75": float(np.percentile(final_equity, 75)),
        "final_equity_p95": float(np.percentile(final_equity, 95)),
        "final_equity_p99": float(np.percentile(final_equity, 99)),
        "final_equity_mean": float(np.mean(final_equity)),
        "final_equity_median": float(np.median(final_equity)),
        "return_p5": return_var_95,
        "return_p1": float(np.percentile(total_return, 1)),
        "return_p25": float(np.percentile(total_return, 25)),
        "return_p50": float(np.percentile(total_return, 50)),
        "return_p75": float(np.percentile(total_return, 75)),
        "return_p95": float(np.percentile(total_return, 95)),
        "return_p99": float(np.percentile(total_return, 99)),
        "probability_profit": float(np.mean(final_equity > initial_capital)),
        "probability_loss_over_10pct": float(np.mean(total_return <= -0.10)),
        "probability_loss_over_20pct": float(np.mean(total_return <= -0.20)),
        "probability_loss_over_30pct": float(np.mean(total_return <= -0.30)),
        "probability_loss_over_50pct": float(np.mean(total_return <= -0.50)),
        "probability_return_over_25pct": float(np.mean(total_return >= 0.25)),
        "probability_return_over_50pct": float(np.mean(total_return >= 0.50)),
        "probability_return_over_100pct": float(np.mean(total_return >= 1.00)),
        "probability_return_over_200pct": float(np.mean(total_return >= 2.00)),
        "probability_return_over_500pct": float(np.mean(total_return >= 5.00)),
        "probability_return_over_1000pct": float(np.mean(total_return >= 10.00)),
        "expected_final_equity": float(np.mean(final_equity)),
        "expected_return": float(np.mean(total_return)),
        "median_return": float(np.median(total_return)),
        "return_mean": float(np.mean(total_return)),
        "return_median": float(np.median(total_return)),
        "return_std": float(np.std(total_return, ddof=0)),
        "best_return": float(np.max(total_return)),
        "worst_return": float(np.min(total_return)),
        "final_equity_std": float(np.std(final_equity, ddof=0)),
        "cagr_mean": avg_cagr,
        "cagr_median": med_cagr,
        "cagr_std": float(np.std(cagr_dist, ddof=0)),
        "cagr_p5": float(np.percentile(cagr_dist, 5)),
        "cagr_p50": float(np.percentile(cagr_dist, 50)),
        "cagr_p95": float(np.percentile(cagr_dist, 95)),
        "expected_max_drawdown": float(np.mean(max_dds)),
        "max_drawdown_mean": float(np.mean(max_dds)),
        "max_drawdown_p50": float(np.percentile(max_dds, 50)),
        "max_drawdown_median": float(np.percentile(max_dds, 50)),
        "max_drawdown_worst": float(np.min(max_dds)),
        "max_drawdown_p95": float(np.percentile(max_dds, 95)),
        "max_drawdown_p95_worst": float(np.percentile(max_dds, 5)),
        "drawdown_duration_mean": float(np.mean(dd_durations)),
        "drawdown_duration_median": float(np.median(dd_durations)),
        "drawdown_duration_worst": float(np.max(dd_durations)),
        "recovery_bars_mean": float(np.mean(dd_recovery_bars)),
        "recovery_bars_median": float(np.median(dd_recovery_bars)),
        "recovery_bars_worst": float(np.max(dd_recovery_bars)),
        "ulcer_index": float(np.mean(ulcer_per_path)),
        "var_95": return_var_95,
        "cvar_95": cvar_95,
        "var_90": return_var_90,
        "cvar_90": cvar_90,
        "var_99": return_var_99,
        "cvar_99": cvar_99,
        "probability_drawdown_worse_than_20pct": float(np.mean(max_dds <= -0.20)),
        "probability_drawdown_worse_than_30pct": float(np.mean(max_dds <= -0.30)),
        "probability_return_below_baseline": float(np.mean(final_equity < baseline_final)),
        "probability_ruin_75pct": float(np.mean(min_equity <= 0.75 * initial_capital)),
        "probability_ruin_50pct": float(np.mean(min_equity <= 0.50 * initial_capital)),
        "probability_ruin_25pct": float(np.mean(min_equity <= 0.25 * initial_capital)),
        "probability_ruin_10pct": float(np.mean(min_equity <= 0.10 * initial_capital)),
        "mean_bar_return": mean_r,
        "bar_return_volatility": std_r,
        "downside_deviation": downside_dev,
        "approx_sharpe": sharpe,
        "approx_sortino": sortino,
        "approx_calmar": calmar,
        "approx_mar": mar,
        "return_over_max_drawdown": float(np.mean(total_return)) / abs(float(np.mean(max_dds))) if float(np.mean(max_dds)) != 0 else 0.0,
        "median_return_over_median_drawdown": float(np.median(total_return)) / abs(float(np.median(max_dds))) if float(np.median(max_dds)) != 0 else 0.0,
        "return_skew": float(pd.Series(total_return).skew()),
        "return_kurtosis": float(pd.Series(total_return).kurtosis()),
    }

    return MonteCarloResult(
        initial_capital=float(initial_capital),
        baseline_final_equity=baseline_final,
        simulations=simulations,
        horizon_bars=horizon,
        seed=seed,
        method=("bootstrap_returns" if block == 1 else f"block_bootstrap_{block}") + f"_threads_{worker_count}",
        equity_paths=equity_paths,
        summary=summary,
    )


def _compute_monte_carlo_analytics(result: MonteCarloResult) -> dict[str, object]:
    equity = result.equity_paths.astype(float)
    equity_np = equity.to_numpy(dtype="float64")
    if equity_np.size == 0:
        equity_np = np.full((1, 1), result.initial_capital, dtype="float64")
    final_equity = equity_np[:, -1]
    total_return = final_equity / result.initial_capital - 1.0
    periods_per_year = int(result.summary.get("periods_per_year", 252))
    cagr = _annualized_return(final_equity, result.initial_capital, result.horizon_bars, periods_per_year=periods_per_year)

    drawdown_paths = _path_drawdowns(equity_np)
    max_drawdown = drawdown_paths.min(axis=1)

    # Path-level risk adjusted metrics
    path_sharpe = []
    path_sortino = []
    for row in equity_np:
        rets = row[1:] / np.maximum(row[:-1], 1e-12) - 1.0 if len(row) > 1 else np.array([], dtype="float64")
        std = float(np.std(rets, ddof=0)) if rets.size else 0.0
        downside = np.minimum(rets, 0.0) if rets.size else np.array([], dtype="float64")
        downside_dev = float(np.sqrt(np.mean(np.square(downside)))) if downside.size else 0.0
        mean_r = float(np.mean(rets)) if rets.size else 0.0
        path_sharpe.append(0.0 if std == 0 else (mean_r / std) * np.sqrt(periods_per_year))
        path_sortino.append(0.0 if downside_dev == 0 else (mean_r / downside_dev) * np.sqrt(periods_per_year))

    pcts = [1, 5, 25, 50, 75, 95, 99]
    checkpoints = sorted(set([max(1, int(result.horizon_bars * r)) for r in [0.1, 0.25, 0.5, 0.75, 1.0]]))

    def probs_at_checkpoints() -> list[dict[str, float]]:
        out = []
        for cp in checkpoints:
            idx = min(cp - 1, equity_np.shape[1] - 1)
            vals = equity_np[:, idx]
            out.append(
                {
                    "bar": float(cp),
                    "profit_prob": float(np.mean(vals > result.initial_capital)),
                    "below_start_prob": float(np.mean(vals < result.initial_capital)),
                }
            )
        return out

    def time_to_target(mult: float) -> np.ndarray:
        target = result.initial_capital * mult
        times = np.full(equity_np.shape[0], np.nan, dtype="float64")
        for i, row in enumerate(equity_np):
            hit = np.where(row >= target)[0]
            if hit.size:
                times[i] = float(hit[0] + 1)
        return times

    def pct_table(values: np.ndarray, name: str) -> list[tuple[str, float]]:
        return [(f"p{p}", float(np.percentile(values, p))) for p in pcts] + [("mean", float(np.mean(values))), ("median", float(np.median(values))), ("std", float(np.std(values, ddof=0)))]

    shuffled_drawdowns = []
    for row in equity_np:
        rets = row[1:] / np.maximum(row[:-1], 1e-12) - 1.0 if len(row) > 1 else np.array([], dtype="float64")
        shuffled = np.random.default_rng(0).permutation(rets) if rets.size else rets
        dd = _max_drawdown_per_path(shuffled.reshape(1, -1))[0]
        shuffled_drawdowns.append(dd)

    summary = result.summary
    flags = []
    if abs(float(summary.get("return_skew", 0.0))) > 1.5:
        flags.append("Extreme skew detected: outcomes may be dominated by asymmetric tails.")
    if float(summary.get("return_kurtosis", 0.0)) > 5.0:
        flags.append("High kurtosis detected: fat-tail risk is elevated.")
    if float(summary.get("return_median", 0.0)) < 0.5 * float(summary.get("return_mean", 0.0)):
        flags.append("Median outcome is materially weaker than mean; outliers may inflate averages.")
    if float(summary.get("probability_return_below_baseline", 0.0)) > 0.6:
        flags.append("Most simulations underperform baseline run; baseline may be unrepresentative.")
    if float(summary.get("max_drawdown_p95_worst", 0.0)) < -0.40:
        flags.append("Worst 5% drawdowns exceed 40%, likely unacceptable for many mandates.")
    if float(summary.get("expected_return", 0.0)) > 0.25 and float(summary.get("approx_sharpe", 0.0)) < 0.75:
        flags.append("High return paired with low Sharpe: returns appear risk-inefficient.")

    return {
        "final_equity": final_equity,
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "drawdown_paths": drawdown_paths,
        "path_sharpe": np.asarray(path_sharpe, dtype="float64"),
        "path_sortino": np.asarray(path_sortino, dtype="float64"),
        "equity_percentiles": {p: np.percentile(equity_np, p, axis=0) for p in [5, 25, 50, 75, 95]},
        "equity_checkpoints": probs_at_checkpoints(),
        "dd_percentiles": {p: np.percentile(drawdown_paths, p, axis=0) for p in [5, 25, 50, 75, 95]},
        "time_to_double": time_to_target(2.0),
        "time_to_5x": time_to_target(5.0),
        "time_to_10x": time_to_target(10.0),
        "table_final_equity": pct_table(final_equity, "final_equity"),
        "table_total_return": pct_table(total_return, "total_return"),
        "table_max_drawdown": pct_table(max_drawdown, "max_drawdown"),
        "table_cagr": pct_table(cagr, "cagr"),
        "scenario_shuffled_drawdown_mean": float(np.mean(shuffled_drawdowns)),
        "flags": flags,
    }


def generate_monte_carlo_pdf_report(
    result: MonteCarloResult,
    output_path: str | Path,
    title: str = "Monte Carlo Strategy Report",
    cli_flags: dict[str, object] | None = None,
    csv_source: str | None = None,
    baseline_trade_count: int | None = None,
    trade_diagnostics: dict[str, float] | None = None,
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    analytics = _compute_monte_carlo_analytics(result)

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def fmt_pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    def fmt_num(value: float) -> str:
        return f"{value:,.2f}"

    def pct_points(values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype="float64") * 100.0

    class PdfBuilder:
        def __init__(self) -> None:
            self.pages: list[list[str]] = []

        def new_page(self) -> list[str]:
            page: list[str] = []
            self.pages.append(page)
            return page

        def save(self, path: Path) -> None:
            objects: list[bytes] = []

            def add_obj(data: str | bytes) -> int:
                body = data.encode("latin-1") if isinstance(data, str) else data
                objects.append(body)
                return len(objects)

            font_id = add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
            page_ids: list[int] = []
            for page in self.pages:
                stream = "\n".join(page).encode("latin-1")
                content_id = add_obj(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")
                page_ids.append(add_obj(f"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 842 595] /Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"))
            pages_id = add_obj(f"<< /Type /Pages /Count {len(page_ids)} /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] >>")
            for page_id in page_ids:
                objects[page_id - 1] = objects[page_id - 1].replace(b"/Parent 0 0 R", f"/Parent {pages_id} 0 R".encode("ascii"))
            catalog_id = add_obj(f"<< /Type /Catalog /Pages {pages_id} 0 R >>")

            data = bytearray(b"%PDF-1.4\n")
            offsets = [0]
            for i, obj in enumerate(objects, start=1):
                offsets.append(len(data))
                data.extend(f"{i} 0 obj\n".encode("ascii"))
                data.extend(obj)
                data.extend(b"\nendobj\n")
            xref_off = len(data)
            data.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
            data.extend(b"0000000000 65535 f \n")
            for off in offsets[1:]:
                data.extend(f"{off:010d} 00000 n \n".encode("ascii"))
            data.extend(f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref_off}\n%%EOF\n".encode("ascii"))
            path.write_bytes(data)

    def t(page: list[str], x: float, y: float, text: str, size: int = 10) -> None:
        page.append(f"BT /F1 {size} Tf {x:.1f} {y:.1f} Td ({esc(text)}) Tj ET")

    def fmt_axis_value(value: float) -> str:
        abs_v = abs(value)
        if abs_v >= 1000:
            return f"{value:,.0f}"
        if abs_v >= 10:
            return f"{value:,.1f}"
        return f"{value:,.2f}"

    def _chart_inner_rect(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
        left_pad = 64.0
        right_pad = 10.0
        top_pad = 36.0
        bottom_pad = 34.0
        ix0 = x + left_pad
        iy0 = y + bottom_pad
        iw = max(w - left_pad - right_pad, 10.0)
        ih = max(h - top_pad - bottom_pad, 10.0)
        return ix0, iy0, iw, ih

    def draw_axes(
        page: list[str],
        x: float,
        y: float,
        w: float,
        h: float,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        x_label: str,
        y_label: str,
        x_ticks: int = 5,
        y_ticks: int = 7,
        y_tick_positions: list[tuple[float, str]] | None = None,
    ) -> tuple[float, float, float, float]:
        ix0, iy0, iw, ih = _chart_inner_rect(x, y, w, h)

        page.append("0.15 w 0.85 0.85 0.85 RG")
        for i in range(x_ticks):
            gx = ix0 + (i / max(x_ticks - 1, 1)) * iw
            page.append(f"{gx:.2f} {iy0:.2f} m {gx:.2f} {iy0 + ih:.2f} l S")

        if y_tick_positions is None:
            y_tick_positions = []
            for i in range(y_ticks):
                ratio = i / max(y_ticks - 1, 1)
                tick_val = ymin + ratio * ((ymax - ymin) or 1.0)
                y_tick_positions.append((ratio, fmt_axis_value(tick_val)))

        for ratio, _ in y_tick_positions:
            gy = iy0 + float(ratio) * ih
            page.append(f"{ix0:.2f} {gy:.2f} m {ix0 + iw:.2f} {gy:.2f} l S")

        page.append("0.3 w 0 0 0 RG")
        page.append(f"{ix0:.2f} {iy0:.2f} m {ix0 + iw:.2f} {iy0:.2f} l S")
        page.append(f"{ix0:.2f} {iy0:.2f} m {ix0:.2f} {iy0 + ih:.2f} l S")

        xspan = (xmax - xmin) or 1.0
        for i in range(x_ticks):
            ratio = i / max(x_ticks - 1, 1)
            tick_x = ix0 + ratio * iw
            tick_val = xmin + ratio * xspan
            t(page, tick_x - 10, iy0 - 14, fmt_axis_value(tick_val), 6)

        for ratio, label in y_tick_positions:
            tick_y = iy0 + float(ratio) * ih
            t(page, x + 2, tick_y - 2, label, 6)

        t(page, ix0 + iw * 0.35, y + 7, x_label, 7)
        t(page, x + 4, iy0 + ih + 8, y_label, 7)
        return ix0, iy0, iw, ih

    def box(page: list[str], x: float, y: float, w: float, h: float, title_text: str, footnote: str, legend: list[str] | None = None) -> None:
        page.append("0.2 w 0 0 0 RG")
        page.append(f"{x:.1f} {y:.1f} {w:.1f} {h:.1f} re S")
        if title_text:
            t(page, x + 4, y + h - 14, title_text, 10)

        if legend:
            legend_y = y + h + 14
            for line in legend:
                t(page, x + 4, legend_y, line[:86], 8)
                legend_y += 10

        note_lines = wrap(f"Note: {footnote}", width=max(52, int(w / 7)))[:2] if footnote else []
        note_y = y - 12
        for line in note_lines:
            t(page, x + 4, note_y, line, 7)
            note_y -= 9

    def _detect_log_growth(values: np.ndarray) -> bool:
        arr = np.asarray(values, dtype="float64")
        arr = arr[np.isfinite(arr)]
        if arr.size < 8 or np.any(arr <= 0):
            return False
        q10 = float(np.quantile(arr, 0.10))
        q90 = float(np.quantile(arr, 0.90))
        if q10 <= 0 or (q90 / q10) < 8.0:
            return False
        idx = np.arange(arr.size, dtype="float64")
        linear_corr = abs(float(np.corrcoef(idx, arr)[0, 1]))
        log_corr = abs(float(np.corrcoef(idx, np.log(arr))[0, 1]))
        return bool(np.isfinite(log_corr) and log_corr > linear_corr * 1.05)

    def draw_line(
        page: list[str],
        values: np.ndarray,
        x: float,
        y: float,
        w: float,
        h: float,
        title_text: str,
        color: tuple[float, float, float],
        x_label: str,
        y_label: str,
        footnote: str,
        decorate: bool = True,
        legend: list[str] | None = None,
        axis_bounds: tuple[float, float] | None = None,
        draw_axes_grid: bool = True,
        log_scale: bool | None = None,
    ) -> None:
        if decorate:
            box(page, x, y, w, h, title_text, footnote, legend=legend)
        if len(values) < 2:
            return

        src_values = np.asarray(values, dtype="float64")
        if axis_bounds is None:
            raw_min = float(np.min(src_values))
            raw_max = float(np.max(src_values))
        else:
            raw_min, raw_max = axis_bounds

        if log_scale is None:
            log_scale = _detect_log_growth(src_values)

        y_tick_positions: list[tuple[float, str]] | None = None
        if log_scale:
            safe_values = np.maximum(src_values, 1e-12)
            safe_min = max(raw_min, 1e-12)
            safe_max = max(raw_max, safe_min * 1.0000001)
            values_plot = np.log10(safe_values)
            vmin = float(np.log10(safe_min))
            vmax = float(np.log10(safe_max))
            y_tick_count = 7
            y_tick_positions = []
            for i in range(y_tick_count):
                ratio = i / max(y_tick_count - 1, 1)
                tick_val = 10 ** (vmin + ratio * ((vmax - vmin) or 1.0))
                y_tick_positions.append((ratio, fmt_axis_value(float(tick_val))))
        else:
            values_plot = src_values
            vmin = raw_min
            vmax = raw_max

        span = (vmax - vmin) or 1.0
        y_tick_count = max(5, min(10, int(max(h - 80, 80) // 26)))
        if draw_axes_grid:
            ix0, iy0, iw, ih = draw_axes(
                page,
                x,
                y,
                w,
                h,
                1.0,
                float(len(values_plot)),
                vmin,
                vmax,
                x_label,
                y_label,
                y_ticks=y_tick_count,
                y_tick_positions=y_tick_positions,
            )
        else:
            ix0, iy0, iw, ih = _chart_inner_rect(x, y, w, h)

        page.append(f"{color[0]:.2f} {color[1]:.2f} {color[2]:.2f} RG 1.0 w")
        for i, v in enumerate(values_plot):
            px = ix0 + (i / (len(values_plot) - 1)) * iw
            py = iy0 + ((float(v) - vmin) / span) * ih
            page.append(f"{px:.2f} {py:.2f} {'m' if i == 0 else 'l'}")
        page.append("S")

    def draw_hist(
        page: list[str],
        values: np.ndarray,
        x: float,
        y: float,
        w: float,
        h: float,
        title_text: str,
        x_label: str,
        y_label: str,
        footnote: str,
        bins: int = 20,
        clip_high_quantile: float | None = None,
    ) -> None:
        box(page, x, y, w, h, title_text, footnote)
        if values.size == 0:
            return
        plotted_values = values
        if clip_high_quantile is not None:
            q = float(np.clip(clip_high_quantile, 0.5, 1.0))
            high = float(np.quantile(values, q))
            clipped = values[values <= high]
            if clipped.size:
                plotted_values = clipped

        counts, edges = np.histogram(plotted_values, bins=min(bins, max(8, int(np.sqrt(plotted_values.size)))))
        cmax = max(int(np.max(counts)), 1)
        ix0, iy0, iw, ih = draw_axes(page, x, y, w, h, float(edges[0]), float(edges[-1]), 0.0, float(cmax), x_label, y_label)
        page.append("0.2 0.45 0.75 rg")
        for i, c in enumerate(counts):
            bw = iw / len(counts)
            bh = (float(c) / cmax) * ih
            px = ix0 + i * bw
            page.append(f"{px:.2f} {iy0:.2f} {max(bw - 1,0.5):.2f} {bh:.2f} re f")
        page.append("0 g")

    def draw_scatter(
        page: list[str],
        x_values: np.ndarray,
        y_values: np.ndarray,
        x: float,
        y: float,
        w: float,
        h: float,
        title_text: str,
        x_label: str,
        y_label: str,
        footnote: str,
        x_clip_high_quantile: float | None = None,
    ) -> None:
        box(page, x, y, w, h, title_text, footnote)
        if x_values.size == 0:
            return
        plotted_x = x_values
        plotted_y = y_values
        if x_clip_high_quantile is not None:
            q = float(np.clip(x_clip_high_quantile, 0.5, 1.0))
            high = float(np.quantile(x_values, q))
            mask = x_values <= high
            if np.any(mask):
                plotted_x = x_values[mask]
                plotted_y = y_values[mask]

        xmin, xmax = float(np.min(plotted_x)), float(np.max(plotted_x))
        ymin, ymax = float(np.min(plotted_y)), float(np.max(plotted_y))
        xspan = (xmax - xmin) or 1.0
        yspan = (ymax - ymin) or 1.0
        ix0, iy0, iw, ih = draw_axes(page, x, y, w, h, xmin, xmax, ymin, ymax, x_label, y_label)
        page.append("0.1 0.45 0.1 rg")
        step = max(1, int(len(plotted_x) / 600))
        for xv, yv in zip(plotted_x[::step], plotted_y[::step]):
            px = ix0 + ((float(xv) - xmin) / xspan) * iw
            py = iy0 + ((float(yv) - ymin) / yspan) * ih
            page.append(f"{px:.2f} {py:.2f} 1.5 1.5 re f")
        page.append("0 g")

    pdf = PdfBuilder()

    # Page 1: Executive summary
    p1 = pdf.new_page()
    t(p1, 40, 565, title, 18)
    t(p1, 40, 548, "Executive Summary", 14)
    t(p1, 40, 532, f"Strategy / method: {result.method}", 10)
    if csv_source:
        t(p1, 40, 518, f"CSV source: {csv_source}", 10)
        stats_start_y = 504
    else:
        stats_start_y = 518
    t(p1, 40, stats_start_y, f"Simulations: {result.simulations} | Horizon bars: {result.horizon_bars} | Initial capital: ${fmt_num(result.initial_capital)}", 10)
    t(p1, 40, stats_start_y - 14, f"Baseline final equity: ${fmt_num(result.baseline_final_equity)} | Expected: ${fmt_num(result.summary['expected_final_equity'])} | Median: ${fmt_num(result.summary['final_equity_p50'])}", 10)
    y_offset = 28
    if baseline_trade_count is not None:
        baseline_bars = int(result.summary.get("baseline_bars", 0))
        projected_trades = result.summary.get("projected_trade_count_at_baseline_cadence")
        t(p1, 40, stats_start_y - 28, f"Baseline trades: {baseline_trade_count}", 10)
        y_offset = 42
        if baseline_bars > 0:
            t(p1, 40, stats_start_y - 42, f"Baseline bars: {baseline_bars}", 10)
            y_offset = 56
        if isinstance(projected_trades, (int, float)) and projected_trades > 0:
            t(p1, 40, stats_start_y - y_offset, f"Projected trades at baseline cadence: {projected_trades:,.1f}", 10)
            y_offset += 14
    t(p1, 40, stats_start_y - y_offset, f"CAGR (mean/median): {fmt_pct(result.summary['cagr_mean'])} / {fmt_pct(result.summary['cagr_median'])}", 10)
    t(p1, 40, stats_start_y - (y_offset + 14), f"Max drawdown (mean/median/worst): {fmt_pct(result.summary['max_drawdown_mean'])} / {fmt_pct(result.summary['max_drawdown_median'])} / {fmt_pct(result.summary['max_drawdown_worst'])}", 10)
    t(p1, 40, stats_start_y - (y_offset + 28), f"Sharpe/Sortino/Calmar/MAR: {result.summary['approx_sharpe']:.2f} / {result.summary['approx_sortino']:.2f} / {result.summary['approx_calmar']:.2f} / {result.summary['approx_mar']:.2f}", 10)
    p1_cone_values = np.vstack([
        np.asarray(analytics["equity_percentiles"][5], dtype="float64"),
        np.asarray(analytics["equity_percentiles"][50], dtype="float64"),
        np.asarray(analytics["equity_percentiles"][95], dtype="float64"),
    ])
    p1_cone_bounds = (float(np.min(p1_cone_values)), float(np.max(p1_cone_values)))
    draw_line(p1, analytics["equity_percentiles"][50], 40, 70, 760, 300, "Equity cone (p5/p50/p95)", (0.1, 0.35, 0.8), "Simulation bar", "Equity ($)", "Median trajectory with downside/upside percentile bands.", axis_bounds=p1_cone_bounds, log_scale=True)
    draw_line(p1, analytics["equity_percentiles"][5], 40, 70, 760, 300, "", (0.8, 0.2, 0.2), "Simulation bar", "Equity ($)", "", decorate=False, axis_bounds=p1_cone_bounds, draw_axes_grid=False, log_scale=True)
    draw_line(p1, analytics["equity_percentiles"][95], 40, 70, 760, 300, "", (0.8, 0.2, 0.2), "Simulation bar", "Equity ($)", "", decorate=False, axis_bounds=p1_cone_bounds, draw_axes_grid=False, log_scale=True)

    # Page 2: Performance metrics + distribution diagnostics
    p2 = pdf.new_page()
    t(p2, 40, 565, "Performance Metrics & Distribution Diagnostics", 15)
    t(p2, 40, 548, "Percentile tables quantify realistic ranges, not just central tendencies.", 10)
    y = 528
    for label, val in analytics["table_final_equity"]:
        t(p2, 40, y, f"Final equity {label}: ${fmt_num(val)}", 9)
        y -= 12
        if y < 380:
            break
    y = 528
    for label, val in analytics["table_total_return"]:
        t(p2, 300, y, f"Total return {label}: {fmt_pct(val)}", 9)
        y -= 12
        if y < 380:
            break
    y = 528
    for label, val in analytics["table_cagr"]:
        t(p2, 560, y, f"CAGR {label}: {fmt_pct(val)}", 9)
        y -= 12
        if y < 380:
            break
    t(p2, 40, 360, f"Skewness: {result.summary['return_skew']:.2f} | Kurtosis: {result.summary['return_kurtosis']:.2f}", 10)
    t(p2, 40, 346, "Interpretation: large skew/kurtosis indicates unstable tails and potential outlier dependence.", 9)
    t(p2, 40, 332, f"Mean vs median return: {fmt_pct(result.summary['return_mean'])} vs {fmt_pct(result.summary['return_median'])}", 9)
    if trade_diagnostics:
        t(p2, 430, 360, "Baseline position/trade diagnostics", 10)
        t(p2, 430, 346, f"Mean position size (USD): {fmt_num(float(trade_diagnostics.get('mean_position_size_usd', 0.0)))}", 9)
        t(p2, 430, 334, f"Median position size (USD): {fmt_num(float(trade_diagnostics.get('median_position_size_usd', 0.0)))}", 9)
        t(p2, 430, 322, f"Mean trade PnL (USD/%): {fmt_num(float(trade_diagnostics.get('mean_trade_pnl_usd', 0.0)))} / {fmt_pct(float(trade_diagnostics.get('mean_trade_pnl_pct', 0.0)))}", 9)
        t(p2, 430, 310, f"Median trade PnL (USD/%): {fmt_num(float(trade_diagnostics.get('median_trade_pnl_usd', 0.0)))} / {fmt_pct(float(trade_diagnostics.get('median_trade_pnl_pct', 0.0)))}", 9)
        t(p2, 430, 298, f"Total slippage paid (est): {fmt_num(float(trade_diagnostics.get('total_slippage_paid', 0.0)))}", 9)
        t(p2, 430, 286, f"Fees per trade: {fmt_num(float(trade_diagnostics.get('fees_per_trade', 0.0)))}", 9)
        t(p2, 430, 274, f"Slippage per trade (est): {fmt_num(float(trade_diagnostics.get('slippage_per_trade', 0.0)))}", 9)
    draw_hist(p2, pct_points(analytics["total_return"]), 40, 80, 240, 220, "Final return histogram", "Total return (%)", "Simulation count", "Distribution of terminal returns across all Monte Carlo runs. Extreme right tail is trimmed at the 95th percentile for readability.", clip_high_quantile=0.95)
    draw_hist(p2, analytics["final_equity"], 300, 80, 240, 220, "Final equity histogram", "Final equity ($)", "Simulation count", "Distribution of ending portfolio values to frame capital outcome ranges. Extreme right tail is trimmed at the 95th percentile for readability.", clip_high_quantile=0.95)
    draw_hist(p2, pct_points(analytics["cagr"]), 560, 80, 240, 220, "CAGR histogram", "Annualized return (%)", "Simulation count", "Annualized return dispersion using inferred bar frequency for accurate compounding horizon. Extreme right tail is trimmed at the 95th percentile for readability.", clip_high_quantile=0.95)

    # Page 3: Volume/fee/slippage turnover diagnostics
    p_turn = pdf.new_page()
    t(p_turn, 40, 565, "Volume / Fees / Slippage Turnover", 15)
    if trade_diagnostics:
        lines = [
            f"Total cumulative volume ($): {fmt_num(float(trade_diagnostics.get('total_cumulative_volume', 0.0)))}",
            f"Total cumulative fees ($): {fmt_num(float(trade_diagnostics.get('total_cumulative_fees', 0.0)))}",
            f"Total cumulative slippage ($ est): {fmt_num(float(trade_diagnostics.get('total_cumulative_slippage', 0.0)))}",
            f"Mean/median volume per trade ($): {fmt_num(float(trade_diagnostics.get('mean_volume_per_trade', 0.0)))} / {fmt_num(float(trade_diagnostics.get('median_volume_per_trade', 0.0)))}",
            f"Mean/median fee per trade ($): {fmt_num(float(trade_diagnostics.get('mean_fee_per_trade', 0.0)))} / {fmt_num(float(trade_diagnostics.get('median_fee_per_trade', 0.0)))}",
            f"Mean/median slippage per trade ($ est): {fmt_num(float(trade_diagnostics.get('mean_slippage_per_trade', 0.0)))} / {fmt_num(float(trade_diagnostics.get('median_slippage_per_trade', 0.0)))}",
        ]
    else:
        lines = ["Trade diagnostics unavailable for this Monte Carlo input."]
    yy = 544
    for line in lines:
        t(p_turn, 40, yy, line, 10)
        yy -= 14

    trade_count_proxy = int(baseline_trade_count or 0)
    if trade_diagnostics and trade_count_proxy > 0:
        total_volume = float(trade_diagnostics.get('total_cumulative_volume', 0.0))
        total_fees = float(trade_diagnostics.get('total_cumulative_fees', 0.0))
        total_slippage = float(trade_diagnostics.get('total_cumulative_slippage', 0.0))

        x = np.arange(1, trade_count_proxy + 1, dtype='float64')
        cumulative_abs_volume = (x / max(trade_count_proxy, 1)) * total_volume
        cumulative_fees = (x / max(trade_count_proxy, 1)) * total_fees
        cumulative_slippage = (x / max(trade_count_proxy, 1)) * total_slippage

        draw_line(p_turn, cumulative_abs_volume.tolist(), 40, 300, 760, 150, "Cumulative turnover by trade", (0.10, 0.35, 0.8), "Trade #", "$", "Baseline-trade-indexed turnover trajectory using aggregate diagnostics.")
        draw_line(p_turn, cumulative_fees.tolist(), 40, 130, 370, 140, "Cumulative fees", (0.8, 0.45, 0.1), "Trade #", "$", "Fee path distributed over baseline trade count.")
        draw_line(p_turn, cumulative_slippage.tolist(), 430, 130, 370, 140, "Cumulative slippage", (0.8, 0.15, 0.15), "Trade #", "$", "Slippage path distributed over baseline trade count.")
    else:
        t(p_turn, 40, 300, "No baseline trade count available for turnover charts.", 10)

    # Page 4: Tail risk and downside
    p3 = pdf.new_page()
    t(p3, 40, 565, "Downside / Tail-Risk Analysis", 15)
    t(p3, 40, 548, "Tail diagnostics measure capital impairment risk and drawdown tolerability.", 10)
    risk_lines = [
        f"VaR 90/95/99: {fmt_pct(result.summary['var_90'])} / {fmt_pct(result.summary['var_95'])} / {fmt_pct(result.summary['var_99'])}",
        f"CVaR 90/95/99: {fmt_pct(result.summary['cvar_90'])} / {fmt_pct(result.summary['cvar_95'])} / {fmt_pct(result.summary['cvar_99'])}",
        f"P(loss >10/20/30/50%): {fmt_pct(result.summary['probability_loss_over_10pct'])} / {fmt_pct(result.summary['probability_loss_over_20pct'])} / {fmt_pct(result.summary['probability_loss_over_30pct'])} / {fmt_pct(result.summary['probability_loss_over_50pct'])}",
        f"Risk of ruin (equity below 75/50/25/10%): {fmt_pct(result.summary['probability_ruin_75pct'])} / {fmt_pct(result.summary['probability_ruin_50pct'])} / {fmt_pct(result.summary['probability_ruin_25pct'])} / {fmt_pct(result.summary['probability_ruin_10pct'])}",
        f"Drawdown duration mean/median/worst (bars): {result.summary['drawdown_duration_mean']:.1f} / {result.summary['drawdown_duration_median']:.1f} / {result.summary['drawdown_duration_worst']:.0f}",
        f"Recovery bars mean/median/worst: {result.summary['recovery_bars_mean']:.1f} / {result.summary['recovery_bars_median']:.1f} / {result.summary['recovery_bars_worst']:.0f}",
        f"Ulcer index: {result.summary['ulcer_index']:.4f} | Downside deviation: {result.summary['downside_deviation']:.4f}",
    ]
    y = 528
    for line in risk_lines:
        t(p3, 40, y, line, 10)
        y -= 14
    draw_hist(p3, pct_points(analytics["max_drawdown"]), 40, 80, 360, 260, "Histogram of max drawdown", "Max drawdown (%)", "Simulation count", "Frequency of worst peak-to-trough losses observed per path.")
    draw_scatter(p3, pct_points(analytics["total_return"]), pct_points(analytics["max_drawdown"]), 430, 80, 370, 260, "Return vs max drawdown", "Total return (%)", "Max drawdown (%)", "Trade-off map between terminal return and deepest path drawdown. Extreme right tail is trimmed at the 95th percentile for readability.", x_clip_high_quantile=0.95)

    # Page 5: Risk-adjusted efficiency + time outcomes
    p4 = pdf.new_page()
    t(p4, 40, 565, "Risk-Adjusted Efficiency & Time-Based Outcomes", 15)
    t(p4, 40, 548, "This section evaluates whether returns are achieved efficiently and how outcomes evolve through time.", 10)
    t(p4, 40, 530, f"Sharpe/Sortino/Calmar/MAR: {result.summary['approx_sharpe']:.2f} / {result.summary['approx_sortino']:.2f} / {result.summary['approx_calmar']:.2f} / {result.summary['approx_mar']:.2f}", 10)
    t(p4, 40, 516, f"Return/maxDD: {result.summary['return_over_max_drawdown']:.2f} | Median return/median maxDD: {result.summary['median_return_over_median_drawdown']:.2f}", 10)
    t(p4, 40, 502, f"P(exceed return 25/50/100/200/500/1000%): {fmt_pct(result.summary['probability_return_over_25pct'])} / {fmt_pct(result.summary['probability_return_over_50pct'])} / {fmt_pct(result.summary['probability_return_over_100pct'])} / {fmt_pct(result.summary['probability_return_over_200pct'])} / {fmt_pct(result.summary['probability_return_over_500pct'])} / {fmt_pct(result.summary['probability_return_over_1000pct'])}", 9)
    t(p4, 40, 488, f"P(underperform baseline): {fmt_pct(result.summary['probability_return_below_baseline'])}", 10)

    y = 468
    for row in analytics["equity_checkpoints"]:
        t(p4, 40, y, f"Checkpoint bar {int(row['bar'])}: P(profitable)={fmt_pct(row['profit_prob'])}, P(below start)={fmt_pct(row['below_start_prob'])}", 9)
        y -= 12

    for label, arr, yy in [("Time to double", analytics["time_to_double"], 200), ("Time to 5x", analytics["time_to_5x"], 130), ("Time to 10x", analytics["time_to_10x"], 60)]:
        valid = arr[~np.isnan(arr)]
        if valid.size:
            txt = f"{label}: median={np.median(valid):.0f} bars, p75={np.percentile(valid,75):.0f}, reached in {fmt_pct(valid.size / len(arr))}"
        else:
            txt = f"{label}: not reached in simulations"
        t(p4, 40, yy, txt, 10)

    draw_scatter(p4, pct_points(analytics["cagr"]), pct_points(analytics["max_drawdown"]), 430, 70, 370, 220, "CAGR vs max drawdown", "CAGR (%)", "Max drawdown (%)", "Annualized return efficiency versus downside severity across simulations. Extreme right tail is trimmed at the 95th percentile for readability.", x_clip_high_quantile=0.95)

    # Page 6: Cone, sample paths, underwater
    p5 = pdf.new_page()
    t(p5, 40, 565, "Path-Level Diagnostics", 15)
    cone_values = np.vstack([
        np.asarray(analytics["equity_percentiles"][5], dtype="float64"),
        np.asarray(analytics["equity_percentiles"][50], dtype="float64"),
        np.asarray(analytics["equity_percentiles"][95], dtype="float64"),
    ])
    cone_bounds = (float(np.min(cone_values)), float(np.max(cone_values)))
    draw_line(
        p5,
        analytics["equity_percentiles"][5],
        40,
        300,
        370,
        230,
        "Rolling equity cone p5/p50/p95",
        (0.75, 0.2, 0.2),
        "Simulation bar",
        "Equity ($)",
        "Cone showing downside/median/upside equity paths.",
        legend=["Legend: red=p5/p95 envelope, blue=p50 median"],
        axis_bounds=cone_bounds,
        log_scale=True,
    )
    draw_line(p5, analytics["equity_percentiles"][50], 40, 300, 370, 230, "", (0.1, 0.3, 0.8), "Simulation bar", "Equity ($)", "", decorate=False, axis_bounds=cone_bounds, draw_axes_grid=False, log_scale=True)
    draw_line(p5, analytics["equity_percentiles"][95], 40, 300, 370, 230, "", (0.75, 0.2, 0.2), "Simulation bar", "Equity ($)", "", decorate=False, axis_bounds=cone_bounds, draw_axes_grid=False, log_scale=True)

    sample_idx = np.linspace(0, result.equity_paths.shape[0] - 1, num=min(12, result.equity_paths.shape[0]), dtype=int)
    sample_paths = result.equity_paths.iloc[sample_idx].to_numpy(dtype="float64")
    sample_bounds = (float(np.min(sample_paths)), float(np.max(sample_paths)))
    draw_line(p5, result.equity_paths.iloc[sample_idx[0]].to_numpy(dtype="float64"), 430, 300, 370, 230, "Sample equity paths (random sims)", (0.2, 0.5, 0.2), "Simulation bar", "Equity ($)", "Representative individual paths illustrating dispersion.", axis_bounds=sample_bounds, log_scale=True)
    for i in sample_idx[1:4]:
        draw_line(p5, result.equity_paths.iloc[int(i)].to_numpy(dtype="float64"), 430, 300, 370, 230, "", (0.2, 0.5, 0.2), "Simulation bar", "Equity ($)", "", decorate=False, axis_bounds=sample_bounds, draw_axes_grid=False, log_scale=True)

    worst_i = int(np.argmin(analytics["final_equity"]))
    med_i = int(np.argsort(analytics["final_equity"])[len(analytics["final_equity"]) // 2])
    best_i = int(np.argmax(analytics["final_equity"]))
    worst_med_best = np.vstack([
        result.equity_paths.iloc[worst_i].to_numpy(dtype="float64"),
        result.equity_paths.iloc[med_i].to_numpy(dtype="float64"),
        result.equity_paths.iloc[best_i].to_numpy(dtype="float64"),
    ])
    wmb_bounds = (float(np.min(worst_med_best)), float(np.max(worst_med_best)))
    draw_line(
        p5,
        result.equity_paths.iloc[worst_i].to_numpy(dtype="float64"),
        40,
        60,
        240,
        200,
        "Worst/Median/Best equity",
        (0.75, 0.2, 0.2),
        "Simulation bar",
        "Equity ($)",
        "Worst, median, and best realized paths.",
        legend=["Legend: red=worst, blue=median, green=best"],
        axis_bounds=wmb_bounds,
        log_scale=True,
    )
    draw_line(p5, result.equity_paths.iloc[med_i].to_numpy(dtype="float64"), 40, 60, 240, 200, "", (0.1, 0.3, 0.8), "Simulation bar", "Equity ($)", "", decorate=False, axis_bounds=wmb_bounds, draw_axes_grid=False, log_scale=True)
    draw_line(p5, result.equity_paths.iloc[best_i].to_numpy(dtype="float64"), 40, 60, 240, 200, "", (0.2, 0.6, 0.2), "Simulation bar", "Equity ($)", "", decorate=False, axis_bounds=wmb_bounds, draw_axes_grid=False, log_scale=True)
    draw_line(p5, pct_points(analytics["drawdown_paths"][med_i]), 300, 60, 240, 200, "Representative underwater plot", (0.2, 0.2, 0.2), "Simulation bar", "Drawdown (%)", "Underwater curve for median path showing depth and persistence below peaks.")
    draw_line(p5, pct_points(analytics["dd_percentiles"][5]), 560, 60, 240, 200, "Drawdown percentile path (worst band)", (0.8, 0.2, 0.2), "Simulation bar", "Drawdown (%)", "5th percentile drawdown path indicating persistent stress envelope over time.")

    # Page 7: Stress/scenario, methodology, flags and conclusion
    p6 = pdf.new_page()
    t(p6, 40, 565, "Scenario Sensitivity, Methodology, and Analyst Conclusion", 15)
    t(p6, 40, 548, "Stress and scenario analysis", 12)
    stressed_returns = analytics["total_return"] * 0.75
    stressed_dd = np.minimum(analytics["max_drawdown"] * 1.25, 0.0)
    t(p6, 40, 532, f"Base expected return: {fmt_pct(np.mean(analytics['total_return']))} | Stress return (25% haircut): {fmt_pct(np.mean(stressed_returns))}", 10)
    t(p6, 40, 518, f"Base mean maxDD: {fmt_pct(np.mean(analytics['max_drawdown']))} | Stress maxDD (+25%): {fmt_pct(np.mean(stressed_dd))}", 10)
    t(p6, 40, 504, f"Block-bootstrap mean maxDD: {fmt_pct(np.mean(analytics['max_drawdown']))} | Fully shuffled-sequence mean maxDD: {fmt_pct(analytics['scenario_shuffled_drawdown_mean'])}", 10)

    t(p6, 40, 482, "Methodology and reproducibility", 12)
    methodology = [
        f"Simulation method: {result.method}",
        f"Random seed: {result.seed}",
        f"Simulations: {result.simulations}; horizon bars: {result.horizon_bars}",
        "Assumptions: historical return bootstrap; path dependence retained only with block bootstrap.",
        "Limitations: does not guarantee future performance and may understate structural regime breaks.",
    ]
    if cli_flags:
        methodology.append("CLI flags used:")
        methodology.append("(Full flag list appears on dedicated CLI pages.)")
    y = 466
    for line in methodology:
        t(p6, 40, y, line, 9)
        y -= 12

    warning_header_y = y - 6
    t(p6, 40, warning_header_y, "Automated warning flags", 12)
    y = warning_header_y - 16
    flags = analytics["flags"]
    if not flags:
        t(p6, 40, y, "No severe quantitative flags triggered under current thresholds.", 9)
    else:
        for flag in flags[:8]:
            t(p6, 40, y, f"- {flag}", 9)
            y -= 12

    t(p6, 40, 280, "Analyst Conclusion", 12)
    concl = [
        f"Expected upside: {fmt_pct(result.summary['expected_return'])} mean return with {fmt_pct(result.summary['probability_profit'])} probability of profit.",
        f"Median realistic outcome: {fmt_pct(result.summary['return_median'])}, which should anchor practical expectations.",
        f"Downside risk: worst 5% drawdown near {fmt_pct(result.summary['max_drawdown_p95_worst'])} and CVaR95 at {fmt_pct(result.summary['cvar_95'])}.",
        "Robustness assessment should prioritize median, tail losses, and baseline underperformance probability over headline mean.",
        "If drawdown tolerance is limited, reduce position sizing and perform additional stress testing before live deployment.",
    ]
    y = 264
    for line in concl:
        t(p6, 40, y, line, 10)
        y -= 14

    if cli_flags:
        items = sorted(cli_flags.items(), key=lambda kv: str(kv[0]))
        per_page = 34
        for idx, start in enumerate(range(0, len(items), per_page), start=1):
            p_flags = pdf.new_page()
            suffix = "" if len(items) <= per_page else f" (page {idx})"
            t(p_flags, 40, 565, f"CLI Flags Used for This Run{suffix}", 15)
            t(p_flags, 40, 548, "All command-line options and values captured for reproducibility.", 10)
            y = 528
            for key, value in items[start:start + per_page]:
                t(p_flags, 40, y, f"--{str(key).replace('_', '-')}: {value}", 9)
                y -= 14

    pdf.save(out)
    return out
