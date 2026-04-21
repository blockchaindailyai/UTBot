from __future__ import annotations

import pandas as pd


TIMEFRAME_ALIASES = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "12h": "12h",
    "1d": "1D",
    "1w": "1W",
}


def infer_source_timeframe_label(index: pd.Index) -> str:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return "unknown"

    deltas = index.to_series().diff().dropna().dt.total_seconds()
    if deltas.empty:
        return "unknown"

    seconds = int(round(float(deltas.median())))
    if seconds <= 0:
        return "unknown"
    if seconds % 86_400 == 0:
        return f"{seconds // 86_400}d"
    if seconds % 3_600 == 0:
        return f"{seconds // 3_600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def normalize_timeframe(timeframe: str) -> str:
    key = timeframe.strip().lower()
    return TIMEFRAME_ALIASES.get(key, timeframe)


def resample_ohlcv(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV data to a target timeframe.

    Expects DatetimeIndex and OHLCV columns.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a pandas DatetimeIndex for resampling")

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns for resampling: {sorted(missing)}")

    inferred_source_freq = data.index.to_series().diff().dropna().median()
    rule = normalize_timeframe(timeframe)
    target_freq = pd.to_timedelta(rule)

    if pd.notna(inferred_source_freq) and target_freq < inferred_source_freq:
        raise ValueError(
            "Cannot upsample OHLCV bars from "
            f"{inferred_source_freq} to {target_freq}. "
            "Provide source data at or below the requested timeframe."
        )

    resampled = data.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return resampled.dropna(subset=["open", "high", "low", "close"])
