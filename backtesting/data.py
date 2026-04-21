from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
REQUIRED_COLUMN_ALIASES = {
    "timestamp": ["timestamp", "time"],
    "open": ["open"],
    "high": ["high"],
    "low": ["low"],
    "close": ["close"],
    "volume": ["volume", "Volume"],
}


def _parse_timestamp_column(values: pd.Series) -> pd.Series:
    """Parse timestamp values supporting ISO strings and unix epochs."""
    if pd.api.types.is_numeric_dtype(values):
        numeric_values = pd.to_numeric(values, errors="coerce")
        if numeric_values.isna().any():
            raise ValueError("Detected non-numeric values in timestamp column")

        max_abs = float(numeric_values.abs().max()) if len(numeric_values) else 0.0
        if max_abs >= 1e17:
            unit = "ns"
        elif max_abs >= 1e14:
            unit = "us"
        elif max_abs >= 1e11:
            unit = "ms"
        else:
            unit = "s"

        return pd.to_datetime(numeric_values, unit=unit, utc=True)

    return pd.to_datetime(values, utc=True)


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    """Load OHLCV CSV data into a standardized DataFrame.

    Returns a DataFrame indexed by UTC timestamp and sorted ascending.
    """
    df = pd.read_csv(path)

    rename_map: dict[str, str] = {}
    missing: list[str] = []
    for canonical, aliases in REQUIRED_COLUMN_ALIASES.items():
        matched = next((alias for alias in aliases if alias in df.columns), None)
        if matched is None:
            missing.append(canonical)
            continue
        if matched != canonical:
            rename_map[matched] = canonical

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if rename_map:
        df = df.rename(columns=rename_map)

    df = df.copy()
    df["timestamp"] = _parse_timestamp_column(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[numeric_cols].isna().any().any():
        raise ValueError("Detected non-numeric values in OHLCV columns")

    return df


def filter_ohlcv_by_date(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Filter OHLCV data by optional start/end date.

    Args:
        df: DataFrame indexed by timestamp.
        start: Inclusive start date/time string.
        end: Inclusive end date/time string. If only a date is provided
            (YYYY-MM-DD), all bars from that day are included.
    """
    if start is None and end is None:
        return df

    filtered = df
    start_ts: pd.Timestamp | None = None
    end_ts: pd.Timestamp | None = None

    if start is not None:
        try:
            start_ts = pd.to_datetime(start, utc=True)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid --start date: {start}") from exc
        filtered = filtered[filtered.index >= start_ts]

    if end is not None:
        try:
            end_ts = pd.to_datetime(end, utc=True)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid --end date: {end}") from exc

        if len(end.strip()) <= 10:
            end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        filtered = filtered[filtered.index <= end_ts]

    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError("--start must be before or equal to --end")

    return filtered
