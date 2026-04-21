from __future__ import annotations

import pandas as pd
import pytest

from backtesting.data import filter_ohlcv_by_date, load_ohlcv_csv


def test_load_ohlcv_csv_accepts_time_and_capital_volume(tmp_path) -> None:
    csv_path = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "time": ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "Volume": [1000, 1100],
        }
    )
    df.to_csv(csv_path, index=False)

    loaded = load_ohlcv_csv(str(csv_path))

    assert loaded.index.name == "timestamp"
    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
    assert loaded["volume"].tolist() == [1000, 1100]


def test_load_ohlcv_csv_supports_unix_epoch_milliseconds(tmp_path) -> None:
    csv_path = tmp_path / "epoch_ms.csv"
    base = 1_704_067_200_000
    df = pd.DataFrame(
        {
            "timestamp": [base, base + 300_000, base + 600_000],
            "open": [100.0, 100.5, 101.0],
            "high": [101.0, 101.5, 102.0],
            "low": [99.5, 100.0, 100.5],
            "close": [100.5, 101.0, 101.5],
            "volume": [10, 11, 12],
        }
    )
    df.to_csv(csv_path, index=False)

    loaded = load_ohlcv_csv(str(csv_path))

    deltas = loaded.index.to_series().diff().dropna().dt.total_seconds()
    assert deltas.eq(300).all()
    assert loaded.index[0] == pd.Timestamp("2024-01-01T00:00:00Z")


def test_filter_ohlcv_by_date_includes_entire_end_day() -> None:
    idx = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        },
        index=idx,
    )

    filtered = filter_ohlcv_by_date(df, start="2024-01-02", end="2024-01-02")

    assert len(filtered) == 24
    assert filtered.index[0] == pd.Timestamp("2024-01-02T00:00:00Z")
    assert filtered.index[-1] == pd.Timestamp("2024-01-02T23:00:00Z")


def test_filter_ohlcv_by_date_rejects_invalid_range() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0, 1.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
        },
        index=idx,
    )

    with pytest.raises(ValueError, match="--start must be before or equal to --end"):
        filter_ohlcv_by_date(df, start="2024-01-03", end="2024-01-01")
