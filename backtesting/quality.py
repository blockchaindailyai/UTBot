from __future__ import annotations

import numpy as np
import pandas as pd


def generate_data_quality_report(data: pd.DataFrame) -> dict[str, float | bool]:
    if not isinstance(data.index, pd.DatetimeIndex):
        return {
            "is_datetime_index": False,
            "timezone_aware": False,
            "duplicate_timestamps": 0.0,
            "missing_bars": 0.0,
            "outlier_bars": 0.0,
        }

    idx = data.index
    duplicate_timestamps = float(idx.duplicated().sum())

    if len(idx) >= 3:
        deltas = idx.to_series().diff().dropna()
        median_delta = deltas.median()
        expected = pd.date_range(start=idx.min(), end=idx.max(), freq=median_delta)
        missing_bars = float(len(expected.difference(idx)))
    else:
        missing_bars = 0.0

    if "close" in data.columns and len(data) > 5:
        rets = data["close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if len(rets) > 5 and rets.std(ddof=0) > 0:
            z = (rets - rets.mean()) / rets.std(ddof=0)
            outlier_bars = float((z.abs() > 5).sum())
        else:
            outlier_bars = 0.0
    else:
        outlier_bars = 0.0

    return {
        "is_datetime_index": True,
        "timezone_aware": idx.tz is not None,
        "duplicate_timestamps": duplicate_timestamps,
        "missing_bars": missing_bars,
        "outlier_bars": outlier_bars,
    }
