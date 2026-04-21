from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"high", "low"}


def _detect_extreme_fractals(values: np.ndarray, *, is_high: bool) -> np.ndarray:
    """Return a boolean mask for Williams fractals on contiguous equal-value plateaus.

    A plateau qualifies when there are at least 2 bars on both sides that are strictly
    lower (high fractal) or strictly higher (low fractal). If multiple adjacent bars have
    the same extreme value, the right-most bar in that run is selected.
    """

    n = len(values)
    out = np.zeros(n, dtype=bool)
    if n < 5:
        return out

    i = 0
    while i < n:
        run_start = i
        run_value = values[i]
        while i + 1 < n and values[i + 1] == run_value:
            i += 1
        run_end = i

        if run_start >= 2 and run_end + 2 < n:
            left = values[run_start - 2 : run_start]
            right = values[run_end + 1 : run_end + 3]
            if is_high:
                qualifies = bool(np.all(left < run_value) and np.all(right < run_value))
            else:
                qualifies = bool(np.all(left > run_value) and np.all(right > run_value))

            if qualifies:
                out[run_end] = True

        i += 1

    return out


def detect_williams_fractals(data: pd.DataFrame, tick_size: float = 1.0) -> pd.DataFrame:
    """Detect Williams fractals using 2-left/2-right rules with right-most plateau tie-break.

    Returns a DataFrame aligned to the input index with:
    - `up_fractal`: high fractal flag (middle/plateau extreme high)
    - `down_fractal`: low fractal flag (middle/plateau extreme low)
    - `up_fractal_price`: plotted one tick above fractal high
    - `down_fractal_price`: plotted one tick below fractal low
    """

    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"data is missing required columns: {missing_cols}")
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")

    highs = data["high"].to_numpy(dtype="float64")
    lows = data["low"].to_numpy(dtype="float64")

    up_mask = _detect_extreme_fractals(highs, is_high=True)
    down_mask = _detect_extreme_fractals(lows, is_high=False)

    up_price = np.full(len(data), np.nan, dtype="float64")
    down_price = np.full(len(data), np.nan, dtype="float64")
    up_price[up_mask] = highs[up_mask] + tick_size
    down_price[down_mask] = lows[down_mask] - tick_size

    return pd.DataFrame(
        {
            "up_fractal": up_mask,
            "down_fractal": down_mask,
            "up_fractal_price": up_price,
            "down_fractal_price": down_price,
        },
        index=data.index,
    )
