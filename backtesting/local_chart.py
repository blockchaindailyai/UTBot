from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .batch import BatchBacktestResult
from .engine import BacktestConfig, BacktestEngine, ExecutionEvent, Trade
from .fractals import detect_williams_fractals


_MARKER_PRICE_OFFSET = 10.0


def _timestamp_seconds(ts: pd.Timestamp) -> int:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.tz_convert("UTC").timestamp())




def _infer_tick_size(data: pd.DataFrame) -> float:
    values = pd.concat([data["open"], data["high"], data["low"], data["close"]]).astype("float64")
    diffs = values.sort_values().diff().abs()
    positive = diffs[diffs > 0]
    if positive.empty:
        return 1.0
    return float(positive.min())


def _williams_fractal_markers(data: pd.DataFrame) -> list[dict[str, str | int | float]]:
    tick_size = _infer_tick_size(data)
    fractals = detect_williams_fractals(data, tick_size=tick_size)

    markers: list[dict[str, str | int | float]] = []
    for ts, row in fractals.iterrows():
        if bool(row["up_fractal"]):
            markers.append(
                {
                    "time": _timestamp_seconds(ts),
                    "position": "aboveBar",
                    "price": float(row["up_fractal_price"]),
                    "color": "#22c55e",
                    "shape": "arrowDown",
                    "text": "F",
                }
            )
        if bool(row["down_fractal"]):
            markers.append(
                {
                    "time": _timestamp_seconds(ts),
                    "position": "belowBar",
                    "price": float(row["down_fractal_price"]),
                    "color": "#ef4444",
                    "shape": "arrowUp",
                    "text": "F",
                }
            )
    return markers


def _valid_third_wiseman_fractal_markers(
    data: pd.DataFrame,
    third_setup_side: pd.Series | None = None,
) -> list[dict[str, str | int | float]]:
    # Third-Wiseman setup arrows are intentionally hidden from chart overlays.
    return []

    tick_size = _infer_tick_size(data)
    fractals = detect_williams_fractals(data, tick_size=tick_size)

    median_price = (data["high"] + data["low"]) / 2
    jaw = _smma(median_price, 13).shift(8)
    teeth = _smma(median_price, 8).shift(5)
    lips = _smma(median_price, 5).shift(3)

    red_to_green_distance = (teeth - lips).abs()
    green_to_midpoint_distance = (lips - median_price).abs()
    gator_up = (lips > teeth) & (teeth > jaw)
    gator_down = (lips < teeth) & (teeth < jaw)
    gator_width_valid = red_to_green_distance < green_to_midpoint_distance

    ao = median_price.rolling(5).mean() - median_price.rolling(34).mean()
    ao_green = ao > ao.shift(1)
    ao_red = ao < ao.shift(1)

    if third_setup_side is not None:
        aligned_side = third_setup_side.reindex(data.index).fillna(0).astype("int8")
        markers: list[dict[str, str | int | float]] = []
        for ts, side in aligned_side.items():
            if int(side) == 0:
                continue
            row = data.loc[ts]
            high = float(row["high"])
            low = float(row["low"])
            is_short_setup = int(side) == -1
            markers.append(
                {
                    "time": _timestamp_seconds(ts),
                    "position": "aboveBar" if is_short_setup else "belowBar",
                    "price": high + _MARKER_PRICE_OFFSET if is_short_setup else low - _MARKER_PRICE_OFFSET,
                    "color": "#ef4444" if is_short_setup else "#22c55e",
                    "shape": "arrowDown" if is_short_setup else "arrowUp",
                    "text": "",
                }
            )
        return markers

    markers: list[dict[str, str | int | float]] = []
    pending_setups: list[dict[str, float | int]] = []
    watched_setup: dict[str, float | int] | None = None
    blocked_side: int | None = None

    for i in range(2, len(data)):
        high_now = float(data["high"].iloc[i])
        low_now = float(data["low"].iloc[i])

        high_two_back = float(data["high"].iloc[i - 2])
        high_one_back = float(data["high"].iloc[i - 1])
        low_two_back = float(data["low"].iloc[i - 2])
        low_one_back = float(data["low"].iloc[i - 1])

        is_local_peak = high_one_back > high_two_back and high_one_back > high_now
        bearish_body = float(data["open"].iloc[i - 1]) > float(data["close"].iloc[i - 1])
        if is_local_peak and bearish_body and bool(gator_up.iloc[i - 1]) and bool(gator_width_valid.iloc[i - 1]) and bool(ao_green.iloc[i - 1]):
            pending_setups.append({"side": -1, "high": high_one_back, "low": low_one_back, "bar": i - 1})

        is_local_trough = low_one_back < low_two_back and low_one_back < low_now
        bullish_body = float(data["close"].iloc[i - 1]) > float(data["open"].iloc[i - 1])
        if is_local_trough and bullish_body and bool(gator_down.iloc[i - 1]) and bool(gator_width_valid.iloc[i - 1]) and bool(ao_red.iloc[i - 1]):
            pending_setups.append(
                {"side": 1, "high": float(data["high"].iloc[i - 1]), "low": low_one_back, "bar": i - 1}
            )

        if watched_setup is None:
            survivors: list[dict[str, float | int]] = []
            for setup in pending_setups:
                setup_side = int(setup["side"])
                if blocked_side is not None and setup_side == blocked_side:
                    continue
                if int(setup["side"]) == -1:
                    if high_now > float(setup["high"]):
                        continue
                    if low_now <= float(setup["low"]):
                        watched_setup = setup
                        if blocked_side is not None and blocked_side != setup_side:
                            blocked_side = None
                        break
                else:
                    if low_now < float(setup["low"]):
                        continue
                    if high_now >= float(setup["high"]):
                        watched_setup = setup
                        if blocked_side is not None and blocked_side != setup_side:
                            blocked_side = None
                        break
                survivors.append(setup)
            pending_setups = survivors

        if watched_setup is not None and i >= 4:
            fractal_i = i - 2
            if fractal_i > int(watched_setup["bar"]):
                ts = data.index[fractal_i]
                if int(watched_setup["side"]) == -1 and bool(fractals["down_fractal"].iloc[fractal_i]):
                    fractal_low = float(data["low"].iloc[fractal_i])
                    if pd.notna(teeth.iloc[fractal_i]) and fractal_low < float(teeth.iloc[fractal_i]):
                        markers.append(
                            {
                                "time": _timestamp_seconds(ts),
                                "position": "aboveBar",
                                "price": float(data["high"].iloc[fractal_i]) + _MARKER_PRICE_OFFSET,
                                "color": "#ef4444",
                                "shape": "arrowDown",
                                "text": "",
                            }
                        )
                        blocked_side = -1
                        watched_setup = None
                elif int(watched_setup["side"]) == 1 and bool(fractals["up_fractal"].iloc[fractal_i]):
                    fractal_high = float(data["high"].iloc[fractal_i])
                    if pd.notna(teeth.iloc[fractal_i]) and fractal_high > float(teeth.iloc[fractal_i]):
                        markers.append(
                            {
                                "time": _timestamp_seconds(ts),
                                "position": "belowBar",
                                "price": float(data["low"].iloc[fractal_i]) - _MARKER_PRICE_OFFSET,
                                "color": "#22c55e",
                                "shape": "arrowUp",
                                "text": "",
                            }
                        )
                        blocked_side = 1
                        watched_setup = None

    return markers


def _second_wiseman_markers(
    data: pd.DataFrame,
    second_fill_prices: pd.Series | None,
    second_setup_side: pd.Series | None = None,
) -> list[dict[str, str | int | float]]:
    if second_setup_side is not None:
        side_aligned = second_setup_side.reindex(data.index).fillna(0).astype("int8")
        markers: list[dict[str, str | int | float]] = []
        for ts, side in side_aligned.items():
            if int(side) == 0:
                continue
            row = data.loc[ts]
            high = float(row["high"])
            low = float(row["low"])
            is_short_setup = int(side) == -1
            markers.append(
                {
                    "time": _timestamp_seconds(ts),
                    "position": "aboveBar" if is_short_setup else "belowBar",
                    "price": high + _MARKER_PRICE_OFFSET if is_short_setup else low - _MARKER_PRICE_OFFSET,
                    "color": "#dc2626" if is_short_setup else "#16a34a",
                    "shape": "arrowDown" if is_short_setup else "arrowUp",
                    "text": "2W",
                }
            )
        return markers

    if second_fill_prices is None:
        return []

    aligned = second_fill_prices.reindex(data.index)
    markers: list[dict[str, str | int | float]] = []
    for ts, fill_price_raw in aligned.items():
        if pd.isna(fill_price_raw):
            continue

        fill_price = float(fill_price_raw)
        row = data.loc[ts]
        high = float(row["high"])
        low = float(row["low"])
        is_short_add = abs(fill_price - low) <= abs(fill_price - high)

        markers.append(
            {
                "time": _timestamp_seconds(ts),
                "position": "aboveBar" if is_short_add else "belowBar",
                "price": high + _MARKER_PRICE_OFFSET if is_short_add else low - _MARKER_PRICE_OFFSET,
                "color": "#dc2626" if is_short_add else "#16a34a",
                "shape": "arrowDown" if is_short_add else "arrowUp",
                "text": "2W",
            }
        )

    return markers


def _wiseman_fill_entry_markers(
    data: pd.DataFrame,
    fill_prices: pd.Series | None,
    fill_side: pd.Series | None = None,
    label: str | None = None,
) -> list[dict[str, str | int | float]]:
    if fill_prices is None:
        return []

    long_text = "LE" if label is None else f"LE-{label}"
    short_text = "SE" if label is None else f"SE-{label}"

    if fill_side is not None:
        aligned_side = fill_side.reindex(data.index).fillna(0).astype("int8")
        aligned_prices = fill_prices.reindex(data.index)
        markers: list[dict[str, str | int | float]] = []
        for ts, side in aligned_side.items():
            if int(side) == 0:
                continue
            if pd.isna(aligned_prices.loc[ts]):
                continue
            row = data.loc[ts]
            high = float(row["high"])
            low = float(row["low"])
            is_short_entry = int(side) == -1
            markers.append(
                {
                    "time": _timestamp_seconds(ts),
                    "position": "aboveBar" if is_short_entry else "belowBar",
                    "price": high + _MARKER_PRICE_OFFSET if is_short_entry else low - _MARKER_PRICE_OFFSET,
                    "color": "#dc2626" if is_short_entry else "#16a34a",
                    "shape": "arrowDown" if is_short_entry else "arrowUp",
                    "text": short_text if is_short_entry else long_text,
                }
            )
        return markers

    aligned = fill_prices.reindex(data.index)
    markers: list[dict[str, str | int | float]] = []
    for ts, fill_price_raw in aligned.items():
        if pd.isna(fill_price_raw):
            continue

        fill_price = float(fill_price_raw)
        row = data.loc[ts]
        high = float(row["high"])
        low = float(row["low"])
        is_short_entry = abs(fill_price - low) <= abs(fill_price - high)
        markers.append(
            {
                "time": _timestamp_seconds(ts),
                "position": "aboveBar" if is_short_entry else "belowBar",
                "price": high + _MARKER_PRICE_OFFSET if is_short_entry else low - _MARKER_PRICE_OFFSET,
                "color": "#dc2626" if is_short_entry else "#16a34a",
                "shape": "arrowDown" if is_short_entry else "arrowUp",
                "text": short_text if is_short_entry else long_text,
            }
        )

    return markers


def _first_wiseman_ignored_markers(
    data: pd.DataFrame,
    setup_side: pd.Series | None,
    ignored_reason: pd.Series | None,
) -> list[dict[str, str | int | float]]:
    if setup_side is None or ignored_reason is None:
        return []

    aligned_side = setup_side.reindex(data.index).fillna(0).astype("int8")
    aligned_reason = ignored_reason.reindex(data.index).fillna("").astype("string")

    reason_labels = {
        "invalidation_before_trigger": "1W-I",
        "same_bar_stop_before_reversal_window": "1W-C",
        "no_breakout_until_end_of_data": "1W-N",
        "gator_closed_canceled": "1W-A",
        "gator_open_percentile_filter": "1W-G",
        "weaker_than_active_setup": "1W-W",
        "reversal_cooldown_active": "1W-CD",
        "ao_divergence_filter": "1W-D",
        "signal_disabled_zero_contracts": "1W-X",
    }

    markers: list[dict[str, str | int | float]] = []
    for ts, reason in aligned_reason.items():
        reason_text = str(reason)
        if reason_text == "" or reason_text not in reason_labels:
            continue

        side = int(aligned_side.loc[ts])
        if side == 0:
            continue

        row = data.loc[ts]
        high = float(row["high"])
        low = float(row["low"])
        is_short_setup = side == -1
        markers.append(
            {
                "time": _timestamp_seconds(ts),
                "position": "aboveBar" if is_short_setup else "belowBar",
                "price": high + (_MARKER_PRICE_OFFSET * 1.2) if is_short_setup else low - (_MARKER_PRICE_OFFSET * 1.2),
                "color": "#f59e0b",
                "shape": "circle",
                "size": 0,
                "text": reason_labels[reason_text],
            }
        )

    return markers


def _first_wiseman_engine_markers(
    data: pd.DataFrame,
    setup_side: pd.Series | None,
    ignored_reason: pd.Series | None,
    reversal_side: pd.Series | None,
    *,
    include_bearish_wiseman: bool = True,
    include_bullish_wiseman: bool = True,
) -> list[dict[str, str | int | float]]:
    if setup_side is None or ignored_reason is None or reversal_side is None:
        return []

    aligned_side = setup_side.reindex(data.index).fillna(0).astype("int8")
    aligned_reason = ignored_reason.reindex(data.index).fillna("").astype("string")
    aligned_reversal = reversal_side.reindex(data.index).fillna(0).astype("int8")

    markers: list[dict[str, str | int | float]] = []

    for ts in data.index:
        side = int(aligned_side.loc[ts])
        reason = str(aligned_reason.loc[ts])
        if side != 0 and reason == "":
            if side == -1 and not include_bearish_wiseman:
                continue
            if side == 1 and not include_bullish_wiseman:
                continue
            row = data.loc[ts]
            high = float(row["high"])
            low = float(row["low"])
            is_short_setup = side == -1
            markers.append(
                {
                    "time": _timestamp_seconds(ts),
                    "position": "aboveBar" if is_short_setup else "belowBar",
                    "price": high + _MARKER_PRICE_OFFSET if is_short_setup else low - _MARKER_PRICE_OFFSET,
                    "color": "#dc2626" if is_short_setup else "#16a34a",
                    "shape": "arrowDown" if is_short_setup else "arrowUp",
                    "text": "1W",
                }
            )

        reverse = int(aligned_reversal.loc[ts])
        if reverse == 0:
            continue
        if reverse == -1 and not include_bearish_wiseman:
            continue
        if reverse == 1 and not include_bullish_wiseman:
            continue
        row = data.loc[ts]
        high = float(row["high"])
        low = float(row["low"])
        is_short_reversal = reverse == -1
        markers.append(
            {
                "time": _timestamp_seconds(ts),
                "position": "aboveBar" if is_short_reversal else "belowBar",
                "price": high + _MARKER_PRICE_OFFSET if is_short_reversal else low - _MARKER_PRICE_OFFSET,
                "color": "#dc2626" if is_short_reversal else "#16a34a",
                "shape": "arrowDown" if is_short_reversal else "arrowUp",
                "text": "1W-R",
            }
        )

    return markers


def _bearish_first_wiseman_markers(
    data: pd.DataFrame,
    spread_lookback: int = 50,
    spread_multiplier: float = 1.0,
) -> list[dict[str, str | int | float]]:
    median_price = (data["high"] + data["low"]) / 2
    jaw = _smma(median_price, 13).shift(8)
    teeth = _smma(median_price, 8).shift(5)
    lips = _smma(median_price, 5).shift(3)

    candidate_midpoint = (data["high"] + data["low"]) / 2
    red_to_green_distance = (teeth - lips).abs()
    green_to_midpoint_distance = (lips - candidate_midpoint).abs()
    gator_open_up = (lips > teeth) & (teeth > jaw)
    gator_width_valid = red_to_green_distance < green_to_midpoint_distance

    ao = median_price.rolling(5).mean() - median_price.rolling(34).mean()
    ao_green = ao > ao.shift(1)

    markers: list[dict[str, str | int | float]] = []
    active_candidate = False
    candidate_high = 0.0
    candidate_low = 0.0
    candidate_ts: pd.Timestamp | None = None

    reverse_watch_active = False
    reverse_source_high = 0.0
    reverse_confirmed_on_i = -1

    for i in range(2, len(data)):
        high_two_back = float(data["high"].iloc[i - 2])
        high_one_back = float(data["high"].iloc[i - 1])
        high_now = float(data["high"].iloc[i])
        low_now = float(data["low"].iloc[i])

        is_local_peak = high_one_back > high_two_back and high_one_back > high_now
        candidate_bearish_body = float(data["open"].iloc[i - 1]) > float(data["close"].iloc[i - 1])
        candidate_qualifies = (
            is_local_peak
            and candidate_bearish_body
            and bool(gator_open_up.iloc[i - 1])
            and bool(gator_width_valid.iloc[i - 1])
            and bool(ao_green.iloc[i - 1])
        )

        if candidate_qualifies and not reverse_watch_active:
            active_candidate = True
            candidate_high = high_one_back
            candidate_low = float(data["low"].iloc[i - 1])
            candidate_ts = data.index[i - 1]

        if active_candidate:
            if high_now > candidate_high:
                active_candidate = False
                candidate_ts = None
            elif low_now < candidate_low and candidate_ts is not None:
                markers.append(
                    {
                        "time": _timestamp_seconds(candidate_ts),
                        "position": "aboveBar",
                        "price": candidate_high + _MARKER_PRICE_OFFSET,
                        "color": "#dc2626",
                        "shape": "arrowDown",
                        "text": "1W",
                    }
                )
                reverse_watch_active = True
                reverse_source_high = candidate_high
                reverse_confirmed_on_i = i
                active_candidate = False
                candidate_ts = None

        if reverse_watch_active and i > reverse_confirmed_on_i and high_now > reverse_source_high:
            markers.append(
                {
                    "time": _timestamp_seconds(data.index[i]),
                    "position": "belowBar",
                    "price": float(data["low"].iloc[i]) - _MARKER_PRICE_OFFSET,
                    "color": "#16a34a",
                    "shape": "arrowUp",
                    "text": "1W-R",
                }
            )
            reverse_watch_active = False

    return markers


def _bullish_first_wiseman_markers(
    data: pd.DataFrame,
    spread_lookback: int = 50,
    spread_multiplier: float = 1.0,
) -> list[dict[str, str | int | float]]:
    median_price = (data["high"] + data["low"]) / 2
    jaw = _smma(median_price, 13).shift(8)
    teeth = _smma(median_price, 8).shift(5)
    lips = _smma(median_price, 5).shift(3)

    candidate_midpoint = (data["high"] + data["low"]) / 2
    red_to_green_distance = (teeth - lips).abs()
    green_to_midpoint_distance = (lips - candidate_midpoint).abs()
    gator_open_down = (lips < teeth) & (teeth < jaw)
    gator_width_valid = red_to_green_distance < green_to_midpoint_distance

    ao = median_price.rolling(5).mean() - median_price.rolling(34).mean()
    ao_red = ao < ao.shift(1)

    markers: list[dict[str, str | int | float]] = []
    active_candidate = False
    candidate_high = 0.0
    candidate_low = 0.0
    candidate_ts: pd.Timestamp | None = None

    reverse_watch_active = False
    reverse_source_low = 0.0
    reverse_confirmed_on_i = -1

    for i in range(2, len(data)):
        low_two_back = float(data["low"].iloc[i - 2])
        low_one_back = float(data["low"].iloc[i - 1])
        low_now = float(data["low"].iloc[i])
        high_now = float(data["high"].iloc[i])

        is_local_trough = low_one_back < low_two_back and low_one_back < low_now
        candidate_bullish_body = float(data["close"].iloc[i - 1]) > float(data["open"].iloc[i - 1])
        candidate_qualifies = (
            is_local_trough
            and candidate_bullish_body
            and bool(gator_open_down.iloc[i - 1])
            and bool(gator_width_valid.iloc[i - 1])
            and bool(ao_red.iloc[i - 1])
        )

        if candidate_qualifies and not reverse_watch_active:
            active_candidate = True
            candidate_high = float(data["high"].iloc[i - 1])
            candidate_low = low_one_back
            candidate_ts = data.index[i - 1]

        if active_candidate:
            if low_now < candidate_low:
                active_candidate = False
                candidate_ts = None
            elif high_now > candidate_high and candidate_ts is not None:
                markers.append(
                    {
                        "time": _timestamp_seconds(candidate_ts),
                        "position": "belowBar",
                        "price": candidate_low - _MARKER_PRICE_OFFSET,
                        "color": "#16a34a",
                        "shape": "arrowUp",
                        "text": "1W",
                    }
                )
                reverse_watch_active = True
                reverse_source_low = candidate_low
                reverse_confirmed_on_i = i
                active_candidate = False
                candidate_ts = None

        if reverse_watch_active and i > reverse_confirmed_on_i and low_now < reverse_source_low:
            markers.append(
                {
                    "time": _timestamp_seconds(data.index[i]),
                    "position": "aboveBar",
                    "price": float(data["high"].iloc[i]) + _MARKER_PRICE_OFFSET,
                    "color": "#dc2626",
                    "shape": "arrowDown",
                    "text": "1W-R",
                }
            )
            reverse_watch_active = False

    return markers


def _execution_event_lines(
    execution_events: list[ExecutionEvent],
    index: pd.DatetimeIndex,
) -> list[dict[str, str | list[dict[str, float | int]]]]:
    index_lookup = {ts: i for i, ts in enumerate(index)}

    def _line_points(ts: pd.Timestamp, price: float) -> list[dict[str, float | int]]:
        i = index_lookup.get(ts)
        if i is None:
            return []
        points: list[dict[str, float | int]] = []
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(index):
                points.append({"time": _timestamp_seconds(index[j]), "value": price})
        return points

    lines: list[dict[str, str | list[dict[str, float | int]]]] = []
    for event in execution_events:
        color = "#16a34a" if event.side == "buy" else "#dc2626"
        label = event.event_type.upper()
        points = _line_points(event.time, float(event.price))
        if points:
            lines.append({"label": label, "color": color, "points": points})
    return lines


def _execution_trade_path_lines(
    execution_events: list[ExecutionEvent],
    index: pd.DatetimeIndex,
) -> list[dict[str, str | list[dict[str, float | int]]]]:
    index_lookup = {ts: i for i, ts in enumerate(index)}
    path_lines: list[dict[str, str | list[dict[str, float | int]]]] = []
    open_legs: list[dict[str, float | int | pd.Timestamp]] = []

    for event in execution_events:
        if event.event_type in {"entry", "add"}:
            open_legs.append({"time": event.time, "price": float(event.price), "units": float(event.units)})
            continue

        if event.event_type in {"reduce", "exit"}:
            remaining = float(event.units)
            while remaining > 1e-12 and open_legs:
                leg = open_legs[0]
                leg_units = float(leg["units"])
                used = min(leg_units, remaining)
                entry_i = index_lookup.get(leg["time"])
                exit_i = index_lookup.get(event.time)
                if entry_i is not None and exit_i is not None:
                    path_lines.append(
                        {
                            "label": "Trade Path",
                            "color": "#16a34a" if event.side == "sell" else "#dc2626",
                            "points": [
                                {"time": _timestamp_seconds(index[entry_i]), "value": float(leg["price"])},
                                {"time": _timestamp_seconds(index[exit_i]), "value": float(event.price)},
                            ],
                        }
                    )
                leg_units -= used
                remaining -= used
                if leg_units <= 1e-12:
                    open_legs.pop(0)
                else:
                    leg["units"] = leg_units
    return path_lines


def _trade_event_lines(
    trades: list[Trade],
    index: pd.DatetimeIndex,
) -> list[dict[str, str | list[dict[str, float | int]]]]:
    index_lookup = {ts: i for i, ts in enumerate(index)}

    def _line_points(ts: pd.Timestamp, price: float) -> list[dict[str, float | int]]:
        i = index_lookup.get(ts)
        if i is None:
            return []
        points: list[dict[str, float | int]] = []
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(index):
                points.append({"time": _timestamp_seconds(index[j]), "value": price})
        return points

    event_lines: list[dict[str, str | list[dict[str, float | int]]]] = []
    for trade in trades:
        is_long = trade.side == "long"

        entry_label = "LE" if is_long else "SE"
        entry_color = "#16a34a" if is_long else "#dc2626"
        entry_points = _line_points(trade.entry_time, float(trade.entry_price))
        if entry_points:
            event_lines.append({"label": entry_label, "color": entry_color, "points": entry_points})

        exit_label = "LX" if is_long else "SX"
        exit_color = "#dc2626" if is_long else "#16a34a"
        exit_points = _line_points(trade.exit_time, float(trade.exit_price))
        if exit_points:
            event_lines.append({"label": exit_label, "color": exit_color, "points": exit_points})

    return event_lines


def _trade_entry_exit_lines(
    trades: list[Trade],
    index: pd.DatetimeIndex,
) -> list[dict[str, str | list[dict[str, float | int]]]]:
    index_lookup = {ts: i for i, ts in enumerate(index)}
    entry_exit_lines: list[dict[str, str | list[dict[str, float | int]]]] = []

    for trade in trades:
        entry_i = index_lookup.get(trade.entry_time)
        exit_i = index_lookup.get(trade.exit_time)
        if entry_i is None or exit_i is None:
            continue

        points = [
            {"time": _timestamp_seconds(index[entry_i]), "value": float(trade.entry_price)},
            {"time": _timestamp_seconds(index[exit_i]), "value": float(trade.exit_price)},
        ]
        entry_exit_lines.append(
            {
                "label": "Trade Path",
                "color": "#16a34a" if trade.side == "long" else "#dc2626",
                "points": points,
            }
        )

    return entry_exit_lines


def _trade_execution_markers(
    trades: list[Trade],
    data: pd.DataFrame,
) -> list[dict[str, str | int | float]]:
    def _trade_label(trade: Trade, signal_attr: str, reason_attr: str) -> str:
        value = getattr(trade, signal_attr, None)
        if value is None or (isinstance(value, str) and not value.strip()):
            value = getattr(trade, reason_attr, None)
        return str(value).strip() if value is not None and str(value).strip() else "Unknown"

    marker_buckets: dict[tuple[int, str], dict[str, str | int | float]] = {}
    for trade in trades:
        is_long = trade.side == "long"

        entry_ts = trade.entry_time
        exit_ts = trade.exit_time
        entry_label = _trade_marker_text("LE" if is_long else "SE", _trade_label(trade, "entry_signal", "entry_reason"))
        exit_label = _trade_marker_text("LX" if is_long else "SX", _trade_label(trade, "exit_signal", "exit_reason"))

        entry_row = data.loc[entry_ts] if entry_ts in data.index else None
        if entry_row is not None:
            entry_position = "belowBar" if is_long else "aboveBar"
            entry_key = (_timestamp_seconds(entry_ts), entry_position)
            existing = marker_buckets.get(entry_key)
            text = entry_label if existing is None else f"{existing['text']}/{entry_label}"
            marker_buckets[entry_key] = {
                "time": entry_key[0],
                "position": entry_position,
                "price": (
                    float(entry_row["low"]) - _MARKER_PRICE_OFFSET
                    if is_long
                    else float(entry_row["high"]) + _MARKER_PRICE_OFFSET
                ),
                "color": "#16a34a" if is_long else "#dc2626",
                "shape": "arrowUp" if is_long else "arrowDown",
                "text": text,
            }

        exit_row = data.loc[exit_ts] if exit_ts in data.index else None
        if exit_row is not None:
            exit_position = "aboveBar" if is_long else "belowBar"
            exit_key = (_timestamp_seconds(exit_ts), exit_position)
            existing = marker_buckets.get(exit_key)
            text = exit_label if existing is None else f"{existing['text']}/{exit_label}"
            marker_buckets[exit_key] = {
                "time": exit_key[0],
                "position": exit_position,
                "price": (
                    float(exit_row["high"]) + _MARKER_PRICE_OFFSET
                    if is_long
                    else float(exit_row["low"]) - _MARKER_PRICE_OFFSET
                ),
                "color": "#dc2626" if is_long else "#16a34a",
                "shape": "arrowDown" if is_long else "arrowUp",
                "text": text,
            }

    return sorted(marker_buckets.values(), key=lambda marker: (int(marker["time"]), str(marker["position"])))


def _execution_event_markers(
    execution_events: list[ExecutionEvent],
    data: pd.DataFrame,
) -> list[dict[str, str | int | float]]:
    marker_buckets: dict[tuple[int, str], dict[str, str | int | float]] = {}
    for event in execution_events:
        event_type = str(event.event_type)
        if event_type not in {"entry", "add", "reduce", "exit"}:
            continue
        ts = pd.Timestamp(event.time)
        if ts not in data.index:
            continue
        row = data.loc[ts]
        side = str(event.side)
        is_buy = side == "buy"
        if event_type in {"entry", "add"}:
            prefix = "LE" if is_buy else "SE"
            position = "belowBar" if is_buy else "aboveBar"
            color = "#16a34a" if is_buy else "#dc2626"
            shape = "arrowUp" if is_buy else "arrowDown"
            price = float(row["low"]) - _MARKER_PRICE_OFFSET if is_buy else float(row["high"]) + _MARKER_PRICE_OFFSET
        else:
            prefix = "SX" if is_buy else "LX"
            position = "belowBar" if is_buy else "aboveBar"
            color = "#16a34a" if is_buy else "#dc2626"
            shape = "arrowUp" if is_buy else "arrowDown"
            price = float(row["low"]) - _MARKER_PRICE_OFFSET if is_buy else float(row["high"]) + _MARKER_PRICE_OFFSET

        label = _trade_marker_text(prefix, getattr(event, "strategy_reason", None))
        key = (_timestamp_seconds(ts), position)
        existing = marker_buckets.get(key)
        text = label if existing is None else f"{existing['text']}/{label}"
        marker_buckets[key] = {
            "time": key[0],
            "position": position,
            "price": price,
            "color": color,
            "shape": shape,
            "text": text,
        }

    return sorted(marker_buckets.values(), key=lambda marker: (int(marker["time"]), str(marker["position"])))


def _strategy_reason_marker_label(reason: str | None) -> str:
    normalized = str(reason or "").strip()
    if normalized in {"Strategy Profit Protection Red Gator", "Red Gator PP", "Red Gator"}:
        return "Red PP"
    if normalized in {"Strategy Profit Protection Green Gator", "Green Gator PP", "Green Gator"}:
        return "Green PP"
    return normalized


def _compact_trade_reason(reason: str | None) -> str | None:
    normalized = str(reason or "").strip()
    if normalized == "":
        return None

    direct_mapping = {
        "Bullish 1W": "1W",
        "Bearish 1W": "1W",
        "Bullish Fractal": "F",
        "Bearish Fractal": "F",
        "Bullish Add-on Fractal": "AF",
        "Bearish Add-on Fractal": "AF",
        "Bullish 1W-R": "1W-R",
        "Bearish 1W-R": "1W-R",
        "Bullish 1W Reversal": "1W-R",
        "Bearish 1W Reversal": "1W-R",
        "Bullish 2W": "2W",
        "Bearish 2W": "2W",
        "Bullish 3W": None,
        "Bearish 3W": None,
        "Strategy Profit Protection Green Gator": "G",
        "Strategy Profit Protection Red Gator": "R",
        "Green Gator PP": "G",
        "Red Gator PP": "R",
        "Green Gator": "G",
        "Red Gator": "R",
        "Strategy Stop Loss Bullish 1W": "S",
        "Strategy Stop Loss Bearish 1W": "S",
        "1W Reversal Stop": "S",
        "NTD Entry Stop": "S",
        "Williams Zone PP": "Z",
        "protective_stop": "S",
        "take_profit": "TP",
        "Reduce": "RED",
    }
    if normalized in direct_mapping:
        return direct_mapping[normalized]

    if normalized.startswith("Signal Intent Flat from "):
        return _compact_trade_reason(normalized.removeprefix("Signal Intent Flat from "))
    if normalized.startswith("Signal Intent Flip to "):
        return _compact_trade_reason(normalized.removeprefix("Signal Intent Flip to "))
    if normalized.startswith("Signal Intent Reduce from "):
        return _compact_trade_reason(normalized.removeprefix("Signal Intent Reduce from "))
    if normalized.startswith("Strategy Exit Reason: "):
        return _compact_trade_reason(normalized.removeprefix("Strategy Exit Reason: "))
    if normalized.startswith("Strategy Reversal to "):
        return _compact_trade_reason(normalized.removeprefix("Strategy Reversal to "))
    if normalized.startswith("Bullish ") or normalized.startswith("Bearish "):
        parts = normalized.split(" ", maxsplit=1)
        if len(parts) == 2:
            return _compact_trade_reason(parts[1]) or parts[1]
    return normalized


def _trade_marker_text(prefix: str, reason: str | None) -> str:
    compact_reason = _compact_trade_reason(reason)
    return f"{prefix}-{compact_reason}" if compact_reason else prefix


def _missing_gator_profit_protection_trades(
    data: pd.DataFrame,
    strategy_template: Any,
    existing_trades: list[Any],
) -> list[Trade]:
    if data.empty or strategy_template is None:
        return []

    replay_strategy = copy.deepcopy(strategy_template)
    replay_result = BacktestEngine(
        BacktestConfig(
            initial_capital=10_000.0,
            fee_rate=0.0,
            slippage_rate=0.0,
            spread_rate=0.0,
            trade_size_mode="units",
            trade_size_value=1.0,
            close_open_position_on_last_bar=False,
        )
    ).run(data.copy(), replay_strategy)

    existing_keys = {
        (pd.Timestamp(trade.entry_time), pd.Timestamp(trade.exit_time), str(trade.side))
        for trade in existing_trades
    }
    return [
        trade
        for trade in replay_result.trades
        if _strategy_reason_marker_label(trade.exit_signal) in {"Red PP", "Green PP"}
        and (pd.Timestamp(trade.entry_time), pd.Timestamp(trade.exit_time), str(trade.side)) not in existing_keys
    ]


def _gator_profit_protection_fallback_overlays(
    data: pd.DataFrame,
    strategy_template: Any,
    existing_trades: list[Any],
) -> dict[str, list[dict[str, str | int | float]] | list[dict[str, str | list[dict[str, float | int]]]]]:
    fallback_trades = _missing_gator_profit_protection_trades(data, strategy_template, existing_trades)
    return {
        "markers": _trade_execution_markers(fallback_trades, data),
        "trade_event_lines": _trade_event_lines(fallback_trades, data.index),
        "trade_path_lines": _trade_entry_exit_lines(fallback_trades, data.index),
    }


def _combine_markers(*marker_sets: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    combined: list[dict[str, str | int | float]] = []
    for markers in marker_sets:
        combined.extend(markers)
    return sorted(combined, key=lambda marker: int(marker["time"]))


def _wiseman_markers(
    data: pd.DataFrame,
    *,
    include_bearish_wiseman: bool = True,
    include_bullish_wiseman: bool = True,
) -> dict[str, list[dict[str, str | int | float]]]:
    return {
        "bearish": _bearish_first_wiseman_markers(data) if include_bearish_wiseman else [],
        "bullish": _bullish_first_wiseman_markers(data) if include_bullish_wiseman else [],
    }


def _build_markers(
    trades: list[Trade],
    data: pd.DataFrame | None = None,
    *,
    include_bearish_wiseman: bool = True,
    include_bullish_wiseman: bool = True,
) -> list[dict[str, str | int | float]]:
    markers: list[dict[str, str | int | float]] = []
    if data is not None:
        wiseman = _wiseman_markers(
            data,
            include_bearish_wiseman=include_bearish_wiseman,
            include_bullish_wiseman=include_bullish_wiseman,
        )
        markers.extend(wiseman["bearish"])
        markers.extend(wiseman["bullish"])
        markers.extend(_valid_third_wiseman_fractal_markers(data))
    return markers


def summarize_wiseman_markers(data: pd.DataFrame) -> dict[str, int]:
    """Return counts for each Wiseman marker type on the supplied OHLCV data."""
    bearish_markers = _bearish_first_wiseman_markers(data)
    bullish_markers = _bullish_first_wiseman_markers(data)
    return {
        "bearish_first_wiseman": sum(marker["text"] == "1W" for marker in bearish_markers),
        "bearish_reverse": sum(marker["text"] == "1W-R" for marker in bearish_markers),
        "bullish_first_wiseman": sum(marker["text"] in {"1W", "1W+"} for marker in bullish_markers),
        "bullish_reverse": sum(marker["text"] == "1W+-R" for marker in bullish_markers),
    }


def _candles_from_data(
    data: pd.DataFrame, zone_colors: pd.Series | None = None
) -> list[dict[str, float | int | str]]:
    candles: list[dict[str, float | int | str]] = []
    for ts, row in data[["open", "high", "low", "close"]].iterrows():
        point: dict[str, float | int | str] = {
            "time": _timestamp_seconds(ts),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
        }
        if zone_colors is not None and ts in zone_colors.index and pd.notna(zone_colors.loc[ts]):
            color = str(zone_colors.loc[ts])
            point["color"] = color
            point["borderColor"] = color
            point["wickColor"] = color
        candles.append(point)
    return candles


def _smma(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1 / period, adjust=False).mean()


def _alligator_series_from_data(data: pd.DataFrame) -> dict[str, list[dict[str, float | int]]]:
    median_price = (data["high"] + data["low"]) / 2
    jaw = _smma(median_price, 13).shift(8)
    teeth = _smma(median_price, 8).shift(5)
    lips = _smma(median_price, 5).shift(3)

    def _to_points(series: pd.Series) -> list[dict[str, float | int]]:
        return [
            {"time": _timestamp_seconds(ts), "value": float(value)}
            for ts, value in series.items()
            if pd.notna(value)
        ]

    return {"jaw": _to_points(jaw), "teeth": _to_points(teeth), "lips": _to_points(lips)}


def _log_scaled_ao_series(
    data: pd.DataFrame,
    fast_len: int = 5,
    slow_len: int = 34,
    show_pct: bool = True,
    log_base: str = "Natural (ln)",
) -> pd.Series:
    median_price = (data["high"] + data["low"]) / 2
    safe_median = median_price.where(median_price > 0)
    if log_base == "Natural (ln)":
        log_price = np.log(safe_median)
    else:
        log_price = np.log10(safe_median)

    fast_ma = log_price.rolling(fast_len).mean()
    slow_ma = log_price.rolling(slow_len).mean()
    ao_log = fast_ma - slow_ma
    ao_pct = ((np.exp(fast_ma) - np.exp(slow_ma)) / np.exp(slow_ma)) * 100
    return ao_pct if show_pct else ao_log


def _log_scaled_ac_series(ao_series: pd.Series, acc_len: int = 5) -> pd.Series:
    return ao_series - ao_series.rolling(acc_len).mean()


def _histogram_bar_colors(histogram: pd.Series) -> pd.Series:
    previous = histogram.shift(1)
    return pd.Series(
        index=histogram.index,
        data=[
            "#64748b" if pd.isna(v) or pd.isna(p) else "#22c55e" if v >= p else "#ef4444"
            for v, p in zip(histogram, previous, strict=False)
        ],
        dtype="object",
    )


def _histogram_points(histogram: pd.Series, colors: pd.Series) -> list[dict[str, float | int | str]]:
    points: list[dict[str, float | int | str]] = []
    for ts, value in histogram.items():
        if pd.isna(value):
            # Preserve the full timeline so indicator panes stay aligned with candle bars.
            points.append({"time": _timestamp_seconds(ts)})
            continue
        points.append(
            {
                "time": _timestamp_seconds(ts),
                "value": float(value),
                "color": str(colors.loc[ts]),
            }
        )
    return points


def _ao_histogram_from_data(data: pd.DataFrame) -> tuple[list[dict[str, float | int | str]], pd.Series]:
    ao_hist = _log_scaled_ao_series(data, fast_len=5, slow_len=34, show_pct=True, log_base="Natural (ln)")
    ao_colors = _histogram_bar_colors(ao_hist)
    return _histogram_points(ao_hist, ao_colors), ao_colors


def _ac_histogram_from_data(data: pd.DataFrame) -> tuple[list[dict[str, float | int | str]], pd.Series]:
    ao_series = _log_scaled_ao_series(data, fast_len=5, slow_len=34, show_pct=True, log_base="Natural (ln)")
    ac_hist = _log_scaled_ac_series(ao_series, acc_len=5)
    ac_colors = _histogram_bar_colors(ac_hist)
    return _histogram_points(ac_hist, ac_colors), ac_colors


def _williams_zones_colors(ao_colors: pd.Series, ac_colors: pd.Series) -> pd.Series:
    zone_colors = pd.Series("#94a3b8", index=ao_colors.index, dtype="object")
    both_green = (ao_colors == "#22c55e") & (ac_colors == "#22c55e")
    both_red = (ao_colors == "#ef4444") & (ac_colors == "#ef4444")
    zone_colors[both_green] = "#22c55e"
    zone_colors[both_red] = "#ef4444"
    return zone_colors


def _build_base_html(title: str, script_body: str) -> str:
    return f"""<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>{title}</title>
    <style>
      body {{
        margin: 0;
        background: #0f172a;
        color: #e2e8f0;
        font-family: Arial, sans-serif;
      }}
      #header {{
        padding: 10px 16px;
        border-bottom: 1px solid #334155;
        display: flex;
        justify-content: space-between;
        align-items: center;
      }}
      #chart-stack {{
        width: 100%;
        height: calc(100vh - 52px);
        display: flex;
        flex-direction: column;
        gap: 6px;
        padding: 6px;
        box-sizing: border-box;
      }}
      .chart-pane {{
        position: relative;
        border: 1px solid #334155;
        border-radius: 8px;
        overflow: hidden;
        background: #0b1220;
      }}
      .ohlc-overlay {{
        position: absolute;
        top: 8px;
        left: 10px;
        z-index: 10;
        pointer-events: none;
        font-size: 12px;
        font-weight: 600;
        color: #e2e8f0;
        background: rgba(11, 18, 32, 0.72);
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 4px 8px;
      }}
      #price-pane {{
        flex: 7;
      }}
      #ao-pane,
      #ac-pane {{
        flex: 2;
      }}
      select {{
        background: #0b1220;
        color: #e2e8f0;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 4px 8px;
      }}
    </style>
  </head>
  <body>
    {script_body}
  </body>
</html>
"""


def _chart_library_script_tag() -> str:
    """Pin a stable Lightweight Charts version that matches our chart API usage."""
    return '<script src="https://unpkg.com/lightweight-charts@4.2.3/dist/lightweight-charts.standalone.production.js"></script>'


def _set_markers_js(series_var: str, markers_var: str) -> str:
    """Set markers across Lightweight Charts major versions."""
    return f"""
      if (typeof {series_var}.setMarkers === 'function') {{
        {series_var}.setMarkers({markers_var});
      }} else if (typeof LightweightCharts.createSeriesMarkers === 'function') {{
        LightweightCharts.createSeriesMarkers({series_var}, {markers_var});
      }}
    """


def generate_local_tradingview_chart(
    data: pd.DataFrame,
    trades: list[Trade],
    output_path: str,
    execution_events: list[ExecutionEvent] | None = None,
    title: str = "Backtest Local TradingView Chart",
    *,
    include_ao: bool = True,
    include_ac: bool = True,
    include_gator: bool = True,
    include_bearish_wiseman: bool = True,
    include_bullish_wiseman: bool = True,
    second_fill_prices: pd.Series | None = None,
    third_fill_prices: pd.Series | None = None,
    second_setup_side: pd.Series | None = None,
    second_fill_side: pd.Series | None = None,
    third_fill_side: pd.Series | None = None,
    third_setup_side: pd.Series | None = None,
    first_setup_side: pd.Series | None = None,
    first_ignored_reason: pd.Series | None = None,
    first_reversal_side: pd.Series | None = None,
    disambiguate_bw_add_on_markers: bool = False,
) -> str:
    """Generate a local HTML chart powered by TradingView Lightweight Charts."""
    required = {"open", "high", "low", "close"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required OHLC columns for chart: {sorted(missing)}")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a pandas DatetimeIndex")

    ao_json_payload, ao_colors = _ao_histogram_from_data(data)
    ac_json_payload, ac_colors = _ac_histogram_from_data(data)
    zone_colors = _williams_zones_colors(ao_colors, ac_colors) if include_ao and include_ac else None
    candles_json = json.dumps(_candles_from_data(data, zone_colors=zone_colors))
    trade_event_lines = _execution_event_lines(execution_events, data.index) if execution_events else _trade_event_lines(trades, data.index)
    trade_paths = _execution_trade_path_lines(execution_events, data.index) if execution_events else _trade_entry_exit_lines(trades, data.index)
    trade_event_lines_json = json.dumps(trade_event_lines)
    trade_entry_exit_lines_json = json.dumps(trade_paths)
    engine_first_wiseman_markers = _first_wiseman_engine_markers(
        data,
        first_setup_side,
        first_ignored_reason,
        first_reversal_side,
        include_bearish_wiseman=include_bearish_wiseman,
        include_bullish_wiseman=include_bullish_wiseman,
    )
    if engine_first_wiseman_markers:
        wiseman_markers = {"bearish": [], "bullish": []}
    else:
        wiseman_markers = _wiseman_markers(
            data,
            include_bearish_wiseman=include_bearish_wiseman,
            include_bullish_wiseman=include_bullish_wiseman,
        )
    second_wiseman_markers = _second_wiseman_markers(data, second_fill_prices, second_setup_side)
    fractal_markers = _valid_third_wiseman_fractal_markers(data, third_setup_side)
    second_entry_markers = _wiseman_fill_entry_markers(
        data,
        second_fill_prices,
        second_fill_side,
        label="2W" if disambiguate_bw_add_on_markers else None,
    )
    third_entry_markers = _wiseman_fill_entry_markers(
        data,
        third_fill_prices,
        third_fill_side,
        label="3W" if disambiguate_bw_add_on_markers else None,
    )
    execution_markers = _execution_event_markers(execution_events, data) if execution_events else _trade_execution_markers(trades, data)
    ignored_first_markers = _first_wiseman_ignored_markers(data, first_setup_side, first_ignored_reason)
    combined_markers_json = json.dumps(
        _combine_markers(
            wiseman_markers["bearish"],
            wiseman_markers["bullish"],
            engine_first_wiseman_markers,
            ignored_first_markers,
            second_wiseman_markers,
            fractal_markers,
            second_entry_markers,
            third_entry_markers,
            execution_markers,
        )
    )
    alligator_json = json.dumps(_alligator_series_from_data(data))
    ao_json = json.dumps(ao_json_payload)
    ac_json = json.dumps(ac_json_payload)

    script_body = f"""
    <div id="header"><span>{title}</span></div>
    <div id="chart-stack">
      <div id="price-pane" class="chart-pane"><div id="price-ohlc" class="ohlc-overlay"></div></div>
      <div id="ao-pane" class="chart-pane"></div>
      <div id="ac-pane" class="chart-pane"></div>
    </div>
    {_chart_library_script_tag()}
    <script>
      const makeChart = (container) => LightweightCharts.createChart(container, {{
        layout: {{ background: {{ color: '#0f172a' }}, textColor: '#cbd5e1' }},
        grid: {{ vertLines: {{ color: '#1e293b' }}, horzLines: {{ color: '#1e293b' }} }},
        rightPriceScale: {{ borderColor: '#334155' }},
        timeScale: {{ borderColor: '#334155', timeVisible: true, secondsVisible: false }},
        crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
      }});

      const priceChart = makeChart(document.getElementById('price-pane'));
      const aoChart = makeChart(document.getElementById('ao-pane'));
      const acChart = makeChart(document.getElementById('ac-pane'));
      const ohlcOverlay = document.getElementById('price-ohlc');
      const formatPrice = (value) => Number.isFinite(Number(value)) ? Number(value).toFixed(4) : '—';
      const formatOhlc = (bar) =>
        `O ${{formatPrice(bar.open)}}  H ${{formatPrice(bar.high)}}  L ${{formatPrice(bar.low)}}  C ${{formatPrice(bar.close)}}`;
      const candlesByTime = new Map();
      const cacheCandlesByTime = (candles) => {{
        candlesByTime.clear();
        candles.forEach((bar) => candlesByTime.set(String(bar.time), bar));
      }};
      const renderOhlc = (bar) => {{
        ohlcOverlay.textContent = bar ? formatOhlc(bar) : 'O —  H —  L —  C —';
      }};

      const candleSeries = typeof priceChart.addCandlestickSeries === 'function'
        ? priceChart.addCandlestickSeries({{
            upColor: '#22c55e', downColor: '#ef4444', borderVisible: false, wickUpColor: '#22c55e', wickDownColor: '#ef4444'
          }})
        : priceChart.addSeries(LightweightCharts.CandlestickSeries, {{
            upColor: '#22c55e', downColor: '#ef4444', borderVisible: false, wickUpColor: '#22c55e', wickDownColor: '#ef4444'
          }});

      let tradeEventLineSeries = [];
      const renderTradeEventLines = (eventLines) => {{
        tradeEventLineSeries.forEach((series) => priceChart.removeSeries(series));
        tradeEventLineSeries = [];
        (eventLines || []).forEach((eventLine) => {{
          const series = priceChart.addLineSeries({{
            color: eventLine.color,
            lineWidth: 2,
            lineStyle: 2,
            lastValueVisible: false,
            priceLineVisible: false,
            crosshairMarkerVisible: false,
            title: eventLine.label,
          }});
          series.setData(eventLine.points);
          tradeEventLineSeries.push(series);
        }});
      }};

      let tradePathLineSeries = [];
      const renderTradePathLines = (tradePaths) => {{
        tradePathLineSeries.forEach((series) => priceChart.removeSeries(series));
        tradePathLineSeries = [];
        (tradePaths || []).forEach((tradePath) => {{
          const series = priceChart.addLineSeries({{
            color: tradePath.color,
            lineWidth: 2,
            lastValueVisible: false,
            priceLineVisible: false,
            crosshairMarkerVisible: false,
            title: tradePath.label,
          }});
          series.setData(tradePath.points);
          tradePathLineSeries.push(series);
        }});
      }};

      const alligator = {alligator_json};
      const includeGator = {str(include_gator).lower()};
      const includeAO = {str(include_ao).lower()};
      const includeAC = {str(include_ac).lower()};

      const jawSeries = includeGator
        ? priceChart.addLineSeries({{ color: '#3b82f6', lineWidth: 2, title: 'Alligator Jaw (13, shift 8)' }})
        : null;
      const teethSeries = includeGator
        ? priceChart.addLineSeries({{ color: '#ef4444', lineWidth: 2, title: 'Alligator Teeth (8, shift 5)' }})
        : null;
      const lipsSeries = includeGator
        ? priceChart.addLineSeries({{ color: '#22c55e', lineWidth: 2, title: 'Alligator Lips (5, shift 3)' }})
        : null;
      const aoSeries = includeAO ? aoChart.addHistogramSeries({{
        title: 'AO Histogram (5,34) Log-Scaled %',
        priceFormat: {{ type: 'price', precision: 4, minMove: 0.0001 }},
      }}) : null;
      const acSeries = includeAC ? acChart.addHistogramSeries({{
        title: 'Williams AC Histogram (5,34,5) Log-Scaled %',
        priceFormat: {{ type: 'price', precision: 4, minMove: 0.0001 }},
      }}) : null;
      const aoZero = includeAO ? aoChart.addLineSeries({{ color: '#64748b', lineWidth: 1, lineStyle: 2 }}) : null;
      const acZero = includeAC ? acChart.addLineSeries({{ color: '#64748b', lineWidth: 1, lineStyle: 2 }}) : null;

      const candlesData = {candles_json};
      candleSeries.setData(candlesData);
      cacheCandlesByTime(candlesData);
      renderOhlc(candlesData.length ? candlesData[candlesData.length - 1] : null);
      if (jawSeries && teethSeries && lipsSeries) {{
        jawSeries.setData(alligator.jaw);
        teethSeries.setData(alligator.teeth);
        lipsSeries.setData(alligator.lips);
      }}
      if (aoSeries && aoZero) {{
        aoSeries.setData({ao_json});
        aoZero.setData({ao_json}.map((point) => ({{ time: point.time, value: 0 }})));
      }}
      if (acSeries && acZero) {{
        acSeries.setData({ac_json});
        acZero.setData({ac_json}.map((point) => ({{ time: point.time, value: 0 }})));
      }}
      renderTradeEventLines({trade_event_lines_json});
      renderTradePathLines({trade_entry_exit_lines_json});
      {_set_markers_js('candleSeries', combined_markers_json)}
      priceChart.subscribeCrosshairMove((param) => {{
        const seriesData = param && param.seriesData ? param.seriesData : null;
        const hoveredBar = seriesData && typeof seriesData.get === 'function' ? seriesData.get(candleSeries) : null;
        if (hoveredBar && hoveredBar.open !== undefined) {{
          renderOhlc(hoveredBar);
          return;
        }}
        if (param && param.time !== undefined) {{
          renderOhlc(candlesByTime.get(String(param.time)) || null);
          return;
        }}
        renderOhlc(candlesData.length ? candlesData[candlesData.length - 1] : null);
      }});

      const syncCharts = [priceChart, aoChart, acChart];
      syncCharts.forEach((sourceChart) => {{
        sourceChart.timeScale().subscribeVisibleLogicalRangeChange((range) => {{
          if (!range) return;
          syncCharts.forEach((chart) => {{
            if (chart !== sourceChart) chart.timeScale().setVisibleLogicalRange(range);
          }});
        }});
      }});

      aoChart.timeScale().applyOptions({{ visible: false }});
      acChart.timeScale().applyOptions({{ visible: true }});

      const resizeCharts = () => {{
        const width = window.innerWidth - 12;
        const chartHeight = window.innerHeight - 64;
        const priceHeight = Math.floor(chartHeight * 0.64);
        const aoHeight = Math.floor(chartHeight * 0.18);
        const acHeight = chartHeight - priceHeight - aoHeight;
        priceChart.applyOptions({{ width, height: priceHeight }});
        aoChart.applyOptions({{ width, height: aoHeight }});
        acChart.applyOptions({{ width, height: acHeight }});
      }};

      window.addEventListener('resize', resizeCharts);
      resizeCharts();
      priceChart.timeScale().fitContent();
    </script>
    """

    output = Path(output_path)
    output.write_text(_build_base_html(title, script_body), encoding="utf-8")
    return str(output)


def generate_batch_local_tradingview_chart(
    batch_result: BatchBacktestResult,
    output_path: str,
    title: str = "Batch Backtest TradingView Chart",
) -> str:
    """Generate a local HTML chart with a selector to switch asset/timeframe runs."""
    datasets = {
        key: {
            "ao": _ao_histogram_from_data(batch_result.run_data[key]),
            "ac": _ac_histogram_from_data(batch_result.run_data[key]),
            "trade_event_lines": _execution_event_lines(batch_result.run_results[key].execution_events, batch_result.run_data[key].index),
            "trade_entry_exit_lines": _execution_trade_path_lines(batch_result.run_results[key].execution_events, batch_result.run_data[key].index),
            "markers": _combine_markers(
                _bearish_first_wiseman_markers(batch_result.run_data[key]),
                _bullish_first_wiseman_markers(batch_result.run_data[key]),
                _trade_execution_markers(batch_result.run_results[key].trades, batch_result.run_data[key]),
            ),
            "alligator": _alligator_series_from_data(batch_result.run_data[key]),
        }
        for key in batch_result.run_results
    }
    for key in datasets:
        ao_payload, ao_colors = datasets[key]["ao"]
        ac_payload, ac_colors = datasets[key]["ac"]
        run_data = batch_result.run_data[key]
        datasets[key]["ao"] = ao_payload
        datasets[key]["ac"] = ac_payload
        datasets[key]["candles"] = _candles_from_data(run_data, zone_colors=_williams_zones_colors(ao_colors, ac_colors))

    keys = sorted(datasets.keys())
    default_key = keys[0]
    datasets_json = json.dumps(datasets)
    options_html = "".join([f"<option value=\"{k}\">{k}</option>" for k in keys])

    script_body = f"""
    <div id="header">
      <span>{title}</span>
      <label>View: <select id="runSelect">{options_html}</select></label>
    </div>
    <div id="chart-stack">
      <div id="price-pane" class="chart-pane"><div id="price-ohlc" class="ohlc-overlay"></div></div>
      <div id="ao-pane" class="chart-pane"></div>
      <div id="ac-pane" class="chart-pane"></div>
    </div>
    {_chart_library_script_tag()}
    <script>
      const datasets = {datasets_json};
      const runSelect = document.getElementById('runSelect');
      const makeChart = (container) => LightweightCharts.createChart(container, {{
        layout: {{ background: {{ color: '#0f172a' }}, textColor: '#cbd5e1' }},
        grid: {{ vertLines: {{ color: '#1e293b' }}, horzLines: {{ color: '#1e293b' }} }},
        rightPriceScale: {{ borderColor: '#334155' }},
        timeScale: {{ borderColor: '#334155', timeVisible: true, secondsVisible: false }},
        crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
      }});

      const priceChart = makeChart(document.getElementById('price-pane'));
      const aoChart = makeChart(document.getElementById('ao-pane'));
      const acChart = makeChart(document.getElementById('ac-pane'));
      const ohlcOverlay = document.getElementById('price-ohlc');
      const formatPrice = (value) => Number.isFinite(Number(value)) ? Number(value).toFixed(4) : '—';
      const formatOhlc = (bar) =>
        `O ${{formatPrice(bar.open)}}  H ${{formatPrice(bar.high)}}  L ${{formatPrice(bar.low)}}  C ${{formatPrice(bar.close)}}`;
      const candlesByTime = new Map();
      const cacheCandlesByTime = (candles) => {{
        candlesByTime.clear();
        candles.forEach((bar) => candlesByTime.set(String(bar.time), bar));
      }};
      const renderOhlc = (bar) => {{
        ohlcOverlay.textContent = bar ? formatOhlc(bar) : 'O —  H —  L —  C —';
      }};

      const candleSeries = typeof priceChart.addCandlestickSeries === 'function'
        ? priceChart.addCandlestickSeries({{
            upColor: '#22c55e', downColor: '#ef4444', borderVisible: false, wickUpColor: '#22c55e', wickDownColor: '#ef4444'
          }})
        : priceChart.addSeries(LightweightCharts.CandlestickSeries, {{
            upColor: '#22c55e', downColor: '#ef4444', borderVisible: false, wickUpColor: '#22c55e', wickDownColor: '#ef4444'
          }});
      let tradeEventLineSeries = [];
      const renderTradeEventLines = (eventLines) => {{
        tradeEventLineSeries.forEach((series) => priceChart.removeSeries(series));
        tradeEventLineSeries = [];
        (eventLines || []).forEach((eventLine) => {{
          const series = priceChart.addLineSeries({{
            color: eventLine.color,
            lineWidth: 2,
            lineStyle: 2,
            lastValueVisible: false,
            priceLineVisible: false,
            crosshairMarkerVisible: false,
            title: eventLine.label,
          }});
          series.setData(eventLine.points);
          tradeEventLineSeries.push(series);
        }});
      }};
      let tradePathLineSeries = [];
      const renderTradePathLines = (tradePaths) => {{
        tradePathLineSeries.forEach((series) => priceChart.removeSeries(series));
        tradePathLineSeries = [];
        (tradePaths || []).forEach((tradePath) => {{
          const series = priceChart.addLineSeries({{
            color: tradePath.color,
            lineWidth: 2,
            lastValueVisible: false,
            priceLineVisible: false,
            crosshairMarkerVisible: false,
            title: tradePath.label,
          }});
          series.setData(tradePath.points);
          tradePathLineSeries.push(series);
        }});
      }};
      const jawSeries = priceChart.addLineSeries({{ color: '#3b82f6', lineWidth: 2, title: 'Alligator Jaw (13, shift 8)' }});
      const teethSeries = priceChart.addLineSeries({{ color: '#ef4444', lineWidth: 2, title: 'Alligator Teeth (8, shift 5)' }});
      const lipsSeries = priceChart.addLineSeries({{ color: '#22c55e', lineWidth: 2, title: 'Alligator Lips (5, shift 3)' }});
      const aoSeries = aoChart.addHistogramSeries({{
        title: 'AO Histogram (5,34) Log-Scaled %',
        priceFormat: {{ type: 'price', precision: 4, minMove: 0.0001 }},
      }});
      const acSeries = acChart.addHistogramSeries({{
        title: 'Williams AC Histogram (5,34,5) Log-Scaled %',
        priceFormat: {{ type: 'price', precision: 4, minMove: 0.0001 }},
      }});
      const aoZero = aoChart.addLineSeries({{ color: '#64748b', lineWidth: 1, lineStyle: 2 }});
      const acZero = acChart.addLineSeries({{ color: '#64748b', lineWidth: 1, lineStyle: 2 }});

      function renderRun(key) {{
        const payload = datasets[key];
        candleSeries.setData(payload.candles);
        cacheCandlesByTime(payload.candles);
        renderOhlc(payload.candles.length ? payload.candles[payload.candles.length - 1] : null);
        jawSeries.setData(payload.alligator.jaw);
        teethSeries.setData(payload.alligator.teeth);
        lipsSeries.setData(payload.alligator.lips);
        aoSeries.setData(payload.ao);
        acSeries.setData(payload.ac);
        aoZero.setData(payload.ao.map((point) => ({{ time: point.time, value: 0 }})));
        acZero.setData(payload.ac.map((point) => ({{ time: point.time, value: 0 }})));
        renderTradeEventLines(payload.trade_event_lines);
        renderTradePathLines(payload.trade_entry_exit_lines);
        {_set_markers_js('candleSeries', 'payload.markers')}
        priceChart.timeScale().fitContent();
      }}
      priceChart.subscribeCrosshairMove((param) => {{
        const seriesData = param && param.seriesData ? param.seriesData : null;
        const hoveredBar = seriesData && typeof seriesData.get === 'function' ? seriesData.get(candleSeries) : null;
        if (hoveredBar && hoveredBar.open !== undefined) {{
          renderOhlc(hoveredBar);
          return;
        }}
        if (param && param.time !== undefined) {{
          renderOhlc(candlesByTime.get(String(param.time)) || null);
          return;
        }}
        const selectedKey = runSelect.value;
        const candles = datasets[selectedKey] ? datasets[selectedKey].candles : [];
        renderOhlc(candles.length ? candles[candles.length - 1] : null);
      }});

      const syncCharts = [priceChart, aoChart, acChart];
      syncCharts.forEach((sourceChart) => {{
        sourceChart.timeScale().subscribeVisibleLogicalRangeChange((range) => {{
          if (!range) return;
          syncCharts.forEach((chart) => {{
            if (chart !== sourceChart) chart.timeScale().setVisibleLogicalRange(range);
          }});
        }});
      }});

      aoChart.timeScale().applyOptions({{ visible: false }});
      acChart.timeScale().applyOptions({{ visible: true }});

      const resizeCharts = () => {{
        const width = window.innerWidth - 12;
        const chartHeight = window.innerHeight - 64;
        const priceHeight = Math.floor(chartHeight * 0.64);
        const aoHeight = Math.floor(chartHeight * 0.18);
        const acHeight = chartHeight - priceHeight - aoHeight;
        priceChart.applyOptions({{ width, height: priceHeight }});
        aoChart.applyOptions({{ width, height: aoHeight }});
        acChart.applyOptions({{ width, height: acHeight }});
      }};

      runSelect.addEventListener('change', (e) => renderRun(e.target.value));
      runSelect.value = '{default_key}';
      renderRun('{default_key}');
      window.addEventListener('resize', resizeCharts);
      resizeCharts();
    </script>
    """

    output = Path(output_path)
    output.write_text(_build_base_html(title, script_body), encoding="utf-8")
    return str(output)
