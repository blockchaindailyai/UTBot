from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000
INTERVAL_TO_MS = {
    "1s": 1_000,
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
    "1M": 2_592_000_000,
}


def parse_utc_timestamp(value: str) -> int:
    """Parse a date/datetime string into epoch milliseconds (UTC)."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def format_utc_timestamp(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def fetch_chunk(
    symbol: str,
    interval: str,
    start_time_ms: int,
    end_time_ms: int,
    pause_seconds: float,
    max_retries: int,
) -> list[list[Any]]:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time_ms,
        "endTime": end_time_ms,
        "limit": MAX_LIMIT,
    }
    query = urlencode(params)
    url = f"{BINANCE_KLINES_URL}?{query}"

    rows: list[list[Any]] | None = None
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with urlopen(url, timeout=30) as response:
                rows = json.loads(response.read().decode("utf-8"))
            break
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to fetch Binance klines after {max_retries} attempts for "
                    f"{format_utc_timestamp(start_time_ms)} -> {format_utc_timestamp(end_time_ms)}"
                ) from last_error
            time.sleep(min(2**attempt, 10))

    if pause_seconds > 0:
        time.sleep(pause_seconds)

    return rows or []


def fetch_klines(
    symbol: str,
    interval: str,
    start_time_ms: int,
    end_time_ms: int,
    pause_seconds: float,
    max_retries: int,
) -> list[list[Any]]:
    """Fetch Binance klines sequentially and return all rows."""
    if interval not in INTERVAL_TO_MS:
        supported = ", ".join(sorted(INTERVAL_TO_MS))
        raise ValueError(f"Unsupported interval '{interval}'. Supported intervals: {supported}")

    symbol = symbol.upper()
    step_ms = INTERVAL_TO_MS[interval]
    chunk_span_ms = step_ms * (MAX_LIMIT - 1)

    all_rows: list[list[Any]] = []
    cursor = start_time_ms
    while cursor <= end_time_ms:
        chunk_end = min(cursor + chunk_span_ms, end_time_ms)
        chunk_rows = fetch_chunk(
            symbol=symbol,
            interval=interval,
            start_time_ms=cursor,
            end_time_ms=chunk_end,
            pause_seconds=pause_seconds,
            max_retries=max_retries,
        )

        all_rows.extend(chunk_rows)
        cursor += step_ms * MAX_LIMIT

    return all_rows


def scan_rows(rows: list[list[Any]], step_ms: int) -> dict[str, Any]:
    """Return a data quality report for kline rows."""
    ordered_times = [int(row[0]) for row in rows if len(row) >= 1]
    unique_times = sorted(set(ordered_times))

    duplicate_counts: dict[int, int] = {}
    for ts in ordered_times:
        duplicate_counts[ts] = duplicate_counts.get(ts, 0) + 1
    duplicate_timestamps = [ts for ts, count in duplicate_counts.items() if count > 1]

    missing_bars: list[int] = []
    gaps: list[tuple[int, int]] = []
    for prev, curr in zip(unique_times, unique_times[1:]):
        delta = curr - prev
        if delta > step_ms:
            gaps.append((prev, curr))
            missing_count = (delta // step_ms) - 1
            for idx in range(1, missing_count + 1):
                missing_bars.append(prev + idx * step_ms)

    non_monotonic_indices: list[int] = []
    for idx in range(1, len(ordered_times)):
        if ordered_times[idx] <= ordered_times[idx - 1]:
            non_monotonic_indices.append(idx)

    malformed_rows: list[tuple[int, str]] = []
    invalid_ohlc_rows: list[tuple[int, str]] = []

    for idx, row in enumerate(rows):
        if len(row) < 6:
            malformed_rows.append((idx, "row has fewer than 6 fields"))
            continue

        try:
            open_p = float(row[1])
            high_p = float(row[2])
            low_p = float(row[3])
            close_p = float(row[4])
            volume = float(row[5])
        except (TypeError, ValueError):
            malformed_rows.append((idx, "OHLCV fields are not numeric"))
            continue

        if high_p < low_p:
            invalid_ohlc_rows.append((idx, "high < low"))
        if high_p < max(open_p, close_p):
            invalid_ohlc_rows.append((idx, "high below open/close"))
        if low_p > min(open_p, close_p):
            invalid_ohlc_rows.append((idx, "low above open/close"))
        if volume < 0:
            invalid_ohlc_rows.append((idx, "negative volume"))

    coverage = None
    if unique_times:
        expected_points = ((unique_times[-1] - unique_times[0]) // step_ms) + 1
        coverage = len(unique_times) / expected_points if expected_points else 1.0

    return {
        "row_count": len(rows),
        "unique_bar_count": len(unique_times),
        "duplicate_timestamp_count": len(duplicate_timestamps),
        "duplicate_timestamps": duplicate_timestamps,
        "missing_bar_count": len(missing_bars),
        "missing_bars": missing_bars,
        "gap_count": len(gaps),
        "gaps": gaps,
        "non_monotonic_count": len(non_monotonic_indices),
        "non_monotonic_indices": non_monotonic_indices,
        "malformed_row_count": len(malformed_rows),
        "malformed_rows": malformed_rows,
        "invalid_ohlc_row_count": len(invalid_ohlc_rows),
        "invalid_ohlc_rows": invalid_ohlc_rows,
        "coverage_ratio": coverage,
    }


def dedupe_and_sort(rows: list[list[Any]]) -> list[list[Any]]:
    deduped: dict[int, list[Any]] = {}
    for row in rows:
        if not row:
            continue
        deduped[int(row[0])] = row
    return [deduped[key] for key in sorted(deduped)]


def print_scan_report(report: dict[str, Any], step_ms: int) -> None:
    print("\n=== Binance OHLCV Scan Report ===")
    print(f"Rows fetched: {report['row_count']}")
    print(f"Unique bars: {report['unique_bar_count']}")
    print(f"Duplicates: {report['duplicate_timestamp_count']}")
    print(f"Missing bars: {report['missing_bar_count']}")
    print(f"Timestamp gaps: {report['gap_count']}")
    print(f"Non-monotonic positions: {report['non_monotonic_count']}")
    print(f"Malformed rows: {report['malformed_row_count']}")
    print(f"Invalid OHLC rows: {report['invalid_ohlc_row_count']}")

    coverage_ratio = report["coverage_ratio"]
    if coverage_ratio is not None:
        print(f"Coverage ratio (unique/expected): {coverage_ratio:.4f}")

    if report["duplicate_timestamps"]:
        preview = report["duplicate_timestamps"][:5]
        formatted = ", ".join(format_utc_timestamp(ts) for ts in preview)
        print(f"Duplicate timestamp examples ({len(preview)} shown): {formatted}")

    if report["gaps"]:
        preview = report["gaps"][:3]
        gap_strings = []
        for start, end in preview:
            missing_between = (end - start) // step_ms - 1
            gap_strings.append(
                f"{format_utc_timestamp(start)} -> {format_utc_timestamp(end)} (missing {missing_between})"
            )
        print(f"Gap examples ({len(preview)} shown): {'; '.join(gap_strings)}")

    if report["malformed_rows"]:
        preview = report["malformed_rows"][:3]
        msg = "; ".join(f"row#{idx}: {reason}" for idx, reason in preview)
        print(f"Malformed row examples ({len(preview)} shown): {msg}")

    if report["invalid_ohlc_rows"]:
        preview = report["invalid_ohlc_rows"][:3]
        msg = "; ".join(f"row#{idx}: {reason}" for idx, reason in preview)
        print(f"Invalid OHLC examples ({len(preview)} shown): {msg}")


def write_rows_to_csv(rows: list[list[Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for row in rows:
            writer.writerow(
                [
                    format_utc_timestamp(int(row[0])),
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Binance OHLCV data into backtesting CSV format")
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance symbol (e.g. BTCUSDT, ETHUSDT)")
    parser.add_argument("--interval", default="1h", help="Binance kline interval (e.g. 1m, 5m, 1h, 1d)")
    parser.add_argument(
        "--start",
        default="2017-01-01T00:00:00+00:00",
        help="Inclusive start datetime in ISO format (default: 2017-01-01 UTC)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Inclusive end datetime in ISO format (default: now UTC)",
    )
    parser.add_argument("--out", default="examples/BINANCE_BTCUSDT_1h.csv", help="Output CSV path")
    parser.add_argument("--pause-seconds", type=float, default=0.15, help="Sleep between API requests")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries per API request")
    args = parser.parse_args()

    start_ms = parse_utc_timestamp(args.start)
    end_ms = parse_utc_timestamp(args.end) if args.end else int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    if start_ms > end_ms:
        raise ValueError("--start must be before or equal to --end")

    interval = args.interval
    if interval not in INTERVAL_TO_MS:
        supported = ", ".join(sorted(INTERVAL_TO_MS))
        raise ValueError(f"Unsupported interval '{interval}'. Supported intervals: {supported}")

    rows = fetch_klines(
        symbol=args.symbol,
        interval=interval,
        start_time_ms=start_ms,
        end_time_ms=end_ms,
        pause_seconds=max(args.pause_seconds, 0.0),
        max_retries=max(args.max_retries, 1),
    )
    if not rows:
        raise RuntimeError("No candles returned. Check symbol/interval/date range.")

    step_ms = INTERVAL_TO_MS[interval]
    scan_report = scan_rows(rows, step_ms=step_ms)
    clean_rows = dedupe_and_sort(rows)

    output_path = Path(args.out)
    write_rows_to_csv(clean_rows, output_path)

    print(f"Downloaded {len(rows)} raw candles for {args.symbol.upper()} {args.interval}")
    print(f"Rows written after dedupe/sort: {len(clean_rows)}")
    print(f"Date range: {format_utc_timestamp(int(clean_rows[0][0]))} -> {format_utc_timestamp(int(clean_rows[-1][0]))}")
    print(f"CSV written to: {output_path}")
    print_scan_report(scan_report, step_ms=step_ms)


if __name__ == "__main__":
    main()
