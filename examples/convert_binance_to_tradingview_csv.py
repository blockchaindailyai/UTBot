from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path


INPUT_TIMESTAMP_COLUMNS = ("timestamp", "time")
INPUT_VOLUME_COLUMNS = ("volume", "Volume")
OUTPUT_HEADER = ["time", "open", "high", "low", "close", "Volume"]


def _format_time_as_utc_z(value: str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_decimal(value: str) -> str:
    try:
        as_decimal = Decimal(value)
    except (InvalidOperation, ValueError):
        return value

    normalized = format(as_decimal, "f").rstrip("0").rstrip(".")
    if normalized in ("", "-0"):
        return "0"
    return normalized


def convert_binance_csv_to_tradingview(source_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with source_path.open("r", newline="", encoding="utf-8") as src_handle:
        reader = csv.DictReader(src_handle)
        if not reader.fieldnames:
            raise ValueError("Input CSV is missing a header row")

        field_set = set(reader.fieldnames)
        time_col = next((col for col in INPUT_TIMESTAMP_COLUMNS if col in field_set), None)
        volume_col = next((col for col in INPUT_VOLUME_COLUMNS if col in field_set), None)

        if time_col is None:
            raise ValueError("Input CSV must contain either 'timestamp' or 'time'")
        if volume_col is None:
            raise ValueError("Input CSV must contain either 'volume' or 'Volume'")

        required_price_cols = ["open", "high", "low", "close"]
        missing = [col for col in required_price_cols if col not in field_set]
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")

        with output_path.open("w", newline="", encoding="utf-8") as out_handle:
            writer = csv.writer(out_handle)
            writer.writerow(OUTPUT_HEADER)

            row_count = 0
            for row in reader:
                writer.writerow(
                    [
                        _format_time_as_utc_z(row[time_col]),
                        _normalize_decimal(row["open"]),
                        _normalize_decimal(row["high"]),
                        _normalize_decimal(row["low"]),
                        _normalize_decimal(row["close"]),
                        _normalize_decimal(row[volume_col]),
                    ]
                )
                row_count += 1

    return row_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Binance OHLCV CSV into TradingView-friendly OHLCV CSV format"
    )
    parser.add_argument("--input", required=True, help="Path to Binance CSV input")
    parser.add_argument("--output", required=True, help="Path to converted TradingView CSV output")
    args = parser.parse_args()

    source_path = Path(args.input)
    output_path = Path(args.output)

    rows_written = convert_binance_csv_to_tradingview(source_path, output_path)
    print(f"Converted {rows_written} rows")
    print(f"Input: {source_path}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
