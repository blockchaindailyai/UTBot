from __future__ import annotations

import argparse
from pathlib import Path

from backtesting import (
    generate_first_wiseman_bearish_pinescript,
    generate_first_wiseman_bullish_pinescript,
    generate_local_tradingview_chart,
    load_ohlcv_csv,
    summarize_wiseman_markers,
)

AVAILABLE_INDICATORS = {
    "ao": "ao",
    "ac": "ac",
    "gator": "gator",
    "bearish-wiseman": "bearish-wiseman",
    "bullish-wiseman": "bullish-wiseman",
}


def _parse_disabled_indicators(disabled: list[str] | None) -> set[str]:
    disabled_set: set[str] = set()
    for indicator in disabled or []:
        normalized = indicator.strip().lower()
        if normalized not in AVAILABLE_INDICATORS:
            supported = ", ".join(sorted(AVAILABLE_INDICATORS.values()))
            raise ValueError(
                f"Unsupported indicator '{indicator}'. Supported values: {supported}"
            )
        disabled_set.add(AVAILABLE_INDICATORS[normalized])
    return disabled_set


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an indicator-only dry-run chart without trade markers/clutter."
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="artifacts_dry_run")
    parser.add_argument(
        "--disable",
        action="append",
        metavar="INDICATOR",
        help=(
            "Disable one indicator. Can be repeated. Supported values: "
            "AO, AC, gator, bearish-wiseman, bullish-wiseman"
        ),
    )
    args = parser.parse_args()

    disabled = _parse_disabled_indicators(args.disable)
    include_ao = "ao" not in disabled
    include_ac = "ac" not in disabled
    include_gator = "gator" not in disabled
    include_bearish_wiseman = "bearish-wiseman" not in disabled
    include_bullish_wiseman = "bullish-wiseman" not in disabled

    data = load_ohlcv_csv(args.csv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    wiseman_pine_path = out_dir / "first_wiseman_bearish.pine"
    if include_bearish_wiseman:
        generate_first_wiseman_bearish_pinescript(str(wiseman_pine_path))

    bullish_wiseman_pine_path = out_dir / "first_wiseman_bullish.pine"
    if include_bullish_wiseman:
        generate_first_wiseman_bullish_pinescript(str(bullish_wiseman_pine_path))

    wiseman_summary = summarize_wiseman_markers(data)

    local_chart_path = out_dir / "tradingview_indicator_dry_run.html"
    generate_local_tradingview_chart(
        data,
        [],
        str(local_chart_path),
        title="Indicator Dry-Run (No Trade Markers)",
        include_ao=include_ao,
        include_ac=include_ac,
        include_gator=include_gator,
        include_bearish_wiseman=include_bearish_wiseman,
        include_bullish_wiseman=include_bullish_wiseman,
    )

    print("Indicator dry-run complete")
    print(f"Bars: {len(data)}")
    if include_bearish_wiseman:
        print(f"TradingView bearish 1st Wiseman Pine written to: {wiseman_pine_path}")
    if include_bullish_wiseman:
        print(f"TradingView bullish 1st Wiseman Pine written to: {bullish_wiseman_pine_path}")
    print(f"Local TradingView indicator chart written to: {local_chart_path}")
    enabled = [
        indicator
        for indicator in AVAILABLE_INDICATORS.values()
        if indicator not in disabled
    ]
    enabled_summary = ', '.join(enabled) if enabled else '(none)'
    print(f"Enabled indicators: {enabled_summary}")
    if include_bearish_wiseman:
        print(
            "Bearish Wiseman markers on dataset: "
            f"1W={wiseman_summary['bearish_first_wiseman']}, "
            f"1W-R={wiseman_summary['bearish_reverse']}"
        )
        if wiseman_summary['bearish_first_wiseman'] == 0:
            print("Note: bearish-wiseman is enabled, but no bars matched the bearish 1st Wiseman rules.")
    if include_bullish_wiseman:
        print(
            "Bullish Wiseman markers on dataset: "
            f"1W+={wiseman_summary['bullish_first_wiseman']}, "
            f"1W+-R={wiseman_summary['bullish_reverse']}"
        )


if __name__ == "__main__":
    main()
