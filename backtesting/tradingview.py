from __future__ import annotations

from datetime import timezone
from pathlib import Path

from .engine import Trade


def _tv_timestamp_ms(ts) -> int:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.tz_convert(timezone.utc).timestamp() * 1000)


def generate_trade_marker_pinescript(
    trades: list[Trade],
    output_path: str,
    title: str = "Backtest Trade Markers",
) -> str:
    """Generate a Pine Script that plots entry/exit markers for completed trades."""
    entry_times = [_tv_timestamp_ms(t.entry_time) for t in trades]
    exit_times = [_tv_timestamp_ms(t.exit_time) for t in trades]
    sides = [1 if t.side == "long" else -1 for t in trades]

    def fmt_array(values: list[int]) -> str:
        if not values:
            return "array.from()"
        return "array.from(" + ", ".join(str(v) for v in values) + ")"

    pine = f'''//@version=5
indicator("{title}", overlay=true, max_labels_count=500)

var entryTimes = {fmt_array(entry_times)}
var exitTimes = {fmt_array(exit_times)}
var sides = {fmt_array(sides)}

isEntry = false
isExit = false
isLong = false

for i = 0 to array.size(entryTimes) - 1
    if time == array.get(entryTimes, i)
        isEntry := true
        isLong := array.get(sides, i) == 1

for i = 0 to array.size(exitTimes) - 1
    if time == array.get(exitTimes, i)
        isExit := true
        isLong := array.get(sides, i) == 1

plotshape(isEntry and isLong, title="Long Entry", style=shape.triangleup, color=color.new(color.green, 0), location=location.belowbar, size=size.small, text="LE")
plotshape(isExit and isLong, title="Long Exit", style=shape.triangledown, color=color.new(color.green, 0), location=location.abovebar, size=size.small, text="LX")
plotshape(isEntry and not isLong, title="Short Entry", style=shape.triangledown, color=color.new(color.red, 0), location=location.abovebar, size=size.small, text="SE")
plotshape(isExit and not isLong, title="Short Exit", style=shape.triangleup, color=color.new(color.red, 0), location=location.belowbar, size=size.small, text="SX")
'''

    output = Path(output_path)
    output.write_text(pine, encoding="utf-8")
    return str(output)


def generate_first_wiseman_bearish_pinescript(
    output_path: str,
    title: str = "1st Wiseman Detector (Bearish)",
) -> str:
    """Generate a Pine Script indicator for bearish 1st Wiseman detection."""
    pine = f'''//@version=5
indicator("{title}", overlay=true, max_labels_count=500)

// Bill Williams Alligator defaults
jawLength = input.int(13, "Jaw Length")
jawShift = input.int(8, "Jaw Shift")
teethLength = input.int(8, "Teeth Length")
teethShift = input.int(5, "Teeth Shift")
lipsLength = input.int(5, "Lips Length")
lipsShift = input.int(3, "Lips Shift")

// Gator width filter

medianPrice = hl2
jaw = ta.smma(medianPrice, jawLength)[jawShift]
teeth = ta.smma(medianPrice, teethLength)[teethShift]
lips = ta.smma(medianPrice, lipsLength)[lipsShift]

gatorOpenUp = lips > teeth and teeth > jaw
candidateMidpoint = (high[1] + low[1]) / 2.0
gatorWidthValid = math.abs(teeth[1] - lips[1]) < math.abs(lips[1] - candidateMidpoint)

// AO as classic Bill Williams AO (SMA5-SMA34 on median price)
ao = ta.sma(medianPrice, 5) - ta.sma(medianPrice, 34)
aoGreen = ao > ao[1]

// Candidate bearish 1st wiseman: pivot high candle with bearish body and filters on the candidate bar.
isLocalPeak = high[1] > high[2] and high[1] > high
candidateBearishBody = open[1] > close[1]
candidateQualifies = isLocalPeak and candidateBearishBody and gatorOpenUp[1] and gatorWidthValid and aoGreen[1]

var bool activeCandidate = false
var int candidateIndex = na
var float candidateHigh = na
var float candidateLow = na
var bool reverseWatchActive = false
var int reverseSourceIndex = na
var float reverseSourceHigh = na
var int reverseConfirmedOnIndex = na

labelOffsetTicks = input.int(20, "1W Label Offset (ticks)", minval=1)
labelOffset = syminfo.mintick * labelOffsetTicks

// New pivot candidate (bar [1] is now confirmed as a local high)
if candidateQualifies
    activeCandidate := true
    candidateIndex := bar_index - 1
    candidateHigh := high[1]
    candidateLow := low[1]

bearishWiseman = false
bullishReverseSignal = false

if activeCandidate
    // Cancel if high is broken first
    if high > candidateHigh
        activeCandidate := false
        candidateIndex := na
        candidateHigh := na
        candidateLow := na
    // Confirm if low is broken before any high break
    else if low < candidateLow
        bearishWiseman := true
        label.new(candidateIndex, candidateHigh + labelOffset, "1W", style=label.style_triangledown, color=color.new(color.red, 0), textcolor=color.white, size=size.small)
        reverseWatchActive := true
        reverseSourceIndex := candidateIndex
        reverseSourceHigh := candidateHigh
        reverseConfirmedOnIndex := bar_index
        activeCandidate := false
        candidateIndex := na
        candidateHigh := na
        candidateLow := na

if reverseWatchActive and bar_index >= reverseConfirmedOnIndex + 3 and high > reverseSourceHigh
    bullishReverseSignal := true
    label.new(bar_index, low, "", style=label.style_triangleup, color=color.new(color.green, 0), textcolor=color.new(color.green, 0), size=size.small)
    reverseWatchActive := false
    reverseSourceIndex := na
    reverseSourceHigh := na
    reverseConfirmedOnIndex := na

alertcondition(bearishWiseman, "Bearish 1st Wiseman", "Bearish 1st Wiseman detected")
alertcondition(bullishReverseSignal, "Bullish Reverse After Bearish 1st Wiseman", "Bullish reverse signal after bearish 1st Wiseman")
'''

    output = Path(output_path)
    output.write_text(pine, encoding="utf-8")
    return str(output)


def generate_first_wiseman_bullish_pinescript(
    output_path: str,
    title: str = "1st Wiseman Detector (Bullish)",
) -> str:
    """Generate a Pine Script indicator for bullish 1st Wiseman detection."""
    pine = f'''//@version=5
indicator("{title}", overlay=true, max_labels_count=500)

// Bill Williams Alligator defaults
jawLength = input.int(13, "Jaw Length")
jawShift = input.int(8, "Jaw Shift")
teethLength = input.int(8, "Teeth Length")
teethShift = input.int(5, "Teeth Shift")
lipsLength = input.int(5, "Lips Length")
lipsShift = input.int(3, "Lips Shift")

// Gator width filter

medianPrice = hl2
jaw = ta.smma(medianPrice, jawLength)[jawShift]
teeth = ta.smma(medianPrice, teethLength)[teethShift]
lips = ta.smma(medianPrice, lipsLength)[lipsShift]

gatorOpenDown = lips < teeth and teeth < jaw
candidateMidpoint = (high[1] + low[1]) / 2.0
gatorWidthValid = math.abs(teeth[1] - lips[1]) < math.abs(lips[1] - candidateMidpoint)

// AO as classic Bill Williams AO (SMA5-SMA34 on median price)
ao = ta.sma(medianPrice, 5) - ta.sma(medianPrice, 34)
aoRed = ao < ao[1]

// Candidate bullish 1st wiseman: pivot low candle with bullish body and filters on the candidate bar.
isLocalTrough = low[1] < low[2] and low[1] < low
candidateBullishBody = close[1] > open[1]
candidateQualifies = isLocalTrough and candidateBullishBody and gatorOpenDown[1] and gatorWidthValid and aoRed[1]

var bool activeCandidate = false
var int candidateIndex = na
var float candidateHigh = na
var float candidateLow = na
var bool reverseWatchActive = false
var float reverseSourceLow = na
var int reverseConfirmedOnIndex = na

labelOffsetTicks = input.int(20, "1W Label Offset (ticks)", minval=1)
labelOffset = syminfo.mintick * labelOffsetTicks

// New pivot candidate (bar [1] is now confirmed as a local low)
if candidateQualifies
    activeCandidate := true
    candidateIndex := bar_index - 1
    candidateHigh := high[1]
    candidateLow := low[1]

bullishWiseman = false
bearishReverseSignal = false

if activeCandidate
    // Cancel if low is broken first
    if low < candidateLow
        activeCandidate := false
        candidateIndex := na
        candidateHigh := na
        candidateLow := na
    // Confirm if high is broken before any low break
    else if high > candidateHigh
        bullishWiseman := true
        label.new(candidateIndex, candidateLow - labelOffset, "1W", style=label.style_triangleup, color=color.new(color.green, 0), textcolor=color.white, size=size.small)
        reverseWatchActive := true
        reverseSourceLow := candidateLow
        reverseConfirmedOnIndex := bar_index
        activeCandidate := false
        candidateIndex := na
        candidateHigh := na
        candidateLow := na

if reverseWatchActive and bar_index >= reverseConfirmedOnIndex + 3 and low < reverseSourceLow
    bearishReverseSignal := true
    label.new(bar_index, high, "", style=label.style_triangledown, color=color.new(color.red, 0), textcolor=color.new(color.red, 0), size=size.small)
    reverseWatchActive := false
    reverseSourceLow := na
    reverseConfirmedOnIndex := na

alertcondition(bullishWiseman, "Bullish 1st Wiseman", "Bullish 1st Wiseman detected")
alertcondition(bearishReverseSignal, "Bearish Reverse After Bullish 1st Wiseman", "Bearish reverse signal after bullish 1st Wiseman")
'''

    output = Path(output_path)
    output.write_text(pine, encoding="utf-8")
    return str(output)


def generate_ut_bot_strategy_pinescript(
    output_path: str,
    title: str = "UT Bot Alerts Strategy",
) -> str:
    """Generate a Pine Script strategy based on UT Bot buy/sell alert logic."""
    pine = f'''//@version=5
strategy("{title}", overlay=true, pyramiding=0, process_orders_on_close=true)

// Inputs
a = input.float(1.0, title="Key Value. 'This changes the sensitivity'")
c = input.int(10, title="ATR Period")
h = input.bool(false, title="Signals from Heikin Ashi Candles")

xATR = ta.atr(c)
nLoss = a * xATR

haClose = request.security(ticker.heikinashi(syminfo.tickerid), timeframe.period, close, lookahead=barmerge.lookahead_off)
src = h ? haClose : close

var float xATRTrailingStop = na
prevStop = nz(xATRTrailingStop[1], src)
if src > prevStop and src[1] > prevStop
    xATRTrailingStop := math.max(prevStop, src - nLoss)
else if src < prevStop and src[1] < prevStop
    xATRTrailingStop := math.min(prevStop, src + nLoss)
else if src > prevStop
    xATRTrailingStop := src - nLoss
else
    xATRTrailingStop := src + nLoss

ema1 = ta.ema(src, 1)
above = ta.crossover(ema1, xATRTrailingStop)
below = ta.crossover(xATRTrailingStop, ema1)

buy = src > xATRTrailingStop and above
sell = src < xATRTrailingStop and below

if buy and strategy.position_size <= 0
    strategy.close("Short")
    strategy.entry("Long", strategy.long)

if sell and strategy.position_size >= 0
    strategy.close("Long")
    strategy.entry("Short", strategy.short)

plotshape(buy, title="Buy", text="Buy", style=shape.labelup, location=location.belowbar, color=color.new(color.green, 0), textcolor=color.white, size=size.tiny)
plotshape(sell, title="Sell", text="Sell", style=shape.labeldown, location=location.abovebar, color=color.new(color.red, 0), textcolor=color.white, size=size.tiny)

barcolor(src > xATRTrailingStop ? color.new(color.green, 0) : src < xATRTrailingStop ? color.new(color.red, 0) : na)
plot(xATRTrailingStop, "ATR Trailing Stop", color=color.new(color.orange, 0), linewidth=2)

alertcondition(buy, "UT Long", "UT Long")
alertcondition(sell, "UT Short", "UT Short")
'''

    output = Path(output_path)
    output.write_text(pine, encoding="utf-8")
    return str(output)
