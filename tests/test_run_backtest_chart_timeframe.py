from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / 'examples' / 'run_backtest.py'
SPEC = importlib.util.spec_from_file_location('run_backtest_chart_tf', MODULE_PATH)
assert SPEC and SPEC.loader
run_backtest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_backtest)


def test_chart_uses_signal_timeframe_data(monkeypatch, tmp_path) -> None:
    idx = pd.date_range('2024-01-01', periods=48, freq='1h', tz='UTC')
    close = pd.Series(range(len(idx)), index=idx, dtype='float64') + 100.0
    df = pd.DataFrame(
        {
            'timestamp': idx,
            'open': close - 0.25,
            'high': close + 0.5,
            'low': close - 0.5,
            'close': close,
            'volume': 1_000.0,
        }
    )
    csv_path = tmp_path / 'sample.csv'
    df.to_csv(csv_path, index=False)

    captured = {'candles': 0}

    def _fake_chart(data: pd.DataFrame, result: object, output_path: Path) -> Path:
        captured['candles'] = len(data)
        Path(output_path).write_text('ok', encoding='utf-8')
        return Path(output_path)

    monkeypatch.setattr(run_backtest, 'generate_local_tradingview_chart', _fake_chart)
    monkeypatch.setattr(run_backtest, 'generate_backtest_pdf_report', lambda result, output_path: Path(output_path).write_text('ok', encoding='utf-8'))
    monkeypatch.setattr(run_backtest, 'generate_ut_bot_strategy_pinescript', lambda output_path: Path(output_path).write_text('ok', encoding='utf-8'))

    argv = [
        'run_backtest.py',
        '--csv',
        str(csv_path),
        '--out',
        str(tmp_path / 'out'),
        '--strategy',
        'ut_bot',
        '--signal-timeframe',
        '1D',
    ]
    monkeypatch.setattr(sys, 'argv', argv)

    run_backtest.main()

    assert captured['candles'] == 2
