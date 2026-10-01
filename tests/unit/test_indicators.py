from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import indicators as ind


def make_series(values):
    return pd.Series(values, dtype=float)


def test_sma_known_values():
    s = make_series([1, 2, 3, 4, 5])
    out = ind.sma(s, 3)
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_rsi_bounds_and_trend():
    up = make_series(np.cumsum(np.ones(100)))
    rsi_up = ind.rsi(up, 14)
    assert rsi_up.dropna().between(99.9, 100.0).all()
    down = make_series(100 - np.cumsum(np.ones(100)))
    rsi_down = ind.rsi(down, 14)
    assert rsi_down.dropna().between(0.0, 0.1).all()


def test_atr_positive():
    n = 200
    rng = np.random.default_rng(0)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, n)))
    high = close + np.abs(rng.normal(0, 0.2, n))
    low = close - np.abs(rng.normal(0, 0.2, n))
    atr = ind.atr(high, low, close, 14)
    assert (atr.dropna() > 0).all()


def test_macd_shapes():
    s = make_series(np.linspace(10, 20, 100) + np.sin(np.linspace(0, 10, 100)))
    line, sig, hist = ind.macd(s)
    assert len(line) == len(sig) == len(hist) == 100
    assert (hist.dropna() == (line - sig).dropna()).all()


def test_bollinger_pct_b_bounded_mostly():
    rng = np.random.default_rng(1)
    s = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))))
    _, _, width, pct_b = ind.bollinger(s, 20, 2.0)
    valid = pct_b.dropna()
    assert (valid > -1).mean() > 0.99
    assert (width.dropna() >= 0).all()


def test_calendar_features_range():
    idx = pd.date_range("2024-01-01", periods=48, freq="1h", tz="UTC")
    cal = ind.calendar_features(idx)
    for c in cal.columns:
        assert cal[c].abs().max() <= 1.0 + 1e-12
