from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build import build_features
from src.models.common import fit_scaler_on_train, make_windows
from src.targets import build_targets


def make_ohlcv(n=600):
    rng = np.random.default_rng(11)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(np.abs(rng.normal(1000, 100, n)))
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_.to_numpy(), "high": high.to_numpy(), "low": low.to_numpy(),
         "close": close.to_numpy(), "volume": volume.to_numpy()},
        index=idx,
    )


FEAT_CFG = {
    "windows": {
        "sma_fast": 12, "sma_slow": 48, "ema": 21, "rsi": 14, "atr": 14,
        "bb": 20, "bb_k": 2.0, "roc": 12, "rv": 24, "relvol": 24, "dd": 168, "hi_lo": 168,
    },
    "include_calendar": True,
}


def test_features_prefix_stability_no_lookahead():
    """Las features en t no deben cambiar al anexar datos futuros."""
    df = make_ohlcv(500)
    full_X, _ = build_features(df, FEAT_CFG)
    k = 400
    part_X, _ = build_features(df.iloc[:k], FEAT_CFG)
    cols = [c for c in full_X.columns]
    a = full_X.iloc[:k][cols].reset_index(drop=True)
    b = part_X[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        a.dropna().round(10), b.dropna().round(10), check_dtype=False
    )


def test_targets_are_future_only():
    df = make_ohlcv(300)
    tg = build_targets(df, [4])
    pert = df.copy()
    pert.iloc[250:, :] = pert.iloc[250:, :].to_numpy() * 1.5
    tg_pert = build_targets(pert, [4])
    assert np.allclose(tg["fwd_ret_4h"].iloc[:240], tg_pert["fwd_ret_4h"].iloc[:240])


def test_scaler_fit_on_train_only():
    df = make_ohlcv(500)
    X, _ = build_features(df, FEAT_CFG)
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    train_mask = np.zeros(len(X), dtype=bool)
    train_mask[:300] = True
    scaler = fit_scaler_on_train(X, train_mask)
    tr_mean = X.iloc[train_mask].mean().to_numpy()
    assert np.allclose(scaler.mean_, tr_mean, rtol=1e-6)


def test_windows_trailing_not_centered():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(100, 2)).astype(np.float32)
    win = make_windows(x, lookback=10)
    assert win.shape == (91, 10, 2)
    assert np.allclose(win[50], x[50:60])
    t = 60
    assert np.allclose(win[t - 9], x[t - 9 : t + 1])
