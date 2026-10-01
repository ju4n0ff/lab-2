from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.targets import build_targets, forward_mae, forward_mfe, forward_return, forward_volatility


def make_ohlcv(n=300):
    rng = np.random.default_rng(7)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(np.abs(rng.normal(1000, 100, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_.to_numpy(), "high": high.to_numpy(), "low": low.to_numpy(),
         "close": close.to_numpy(), "volume": volume.to_numpy()},
        index=idx,
    )


def test_forward_return_matches_manual():
    df = make_ohlcv(50)
    h = 5
    fr = forward_return(df["close"], h)
    for t in [10, 25, 44]:
        assert fr.iloc[t] == pytest.approx(df["close"].iloc[t + h] / df["close"].iloc[t] - 1)
    assert fr.iloc[-h:].isna().all()


def test_forward_volatility_matches_manual():
    df = make_ohlcv(60)
    logret = np.log(df["close"] / df["close"].shift(1))
    fv = forward_volatility(logret, 6)
    t = 20
    expected = logret.iloc[t + 1 : t + 7].std(ddof=1)
    assert fv.iloc[t] == pytest.approx(expected)


def test_mfe_mae_signs():
    df = make_ohlcv(80)
    mfe = forward_mfe(df["high"], df["close"], 4)
    mae = forward_mae(df["low"], df["close"], 4)
    valid = mfe.notna() & mae.notna()
    assert (mfe[valid] >= -0.05).all()
    assert (mae[valid] <= 0.05).all()
    assert (mfe[valid] >= mae[valid]).all()


def test_build_targets_columns_and_tail_nan():
    df = make_ohlcv(200)
    tg = build_targets(df, [4, 24])
    for h in [4, 24]:
        for base in ["fwd_ret", "fwd_dir", "fwd_vol", "fwd_mfe", "fwd_mae"]:
            assert f"{base}_{h}h" in tg.columns
    assert tg[f"fwd_ret_24h"].iloc[-24:].isna().all()
