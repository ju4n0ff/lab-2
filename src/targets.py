from __future__ import annotations

import numpy as np
import pandas as pd

from .features import indicators as ind


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """R_h = P_{t+h}/P_t - 1. Label: usa solo informacion futura por definicion."""
    return close.shift(-horizon) / close - 1.0


def forward_direction(close: pd.Series, horizon: int) -> pd.Series:
    r = forward_return(close, horizon)
    return np.sign(r)


def forward_volatility(logret: pd.Series, horizon: int) -> pd.Series:
    arr = logret.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    for t in range(len(arr) - horizon):
        seg = arr[t + 1 : t + 1 + horizon]
        out[t] = np.std(seg, ddof=1) if horizon > 1 else abs(seg[0])
    return pd.Series(out, index=logret.index)


def forward_mfe(high: pd.Series, close: pd.Series, horizon: int) -> pd.Series:
    """Maximum favorable excursion relativo al close de t."""
    hi = high.to_numpy(dtype=float)
    cl = close.to_numpy(dtype=float)
    out = np.full(len(cl), np.nan)
    for t in range(len(cl) - horizon):
        out[t] = hi[t + 1 : t + 1 + horizon].max() / cl[t] - 1.0
    return pd.Series(out, index=close.index)


def forward_mae(low: pd.Series, close: pd.Series, horizon: int) -> pd.Series:
    lo = low.to_numpy(dtype=float)
    cl = close.to_numpy(dtype=float)
    out = np.full(len(cl), np.nan)
    for t in range(len(cl) - horizon):
        out[t] = lo[t + 1 : t + 1 + horizon].min() / cl[t] - 1.0
    return pd.Series(out, index=close.index)


def build_targets(df: pd.DataFrame, horizons_hours: list[int]) -> pd.DataFrame:
    close = df["close"]
    logret = ind.log_returns(close)
    parts = {}
    for h in horizons_hours:
        suffix = f"{h}h"
        parts[f"fwd_ret_{suffix}"] = forward_return(close, h)
        parts[f"fwd_dir_{suffix}"] = forward_direction(close, h)
        parts[f"fwd_vol_{suffix}"] = forward_volatility(logret, h)
        parts[f"fwd_mfe_{suffix}"] = forward_mfe(df["high"], close, h)
        parts[f"fwd_mae_{suffix}"] = forward_mae(df["low"], close, h)
    return pd.DataFrame(parts, index=df.index)
