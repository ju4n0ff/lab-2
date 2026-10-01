from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def simple_returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def ema(close: pd.Series, window: int) -> pd.Series:
    return close.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(avg_loss != 0, 100.0)
    out[avg_gain.isna() | avg_loss.isna()] = np.nan
    return out


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = line - sig
    return line, sig, hist


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    a = high - low
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def bollinger(
    close: pd.Series, window: int = 20, k: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    mid = sma(close, window)
    sd = close.rolling(window, min_periods=window).std(ddof=1)
    upper = mid + k * sd
    lower = mid - k * sd
    width = (upper - lower) / mid.replace(0.0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return upper, lower, width, pct_b


def roc(close: pd.Series, window: int = 12) -> pd.Series:
    return close.pct_change(periods=window)


def realized_vol(logret: pd.Series, window: int = 24) -> pd.Series:
    return logret.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(window)


def relative_volume(volume: pd.Series, window: int = 24) -> pd.Series:
    base = volume.rolling(window, min_periods=window).mean().replace(0.0, np.nan)
    return volume / base


def drawdown_from_peak(close: pd.Series, window: int | None = None) -> pd.Series:
    if window is None:
        peak = close.cummax()
    else:
        peak = close.rolling(window, min_periods=window).max()
    return close / peak - 1.0


def distance_from_high(close: pd.Series, window: int) -> pd.Series:
    return close / close.rolling(window, min_periods=window).max() - 1.0


def distance_from_low(close: pd.Series, window: int) -> pd.Series:
    return close / close.rolling(window, min_periods=window).min() - 1.0


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    return pd.DataFrame(
        {
            "cal_hour_sin": np.sin(2 * np.pi * hour / 24),
            "cal_hour_cos": np.cos(2 * np.pi * hour / 24),
            "cal_dow_sin": np.sin(2 * np.pi * dow / 7),
            "cal_dow_cos": np.cos(2 * np.pi * dow / 7),
        },
        index=index,
    )
