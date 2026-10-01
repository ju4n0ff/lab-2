from __future__ import annotations

import numpy as np


def buy_and_hold(n: int) -> np.ndarray:
    return np.ones(n)


def no_trade(n: int) -> np.ndarray:
    return np.zeros(n)


def naive_momentum_signal(close: np.ndarray) -> np.ndarray:
    """Long si el ultimo retorno fue positivo. Decision en t con info <= t."""
    ret = np.diff(close, prepend=close[0])
    return (ret > 0).astype(float)


def ma_crossover_signal(close: np.ndarray, fast: int = 12, slow: int = 48) -> np.ndarray:
    s_fast = _rolling_mean(close, fast)
    s_slow = _rolling_mean(close, slow)
    sig = (s_fast > s_slow).astype(float)
    sig[np.isnan(s_fast) | np.isnan(s_slow)] = 0.0
    return sig


def momentum_signal(close: np.ndarray, window: int = 12) -> np.ndarray:
    past = np.full_like(close, np.nan, dtype=float)
    past[window:] = close[:-window]
    roc = close / past - 1.0
    sig = (roc > 0).astype(float)
    sig[np.isnan(roc)] = 0.0
    return sig


def rsi_mean_reversion_signal(close: np.ndarray, window: int = 14, lo: float = 30.0, hi: float = 70.0) -> np.ndarray:
    rsi = _rsi_np(close, window)
    n = len(close)
    pos = np.zeros(n)
    state = 0.0
    for t in range(n):
        v = rsi[t]
        if not np.isnan(v):
            if state == 0.0:
                if v < lo:
                    state = 1.0
            else:
                if v > hi:
                    state = 0.0
        pos[t] = state
    return pos


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    s = pd_rolling_mean(x, w)
    return s


def pd_rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    import pandas as pd

    return pd.Series(x).rolling(w, min_periods=w).mean().to_numpy()


def _rsi_np(close: np.ndarray, w: int) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.clip(delta, 0.0, None)
    loss = np.clip(-delta, None, 0.0)
    avg_gain = _wilder(gain, w)
    avg_loss = _wilder(loss, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss != 0)
        rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi[avg_loss == 0] = 100.0
    warmup = min(w * 2, len(close))
    rsi[:warmup] = np.nan
    return rsi


def _wilder(x: np.ndarray, w: int) -> np.ndarray:
    import pandas as pd

    return pd.Series(x).ewm(alpha=1.0 / w, adjust=False, min_periods=w).mean().to_numpy()
