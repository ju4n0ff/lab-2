from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def fit_scaler_on_train(X: pd.DataFrame, train_mask: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X.iloc[train_mask])
    return scaler


def transform_with_scaler(X: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(X).astype(np.float32)


def make_windows(x2d: np.ndarray, lookback: int) -> np.ndarray:
    """Ventana en posicion i cubre [i, i+lookback-1].

    Para una ventana terminada en t (trailing, sin look-ahead) usar indice t-lookback+1.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    if len(x2d) < lookback:
        raise ValueError("serie mas corta que lookback")
    win = sliding_window_view(x2d, lookback, axis=0)
    out = np.ascontiguousarray(win.transpose(0, 2, 1))
    return out.astype(np.float32)


def pinball_loss_np(y_pred: np.ndarray, y_true: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """y_pred (N,H,Q), y_true (N,H). Devuelve pinball promedio por fila (N,H)."""
    e = y_true[:, :, None] - y_pred
    loss = np.maximum(quantiles[None, None, :] * e, (quantiles[None, None, :] - 1.0) * e)
    return loss.mean(axis=2)


def pinball_loss_tf(quantiles: list[float]):
    import tensorflow as tf

    q = tf.constant(quantiles, dtype=tf.float32)

    def loss(y_true, y_pred):
        e = y_true[:, :, None] - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1.0) * e))

    return loss


def decoder_windows(cal: np.ndarray, dec_len: int, positions: np.ndarray) -> np.ndarray:
    """Inputs conocidos futuros para posiciones p: cal[p+1 .. p+dec_len]. Shape (n, dec_len, K)."""
    from numpy.lib.stride_tricks import sliding_window_view

    win = sliding_window_view(cal, dec_len, axis=0)
    out = np.ascontiguousarray(win[positions + 1].transpose(0, 2, 1))
    return out.astype(np.float32)


def drop_warmup(
    df: pd.DataFrame, X: pd.DataFrame, targets: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Elimina warmup inicial y cualquier fila con NaN/inf en features. No imputa."""
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1).to_numpy()
    if not mask.any():
        raise ValueError("todas las filas tienen NaN en features")
    return (
        df.iloc[mask].copy(),
        X.iloc[mask].copy(),
        targets.iloc[mask].copy(),
    )
