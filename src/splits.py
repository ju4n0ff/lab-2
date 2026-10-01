from __future__ import annotations

import numpy as np
import pandas as pd


def chrono_masks(
    index: pd.DatetimeIndex,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str | None,
) -> dict[str, np.ndarray]:
    ts = index.tz_localize(None) if index.tz is None else index.tz_convert("UTC").tz_localize(None)
    masks = {
        "train": (ts >= pd.Timestamp(train_start)) & (ts <= pd.Timestamp(train_end)),
        "val": (ts >= pd.Timestamp(val_start)) & (ts <= pd.Timestamp(val_end)),
        "test": (ts >= pd.Timestamp(test_start)),
    }
    if test_end is not None:
        masks["test"] &= ts <= pd.Timestamp(test_end)
    for name, m in masks.items():
        if m.sum() == 0:
            raise ValueError(f"split vacio: {name}")
    return masks


def walk_forward_folds(
    n_samples: int, n_folds: int, val_fraction: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds expandentes dentro de train+val. El test final nunca participa."""
    if n_folds < 1:
        raise ValueError("n_folds debe ser >= 1")
    val_size = int(n_samples * val_fraction)
    fold_size = max(val_size // n_folds, 1)
    folds = []
    for f in range(n_folds):
        val_start = n_samples - val_size + f * fold_size
        val_end = val_start + fold_size if f < n_folds - 1 else n_samples
        train_idx = np.arange(0, val_start)
        val_idx = np.arange(val_start, val_end)
        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append((train_idx, val_idx))
    return folds
