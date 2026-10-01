from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def p_up_from_quantiles(pred: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """P(retorno > 0) por interpolacion de la CDF de quantiles. pred (N,H,Q)."""
    n, h, _ = pred.shape
    out = np.zeros((n, h))
    for i in range(n):
        for j in range(h):
            vals = pred[i, j]
            order = np.argsort(vals)
            v_sorted = vals[order]
            q_sorted = quantiles[order]
            if 0.0 <= v_sorted[0]:
                out[i, j] = 1.0 - float(q_sorted[0])
                continue
            if 0.0 > v_sorted[-1]:
                out[i, j] = 1.0 - float(q_sorted[-1])
                continue
            f0 = float(np.interp(0.0, v_sorted, q_sorted))
            out[i, j] = 1.0 - f0
    return np.clip(out, 1e-6, 1 - 1e-6)


class PlattCalibrator:
    """Calibracion de P(up) con regresion logistica ajustada en validacion."""

    def __init__(self):
        self.model: LogisticRegression | None = None

    @staticmethod
    def _feature(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p_raw: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        self.model = LogisticRegression()
        self.model.fit(self._feature(p_raw), labels.astype(int))
        return self

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("calibrador sin ajustar")
        return self.model.predict_proba(self._feature(p_raw))[:, 1]


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss_score(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(p, edges[1:-1])
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        ece += (mask.sum() / len(p)) * abs(conf - acc)
    return float(ece)


def reliability_bins(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(p, edges[1:-1])
    out = {"bin_center": [], "predicted_mean": [], "empirical_freq": [], "count": []}
    for b in range(bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        out["bin_center"].append(float((edges[b] + edges[b + 1]) / 2))
        out["predicted_mean"].append(float(p[mask].mean()))
        out["empirical_freq"].append(float(y[mask].mean()))
        out["count"].append(int(mask.sum()))
    return out
