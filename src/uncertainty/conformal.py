from __future__ import annotations

import numpy as np


class SplitConformal:
    """Split conformal por horizonte sobre residuos absolutos de validacion.

    q_hat = quantil_{ceil((n+1)(1-alpha))/n}(|y - pred|) en validacion.
    Intervalo: pred ± q_hat. Cobertura marginal garantizada bajo intercambiabilidad.
    """

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.q_hat: np.ndarray | None = None

    def fit(self, residuals_abs: np.ndarray) -> dict:
        n = residuals_abs.shape[0]
        level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.q_hat = np.quantile(residuals_abs, level, axis=0).astype(np.float64)
        return {"alpha": self.alpha, "level": float(level), "q_hat": self.q_hat.tolist()}

    def intervals(self, median_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.q_hat is None:
            raise RuntimeError("conformal sin ajustar")
        return median_pred - self.q_hat, median_pred + self.q_hat


def empirical_coverage(lower: np.ndarray, upper: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ((y >= lower) & (y <= upper)).mean(axis=0)


def interval_stats(lower: np.ndarray, upper: np.ndarray) -> dict:
    width = upper - lower
    return {
        "mean_width": width.mean(axis=0).tolist(),
        "median_width": np.median(width, axis=0).tolist(),
    }
