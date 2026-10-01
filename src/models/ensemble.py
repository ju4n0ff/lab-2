from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .common import pinball_loss_np


class QuantileEnsemble:
    """Combina forecasts de quantiles. Pesos aprendidos SOLO en validacion."""

    def __init__(self, method: str = "pinball_weighted", quantiles: list[float] | None = None):
        self.method = method
        self.quantiles = np.asarray(quantiles, dtype=float)
        self.weights: dict[str, float] | None = None

    def fit(self, val_preds: dict[str, np.ndarray], Y_val: np.ndarray) -> dict:
        names = sorted(val_preds.keys())
        if self.method == "simple_average" or len(names) == 1:
            self.weights = {n: 1.0 / len(names) for n in names}
            return {"weights": dict(self.weights), "method": "simple_average"}

        P = np.stack([val_preds[n] for n in names])
        qs = self.quantiles

        def obj(w: np.ndarray) -> float:
            pred = np.tensordot(w, P, axes=1)
            return float(pinball_loss_np(pred, Y_val, qs).mean())

        w0 = np.full(len(names), 1.0 / len(names))
        res = minimize(
            obj,
            w0,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(names),
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        )
        if res.success:
            w = res.x
            method_used = "pinball_weighted"
        else:
            w = w0
            method_used = "uniform_fallback"
        self.weights = {n: float(wi) for n, wi in zip(names, w)}
        return {"weights": dict(self.weights), "method": method_used}

    def predict(self, preds: dict[str, np.ndarray]) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("ensemble sin ajustar")
        names = [n for n in sorted(preds.keys()) if n in self.weights]
        out = None
        for n in names:
            term = self.weights[n] * preds[n]
            out = term if out is None else out + term
        return out.astype(np.float32)
