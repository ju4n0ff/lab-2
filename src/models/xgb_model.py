from __future__ import annotations

import numpy as np

try:
    from xgboost import XGBRegressor

    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class XGBQuantileModels:
    """Un modelo por (horizonte, quantil). Entrenamiento solo con train; early stopping con val."""

    def __init__(self, quantiles: list[float], params: dict):
        if not HAS_XGB:
            raise ImportError("xgboost no instalado")
        self.quantiles = quantiles
        self.params = params
        self.models: dict[tuple[int, float], object] = {}

    def fit(self, Xtr: np.ndarray, Ytr: np.ndarray, Xval: np.ndarray, Yval: np.ndarray) -> None:
        n_h = Ytr.shape[1]
        es = self.params.get("early_stopping_rounds", 50)
        for h in range(n_h):
            for q in self.quantiles:
                model = XGBRegressor(
                    n_estimators=self.params.get("n_estimators", 400),
                    max_depth=self.params.get("max_depth", 6),
                    learning_rate=self.params.get("learning_rate", 0.05),
                    subsample=self.params.get("subsample", 0.9),
                    colsample_bytree=self.params.get("colsample_bytree", 0.9),
                    objective="reg:quantileerror",
                    quantile_alpha=q,
                    tree_method="hist",
                    early_stopping_rounds=es,
                    verbosity=0,
                    n_jobs=-1,
                )
                model.fit(
                    Xtr,
                    Ytr[:, h],
                    eval_set=[(Xval, Yval[:, h])],
                    verbose=False,
                )
                self.models[(h, q)] = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        n_h = len({h for h, _ in self.models})
        out = np.zeros((n, n_h, len(self.quantiles)), dtype=np.float32)
        for h in range(n_h):
            for qi, q in enumerate(self.quantiles):
                out[:, h, qi] = self.models[(h, q)].predict(X)
        return out
