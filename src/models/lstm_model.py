from __future__ import annotations

import numpy as np

from .common import pinball_loss_tf


class LSTMQuantile:
    """LSTM Keras multi-horizonte multi-quantil. Benchmark deep learning."""

    def __init__(self, lookback: int, n_horizons: int, quantiles: list[float], cfg: dict):
        import tensorflow as tf

        self.lookback = lookback
        self.n_horizons = n_horizons
        self.quantiles = quantiles
        self.cfg = cfg
        self.model = None
        self._tf = tf

    def _build(self, n_feat: int):
        tf = self._tf
        layers = tf.keras.layers
        inp = layers.Input((self.lookback, n_feat))
        x = layers.LSTM(self.cfg.get("units", 64))(inp)
        x = layers.Dropout(0.1)(x)
        out = layers.Dense(self.n_horizons * len(self.quantiles))(x)
        out = layers.Reshape((self.n_horizons, len(self.quantiles)))(out)
        model = tf.keras.Model(inp, out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                self.cfg.get("learning_rate", 1e-3), clipnorm=1.0
            ),
            loss=pinball_loss_tf(self.quantiles),
        )
        return model

    def fit(
        self,
        Xtr: np.ndarray,
        Ytr: np.ndarray,
        Xval: np.ndarray,
        Yval: np.ndarray,
    ) -> dict:
        tf = self._tf
        self.model = self._build(Xtr.shape[-1])
        hist = self.model.fit(
            Xtr,
            Ytr,
            validation_data=(Xval, Yval),
            epochs=self.cfg.get("epochs", 15),
            batch_size=self.cfg.get("batch_size", 128),
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.cfg.get("patience", 3),
                    restore_best_weights=True,
                )
            ],
        )
        return {"history": {k: [float(v) for v in vs] for k, vs in hist.history.items()}}

    def predict(self, Xw: np.ndarray) -> np.ndarray:
        return self.model.predict(Xw, verbose=0).astype(np.float32)
