from __future__ import annotations

import numpy as np

from .common import pinball_loss_tf


class GRN:
    """Gated Residual Network simplificada (TFT-inspired)."""

    def __init__(self, d_model: int):
        self.d_model = d_model

    def __call__(self, x):
        import tensorflow as tf
        from tensorflow.keras import layers

        units = x.shape[-1]
        y = layers.Dense(self.d_model)(x)
        y = layers.ELU()(y)
        y = layers.Dense(self.d_model)(y)
        gate = layers.Dense(self.d_model, activation="sigmoid")(y)
        y = layers.Multiply()([y, gate])
        if units != self.d_model:
            x = layers.Dense(self.d_model)(x)
        out = layers.LayerNormalization()(layers.Add()([x, y]))
        return out


class VSN:
    """Variable Selection Network simplificada: pesos softmax por variable."""

    def __init__(self, d_model: int):
        self.d_model = d_model

    def __call__(self, x):
        import tensorflow as tf
        from tensorflow.keras import layers

        n_feat = int(x.shape[-1])
        proj = [layers.Dense(self.d_model)(x[..., i : i + 1]) for i in range(n_feat)]
        scores = layers.Dense(n_feat)(x)
        weights = layers.Softmax(axis=-1)(scores)
        sel = sum(weights[..., i : i + 1] * proj[i] for i in range(n_feat))
        return sel


def build_tft(
    lookback: int,
    n_observed: int,
    dec_len: int,
    n_known: int,
    n_horizons: int,
    quantiles: list[float],
    d_model: int = 48,
    lstm_units: int = 48,
    heads: int = 1,
    lr: float = 1e-3,
):
    """TFT-lite: VSN + LSTM encoder/decoder + atencion interpretable + salida multi-quantil."""
    import tensorflow as tf
    from tensorflow.keras import layers

    enc_in = layers.Input((lookback, n_observed), name="enc_in")
    dec_in = layers.Input((dec_len, n_known), name="dec_in")

    sel_e = VSN(d_model)(enc_in)
    sel_d = VSN(d_model)(dec_in)

    enc_seq, *state = layers.LSTM(lstm_units, return_sequences=True, return_state=True)(sel_e)
    dec_seq = layers.LSTM(lstm_units, return_sequences=True)(sel_d, initial_state=state)

    att = layers.MultiHeadAttention(num_heads=heads, key_dim=max(8, d_model // 2))(
        dec_seq, enc_seq
    )
    att = GRN(d_model)(att)
    fused = GRN(d_model)(layers.Add()([dec_seq, att]))

    last = layers.Lambda(lambda t: t[:, -1, :])(fused)
    out = layers.Dense(n_horizons * len(quantiles))(last)
    out = layers.Reshape((n_horizons, len(quantiles)))(out)

    model = tf.keras.Model([enc_in, dec_in], out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr, clipnorm=1.0),
        loss=pinball_loss_tf(quantiles),
    )
    return model


class TFTQuantile:
    def __init__(self, lookback: int, dec_len: int, n_horizons: int, quantiles: list[float], cfg: dict):
        self.lookback = lookback
        self.dec_len = dec_len
        self.n_horizons = n_horizons
        self.quantiles = quantiles
        self.cfg = cfg
        self.model = None

    def fit(
        self,
        Xtr: np.ndarray,
        Dtr: np.ndarray,
        Ytr: np.ndarray,
        Xval: np.ndarray,
        Dval: np.ndarray,
        Yval: np.ndarray,
    ) -> dict:
        import tensorflow as tf

        self.model = build_tft(
            lookback=self.lookback,
            n_observed=Xtr.shape[-1],
            dec_len=self.dec_len,
            n_known=Dtr.shape[-1],
            n_horizons=self.n_horizons,
            quantiles=self.quantiles,
            d_model=self.cfg.get("d_model", 48),
            lstm_units=self.cfg.get("lstm_units", 48),
            heads=self.cfg.get("heads", 1),
            lr=self.cfg.get("learning_rate", 1e-3),
        )
        hist = self.model.fit(
            [Xtr, Dtr],
            Ytr,
            validation_data=([Xval, Dval], Yval),
            epochs=self.cfg.get("epochs", 12),
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

    def predict(self, Xw: np.ndarray, Dw: np.ndarray) -> np.ndarray:
        return self.model.predict([Xw, Dw], verbose=0).astype(np.float32)
