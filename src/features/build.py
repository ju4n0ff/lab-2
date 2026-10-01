from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


FEATURE_META: list[dict] = []


def _meta(name: str, category: str, role: str, description: str) -> dict:
    return {
        "name": name,
        "category": category,
        "role": role,
        "description": description,
        "leakage_risk": "bajo: solo ventanas trailing",
    }


def build_features(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Construye features causales (solo ventanas trailing). Devuelve (X, meta)."""
    w = cfg["windows"]
    close = df["close"]
    logret = ind.log_returns(close)

    macd_line, macd_sig, macd_hist = ind.macd(close)
    bb_up, bb_lo, bb_w, bb_pct = ind.bollinger(close, w["bb"], w["bb_k"])

    feats: dict[str, pd.Series] = {
        "ret_1": ind.simple_returns(close),
        "logret_1": logret,
        "logret_4": np.log(close / close.shift(4)),
        "logret_24": np.log(close / close.shift(24)),
        "sma_ratio_fast": close / ind.sma(close, w["sma_fast"]) - 1.0,
        "sma_ratio_slow": close / ind.sma(close, w["sma_slow"]) - 1.0,
        "ema_ratio": close / ind.ema(close, w["ema"]) - 1.0,
        "rsi": ind.rsi(close, w["rsi"]),
        "macd_norm": macd_line / close,
        "macd_signal_norm": macd_sig / close,
        "macd_hist_norm": macd_hist / close,
        "atr_norm": ind.atr(df["high"], df["low"], close, w["atr"]) / close,
        "bb_width": bb_w,
        "bb_pct_b": bb_pct,
        "roc": ind.roc(close, w["roc"]),
        "realized_vol": ind.realized_vol(logret, w["rv"]),
        "rel_volume": ind.relative_volume(df["volume"], w["relvol"]),
        "log_volume_change": np.log(df["volume"].replace(0.0, np.nan)).diff(),
        "drawdown_rolling": ind.drawdown_from_peak(close, w["dd"]),
        "dist_high": ind.distance_from_high(close, w["hi_lo"]),
        "dist_low": ind.distance_from_low(close, w["hi_lo"]),
    }

    X = pd.DataFrame(feats, index=df.index)

    meta = [
        _meta("ret_1", "price", "observed", "retorno simple 1 vela"),
        _meta("logret_1", "price", "observed", "log retorno 1 vela"),
        _meta("logret_4", "price", "observed", "log retorno 4 velas"),
        _meta("logret_24", "price", "observed", "log retorno 24 velas"),
        _meta("sma_ratio_fast", "trend", "observed", "precio vs SMA corta"),
        _meta("sma_ratio_slow", "trend", "observed", "precio vs SMA larga"),
        _meta("ema_ratio", "trend", "observed", "precio vs EMA"),
        _meta("rsi", "momentum", "observed", "RSI Wilder"),
        _meta("macd_norm", "momentum", "observed", "MACD normalizado"),
        _meta("macd_signal_norm", "momentum", "observed", "senal MACD normalizada"),
        _meta("macd_hist_norm", "momentum", "observed", "histograma MACD"),
        _meta("atr_norm", "volatility", "observed", "ATR relativo"),
        _meta("bb_width", "volatility", "observed", "ancho Bandas de Bollinger"),
        _meta("bb_pct_b", "volatility", "observed", "%B de Bollinger"),
        _meta("roc", "momentum", "observed", "rate of change"),
        _meta("realized_vol", "volatility", "observed", "volatilidad realizada rolling"),
        _meta("rel_volume", "volume", "observed", "volumen relativo"),
        _meta("log_volume_change", "volume", "observed", "cambio log volumen"),
        _meta("drawdown_rolling", "market_structure", "observed", "drawdown desde maximo rolling"),
        _meta("dist_high", "market_structure", "observed", "distancia al maximo rolling"),
        _meta("dist_low", "market_structure", "observed", "distancia al minimo rolling"),
    ]

    if cfg.get("include_calendar", True):
        cal = ind.calendar_features(df.index)
        for col in cal.columns:
            X[col] = cal[col]
            meta.append(_meta(col, "calendar", "known_future", "feature calendario determinista"))

    global FEATURE_META
    FEATURE_META = meta
    return X, meta
