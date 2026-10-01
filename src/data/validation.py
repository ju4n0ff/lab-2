from __future__ import annotations

import numpy as np
import pandas as pd


def validate_ohlcv(df: pd.DataFrame, freq_ms: int) -> tuple[pd.DataFrame, dict]:
    """Limpia y valida OHLCV. Devuelve (df_limpio, reporte). Nunca inventa datos."""
    rep: dict = {"n_rows_in": int(len(df))}
    df = df.copy()

    dup_mask = df["open_time"].duplicated(keep="first")
    rep["duplicates_dropped"] = int(dup_mask.sum())
    df = df.loc[~dup_mask].copy()

    df = df.sort_values("open_time").reset_index(drop=True)

    required = ["open", "high", "low", "close", "volume"]
    nan_counts = {c: int(df[c].isna().sum()) for c in required}
    rep["nan_per_column"] = nan_counts
    df = df.dropna(subset=required).reset_index(drop=True)

    bad_ohlc = (df["high"] < df[["open", "close", "low"]].max(axis=1)) | (
        df["low"] > df[["open", "close", "high"]].min(axis=1)
    )
    nonpositive = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    invalid = bad_ohlc | nonpositive
    rep["invalid_rows_dropped"] = int(invalid.sum())
    df = df.loc[~invalid].reset_index(drop=True)

    ts_ns = df["open_time"].astype("datetime64[ns, UTC]").astype("int64")
    ts = ts_ns // 10**6
    gaps = np.diff(ts.values) != freq_ms
    n_gaps = int(gaps.sum())
    rep["gaps"] = n_gaps
    rep["gap_ratio"] = float(n_gaps / max(len(df) - 1, 1))
    rep["start"] = str(df["open_time"].iloc[0])
    rep["end"] = str(df["open_time"].iloc[-1])
    rep["n_rows_out"] = int(len(df))

    score = 100.0
    score -= min(rep["gap_ratio"] * 100, 40.0)
    score -= min(rep["duplicates_dropped"] / max(len(df), 1) * 100, 20.0)
    score -= min(sum(nan_counts.values()) / max(rep["n_rows_in"], 1) * 100, 20.0)
    score -= min(rep["invalid_rows_dropped"] / max(len(df), 1) * 100, 20.0)
    rep["quality_score"] = round(max(score, 0.0), 1)

    tz_ok = str(df["open_time"].dt.tz) == "UTC"
    monotonic = df["open_time"].is_monotonic_increasing
    unique = df["open_time"].is_unique
    rep["timestamp_checks"] = {
        "tz_utc": bool(tz_ok),
        "monotonic": bool(monotonic),
        "unique": bool(unique),
    }
    if not (tz_ok and monotonic and unique):
        raise ValueError("validacion de timestamps fallo; pipeline detenido")
    return df, rep
