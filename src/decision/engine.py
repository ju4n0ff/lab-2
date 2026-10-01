from __future__ import annotations

import numpy as np


def choose_horizon(
    expected_returns: np.ndarray,
    vols: np.ndarray,
    cost_estimates: np.ndarray,
    lambda_risk: float,
    lambda_cost: float,
) -> int:
    """h* = argmax E[R] - lambda_risk*sigma - lambda_cost*cost. Utilidad esperada ajustada por riesgo."""
    utility = expected_returns - lambda_risk * vols - lambda_cost * cost_estimates
    return int(np.argmax(utility))


def decide_action(
    p_up_calibrated: float,
    expected_net_return: float,
    data_quality: float,
    min_p_up: float,
    min_edge: float,
    min_quality: float,
    ood_flag: bool,
) -> tuple[str, list[str]]:
    """Reglas documentadas: BUY / WAIT / NO TRADE para cuenta flat."""
    reasons: list[str] = []
    if data_quality < min_quality:
        reasons.append(f"data quality {data_quality:.0f} < {min_quality}")
        return "NO TRADE", reasons
    if ood_flag:
        reasons.append("intervalo de prediccion fuera de distribucion (ancho extremo)")
        return "NO TRADE", reasons
    if p_up_calibrated >= min_p_up and expected_net_return >= min_edge:
        return "BUY", [f"p_up={p_up_calibrated:.2f}>={min_p_up}", f"edge neto={expected_net_return:.4f}>={min_edge}"]
    reasons.append(
        f"condiciones no cumplidas: p_up={p_up_calibrated:.2f} (min {min_p_up}), "
        f"edge={expected_net_return:.4f} (min {min_edge})"
    )
    return "WAIT", reasons


def confidence_score(
    p_up_calibrated: float, interval_width_over_vol: float, data_quality: float
) -> float:
    """confidence = p_calibrada * (1 - width_norm) * calidad_datos/100.

    width_norm = clip(ancho_intervalo / (8 * sigma_estimada), 0, 1).
    Justificacion: un PI-90% bien calibrado mide ~3.3 sigma (penalizacion ~40%);
    anchos >= 8 sigma indican incertidumbre extrema y llevan la confianza a 0.
    Formula explicita y documentada.
    """
    width_norm = float(np.clip(interval_width_over_vol / 8.0, 0.0, 1.0))
    return round(float(p_up_calibrated * (1.0 - width_norm) * (data_quality / 100.0)), 4)


def position_size(max_position: float, p_up_calibrated: float) -> float:
    """Tamano proporcional a probabilidad calibrada, acotado por max_position."""
    return round(float(np.clip(max_position * p_up_calibrated, 0.0, max_position)), 4)


def prob_target_stop(
    median_pred: np.ndarray,
    quantiles: np.ndarray,
    target_pct: float,
    stop_pct: float,
) -> tuple[float, float]:
    """Aproximacion desde CDF de quantiles del horizonte elegido.

    LIMITACION conocida: usa la distribucion marginal del retorno, no la ruta
    (path) del precio; subestima P(target hit) y sobreestima P(stop hit).
    """
    order = np.argsort(median_pred)
    v = median_pred[order]
    q = quantiles[order]
    f_target = float(np.interp(target_pct, v, q)) if target_pct <= v[-1] else 1.0
    if target_pct < v[0]:
        f_target = 0.0
    f_stop = float(np.interp(-stop_pct, v, q)) if -stop_pct <= v[-1] else 1.0
    if -stop_pct < v[0]:
        f_stop = 0.0
    p_target = float(np.clip(1.0 - f_target, 0.0, 1.0))
    p_stop = float(np.clip(f_stop, 0.0, 1.0))
    return p_target, p_stop
