from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtesting.backtester import run_backtest
from src.config import load_config, path_under_root
from src.data.validation import validate_ohlcv
from src.decision.engine import (
    choose_horizon,
    confidence_score,
    decide_action,
    position_size,
    prob_target_stop,
)
from src.evaluation.metrics import predictive_metrics, strategy_metrics
from src.features.build import build_features
from src.models.baselines import (
    buy_and_hold,
    ma_crossover_signal,
    momentum_signal,
    naive_momentum_signal,
    no_trade,
    rsi_mean_reversion_signal,
)
from src.models.common import (
    decoder_windows,
    drop_warmup,
    fit_scaler_on_train,
    make_windows,
    set_seeds,
    transform_with_scaler,
)
from src.models.ensemble import QuantileEnsemble
from src.models.lstm_model import LSTMQuantile
from src.models.tft import TFTQuantile
from src.models.xgb_model import XGBQuantileModels
from src.reporting.reports import (
    log_experiment,
    write_backtest_report,
    write_calibration_report,
    write_model_card,
    write_research_report,
)
from src.splits import chrono_masks, walk_forward_folds
from src.targets import build_targets
from src.uncertainty.calibration import (
    PlattCalibrator,
    brier_score,
    expected_calibration_error,
    log_loss_score,
    p_up_from_quantiles,
    reliability_bins,
)
from src.uncertainty.conformal import SplitConformal, empirical_coverage, interval_stats


def main() -> None:
    data_cfg = load_config("data")
    feat_cfg = load_config("features")
    models_cfg = load_config("models")
    bt_cfg = load_config("backtest")
    risk_cfg = load_config("risk")

    seed = int(models_cfg.get("seed", 42))
    set_seeds(seed)
    experiment_id = str(uuid.uuid4())[:8]

    if models_cfg.get("quick"):
        models_cfg["lstm"]["epochs"] = 2
        models_cfg["tft"]["epochs"] = 2
        models_cfg["xgboost"]["n_estimators"] = 60
        print("MODO QUICK: epocas reducidas (smoke test)")

    symbol = data_cfg["market"]["symbol"]
    timeframe = data_cfg["market"]["timeframe"]
    raw_dir = path_under_root(data_cfg["paths"]["raw"])
    validated_path = raw_dir / f"{symbol.lower()}_{timeframe}_validated.parquet"
    if not validated_path.exists():
        raise SystemExit("Datos validados no encontrados. Ejecuta scripts/download_data.py primero.")

    df = pd.read_parquet(validated_path)
    df, val_rep = validate_ohlcv(df, 3_600_000)
    df = df.set_index("open_time")
    quality = val_rep["quality_score"]
    print(f"[1/8] Datos: {len(df)} velas, quality={quality}")

    X_raw, meta = build_features(df, feat_cfg)
    targets_raw = build_targets(df, feat_cfg["horizons_hours"])
    df, X, targets = drop_warmup(df, X_raw, targets_raw)
    print(f"[2/8] Features: {X.shape[1]} | Targets: {targets.shape[1]} | filas utiles: {len(df)}")

    sp = data_cfg["splits"]
    masks = chrono_masks(
        df.index,
        sp["train_start"],
        sp["train_end"],
        sp["val_start"],
        sp["val_end"],
        sp["test_start"],
        sp.get("test_end"),
    )
    train_end_pos = int(np.where(masks["val"])[0][0])
    val_end_pos = int(np.where(masks["test"])[0][0])

    horizons = np.array(feat_cfg["horizons_hours"])
    quantiles = np.array(feat_cfg["quantiles"], dtype=float)

    scaler = fit_scaler_on_train(X, masks["train"])
    Xs = transform_with_scaler(X, scaler)

    lookback = int(feat_cfg["lookback"])
    dec_len = int(feat_cfg.get("decoder_len", 24))
    cal_cols = [c for c in X.columns if c.startswith("cal_")]
    cal_all = Xs[:, [X.columns.get_loc(c) for c in cal_cols]].astype(np.float32)

    max_h = int(horizons.max())
    n_usable = len(df) - max_h - dec_len
    windows = make_windows(Xs[:n_usable], lookback)
    y_matrix = targets[[f"fwd_ret_{h}h" for h in horizons]].to_numpy()[:n_usable].astype(np.float32)

    pos_valid = np.arange(lookback - 1, n_usable)
    first_val_rel = int(np.searchsorted(pos_valid, train_end_pos))
    last_val_rel = int(np.searchsorted(pos_valid, val_end_pos))
    tv_rel = np.arange(0, last_val_rel)
    te_rel = np.arange(last_val_rel, len(pos_valid))

    val_positions = pos_valid[first_val_rel:last_val_rel]
    test_positions = pos_valid[last_val_rel:]

    n_folds = int(models_cfg["walk_forward"]["folds"])
    val_frac = float(models_cfg["walk_forward"].get("val_fraction", 0.15))
    folds = walk_forward_folds(len(tv_rel), n_folds, val_frac)

    enabled = []
    if models_cfg["xgboost"].get("enabled", True):
        enabled.append("xgboost")
    if models_cfg["lstm"].get("enabled", True):
        enabled.append("lstm")
    if models_cfg["tft"].get("enabled", True):
        enabled.append("tft")

    val_preds_by_fold: list[dict[str, np.ndarray]] = []
    test_preds_by_fold: list[dict[str, np.ndarray]] = []
    artifacts_dir = path_under_root("models/artifacts")
    fresh = bool(os.environ.get("AOTTS_FRESH"))

    def load_or_train(name: str, fi: int, train_fn, expected_val_len: int, expected_test_len: int) -> tuple[np.ndarray, np.ndarray]:
        cache = artifacts_dir / f"{name}_fold{fi}.npz"
        if cache.exists() and not fresh:
            z = np.load(cache)
            if z["val"].shape[0] == expected_val_len and z["test"].shape[0] == expected_test_len:
                print(f"       {name} fold{fi}: cargado de cache")
                return z["val"], z["test"]
            else:
                print(f"       {name} fold{fi}: cache obsoleto (val {z['val'].shape[0]}!={expected_val_len}, test {z['test'].shape[0]}!={expected_test_len}), reentrenando...")
                cache.unlink()
        import time as _time

        t0 = _time.time()
        pv, pt = train_fn()
        np.savez_compressed(cache, val=pv, test=pt)
        print(f"       {name} fold{fi}: entrenado en {_time.time() - t0:.0f}s")
        return pv, pt

    for fi, (tr_rel_f, va_rel_f) in enumerate(folds):
        tr_pos = pos_valid[tv_rel[tr_rel_f]]
        va_pos = pos_valid[tv_rel[va_rel_f]]
        tr_w = tv_rel[tr_rel_f]
        va_w = tv_rel[va_rel_f]
        print(
            f"[3/8] Fold {fi + 1}/{len(folds)}: train={len(tr_pos)} val={len(va_pos)} "
            f"modelos={enabled}"
        )

        preds_val_f: dict[str, np.ndarray] = {}
        preds_test_f: dict[str, np.ndarray] = {}

        if "xgboost" in enabled:

            def train_xgb():
                xgbm = XGBQuantileModels(list(quantiles), models_cfg["xgboost"])
                xgbm.fit(Xs[tr_pos], y_matrix[tr_pos], Xs[va_pos], y_matrix[va_pos])
                return xgbm.predict(Xs[va_pos]), xgbm.predict(Xs[test_positions])

            preds_val_f["xgboost"], preds_test_f["xgboost"] = load_or_train("xgboost", fi, train_xgb, len(va_pos), len(test_positions))

        if "lstm" in enabled:

            def train_lstm():
                lstm = LSTMQuantile(lookback, len(horizons), list(quantiles), models_cfg["lstm"])
                lstm.fit(windows[tr_w], y_matrix[tr_pos], windows[va_w], y_matrix[va_pos])
                return lstm.predict(windows[va_w]), lstm.predict(windows[te_rel])

            preds_val_f["lstm"], preds_test_f["lstm"] = load_or_train("lstm", fi, train_lstm, len(va_w), len(te_rel))

        if "tft" in enabled:

            def train_tft():
                tft = TFTQuantile(lookback, dec_len, len(horizons), list(quantiles), models_cfg["tft"])
                tft.fit(
                    windows[tr_w],
                    decoder_windows(cal_all, dec_len, tr_pos),
                    y_matrix[tr_pos],
                    windows[va_w],
                    decoder_windows(cal_all, dec_len, va_pos),
                    y_matrix[va_pos],
                )
                return (
                    tft.predict(windows[va_w], decoder_windows(cal_all, dec_len, va_pos)),
                    tft.predict(windows[te_rel], decoder_windows(cal_all, dec_len, test_positions)),
                )

            preds_val_f["tft"], preds_test_f["tft"] = load_or_train("tft", fi, train_tft, len(va_w), len(te_rel))

        val_preds_by_fold.append(preds_val_f)
        test_preds_by_fold.append(preds_test_f)

    names_common = sorted(set.intersection(*[set(d.keys()) for d in val_preds_by_fold]))
    val_preds = {n: np.concatenate([d[n] for d in val_preds_by_fold], axis=0) for n in names_common}
    test_preds = {n: np.mean([d[n] for d in test_preds_by_fold], axis=0) for n in names_common}
    y_val_concat = np.concatenate(
        [y_matrix[pos_valid[tv_rel[va_rel_f]]] for _, va_rel_f in folds], axis=0
    )

    ensemble = QuantileEnsemble(models_cfg["ensemble"]["method"], list(quantiles))
    ens_info = ensemble.fit(val_preds, y_val_concat)
    print(f"[4/8] Ensemble weights: {ens_info['weights']}")

    val_ens = ensemble.predict(val_preds)
    test_ens = ensemble.predict(test_preds)

    median_qi = int(np.where(quantiles == 0.5)[0][0])
    resid_val = np.abs(y_val_concat - val_ens[:, :, median_qi])
    conformal = SplitConformal(risk_cfg["conformal_alpha"])
    conf_info = conformal.fit(resid_val)
    lo_test, hi_test = conformal.intervals(test_ens[:, :, median_qi])
    cov_test = empirical_coverage(lo_test, hi_test, y_matrix[test_positions])
    istats = interval_stats(lo_test, hi_test)
    print(f"[5/8] Conformal q_hat={np.round(conf_info['q_hat'], 4)} coverage_test={np.round(cov_test, 3)}")

    p_up_raw_val = p_up_from_quantiles(val_ens, quantiles)
    labels_val = (y_val_concat > 0).astype(int)
    calibrators = [
        PlattCalibrator().fit(p_up_raw_val[:, j], labels_val[:, j]) for j in range(len(horizons))
    ]
    p_up_raw_test = p_up_from_quantiles(test_ens, quantiles)
    p_up_cal_test = np.column_stack([c.predict(p_up_raw_test[:, j]) for j, c in enumerate(calibrators)])
    labels_test = (y_matrix[test_positions] > 0).astype(int)

    brier_raw = [brier_score(p_up_raw_test[:, j], labels_test[:, j]) for j in range(len(horizons))]
    brier_cal = [brier_score(p_up_cal_test[:, j], labels_test[:, j]) for j in range(len(horizons))]
    ll_raw = [log_loss_score(p_up_raw_test[:, j], labels_test[:, j]) for j in range(len(horizons))]
    ll_cal = [log_loss_score(p_up_cal_test[:, j], labels_test[:, j]) for j in range(len(horizons))]
    ece_raw = [
        expected_calibration_error(p_up_raw_test[:, j], labels_test[:, j])
        for j in range(len(horizons))
    ]
    ece_cal = [
        expected_calibration_error(p_up_cal_test[:, j], labels_test[:, j])
        for j in range(len(horizons))
    ]
    rel = reliability_bins(p_up_cal_test[:, 1], labels_test[:, 1])
    print(f"[6/8] Calibracion Brier raw={np.round(brier_raw, 3)} cal={np.round(brier_cal, 3)}")

    med_test = test_ens[:, :, median_qi]
    pred_metrics = predictive_metrics(y_matrix[test_positions], med_test)

    cost_rate_bps = {
        "fee_bps": bt_cfg["fee_bps"],
        "half_spread_bps": bt_cfg["half_spread_bps"],
        "slippage_bps": bt_cfg["slippage_bps"],
    }
    total_cost = sum(cost_rate_bps.values()) / 10_000

    fwd_vol_val = (
        targets[[f"fwd_vol_{h}h" for h in horizons]]
        .iloc[val_positions]
        .to_numpy(dtype=float)
    )
    vol_per_h_scaled = np.nanmean(fwd_vol_val, axis=0) * np.sqrt(horizons)
    h_star = choose_horizon(
        med_test.mean(axis=0), vol_per_h_scaled, np.full(len(horizons), total_cost),
        risk_cfg["lambda_risk"], risk_cfg["lambda_cost"],
    )
    h_star_hours = int(horizons[h_star])
    print(f"[7/8] Horizonte optimo por utilidad esperada: {h_star_hours}h")

    idx = df.index
    close_px = df["close"].to_numpy()

    p_up_series = pd.Series(p_up_cal_test[:, h_star], index=idx[test_positions])
    edge_series = pd.Series(med_test[:, h_star], index=idx[test_positions])
    target_exposure = np.where(
        (p_up_series >= risk_cfg["min_p_up"]) & (edge_series >= risk_cfg["min_edge_net"]),
        risk_cfg["max_position"] * p_up_series,
        0.0,
    )
    target_s = pd.Series(target_exposure, index=idx[test_positions])
    smooth_s = target_s.ewm(alpha=0.5, adjust=False).mean()

    rows = []
    strategies = {
        "buy_and_hold": pd.Series(buy_and_hold(len(te_rel)), index=idx[test_positions]),
        "no_trade": pd.Series(no_trade(len(te_rel)), index=idx[test_positions]),
        "naive_momentum": pd.Series(
            naive_momentum_signal(close_px)[test_positions], index=idx[test_positions]
        ),
        "ma_crossover": pd.Series(
            ma_crossover_signal(close_px)[test_positions], index=idx[test_positions]
        ),
        "momentum": pd.Series(momentum_signal(close_px)[test_positions], index=idx[test_positions]),
        "rsi_meanrev": pd.Series(
            rsi_mean_reversion_signal(close_px)[test_positions], index=idx[test_positions]
        ),
        "model_strategy": smooth_s,
    }
    prices_bt = df.iloc[test_positions][["open", "close"]]
    for name, sig in strategies.items():
        res = run_backtest(
            prices_bt, sig,
            bt_cfg["fee_bps"], bt_cfg["half_spread_bps"], bt_cfg["slippage_bps"],
            bt_cfg["initial_capital"], risk_cfg["max_position"],
        )
        m = strategy_metrics(
            res.equity, res.strategy_returns_net, res.positions_executed, res.trades,
            bt_cfg["annualization_factor"], bt_cfg["risk_free_annual"],
        )
        rows.append({"name": name, **m})

    baseline_sharpes = [r["sharpe"] for r in rows[:-1] if r["name"] != "no_trade"]
    beats = rows[-1]["sharpe"] > max(baseline_sharpes)
    status = "CANDIDATE" if beats else "RESEARCH"
    print(f"[8/8] Modelo vs baselines: sharpe={rows[-1]['sharpe']:.2f} -> estado {status}")

    reports_dir = path_under_root("reports")
    write_backtest_report(reports_dir / "backtest_report.md", rows, cost_rate_bps)

    last_rel = te_rel[-1]
    med_last = med_test[-1]
    width_last = float(hi_test[-1, h_star] - lo_test[-1, h_star])
    ood_width_threshold = float(np.quantile(hi_test[:, h_star] - lo_test[:, h_star], 0.95))
    ood_flag = width_last > ood_width_threshold
    exp_ret_last = float(med_last[h_star])
    vol_scaled_last = float(targets[f"fwd_vol_{h_star_hours}h"].iloc[last_rel + lookback - 1]) * np.sqrt(
        h_star_hours
    )
    net_exp = (
        exp_ret_last
        - risk_cfg["lambda_risk"] * vol_scaled_last
        - risk_cfg["lambda_cost"] * total_cost
    )
    p_up_last = float(p_up_cal_test[-1, h_star])
    action, reasons = decide_action(
        p_up_last, net_exp, quality,
        risk_cfg["min_p_up"], risk_cfg["min_edge_net"], risk_cfg["min_data_quality"],
        ood_flag,
    )
    width_ratio = width_last / max(vol_scaled_last, 1e-9)
    conf = confidence_score(p_up_last, width_ratio, quality)
    alloc = position_size(risk_cfg["max_position"], p_up_last) if action == "BUY" else 0.0
    p_tgt, p_stop = prob_target_stop(
        test_ens[-1, h_star, :], quantiles, risk_cfg["target_pct"], risk_cfg["stop_pct"]
    )

    snapshot = {
        "asset": f"{symbol[:-4]}/{symbol[-4:]}",
        "timestamp": str(idx[test_positions[-1]]),
        "action": action,
        "confidence": conf,
        "horizon_hours": h_star_hours,
        "expected_return": round(exp_ret_last, 6),
        "expected_return_net_of_risk_cost": round(net_exp, 6),
        "prediction_interval": {
            "lower": round(float(lo_test[-1, h_star]), 6),
            "upper": round(float(hi_test[-1, h_star]), 6),
            "coverage_target": round(1 - risk_cfg["conformal_alpha"], 2),
        },
        "probabilities": {
            "positive_return": round(p_up_last, 4),
            "target_hit": round(p_tgt, 4),
            "stop_hit": round(p_stop, 4),
        },
        "strategy": {"number_of_transactions": 1, "allocation_pct": alloc},
        "risk": {
            "stop_pct": risk_cfg["stop_pct"],
            "target_pct": risk_cfg["target_pct"],
            "status": "PASS" if action != "NO TRADE" else "BLOCKED",
            "reasons": reasons,
        },
        "model": {
            "forecast_models": enabled,
            "ensemble_weights": ens_info["weights"],
            "decision_rule": "rule_based_v1 (RL en fase posterior)",
            "experiment_id": experiment_id,
        },
        "status": "VALID",
    }
    (reports_dir / "decision_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    ctx = {
        "symbol": symbol,
        "timeframe": timeframe,
        "n_rows": len(df),
        "start": str(idx[0]),
        "end": str(idx[-1]),
        "quality_score": quality,
        "train_period": f"{sp['train_start']}..{sp['train_end']}",
        "val_period": f"{sp['val_start']}..{sp['val_end']}",
        "test_period": f"{sp['test_start']}..{sp.get('test_end') or 'now'}",
        "models": enabled,
        "ensemble_weights": ens_info["weights"],
        "q_hat": conf_info["q_hat"],
        "alpha": risk_cfg["conformal_alpha"],
        "coverage_test": cov_test.tolist(),
        "mean_width": istats["mean_width"],
        "pred_metrics_test": pred_metrics,
        "brier_test": brier_cal,
        "ece_test": ece_cal,
    }
    write_research_report(reports_dir / "research_report.md", ctx)

    card = {
        "symbol": symbol,
        "timeframe": timeframe,
        "n_rows": len(df),
        "start": str(idx[0]),
        "end": str(idx[-1]),
        "n_features": X.shape[1],
        "horizons": feat_cfg["horizons_hours"],
        "quantiles": feat_cfg["quantiles"],
        "metrics_test": {**pred_metrics, "brier_cal": brier_cal, "coverage": cov_test.tolist()},
        "version": f"0.1.0+{experiment_id}",
        "status": status,
    }
    write_model_card(reports_dir / "model_card.md", card)

    cal_ctx = {
        "alpha": risk_cfg["conformal_alpha"],
        "coverage_test": cov_test.tolist(),
        "mean_width": istats["mean_width"],
        "brier_raw_test": brier_raw,
        "brier_test": brier_cal,
        "logloss_raw_test": ll_raw,
        "logloss_test": ll_cal,
        "ece_raw_test": ece_raw,
        "ece_test": ece_cal,
        "reliability": rel,
    }
    write_calibration_report(reports_dir / "calibration_report.md", cal_ctx)

    path_under_root("experiments")
    log_experiment(ROOT / "experiments" / "log.jsonl", {
        "experiment_id": experiment_id,
        "dataset_version": validated_path.name,
        "models": enabled,
        "ensemble_weights": ens_info["weights"],
        "seed": seed,
        "test_metrics": {**pred_metrics, "brier_cal": brier_cal, "coverage": cov_test.tolist()},
        "strategy_sharpe": rows[-1]["sharpe"],
        "status": status,
    })

    registry_path = path_under_root("models") / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {"models": []}
    registry["models"].append({
        "experiment_id": experiment_id,
        "version": card["version"],
        "status": status,
        "date": str(pd.Timestamp.now(tz="UTC")),
    })
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print("\n=== DECISION SNAPSHOT ===")
    print(json.dumps(snapshot, indent=2))
    print("\nReportes generados en reports/")


if __name__ == "__main__":
    main()
