from __future__ import annotations

import numpy as np
import pandas as pd


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series, ann_factor: float) -> float:
    years = len(equity) / ann_factor
    if years <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0)


def annualized_volatility(returns: pd.Series, ann_factor: float) -> float:
    return float(returns.std(ddof=1) * np.sqrt(ann_factor))


def sharpe_ratio(returns: pd.Series, ann_factor: float, rf_annual: float = 0.0) -> float:
    excess = returns - (rf_annual / ann_factor)
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(ann_factor))


def downside_deviation(returns: pd.Series, ann_factor: float, mar: float = 0.0) -> float:
    diff = np.minimum(returns - mar, 0.0)
    return float(np.sqrt(np.mean(diff**2)) * np.sqrt(ann_factor))


def sortino_ratio(returns: pd.Series, ann_factor: float, mar: float = 0.0) -> float:
    dd = downside_deviation(returns, ann_factor, mar)
    if dd == 0 or np.isnan(dd):
        return float("nan")
    return float(returns.mean() * ann_factor / dd)


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    dd = drawdown_series(equity)
    return float(dd.min())


def average_drawdown(equity: pd.Series) -> float:
    dd = drawdown_series(equity)
    neg = dd[dd < 0]
    return float(neg.mean()) if len(neg) else 0.0


def calmar_ratio(equity: pd.Series, ann_factor: float) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return float("nan")
    return float(cagr(equity, ann_factor) / mdd)


def win_rate(trades: pd.DataFrame) -> float:
    if len(trades) == 0:
        return float("nan")
    return float((trades["gross_return"] > 0).mean())


def profit_factor(trades: pd.DataFrame) -> float:
    if len(trades) == 0:
        return float("nan")
    gains = trades.loc[trades["gross_return"] > 0, "gross_return"].sum()
    losses = -trades.loc[trades["gross_return"] < 0, "gross_return"].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def expectancy(trades: pd.DataFrame) -> float:
    if len(trades) == 0:
        return float("nan")
    return float(trades["gross_return"].mean())


def turnover(positions: pd.Series) -> float:
    return float(positions.diff().abs().sum() / max(len(positions), 1))


def exposure(positions: pd.Series) -> float:
    return float((positions.abs() > 1e-12).mean())


def strategy_metrics(
    equity: pd.Series,
    returns_net: pd.Series,
    positions: pd.Series,
    trades: pd.DataFrame,
    ann_factor: float,
    rf_annual: float = 0.0,
) -> dict:
    return {
        "total_return": total_return(equity),
        "cagr": cagr(equity, ann_factor),
        "ann_volatility": annualized_volatility(returns_net, ann_factor),
        "sharpe": sharpe_ratio(returns_net, ann_factor, rf_annual),
        "sortino": sortino_ratio(returns_net, ann_factor),
        "calmar": calmar_ratio(equity, ann_factor),
        "max_drawdown": max_drawdown(equity),
        "avg_drawdown": average_drawdown(equity),
        "win_rate": win_rate(trades),
        "profit_factor": profit_factor(trades),
        "expectancy": expectancy(trades),
        "n_trades": int(len(trades)),
        "avg_bars_held": float(trades["bars_held"].mean()) if len(trades) else float("nan"),
        "turnover_per_bar": turnover(positions),
        "exposure": exposure(positions),
        "downside_deviation": downside_deviation(returns_net, ann_factor),
    }


def predictive_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "directional_accuracy": dir_acc,
    }
