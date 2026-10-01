from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series
    strategy_returns_net: pd.Series
    positions_executed: pd.Series
    trades: pd.DataFrame
    metrics: dict = field(default_factory=dict)


def run_backtest(
    prices: pd.DataFrame,
    signal_positions: pd.Series,
    fee_bps: float,
    half_spread_bps: float,
    slippage_bps: float,
    initial_capital: float = 10_000.0,
    max_position: float = 1.0,
) -> BacktestResult:
    """Backtest con ejecucion causal.

    La senal decidida al cierre de t se ejecuta en la apertura de t+1.
    El retorno del activo se mide apertura a apertura. Costes sobre el turnover:
    comision + medio spread + slippage por unidad de posicion cambiada.
    """
    idx = prices.index
    sig = signal_positions.reindex(idx).fillna(0.0).clip(0.0, max_position).astype(float)

    exec_pos = sig.shift(1).fillna(0.0)
    open_ret = prices["open"].pct_change().fillna(0.0)

    gross_ret = exec_pos.shift(1).fillna(0.0) * open_ret

    turnover = exec_pos.diff().abs()
    turnover.iloc[0] = abs(exec_pos.iloc[0])
    cost_rate = (fee_bps + half_spread_bps + slippage_bps) / 10_000.0
    costs = turnover * cost_rate

    net_ret = gross_ret - costs
    equity = initial_capital * (1.0 + net_ret).cumprod()

    trades = _extract_trades(exec_pos, prices["open"], cost_rate)

    return BacktestResult(
        equity=equity,
        strategy_returns_net=net_ret,
        positions_executed=exec_pos,
        trades=trades,
        metrics={},
    )


def _extract_trades(exec_pos: pd.Series, open_price: pd.Series, cost_rate: float) -> pd.DataFrame:
    pos = exec_pos.to_numpy()
    px = open_price.to_numpy()
    ts = exec_pos.index
    rows = []
    in_trade = False
    entry_i = None
    entry_px = None
    for i in range(len(pos)):
        if not in_trade and pos[i] > 1e-12:
            in_trade = True
            entry_i = i
            entry_px = px[i]
        elif in_trade and pos[i] <= 1e-12:
            exit_i = i - 1 if i > 0 else 0
            rows.append(
                {
                    "entry_time": ts[entry_i],
                    "exit_time": ts[i],
                    "entry_price": entry_px,
                    "exit_price": px[exit_i],
                    "gross_return": px[exit_i] / entry_px - 1.0,
                    "costs_roundtrip_pct": 2 * cost_rate,
                    "bars_held": exit_i - entry_i + 1,
                }
            )
            in_trade = False
    if in_trade and entry_i is not None:
        last = len(pos) - 1
        rows.append(
            {
                "entry_time": ts[entry_i],
                "exit_time": ts[last],
                "entry_price": entry_px,
                "exit_price": px[last],
                "gross_return": px[last] / entry_px - 1.0,
                "costs_roundtrip_pct": 2 * cost_rate,
                "bars_held": last - entry_i + 1,
            }
        )
    return pd.DataFrame(rows)
