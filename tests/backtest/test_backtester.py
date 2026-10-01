from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting.backtester import run_backtest


def make_prices(n=100, drift=0.0):
    rng = np.random.default_rng(3)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.001, n))))
    open_ = close.shift(1).fillna(close.iloc[0])
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_.to_numpy(), "close": close.to_numpy()}, index=idx)


def test_no_same_bar_execution():
    prices = make_prices(50)
    sig = pd.Series(0.0, index=prices.index)
    sig.iloc[10] = 1.0
    res = run_backtest(prices, sig, fee_bps=0, half_spread_bps=0, slippage_bps=0)
    assert res.positions_executed.iloc[10] == 0.0
    assert res.positions_executed.iloc[11] == 1.0


def test_costs_applied_on_entry_and_exit():
    prices = make_prices(20)
    sig = pd.Series(0.0, index=prices.index)
    sig.iloc[5:15] = 1.0
    cost_rate = (10 + 2 + 5) / 10_000
    res_free = run_backtest(prices, sig, 0, 0, 0)
    res_cost = run_backtest(prices, sig, 10, 2, 5)
    expected_drag = 2 * cost_rate * 10_000
    diff_bps = (res_free.equity.iloc[-1] / res_cost.equity.iloc[-1] - 1) * 10_000
    assert diff_bps == pytest.approx(expected_drag, rel=0.05)


def test_long_only_clipped_by_max_position():
    prices = make_prices(30)
    sig = pd.Series(2.5, index=prices.index)
    res = run_backtest(prices, sig, 10, 2, 5, max_position=1.0)
    assert (res.positions_executed <= 1.0).all()


def test_no_trade_flat_equity_with_costs():
    prices = make_prices(40)
    sig = pd.Series(0.0, index=prices.index)
    res = run_backtest(prices, sig, 10, 2, 5)
    assert res.equity.nunique() == 1
