from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    max_drawdown,
    sharpe_ratio,
    strategy_metrics,
    total_return,
    win_rate,
)


def test_total_return():
    eq = pd.Series([100.0, 110.0, 121.0])
    assert total_return(eq) == pytest.approx(0.21)


def test_sharpe_constant_positive_returns_is_high():
    r = pd.Series([0.001] * 100)
    s = sharpe_ratio(r, ann_factor=8760)
    assert s > 100


def test_max_drawdown_known_curve():
    eq = pd.Series([100, 120, 90, 95, 130])
    assert max_drawdown(eq) == pytest.approx(90 / 120 - 1)


def test_win_rate():
    trades = pd.DataFrame({"gross_return": [0.1, -0.05, 0.2]})
    assert win_rate(trades) == pytest.approx(2 / 3)


def test_strategy_metrics_keys():
    eq = pd.Series([100, 101, 99, 102])
    rets = pd.Series([0.01, -0.02, 0.03])
    pos = pd.Series([1.0, 1.0, 0.0])
    trades = pd.DataFrame({"gross_return": [0.02], "bars_held": [3]})
    m = strategy_metrics(eq, rets, pos, trades, 8760)
    for key in ["total_return", "sharpe", "max_drawdown", "n_trades", "exposure"]:
        assert key in m
