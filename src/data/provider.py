from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class MarketDataProvider(ABC):
    """Interface desacoplada de cualquier proveedor concreto."""

    @abstractmethod
    def get_ohlcv(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        raise NotImplementedError
