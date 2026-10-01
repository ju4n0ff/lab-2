from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import requests

from .provider import MarketDataProvider

INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
]


class BinanceProvider(MarketDataProvider):
    """Klines publicos de Binance. No requiere credenciales para datos historicos."""

    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or os.environ.get("BINANCE_API_URL", "https://api.binance.com")
        self.session = requests.Session()

    def get_ohlcv(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        if interval not in INTERVAL_MS:
            raise ValueError(f"intervalo no soportado: {interval}")
        url = f"{self.base_url}/api/v3/klines"
        rows: list[list] = []
        cursor = start_ms
        while cursor < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            }
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            rows.extend(batch)
            last_open = batch[-1][0]
            next_cursor = last_open + INTERVAL_MS[interval]
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            time.sleep(0.15)

        if not rows:
            raise RuntimeError("Binance devolvio 0 velas para el rango solicitado")

        arr = np.array(rows, dtype=object)
        df = pd.DataFrame(
            {
                "open_time": pd.to_datetime(arr[:, 0].astype(np.int64), unit="ms", utc=True),
                "open": arr[:, 1].astype(float),
                "high": arr[:, 2].astype(float),
                "low": arr[:, 3].astype(float),
                "close": arr[:, 4].astype(float),
                "volume": arr[:, 5].astype(float),
                "close_time": pd.to_datetime(arr[:, 6].astype(np.int64), unit="ms", utc=True),
                "quote_volume": arr[:, 7].astype(float),
                "trades": arr[:, 8].astype(int),
            }
        )
        return df[COLUMNS]
