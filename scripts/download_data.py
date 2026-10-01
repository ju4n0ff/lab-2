from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, path_under_root
from src.data.binance import BinanceProvider, INTERVAL_MS
from src.data.validation import validate_ohlcv


def main() -> None:
    cfg = load_config("data")
    symbol = cfg["market"]["symbol"]
    timeframe = cfg["market"]["timeframe"]
    start = cfg["history"]["start"]
    end = cfg["history"].get("end")

    provider = BinanceProvider()
    freq_ms = INTERVAL_MS[timeframe]
    start_ms = int(__import__("pandas").Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(
        (
            __import__("pandas").Timestamp(end, tz="UTC")
            if end
            else __import__("pandas").Timestamp.now(tz="UTC")
        ).timestamp()
        * 1000
    )

    print(f"Descargando {symbol} {timeframe} desde {start} ...")
    raw = provider.get_ohlcv(symbol, timeframe, start_ms, end_ms)
    raw_dir = path_under_root(cfg["paths"]["raw"])
    out_path = raw_dir / f"{symbol.lower()}_{timeframe}.parquet"
    raw.to_parquet(out_path, index=False)
    print(f"Guardado: {out_path} ({len(raw)} filas)")

    clean, rep = validate_ohlcv(raw, freq_ms)
    clean_path = raw_dir / f"{symbol.lower()}_{timeframe}_validated.parquet"
    clean.to_parquet(clean_path, index=False)
    rep_path = raw_dir / f"{symbol.lower()}_{timeframe}_validation.json"
    rep_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"Validacion: quality={rep['quality_score']} gaps={rep['gaps']} -> {clean_path}")


if __name__ == "__main__":
    main()
