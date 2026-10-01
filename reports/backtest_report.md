# Backtest Report

Generado: 2026-08-27T17:36:45.920723+00:00

Costes por lado: fee=10bps, half_spread=2bps, slippage=5bps. Ejecucion: senal t -> apertura t+1.

| Estrategia | TotalRet | CAGR | Sharpe | Sortino | MaxDD | WinRate | Trades | Exposure |
|---|---|---|---|---|---|---|---|---|
| buy_and_hold | -36.35% | -32.80% | -0.74 | -1.03 | -53.74% | +0.00% | 1 | +99.99% |
| no_trade | +0.00% | +0.00% | n/d | n/d | +0.00% | n/d | 0 | +0.00% |
| naive_momentum | -99.99% | -99.96% | -23.36 | -28.71 | -99.99% | +51.20% | 2574 | +50.02% |
| ma_crossover | -48.30% | -44.04% | -2.01 | -2.80 | -57.45% | +39.20% | 125 | +50.29% |
| momentum | -92.50% | -89.77% | -7.60 | -10.21 | -93.52% | +44.81% | 703 | +50.89% |
| rsi_meanrev | -36.35% | -32.80% | -0.74 | -1.03 | -53.74% | +0.00% | 1 | +99.99% |
| model_strategy | +1.85% | +1.62% | 0.91 | 3.39 | -0.53% | +0.00% | 1 | +0.83% |

## FACT
- Todas las estrategias evaluadas sobre el mismo periodo TEST con los mismos costes.

## LIMITATION
- Ejecucion aproximada apertura-a-apertura sin partial fills ni latencia.
- Una estrategia que colapse duplicando costes debe clasificarse fragil (ver stress testing pendiente).