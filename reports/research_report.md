# Research Report

Generado: 2026-08-27T17:36:45.931685+00:00

## FACT
- Activo: BTCUSDT 1h
- Observaciones: 49369 velas (2021-01-07 23:00:00+00:00 a 2026-08-27 17:00:00+00:00)
- Data quality score: 100.0/100
- Splits temporales: train 2021-01-01..2024-06-30, val 2024-07-01..2025-06-30, test 2025-07-01..now
- Modelos entrenados: xgboost
- Pesos ensemble (validacion): {"xgboost": 1.0}
- Conformal q_hat por horizonte: [0.01631228028358064, 0.04047252279146618, 0.09979383852415276]
- Cobertura empirica en test (target 90%): [0.9311834438416717, 0.9374120956399438, 0.9114928671890697]

## INFERENCE
- Metricas predictivas del ensemble en TEST: {"mae": 0.02175598405301571, "rmse": 0.03727291524410248, "directional_accuracy": 0.5201259125309758}
- Calibracion P(up) en TEST: Brier=[0.2495951004699469, 0.24928084019165803, 0.25997459687556046], ECE=[0.020546423851924837, 0.017119626137117674, 0.11018544220245675]
- Comparacion contra baselines netos de costes: ver backtest_report.md.

## ASSUMPTION
- Los costes modelados (fee/spread/slippage) representan razonablemente la ejecucion real.
- La liquidez horaria de BTC/USDT absorbe el tamano de posicion simulado sin impacto de mercado.
- La distribucion de retornos es suficientemente estable para que conformal calibre fuera de muestra.

## LIMITATION
- Sin datos de order book; no se modela impacto de mercado ni partial fills.
- Sin variables macro/sentiment; el modelo solo ve precio/volumen/calendario.
- RL (PPO/SAC), monitoring de drift y paper trading quedan para fases siguientes.
- Resultados historicos no garantizan rendimiento futuro.