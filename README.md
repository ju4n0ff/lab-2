# AOTTS-MVP — Timing Óptimo de Compra/Venta (MVP de investigación)

Sistema de investigación cuantitativa que estima probabilísticamente momentos favorables de
entrada/salida usando **solo información disponible hasta cada instante de decisión**.
Implementa el MVP (fases 0–9) del `PROMPT_MAESTRO_AOTTS_OPENCODE.txt`.

> **Aviso**: es un sistema de decision support / investigación. NO garantiza ganancias,
> NO es asesoramiento financiero. Toda salida incluye incertidumbre y riesgos.

## Qué hace

1. **Datos**: descarga OHLCV 1h de BTC/USDT desde la API pública de Binance (~49k velas,
   2021→hoy), valida timestamps/gaps/duplicados y asigna un data quality score.
2. **Features causales**: 25 features (precio, tendencia, momentum, volatilidad, volumen,
   estructura, calendario) construidas solo con ventanas trailing.
3. **Targets multi-horizonte**: retorno, dirección, volatilidad, MFE/MAE a 4h/24h/168h.
4. **Modelos**: XGBoost quantile + LSTM (TensorFlow/Keras) + TFT-lite (TensorFlow/Keras,
   con VSN, encoder-decoder LSTM y atención interpretable).
5. **Ensemble**: pesos aprendidos solo en validación (pinball-weighted o fallback uniforme).
6. **Incertidumbre**: split conformal por horizonte (PI ~90%) + calibración Platt de P(up)
   con Brier/LogLoss/ECE antes y después.
7. **Backtesting**: ejecución causal (señal t → apertura t+1), fee+spread+slippage,
   métricas Sharpe/Sortino/Calmar/maxDD/win rate/etc., comparado contra 6 baselines.
8. **Decisión**: horizonte óptimo por utilidad esperada ajustada por riesgo, reglas
   BUY/WAIT/NO TRADE documentadas, snapshot JSON estructurado.
9. **Reportes**: research/backtest/model card/calibration en Markdown con secciones
   FACT / INFERENCE / ASSUMPTION / LIMITATION.

## Resultados actuales (honestos)

En el test out-of-sample (jul 2025 – ago 2026, mercado bajista: buy&hold -41%) el modelo
**NO supera a los baselines netos de costes** → estado del registry: `RESEARCH`.
La cobertura conformal empírica fue 93–94% (objetivo 90%). Estos resultados negativos se
reportan tal cual, como exige la metodología.

## Instalación

Requiere Python 3.13 (TensorFlow 2.21 no publica wheels para 3.14).

```powershell
py -3.13 -m venv .venv
.venv\Scripts\pip install -e .
# o: .venv\Scripts\pip install tensorflow-cpu xgboost lightgbm scikit-learn pandas numpy pyarrow pyyaml optuna pydantic requests scipy pytest
```

## Uso

```powershell
.venv\Scripts\python scripts\download_data.py   # descarga + validación
.venv\Scripts\python scripts\run_pipeline.py    # pipeline end-to-end
.venv\Scripts\python -m pytest tests -q         # tests (unitarios, leakage, backtest)
```

- El entrenamiento es **reanudable**: cada modelo/fold cachea sus predicciones en
  `models/artifacts/`. Borra esa carpeta o usa `AOTTS_FRESH=1` para reentrenar.
- `configs/models.yaml` tiene `quick: true` para smoke tests (épocas reducidas).

## Arquitectura

```
Binance API -> validación -> features -> targets
                                   |
        +------------+-------------+-------------+
        |            |                           |
     XGBoost      LSTM (TF)                  TFT-lite (TF)
        +------------+----------+----------------+
                     |
                 ENSEMBLE (pesos en val)
                     |
        CONFORMAL (PI) + CALIBRACIÓN (P up)
                     |
              DECISION ENGINE -> RISK CHECKS -> snapshot JSON
                     |
        BACKTEST (costes reales) vs BASELINES -> REPORTES
```

Separación estricta: `data -> features -> prediction -> uncertainty -> decision -> risk -> execution`.

## Anti-leakage (verificado con tests)

- Features solo trailing (`tests/leakage`): estabilidad de prefijos, ventanas no centradas.
- Scaler ajustado solo con train; targets son labels, nunca features.
- Splits estrictamente temporales; walk-forward expandente dentro de train+val;
  TEST final nunca se usa para tuning; pesos/calibración/conformal solo en validación.

## Configuración

Todo en `configs/*.yaml`: datos y splits (`data.yaml`), features/horizontes/cuantiles
(`features.yaml`), hiperparámetros (`models.yaml`), costes (`backtest.yaml`),
límites de riesgo y reglas de decisión (`risk.yaml`). Sin credenciales en código
(`.env.example`; los klines públicos no requieren keys).

## Limitaciones conocidas

- Sin order book: sin impacto de mercado ni partial fills; slippage fijo.
- P(target)/P(stop) derivan de la CDF marginal, no de la ruta del precio.
- Cobertura conformal marginal, no condicional por régimen.
- RL (PPO/SAC), risk overlay completo, monitoring/drift, API y paper trading:
  fases siguientes del prompt maestro (10–16).

## Próximas fases

PPO/SAC sobre el entorno ya definido, risk overlay con autoridad de bloqueo,
no-trade engine explícito, monitoring de drift con SAFE MODE, FastAPI y paper trading.
