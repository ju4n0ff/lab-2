# Model Card

Generado: 2026-08-27T17:36:45.937914+00:00

- **Proposito**: forecast probabilistico multi-horizonte de retornos BTCUSDT.
- **Uso previsto**: decision support de investigacion. **Uso prohibido**: asesoramiento financiero, trading automatico live.
- **Datos**: 49369 velas 1h (2021-01-07 23:00:00+00:00 a 2026-08-27 17:00:00+00:00).
- **Features**: 25 features causales (precio/tendencia/momentum/volatilidad/volumen/calendario).
- **Target**: retornos forward a [4, 24, 168] con quantiles [0.1, 0.25, 0.5, 0.75, 0.9].
- **Arquitectura**: TFT-lite (VSN+LSTM+atencion) y LSTM en TensorFlow/Keras; XGBoost quantile; ensemble ponderado.
- **Metricas TEST**: {"mae": 0.02175598405301571, "rmse": 0.03727291524410248, "directional_accuracy": 0.5201259125309758, "brier_cal": [0.2495951004699469, 0.24928084019165803, 0.25997459687556046], "coverage": [0.9311834438416717, 0.9374120956399438, 0.9114928671890697]}
- **Calibracion**: Platt sobre P(up); Brier/ECE reportados en calibration_report.md.
- **Limitaciones**: sin order book, sin sentimiento, sin macro; cobertura conformal marginal no condicional.
- **Modos de fallo conocidos**: regimenes de alta volatilidad, cambios estructurales post-entrenamiento.
- **Version**: 0.1.0+4161bda3 - Estado registry: CANDIDATE