# Calibration Report

Generado: 2026-08-27T17:36:45.939225+00:00

## Cobertura conformal (TEST)

- Alpha: 0.1, cobertura objetivo: 90%
- Cobertura empirica por horizonte: [0.9311834438416717, 0.9374120956399438, 0.9114928671890697]
- Ancho medio de intervalo: [0.03262456056716701, 0.08094504558292266, 0.19958767704833724]

## P(up) calibrada (TEST)

- Brier antes: [0.2492679347723226, 0.2491322875405509, 0.24969736899238354] / despues: [0.2495951004699469, 0.24928084019165803, 0.25997459687556046]
- LogLoss antes: [0.6916825423531244, 0.691376247105652, 0.6925664960585994] / despues: [0.692345512065497, 0.6916542395349745, 0.7136801704373957]
- ECE antes: [0.011497825330039107, 0.01686968650799009, 0.03720058466865296] / despues: [0.020546423851924837, 0.017119626137117674, 0.11018544220245675]

## Reliability bins (post-calibracion)

{
  "bin_center": [
    0.25,
    0.35000000000000003,
    0.45,
    0.55,
    0.6500000000000001,
    0.75,
    0.8500000000000001
  ],
  "predicted_mean": [
    0.21339031903057387,
    0.37852581303072885,
    0.49398879224633213,
    0.5253301506358504,
    0.6133541043327658,
    0.7564266360292159,
    0.8118118414135999
  ],
  "empirical_freq": [
    0.0,
    0.0,
    0.46403712296983757,
    0.5102108400485705,
    0.8421052631578947,
    1.0,
    1.0
  ],
  "count": [
    7,
    1,
    862,
    9059,
    19,
    5,
    1
  ]
}

## FACT / LIMITATION
- FACT: coberturas y scores calculados solo sobre TEST nunca visto durante tuning.
- LIMITATION: cobertura marginal; la cobertura condicional por regimen puede diferir.