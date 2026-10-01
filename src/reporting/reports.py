from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _fmt(x, pct: bool = False, nd: int = 4) -> str:
    if x is None:
        return "n/d"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if xf != xf:
        return "n/d"
    if pct:
        return f"{xf * 100:+.2f}%"
    return f"{xf:.{nd}f}"


def write_research_report(path: Path, ctx: dict) -> None:
    lines = [
        "# Research Report",
        "",
        f"Generado: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## FACT",
        f"- Activo: {ctx['symbol']} {ctx['timeframe']}",
        f"- Observaciones: {ctx['n_rows']} velas ({ctx['start']} a {ctx['end']})",
        f"- Data quality score: {ctx['quality_score']}/100",
        f"- Splits temporales: train {ctx['train_period']}, val {ctx['val_period']}, test {ctx['test_period']}",
        f"- Modelos entrenados: {', '.join(ctx['models'])}",
        f"- Pesos ensemble (validacion): {json.dumps(ctx['ensemble_weights'])}",
        f"- Conformal q_hat por horizonte: {json.dumps(ctx['q_hat'])}",
        f"- Cobertura empirica en test (target {1 - ctx['alpha']:.0%}): {json.dumps(ctx['coverage_test'])}",
        "",
        "## INFERENCE",
        f"- Metricas predictivas del ensemble en TEST: {json.dumps(ctx['pred_metrics_test'], default=str)}",
        f"- Calibracion P(up) en TEST: Brier={_fmt(ctx.get('brier_test'))}, ECE={_fmt(ctx.get('ece_test'))}",
        "- Comparacion contra baselines netos de costes: ver backtest_report.md.",
        "",
        "## ASSUMPTION",
        "- Los costes modelados (fee/spread/slippage) representan razonablemente la ejecucion real.",
        "- La liquidez horaria de BTC/USDT absorbe el tamano de posicion simulado sin impacto de mercado.",
        "- La distribucion de retornos es suficientemente estable para que conformal calibre fuera de muestra.",
        "",
        "## LIMITATION",
        "- Sin datos de order book; no se modela impacto de mercado ni partial fills.",
        "- Sin variables macro/sentiment; el modelo solo ve precio/volumen/calendario.",
        "- RL (PPO/SAC), monitoring de drift y paper trading quedan para fases siguientes.",
        "- Resultados historicos no garantizan rendimiento futuro.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_backtest_report(path: Path, rows: list[dict], costs: dict) -> None:
    lines = [
        "# Backtest Report",
        "",
        f"Generado: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Costes por lado: fee={costs['fee_bps']}bps, half_spread={costs['half_spread_bps']}bps, "
        f"slippage={costs['slippage_bps']}bps. Ejecucion: senal t -> apertura t+1.",
        "",
        "| Estrategia | TotalRet | CAGR | Sharpe | Sortino | MaxDD | WinRate | Trades | Exposure |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {_fmt(r['total_return'], pct=True)} | {_fmt(r['cagr'], pct=True)} "
            f"| {_fmt(r['sharpe'], nd=2)} | {_fmt(r['sortino'], nd=2)} | {_fmt(r['max_drawdown'], pct=True)} "
            f"| {_fmt(r['win_rate'], pct=True)} | {r['n_trades']} | {_fmt(r['exposure'], pct=True)} |"
        )
    lines += [
        "",
        "## FACT",
        "- Todas las estrategias evaluadas sobre el mismo periodo TEST con los mismos costes.",
        "",
        "## LIMITATION",
        "- Ejecucion aproximada apertura-a-apertura sin partial fills ni latencia.",
        "- Una estrategia que colapse duplicando costes debe clasificarse fragil (ver stress testing pendiente).",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_model_card(path: Path, card: dict) -> None:
    lines = [
        "# Model Card",
        "",
        f"Generado: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- **Proposito**: forecast probabilistico multi-horizonte de retornos {card['symbol']}.",
        f"- **Uso previsto**: decision support de investigacion. **Uso prohibido**: asesoramiento financiero, trading automatico live.",
        f"- **Datos**: {card['n_rows']} velas {card['timeframe']} ({card['start']} a {card['end']}).",
        f"- **Features**: {card['n_features']} features causales (precio/tendencia/momentum/volatilidad/volumen/calendario).",
        f"- **Target**: retornos forward a {card['horizons']} con quantiles {card['quantiles']}.",
        f"- **Arquitectura**: TFT-lite (VSN+LSTM+atencion) y LSTM en TensorFlow/Keras; XGBoost quantile; ensemble ponderado.",
        f"- **Metricas TEST**: {json.dumps(card['metrics_test'], default=str)}",
        f"- **Calibracion**: Platt sobre P(up); Brier/ECE reportados en calibration_report.md.",
        f"- **Limitaciones**: sin order book, sin sentimiento, sin macro; cobertura conformal marginal no condicional.",
        f"- **Modos de fallo conocidos**: regimenes de alta volatilidad, cambios estructurales post-entrenamiento.",
        f"- **Version**: {card['version']} - Estado registry: {card['status']}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_calibration_report(path: Path, ctx: dict) -> None:
    lines = [
        "# Calibration Report",
        "",
        f"Generado: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Cobertura conformal (TEST)",
        "",
        f"- Alpha: {ctx['alpha']}, cobertura objetivo: {1 - ctx['alpha']:.0%}",
        f"- Cobertura empirica por horizonte: {json.dumps(ctx['coverage_test'])}",
        f"- Ancho medio de intervalo: {json.dumps(ctx['mean_width'])}",
        "",
        "## P(up) calibrada (TEST)",
        "",
        f"- Brier antes: {_fmt(ctx.get('brier_raw_test'))} / despues: {_fmt(ctx.get('brier_test'))}",
        f"- LogLoss antes: {_fmt(ctx.get('logloss_raw_test'))} / despues: {_fmt(ctx.get('logloss_test'))}",
        f"- ECE antes: {_fmt(ctx.get('ece_raw_test'))} / despues: {_fmt(ctx.get('ece_test'))}",
        "",
        "## Reliability bins (post-calibracion)",
        "",
        json.dumps(ctx.get("reliability", {}), indent=2),
        "",
        "## FACT / LIMITATION",
        "- FACT: coberturas y scores calculados solo sobre TEST nunca visto durante tuning.",
        "- LIMITATION: cobertura marginal; la cobertura condicional por regimen puede diferir.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def log_experiment(log_path: Path, record: dict) -> None:
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
