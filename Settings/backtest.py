"""
backtest.py
───────────
Roda o backtest na janela de teste (ou dataset completo) e salva os resultados.

Uso:
    python backtest.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from features import engineer_features
from settings import (
    BACKTEST_WINDOW,
    DATA_PATH,
    EXIT_GAP,
    FEE_RATE,
    HORIZON,
    LABEL_COLUMN,
    LABEL_STOP_TOLERANCE,
    MAX_HOLD_BARS,
    MIN_RETURN,
    MODEL_PATH,
    PRICE_COLUMNS,
    REGIME_FILTER_COLUMN,
    STOP_LOSS,
    TAKE_PROFIT,
    THRESHOLDS,
    TRAIN_RATIO,
    USE_REGIME_FILTER,
)
from simulation import (
    build_feature_matrix,
    load_model_artifact,
    predict_probabilities,
    simulate_threshold,
)

OUTPUT_DIR = Path("backtest_results")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_prepare_data() -> tuple[pd.DataFrame, str | None, list[str]]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV não encontrado em '{DATA_PATH}'.")

    df_raw = pd.read_csv(DATA_PATH)
    df_features, timestamp_col, feature_columns = engineer_features(
        df_raw,
        horizon=HORIZON,
        min_return=MIN_RETURN,
        label_stop_tolerance=LABEL_STOP_TOLERANCE,
        label_column=LABEL_COLUMN,
        drop_target_na=True,
    )

    if df_features.empty:
        raise ValueError("DataFrame vazio após feature engineering.")

    return df_features, timestamp_col, feature_columns


def get_backtest_window(df: pd.DataFrame) -> pd.DataFrame:
    if BACKTEST_WINDOW == "test_only":
        split = int(len(df) * TRAIN_RATIO)
        df_window = df.iloc[split:].copy()
    elif BACKTEST_WINDOW == "full":
        df_window = df.copy()
    else:
        raise ValueError(f"BACKTEST_WINDOW inválido: '{BACKTEST_WINDOW}'. Use 'test_only' ou 'full'.")

    df_window = df_window.reset_index(drop=True)

    if len(df_window) < 10:
        raise ValueError(
            f"Janela de backtest muito pequena ({len(df_window)} linhas). "
            "Verifique o CSV ou mude BACKTEST_WINDOW."
        )
    return df_window


def save_summary_text(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "BACKTEST SUMMARY",
        "=" * 40,
        f"CSV         : {DATA_PATH}",
        f"Janela      : {BACKTEST_WINDOW}",
        f"Horizon     : {HORIZON}",
        f"Min return  : {MIN_RETURN}",
        f"Thresholds  : {THRESHOLDS}",
        f"Fee rate    : {FEE_RATE}",
        f"Stop loss   : {STOP_LOSS}",
        f"Take profit : {TAKE_PROFIT}",
        f"Max hold    : {MAX_HOLD_BARS}",
        f"Exit gap    : {EXIT_GAP}",
        f"Regime filter: {'ATIVO — ' + REGIME_FILTER_COLUMN if USE_REGIME_FILTER else 'DESLIGADO'}",
        "",
        f"Best threshold        : {metrics['threshold']:.2f}",
        f"Strategy total return : {metrics['strategy_total_return']:.4%}",
        f"Buy and hold return   : {metrics['buy_and_hold_return']:.4%}",
        f"Alpha vs B&H          : {metrics['alpha_vs_buy_hold']:.4%}",
        f"Max drawdown          : {metrics['max_drawdown']:.4%}",
        f"Sharpe (aprox.)       : {metrics['sharpe']:.4f}",
        f"Profit factor         : {metrics['profit_factor']:.4f}",
        f"Exposure              : {metrics['exposure']:.4%}",
        f"Total trades          : {metrics['total_trades']}",
        f"Winning trades        : {metrics['winning_trades']}",
        f"Losing trades         : {metrics['losing_trades']}",
        f"Win rate              : {metrics['win_rate']:.4%}",
        f"Avg trade return      : {metrics['average_trade_return']:.4%}",
        f"Median trade return   : {metrics['median_trade_return']:.4%}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("BACKTEST")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # carrega modelo e dados
    model, artifact_features, _ = load_model_artifact(MODEL_PATH)
    df_features, timestamp_col, engineered_features = load_and_prepare_data()
    df_window = get_backtest_window(df_features)

    print(f"Barras na janela de backtest: {len(df_window)}")
    print(f"Regime filter: {'ATIVO' if USE_REGIME_FILTER else 'DESLIGADO'}")

    if USE_REGIME_FILTER and REGIME_FILTER_COLUMN in df_window.columns:
        regime_pct = df_window[REGIME_FILTER_COLUMN].mean()
        print(f"  → {regime_pct:.1%} das barras passam no filtro de regime")

    feature_columns = artifact_features or engineered_features
    close_col = PRICE_COLUMNS["close"]

    X_window = build_feature_matrix(df_window, feature_columns)
    probabilities = predict_probabilities(model, X_window)

    print(f"\nDistribuição de probabilidades na janela:")
    proba_s = pd.Series(probabilities)
    print(f"  min={proba_s.min():.4f} | "
          f"p25={proba_s.quantile(0.25):.4f} | "
          f"median={proba_s.median():.4f} | "
          f"p75={proba_s.quantile(0.75):.4f} | "
          f"max={proba_s.max():.4f}")

    # simula todos os thresholds
    all_metrics: list[dict[str, Any]] = []
    detailed_by_threshold: dict[float, pd.DataFrame] = {}
    trades_by_threshold: dict[float, pd.DataFrame] = {}

    print()
    for threshold in THRESHOLDS:
        metrics, detailed_df, trades_df = simulate_threshold(
            df=df_window,
            timestamp_col=timestamp_col,
            close_col=close_col,
            probabilities=probabilities,
            threshold=threshold,
        )
        all_metrics.append(metrics)
        detailed_by_threshold[threshold] = detailed_df
        trades_by_threshold[threshold] = trades_df

        print(
            f"threshold={threshold:.2f} | "
            f"trades={metrics['total_trades']:>3} | "
            f"win_rate={metrics['win_rate']:.2%} | "
            f"return={metrics['strategy_total_return']:+.4%} | "
            f"sharpe={metrics['sharpe']:+.3f} | "
            f"alpha={metrics['alpha_vs_buy_hold']:+.4%}"
        )

    # escolhe melhor threshold por retorno total
    threshold_df = (
        pd.DataFrame(all_metrics)
        .sort_values("strategy_total_return", ascending=False)
        .reset_index(drop=True)
    )

    best_threshold = float(threshold_df.iloc[0]["threshold"])
    best_metrics = next(m for m in all_metrics if m["threshold"] == best_threshold)
    best_detailed = detailed_by_threshold[best_threshold]
    best_trades = trades_by_threshold[best_threshold]

    # salva resultados
    threshold_df.to_csv(OUTPUT_DIR / "threshold_summary.csv", index=False)
    best_detailed.to_csv(OUTPUT_DIR / "backtest_best.csv", index=False)
    best_trades.to_csv(OUTPUT_DIR / "trades_best.csv", index=False)
    save_summary_text(OUTPUT_DIR / "summary_best.txt", best_metrics)

    # imprime comparação de thresholds
    display_cols = [
        "threshold", "total_trades", "win_rate", "strategy_total_return",
        "buy_and_hold_return", "alpha_vs_buy_hold", "max_drawdown",
        "sharpe", "profit_factor", "exposure",
    ]
    print("\nComparação de thresholds:")
    print(threshold_df[display_cols].to_string(index=False))

    # interpreta o resultado
    print("\n" + "=" * 60)
    print("INTERPRETAÇÃO")
    print("=" * 60)
    bh = best_metrics["buy_and_hold_return"]
    sr = best_metrics["strategy_total_return"]
    alpha = best_metrics["alpha_vs_buy_hold"]
    trades_n = best_metrics["total_trades"]
    drawdown = best_metrics["max_drawdown"]
    sharpe = best_metrics["sharpe"]

    print(f"Melhor threshold : {best_threshold:.2f}")
    print(f"Retorno estratégia: {sr:+.4%}")
    print(f"Buy and hold      : {bh:+.4%}")
    print(f"Alpha             : {alpha:+.4%}  {'✓ positivo' if alpha > 0 else '✗ negativo'}")
    print(f"Trades            : {trades_n}")
    print(f"Max drawdown      : {drawdown:.4%}")
    print(f"Sharpe            : {sharpe:.4f}")
    print()

    if trades_n < 10:
        print(
            "⚠  Poucos trades para conclusão estatística. "
            "Considere:\n"
            "  • USE_REGIME_FILTER = False no settings.py\n"
            "  • Aumentar o período de dados (use get_data.py)\n"
            "  • Reduzir MIN_RETURN ou HORIZON"
        )
    if alpha > 0 and trades_n >= 10:
        print("✓ Estratégia superou buy and hold com trades suficientes.")
    elif alpha <= 0:
        print("✗ Estratégia não superou buy and hold. Veja walk-forward para robustez.")

    if abs(drawdown) > 0.15:
        print(f"⚠  Drawdown alto ({drawdown:.2%}). Considere reduzir STOP_LOSS.")

    print(f"\nResultados salvos em: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
