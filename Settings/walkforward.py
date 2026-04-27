"""
walkforward.py
──────────────
Validação walk-forward: retreina o modelo em cada fold e avalia no período
seguinte.  Produz métricas médias por threshold e relatório de robustez.

Uso:
    python walkforward.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from features import engineer_features
from settings import (
    DATA_PATH,
    HORIZON,
    LABEL_COLUMN,
    LABEL_STOP_TOLERANCE,
    MIN_RETURN,
    MODEL_PATH,
    PRICE_COLUMNS,
    THRESHOLDS,
    WF_INITIAL_TRAIN_SIZE,
    WF_STEP_SIZE,
    WF_TEST_SIZE,
)
from simulation import (
    build_feature_matrix,
    load_model_artifact,
    predict_probabilities,
    simulate_threshold,
)

OUTPUT_DIR = Path("reports")


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

def build_fold_ranges(total_rows: int) -> list[tuple[int, int, int, int]]:
    """
    Expanding window walk-forward.

    Fração inicial de treino  : WF_INITIAL_TRAIN_SIZE  (ex: 0.60 = 60%)
    Tamanho de cada janela test: WF_TEST_SIZE           (ex: 0.10 = 10%)
    Passo entre folds          : WF_STEP_SIZE           (normalmente = WF_TEST_SIZE)

    Com 1250 barras e os defaults (60/10/10):
      fold 1: treino[0:750]  teste[750:875]
      fold 2: treino[0:875]  teste[875:1000]
      fold 3: treino[0:1000] teste[1000:1125]
      fold 4: treino[0:1125] teste[1125:1250]
    """
    initial_train = int(total_rows * WF_INITIAL_TRAIN_SIZE)
    test_size = int(total_rows * WF_TEST_SIZE)
    step_size = int(total_rows * WF_STEP_SIZE)

    if test_size < 20:
        raise ValueError(
            f"Janela de teste muito pequena ({test_size} barras). "
            "Aumente o dataset ou WF_TEST_SIZE."
        )

    folds: list[tuple[int, int, int, int]] = []
    train_end = initial_train
    while True:
        test_start = train_end
        test_end = test_start + test_size
        if test_end > total_rows:
            break
        folds.append((0, train_end, test_start, test_end))
        train_end += step_size

    return folds


# ---------------------------------------------------------------------------
# Relatório de robustez
# ---------------------------------------------------------------------------

def print_robustness_report(summary_df: pd.DataFrame) -> None:
    """
    Imprime uma análise de robustez focada nas métricas que importam
    para avaliar se a estratégia é consistente ao longo do tempo.
    """
    print("\n" + "=" * 70)
    print("RELATÓRIO DE ROBUSTEZ — WALK-FORWARD")
    print("=" * 70)

    for _, row in summary_df.iterrows():
        t = row["threshold"]
        folds_pos = row.get("folds_positive", "?")
        n_folds = row.get("folds", "?")
        avg_ret = row["avg_total_return"]
        median_ret = row["median_total_return"]
        avg_sharpe = row["avg_sharpe"]
        avg_alpha = row["avg_alpha_vs_buy_hold"]
        avg_dd = row["avg_max_drawdown"]
        avg_trades = row["avg_trades"]

        print(f"\nThreshold {t:.2f}:")
        print(f"  Folds positivos     : {folds_pos}/{n_folds}")
        print(f"  Retorno médio       : {avg_ret:+.4%}")
        print(f"  Retorno mediano     : {median_ret:+.4%}")
        print(f"  Sharpe médio        : {avg_sharpe:+.4f}")
        print(f"  Alpha médio vs B&H  : {avg_alpha:+.4%}")
        print(f"  Drawdown médio      : {avg_dd:.4%}")
        print(f"  Trades médios/fold  : {avg_trades:.1f}")

        # aviso se estatística insuficiente
        if avg_trades < 3:
            print(
                "  ⚠  Menos de 3 trades por fold em média — "
                "resultado estatisticamente frágil."
            )

    # sugere melhor threshold por consistência (folds positivos) e depois por sharpe
    if "folds_positive" in summary_df.columns:
        best = summary_df.sort_values(
            ["folds_positive", "avg_sharpe"], ascending=[False, False]
        ).iloc[0]
        print(
            f"\n→ Threshold mais consistente: {best['threshold']:.2f} "
            f"({best['folds_positive']:.0f}/{best['folds']:.0f} folds positivos, "
            f"Sharpe médio={best['avg_sharpe']:.4f})"
        )

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("WALK-FORWARD VALIDATION")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # carrega modelo template e dados
    model_template, artifact_features, _ = load_model_artifact(MODEL_PATH)
    df_features, timestamp_col, engineered_features = engineer_features(
        pd.read_csv(DATA_PATH),
        horizon=HORIZON,
        min_return=MIN_RETURN,
        label_stop_tolerance=LABEL_STOP_TOLERANCE,
        label_column=LABEL_COLUMN,
        drop_target_na=True,
    )

    if df_features.empty:
        raise ValueError("DataFrame vazio após feature engineering.")

    df_features[LABEL_COLUMN] = (
        pd.to_numeric(df_features[LABEL_COLUMN], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    feature_columns = artifact_features or engineered_features
    close_col = PRICE_COLUMNS["close"]

    print(f"Total de linhas   : {len(df_features)}")
    print(f"Features          : {len(feature_columns)}")
    print(f"Thresholds        : {THRESHOLDS}")
    print(
        f"Config walk-forward: initial_train={WF_INITIAL_TRAIN_SIZE:.0%} | "
        f"test_size={WF_TEST_SIZE:.0%} | step={WF_STEP_SIZE:.0%}"
    )

    fold_ranges = build_fold_ranges(len(df_features))
    if not fold_ranges:
        raise ValueError(
            "Não foi possível montar folds. "
            "O dataset provavelmente é pequeno demais. "
            "Use get_data.py para baixar dados reais (5 anos diários ≈ 1250 barras)."
        )

    print(f"Folds gerados     : {len(fold_ranges)}")

    fold_records: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []

    for fold_num, (train_start, train_end, test_start, test_end) in enumerate(
        fold_ranges, start=1
    ):
        train_df = df_features.iloc[train_start:train_end].reset_index(drop=True)
        test_df = df_features.iloc[test_start:test_end].reset_index(drop=True)

        X_train = build_feature_matrix(train_df, feature_columns)
        y_train = pd.to_numeric(train_df[LABEL_COLUMN], errors="coerce").fillna(0).astype(int)
        X_test = build_feature_matrix(test_df, feature_columns)
        y_test = pd.to_numeric(test_df[LABEL_COLUMN], errors="coerce").fillna(0).astype(int)

        # retreina do zero em cada fold
        try:
            estimator = clone(model_template)
        except Exception:
            import copy
            estimator = copy.deepcopy(model_template)

        estimator.fit(X_train, y_train)
        probabilities = predict_probabilities(estimator, X_test)

        positive_rate = float(y_test.mean()) if len(y_test) > 0 else 0.0

        print("-" * 70)
        print(
            f"Fold {fold_num}: treino[{train_start}:{train_end}] ({len(train_df)} linhas) | "
            f"teste[{test_start}:{test_end}] ({len(test_df)} linhas) | "
            f"label_positivo={positive_rate:.2%}"
        )

        for threshold in THRESHOLDS:
            y_pred = (probabilities >= threshold).astype(int)

            # métricas de classificação
            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))

            # métricas de ranking (independentes do threshold)
            try:
                auc = float(roc_auc_score(y_test, probabilities))
                ap = float(average_precision_score(y_test, probabilities))
            except Exception:
                auc = 0.0
                ap = 0.0

            # simulação financeira
            sim_metrics, _, trades_df = simulate_threshold(
                df=test_df,
                timestamp_col=timestamp_col,
                close_col=close_col,
                probabilities=probabilities,
                threshold=threshold,
            )

            fold_records.append(
                {
                    "fold": fold_num,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "test_rows": len(test_df),
                    "positive_rate": positive_rate,
                    "threshold": threshold,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "roc_auc": auc,
                    "avg_precision_score": ap,
                    "trades": sim_metrics["total_trades"],
                    "trade_rate": sim_metrics["trade_rate"],
                    "win_rate": sim_metrics["win_rate"],
                    "average_trade_return": sim_metrics["average_trade_return"],
                    "median_trade_return": sim_metrics["median_trade_return"],
                    "total_return": sim_metrics["strategy_total_return"],
                    "buy_and_hold_return": sim_metrics["buy_and_hold_return"],
                    "alpha_vs_buy_hold": sim_metrics["alpha_vs_buy_hold"],
                    "max_drawdown": sim_metrics["max_drawdown"],
                    "sharpe": sim_metrics["sharpe"],
                    "profit_factor": sim_metrics["profit_factor"],
                    "exposure": sim_metrics["exposure"],
                    "winning_trades": sim_metrics["winning_trades"],
                    "losing_trades": sim_metrics["losing_trades"],
                }
            )

            # registra predições individuais
            for i in range(len(test_df)):
                rec_row: dict[str, Any] = {
                    "fold": fold_num,
                    "threshold": threshold,
                    "probability": float(probabilities[i]),
                    "y_true": int(y_test.iloc[i]),
                    "y_pred": int(y_pred[i]),
                }
                if timestamp_col and timestamp_col in test_df.columns:
                    rec_row["timestamp"] = test_df.iloc[i][timestamp_col]
                if close_col in test_df.columns:
                    rec_row["close"] = float(test_df.iloc[i][close_col])
                prediction_records.append(rec_row)

            print(
                f"  t={threshold:.2f} | "
                f"trades={sim_metrics['total_trades']:>3} | "
                f"win={sim_metrics['win_rate']:.2%} | "
                f"ret={sim_metrics['strategy_total_return']:+.4%} | "
                f"sharpe={sim_metrics['sharpe']:+.3f} | "
                f"alpha={sim_metrics['alpha_vs_buy_hold']:+.4%} | "
                f"auc={auc:.4f}"
            )

    print("-" * 70)

    fold_df = pd.DataFrame(fold_records)

    # summary por threshold
    summary_df = (
        fold_df.groupby("threshold", as_index=False)
        .agg(
            folds=("fold", "count"),
            folds_positive=("total_return", lambda x: (x > 0).sum()),
            avg_accuracy=("accuracy", "mean"),
            avg_roc_auc=("roc_auc", "mean"),
            avg_avg_precision=("avg_precision_score", "mean"),
            avg_f1=("f1", "mean"),
            avg_trades=("trades", "mean"),
            avg_trade_rate=("trade_rate", "mean"),
            avg_win_rate=("win_rate", "mean"),
            avg_trade_return=("average_trade_return", "mean"),
            median_trade_return=("median_trade_return", "median"),
            avg_total_return=("total_return", "mean"),
            median_total_return=("total_return", "median"),
            sum_total_return=("total_return", "sum"),
            avg_sharpe=("sharpe", "mean"),
            avg_alpha_vs_buy_hold=("alpha_vs_buy_hold", "mean"),
            avg_max_drawdown=("max_drawdown", "mean"),
            avg_profit_factor=("profit_factor", "mean"),
            avg_exposure=("exposure", "mean"),
        )
        # ordena por folds_positivos desc, depois avg_sharpe desc
        .sort_values(["folds_positive", "avg_sharpe"], ascending=[False, False])
        .reset_index(drop=True)
    )

    # salva
    fold_df.to_csv(OUTPUT_DIR / "walkforward_folds.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "walkforward_summary.csv", index=False)
    pd.DataFrame(prediction_records).to_csv(
        OUTPUT_DIR / "walkforward_predictions.csv", index=False
    )

    # imprime summary
    display_cols = [
        "threshold", "folds", "folds_positive",
        "avg_roc_auc", "avg_f1",
        "avg_trades", "avg_win_rate",
        "avg_total_return", "median_total_return",
        "avg_sharpe", "avg_alpha_vs_buy_hold", "avg_max_drawdown",
    ]
    print("\nResumo por threshold:")
    print(summary_df[display_cols].to_string(index=False))

    print_robustness_report(summary_df)

    best = summary_df.iloc[0]
    print(f"\nArquivos salvos em: {OUTPUT_DIR}/")
    print(
        f"  walkforward_folds.csv    — métricas por fold\n"
        f"  walkforward_summary.csv  — médias por threshold\n"
        f"  walkforward_predictions.csv — predições individuais"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
