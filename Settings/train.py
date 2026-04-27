"""
train.py
────────
Treina o modelo, avalia no conjunto de teste e salva o bundle em models/.

Uso:
    python train.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from features import engineer_features
from settings import (
    HORIZON,
    LABEL_COLUMN,
    LABEL_STOP_TOLERANCE,
    MIN_RETURN,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)

# ---------------------------------------------------------------------------
# Localização dos dados
# ---------------------------------------------------------------------------
DATA_CANDIDATES = [
    Path("data/market_data.csv"),
    Path("data/data.csv"),
    Path("market_data.csv"),
    Path("data.csv"),
    Path("historical_data.csv"),
]


def find_data_file() -> Path:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Nenhum CSV encontrado. Caminhos testados:\n"
        + "\n".join(str(p) for p in DATA_CANDIDATES)
    )


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def build_model(calibrate: bool = True) -> RandomForestClassifier | CalibratedClassifierCV:
    """
    RandomForest com calibração de probabilidade via isotonic regression.

    Por que calibrar?
    Random Forests tendem a ter probabilidades mal calibradas — os valores
    ficam "espremidos" no centro (longe de 0 e 1). Calibrar torna o
    threshold mais interpretável: prob=0.60 realmente significa ~60% de
    chance de acertar.
    """
    base = RandomForestClassifier(
        n_estimators=600,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=6,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    if calibrate:
        # cv=5 usa 5-fold interno para calibrar; não vaza dados de teste
        return CalibratedClassifierCV(base, method="isotonic", cv=5)
    return base


# ---------------------------------------------------------------------------
# Divisão treino/teste — sem embaralhar (série temporal)
# ---------------------------------------------------------------------------

def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(df) * (1 - TEST_SIZE))
    if split_index <= 0 or split_index >= len(df):
        raise ValueError(
            f"Split inválido: {split_index} com TEST_SIZE={TEST_SIZE} e {len(df)} linhas."
        )
    return df.iloc[:split_index].copy(), df.iloc[split_index:].copy()


# ---------------------------------------------------------------------------
# Relatório de features
# ---------------------------------------------------------------------------

def print_feature_importances(
    model,
    feature_columns: list[str],
    top_n: int = 15,
) -> None:
    """
    Tenta extrair importâncias do modelo (funciona com RF puro;
    com CalibratedClassifierCV acessa o estimador base).
    """
    estimator = model
    if hasattr(model, "estimator"):          # CalibratedClassifierCV
        estimator = model.estimator
    if hasattr(model, "calibrated_classifiers_"):   # após fit
        try:
            estimator = model.calibrated_classifiers_[0].estimator
        except Exception:
            pass

    if not hasattr(estimator, "feature_importances_"):
        print("Importâncias de features não disponíveis para este modelo.")
        return

    importance_df = (
        pd.DataFrame(
            {"feature": feature_columns, "importance": estimator.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\nTop {top_n} features por importância:")
    print(importance_df.head(top_n).set_index("feature")["importance"].to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("TREINAMENTO")
    print("=" * 60)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    csv_path = find_data_file()
    print(f"CSV encontrado: {csv_path}")

    df_raw = pd.read_csv(csv_path)
    print(f"Linhas brutas: {len(df_raw)}")

    df_features, timestamp_col, feature_columns = engineer_features(
        df_raw,
        horizon=HORIZON,
        min_return=MIN_RETURN,
        label_stop_tolerance=LABEL_STOP_TOLERANCE,
        drop_target_na=True,
    )

    if df_features.empty:
        raise ValueError(
            "DataFrame vazio após feature engineering. "
            "Verifique o CSV (precisa de pelo menos ~300 linhas para sma_200)."
        )

    print(f"\nLinhas após feature engineering: {len(df_features)}")

    # --- distribuição do label ---
    label_counts = df_features[LABEL_COLUMN].value_counts().sort_index()
    total = len(df_features)
    print("\nDistribuição do label:")
    for lbl, count in label_counts.items():
        print(f"  Classe {lbl}: {count:>6} ({count/total:.1%})")

    if len(label_counts) < 2:
        raise ValueError(
            "Apenas uma classe no label. Verifique MIN_RETURN e HORIZON no settings.py."
        )

    imbalance_ratio = label_counts.min() / label_counts.max()
    if imbalance_ratio < 0.2:
        print(
            f"\n⚠  Dataset desbalanceado (ratio={imbalance_ratio:.2f}). "
            "O modelo usa class_weight='balanced_subsample' para compensar."
        )

    # --- split ---
    train_df, test_df = split_train_test(df_features)
    print(f"\nTreino: {len(train_df)} linhas | Teste: {len(test_df)} linhas")

    X_train = train_df[feature_columns]
    y_train = train_df[LABEL_COLUMN].astype(int)
    X_test = test_df[feature_columns]
    y_test = test_df[LABEL_COLUMN].astype(int)

    print("\nLabel settings:")
    print(f"  Horizon       : {HORIZON}")
    print(f"  Min return    : {MIN_RETURN:.4f}")
    print(f"  Stop tolerance: {LABEL_STOP_TOLERANCE:.4f}")

    # --- treino ---
    print("\nTreinando modelo...")
    model = build_model(calibrate=True)
    model.fit(X_train, y_train)
    print("Treino concluído.")

    # --- avaliação ---
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print(f"\nAcurácia          : {accuracy:.4f}")
    print(f"ROC-AUC           : {roc_auc:.4f}")
    print(f"Average Precision : {avg_precision:.4f}")
    print("\nMatriz de confusão:")
    print(cm)
    print("\nRelatório de classificação:")
    print(report)

    # --- distribuição de probabilidades ---
    proba_series = pd.Series(y_proba)
    print("Distribuição de probabilidades no teste:")
    print(f"  min={proba_series.min():.4f} | "
          f"p25={proba_series.quantile(0.25):.4f} | "
          f"median={proba_series.median():.4f} | "
          f"p75={proba_series.quantile(0.75):.4f} | "
          f"max={proba_series.max():.4f}")

    # --- salva bundle ---
    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "timestamp_col": timestamp_col,
        "horizon": HORIZON,
        "min_return": MIN_RETURN,
        "label_stop_tolerance": LABEL_STOP_TOLERANCE,
        "label_column": LABEL_COLUMN,
        "data_path": str(csv_path),
        "test_start_index": len(train_df),
    }
    joblib.dump(bundle, MODEL_PATH)

    print(f"\nModelo salvo em: {MODEL_PATH}")

    print_feature_importances(model, feature_columns, top_n=15)

    print("\n" + "=" * 60)
    print("TREINAMENTO CONCLUÍDO")
    print("=" * 60)


if __name__ == "__main__":
    main()
