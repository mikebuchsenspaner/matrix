"""
main.py
───────
Inferência: carrega o modelo treinado, calcula features no dado mais recente
e emite a decisão BUY / NO BUY.

Uso:
    python main.py
    python main.py --threshold 0.45
    python main.py --csv data/meu_ativo.csv --threshold 0.40
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from features import engineer_features
from settings import MODEL_PATH, THRESHOLDS

DATA_CANDIDATES = [
    Path("data/market_data.csv"),
    Path("data/data.csv"),
    Path("market_data.csv"),
    Path("data.csv"),
    Path("historical_data.csv"),
]

# threshold padrão: usa o menor da lista (mais agressivo, mais trades)
DEFAULT_THRESHOLD = min(THRESHOLDS)


def find_data_file(saved_path: str | None = None) -> Path:
    if saved_path:
        p = Path(saved_path)
        if p.exists():
            return p

    for path in DATA_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Nenhum CSV encontrado. Coloque o arquivo em:\n"
        + "\n".join(str(p) for p in DATA_CANDIDATES)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferência do modelo de trading")
    parser.add_argument("--csv", type=str, default=None, help="Caminho para o CSV de dados")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Threshold de decisão (default: {DEFAULT_THRESHOLD})",
    )
    args = parser.parse_args()

    threshold = args.threshold

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em '{MODEL_PATH}'. Rode primeiro: python train.py"
        )

    print("=" * 50)
    print("INFERÊNCIA")
    print("=" * 50)

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    timestamp_col = bundle.get("timestamp_col")
    horizon = bundle.get("horizon")
    min_return = bundle.get("min_return")
    saved_data_path = bundle.get("data_path")

    csv_path = find_data_file(args.csv or saved_data_path)
    df_raw = pd.read_csv(csv_path)
    print(f"CSV carregado: {csv_path} ({len(df_raw)} linhas)")

    df_features, detected_ts, _ = engineer_features(df_raw, drop_target_na=False)

    if df_features.empty:
        raise ValueError("Nenhuma linha válida após feature engineering.")

    active_ts = timestamp_col or detected_ts
    latest = df_features.iloc[[-1]].copy()
    X_latest = latest[feature_columns]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_latest)[0]
        prob_no_buy = float(probs[0])
        prob_buy = float(probs[1])
    else:
        raw = int(model.predict(X_latest)[0])
        prob_no_buy = 0.0
        prob_buy = float(raw)

    signal = "BUY" if prob_buy >= threshold else "NO BUY"

    print()
    if active_ts and active_ts in latest.columns:
        print(f"Timestamp mais recente : {latest.iloc[0][active_ts]}")
    if "close" in latest.columns:
        print(f"Fechamento mais recente: {latest.iloc[0]['close']:.4f}")

    print(f"\nHorizon              : {horizon}")
    print(f"Min return do label  : {min_return}")
    print(f"Threshold de decisão : {threshold}")
    print()
    print(f"Probabilidade BUY    : {prob_buy:.4f}")
    print(f"Probabilidade NO BUY : {prob_no_buy:.4f}")
    print()
    print(f"┌─────────────────────┐")
    print(f"│  DECISÃO: {signal:<10} │")
    print(f"└─────────────────────┘")
    print()

    # contexto adicional
    if signal == "BUY":
        margin = prob_buy - threshold
        print(f"Margem acima do threshold: +{margin:.4f}")
    else:
        gap = threshold - prob_buy
        print(f"Faltam {gap:.4f} pontos de probabilidade para BUY.")

    print("=" * 50)


if __name__ == "__main__":
    main()
