from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from features import engineer_features

MODEL_PATH = Path("models/trade_model.pkl")

THRESHOLD = 0.30

DATA_CANDIDATES = [
    Path("data/market_data.csv"),
    Path("data/data.csv"),
    Path("market_data.csv"),
    Path("data.csv"),
    Path("historical_data.csv"),
]


def find_data_file(saved_path: str | None = None) -> Path:
    if saved_path:
        saved = Path(saved_path)
        if saved.exists():
            return saved

    for path in DATA_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Nenhum CSV encontrado para inferência. Coloque seu arquivo em um destes caminhos:\n"
        + "\n".join(str(p) for p in DATA_CANDIDATES)
    )


def load_data(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em '{MODEL_PATH}'. Rode primeiro: python train.py"
        )

    print("Starting inference...")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    timestamp_col = bundle.get("timestamp_col")
    horizon = bundle.get("horizon")
    min_return = bundle.get("min_return")
    saved_data_path = bundle.get("data_path")

    csv_path = find_data_file(saved_data_path)
    df_raw = load_data(csv_path)

    df_features, detected_timestamp_col, _ = engineer_features(
        df_raw,
        drop_target_na=False,
    )

    if df_features.empty:
        raise ValueError("Não há linhas suficientes para gerar features e fazer inferência.")

    active_timestamp_col = timestamp_col or detected_timestamp_col

    latest_row = df_features.iloc[[-1]].copy()
    X_latest = latest_row[feature_columns]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_latest)[0]
        prob_sell = float(probabilities[0])
        prob_buy = float(probabilities[1])
        signal = "BUY" if prob_buy >= THRESHOLD else "NO BUY"
        model_prediction = int(prob_buy >= THRESHOLD)
    else:
        raw_prediction = int(model.predict(X_latest)[0])
        prob_sell = 0.0
        prob_buy = 0.0
        signal = "BUY" if raw_prediction == 1 else "NO BUY"
        model_prediction = raw_prediction

    print("Data loaded successfully.")
    print(f"Using CSV: {csv_path}")
    print()

    if active_timestamp_col and active_timestamp_col in latest_row.columns:
        print(f"Latest timestamp: {latest_row.iloc[0][active_timestamp_col]}")

    if "close" in latest_row.columns:
        print(f"Latest close: {latest_row.iloc[0]['close']}")

    print(f"Horizon: {horizon}")
    print(f"Min return threshold: {min_return}")
    print(f"Decision threshold: {THRESHOLD}")
    print()
    print(f"Predicted signal: {signal}")
    print(f"Model class prediction: {model_prediction}")
    print(f"Probability class 0: {prob_sell:.4f}")
    print(f"Probability class 1: {prob_buy:.4f}")


if __name__ == "__main__":
    main()