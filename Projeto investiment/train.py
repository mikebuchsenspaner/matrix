from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from features import HORIZON, LABEL_STOP_TOLERANCE, MIN_RETURN, engineer_features


MODEL_PATH = Path("models/trade_model.pkl")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

DATA_CANDIDATES = [
    Path("data/market_data.csv"),
    Path("data/data.csv"),
    Path("market_data.csv"),
    Path("data.csv"),
    Path("historical_data.csv"),
]

TEST_SIZE = 0.20
RANDOM_STATE = 42


def find_data_file() -> Path:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Nenhum CSV encontrado para treino. Coloque seu arquivo em um destes caminhos:\n"
        + "\n".join(str(p) for p in DATA_CANDIDATES)
    )


def load_data(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=600,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=6,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def prepare_dataset(
    df_raw: pd.DataFrame,
    horizon: int = HORIZON,
    min_return: float = MIN_RETURN,
) -> tuple[pd.DataFrame, str | None, list[str]]:
    df_features, timestamp_col, feature_columns = engineer_features(
        df_raw,
        horizon=horizon,
        min_return=min_return,
        drop_target_na=True,
    )

    if df_features.empty:
        raise ValueError("Não há linhas suficientes após a engenharia de features.")

    return df_features, timestamp_col, feature_columns


def split_train_test(df_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(df_features) * (1 - TEST_SIZE))

    if split_index <= 0 or split_index >= len(df_features):
        raise ValueError("Split inválido entre treino e teste.")

    train_df = df_features.iloc[:split_index].copy()
    test_df = df_features.iloc[split_index:].copy()

    return train_df, test_df


def print_feature_importances(
    model: RandomForestClassifier,
    feature_columns: list[str],
    top_n: int = 15,
) -> None:
    if not hasattr(model, "feature_importances_"):
        return

    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print()
    print(f"Top {top_n} feature importances:")
    print(importance_df.head(top_n).set_index("feature")["importance"].to_string())


def main() -> None:
    print("Training started...")

    csv_path = find_data_file()
    df_raw = load_data(csv_path)

    df_features, timestamp_col, feature_columns = prepare_dataset(
        df_raw,
        horizon=HORIZON,
        min_return=MIN_RETURN,
    )

    train_df, test_df = split_train_test(df_features)

    X_train = train_df[feature_columns]
    y_train = train_df["label"].astype(int)

    X_test = test_df[feature_columns]
    y_test = test_df["label"].astype(int)

    print(f"Total usable rows: {len(df_features)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print()
    print("Label settings:")
    print(f"Horizon: {HORIZON}")
    print(f"Target upside (min_return): {MIN_RETURN:.4f}")
    print(f"Max adverse move allowed: {LABEL_STOP_TOLERANCE:.4f}")
    print()
    print("Label distribution (full dataset):")
    print(df_features["label"].value_counts().sort_index())

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "timestamp_col": timestamp_col,
        "horizon": HORIZON,
        "min_return": MIN_RETURN,
        "label_stop_tolerance": LABEL_STOP_TOLERANCE,
        "label_mode": "future_high_low_range",
        "data_path": str(csv_path),
    }

    joblib.dump(bundle, MODEL_PATH)

    print()
    print("Training completed successfully.")
    print(f"Model saved to: {MODEL_PATH}")
    print()
    print(f"Test accuracy: {accuracy:.4f}")
    print()
    print("Confusion matrix:")
    print(cm)
    print()
    print("Classification report:")
    print(report)

    print_feature_importances(model, feature_columns, top_n=15)


if __name__ == "__main__":
    main()