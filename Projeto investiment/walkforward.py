from __future__ import annotations

import copy
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from features import (
    HORIZON,
    LABEL_STOP_TOLERANCE,
    MIN_RETURN,
    engineer_features,
)
from settings import LABEL_COLUMN, PRICE_COLUMNS

DATA_PATH = Path("data/market_data.csv")
MODEL_PATH = Path("models/trade_model.pkl")
OUTPUT_DIR = Path("reports")

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50]

INITIAL_TRAIN_SIZE = 500
TEST_SIZE = 120
STEP_SIZE = 120

FEE_RATE = 0.001
MAX_HOLD_BARS = 5
STOP_LOSS = -0.02
TAKE_PROFIT = 0.04
EXIT_GAP = 0.10

USE_REGIME_FILTER = True
REGIME_FILTER_COLUMN = "regime_entry_filter"


def load_model_artifact(path: Path) -> tuple[Any, list[str] | None, dict[str, Any] | None]:
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em '{path}'. Rode primeiro: python train.py"
        )

    artifact = None
    last_error = None

    if joblib is not None:
        try:
            artifact = joblib.load(path)
        except Exception as exc:  # pragma: no cover
            last_error = exc

    if artifact is None:
        try:
            with path.open("rb") as file:
                artifact = pickle.load(file)
        except Exception as exc:  # pragma: no cover
            last_error = exc

    if artifact is None:
        raise RuntimeError(f"Não foi possível carregar o modelo de '{path}': {last_error}")

    if isinstance(artifact, dict):
        model = (
            artifact.get("model")
            or artifact.get("estimator")
            or artifact.get("classifier")
            or artifact.get("pipeline")
        )
        if model is None:
            raise ValueError(
                "O arquivo do modelo foi carregado, mas não contém uma chave como "
                "'model', 'estimator', 'classifier' ou 'pipeline'."
            )
        feature_columns = artifact.get("feature_columns")
        return model, feature_columns, artifact

    return artifact, None, None


def make_fresh_estimator(model_template: Any) -> Any:
    try:
        return clone(model_template)
    except Exception:
        return copy.deepcopy(model_template)


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
        raise ValueError("O DataFrame de features ficou vazio após o processamento.")

    if LABEL_COLUMN not in df_features.columns:
        raise ValueError(
            f"A coluna de label '{LABEL_COLUMN}' não foi encontrada após engineer_features()."
        )

    df_features[LABEL_COLUMN] = pd.to_numeric(
        df_features[LABEL_COLUMN], errors="coerce"
    ).fillna(0).astype(int)

    return df_features, timestamp_col, feature_columns


def build_feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)

    for column in feature_columns:
        if column in df.columns:
            x[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            x[column] = 0.0

    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x


def predict_probabilities(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        probabilities = np.asarray(probabilities)

        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1]

        if probabilities.ndim == 1:
            return probabilities

        return probabilities.ravel()

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x), dtype=float).ravel()
        return 1.0 / (1.0 + np.exp(-scores))

    predictions = np.asarray(model.predict(x), dtype=float).ravel()
    return predictions


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def compute_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    std = float(returns.std(ddof=0))

    if std == 0.0:
        return 0.0

    mean = float(returns.mean())
    return float((mean / std) * math.sqrt(periods_per_year))


def simulate_threshold(
    df: pd.DataFrame,
    timestamp_col: str | None,
    close_col: str,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    data = df.copy().reset_index(drop=True)
    data["probability"] = pd.Series(probabilities, index=data.index).astype(float)
    data["signal"] = (data["probability"] >= threshold).astype(int)

    if USE_REGIME_FILTER:
        if REGIME_FILTER_COLUMN not in data.columns:
            raise ValueError(
                f"A coluna de filtro de regime '{REGIME_FILTER_COLUMN}' não existe no DataFrame."
            )
        data["regime_ok"] = pd.to_numeric(
            data[REGIME_FILTER_COLUMN], errors="coerce"
        ).fillna(0).astype(int)
    else:
        data["regime_ok"] = 1

    data[close_col] = pd.to_numeric(data[close_col], errors="coerce")
    data = data.dropna(subset=[close_col]).reset_index(drop=True)

    if len(data) < 2:
        raise ValueError("Dados insuficientes para simular o walk-forward.")

    close = data[close_col].astype(float)
    market_returns = close.pct_change().fillna(0.0)

    strategy_returns = np.zeros(len(data), dtype=float)
    position_end_of_bar = np.zeros(len(data), dtype=int)
    entry_signal = np.zeros(len(data), dtype=int)
    exit_signal = np.zeros(len(data), dtype=int)
    exit_reason_col = [""] * len(data)

    in_position = False
    entry_index = None
    entry_price = None
    entry_probability = None

    trades: list[dict[str, Any]] = []

    for i in range(len(data)):
        if in_position and i > 0:
            strategy_returns[i] += float(market_returns.iloc[i])

        if in_position:
            current_price = float(close.iloc[i])
            gross_trade_return = current_price / float(entry_price) - 1.0
            bars_held = i - int(entry_index)

            exit_probability_level = max(0.0, threshold - EXIT_GAP)
            probability_exit = float(data.loc[i, "probability"]) < exit_probability_level
            stop_exit = gross_trade_return <= STOP_LOSS
            take_exit = gross_trade_return >= TAKE_PROFIT
            max_hold_exit = bars_held >= MAX_HOLD_BARS
            last_bar_exit = i == len(data) - 1

            exit_reason = None
            if stop_exit:
                exit_reason = "stop_loss"
            elif take_exit:
                exit_reason = "take_profit"
            elif probability_exit:
                exit_reason = "probability_exit"
            elif max_hold_exit:
                exit_reason = "max_hold"
            elif last_bar_exit:
                exit_reason = "end_of_data"

            if exit_reason is not None:
                strategy_returns[i] -= FEE_RATE
                exit_signal[i] = 1
                exit_reason_col[i] = exit_reason

                net_trade_return = (
                    ((1.0 + gross_trade_return) * (1.0 - FEE_RATE) * (1.0 - FEE_RATE)) - 1.0
                )

                trade_row = {
                    "threshold": threshold,
                    "entry_index": int(entry_index),
                    "exit_index": int(i),
                    "entry_price": float(entry_price),
                    "exit_price": current_price,
                    "bars_held": int(bars_held),
                    "entry_probability": float(entry_probability),
                    "exit_probability": float(data.loc[i, "probability"]),
                    "gross_return": float(gross_trade_return),
                    "net_return": float(net_trade_return),
                    "exit_reason": exit_reason,
                }

                if timestamp_col and timestamp_col in data.columns:
                    trade_row["entry_time"] = data.loc[int(entry_index), timestamp_col]
                    trade_row["exit_time"] = data.loc[int(i), timestamp_col]

                trades.append(trade_row)

                in_position = False
                entry_index = None
                entry_price = None
                entry_probability = None

        can_enter = (
            not in_position
            and i < len(data) - 1
            and int(data.loc[i, "signal"]) == 1
            and int(data.loc[i, "regime_ok"]) == 1
        )

        if can_enter:
            in_position = True
            entry_index = i
            entry_price = float(close.iloc[i])
            entry_probability = float(data.loc[i, "probability"])

            strategy_returns[i] -= FEE_RATE
            entry_signal[i] = 1

        position_end_of_bar[i] = 1 if in_position else 0

    data["market_return"] = market_returns
    data["strategy_return"] = strategy_returns
    data["position"] = position_end_of_bar
    data["entry_signal"] = entry_signal
    data["exit_signal"] = exit_signal
    data["exit_reason"] = exit_reason_col
    data["strategy_equity"] = (1.0 + data["strategy_return"]).cumprod()
    data["buy_hold_equity"] = (1.0 + data["market_return"]).cumprod()

    trades_df = pd.DataFrame(trades)

    strategy_total_return = float(data["strategy_equity"].iloc[-1] - 1.0)
    buy_hold_return = float(close.iloc[-1] / close.iloc[0] - 1.0)
    alpha_vs_buy_hold = float(strategy_total_return - buy_hold_return)
    max_drawdown = compute_max_drawdown(data["strategy_equity"])
    sharpe = compute_sharpe(data["strategy_return"])
    exposure = float(data["position"].mean())

    total_trades = int(len(trades_df))
    winning_trades = int((trades_df["net_return"] > 0).sum()) if total_trades > 0 else 0
    losing_trades = int((trades_df["net_return"] <= 0).sum()) if total_trades > 0 else 0
    win_rate = float(winning_trades / total_trades) if total_trades > 0 else 0.0
    average_trade_return = (
        float(trades_df["net_return"].mean()) if total_trades > 0 else 0.0
    )
    median_trade_return = (
        float(trades_df["net_return"].median()) if total_trades > 0 else 0.0
    )
    trade_rate = float(total_trades / len(data)) if len(data) > 0 else 0.0

    metrics = {
        "threshold": threshold,
        "strategy_total_return": strategy_total_return,
        "buy_and_hold_return": buy_hold_return,
        "alpha_vs_buy_hold": alpha_vs_buy_hold,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "exposure": exposure,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "average_trade_return": average_trade_return,
        "median_trade_return": median_trade_return,
        "trade_rate": trade_rate,
    }

    return metrics, data, trades_df


def build_fold_ranges(total_rows: int) -> list[tuple[int, int, int, int]]:
    folds: list[tuple[int, int, int, int]] = []

    train_end = INITIAL_TRAIN_SIZE
    while True:
        test_start = train_end
        test_end = test_start + TEST_SIZE

        if test_end > total_rows:
            break

        folds.append((0, train_end, test_start, test_end))
        train_end += STEP_SIZE

    return folds


def main() -> None:
    print("Starting walk-forward validation.")
    print(f"Data file: {DATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_template, artifact_feature_columns, _ = load_model_artifact(MODEL_PATH)
    df_features, timestamp_col, engineered_feature_columns = load_and_prepare_data()

    feature_columns = artifact_feature_columns or engineered_feature_columns
    close_col = PRICE_COLUMNS["close"]

    print(f"Rows after features/dropna: {len(df_features)}")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Using thresholds: {THRESHOLDS}")

    fold_ranges = build_fold_ranges(len(df_features))
    if not fold_ranges:
        raise ValueError(
            "Não foi possível montar folds. Verifique INITIAL_TRAIN_SIZE, TEST_SIZE e o tamanho da base."
        )

    print(f"\nTotal folds: {len(fold_ranges)}")

    fold_metrics_records: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []

    for fold_number, (train_start, train_end, test_start, test_end) in enumerate(
        fold_ranges, start=1
    ):
        train_df = df_features.iloc[train_start:train_end].copy().reset_index(drop=True)
        test_df = df_features.iloc[test_start:test_end].copy().reset_index(drop=True)

        x_train = build_feature_matrix(train_df, feature_columns)
        y_train = pd.to_numeric(train_df[LABEL_COLUMN], errors="coerce").fillna(0).astype(int)

        x_test = build_feature_matrix(test_df, feature_columns)
        y_test = pd.to_numeric(test_df[LABEL_COLUMN], errors="coerce").fillna(0).astype(int)

        estimator = make_fresh_estimator(model_template)
        estimator.fit(x_train, y_train)

        probabilities = predict_probabilities(estimator, x_test)
        positive_rate = float(y_test.mean()) if len(y_test) > 0 else 0.0

        print("-" * 80)
        print(
            f"Fold {fold_number}: "
            f"train[{train_start}:{train_end}] ({len(train_df)} rows) | "
            f"test[{test_start}:{test_end}] ({len(test_df)} rows) | "
            f"positive_rate={positive_rate:.4f}"
        )

        for threshold in THRESHOLDS:
            y_pred = (probabilities >= threshold).astype(int)

            accuracy = float(accuracy_score(y_test, y_pred))
            precision = float(precision_score(y_test, y_pred, zero_division=0))
            recall = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))

            sim_metrics, detailed_df, trades_df = simulate_threshold(
                df=test_df,
                timestamp_col=timestamp_col,
                close_col=close_col,
                probabilities=probabilities,
                threshold=threshold,
            )

            fold_record = {
                "fold": fold_number,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "test_rows": len(test_df),
                "positive_rate": positive_rate,
                "threshold": threshold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
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
                "exposure": sim_metrics["exposure"],
                "winning_trades": sim_metrics["winning_trades"],
                "losing_trades": sim_metrics["losing_trades"],
                "regime_filter_active": USE_REGIME_FILTER,
                "regime_filter_column": REGIME_FILTER_COLUMN if USE_REGIME_FILTER else "",
            }
            fold_metrics_records.append(fold_record)

            regime_ok_series = (
                pd.to_numeric(test_df[REGIME_FILTER_COLUMN], errors="coerce").fillna(0).astype(int)
                if USE_REGIME_FILTER and REGIME_FILTER_COLUMN in test_df.columns
                else pd.Series(1, index=test_df.index, dtype=int)
            )

            for i in range(len(test_df)):
                record = {
                    "fold": fold_number,
                    "threshold": threshold,
                    "row_index_in_fold": i,
                    "probability": float(probabilities[i]),
                    "y_true": int(y_test.iloc[i]),
                    "y_pred": int(y_pred[i]),
                    "regime_ok": int(regime_ok_series.iloc[i]),
                }

                if timestamp_col and timestamp_col in test_df.columns:
                    record["timestamp"] = test_df.iloc[i][timestamp_col]

                if close_col in test_df.columns:
                    record["close"] = float(test_df.iloc[i][close_col])

                prediction_records.append(record)

            print(
                f"  threshold={threshold:.2f} | "
                f"accuracy={accuracy:.4f} | "
                f"f1={f1:.4f} | "
                f"trades={sim_metrics['total_trades']} | "
                f"win_rate={sim_metrics['win_rate']:.4f} | "
                f"total_return={sim_metrics['strategy_total_return']:.4f} | "
                f"sharpe={sim_metrics['sharpe']:.4f}"
            )

    print("-" * 80)

    fold_metrics_df = pd.DataFrame(fold_metrics_records)
    predictions_df = pd.DataFrame(prediction_records)

    summary_df = (
        fold_metrics_df.groupby("threshold", as_index=False)
        .agg(
            folds=("fold", "count"),
            avg_accuracy=("accuracy", "mean"),
            avg_precision=("precision", "mean"),
            avg_recall=("recall", "mean"),
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
            avg_exposure=("exposure", "mean"),
        )
        .sort_values(by="avg_total_return", ascending=False)
        .reset_index(drop=True)
    )

    best_threshold = float(summary_df.iloc[0]["threshold"])

    folds_path = OUTPUT_DIR / "walkforward_folds.csv"
    summary_path = OUTPUT_DIR / "walkforward_summary.csv"
    predictions_path = OUTPUT_DIR / "walkforward_predictions.csv"

    fold_metrics_df.to_csv(folds_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print("\nWalk-forward finished.")
    print(f"Best threshold by avg_total_return: {best_threshold:.2f}")

    display_cols = [
        "threshold",
        "folds",
        "avg_accuracy",
        "avg_precision",
        "avg_recall",
        "avg_f1",
        "avg_trades",
        "avg_trade_rate",
        "avg_win_rate",
        "avg_trade_return",
        "median_trade_return",
        "avg_total_return",
        "median_total_return",
        "sum_total_return",
        "avg_sharpe",
    ]

    print("\nSummary:")
    print(summary_df[display_cols].to_string(index=False))

    print(f"\nSaved fold metrics to: {folds_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved predictions to: {predictions_path}")


if __name__ == "__main__":
    main()