from __future__ import annotations

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

from features import (
    HORIZON,
    LABEL_STOP_TOLERANCE,
    MIN_RETURN,
    engineer_features,
)
from settings import LABEL_COLUMN, PRICE_COLUMNS

DATA_PATH = Path("data/market_data.csv")
MODEL_PATH = Path("models/trade_model.pkl")
OUTPUT_DIR = Path("backtest_results")

TRAIN_RATIO = 0.80
WINDOW_NAME = "test_only"

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50]

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

    return df_features, timestamp_col, feature_columns


def get_backtest_window(df: pd.DataFrame) -> pd.DataFrame:
    split_index = int(len(df) * TRAIN_RATIO)

    if WINDOW_NAME == "test_only":
        df_window = df.iloc[split_index:].copy()
    elif WINDOW_NAME == "full":
        df_window = df.copy()
    else:
        raise ValueError(
            f"WINDOW_NAME inválido: '{WINDOW_NAME}'. Use 'test_only' ou 'full'."
        )

    df_window = df_window.reset_index(drop=True)

    if len(df_window) < 10:
        raise ValueError(
            "Janela de backtest muito pequena. Verifique o CSV, o split ou as features."
        )

    return df_window


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
        raise ValueError("Dados insuficientes para simular o backtest.")

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

                net_trade_return = ((1.0 + gross_trade_return) * (1.0 - FEE_RATE) * (1.0 - FEE_RATE)) - 1.0

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
    }

    return metrics, data, trades_df


def save_summary_text(
    summary_path: Path,
    best_metrics: dict[str, Any],
) -> None:
    lines = [
        f"CSV usado: {DATA_PATH}",
        f"Janela usada: {WINDOW_NAME}",
        f"Horizon: {HORIZON}",
        f"Min return threshold do label: {MIN_RETURN}",
        f"Thresholds testados: {THRESHOLDS}",
        f"Fee rate: {FEE_RATE}",
        f"Max hold bars: {MAX_HOLD_BARS}",
        f"Stop loss: {STOP_LOSS}",
        f"Take profit: {TAKE_PROFIT}",
        f"Exit gap: {EXIT_GAP}",
        f"Regime filter ativo: {USE_REGIME_FILTER}",
        f"Regime filter: {REGIME_FILTER_COLUMN if USE_REGIME_FILTER else 'None'}",
        "",
        f"Best threshold: {best_metrics['threshold']:.2f}",
        f"Strategy total return: {best_metrics['strategy_total_return']:.4%}",
        f"Buy and hold return: {best_metrics['buy_and_hold_return']:.4%}",
        f"Alpha vs buy and hold: {best_metrics['alpha_vs_buy_hold']:.4%}",
        f"Max drawdown: {best_metrics['max_drawdown']:.4%}",
        f"Sharpe (aprox.): {best_metrics['sharpe']:.4f}",
        f"Exposure: {best_metrics['exposure']:.4%}",
        f"Total trades: {best_metrics['total_trades']}",
        f"Winning trades: {best_metrics['winning_trades']}",
        f"Losing trades: {best_metrics['losing_trades']}",
        f"Win rate: {best_metrics['win_rate']:.4%}",
        f"Average trade return: {best_metrics['average_trade_return']:.4%}",
    ]

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("Starting backtest.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, artifact_feature_columns, _ = load_model_artifact(MODEL_PATH)
    df_features, timestamp_col, engineered_feature_columns = load_and_prepare_data()
    df_window = get_backtest_window(df_features)

    close_col = PRICE_COLUMNS["close"]

    feature_columns = artifact_feature_columns or engineered_feature_columns
    x_window = build_feature_matrix(df_window, feature_columns)
    probabilities = predict_probabilities(model, x_window)

    all_metrics: list[dict[str, Any]] = []
    detailed_results: dict[float, pd.DataFrame] = {}
    trade_results: dict[float, pd.DataFrame] = {}

    for threshold in THRESHOLDS:
        metrics, detailed_df, trades_df = simulate_threshold(
            df=df_window,
            timestamp_col=timestamp_col,
            close_col=close_col,
            probabilities=probabilities,
            threshold=threshold,
        )
        all_metrics.append(metrics)
        detailed_results[threshold] = detailed_df
        trade_results[threshold] = trades_df

    threshold_df = pd.DataFrame(all_metrics)
    threshold_df = threshold_df.sort_values(
        by="strategy_total_return", ascending=False
    ).reset_index(drop=True)

    best_threshold = float(threshold_df.iloc[0]["threshold"])
    best_metrics = all_metrics[THRESHOLDS.index(best_threshold)] if best_threshold in THRESHOLDS else next(
        item for item in all_metrics if float(item["threshold"]) == best_threshold
    )

    best_detailed = detailed_results[best_threshold]
    best_trades = trade_results[best_threshold]

    threshold_summary_path = OUTPUT_DIR / f"threshold_summary_{WINDOW_NAME}.csv"
    detailed_path = OUTPUT_DIR / f"backtest_best_{WINDOW_NAME}.csv"
    trades_path = OUTPUT_DIR / f"trades_best_{WINDOW_NAME}.csv"
    summary_path = OUTPUT_DIR / f"summary_best_{WINDOW_NAME}.txt"

    threshold_df.to_csv(threshold_summary_path, index=False)
    best_detailed.to_csv(detailed_path, index=False)
    best_trades.to_csv(trades_path, index=False)
    save_summary_text(summary_path, best_metrics)

    display_cols = [
        "threshold",
        "strategy_total_return",
        "buy_and_hold_return",
        "alpha_vs_buy_hold",
        "max_drawdown",
        "sharpe",
        "exposure",
        "total_trades",
        "win_rate",
        "average_trade_return",
    ]

    print("\nThreshold comparison:")
    print(threshold_df[display_cols].to_string(index=False))

    print("\nBacktest completed successfully.")
    print(f"CSV usado: {DATA_PATH}")
    print(f"Janela usada: {WINDOW_NAME}")
    print(f"Horizon: {HORIZON}")
    print(f"Min return threshold do label: {MIN_RETURN}")
    print(f"Thresholds testados: {THRESHOLDS}")
    print(f"Fee rate: {FEE_RATE}")
    print(f"Max hold bars: {MAX_HOLD_BARS}")
    print(f"Stop loss: {STOP_LOSS}")
    print(f"Take profit: {TAKE_PROFIT}")
    print(f"Exit gap: {EXIT_GAP}")
    print(f"Regime filter ativo: {USE_REGIME_FILTER}")
    print(f"Regime filter: {REGIME_FILTER_COLUMN if USE_REGIME_FILTER else 'None'}")

    print(f"\nBest threshold: {best_metrics['threshold']:.2f}")
    print(f"Strategy total return: {best_metrics['strategy_total_return']:.4%}")
    print(f"Buy and hold return: {best_metrics['buy_and_hold_return']:.4%}")
    print(f"Alpha vs buy and hold: {best_metrics['alpha_vs_buy_hold']:.4%}")
    print(f"Max drawdown: {best_metrics['max_drawdown']:.4%}")
    print(f"Sharpe (aprox.): {best_metrics['sharpe']:.4f}")
    print(f"Exposure: {best_metrics['exposure']:.4%}")
    print(f"Total trades: {best_metrics['total_trades']}")
    print(f"Winning trades: {best_metrics['winning_trades']}")
    print(f"Losing trades: {best_metrics['losing_trades']}")
    print(f"Win rate: {best_metrics['win_rate']:.4%}")
    print(f"Average trade return: {best_metrics['average_trade_return']:.4%}")

    print(f"\nThreshold summary saved to: {threshold_summary_path}")
    print(f"Detailed best backtest saved to: {detailed_path}")
    print(f"Trades saved to: {trades_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()