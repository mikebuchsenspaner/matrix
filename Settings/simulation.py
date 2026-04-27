"""
simulation.py
─────────────
Lógica de simulação barra a barra compartilhada entre backtest.py e
walkforward.py.  Importar daqui evita duplicação de código.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from settings import (
    EXIT_GAP,
    FEE_RATE,
    MAX_HOLD_BARS,
    REGIME_FILTER_COLUMN,
    STOP_LOSS,
    TAKE_PROFIT,
    USE_REGIME_FILTER,
)


# ---------------------------------------------------------------------------
# Helpers estatísticos
# ---------------------------------------------------------------------------

def compute_max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def compute_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    std = float(returns.std(ddof=0))
    if std == 0.0:
        return 0.0
    return float((returns.mean() / std) * math.sqrt(periods_per_year))


def compute_profit_factor(trades_df: pd.DataFrame) -> float:
    if trades_df.empty:
        return 0.0
    gross_profit = trades_df.loc[trades_df["net_return"] > 0, "net_return"].sum()
    gross_loss = trades_df.loc[trades_df["net_return"] < 0, "net_return"].sum()
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / abs(gross_loss))


# ---------------------------------------------------------------------------
# Simulação barra a barra
# ---------------------------------------------------------------------------

def simulate_threshold(
    df: pd.DataFrame,
    timestamp_col: str | None,
    close_col: str,
    probabilities: np.ndarray,
    threshold: float,
    fee_rate: float = FEE_RATE,
    stop_loss: float = STOP_LOSS,
    take_profit: float = TAKE_PROFIT,
    max_hold_bars: int = MAX_HOLD_BARS,
    exit_gap: float = EXIT_GAP,
    use_regime_filter: bool = USE_REGIME_FILTER,
    regime_filter_column: str = REGIME_FILTER_COLUMN,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Simula a estratégia barra a barra para um threshold específico.

    Regras de entrada:
      - Sinal do modelo >= threshold
      - Regime filter OK (se use_regime_filter = True)
      - Não está em posição

    Regras de saída (em ordem de prioridade):
      1. Stop loss atingido
      2. Take profit atingido
      3. Probabilidade caiu abaixo de (threshold - exit_gap)
      4. Máximo de barras atingido
      5. Última barra dos dados

    Retorna:
      - dict com métricas agregadas
      - DataFrame detalhado barra a barra
      - DataFrame de trades individuais
    """
    data = df.copy().reset_index(drop=True)
    data["probability"] = pd.Series(probabilities, index=data.index).astype(float)
    data["signal"] = (data["probability"] >= threshold).astype(int)

    # regime filter
    if use_regime_filter:
        if regime_filter_column not in data.columns:
            raise ValueError(
                f"Coluna de regime '{regime_filter_column}' não encontrada. "
                "Defina USE_REGIME_FILTER = False no settings.py ou verifique o nome da coluna."
            )
        data["regime_ok"] = (
            pd.to_numeric(data[regime_filter_column], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    else:
        data["regime_ok"] = 1

    data[close_col] = pd.to_numeric(data[close_col], errors="coerce")
    data = data.dropna(subset=[close_col]).reset_index(drop=True)

    if len(data) < 2:
        raise ValueError("Dados insuficientes para simular (< 2 barras válidas).")

    close = data[close_col].astype(float)
    market_returns = close.pct_change().fillna(0.0)

    strategy_returns = np.zeros(len(data), dtype=float)
    position_end_of_bar = np.zeros(len(data), dtype=int)
    entry_signal_arr = np.zeros(len(data), dtype=int)
    exit_signal_arr = np.zeros(len(data), dtype=int)
    exit_reason_arr = [""] * len(data)

    in_position = False
    entry_index: int | None = None
    entry_price: float | None = None
    entry_probability: float | None = None

    trades: list[dict[str, Any]] = []

    for i in range(len(data)):
        # acumula retorno de mercado enquanto em posição
        if in_position and i > 0:
            strategy_returns[i] += float(market_returns.iloc[i])

        # verifica saída
        if in_position:
            current_price = float(close.iloc[i])
            gross_return = current_price / float(entry_price) - 1.0
            bars_held = i - int(entry_index)

            prob_exit_level = max(0.0, threshold - exit_gap)
            prob_exit = float(data.loc[i, "probability"]) < prob_exit_level

            exit_reason: str | None = None
            if gross_return <= -stop_loss:
                exit_reason = "stop_loss"
            elif gross_return >= take_profit:
                exit_reason = "take_profit"
            elif prob_exit:
                exit_reason = "probability_exit"
            elif bars_held >= max_hold_bars:
                exit_reason = "max_hold"
            elif i == len(data) - 1:
                exit_reason = "end_of_data"

            if exit_reason is not None:
                # taxa na saída
                strategy_returns[i] -= fee_rate
                exit_signal_arr[i] = 1
                exit_reason_arr[i] = exit_reason

                net_return = (
                    (1.0 + gross_return) * (1.0 - fee_rate) * (1.0 - fee_rate)
                ) - 1.0

                trade: dict[str, Any] = {
                    "threshold": threshold,
                    "entry_index": int(entry_index),
                    "exit_index": i,
                    "entry_price": float(entry_price),
                    "exit_price": current_price,
                    "bars_held": bars_held,
                    "entry_probability": float(entry_probability),
                    "exit_probability": float(data.loc[i, "probability"]),
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "exit_reason": exit_reason,
                }

                if timestamp_col and timestamp_col in data.columns:
                    trade["entry_time"] = data.loc[int(entry_index), timestamp_col]
                    trade["exit_time"] = data.loc[i, timestamp_col]

                trades.append(trade)
                in_position = False
                entry_index = entry_price = entry_probability = None

        # verifica entrada
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
            strategy_returns[i] -= fee_rate  # taxa na entrada
            entry_signal_arr[i] = 1

        position_end_of_bar[i] = 1 if in_position else 0

    # monta DataFrames de resultado
    data["market_return"] = market_returns
    data["strategy_return"] = strategy_returns
    data["position"] = position_end_of_bar
    data["entry_signal"] = entry_signal_arr
    data["exit_signal"] = exit_signal_arr
    data["exit_reason"] = exit_reason_arr
    data["strategy_equity"] = (1.0 + data["strategy_return"]).cumprod()
    data["buy_hold_equity"] = (1.0 + data["market_return"]).cumprod()

    trades_df = pd.DataFrame(trades)

    # métricas agregadas
    strategy_total_return = float(data["strategy_equity"].iloc[-1] - 1.0)
    buy_hold_return = float(close.iloc[-1] / close.iloc[0] - 1.0)
    alpha = float(strategy_total_return - buy_hold_return)
    max_drawdown = compute_max_drawdown(data["strategy_equity"])
    sharpe = compute_sharpe(data["strategy_return"])
    exposure = float(data["position"].mean())
    profit_factor = compute_profit_factor(trades_df)

    n = len(trades_df)
    winning = int((trades_df["net_return"] > 0).sum()) if n > 0 else 0
    losing = int((trades_df["net_return"] <= 0).sum()) if n > 0 else 0
    win_rate = float(winning / n) if n > 0 else 0.0
    avg_return = float(trades_df["net_return"].mean()) if n > 0 else 0.0
    median_return = float(trades_df["net_return"].median()) if n > 0 else 0.0

    metrics: dict[str, Any] = {
        "threshold": threshold,
        "strategy_total_return": strategy_total_return,
        "buy_and_hold_return": buy_hold_return,
        "alpha_vs_buy_hold": alpha,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "profit_factor": profit_factor,
        "exposure": exposure,
        "total_trades": n,
        "winning_trades": winning,
        "losing_trades": losing,
        "win_rate": win_rate,
        "average_trade_return": avg_return,
        "median_trade_return": median_return,
        "trade_rate": float(n / len(data)) if len(data) > 0 else 0.0,
    }

    return metrics, data, trades_df


# ---------------------------------------------------------------------------
# Carregamento de modelo
# ---------------------------------------------------------------------------

def load_model_artifact(
    path,
) -> tuple[Any, list[str] | None, dict[str, Any] | None]:
    """
    Carrega o bundle salvo pelo train.py (joblib ou pickle).
    Retorna (model, feature_columns, artifact_dict).
    """
    import pickle
    from pathlib import Path

    try:
        import joblib
        _joblib_available = True
    except ImportError:
        _joblib_available = False

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em '{path}'. Rode primeiro: python train.py"
        )

    artifact = None
    last_error = None

    if _joblib_available:
        try:
            artifact = joblib.load(path)
        except Exception as exc:
            last_error = exc

    if artifact is None:
        try:
            with path.open("rb") as fh:
                artifact = pickle.load(fh)
        except Exception as exc:
            last_error = exc

    if artifact is None:
        raise RuntimeError(
            f"Não foi possível carregar o modelo de '{path}': {last_error}"
        )

    if isinstance(artifact, dict):
        model = (
            artifact.get("model")
            or artifact.get("estimator")
            or artifact.get("classifier")
            or artifact.get("pipeline")
        )
        if model is None:
            raise ValueError(
                "Arquivo carregado mas sem chave 'model'/'estimator'/'classifier'/'pipeline'."
            )
        feature_columns = artifact.get("feature_columns")
        return model, feature_columns, artifact

    return artifact, None, None


# ---------------------------------------------------------------------------
# Predição de probabilidades
# ---------------------------------------------------------------------------

def predict_probabilities(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(x))
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1]
        return probs.ravel()

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x), dtype=float).ravel()
        return 1.0 / (1.0 + np.exp(-scores))

    return np.asarray(model.predict(x), dtype=float).ravel()


# ---------------------------------------------------------------------------
# Construção da matriz de features
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    for col in feature_columns:
        if col in df.columns:
            x[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            x[col] = 0.0
    return x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
