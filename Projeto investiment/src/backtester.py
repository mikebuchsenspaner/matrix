from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import (
    BUY_THRESHOLD,
    EXIT_THRESHOLD,
    FEE_RATE,
    INITIAL_CAPITAL,
    MAX_HOLDING_BARS,
    POSITION_SIZE_FRACTION,
    PRICE_COLUMNS,
    SLIPPAGE_RATE,
    STOP_LOSS,
    TAKE_PROFIT,
    TIMESTAMP_CANDIDATES,
)


@dataclass
class Position:
    entry_time: Any
    entry_index: int
    entry_price: float
    units: float
    entry_cost: float
    entry_fee: float
    entry_probability: float
    signal_time: Any


def detect_timestamp_column(df: pd.DataFrame) -> str | None:
    for col in TIMESTAMP_CANDIDATES:
        if col in df.columns:
            return col
    return None


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    loaded = joblib.load(model_path)

    if isinstance(loaded, dict) and "model" in loaded:
        return loaded

    feature_columns = list(getattr(loaded, "feature_names_in_", []))
    return {
        "model": loaded,
        "feature_columns": feature_columns,
        "label_column": "label",
        "timestamp_column": None,
        "price_columns": PRICE_COLUMNS,
        "test_start_index": None,
    }


def preprocess_market_data(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    label_column: str | None = None,
) -> tuple[pd.DataFrame, str | None]:
    df = df.copy()

    timestamp_col = detect_timestamp_column(df)
    if timestamp_col:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)

    ignore_cols = set()
    if timestamp_col:
        ignore_cols.add(timestamp_col)
    if label_column and label_column in df.columns:
        ignore_cols.add(label_column)

    for col in df.columns:
        if col not in ignore_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if not feature_columns:
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        feature_columns = [col for col in numeric_columns if col != label_column]

    required_cols = list(feature_columns)

    close_col = PRICE_COLUMNS["close"]
    if close_col in df.columns:
        required_cols.append(close_col)

    required_cols = [col for col in required_cols if col in df.columns]
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    return df, timestamp_col


def get_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1]
        return probabilities.ravel()

    predictions = model.predict(X)
    return np.asarray(predictions, dtype=float)


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return float(drawdown.min()) if len(drawdown) else 0.0


def calculate_profit_factor(trades_df: pd.DataFrame) -> float:
    if trades_df.empty:
        return 0.0

    gross_profit = trades_df.loc[trades_df["pnl_value"] > 0, "pnl_value"].sum()
    gross_loss = trades_df.loc[trades_df["pnl_value"] < 0, "pnl_value"].sum()

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return float(gross_profit / abs(gross_loss))


def calculate_sharpe(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0

    returns = equity_curve.pct_change().dropna()
    if returns.std(ddof=0) == 0:
        return 0.0

    return float((returns.mean() / returns.std(ddof=0)) * sqrt(252))


def run_backtest(
    df: pd.DataFrame,
    model_bundle: dict[str, Any],
    initial_capital: float = INITIAL_CAPITAL,
    position_size_fraction: float = POSITION_SIZE_FRACTION,
    buy_threshold: float = BUY_THRESHOLD,
    exit_threshold: float = EXIT_THRESHOLD,
    stop_loss: float = STOP_LOSS,
    take_profit: float = TAKE_PROFIT,
    max_holding_bars: int = MAX_HOLDING_BARS,
    fee_rate: float = FEE_RATE,
    slippage_rate: float = SLIPPAGE_RATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    model = model_bundle["model"]
    label_column = model_bundle.get("label_column", "label")
    feature_columns = model_bundle.get("feature_columns", [])
    test_start_index = model_bundle.get("test_start_index")

    df, timestamp_col = preprocess_market_data(
        df=df,
        feature_columns=feature_columns,
        label_column=label_column,
    )

    if test_start_index is not None and 0 < test_start_index < len(df):
        df = df.iloc[test_start_index:].reset_index(drop=True)

    if not feature_columns:
        feature_columns = list(getattr(model, "feature_names_in_", []))

    if not feature_columns:
        raise ValueError(
            "Não foi possível descobrir as feature columns. "
            "Retreine o modelo com o train.py novo."
        )

    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Features ausentes no CSV: {missing_features}")

    close_col = PRICE_COLUMNS["close"]
    open_col = PRICE_COLUMNS["open"] if PRICE_COLUMNS["open"] in df.columns else close_col
    high_col = PRICE_COLUMNS["high"] if PRICE_COLUMNS["high"] in df.columns else close_col
    low_col = PRICE_COLUMNS["low"] if PRICE_COLUMNS["low"] in df.columns else close_col

    if close_col not in df.columns:
        raise ValueError(f"A coluna de preço '{close_col}' não existe no CSV.")

    X = df[feature_columns]
    probabilities = get_probabilities(model, X)

    cash = initial_capital
    position: Position | None = None
    pending_entry: dict[str, Any] | None = None

    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    for i, row in df.iterrows():
        current_time = row[timestamp_col] if timestamp_col else i
        open_price = float(row[open_col])
        high_price = float(row[high_col])
        low_price = float(row[low_col])
        close_price = float(row[close_col])
        current_probability = float(probabilities[i])

        if pending_entry is not None and position is None:
            entry_fill_price = open_price * (1 + slippage_rate)

            deployable_cash = cash * position_size_fraction
            units = deployable_cash / (entry_fill_price * (1 + fee_rate))

            entry_cost = units * entry_fill_price
            entry_fee = entry_cost * fee_rate

            cash -= (entry_cost + entry_fee)

            position = Position(
                entry_time=current_time,
                entry_index=i,
                entry_price=entry_fill_price,
                units=units,
                entry_cost=entry_cost,
                entry_fee=entry_fee,
                entry_probability=float(pending_entry["probability"]),
                signal_time=pending_entry["signal_time"],
            )
            pending_entry = None

        if position is not None:
            holding_bars = i - position.entry_index + 1

            stop_price = position.entry_price * (1 - stop_loss)
            take_price = position.entry_price * (1 + take_profit)

            exit_reason = None
            raw_exit_price = None

            if stop_loss > 0 and low_price <= stop_price:
                exit_reason = "stop_loss"
                raw_exit_price = stop_price
            elif take_profit > 0 and high_price >= take_price:
                exit_reason = "take_profit"
                raw_exit_price = take_price
            elif current_probability <= exit_threshold:
                exit_reason = "model_exit"
                raw_exit_price = close_price
            elif holding_bars >= max_holding_bars:
                exit_reason = "time_exit"
                raw_exit_price = close_price
            elif i == len(df) - 1:
                exit_reason = "end_of_data"
                raw_exit_price = close_price

            if exit_reason is not None and raw_exit_price is not None:
                exit_fill_price = raw_exit_price * (1 - slippage_rate)
                gross_exit_value = position.units * exit_fill_price
                exit_fee = gross_exit_value * fee_rate
                net_exit_value = gross_exit_value - exit_fee

                cash += net_exit_value

                total_entry_value = position.entry_cost + position.entry_fee
                pnl_value = net_exit_value - total_entry_value
                net_return_pct = pnl_value / total_entry_value if total_entry_value else 0.0

                trades.append(
                    {
                        "signal_time": position.signal_time,
                        "entry_time": position.entry_time,
                        "exit_time": current_time,
                        "entry_probability": position.entry_probability,
                        "exit_probability": current_probability,
                        "entry_price": position.entry_price,
                        "exit_price": exit_fill_price,
                        "bars_held": holding_bars,
                        "gross_entry_value": position.entry_cost,
                        "gross_exit_value": gross_exit_value,
                        "entry_fee": position.entry_fee,
                        "exit_fee": exit_fee,
                        "pnl_value": pnl_value,
                        "net_return_pct": net_return_pct,
                        "exit_reason": exit_reason,
                    }
                )

                position = None

        equity_value = cash
        if position is not None:
            equity_value += position.units * close_price

        equity_rows.append(
            {
                "time": current_time,
                "close": close_price,
                "probability": current_probability,
                "equity": equity_value,
            }
        )

        if position is None and i < len(df) - 1 and current_probability >= buy_threshold:
            pending_entry = {
                "probability": current_probability,
                "signal_time": current_time,
            }

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)

    final_equity = float(equity_df["equity"].iloc[-1]) if not equity_df.empty else initial_capital
    total_return_pct = (final_equity / initial_capital) - 1.0

    number_of_trades = int(len(trades_df))
    win_rate = (
        float((trades_df["pnl_value"] > 0).mean()) if number_of_trades > 0 else 0.0
    )
    avg_trade_return = (
        float(trades_df["net_return_pct"].mean()) if number_of_trades > 0 else 0.0
    )
    max_drawdown = calculate_max_drawdown(equity_df["equity"]) if not equity_df.empty else 0.0
    profit_factor = calculate_profit_factor(trades_df)
    sharpe_ratio = calculate_sharpe(equity_df["equity"]) if not equity_df.empty else 0.0

    summary = {
        "initial_capital": float(initial_capital),
        "final_equity": final_equity,
        "total_return_pct": float(total_return_pct),
        "number_of_trades": float(number_of_trades),
        "win_rate": float(win_rate),
        "avg_trade_return_pct": float(avg_trade_return),
        "max_drawdown_pct": float(max_drawdown),
        "profit_factor": float(profit_factor),
        "sharpe_252_assumption": float(sharpe_ratio),
        "buy_threshold": float(buy_threshold),
        "exit_threshold": float(exit_threshold),
        "stop_loss": float(stop_loss),
        "take_profit": float(take_profit),
        "max_holding_bars": float(max_holding_bars),
        "fee_rate": float(fee_rate),
        "slippage_rate": float(slippage_rate),
    }

    return trades_df, equity_df, summary