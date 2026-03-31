import numpy as np
import pandas as pd

from config.config import LABEL_HORIZON, STOP_LOSS_PCT, TAKE_PROFIT_PCT


def label_trades(
    data: pd.DataFrame,
    take_profit_pct: float = TAKE_PROFIT_PCT,
    stop_loss_pct: float = STOP_LOSS_PCT,
    horizon: int = LABEL_HORIZON,
) -> pd.DataFrame:
    df = data.copy()
    labels = []

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    total_rows = len(df)

    for i in range(total_rows):
        if i + 1 >= total_rows:
            labels.append(np.nan)
            continue

        entry_price = closes[i]
        take_profit_price = entry_price * (1 + take_profit_pct)
        stop_loss_price = entry_price * (1 - stop_loss_pct)

        end_idx = min(i + 1 + horizon, total_rows)

        future_highs = highs[i + 1 : end_idx]
        future_lows = lows[i + 1 : end_idx]

        label = np.nan

        for high_value, low_value in zip(future_highs, future_lows):
            tp_hit = high_value >= take_profit_price
            sl_hit = low_value <= stop_loss_price

            if tp_hit and sl_hit:
                label = 0
                break

            if tp_hit:
                label = 1
                break

            if sl_hit:
                label = 0
                break

        labels.append(label)

    df["label"] = labels
    return df