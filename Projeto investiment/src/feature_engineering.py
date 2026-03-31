import numpy as np
import pandas as pd


def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def generate_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    df["return_1"] = df["close"].pct_change(1)
    df["return_5"] = df["close"].pct_change(5)
    df["return_10"] = df["close"].pct_change(10)

    df["sma_5"] = df["close"].rolling(window=5, min_periods=5).mean()
    df["sma_10"] = df["close"].rolling(window=10, min_periods=10).mean()
    df["sma_20"] = df["close"].rolling(window=20, min_periods=20).mean()

    df["price_vs_sma_5"] = df["close"] / df["sma_5"] - 1
    df["price_vs_sma_10"] = df["close"] / df["sma_10"] - 1
    df["price_vs_sma_20"] = df["close"] / df["sma_20"] - 1

    df["volatility_5"] = df["return_1"].rolling(window=5, min_periods=5).std()
    df["volatility_10"] = df["return_1"].rolling(window=10, min_periods=10).std()

    df["momentum_5"] = df["close"] - df["close"].shift(5)
    df["momentum_10"] = df["close"] - df["close"].shift(10)

    df["volume_change_1"] = df["volume"].pct_change(1)
    df["volume_sma_5"] = df["volume"].rolling(window=5, min_periods=5).mean()
    df["volume_ratio_5"] = df["volume"] / df["volume_sma_5"]

    df["rsi_14"] = _calculate_rsi(df["close"], period=14)

    return df