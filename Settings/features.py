from __future__ import annotations

import numpy as np
import pandas as pd

from settings import (
    HORIZON,
    LABEL_COLUMN,
    LABEL_STOP_TOLERANCE,
    MIN_RETURN,
    PRICE_COLUMNS,
    TIMESTAMP_CANDIDATES,
    VOLUME_COLUMN,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_timestamp_column(df: pd.DataFrame) -> str | None:
    for col in TIMESTAMP_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _safe_div(numerator: pd.Series, denominator) -> pd.Series:
    if isinstance(denominator, pd.Series):
        denominator = denominator.replace(0, np.nan)
    elif denominator == 0:
        denominator = np.nan
    return numerator / denominator


# ---------------------------------------------------------------------------
# Indicadores
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume.fillna(0.0)).cumsum()


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------

def create_label(
    df: pd.DataFrame,
    horizon: int = HORIZON,
    min_return: float = MIN_RETURN,
    label_stop_tolerance: float = LABEL_STOP_TOLERANCE,
    label_column: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """
    Label = 1 se o retorno em `horizon` barras for > min_return
    (e a queda mínima no período não exceder label_stop_tolerance, se > 0).
    As últimas `horizon` linhas são dropadas pois o futuro não existe.
    """
    df = df.copy()

    close_col = PRICE_COLUMNS["close"]
    low_col = PRICE_COLUMNS.get("low")

    if close_col not in df.columns:
        raise ValueError(
            f"Coluna '{close_col}' não encontrada. Ajuste PRICE_COLUMNS['close'] no settings.py."
        )

    future_close = df[close_col].shift(-horizon)
    future_return = future_close / df[close_col] - 1.0

    if low_col in df.columns and label_stop_tolerance > 0:
        future_min_low = (
            df[low_col]
            .shift(-1)
            .rolling(window=horizon, min_periods=horizon)
            .min()
            .shift(-(horizon - 1))
        )
        stop_return = future_min_low / df[close_col] - 1.0
        label = (
            (future_return > min_return) & (stop_return > -label_stop_tolerance)
        ).astype(int)
    else:
        label = (future_return > min_return).astype(int)

    df["future_return"] = future_return
    df[label_column] = label

    # Remove as últimas `horizon` linhas (futuro desconhecido)
    df = df.iloc[:-horizon].reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Feature engineering principal
# ---------------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    horizon: int = HORIZON,
    min_return: float = MIN_RETURN,
    label_stop_tolerance: float = LABEL_STOP_TOLERANCE,
    label_column: str = LABEL_COLUMN,
    drop_target_na: bool = False,
) -> tuple[pd.DataFrame, str | None, list[str]]:
    """
    Recebe o CSV bruto e retorna:
      - DataFrame com todas as features e (opcionalmente) o label
      - Nome da coluna de timestamp (ou None)
      - Lista dos nomes das features usadas pelo modelo
    """
    df = df.copy()

    # --- timestamp ---
    timestamp_col = detect_timestamp_column(df)
    if timestamp_col:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = (
            df.dropna(subset=[timestamp_col])
            .sort_values(timestamp_col)
            .reset_index(drop=True)
        )

    # --- converte tudo para numérico ---
    for col in df.columns:
        if col != timestamp_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- colunas de preço ---
    open_col = PRICE_COLUMNS["open"]
    high_col = PRICE_COLUMNS["high"]
    low_col = PRICE_COLUMNS["low"]
    close_col = PRICE_COLUMNS["close"]
    volume_col = VOLUME_COLUMN

    if close_col not in df.columns:
        raise ValueError(
            f"Coluna '{close_col}' não encontrada no CSV. "
            "Ajuste PRICE_COLUMNS['close'] no settings.py."
        )

    close = df[close_col]
    open_price = df[open_col] if open_col in df.columns else close.copy()
    high_price = df[high_col] if high_col in df.columns else close.copy()
    low_price = df[low_col] if low_col in df.columns else close.copy()

    if volume_col in df.columns:
        volume = df[volume_col].fillna(0.0)
    else:
        volume = pd.Series(0.0, index=df.index, dtype=float)
        df[volume_col] = volume

    # -----------------------------------------------------------------------
    # Retornos
    # -----------------------------------------------------------------------
    df["return_1"] = close.pct_change(1)
    df["return_2"] = close.pct_change(2)
    df["return_3"] = close.pct_change(3)
    df["return_5"] = close.pct_change(5)
    df["return_10"] = close.pct_change(10)
    df["return_20"] = close.pct_change(20)
    df["log_return_1"] = np.log(close / close.shift(1))
    df["gap_open"] = _safe_div(open_price - close.shift(1), close.shift(1))
    df["intraday_return"] = _safe_div(close - open_price, open_price)

    # -----------------------------------------------------------------------
    # Médias móveis
    # -----------------------------------------------------------------------
    df["sma_5"] = close.rolling(5, min_periods=5).mean()
    df["sma_10"] = close.rolling(10, min_periods=10).mean()
    df["sma_20"] = close.rolling(20, min_periods=20).mean()
    df["sma_50"] = close.rolling(50, min_periods=50).mean()
    df["sma_100"] = close.rolling(100, min_periods=100).mean()
    df["sma_200"] = close.rolling(200, min_periods=200).mean()

    df["ema_5"] = close.ewm(span=5, adjust=False).mean()
    df["ema_10"] = close.ewm(span=10, adjust=False).mean()
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()

    # preço relativo às médias
    df["price_to_sma_5"] = _safe_div(close, df["sma_5"]) - 1.0
    df["price_to_sma_10"] = _safe_div(close, df["sma_10"]) - 1.0
    df["price_to_sma_20"] = _safe_div(close, df["sma_20"]) - 1.0
    df["price_to_sma_50"] = _safe_div(close, df["sma_50"]) - 1.0
    df["price_to_sma_200"] = _safe_div(close, df["sma_200"]) - 1.0
    df["price_to_ema_10"] = _safe_div(close, df["ema_10"]) - 1.0
    df["price_to_ema_20"] = _safe_div(close, df["ema_20"]) - 1.0
    df["price_to_ema_50"] = _safe_div(close, df["ema_50"]) - 1.0

    # inclinação das médias
    df["sma_20_slope_5"] = _safe_div(df["sma_20"] - df["sma_20"].shift(5), df["sma_20"].shift(5))
    df["sma_50_slope_10"] = _safe_div(df["sma_50"] - df["sma_50"].shift(10), df["sma_50"].shift(10))
    df["ema_20_slope_5"] = _safe_div(df["ema_20"] - df["ema_20"].shift(5), df["ema_20"].shift(5))

    # -----------------------------------------------------------------------
    # Momentum
    # -----------------------------------------------------------------------
    df["momentum_3"] = close.diff(3)
    df["momentum_5"] = close.diff(5)
    df["momentum_10"] = close.diff(10)
    df["momentum_20"] = close.diff(20)

    # -----------------------------------------------------------------------
    # Volatilidade
    # -----------------------------------------------------------------------
    df["volatility_5"] = df["return_1"].rolling(5, min_periods=5).std()
    df["volatility_10"] = df["return_1"].rolling(10, min_periods=10).std()
    df["volatility_20"] = df["return_1"].rolling(20, min_periods=20).std()

    # -----------------------------------------------------------------------
    # Volume
    # -----------------------------------------------------------------------
    df["volume_change_1"] = volume.pct_change(1)
    df["volume_sma_5"] = volume.rolling(5, min_periods=5).mean()
    df["volume_sma_20"] = volume.rolling(20, min_periods=20).mean()
    df["volume_ratio_5"] = _safe_div(volume, df["volume_sma_5"])
    df["volume_ratio_20"] = _safe_div(volume, df["volume_sma_20"])

    # -----------------------------------------------------------------------
    # Indicadores técnicos
    # -----------------------------------------------------------------------
    df["rsi_14"] = compute_rsi(close, period=14)
    df["rsi_14_change_3"] = df["rsi_14"].diff(3)

    df["atr_14"] = compute_atr(high_price, low_price, close, period=14)
    df["atr_pct_14"] = _safe_div(df["atr_14"], close)

    df["obv"] = compute_obv(close, volume)
    df["obv_slope_5"] = df["obv"].diff(5)
    df["obv_zscore_20"] = _safe_div(
        df["obv"] - df["obv"].rolling(20, min_periods=20).mean(),
        df["obv"].rolling(20, min_periods=20).std(),
    )

    # -----------------------------------------------------------------------
    # Candle
    # -----------------------------------------------------------------------
    df["candle_range_pct"] = _safe_div(high_price - low_price, close)
    df["candle_body_pct"] = _safe_div(close - open_price, open_price)

    max_oc = pd.concat([open_price, close], axis=1).max(axis=1)
    min_oc = pd.concat([open_price, close], axis=1).min(axis=1)
    df["upper_wick_pct"] = _safe_div(high_price - max_oc, close)
    df["lower_wick_pct"] = _safe_div(min_oc - low_price, close)

    # -----------------------------------------------------------------------
    # Posição relativa a máximas/mínimas
    # -----------------------------------------------------------------------
    df["price_vs_high_20"] = _safe_div(close, high_price.rolling(20, min_periods=20).max()) - 1.0
    df["price_vs_low_20"] = _safe_div(close, low_price.rolling(20, min_periods=20).min()) - 1.0

    # -----------------------------------------------------------------------
    # Flags binárias e regime filter
    # -----------------------------------------------------------------------
    df["price_above_sma_200"] = (close > df["sma_200"]).astype(int)
    df["sma_50_above_sma_200"] = (df["sma_50"] > df["sma_200"]).astype(int)
    df["ema_20_above_ema_50"] = (df["ema_20"] > df["ema_50"]).astype(int)
    df["rsi_14_below_70"] = (df["rsi_14"] < 70).astype(int)
    df["rsi_14_above_45"] = (df["rsi_14"] > 45).astype(int)
    df["volume_above_sma_20"] = (volume > df["volume_sma_20"]).astype(int)
    df["atr_pct_14_below_05"] = (df["atr_pct_14"] < 0.05).astype(int)

    df["regime_bull_trend"] = (
        (df["price_above_sma_200"] == 1)
        & (df["sma_50_above_sma_200"] == 1)
        & (df["ema_20_above_ema_50"] == 1)
    ).astype(int)

    df["regime_momentum_confirmed"] = (
        (df["rsi_14_below_70"] == 1)
        & (df["rsi_14_above_45"] == 1)
        & (df["momentum_5"] > 0)
    ).astype(int)

    df["regime_volume_confirmed"] = (df["volume_above_sma_20"] == 1).astype(int)

    # regime_entry_filter: AND de todas as condições
    # (usado pelo backtest/walkforward quando USE_REGIME_FILTER = True)
    df["regime_entry_filter"] = (
        (df["regime_bull_trend"] == 1)
        & (df["regime_momentum_confirmed"] == 1)
        & (df["regime_volume_confirmed"] == 1)
        & (df["atr_pct_14_below_05"] == 1)
    ).astype(int)

    # -----------------------------------------------------------------------
    # Label (só quando pedido — treino/backtest, não inferência)
    # -----------------------------------------------------------------------
    if drop_target_na:
        df = create_label(
            df,
            horizon=horizon,
            min_return=min_return,
            label_stop_tolerance=label_stop_tolerance,
            label_column=label_column,
        )

    # -----------------------------------------------------------------------
    # Lista de features que o modelo usa
    # -----------------------------------------------------------------------
    feature_columns = [
        "return_1", "return_2", "return_3", "return_5", "return_10", "return_20",
        "log_return_1", "gap_open", "intraday_return",
        "sma_5", "sma_10", "sma_20", "sma_50", "sma_100", "sma_200",
        "ema_5", "ema_10", "ema_20", "ema_50",
        "price_to_sma_5", "price_to_sma_10", "price_to_sma_20",
        "price_to_sma_50", "price_to_sma_200",
        "price_to_ema_10", "price_to_ema_20", "price_to_ema_50",
        "sma_20_slope_5", "sma_50_slope_10", "ema_20_slope_5",
        "momentum_3", "momentum_5", "momentum_10", "momentum_20",
        "volatility_5", "volatility_10", "volatility_20",
        "volume_change_1", "volume_sma_5", "volume_sma_20",
        "volume_ratio_5", "volume_ratio_20",
        "rsi_14", "rsi_14_change_3",
        "atr_14", "atr_pct_14",
        "obv", "obv_slope_5", "obv_zscore_20",
        "candle_range_pct", "candle_body_pct",
        "upper_wick_pct", "lower_wick_pct",
        "price_vs_high_20", "price_vs_low_20",
        "price_above_sma_200", "sma_50_above_sma_200", "ema_20_above_ema_50",
        "rsi_14_below_70", "rsi_14_above_45",
        "volume_above_sma_20", "atr_pct_14_below_05",
        "regime_bull_trend", "regime_momentum_confirmed",
        "regime_volume_confirmed", "regime_entry_filter",
    ]

    # --- limpeza final ---
    required = feature_columns + [close_col]
    existing = [c for c in required if c in df.columns]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=existing).reset_index(drop=True)

    if drop_target_na and label_column in df.columns:
        df[label_column] = pd.to_numeric(df[label_column], errors="coerce")
        df = df.dropna(subset=[label_column]).reset_index(drop=True)
        df[label_column] = df[label_column].astype(int)

    return df, timestamp_col, feature_columns
