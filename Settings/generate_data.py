"""
generate_data.py
────────────────
Gera dados sintéticos usando Geometric Brownian Motion (GBM) com
parâmetros calibrados em ações reais (IBOV/SPY anualizados).

Use este script APENAS para testes rápidos.
Para resultados reais, use: python get_data.py

Uso:
    python generate_data.py
    python generate_data.py --rows 2000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Parâmetros GBM calibrados aproximadamente em SPY diário (20 anos)
ANNUAL_MU = 0.10        # retorno esperado anual (~10% ao ano)
ANNUAL_SIGMA = 0.18     # volatilidade anual (~18% ao ano)
TRADING_DAYS = 252

DAILY_MU = ANNUAL_MU / TRADING_DAYS
DAILY_SIGMA = ANNUAL_SIGMA / np.sqrt(TRADING_DAYS)

DEFAULT_ROWS = 2000     # mínimo recomendado para sma_200 + walk-forward
START_PRICE = 100.0
START_DATE = pd.Timestamp("2020-01-01")

OUTPUT_PATH = Path("data/market_data.csv")


def generate_gbm(
    n_rows: int,
    mu: float = DAILY_MU,
    sigma: float = DAILY_SIGMA,
    s0: float = START_PRICE,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # log-retornos diários
    log_returns = rng.normal(
        loc=mu - 0.5 * sigma**2,
        scale=sigma,
        size=n_rows,
    )
    prices = s0 * np.exp(np.cumsum(log_returns))
    prices = np.insert(prices, 0, s0)[:-1]  # começa em s0

    # gera OHLCV realista a partir do close
    intraday_vol = sigma / np.sqrt(6.5)  # ~6.5 horas de pregão
    opens = prices * np.exp(rng.normal(0, intraday_vol * 0.3, n_rows))
    highs = np.maximum(opens, prices) * (1 + rng.exponential(intraday_vol, n_rows))
    lows = np.minimum(opens, prices) * (1 - rng.exponential(intraday_vol, n_rows))
    lows = np.minimum(lows, opens)  # low nunca acima do open
    highs = np.maximum(highs, prices)  # high nunca abaixo do close

    # volume com clustering de volatilidade (mais volume em dias de alta vol)
    base_volume = 1_000_000
    vol_multiplier = 1 + 3 * np.abs(log_returns) / sigma
    volumes = (base_volume * vol_multiplier * rng.lognormal(0, 0.3, n_rows)).astype(int)

    # timestamps em dias úteis
    timestamps = pd.bdate_range(start=START_DATE, periods=n_rows)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.round(opens, 4),
            "high": np.round(highs, 4),
            "low": np.round(lows, 4),
            "close": np.round(prices, 4),
            "volume": volumes,
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera dados sintéticos GBM")
    parser.add_argument(
        "--rows", type=int, default=DEFAULT_ROWS,
        help=f"Número de linhas (default: {DEFAULT_ROWS})"
    )
    args = parser.parse_args()

    n = args.rows
    if n < 300:
        print(f"⚠  {n} linhas é muito pouco. Usando mínimo de 300.")
        n = 300

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_gbm(n_rows=n)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset sintético gerado com sucesso!")
    print(f"  Linhas  : {len(df)}")
    print(f"  Período : {df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}")
    print(f"  Preço inicial: {df['close'].iloc[0]:.2f}")
    print(f"  Preço final  : {df['close'].iloc[-1]:.2f}")
    print(f"  Salvo em: {OUTPUT_PATH}")
    print()
    print("Para dados reais (recomendado), rode: python get_data.py")


if __name__ == "__main__":
    main()
