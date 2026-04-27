
import yfinance as yf
import pandas as pd

# escolher ativo
ticker = "AAPL"

# baixar dados (daily, últimos 5 anos)
df = yf.download(ticker, period="5y", interval="1d")

# reset index pra virar coluna
df = df.reset_index()

# renomear colunas
df = df.rename(columns={
    "Date": "timestamp",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume"
})

# manter só o necessário
df = df[["timestamp", "open", "high", "low", "close", "volume"]]

# salvar no CSV do projeto
df.to_csv("data/market_data.csv", index=False)

print("Dados baixados com sucesso!")
print(f"Linhas: {len(df)}")