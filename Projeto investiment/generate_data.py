import pandas as pd
import numpy as np

rows = 300
price = 100

data = []

for i in range(rows):
    timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i)

    change = np.random.normal(0.05, 0.5)
    open_price = price
    close_price = price + change

    high = max(open_price, close_price) + abs(np.random.normal(0.2, 0.2))
    low = min(open_price, close_price) - abs(np.random.normal(0.2, 0.2))

    volume = int(np.random.normal(1500, 300))

    data.append([timestamp, open_price, high, low, close_price, volume])

    price = close_price

df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
df.to_csv("data/market_data.csv", index=False)

print("Dataset criado com sucesso!")