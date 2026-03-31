import pandas as pd

from config.config import DATA_FILE


def load_data(file_path: str = DATA_FILE) -> pd.DataFrame:
    data = pd.read_csv(file_path)

    required_columns = {"timestamp", "open", "high", "low", "close", "volume"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in CSV: {missing_str}")

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp"]).copy()
    data = data.sort_values("timestamp").reset_index(drop=True)

    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=numeric_columns).copy()
    data = data.set_index("timestamp")

    return data