from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "market_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "trade_model.pkl"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

LABEL_COLUMN = "label"

TIMESTAMP_CANDIDATES = ["timestamp", "datetime", "date", "time"]
PRICE_COLUMNS = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
}

TEST_SIZE = 0.20
RANDOM_STATE = 42

INITIAL_CAPITAL = 10_000.0
POSITION_SIZE_FRACTION = 1.0

BUY_THRESHOLD = 0.60
EXIT_THRESHOLD = 0.45

STOP_LOSS = 0.02
TAKE_PROFIT = 0.04
MAX_HOLDING_BARS = 5

FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005