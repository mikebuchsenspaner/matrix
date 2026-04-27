from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "market_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "trade_model.pkl"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Colunas
# ---------------------------------------------------------------------------
LABEL_COLUMN = "label"

TIMESTAMP_CANDIDATES = ["timestamp", "datetime", "date", "time"]

PRICE_COLUMNS = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
}

VOLUME_COLUMN = "volume"

# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20          # fração reservada para teste no train.py
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------
HORIZON = 5               # quantas barras à frente o label olha
MIN_RETURN = 0.003        # retorno mínimo para label = 1
LABEL_STOP_TOLERANCE = 0.0  # tolerância de queda máxima (0 = desligado)

# ---------------------------------------------------------------------------
# Backtest / simulação
# ---------------------------------------------------------------------------
FEE_RATE = 0.001          # taxa por entrada e saída (0.1%)
SLIPPAGE_RATE = 0.0005    # slippage por operação (0.05%)
STOP_LOSS = 0.02          # stop loss como fração do preço de entrada
TAKE_PROFIT = 0.04        # take profit como fração do preço de entrada
MAX_HOLD_BARS = 5         # máximo de barras em posição
EXIT_GAP = 0.10           # probabilidade cai (threshold - EXIT_GAP) → sai

INITIAL_CAPITAL = 10_000.0
POSITION_SIZE_FRACTION = 1.0  # fração do capital alocada por trade

# Thresholds testados no backtest e walk-forward
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50]

# Janela usada no backtest standalone
# "test_only" → só a fatia de teste  |  "full" → dataset inteiro
BACKTEST_WINDOW = "test_only"
TRAIN_RATIO = 1.0 - TEST_SIZE  # deve ser consistente com TEST_SIZE

# ---------------------------------------------------------------------------
# Regime filter
# ---------------------------------------------------------------------------
# True  → entradas só quando regime_entry_filter == 1
# False → entradas sem restrição de regime (mais trades, mais risco)
USE_REGIME_FILTER = False
REGIME_FILTER_COLUMN = "regime_entry_filter"

# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
# Tamanhos em número de barras (linhas do dataset após feature engineering)
WF_INITIAL_TRAIN_SIZE = 0.60   # fração do dataset para o primeiro treino
WF_TEST_SIZE = 0.10            # fração do dataset por janela de teste
WF_STEP_SIZE = 0.10            # quanto avança a cada fold (= WF_TEST_SIZE para expanding)
