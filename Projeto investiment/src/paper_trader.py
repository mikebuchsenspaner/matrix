import pandas as pd
from config.config import MAX_TRADES_PER_DAY

class PaperTrader:
    def __init__(self, model):
        self.model = model
        self.trade_count = 0
        self.log = []

    def simulate(self, new_data):
        for i in range(len(new_data)):
            if self.trade_count < MAX_TRADES_PER_DAY:
                prob = self.model.predict(new_data.iloc[i:i+1])[0]
                if prob >= 0.7:  # Example threshold
                    self.log.append({'timestamp': new_data.index[i], 'action': 'BUY', 'price': new_data['close'].iloc[i]})
                    self.trade_count += 1

    def get_log(self):
        return pd.DataFrame(self.log)