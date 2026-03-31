from config.config import PROBABILITY_THRESHOLD


class DecisionEngine:
    def __init__(self, threshold: float = PROBABILITY_THRESHOLD):
        self.threshold = threshold

    def make_decision(self, probability: float) -> str:
        if probability >= self.threshold:
            return "BUY"
        return "NO BUY"