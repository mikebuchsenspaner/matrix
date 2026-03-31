from __future__ import annotations

import os

import joblib
import pandas as pd
from xgboost import XGBClassifier

from config.config import MODEL_FILE, RANDOM_STATE
from src.decision_engine import DecisionEngine


class TradeModel:
    def __init__(self) -> None:
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=1.5,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        )
        self.feature_columns: list[str] | None = None
        self.decision_engine = DecisionEngine()

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        if X.empty:
            raise ValueError("Training features are empty.")

        if y.empty:
            raise ValueError("Training labels are empty.")

        self.feature_columns = list(X.columns)
        self.model.fit(X, y)

    def predict_probability(self, X: pd.DataFrame) -> float:
        if self.feature_columns is None:
            raise ValueError("Model feature columns are not loaded.")

        X_aligned = X[self.feature_columns]
        probability = float(self.model.predict_proba(X_aligned)[:, 1][0])
        return probability

    def predict(self, X: pd.DataFrame) -> tuple[float, str]:
        probability = self.predict_probability(X)
        decision = self.decision_engine.make_decision(probability)
        return probability, decision

    def save_model(self, file_path: str = MODEL_FILE) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        payload = {
            "model": self.model,
            "feature_columns": self.feature_columns,
        }
        joblib.dump(payload, file_path)

    def load_model(self, file_path: str = MODEL_FILE) -> None:
        payload = joblib.load(file_path)

        self.model = payload["model"]
        self.feature_columns = payload["feature_columns"]