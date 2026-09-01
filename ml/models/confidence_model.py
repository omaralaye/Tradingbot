"""
ml/models/confidence_model.py
------------------------------
Predicts the probability that a given trade setup will hit Take Profit
before Stop Loss (i.e. is the setup actually worth taking?).

Trained on historical rule-based signals with known outcomes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from loguru import logger

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


class ConfidenceModel:
    """Binary classifier: will this setup hit TP before SL?

    Label = 1 (win) or 0 (loss), trained on historical signal outcomes.
    Output: win probability 0.0-1.0.
    """

    DEFAULT_PARAMS = {
        "n_estimators":  150,
        "max_depth":     4,
        "learning_rate": 0.10,
        "subsample":     0.8,
        "random_state":  42,
        "eval_metric":   "logloss",
    }

    def __init__(self, **xgb_params):
        if XGBClassifier is None:
            raise ImportError("xgboost is required for ConfidenceModel.")
        params = {**self.DEFAULT_PARAMS, **xgb_params}
        self._model     = XGBClassifier(**params)
        self._is_fitted = False
        self._feature_names: Optional[list[str]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ConfidenceModel":
        """Train on historical signal outcome data.

        Args:
            X: Feature DataFrame.
            y: Binary label Series (1=win, 0=loss).
        """
        self._feature_names = list(X.columns)
        self._model.fit(X, y)
        self._is_fitted = True
        logger.info("ConfidenceModel fitted on {} samples.", len(X))
        return self

    def predict_win_prob(self, X: pd.DataFrame) -> float:
        """Return the probability (0-1) that this setup hits TP before SL."""
        if not self._is_fitted:
            raise RuntimeError("ConfidenceModel not fitted.")
        proba = self._model.predict_proba(X)[-1]
        return float(proba[1])  # index 1 = win class

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("ConfidenceModel saved to {}", path)

    @classmethod
    def load(cls, path: str | Path) -> "ConfidenceModel":
        model = joblib.load(path)
        logger.info("ConfidenceModel loaded from {}", path)
        return model
