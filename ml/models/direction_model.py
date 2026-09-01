"""
ml/models/direction_model.py
-----------------------------
Predicts the probability of an up-move vs down-move over a defined
forward horizon. Uses XGBoost gradient-boosted trees.

Labels: +1 (up), -1 (down), 0 (neutral/timeout from triple-barrier).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    logger.warning("xgboost not installed. DirectionModel will not function.")


class DirectionModel:
    """XGBoost classifier predicting next-move direction probability.

    Outputs:
        'up':      probability of price moving up (hitting TP first)
        'down':    probability of price moving down (hitting SL first)
        'neutral': probability of timeout (neither TP nor SL hit)
    """

    DEFAULT_PARAMS = {
        "n_estimators":  300,
        "max_depth":     6,
        "learning_rate": 0.05,
        "subsample":     0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric":   "mlogloss",
        "random_state":  42,
    }

    # Triple-barrier labels: map to 0/1/2 for XGBoost multiclass
    LABEL_MAP     = {-1: 0, 0: 1, 1: 2}
    LABEL_MAP_INV = {0: "down", 1: "neutral", 2: "up"}

    def __init__(self, **xgb_params):
        if XGBClassifier is None:
            raise ImportError("xgboost is required for DirectionModel.")
        params = {**self.DEFAULT_PARAMS, **xgb_params}
        self._model = XGBClassifier(**params)
        self._feature_names: Optional[list[str]] = None
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DirectionModel":
        """Train the model on walk-forward split data.

        Args:
            X: Feature DataFrame (rows = bars, columns = feature names).
            y: Label Series with values -1, 0, or +1.

        Returns:
            self, for chaining.
        """
        y_mapped = y.map(self.LABEL_MAP)
        self._feature_names = list(X.columns)
        self._model.fit(X, y_mapped)
        self._is_fitted = True
        logger.info("DirectionModel fitted on {} samples, {} features.", len(X), len(X.columns))
        return self

    def predict_proba(self, X: pd.DataFrame) -> dict[str, float]:
        """Return direction probabilities for one or more samples.

        Args:
            X: Feature DataFrame (one row = one prediction point).

        Returns:
            Dict with keys 'up', 'down', 'neutral' and float probability values.
            If X has multiple rows, returns probabilities for the last row.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        proba = self._model.predict_proba(X)[-1]  # last row for live inference
        return {
            "down":    float(proba[0]),
            "neutral": float(proba[1]),
            "up":      float(proba[2]),
        }

    def feature_importance(self) -> pd.Series:
        """Return feature importances sorted descending."""
        if not self._is_fitted or self._feature_names is None:
            return pd.Series(dtype=float)
        return pd.Series(
            self._model.feature_importances_,
            index=self._feature_names,
        ).sort_values(ascending=False)

    def save(self, path: str | Path) -> None:
        """Persist model to disk using joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("DirectionModel saved to {}", path)

    @classmethod
    def load(cls, path: str | Path) -> "DirectionModel":
        """Load a previously saved model from disk."""
        model = joblib.load(path)
        logger.info("DirectionModel loaded from {}", path)
        return model
