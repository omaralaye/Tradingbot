"""
ml/models/regime_model.py
--------------------------
Predicts market regime: UPTREND / DOWNTREND / RANGING / CHOPPY.
Complements (and may eventually replace) the rule-based regime detector.
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


class RegimeModel:
    """XGBoost classifier predicting market regime category."""

    REGIMES    = ["uptrend", "downtrend", "ranging", "choppy"]
    LABEL_MAP  = {r: i for i, r in enumerate(REGIMES)}
    LABEL_INV  = {i: r for r, i in LABEL_MAP.items()}

    DEFAULT_PARAMS = {
        "n_estimators":  200,
        "max_depth":     5,
        "learning_rate": 0.08,
        "subsample":     0.8,
        "random_state":  42,
        "eval_metric":   "mlogloss",
    }

    def __init__(self, **xgb_params):
        if XGBClassifier is None:
            raise ImportError("xgboost is required for RegimeModel.")
        params = {**self.DEFAULT_PARAMS, **xgb_params}
        self._model        = XGBClassifier(**params)
        self._feature_names: Optional[list[str]] = None
        self._is_fitted    = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RegimeModel":
        """Train on walk-forward split data.

        Args:
            X: Feature DataFrame.
            y: Label Series with string regime labels (from TripleBarrierLabeler.label_regime).
        """
        y_mapped = y.map(self.LABEL_MAP)
        self._feature_names = list(X.columns)
        self._model.fit(X, y_mapped)
        self._is_fitted = True
        logger.info("RegimeModel fitted on {} samples.", len(X))
        return self

    def predict(self, X: pd.DataFrame) -> tuple[str, float]:
        """Predict the most likely regime and its confidence.

        Returns:
            Tuple of (regime_str, confidence_float).
        """
        if not self._is_fitted:
            raise RuntimeError("RegimeModel not fitted.")
        proba     = self._model.predict_proba(X)[-1]
        class_idx = int(proba.argmax())
        return self.LABEL_INV[class_idx], float(proba[class_idx])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("RegimeModel saved to {}", path)

    @classmethod
    def load(cls, path: str | Path) -> "RegimeModel":
        model = joblib.load(path)
        logger.info("RegimeModel loaded from {}", path)
        return model
