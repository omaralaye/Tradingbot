"""
ml/inference.py
---------------
Loads the latest approved ML models from ml/registry/ and runs
inference for live signal generation.

Gracefully degrades: if no trained model is found, the system continues
with rule-based signals alone and logs a warning. This ensures the bot
can run from day one before any ML models have been trained.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


class ModelInference:
    """Loads and runs the three ML models for live inference.

    Models are versioned in ml/registry/ — this class always loads
    the most recently saved version of each model by timestamp.

    Graceful degradation: if a model file is missing, that model's
    predictions are skipped and the system falls back to rule-based logic.
    """

    def __init__(self, registry_dir: str = "ml/registry"):
        self._registry = Path(registry_dir)
        self._direction_model  = None
        self._regime_model     = None
        self._confidence_model = None
        self._load_all_models()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def are_models_loaded(self) -> bool:
        """Return True if at least one model is loaded and ready."""
        return any([
            self._direction_model is not None,
            self._regime_model    is not None,
            self._confidence_model is not None,
        ])

    def predict_direction(self, features: pd.Series) -> dict[str, float]:
        """Run the direction model and return up/down/neutral probabilities.

        Returns:
            Dict {'up': float, 'down': float, 'neutral': float} or empty dict.
        """
        if self._direction_model is None:
            return {}
        try:
            X = features.to_frame().T
            return self._direction_model.predict_proba(X)
        except Exception as exc:
            logger.warning("Direction model inference error: {}", exc)
            return {}

    def predict_regime(self, features: pd.Series) -> tuple[str, float]:
        """Run the regime model.

        Returns:
            Tuple of (regime_string, confidence) or ('unknown', 0.0).
        """
        if self._regime_model is None:
            return "unknown", 0.0
        try:
            X = features.to_frame().T
            return self._regime_model.predict(X)
        except Exception as exc:
            logger.warning("Regime model inference error: {}", exc)
            return "unknown", 0.0

    def predict_confidence(self, features: pd.Series) -> float:
        """Run the confidence model.

        Returns:
            Win probability 0.0-1.0, or 0.5 (neutral) if model unavailable.
        """
        if self._confidence_model is None:
            return 0.5  # neutral default
        try:
            X = features.to_frame().T
            return self._confidence_model.predict_win_prob(X)
        except Exception as exc:
            logger.warning("Confidence model inference error: {}", exc)
            return 0.5

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_latest(self, model_name: str):
        """Load the most recently saved version of a named model.

        Scans the registry for joblib files matching <model_name>_*.joblib,
        sorted by filename (which includes a timestamp), and loads the last one.
        """
        pattern = f"{model_name}_*.joblib"
        candidates = sorted(self._registry.glob(pattern))
        if not candidates:
            logger.warning(
                "No trained model found for '{}' in {}. "
                "Run ml/train.py to generate one. "
                "Bot will use rule-based signals only.",
                model_name, self._registry,
            )
            return None

        latest = candidates[-1]
        try:
            import joblib
            model = joblib.load(latest)
            logger.info("Loaded {} model from {}", model_name, latest.name)
            return model
        except Exception as exc:
            logger.error("Failed to load model {}: {}", latest, exc)
            return None

    def _load_all_models(self) -> None:
        """Attempt to load all three models on initialisation."""
        self._direction_model  = self.load_latest("direction")
        self._regime_model     = self.load_latest("regime")
        self._confidence_model = self.load_latest("confidence")

        loaded = sum([
            self._direction_model  is not None,
            self._regime_model     is not None,
            self._confidence_model is not None,
        ])
        if loaded == 0:
            logger.warning(
                "No ML models loaded. Bot will run with rule-based signals only. "
                "Train models with: python -m ml.train"
            )
        else:
            logger.info("{}/3 ML models loaded.", loaded)
