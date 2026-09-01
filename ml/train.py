"""
ml/train.py
-----------
Walk-forward training pipeline for all ML models.

Uses walk-forward cross-validation (not random k-fold) to respect
time ordering of data and prevent future leakage. Includes purge and
embargo gaps between train and test windows.

Reference: De Prado, M. L. (2018). Advances in Financial Machine Learning,
           Chapter 7 (Cross-Validation in Finance).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
from loguru import logger

from ml.models.direction_model   import DirectionModel
from ml.models.regime_model      import RegimeModel
from ml.models.confidence_model  import ConfidenceModel


class WalkForwardTrainer:
    """Implements walk-forward cross-validation with purge/embargo.

    Walk-forward means: train on past data, test on immediate future,
    roll the window forward. Never train on data that comes after test data.

    Purge:   Remove training samples whose labels overlap with the test window
             (e.g. a triple-barrier window that extends into the test period).
    Embargo: Remove training samples immediately before the test window to
             avoid close temporal correlation contaminating the split.
    """

    def __init__(
        self,
        model_class,
        n_splits:      int = 5,
        purge_bars:    int = 10,
        embargo_bars:  int = 5,
        registry_dir:  str = "ml/registry",
    ):
        """
        Args:
            model_class:   One of DirectionModel, RegimeModel, ConfidenceModel.
            n_splits:      Number of walk-forward splits.
            purge_bars:    Bars to remove from training end (label leakage).
            embargo_bars:  Additional bars to skip after purge (temporal corr).
            registry_dir:  Where to save trained model + metadata.
        """
        self._model_class  = model_class
        self._n_splits     = n_splits
        self._purge        = purge_bars
        self._embargo      = embargo_bars
        self._registry_dir = Path(registry_dir)
        self._registry_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
    ) -> dict:
        """Run walk-forward cross-validation and return fold metrics.

        Args:
            X:          Feature DataFrame, time-ordered.
            y:          Label Series, aligned with X.
            model_name: Identifier for logging and registry (e.g. 'direction').

        Returns:
            Dict with per-fold metrics and aggregate statistics.
        """
        X, y = X.align(y, join="inner", axis=0)
        X    = X.dropna()
        y    = y.loc[X.index].dropna()
        X    = X.loc[y.index]

        n        = len(X)
        metrics  = {"folds": [], "model_name": model_name}
        best_model = None
        best_score = -np.inf

        logger.info("Starting walk-forward training: {} samples, {} splits.", n, self._n_splits)

        for fold_idx, (train_idx, test_idx) in enumerate(self._make_splits(n)):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test  = X.iloc[test_idx]
            y_test  = y.iloc[test_idx]

            if len(X_train) < 50 or len(X_test) < 10:
                logger.warning("Fold {} too small — skipping.", fold_idx)
                continue

            model = self._model_class()
            model.fit(X_train, y_train)

            # Evaluate on test fold
            fold_metrics = self._evaluate(model, X_test, y_test)
            fold_metrics["fold"] = fold_idx
            fold_metrics["train_size"] = len(X_train)
            fold_metrics["test_size"]  = len(X_test)
            metrics["folds"].append(fold_metrics)

            score = fold_metrics.get("accuracy", 0.0)
            if score > best_score:
                best_score = score
                best_model = model

            logger.info(
                "Fold {}/{} | train={} | test={} | accuracy={:.3f}",
                fold_idx + 1, self._n_splits,
                len(X_train), len(X_test), score,
            )

        # Aggregate metrics
        if metrics["folds"]:
            accs = [f["accuracy"] for f in metrics["folds"]]
            metrics["mean_accuracy"] = float(np.mean(accs))
            metrics["std_accuracy"]  = float(np.std(accs))
            logger.info(
                "Walk-forward complete | mean accuracy={:.3f} ± {:.3f}",
                metrics["mean_accuracy"], metrics["std_accuracy"],
            )

        if best_model is not None:
            self.save_to_registry(best_model, metrics, model_name, list(X.columns))

        return metrics

    def save_to_registry(
        self,
        model,
        metrics:       dict,
        model_name:    str,
        feature_names: list[str],
    ) -> None:
        """Save the trained model and metadata JSON to ml/registry/.

        File naming: ml/registry/<model_name>_<timestamp>.joblib
                     ml/registry/<model_name>_<timestamp>_metadata.json
        """
        ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        stem     = f"{model_name}_{ts}"
        model_path = self._registry_dir / f"{stem}.joblib"
        meta_path  = self._registry_dir / f"{stem}_metadata.json"

        model.save(model_path)

        metadata = {
            "model_name":    model_name,
            "train_date":    ts,
            "feature_names": feature_names,
            "metrics":       metrics,
        }
        meta_path.write_text(json.dumps(metadata, indent=2, default=str))
        logger.info("Registry entry saved: {} + {}", model_path.name, meta_path.name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_splits(self, n: int) -> Generator[tuple[list[int], list[int]], None, None]:
        """Yield (train_indices, test_indices) with purge + embargo gaps."""
        fold_size  = n // (self._n_splits + 1)
        test_size  = fold_size

        for i in range(self._n_splits):
            test_start  = fold_size * (i + 1)
            test_end    = test_start + test_size
            if test_end > n:
                break

            train_end = test_start - self._purge - self._embargo
            if train_end <= 0:
                continue

            train_idx = list(range(0, train_end))
            test_idx  = list(range(test_start, test_end))
            yield train_idx, test_idx

    @staticmethod
    def _evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Compute test-fold metrics. Handles different model predict interfaces."""
        from sklearn.metrics import accuracy_score
        try:
            if hasattr(model, "predict"):
                preds = [model.predict(X_test.iloc[[i]])[0] for i in range(len(X_test))]
            elif hasattr(model, "predict_win_prob"):
                preds = [int(model.predict_win_prob(X_test.iloc[[i]]) >= 0.5) for i in range(len(X_test))]
            else:
                return {"accuracy": 0.0}
            accuracy = float(accuracy_score(y_test, preds))
        except Exception as exc:
            logger.warning("Evaluation error: {}", exc)
            accuracy = 0.0
        return {"accuracy": accuracy}
