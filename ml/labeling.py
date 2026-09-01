"""
ml/labeling.py
--------------
Generates training labels for ML models.

WARNING: Labels are computed using FUTURE data and are strictly for
offline training. Never call label() during live inference — it will
look ahead into future bars and produce invalid results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


class TripleBarrierLabeler:
    """Labels each bar using the Triple Barrier Method.

    For each bar, define three barriers:
      - Upper barrier (Take Profit): entry + atr * tp_multiplier
      - Lower barrier (Stop Loss):   entry - atr * sl_multiplier
      - Time barrier:                entry + horizon_bars

    Label = +1 if TP is hit first, -1 if SL is hit first, 0 if time expires.
    This is more realistic than fixed-horizon return labeling because it
    mirrors how trades actually resolve in practice.

    Reference: De Prado, M. L. (2018). Advances in Financial Machine Learning.
    """

    def __init__(
        self,
        tp_multiplier: float = 2.0,
        sl_multiplier: float = 1.0,
        horizon_bars:  int   = 20,
    ):
        """
        Args:
            tp_multiplier: Take-profit distance = atr * this value.
            sl_multiplier: Stop-loss distance   = atr * this value.
            horizon_bars:  Maximum bars to wait before labeling as 0 (timeout).
        """
        self._tp_mult   = tp_multiplier
        self._sl_mult   = sl_multiplier
        self._horizon   = horizon_bars

    def label(self, df: pd.DataFrame, atr: pd.Series) -> pd.Series:
        """Apply triple-barrier labeling to the full DataFrame.

        ⚠️  FUTURE-LEAKING — USE ONLY FOR TRAINING, NOT INFERENCE.

        Args:
            df:  OHLCV DataFrame.
            atr: ATR Series aligned with df index.

        Returns:
            pd.Series of integer labels: +1, -1, or 0.
            Last `horizon_bars` rows are NaN (insufficient future data).
        """
        labels = pd.Series(np.nan, index=df.index)
        closes = df["close"].values
        highs  = df["high"].values
        lows   = df["low"].values
        atrs   = atr.values

        for i in range(len(df) - self._horizon):
            entry = closes[i]
            a     = atrs[i]
            if np.isnan(a) or a == 0:
                continue

            tp_price = entry + a * self._tp_mult
            sl_price = entry - a * self._sl_mult
            label    = 0  # timeout by default

            for j in range(i + 1, min(i + self._horizon + 1, len(df))):
                if highs[j] >= tp_price:
                    label = 1
                    break
                if lows[j] <= sl_price:
                    label = -1
                    break

            labels.iloc[i] = label

        logger.info(
            "Triple-barrier labels: +1={}, -1={}, 0={}, NaN={}",
            (labels == 1).sum(), (labels == -1).sum(),
            (labels == 0).sum(), labels.isna().sum(),
        )
        return labels

    def label_regime(
        self,
        df: pd.DataFrame,
        adx: pd.Series,
        ema_slope: pd.Series,
        adx_threshold: float = 25.0,
    ) -> pd.Series:
        """Label each bar's regime using rule-based logic (ground truth for regime model).

        This produces reproducible, consistent regime labels for training.
        The model then generalises beyond the rule's edge cases.

        Returns:
            pd.Series of strings: 'uptrend', 'downtrend', 'ranging', 'choppy'.
        """
        conditions = [
            (adx > adx_threshold) & (ema_slope > 0),   # uptrend
            (adx > adx_threshold) & (ema_slope <= 0),  # downtrend
            (adx <= adx_threshold * 0.8),               # ranging
        ]
        choices = ["uptrend", "downtrend", "ranging"]
        labels  = pd.Series(
            np.select(conditions, choices, default="choppy"),
            index=df.index,
        )
        logger.debug("Regime label counts:\n{}", labels.value_counts().to_string())
        return labels
