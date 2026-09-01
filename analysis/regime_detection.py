"""
analysis/regime_detection.py
-----------------------------
Classifies the current market regime for a given symbol/timeframe into:
  UPTREND | DOWNTREND | RANGING | CHOPPY

The classification feeds both the signal engine (as a filter/gate) and
the ML feature set (as a rule-based label for the regime model).
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from loguru import logger

from analysis.indicators import add_adx, add_ema


class RegimeType(str, Enum):
    """Market regime classifications."""
    UPTREND   = "uptrend"
    DOWNTREND = "downtrend"
    RANGING   = "ranging"
    CHOPPY    = "choppy"


@dataclass
class RegimeResult:
    """Output of a single regime detection pass.

    Attributes:
        regime:     The detected market regime.
        confidence: Score from 0.0 (uncertain) to 1.0 (very confident).
        adx:        Current ADX value (trend strength indicator).
        ema_slope:  Normalised slope of the trend EMA.
        reasoning:  Human-readable explanation of the classification.
    """
    regime:     RegimeType
    confidence: float
    adx:        float
    ema_slope:  float
    reasoning:  str


class RegimeDetector:
    """Classifies market regime using ADX + EMA slope.

    Logic:
      - ADX > adx_trend_threshold  => Market is trending.
        - EMA slope > 0            => UPTREND
        - EMA slope < 0            => DOWNTREND
      - ADX < adx_range_threshold  => RANGING (oscillating price)
      - Otherwise (between thresholds, conflicting signals) => CHOPPY

    Confidence is derived from:
      - How far ADX is from the decision boundary.
      - How steep the EMA slope is relative to price.
    """

    def __init__(
        self,
        adx_period: int = 14,
        adx_trend_threshold: float = 25.0,
        adx_range_threshold: float = 20.0,
        ema_period: int = 50,
    ):
        """
        Args:
            adx_period:           Lookback for ADX calculation.
            adx_trend_threshold:  ADX above this => trending.
            adx_range_threshold:  ADX below this => ranging.
            ema_period:           EMA period used to measure slope/direction.
        """
        self._adx_period = adx_period
        self._adx_trend  = adx_trend_threshold
        self._adx_range  = adx_range_threshold
        self._ema_period = ema_period

    def detect(self, df: pd.DataFrame) -> RegimeResult:
        """Detect the market regime from the most recent bars of df.

        Args:
            df: OHLCV DataFrame. Should have at least ema_period + adx_period bars.

        Returns:
            RegimeResult with regime, confidence, and reasoning.
        """
        if len(df) < max(self._adx_period, self._ema_period) + 5:
            logger.warning("Insufficient bars for regime detection ({} bars).", len(df))
            return RegimeResult(
                regime=RegimeType.CHOPPY,
                confidence=0.0,
                adx=0.0,
                ema_slope=0.0,
                reasoning="Insufficient data for regime detection.",
            )

        # Compute indicators on the full series, use only the last value
        adx_df   = add_adx(df, self._adx_period)
        ema_vals = add_ema(df, self._ema_period)

        current_adx = float(adx_df["ADX"].iloc[-1])
        if np.isnan(current_adx):
            current_adx = 0.0

        # EMA slope: rate of change over last 5 bars, normalised by price
        ema_recent = ema_vals.iloc[-5:].values
        slope_raw  = float(np.polyfit(range(len(ema_recent)), ema_recent, 1)[0])
        current_price = float(df["close"].iloc[-1])
        ema_slope = slope_raw / current_price if current_price != 0 else 0.0

        # --- Classification logic ---
        if current_adx >= self._adx_trend:
            # Strong trend — direction determined by EMA slope
            if ema_slope > 0:
                regime = RegimeType.UPTREND
                reasoning = (
                    f"ADX={current_adx:.1f} (>{self._adx_trend}) signals strong trend; "
                    f"EMA slope={ema_slope:.6f} > 0 => uptrend."
                )
            else:
                regime = RegimeType.DOWNTREND
                reasoning = (
                    f"ADX={current_adx:.1f} (>{self._adx_trend}) signals strong trend; "
                    f"EMA slope={ema_slope:.6f} < 0 => downtrend."
                )
            # Confidence scales with how far ADX is above the trend threshold
            confidence = min(1.0, (current_adx - self._adx_trend) / 25.0 + 0.5)

        elif current_adx <= self._adx_range:
            # Weak ADX => price oscillating in a range
            regime = RegimeType.RANGING
            reasoning = (
                f"ADX={current_adx:.1f} (<={self._adx_range}) => low trend strength; "
                "price likely oscillating in a range."
            )
            confidence = min(1.0, (self._adx_range - current_adx) / self._adx_range + 0.3)

        else:
            # ADX in the grey zone — treat as choppy/transitional
            regime = RegimeType.CHOPPY
            reasoning = (
                f"ADX={current_adx:.1f} between range ({self._adx_range}) and trend "
                f"({self._adx_trend}) thresholds => transitional/choppy; "
                "be more conservative."
            )
            confidence = 0.3  # inherently uncertain in this zone

        confidence = float(np.clip(confidence, 0.0, 1.0))
        logger.debug(
            "Regime: {} | confidence={:.2f} | ADX={:.1f} | slope={:.6f}",
            regime.value, confidence, current_adx, ema_slope,
        )

        return RegimeResult(
            regime=regime,
            confidence=confidence,
            adx=current_adx,
            ema_slope=ema_slope,
            reasoning=reasoning,
        )
