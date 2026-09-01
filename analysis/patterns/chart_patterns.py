"""
analysis/patterns/chart_patterns.py
-------------------------------------
Detects multi-bar chart patterns: Double Top/Bottom, Triangle, Head & Shoulders.

Each detector returns a confidence score (not just a boolean) because
pattern recognition on real price data is inherently imprecise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ChartPattern:
    """A detected chart pattern.

    Attributes:
        pattern_type: e.g. 'double_top', 'double_bottom', 'triangle_ascending'.
        direction:    'bullish', 'bearish', or 'neutral'.
        confidence:   0.0 (weak) to 1.0 (clear textbook pattern).
        key_levels:   Relevant price levels (e.g. neckline, peaks, troughs).
        description:  Human-readable explanation of why this pattern was detected.
    """
    pattern_type: str
    direction:    str
    confidence:   float
    key_levels:   dict         = field(default_factory=dict)
    description:  str          = ""


class ChartPatternDetector:
    """Scans OHLCV data for classic chart patterns.

    Patterns are detected on the most recent N bars of the provided DataFrame.
    All detectors use swing high/low identification internally.
    """

    def __init__(
        self,
        swing_lookback: int = 5,
        similarity_tolerance: float = 0.015,  # 1.5% price similarity
        min_confidence: float = 0.40,
    ):
        self._swing_lb   = swing_lookback
        self._tolerance  = similarity_tolerance
        self._min_conf   = min_confidence

    def detect_all(self, df: pd.DataFrame) -> list[ChartPattern]:
        """Run all pattern detectors and return high-confidence patterns.

        Returns:
            List of ChartPattern objects, sorted by confidence descending.
        """
        if len(df) < 20:
            return []

        patterns: list[ChartPattern] = []
        detectors = [
            self._detect_double_top,
            self._detect_double_bottom,
            self._detect_triangle,
            self._detect_head_and_shoulders,
        ]
        for detector in detectors:
            try:
                result = detector(df)
                if result and result.confidence >= self._min_conf:
                    patterns.append(result)
            except Exception as exc:
                logger.debug("Chart pattern error in {}: {}", detector.__name__, exc)

        patterns.sort(key=lambda p: p.confidence, reverse=True)
        return patterns

    # ------------------------------------------------------------------
    # Pattern detectors
    # ------------------------------------------------------------------

    def _detect_double_top(self, df: pd.DataFrame) -> Optional[ChartPattern]:
        """Two peaks at similar price levels with a valley between — bearish reversal."""
        highs = self._find_swing_highs(df, n=10)
        if len(highs) < 2:
            return None

        peak1_price, peak1_idx = highs[-2]
        peak2_price, peak2_idx = highs[-1]

        similarity = abs(peak1_price - peak2_price) / max(peak1_price, 1e-10)
        if similarity > self._tolerance:
            return None  # peaks not close enough in price

        # Ensure there's a valley between the peaks
        valley_data = df.iloc[peak1_idx: peak2_idx]
        if valley_data.empty:
            return None
        valley_low  = valley_data["low"].min()
        neckline    = float(valley_low)
        drop_pct    = (peak1_price - neckline) / peak1_price

        if drop_pct < 0.005:  # valley must be meaningful
            return None

        confidence = max(0.0, 1.0 - similarity / self._tolerance) * min(1.0, drop_pct / 0.02)

        return ChartPattern(
            pattern_type="double_top",
            direction="bearish",
            confidence=float(confidence),
            key_levels={"peak1": peak1_price, "peak2": peak2_price, "neckline": neckline},
            description=(
                f"Double top at {peak1_price:.5f}/{peak2_price:.5f} "
                f"(similarity {similarity*100:.2f}%), neckline at {neckline:.5f}."
            ),
        )

    def _detect_double_bottom(self, df: pd.DataFrame) -> Optional[ChartPattern]:
        """Two troughs at similar price levels with a peak between — bullish reversal."""
        lows = self._find_swing_lows(df, n=10)
        if len(lows) < 2:
            return None

        trough1_price, trough1_idx = lows[-2]
        trough2_price, trough2_idx = lows[-1]

        similarity = abs(trough1_price - trough2_price) / max(trough1_price, 1e-10)
        if similarity > self._tolerance:
            return None

        peak_data = df.iloc[trough1_idx: trough2_idx]
        if peak_data.empty:
            return None
        peak_high  = peak_data["high"].max()
        neckline   = float(peak_high)
        rise_pct   = (neckline - trough1_price) / max(trough1_price, 1e-10)

        if rise_pct < 0.005:
            return None

        confidence = max(0.0, 1.0 - similarity / self._tolerance) * min(1.0, rise_pct / 0.02)

        return ChartPattern(
            pattern_type="double_bottom",
            direction="bullish",
            confidence=float(confidence),
            key_levels={"trough1": trough1_price, "trough2": trough2_price, "neckline": neckline},
            description=(
                f"Double bottom at {trough1_price:.5f}/{trough2_price:.5f}, "
                f"neckline at {neckline:.5f}."
            ),
        )

    def _detect_triangle(self, df: pd.DataFrame) -> Optional[ChartPattern]:
        """Ascending, descending, or symmetrical triangle via trendline slope comparison."""
        recent = df.iloc[-30:]
        highs  = recent["high"].values
        lows   = recent["low"].values
        x      = np.arange(len(recent))

        upper_slope = float(np.polyfit(x, highs, 1)[0])
        lower_slope = float(np.polyfit(x, lows,  1)[0])

        norm  = float(recent["close"].mean())
        upper_slope_n = upper_slope / norm
        lower_slope_n = lower_slope / norm

        is_converging = (upper_slope_n < 0 and lower_slope_n > 0) or abs(upper_slope_n - lower_slope_n) < 0.0001

        if not is_converging:
            return None

        if upper_slope_n > 0.00005 and abs(lower_slope_n) < 0.00005:
            ptype, direction = "triangle_ascending", "bullish"
        elif lower_slope_n < -0.00005 and abs(upper_slope_n) < 0.00005:
            ptype, direction = "triangle_descending", "bearish"
        else:
            ptype, direction = "triangle_symmetrical", "neutral"

        confidence = 0.55  # triangles need breakout confirmation — moderate confidence only

        return ChartPattern(
            pattern_type=ptype,
            direction=direction,
            confidence=confidence,
            key_levels={"upper_slope": upper_slope_n, "lower_slope": lower_slope_n},
            description=f"{ptype.replace('_',' ').title()} detected (upper slope {upper_slope_n:.6f}, lower slope {lower_slope_n:.6f}).",
        )

    def _detect_head_and_shoulders(self, df: pd.DataFrame) -> Optional[ChartPattern]:
        """Head and Shoulders — three peaks, middle (head) is the highest — bearish."""
        highs = self._find_swing_highs(df, n=15)
        if len(highs) < 3:
            return None

        left_shoulder_price, _ = highs[-3]
        head_price,          _ = highs[-2]
        right_shoulder_price, _= highs[-1]

        # Head must be higher than both shoulders
        if not (head_price > left_shoulder_price and head_price > right_shoulder_price):
            return None

        # Shoulders should be roughly symmetric
        shoulder_sim = abs(left_shoulder_price - right_shoulder_price) / max(left_shoulder_price, 1e-10)
        if shoulder_sim > self._tolerance * 2:
            return None

        confidence = max(0.0, 1.0 - shoulder_sim / (self._tolerance * 2)) * 0.75

        return ChartPattern(
            pattern_type="head_and_shoulders",
            direction="bearish",
            confidence=float(confidence),
            key_levels={
                "left_shoulder": left_shoulder_price,
                "head": head_price,
                "right_shoulder": right_shoulder_price,
            },
            description=(
                f"H&S: left shoulder {left_shoulder_price:.5f}, "
                f"head {head_price:.5f}, right shoulder {right_shoulder_price:.5f}."
            ),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_swing_highs(self, df: pd.DataFrame, n: int = 10) -> list[tuple[float, int]]:
        """Return (price, index) of local swing highs in the last n bars."""
        lb     = self._swing_lb
        result = []
        highs  = df["high"].values
        for i in range(lb, len(highs) - lb):
            if highs[i] == max(highs[max(0, i - lb): i + lb + 1]):
                result.append((float(highs[i]), i))
        return result[-n:]

    def _find_swing_lows(self, df: pd.DataFrame, n: int = 10) -> list[tuple[float, int]]:
        """Return (price, index) of local swing lows in the last n bars."""
        lb     = self._swing_lb
        result = []
        lows   = df["low"].values
        for i in range(lb, len(lows) - lb):
            if lows[i] == min(lows[max(0, i - lb): i + lb + 1]):
                result.append((float(lows[i]), i))
        return result[-n:]
