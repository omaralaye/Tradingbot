"""
analysis/patterns/support_resistance.py
----------------------------------------
Detects support and resistance levels by finding swing high/low points
and clustering nearby levels based on price proximity.

Levels are ranked by touch count and proximity to current price.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class SRLevel:
    """A support or resistance price level.

    Attributes:
        price:       The level price.
        strength:    0.0 (weak) to 1.0 (very strong), based on touch count and recency.
        level_type:  'support' or 'resistance'.
        touch_count: Number of times price tested this level.
    """
    price:       float
    strength:    float
    level_type:  str
    touch_count: int


class SupportResistanceDetector:
    """Identifies S/R levels from swing high/low clustering.

    Algorithm:
      1. Find local swing highs and swing lows using a rolling lookback window.
      2. Cluster nearby swing points (within cluster_tolerance_pct of each other).
      3. Rank clusters by touch count; return the top N.
    """

    def __init__(
        self,
        swing_lookback: int = 10,
        cluster_tolerance_pct: float = 0.002,  # 0.2% of price
    ):
        """
        Args:
            swing_lookback:         Bars on each side for a point to be a swing high/low.
            cluster_tolerance_pct:  Levels within this % are merged into one cluster.
        """
        self._lookback  = swing_lookback
        self._tolerance = cluster_tolerance_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, df: pd.DataFrame, n_levels: int = 5) -> list[SRLevel]:
        """Find the strongest support and resistance levels.

        Args:
            df:       OHLCV DataFrame.
            n_levels: Maximum number of levels to return (sorted by strength).

        Returns:
            List of SRLevel objects (strongest first).
        """
        if len(df) < self._lookback * 2 + 1:
            logger.warning("Insufficient bars for S/R detection.")
            return []

        highs = self._find_swing_highs(df)
        lows  = self._find_swing_lows(df)

        resistance_levels = self._cluster_levels(highs, "resistance", df)
        support_levels    = self._cluster_levels(lows,  "support",    df)

        all_levels = resistance_levels + support_levels
        all_levels.sort(key=lambda x: x.strength, reverse=True)
        return all_levels[:n_levels]

    def get_nearest_support(
        self,
        df: pd.DataFrame,
        current_price: float,
    ) -> Optional[SRLevel]:
        """Return the nearest support level below the current price."""
        levels = self.detect(df)
        candidates = [l for l in levels if l.level_type == "support" and l.price < current_price]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x.price)  # closest below

    def get_nearest_resistance(
        self,
        df: pd.DataFrame,
        current_price: float,
    ) -> Optional[SRLevel]:
        """Return the nearest resistance level above the current price."""
        levels = self.detect(df)
        candidates = [l for l in levels if l.level_type == "resistance" and l.price > current_price]
        if not candidates:
            return None
        return min(candidates, key=lambda x: x.price)  # closest above

    @staticmethod
    def distance_to_level(current_price: float, level: SRLevel) -> float:
        """Return the percentage distance from current price to a level."""
        if current_price == 0:
            return 0.0
        return abs(current_price - level.price) / current_price

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_swing_highs(self, df: pd.DataFrame) -> list[float]:
        """Find local high points where high > all highs in lookback window."""
        highs = df["high"].values
        swing_highs = []
        lb = self._lookback
        for i in range(lb, len(highs) - lb):
            window = highs[i - lb: i + lb + 1]
            if highs[i] == np.max(window):
                swing_highs.append(float(highs[i]))
        return swing_highs

    def _find_swing_lows(self, df: pd.DataFrame) -> list[float]:
        """Find local low points where low < all lows in lookback window."""
        lows = df["low"].values
        swing_lows = []
        lb = self._lookback
        for i in range(lb, len(lows) - lb):
            window = lows[i - lb: i + lb + 1]
            if lows[i] == np.min(window):
                swing_lows.append(float(lows[i]))
        return swing_lows

    def _cluster_levels(
        self,
        prices: list[float],
        level_type: str,
        df: pd.DataFrame,
    ) -> list[SRLevel]:
        """Merge nearby price levels into clusters and score them."""
        if not prices:
            return []

        current_price = float(df["close"].iloc[-1])
        prices_arr    = np.array(sorted(prices))
        clusters: list[list[float]] = []

        for price in prices_arr:
            placed = False
            for cluster in clusters:
                centroid = np.mean(cluster)
                if abs(price - centroid) / max(centroid, 1e-10) <= self._tolerance:
                    cluster.append(price)
                    placed = True
                    break
            if not placed:
                clusters.append([price])

        levels = []
        for cluster in clusters:
            centroid    = float(np.mean(cluster))
            touch_count = len(cluster)
            # Strength: normalised by max possible touches, discounted by distance from price
            distance_factor = 1.0 / (1.0 + abs(centroid - current_price) / max(current_price, 1e-10) * 10)
            strength = min(1.0, (touch_count / max(len(prices), 1)) * 0.7 + distance_factor * 0.3)

            levels.append(SRLevel(
                price=centroid,
                strength=round(strength, 3),
                level_type=level_type,
                touch_count=touch_count,
            ))

        return levels
