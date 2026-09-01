"""
analysis/patterns/candlestick_patterns.py
-----------------------------------------
Detects single- and multi-candle patterns and returns a confidence score
rather than a simple boolean, since pattern recognition is inherently fuzzy.

Patterns detected: Engulfing, Pin Bar / Hammer, Doji, Morning/Evening Star.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class CandleSignal:
    """A detected candlestick pattern on a specific bar.

    Attributes:
        pattern:    Pattern name (e.g. 'engulfing', 'pin_bar', 'doji').
        direction:  'bullish', 'bearish', or 'neutral'.
        confidence: 0.0 (weak) to 1.0 (textbook pattern).
        bar_index:  Position in the DataFrame (-1 = last bar, -2 = second-to-last, etc.)
    """
    pattern:    str
    direction:  str
    confidence: float
    bar_index:  int = -1


class CandlestickDetector:
    """Scans recent bars for candlestick patterns.

    All detectors examine the last few bars of the supplied DataFrame.
    Confidence scores are derived from geometric ratios (wick/body, etc.)
    to reflect how closely the candle matches the textbook definition.
    """

    def __init__(
        self,
        min_confidence: float = 0.4,
        wick_to_body_ratio: float = 2.0,
        doji_body_pct: float = 0.10,
    ):
        """
        Args:
            min_confidence:    Minimum confidence to include a signal in results.
            wick_to_body_ratio: Pin bar: wick must be >= this multiple of body.
            doji_body_pct:     Doji: body must be <= this fraction of total candle range.
        """
        self._min_conf    = min_confidence
        self._wick_ratio  = wick_to_body_ratio
        self._doji_pct    = doji_body_pct

    def detect_all(self, df: pd.DataFrame) -> list[CandleSignal]:
        """Run all pattern detectors and return confirmed signals.

        Args:
            df: OHLCV DataFrame. Requires at least 3 bars.

        Returns:
            List of CandleSignal objects, sorted by confidence descending.
        """
        if len(df) < 3:
            return []

        signals: list[CandleSignal] = []
        detectors = [
            self._detect_engulfing,
            self._detect_pin_bar,
            self._detect_doji,
            self._detect_morning_evening_star,
        ]
        for detector in detectors:
            try:
                result = detector(df)
                if result and result.confidence >= self._min_conf:
                    signals.append(result)
            except Exception as exc:
                logger.debug("Candlestick detector error: {}", exc)

        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals

    # ------------------------------------------------------------------
    # Individual pattern detectors
    # ------------------------------------------------------------------

    def _detect_engulfing(self, df: pd.DataFrame) -> Optional[CandleSignal]:
        """Bullish or bearish engulfing pattern (2-candle reversal).

        Bullish: current candle is green and fully engulfs the prior red candle.
        Bearish: current candle is red and fully engulfs the prior green candle.
        """
        prev = df.iloc[-2]
        curr = df.iloc[-1]

        prev_body = abs(prev["close"] - prev["open"])
        curr_body = abs(curr["close"] - curr["open"])

        if prev_body == 0 or curr_body == 0:
            return None

        # Bullish engulfing
        if (prev["close"] < prev["open"]          # prior candle bearish
                and curr["close"] > curr["open"]   # current candle bullish
                and curr["open"]  < prev["close"]  # opens below prior close
                and curr["close"] > prev["open"]): # closes above prior open
            confidence = min(1.0, curr_body / prev_body * 0.8)
            return CandleSignal("engulfing", "bullish", confidence, -1)

        # Bearish engulfing
        if (prev["close"] > prev["open"]
                and curr["close"] < curr["open"]
                and curr["open"]  > prev["close"]
                and curr["close"] < prev["open"]):
            confidence = min(1.0, curr_body / prev_body * 0.8)
            return CandleSignal("engulfing", "bearish", confidence, -1)

        return None

    def _detect_pin_bar(self, df: pd.DataFrame) -> Optional[CandleSignal]:
        """Pin bar / hammer / shooting star — long wick reversal signal.

        Bullish (hammer): long lower wick, small body near the top.
        Bearish (shooting star): long upper wick, small body near the bottom.

        Confidence scales with the wick-to-body ratio.
        """
        c = df.iloc[-1]
        body  = abs(c["close"] - c["open"])
        total = c["high"] - c["low"]

        if total == 0:
            return None

        upper_wick = c["high"] - max(c["open"], c["close"])
        lower_wick = min(c["open"], c["close"]) - c["low"]

        # Bullish hammer: long lower wick, short upper wick
        if lower_wick >= self._wick_ratio * max(body, 1e-10) and upper_wick < lower_wick * 0.4:
            ratio      = lower_wick / max(body, 1e-10)
            confidence = min(1.0, (ratio - self._wick_ratio) / self._wick_ratio + 0.5)
            return CandleSignal("pin_bar", "bullish", float(confidence), -1)

        # Bearish shooting star: long upper wick, short lower wick
        if upper_wick >= self._wick_ratio * max(body, 1e-10) and lower_wick < upper_wick * 0.4:
            ratio      = upper_wick / max(body, 1e-10)
            confidence = min(1.0, (ratio - self._wick_ratio) / self._wick_ratio + 0.5)
            return CandleSignal("pin_bar", "bearish", float(confidence), -1)

        return None

    def _detect_doji(self, df: pd.DataFrame) -> Optional[CandleSignal]:
        """Doji — open ≈ close, indicating indecision.

        Confidence is inversely proportional to body size relative to range.
        Direction is 'neutral' — doji alone is not directional.
        """
        c = df.iloc[-1]
        total = c["high"] - c["low"]
        if total == 0:
            return None
        body = abs(c["close"] - c["open"])
        body_pct = body / total

        if body_pct <= self._doji_pct:
            confidence = 1.0 - (body_pct / self._doji_pct)
            return CandleSignal("doji", "neutral", float(confidence), -1)
        return None

    def _detect_morning_evening_star(self, df: pd.DataFrame) -> Optional[CandleSignal]:
        """Morning Star (bullish) / Evening Star (bearish) — 3-candle reversal.

        Morning Star:
          Bar -3: large bearish candle
          Bar -2: small body (indecision), gaps down
          Bar -1: large bullish candle closing into bar -3's body

        Evening Star: mirror pattern.
        """
        if len(df) < 3:
            return None

        a = df.iloc[-3]   # first candle
        b = df.iloc[-2]   # star candle
        c = df.iloc[-1]   # confirmation candle

        a_body = abs(a["close"] - a["open"])
        b_body = abs(b["close"] - b["open"])
        c_body = abs(c["close"] - c["open"])

        if a_body == 0 or c_body == 0:
            return None

        star_is_small = b_body < a_body * 0.3

        # Morning Star: bearish -> star -> bullish
        if (a["close"] < a["open"]   # bearish first candle
                and star_is_small
                and c["close"] > c["open"]  # bullish confirmation
                and c["close"] > a["open"] + a_body * 0.4):
            confidence = min(1.0, c_body / a_body * 0.7 + 0.3)
            return CandleSignal("morning_star", "bullish", float(confidence), -1)

        # Evening Star: bullish -> star -> bearish
        if (a["close"] > a["open"]
                and star_is_small
                and c["close"] < c["open"]
                and c["close"] < a["open"] - a_body * 0.4):
            confidence = min(1.0, c_body / a_body * 0.7 + 0.3)
            return CandleSignal("evening_star", "bearish", float(confidence), -1)

        return None
