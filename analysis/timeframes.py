"""
analysis/timeframes.py
-----------------------
Multi-timeframe alignment utilities.

Higher timeframes (D1, H4) establish directional bias.
Mid timeframes (H1, M30, M15) provide structure.
Lower timeframes (M5, M1) are used only for entry timing.

Confluence across at least two timeframes is required before a signal
is considered valid — this threshold is configurable in config/settings.py.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger


# Canonical ordering from lowest to highest timeframe
TIMEFRAME_ORDER = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]

# Resampling rules for converting lower-TF OHLCV to higher TFs
_RESAMPLE_MAP = {
    "M5":  "5min",
    "M15": "15min",
    "M30": "30min",
    "H1":  "1h",
    "H4":  "4h",
    "D1":  "1D",
    "W1":  "1W",
    "MN1": "1ME",
}


class TimeframeAnalyzer:
    """Utility class for multi-timeframe signal confluence analysis.

    How each timeframe contributes to a trade decision:
      - D1 / H4  → Bias (overall trend direction; highest weight)
      - H1 / M30 / M15 → Structure (swing levels, pattern confirmation)
      - M5 / M1  → Entry timing (trigger candle, precision entry)

    A trade should only be taken when bias + structure timeframes agree,
    and entry is confirmed on a lower timeframe — never on a lower TF alone.
    """

    TIMEFRAMES = TIMEFRAME_ORDER

    @staticmethod
    def get_higher_timeframes(tf: str) -> list[str]:
        """Return all timeframes above the given one (ordered low to high)."""
        if tf not in TIMEFRAME_ORDER:
            raise ValueError(f"Unknown timeframe: {tf}")
        idx = TIMEFRAME_ORDER.index(tf)
        return TIMEFRAME_ORDER[idx + 1:]

    @staticmethod
    def get_bias_timeframes() -> list[str]:
        """High-level timeframes used to determine overall directional bias."""
        return ["D1", "H4"]

    @staticmethod
    def get_structure_timeframes() -> list[str]:
        """Mid-level timeframes used to identify swing structure and key levels."""
        return ["H1", "M30", "M15"]

    @staticmethod
    def get_entry_timeframes() -> list[str]:
        """Low-level timeframes used only for precise entry timing."""
        return ["M5", "M1"]

    @staticmethod
    def align_signals(signals: dict[str, str]) -> dict:
        """Check directional agreement across multiple timeframe signals.

        Args:
            signals: Dict mapping timeframe -> direction string
                     e.g. {'H4': 'bullish', 'H1': 'bullish', 'M15': 'bearish'}

        Returns:
            Dict with:
              'dominant_direction': 'bullish' | 'bearish' | 'neutral'
              'agreement_count':    int — how many TFs agree on dominant direction
              'agreement_score':    float 0-1 — fraction of TFs in agreement
              'details':            the input signals dict
        """
        if not signals:
            return {
                "dominant_direction": "neutral",
                "agreement_count": 0,
                "agreement_score": 0.0,
                "details": {},
            }

        bull_count = sum(1 for v in signals.values() if v == "bullish")
        bear_count = sum(1 for v in signals.values() if v == "bearish")
        total      = len(signals)

        if bull_count > bear_count:
            dominant   = "bullish"
            agree_count = bull_count
        elif bear_count > bull_count:
            dominant   = "bearish"
            agree_count = bear_count
        else:
            dominant   = "neutral"
            agree_count = 0

        return {
            "dominant_direction": dominant,
            "agreement_count":    agree_count,
            "agreement_score":    agree_count / total if total > 0 else 0.0,
            "details":            signals,
        }

    @staticmethod
    def resample_to_higher(
        df: pd.DataFrame,
        from_tf: str,
        to_tf: str,
    ) -> pd.DataFrame:
        """Resample a lower-timeframe OHLCV DataFrame to a higher timeframe.

        Uses standard OHLCV aggregation:
          open  = first, high = max, low = min, close = last, volume = sum

        Args:
            df:      OHLCV DataFrame indexed by UTC datetime.
            from_tf: Source timeframe string (e.g. 'M15').
            to_tf:   Target timeframe string (e.g. 'H1').

        Returns:
            Resampled DataFrame at the target timeframe.
        """
        if to_tf not in _RESAMPLE_MAP:
            raise ValueError(
                f"Cannot resample to '{to_tf}'. Supported targets: {list(_RESAMPLE_MAP.keys())}"
            )

        rule = _RESAMPLE_MAP[to_tf]
        resampled = df.resample(rule).agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}
        ).dropna()

        logger.debug(
            "Resampled {} -> {} ({} bars -> {} bars)",
            from_tf, to_tf, len(df), len(resampled),
        )
        return resampled
