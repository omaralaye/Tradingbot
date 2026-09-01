"""
ml/features.py
--------------
Feature engineering for ML models. All features are derived from data
already available at signal time — NO future-leaking features.

The output is a flat pd.Series of named floats, ready for model input.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from analysis.indicators import add_ema, add_atr, add_adx, add_rsi, add_macd, add_bollinger_bands


class FeatureEngineer:
    """Builds a feature vector from all analysis outputs.

    Feature categories:
      1. Per-timeframe technical indicators (EMA slope, ATR, RSI, ADX, MACD, BB)
      2. Regime classification (one-hot + confidence)
      3. Session context (overlap flag, volatility multiplier)
      4. Pattern signals (candlestick bull/bear confidence, S/R distance, chart pattern)
      5. Price-derived (returns, realized volatility, candle structure)

    All features are normalised or bounded where possible to improve model stability.
    """

    FEATURE_TIMEFRAMES = ["M15", "H1", "H4", "D1"]  # TFs included in feature set

    def build_features(
        self,
        ohlcv_by_tf: dict[str, pd.DataFrame],
        regime_result=None,
        session_info=None,
        candle_signals: list = None,
        sr_levels: list = None,
        chart_patterns: list = None,
    ) -> pd.Series:
        """Build the full feature vector for a single prediction point.

        Args:
            ohlcv_by_tf:    Dict of {timeframe: OHLCV DataFrame}.
            regime_result:  RegimeResult from RegimeDetector (or None).
            session_info:   SessionInfo from SessionContext (or None).
            candle_signals: List of CandleSignal objects (or None).
            sr_levels:      List of SRLevel objects (or None).
            chart_patterns: List of ChartPattern objects (or None).

        Returns:
            pd.Series of floats, indexed by feature name.
        """
        features: dict[str, float] = {}

        # 1. Per-timeframe indicator features
        for tf in self.FEATURE_TIMEFRAMES:
            df = ohlcv_by_tf.get(tf)
            if df is None or len(df) < 50:
                # Fill with NaN if data unavailable — model should handle gracefully
                for fname in self._tf_feature_names(tf):
                    features[fname] = np.nan
                continue
            features.update(self._extract_tf_features(df, tf))

        # 2. Regime features
        features.update(self._extract_regime_features(regime_result))

        # 3. Session features
        features.update(self._extract_session_features(session_info))

        # 4. Pattern features
        features.update(self._extract_pattern_features(
            candle_signals or [], sr_levels or [], chart_patterns or [],
            ohlcv_by_tf.get("H1"),
        ))

        # 5. Price-derived features from H1
        features.update(self._extract_price_features(ohlcv_by_tf.get("H1")))

        return pd.Series(features, dtype=float)

    def get_feature_names(self) -> list[str]:
        """Return the list of all feature names in the expected order."""
        names: list[str] = []
        for tf in self.FEATURE_TIMEFRAMES:
            names.extend(self._tf_feature_names(tf))
        names.extend(self._regime_feature_names())
        names.extend(self._session_feature_names())
        names.extend(self._pattern_feature_names())
        names.extend(self._price_feature_names())
        return names

    # ------------------------------------------------------------------
    # Per-timeframe features
    # ------------------------------------------------------------------

    def _extract_tf_features(self, df: pd.DataFrame, tf: str) -> dict[str, float]:
        """Extract indicator features for one timeframe."""
        close = df["close"]
        prefix = f"{tf}_"

        ema9   = add_ema(df, 9)
        ema21  = add_ema(df, 21)
        ema50  = add_ema(df, 50)
        atr    = add_atr(df, 14)
        rsi    = add_rsi(df, 14)
        adx_df = add_adx(df, 14)
        macd_df= add_macd(df)
        bb_df  = add_bollinger_bands(df)

        price = float(close.iloc[-1])
        atr_val = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 1.0

        # EMA slope: 5-bar slope normalised by price
        def _slope(series: pd.Series) -> float:
            vals = series.iloc[-5:].values
            if len(vals) < 2 or np.isnan(vals).any():
                return 0.0
            raw = float(np.polyfit(range(len(vals)), vals, 1)[0])
            return raw / max(price, 1e-10)

        return {
            f"{prefix}ema9_slope":  _slope(ema9),
            f"{prefix}ema21_slope": _slope(ema21),
            f"{prefix}ema50_slope": _slope(ema50),
            f"{prefix}ema9_21_cross": float((ema9.iloc[-1] - ema21.iloc[-1]) / max(atr_val, 1e-10)),
            f"{prefix}atr_norm":    atr_val / max(price, 1e-10),
            f"{prefix}rsi":         float(rsi.iloc[-1]) / 100.0 if not np.isnan(rsi.iloc[-1]) else 0.5,
            f"{prefix}adx":         float(adx_df["ADX"].iloc[-1]) / 100.0 if not np.isnan(adx_df["ADX"].iloc[-1]) else 0.0,
            f"{prefix}di_diff":     float((adx_df["DI_plus"].iloc[-1] - adx_df["DI_minus"].iloc[-1]) / 100.0),
            f"{prefix}macd_hist":   float(macd_df["histogram"].iloc[-1]) / max(atr_val, 1e-10),
            f"{prefix}bb_pct_b":    float(bb_df["BB_pct_b"].iloc[-1]) if not np.isnan(bb_df["BB_pct_b"].iloc[-1]) else 0.5,
        }

    def _tf_feature_names(self, tf: str) -> list[str]:
        prefix = f"{tf}_"
        return [
            f"{prefix}ema9_slope", f"{prefix}ema21_slope", f"{prefix}ema50_slope",
            f"{prefix}ema9_21_cross", f"{prefix}atr_norm", f"{prefix}rsi",
            f"{prefix}adx", f"{prefix}di_diff", f"{prefix}macd_hist", f"{prefix}bb_pct_b",
        ]

    # ------------------------------------------------------------------
    # Regime features
    # ------------------------------------------------------------------

    def _extract_regime_features(self, regime_result) -> dict[str, float]:
        if regime_result is None:
            return {n: 0.0 for n in self._regime_feature_names()}
        return {
            "regime_is_uptrend":   float(regime_result.regime.value == "uptrend"),
            "regime_is_downtrend": float(regime_result.regime.value == "downtrend"),
            "regime_is_ranging":   float(regime_result.regime.value == "ranging"),
            "regime_is_choppy":    float(regime_result.regime.value == "choppy"),
            "regime_confidence":   float(regime_result.confidence),
            "regime_adx":          float(regime_result.adx) / 100.0,
        }

    def _regime_feature_names(self) -> list[str]:
        return ["regime_is_uptrend", "regime_is_downtrend", "regime_is_ranging",
                "regime_is_choppy", "regime_confidence", "regime_adx"]

    # ------------------------------------------------------------------
    # Session features
    # ------------------------------------------------------------------

    def _extract_session_features(self, session_info) -> dict[str, float]:
        if session_info is None:
            return {n: 0.0 for n in self._session_feature_names()}
        vol_map = {"low": 0.0, "medium": 0.33, "high": 0.67, "very_high": 1.0}
        return {
            "session_is_overlap":     float(session_info.is_overlap),
            "session_london_active":  float("london" in session_info.active_sessions),
            "session_ny_active":      float("new_york" in session_info.active_sessions),
            "session_tokyo_active":   float("tokyo" in session_info.active_sessions),
            "session_volatility":     vol_map.get(session_info.volatility_profile, 0.0),
        }

    def _session_feature_names(self) -> list[str]:
        return ["session_is_overlap", "session_london_active", "session_ny_active",
                "session_tokyo_active", "session_volatility"]

    # ------------------------------------------------------------------
    # Pattern features
    # ------------------------------------------------------------------

    def _extract_pattern_features(
        self,
        candle_signals: list,
        sr_levels: list,
        chart_patterns: list,
        h1_df: Optional[pd.DataFrame],
    ) -> dict[str, float]:
        bull_candle = max((s.confidence for s in candle_signals if s.direction == "bullish"), default=0.0)
        bear_candle = max((s.confidence for s in candle_signals if s.direction == "bearish"), default=0.0)

        current_price = float(h1_df["close"].iloc[-1]) if h1_df is not None and len(h1_df) > 0 else 0.0
        nearest_sr_dist = 1.0
        if sr_levels and current_price > 0:
            distances = [abs(l.price - current_price) / current_price for l in sr_levels]
            nearest_sr_dist = min(distances) if distances else 1.0

        chart_bull = max((p.confidence for p in chart_patterns if p.direction == "bullish"), default=0.0)
        chart_bear = max((p.confidence for p in chart_patterns if p.direction == "bearish"), default=0.0)

        return {
            "pattern_candle_bull_conf": float(bull_candle),
            "pattern_candle_bear_conf": float(bear_candle),
            "pattern_nearest_sr_dist":  float(np.clip(nearest_sr_dist, 0, 1)),
            "pattern_chart_bull_conf":  float(chart_bull),
            "pattern_chart_bear_conf":  float(chart_bear),
        }

    def _pattern_feature_names(self) -> list[str]:
        return ["pattern_candle_bull_conf", "pattern_candle_bear_conf",
                "pattern_nearest_sr_dist", "pattern_chart_bull_conf", "pattern_chart_bear_conf"]

    # ------------------------------------------------------------------
    # Price-derived features
    # ------------------------------------------------------------------

    def _extract_price_features(self, df: Optional[pd.DataFrame]) -> dict[str, float]:
        if df is None or len(df) < 21:
            return {n: 0.0 for n in self._price_feature_names()}

        close = df["close"]
        ret_5  = float((close.iloc[-1] / close.iloc[-6] - 1))  if len(close) >= 6  else 0.0
        ret_10 = float((close.iloc[-1] / close.iloc[-11] - 1)) if len(close) >= 11 else 0.0
        ret_20 = float((close.iloc[-1] / close.iloc[-21] - 1)) if len(close) >= 21 else 0.0
        log_ret = float(np.log(close.iloc[-1] / close.iloc[-2])) if close.iloc[-2] > 0 else 0.0
        rvol   = float(close.pct_change().iloc[-20:].std())

        c = df.iloc[-1]
        body_pct   = abs(c["close"] - c["open"]) / max(c["high"] - c["low"], 1e-10)
        upper_wick = (c["high"] - max(c["open"], c["close"])) / max(c["high"] - c["low"], 1e-10)
        lower_wick = (min(c["open"], c["close"]) - c["low"]) / max(c["high"] - c["low"], 1e-10)

        return {
            "price_ret_5bar":    float(np.clip(ret_5,  -0.1, 0.1)),
            "price_ret_10bar":   float(np.clip(ret_10, -0.1, 0.1)),
            "price_ret_20bar":   float(np.clip(ret_20, -0.1, 0.1)),
            "price_log_return":  float(np.clip(log_ret, -0.05, 0.05)),
            "price_rvol_20":     float(np.clip(rvol, 0, 0.05)),
            "price_body_pct":    float(body_pct),
            "price_upper_wick":  float(upper_wick),
            "price_lower_wick":  float(lower_wick),
        }

    def _price_feature_names(self) -> list[str]:
        return ["price_ret_5bar", "price_ret_10bar", "price_ret_20bar",
                "price_log_return", "price_rvol_20", "price_body_pct",
                "price_upper_wick", "price_lower_wick"]
