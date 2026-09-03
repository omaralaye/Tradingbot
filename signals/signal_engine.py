"""
signals/signal_engine.py
-------------------------
Combines regime, pattern, timeframe confluence, session context, and ML
outputs into a single trade signal: LONG, SHORT, or NO_TRADE.

Every signal includes a full reasoning dict so the trade journal can record
exactly why a trade was (or was not) taken.

No trade is placed without a traceable reason logged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np
from loguru import logger

from analysis.regime_detection import RegimeType
from analysis.timeframes import TimeframeAnalyzer


class SignalType(str, Enum):
    """Possible trade signal outcomes."""
    LONG     = "LONG"
    SHORT    = "SHORT"
    NO_TRADE = "NO_TRADE"


@dataclass
class TradeSignal:
    """The output of one signal engine evaluation.

    Attributes:
        signal:     LONG, SHORT, or NO_TRADE.
        symbol:     Instrument symbol (e.g. 'EURUSD').
        timeframe:  Primary entry timeframe (e.g. 'H1').
        confidence: Composite confidence score 0.0-1.0.
        reasoning:  Full breakdown of why this signal was generated.
        timestamp:  UTC datetime of signal generation.
    """
    signal:     SignalType
    symbol:     str
    timeframe:  str
    confidence: float
    reasoning:  dict  = field(default_factory=dict)
    timestamp:  datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SignalEngine:
    """Orchestrates the full analysis pipeline and produces a trade signal.

    How a signal is generated:
      1. Detect regime on H4 (bias) and H1 (structure).
      2. Check session context — avoid low-liquidity windows if configured.
      3. Detect candlestick and chart patterns on H1.
      4. Find S/R levels on H1.
      5. Check timeframe confluence across H4, H1, M15.
      6. Run ML models (direction, regime, confidence) if loaded.
      7. Compute composite confidence score.
      8. Apply min_confidence threshold and regime filter.
      9. Return TradeSignal with full reasoning.

    Configuration:
      All thresholds (min_confidence, min_tf_agreement, etc.) come from
      the Settings object — never hardcoded here.
    """

    def __init__(
        self,
        settings,
        regime_detector,
        sr_detector,
        candle_detector,
        chart_detector,
        tf_analyzer: TimeframeAnalyzer,
        session_ctx,
        model_inference,
        confidence_scorer,
        online_learner=None,
    ):
        self._settings      = settings
        self._regime        = regime_detector
        self._sr            = sr_detector
        self._candle        = candle_detector
        self._chart         = chart_detector
        self._tf_analyzer   = tf_analyzer
        self._session       = session_ctx
        self._inference     = model_inference
        self._scorer        = confidence_scorer
        self._online_learner = online_learner

    def generate_signal(
        self,
        symbol:       str,
        ohlcv_by_tf:  dict,
    ) -> TradeSignal:
        """Run the full analysis pipeline for one symbol.

        Args:
            symbol:      Instrument to analyse (e.g. 'EURUSD').
            ohlcv_by_tf: Dict of {timeframe_str: OHLCV DataFrame}.

        Returns:
            TradeSignal — always returns one (may be NO_TRADE with reasoning).
        """
        reasoning: dict = {"symbol": symbol, "steps": {}}

        # Step 1: Regime detection
        h4_df = ohlcv_by_tf.get("H4")
        h1_df = ohlcv_by_tf.get("H1")

        h4_regime = self._regime.detect(h4_df) if h4_df is not None else None
        h1_regime = self._regime.detect(h1_df) if h1_df is not None else None

        reasoning["steps"]["regime_h4"] = {
            "regime":     h4_regime.regime.value  if h4_regime else "unknown",
            "confidence": h4_regime.confidence     if h4_regime else 0.0,
            "reasoning":  h4_regime.reasoning      if h4_regime else "",
        }
        reasoning["steps"]["regime_h1"] = {
            "regime":     h1_regime.regime.value  if h1_regime else "unknown",
            "confidence": h1_regime.confidence     if h1_regime else 0.0,
        }

        # Step 2: Session context
        session_info = self._session.get_current_session()
        reasoning["steps"]["session"] = {
            "active":    session_info.active_sessions,
            "overlap":   session_info.overlap_name,
            "volatility": session_info.volatility_profile,
        }

        if self._session.is_avoided_window():
            return self._no_trade(symbol, "In avoided trading window.", reasoning)

        # Step 3: Pattern detection on H1
        candle_signals = self._candle.detect_all(h1_df) if h1_df is not None else []
        chart_patterns = self._chart.detect_all(h1_df)  if h1_df is not None else []
        sr_levels      = self._sr.detect(h1_df)          if h1_df is not None else []

        reasoning["steps"]["patterns"] = {
            "candlestick": [{"pattern": s.pattern, "direction": s.direction, "conf": s.confidence} for s in candle_signals],
            "chart":       [{"type": p.pattern_type, "direction": p.direction, "conf": p.confidence} for p in chart_patterns],
            "sr_levels":   [{"price": l.price, "type": l.level_type, "strength": l.strength} for l in sr_levels],
        }

        # Step 4: Timeframe confluence
        tf_signals: dict[str, str] = {}
        for tf in ["H4", "H1", "M15"]:
            df = ohlcv_by_tf.get(tf)
            if df is None:
                continue
            regime = self._regime.detect(df)
            if regime.regime == RegimeType.UPTREND:
                tf_signals[tf] = "bullish"
            elif regime.regime == RegimeType.DOWNTREND:
                tf_signals[tf] = "bearish"

        tf_agreement = self._tf_analyzer.align_signals(tf_signals)
        reasoning["steps"]["tf_agreement"] = tf_agreement

        if tf_agreement["agreement_count"] < self._settings.min_timeframe_agreement:
            return self._no_trade(
                symbol,
                f"Insufficient TF agreement: {tf_agreement['agreement_count']}"
                f" < {self._settings.min_timeframe_agreement} required.",
                reasoning,
            )

        # Step 5: ML model inference
        ml_outputs: dict = {}
        if self._inference is not None and self._inference.are_models_loaded():
            try:
                from ml.features import FeatureEngineer
                engineer = FeatureEngineer()
                features = engineer.build_features(
                    ohlcv_by_tf, h1_regime, session_info,
                    candle_signals, sr_levels, chart_patterns,
                )
                ml_outputs["direction_prob"] = self._inference.predict_direction(features)
                ml_outputs["regime_ml"]      = self._inference.predict_regime(features)
                ml_outputs["win_prob"]       = self._inference.predict_confidence(features)
            except Exception as exc:
                logger.warning("ML inference error (continuing without ML): {}", exc)
                ml_outputs = {}

        reasoning["steps"]["ml"] = ml_outputs

        # Step 6: Composite confidence scoring
        all_patterns = list(candle_signals) + list(chart_patterns)
        confidence = self._scorer.score(h1_regime, all_patterns, tf_agreement, ml_outputs)
        reasoning["confidence"] = confidence

        if confidence < self._settings.min_signal_confidence:
            return self._no_trade(
                symbol,
                f"Confidence {confidence:.2f} below threshold {self._settings.min_signal_confidence}.",
                reasoning,
            )

        # Step 7: Determine direction from dominant TF agreement
        dominant = tf_agreement["dominant_direction"]
        if dominant == "bullish":
            signal_type = SignalType.LONG
        elif dominant == "bearish":
            signal_type = SignalType.SHORT
        else:
            return self._no_trade(symbol, "No dominant direction from TF agreement.", reasoning)

        # Step 8: Online continuous learning adaptive gate
        if self._online_learner is not None:
            cand_names = [s.pattern for s in candle_signals] if candle_signals else []
            h4_reg_str = reasoning["steps"].get("regime_h4", {}).get("regime", "unknown")
            h1_reg_str = reasoning["steps"].get("regime_h1", {}).get("regime", "unknown")
            sess_str   = str(session_info.active_sessions)
            sess_vol   = session_info.volatility_profile

            eval_res = self._online_learner.evaluate_setup(
                symbol=symbol,
                direction=signal_type.value,
                h4_regime=h4_reg_str,
                h1_regime=h1_reg_str,
                session=sess_str,
                session_volatility=sess_vol,
                patterns=cand_names,
                confidence=confidence,
                tf_agreement=tf_agreement.get("agreement_score", 0.67),
            )
            reasoning["steps"]["online_learning"] = eval_res.to_dict()

            if not eval_res.is_allowed:
                return self._no_trade(symbol, f"Online learner veto: {eval_res.reason}", reasoning)

            if eval_res.confidence_multiplier != 1.0:
                old_conf = confidence
                confidence = float(np.clip(confidence * eval_res.confidence_multiplier, 0.0, 1.0))
                reasoning["confidence"] = confidence
                logger.info(
                    "Online learner adjusted confidence for {} {}: {:.2f} -> {:.2f} (mult: {:.2f}x, action: {})",
                    symbol, signal_type.value, old_conf, confidence, eval_res.confidence_multiplier, eval_res.action,
                )

            if confidence < self._settings.min_signal_confidence:
                return self._no_trade(
                    symbol,
                    f"Confidence {confidence:.2f} after online learner penalty ({eval_res.confidence_multiplier:.2f}x) below threshold {self._settings.min_signal_confidence}.",
                    reasoning,
                )

        logger.info(
            "Signal generated: {} {} | confidence={:.2f} | regime={} | TF agreement={}",
            signal_type.value, symbol, confidence,
            h1_regime.regime.value if h1_regime else "unknown",
            tf_agreement["dominant_direction"],
        )

        return TradeSignal(
            signal=signal_type,
            symbol=symbol,
            timeframe="H1",
            confidence=confidence,
            reasoning=reasoning,
        )

    @staticmethod
    def _no_trade(symbol: str, reason: str, reasoning: dict) -> TradeSignal:
        """Return a NO_TRADE signal with a logged reason."""
        reasoning["no_trade_reason"] = reason
        logger.debug("NO_TRADE for {}: {}", symbol, reason)
        return TradeSignal(
            signal=SignalType.NO_TRADE,
            symbol=symbol,
            timeframe="H1",
            confidence=0.0,
            reasoning=reasoning,
        )
