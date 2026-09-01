"""
signals/confidence_scoring.py
------------------------------
Computes a weighted composite confidence score from all analysis components.

The score is used by the signal engine to decide whether to take a trade.
All weights are configurable so the user can tune importance of each component.
"""

from __future__ import annotations

from loguru import logger


# Default component weights — must sum to 1.0
DEFAULT_WEIGHTS = {
    "regime_confidence":  0.25,   # how confident the regime detector is
    "pattern_confidence": 0.20,   # best candlestick/chart pattern confidence
    "tf_agreement":       0.25,   # fraction of timeframes agreeing on direction
    "ml_direction_prob":  0.15,   # ML direction model probability
    "ml_confidence_prob": 0.15,   # ML confidence model win probability
}


class ConfidenceScorer:
    """Computes a weighted composite confidence score.

    Each component is normalised to [0, 1] before weighting, so the
    output is always in [0, 1].

    Example:
        scorer = ConfidenceScorer()
        score  = scorer.score(regime_result, candle_signals, tf_agreement, ml_outputs)
    """

    def __init__(self, weights: dict | None = None):
        """
        Args:
            weights: Custom weight dict. Must have the same keys as DEFAULT_WEIGHTS.
                     Weights do not need to sum to 1.0 — they are normalised internally.
        """
        raw = weights or DEFAULT_WEIGHTS
        total = sum(raw.values())
        self._weights = {k: v / total for k, v in raw.items()}  # normalise

    def score(
        self,
        regime_result,
        pattern_signals: list,
        tf_agreement:    dict,
        ml_outputs:      dict,
    ) -> float:
        """Compute the composite confidence score.

        Args:
            regime_result:   RegimeResult (or None).
            pattern_signals: List of CandleSignal / ChartPattern objects.
            tf_agreement:    Output of TimeframeAnalyzer.align_signals().
            ml_outputs:      Dict with keys 'direction_prob' (dict) and 'win_prob' (float).

        Returns:
            Float in [0.0, 1.0].
        """
        components = self.get_component_scores(regime_result, pattern_signals, tf_agreement, ml_outputs)
        composite = sum(
            components[k] * self._weights.get(k, 0.0)
            for k in components
        )
        composite = float(max(0.0, min(1.0, composite)))
        logger.debug("Confidence components: {} => composite={:.3f}", components, composite)
        return composite

    def get_component_scores(
        self,
        regime_result,
        pattern_signals: list,
        tf_agreement:    dict,
        ml_outputs:      dict,
    ) -> dict[str, float]:
        """Return individual component scores before weighting.

        Useful for logging and explainability — the trade journal records these
        so every decision is auditable.
        """
        # 1. Regime confidence
        regime_conf = float(regime_result.confidence) if regime_result else 0.5

        # 2. Pattern confidence — best confidence across all detected patterns
        if pattern_signals:
            pattern_conf = max(getattr(s, "confidence", 0.0) for s in pattern_signals)
        else:
            pattern_conf = 0.0

        # 3. Timeframe agreement score
        tf_score = float(tf_agreement.get("agreement_score", 0.0))

        # 4. ML direction probability — probability of the dominant direction
        direction_probs = ml_outputs.get("direction_prob", {})
        if direction_probs:
            ml_dir_prob = max(
                direction_probs.get("up", 0.0),
                direction_probs.get("down", 0.0),
            )
        else:
            ml_dir_prob = 0.5  # neutral default when model not loaded

        # 5. ML win probability from confidence model
        ml_win_prob = float(ml_outputs.get("win_prob", 0.5))

        return {
            "regime_confidence":  regime_conf,
            "pattern_confidence": float(pattern_conf),
            "tf_agreement":       tf_score,
            "ml_direction_prob":  ml_dir_prob,
            "ml_confidence_prob": ml_win_prob,
        }
