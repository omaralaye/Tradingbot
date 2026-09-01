"""
tests/test_regime_detection.py
--------------------------------
Unit tests for RegimeDetector. Regime classification is core signal logic
and must be verifiable with synthetic data.
"""

import numpy as np
import pandas as pd
import pytest

from analysis.regime_detection import RegimeDetector, RegimeType


def make_ohlcv(prices: list[float]) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame from a list of close prices."""
    n = len(prices)
    return pd.DataFrame({
        "open":   prices,
        "high":   [p * 1.001 for p in prices],
        "low":    [p * 0.999 for p in prices],
        "close":  prices,
        "volume": [1000.0] * n,
    })


def make_uptrend_df(n: int = 100) -> pd.DataFrame:
    """Steadily rising prices — should produce UPTREND."""
    prices = [1.0 + i * 0.005 for i in range(n)]
    return make_ohlcv(prices)


def make_flat_df(n: int = 100) -> pd.DataFrame:
    """Oscillating flat prices — should produce RANGING."""
    prices = [1.1 + 0.001 * np.sin(i * 0.5) for i in range(n)]
    return make_ohlcv(prices)


def test_uptrend_detected():
    """A steadily rising price series should be classified as UPTREND."""
    df      = make_uptrend_df(100)
    detector = RegimeDetector(adx_period=10, ema_period=20)
    result   = detector.detect(df)
    assert result.regime == RegimeType.UPTREND, (
        f"Expected UPTREND, got {result.regime}. Reasoning: {result.reasoning}"
    )


def test_ranging_detected():
    """A flat/oscillating price series should be classified as RANGING."""
    df      = make_flat_df(100)
    detector = RegimeDetector(adx_period=10, adx_range_threshold=30, ema_period=20)
    result   = detector.detect(df)
    assert result.regime in (RegimeType.RANGING, RegimeType.CHOPPY), (
        f"Expected RANGING or CHOPPY, got {result.regime}"
    )


def test_confidence_between_0_and_1():
    """Confidence score must always be in [0, 1]."""
    for df in [make_uptrend_df(), make_flat_df()]:
        detector = RegimeDetector(adx_period=10, ema_period=20)
        result   = detector.detect(df)
        assert 0.0 <= result.confidence <= 1.0, (
            f"Confidence {result.confidence} out of bounds."
        )


def test_result_has_reasoning_string():
    """RegimeResult.reasoning must be a non-empty string explaining the classification."""
    df      = make_uptrend_df()
    detector = RegimeDetector(adx_period=10, ema_period=20)
    result   = detector.detect(df)
    assert isinstance(result.reasoning, str)
    assert len(result.reasoning) > 0, "Reasoning string is empty."
