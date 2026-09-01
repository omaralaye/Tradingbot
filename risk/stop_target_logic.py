"""
risk/stop_target_logic.py
--------------------------
Calculates Stop Loss and Take Profit levels using either:
  - Structure-based: place SL beyond the nearest S/R level
  - ATR-based fallback: SL = entry ± ATR * multiplier

SL/TP are always required for every trade — this module enforces that.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from analysis.patterns.support_resistance import SRLevel
from signals.signal_engine import SignalType, TradeSignal


class StopTargetCalculator:
    """Computes SL and TP prices for a given trade signal.

    Stop loss placement priority:
      1. Structure-based: just beyond the nearest S/R level (preferred).
      2. ATR-based fallback: if no nearby S/R level or structure is not
         within a reasonable distance.

    Take Profit is always set based on the risk-reward ratio applied to SL distance.
    """

    def __init__(
        self,
        atr_sl_multiplier: float = 1.5,
        atr_tp_multiplier: float = 2.5,
        use_structure:     bool  = True,
        max_sr_distance_atr: float = 3.0,  # max ATR multiples to look for S/R
    ):
        """
        Args:
            atr_sl_multiplier:     SL = price ± (ATR * this) for ATR-based SL.
            atr_tp_multiplier:     TP = price ± (ATR * this) for ATR-based TP.
            use_structure:         Prefer structure-based SL over ATR when possible.
            max_sr_distance_atr:   Ignore S/R levels further than this many ATR units.
        """
        self._atr_sl  = atr_sl_multiplier
        self._atr_tp  = atr_tp_multiplier
        self._use_sr  = use_structure
        self._max_sr  = max_sr_distance_atr

    def calculate_sl_tp(
        self,
        signal:        TradeSignal,
        df:            pd.DataFrame,
        atr:           float,
        sr_levels:     list[SRLevel],
        current_price: float,
    ) -> dict:
        """Calculate SL and TP for a trade signal.

        Args:
            signal:        The TradeSignal (LONG or SHORT).
            df:            OHLCV DataFrame for recent structure reference.
            atr:           Current ATR value.
            sr_levels:     Detected S/R levels from SupportResistanceDetector.
            current_price: Current market price (entry price).

        Returns:
            Dict with keys: stop_loss, take_profit, sl_pips, tp_pips, risk_reward, method.

        Raises:
            ValueError: If signal is not LONG or SHORT.
        """
        if signal.signal not in (SignalType.LONG, SignalType.SHORT):
            raise ValueError(f"Cannot calculate SL/TP for {signal.signal} signal.")

        is_long = signal.signal == SignalType.LONG

        # Try structure-based SL first
        sl_price = None
        method   = "atr"

        if self._use_sr and sr_levels:
            sl_price = self._structure_sl(is_long, sr_levels, current_price, atr)
            if sl_price is not None:
                method = "structure"

        # Fallback to ATR-based SL
        if sl_price is None:
            sl_price = self._atr_sl_price(is_long, current_price, atr)

        tp_price    = self._calculate_tp(is_long, current_price, sl_price)
        sl_distance = abs(current_price - sl_price)
        tp_distance = abs(tp_price - current_price)
        rr          = tp_distance / sl_distance if sl_distance > 0 else 0.0

        result = {
            "stop_loss":   round(sl_price, 5),
            "take_profit": round(tp_price, 5),
            "sl_distance": round(sl_distance, 5),
            "tp_distance": round(tp_distance, 5),
            "risk_reward": round(rr, 2),
            "method":      method,
        }

        logger.debug(
            "SL/TP | {} {} | entry={:.5f} SL={:.5f} TP={:.5f} RR={:.2f} ({})",
            signal.signal.value, signal.symbol,
            current_price, sl_price, tp_price, rr, method,
        )
        return result

    # ------------------------------------------------------------------
    # SL calculation methods
    # ------------------------------------------------------------------

    def _structure_sl(
        self,
        is_long:       bool,
        sr_levels:     list[SRLevel],
        current_price: float,
        atr:           float,
    ) -> Optional[float]:
        """Place SL just beyond the nearest relevant S/R level.

        For LONG: SL just below nearest support.
        For SHORT: SL just above nearest resistance.
        """
        max_dist = atr * self._max_sr
        buffer   = atr * 0.1  # small buffer beyond the S/R level

        if is_long:
            candidates = [
                l for l in sr_levels
                if l.level_type == "support"
                and l.price < current_price
                and (current_price - l.price) <= max_dist
            ]
            if candidates:
                nearest = max(candidates, key=lambda x: x.price)
                return nearest.price - buffer
        else:
            candidates = [
                l for l in sr_levels
                if l.level_type == "resistance"
                and l.price > current_price
                and (l.price - current_price) <= max_dist
            ]
            if candidates:
                nearest = min(candidates, key=lambda x: x.price)
                return nearest.price + buffer

        return None

    def _atr_sl_price(self, is_long: bool, entry: float, atr: float) -> float:
        """ATR-based stop loss price."""
        if is_long:
            return entry - atr * self._atr_sl
        return entry + atr * self._atr_sl

    def _calculate_tp(self, is_long: bool, entry: float, sl_price: float) -> float:
        """Calculate TP based on risk-reward ratio relative to SL distance."""
        sl_dist = abs(entry - sl_price)
        tp_dist = sl_dist * (self._atr_tp / self._atr_sl)
        if is_long:
            return entry + tp_dist
        return entry - tp_dist
