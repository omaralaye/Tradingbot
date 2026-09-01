"""
risk/position_sizing.py
------------------------
Calculates lot size from account balance, risk percentage, stop-loss distance,
and per-instrument constraints.

IMPORTANT: Risk % comes from settings (user-configured). A fixed lot size
is never used unless explicitly reviewed and approved by the user.
"""

from __future__ import annotations

import math

from loguru import logger


class PositionSizer:
    """Calculates the correct lot size for a trade.

    Formula:
        risk_amount = account_balance * (risk_pct / 100)
        lot_size    = risk_amount / (stop_loss_pips * pip_value_per_lot)

    The result is then clamped to instrument min/max and rounded to lot_step.
    """

    def __init__(
        self,
        risk_pct:  float,
        min_lot:   float = 0.01,
        max_lot:   float = 10.0,
    ):
        """
        Args:
            risk_pct: Percentage of balance to risk per trade (e.g. 1.0 = 1%).
            min_lot:  Minimum lot size allowed.
            max_lot:  Maximum lot size allowed.
        """
        if risk_pct <= 0 or risk_pct > 100:
            raise ValueError(f"risk_pct must be between 0 and 100, got {risk_pct}.")
        self._risk_pct = risk_pct
        self._min_lot  = min_lot
        self._max_lot  = max_lot

    def calculate_lot_size(
        self,
        account_balance:  float,
        stop_loss_pips:   float,
        pip_value:        float,
        instrument_config: dict,
    ) -> float:
        """Calculate position lot size.

        Args:
            account_balance:   Current account balance in account currency.
            stop_loss_pips:    Stop loss distance in pips.
            pip_value:         Value of 1 pip per 1 standard lot in account currency.
            instrument_config: Instrument dict from instruments.yaml
                               (has min_lot, max_lot, lot_step, risk_multiplier).

        Returns:
            Calculated lot size, clamped and rounded to instrument constraints.

        Raises:
            ValueError: If stop_loss_pips <= 0 (undefined stop loss).
        """
        if stop_loss_pips <= 0:
            raise ValueError(
                f"stop_loss_pips must be > 0, got {stop_loss_pips}. "
                "Every trade must have a defined stop loss."
            )
        if pip_value <= 0:
            raise ValueError(f"pip_value must be > 0, got {pip_value}.")

        # Apply per-instrument risk multiplier
        risk_multiplier = float(instrument_config.get("risk_multiplier", 1.0))
        effective_risk  = self._risk_pct * risk_multiplier

        risk_amount = account_balance * (effective_risk / 100.0)
        raw_lot     = risk_amount / (stop_loss_pips * pip_value)

        clamped = self.validate_lot(raw_lot, instrument_config)

        logger.debug(
            "Position size | balance={:.2f} risk={:.2f}% SL={:.1f}pips "
            "pip_val={:.2f} => raw_lot={:.4f} => clamped={:.2f}",
            account_balance, effective_risk, stop_loss_pips, pip_value, raw_lot, clamped,
        )
        return clamped

    def validate_lot(self, lot: float, instrument_config: dict) -> float:
        """Clamp and round lot size to instrument constraints.

        Args:
            lot:               Raw calculated lot size.
            instrument_config: Dict with min_lot, max_lot, lot_step keys.

        Returns:
            Valid lot size for this instrument.
        """
        min_lot  = float(instrument_config.get("min_lot",  self._min_lot))
        max_lot  = float(instrument_config.get("max_lot",  self._max_lot))
        lot_step = float(instrument_config.get("lot_step", 0.01))

        # Round down to nearest lot_step
        steps     = math.floor(lot / lot_step)
        rounded   = steps * lot_step
        validated = float(max(min_lot, min(max_lot, rounded)))

        if validated < min_lot:
            logger.warning("Calculated lot {:.4f} below minimum {}; using minimum.", lot, min_lot)
            validated = min_lot

        return round(validated, 2)
