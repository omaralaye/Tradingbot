"""
tests/test_position_sizing.py
------------------------------
Unit tests for PositionSizer. These cover the core math and boundary conditions.
Risk/position-size calculations are critical — errors here directly cost money.
"""

import pytest
from risk.position_sizing import PositionSizer

INSTRUMENT = {
    "min_lot": 0.01,
    "max_lot": 10.0,
    "lot_step": 0.01,
    "risk_multiplier": 1.0,
}


def test_basic_calculation():
    """1% of $10,000 at 20 pips SL with $10/pip/lot => 0.5 lots."""
    sizer = PositionSizer(risk_pct=1.0)
    lot = sizer.calculate_lot_size(
        account_balance=10_000.0,
        stop_loss_pips=20.0,
        pip_value=10.0,
        instrument_config=INSTRUMENT,
    )
    # risk_amount = 100, lot = 100 / (20 * 10) = 0.5
    assert lot == pytest.approx(0.5, abs=0.01)


def test_lot_clamped_to_max():
    """Very small SL should not produce a lot above max_lot."""
    sizer = PositionSizer(risk_pct=5.0)
    instrument = {**INSTRUMENT, "max_lot": 2.0}
    lot = sizer.calculate_lot_size(
        account_balance=100_000.0,
        stop_loss_pips=1.0,
        pip_value=10.0,
        instrument_config=instrument,
    )
    assert lot <= 2.0


def test_lot_clamped_to_min():
    """Very large SL relative to small account should not produce lot below min_lot."""
    sizer = PositionSizer(risk_pct=1.0)
    lot = sizer.calculate_lot_size(
        account_balance=100.0,
        stop_loss_pips=500.0,
        pip_value=10.0,
        instrument_config=INSTRUMENT,
    )
    assert lot >= INSTRUMENT["min_lot"]


def test_lot_step_rounding():
    """Lot size must be rounded down to the nearest lot_step (0.01)."""
    sizer = PositionSizer(risk_pct=1.0)
    # Produces a fractional lot that needs rounding
    lot = sizer.calculate_lot_size(
        account_balance=10_000.0,
        stop_loss_pips=33.0,
        pip_value=10.0,
        instrument_config=INSTRUMENT,
    )
    # lot = 100 / 330 ≈ 0.303 => rounds down to 0.30
    remainder = round(lot * 100) % 1
    assert remainder == pytest.approx(0, abs=0.001), f"Lot {lot} not rounded to lot_step 0.01"


def test_zero_sl_raises_error():
    """A stop loss of zero must raise ValueError — no trade without a defined SL."""
    sizer = PositionSizer(risk_pct=1.0)
    with pytest.raises(ValueError, match="stop_loss_pips must be > 0"):
        sizer.calculate_lot_size(
            account_balance=10_000.0,
            stop_loss_pips=0.0,
            pip_value=10.0,
            instrument_config=INSTRUMENT,
        )
