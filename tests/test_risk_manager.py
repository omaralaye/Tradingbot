"""
tests/test_risk_manager.py
---------------------------
Unit tests for RiskManager. Risk rules are safety-critical — failures here
can result in real financial losses.
"""

import pytest
from unittest.mock import MagicMock

from risk.risk_manager import RiskManager


def make_settings(max_daily_loss=3.0, max_positions=5):
    s = MagicMock()
    s.max_daily_loss_pct   = max_daily_loss
    s.max_open_positions   = max_positions
    s.risk_per_trade_pct   = 1.0
    return s


def make_signal(direction="LONG"):
    sig = MagicMock()
    sig.signal.value = direction
    return sig


def test_kill_switch_blocks_trading():
    """Once the kill switch is triggered, check_can_trade must return False."""
    rm = RiskManager(make_settings())
    rm.trigger_kill_switch("Test kill switch")
    allowed, reason = rm.check_can_trade("EURUSD", make_signal(), open_positions=[])
    assert allowed is False
    assert "Kill switch" in reason


def test_daily_loss_limit_triggers_kill_switch():
    """Exceeding the daily loss limit should activate the kill switch."""
    rm = RiskManager(make_settings(max_daily_loss=3.0))
    rm.reset_daily_stats(current_balance=10_000.0)
    # Simulate a 4% loss ($400 on $10,000 starting balance)
    rm.update_daily_pnl(-400.0)
    allowed, reason = rm.check_can_trade("EURUSD", make_signal(), open_positions=[])
    assert allowed is False
    assert rm.is_kill_switch_active is True


def test_max_positions_blocks_new_trade():
    """When max_open_positions is reached, no new trades should be allowed."""
    rm = RiskManager(make_settings(max_positions=3))
    fake_positions = [{"symbol": "EURUSD"}, {"symbol": "GBPUSD"}, {"symbol": "USDJPY"}]
    allowed, reason = rm.check_can_trade("AUDUSD", make_signal(), open_positions=fake_positions)
    assert allowed is False
    assert "Max open positions" in reason


def test_reset_daily_stats_clears_pnl():
    """reset_daily_stats should zero out daily PnL and update start balance."""
    rm = RiskManager(make_settings())
    rm.reset_daily_stats(current_balance=10_000.0)  # must set start balance before tracking PnL
    rm.update_daily_pnl(-200.0)
    assert rm.daily_loss_pct > 0

    rm.reset_daily_stats(current_balance=9_800.0)
    assert rm.daily_loss_pct == 0.0
    assert rm._day_start_balance == 9_800.0
