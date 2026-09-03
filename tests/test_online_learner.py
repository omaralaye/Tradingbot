"""
tests/test_online_learner.py
----------------------------
Comprehensive test suite for the Online Continuous Learning Loop & Adaptive Feedback Brain:
  1. Feature extraction and setup fingerprinting
  2. Bayesian win probability and dynamic scoring
  3. Incremental classifier learning via partial_fit
  4. Quarantine triggering on consecutive losses and vetoing
  5. Winning setup confidence boosting
  6. Persistence save/load across restarts
  7. Closed-position syncing from MT5 deal history
  8. TradeJournal realized trade logging
  9. Symbol cooldown enforcement in RiskManager
  10. Stop loss floor enforcement in StopTargetCalculator
  11. SignalEngine online learner integration
"""

import os
import time
from unittest.mock import MagicMock

import pandas as pd
import pytest

from execution.order_manager import OrderManager
from execution.trade_journal import TradeJournal
from ml.online_learner import OnlineAdaptiveLearner, SetupEvaluation
from risk.risk_manager import RiskManager
from risk.stop_target_logic import StopTargetCalculator
from signals.signal_engine import SignalEngine, SignalType, TradeSignal


@pytest.fixture
def tmp_memory_path(tmp_path):
    return str(tmp_path / "test_online_memory.json")


def test_online_learner_initial_evaluation(tmp_memory_path):
    """Initial evaluation with no trade history should allow trade with neutral scores."""
    learner = OnlineAdaptiveLearner(memory_path=tmp_memory_path)
    ev = learner.evaluate_setup(
        symbol="EURUSD",
        direction="LONG",
        h4_regime="uptrend",
        h1_regime="uptrend",
        session="london",
        confidence=0.70,
    )
    assert ev.is_allowed is True
    assert ev.action == "ALLOW"
    assert ev.confidence_multiplier == 1.0
    assert ev.bayesian_win_prob == 0.50
    assert ev.is_quarantined is False


def test_online_learner_quarantine_on_consecutive_losses(tmp_memory_path):
    """After consecutive losses equal to streak threshold, setup must be quarantined."""
    learner = OnlineAdaptiveLearner(
        memory_path=tmp_memory_path,
        quarantine_loss_streak=2,
        quarantine_duration_seconds=3600,
    )

    # First loss
    res1 = learner.on_trade_closed({
        "ticket": 1001,
        "symbol": "EURUSD",
        "direction": "LONG",
        "net_pnl": -1.50,
        "h4_regime": "downtrend",
        "h1_regime": "uptrend",
    })
    assert res1["outcome"] == "loss"
    assert res1["quarantine_triggered"] is False

    # Second loss -> should trigger quarantine
    res2 = learner.on_trade_closed({
        "ticket": 1002,
        "symbol": "EURUSD",
        "direction": "LONG",
        "net_pnl": -1.20,
        "h4_regime": "downtrend",
        "h1_regime": "uptrend",
    })
    assert res2["quarantine_triggered"] is True

    # Next evaluation must be VETOED
    ev = learner.evaluate_setup(
        symbol="EURUSD",
        direction="LONG",
        h4_regime="downtrend",
        h1_regime="uptrend",
    )
    assert ev.is_allowed is False
    assert ev.action == "VETO"
    assert ev.confidence_multiplier == 0.0
    assert ev.is_quarantined is True
    assert "quarantined" in ev.quarantine_reason.lower()


def test_online_learner_winning_setup_boost(tmp_memory_path):
    """Proven winning setup with consecutive wins should receive a confidence boost."""
    learner = OnlineAdaptiveLearner(
        memory_path=tmp_memory_path,
        min_trades_for_stats=2,
    )

    # Record 3 solid wins for USDJPY trend trade
    for i in range(3):
        learner.on_trade_closed({
            "ticket": 2000 + i,
            "symbol": "USDJPY",
            "direction": "SHORT",
            "net_pnl": 2.50,
            "h4_regime": "downtrend",
            "h1_regime": "downtrend",
            "confidence": 0.75,
        })

    ev = learner.evaluate_setup(
        symbol="USDJPY",
        direction="SHORT",
        h4_regime="downtrend",
        h1_regime="downtrend",
        confidence=0.75,
    )
    assert ev.is_allowed is True
    assert ev.bayesian_win_prob > 0.60
    assert ev.confidence_multiplier >= 1.0


def test_online_learner_persistence(tmp_memory_path):
    """Memory must save and reload state accurately across restarts."""
    learner1 = OnlineAdaptiveLearner(memory_path=tmp_memory_path, quarantine_loss_streak=2)
    learner1.on_trade_closed({
        "ticket": 3001,
        "symbol": "GBPUSD",
        "direction": "SHORT",
        "net_pnl": 3.00,
        "h4_regime": "downtrend",
    })

    # Instantiate second learner from same memory file
    learner2 = OnlineAdaptiveLearner(memory_path=tmp_memory_path, quarantine_loss_streak=2)
    summary = learner2.get_summary()

    assert summary["total_processed_trades"] == 1
    assert "GBPUSD:SHORT" in learner2._setup_stats
    assert learner2._setup_stats["GBPUSD:SHORT"]["wins"] == 1
    assert learner2._setup_stats["GBPUSD:SHORT"]["net_pnl"] == 3.00


def test_sync_closed_positions_pairing():
    """OrderManager.sync_closed_positions must pair entry & exit deals and calculate net PnL."""
    connector = MagicMock()
    connector.is_connected.return_value = True

    # Mock deals: 1 open (position 10), 1 closed (position 20)
    mock_deal_entry = MagicMock()
    mock_deal_entry._asdict.return_value = {
        "ticket": 101, "order": 1, "position_id": 20, "type": 0, "entry": 0,
        "symbol": "EURUSDm", "volume": 0.01, "price": 1.1600, "time": 1700000000,
        "profit": 0.0, "commission": -0.04, "swap": 0.0, "comment": "",
    }
    mock_deal_exit = MagicMock()
    mock_deal_exit._asdict.return_value = {
        "ticket": 102, "order": 2, "position_id": 20, "type": 1, "entry": 1,
        "symbol": "EURUSDm", "volume": 0.01, "price": 1.1630, "time": 1700003600,
        "profit": 3.00, "commission": -0.04, "swap": 0.0, "comment": "[tp 1.16300]",
    }

    connector.mt5.history_deals_get.return_value = [mock_deal_entry, mock_deal_exit]

    settings = MagicMock()
    settings.enable_live_trading = False
    om = OrderManager(connector, settings)

    closed = om.sync_closed_positions()
    assert len(closed) == 1
    c = closed[0]
    assert c["position_id"] == 20
    assert c["symbol"] == "EURUSD"
    assert c["net_pnl"] == 2.92  # 3.00 - 0.08 commission
    assert c["outcome"] == "win"
    assert c["exit_reason"] == "[tp 1.16300]"
    assert c["duration_minutes"] == 60.0

    # Calling again should not duplicate
    closed2 = om.sync_closed_positions()
    assert len(closed2) == 0


def test_trade_journal_log_trade_closed(tmp_path):
    """TradeJournal.log_trade_closed should record trade_closed row."""
    j_path = tmp_path / "test_journal.csv"
    journal = TradeJournal(str(j_path))

    closed_trade = {
        "ticket": 5555,
        "symbol": "USDJPY",
        "direction": "sell",
        "entry_price": 159.50,
        "exit_price": 159.00,
        "net_pnl": 4.50,
        "outcome": "win",
        "exit_reason": "[tp 159.000]",
        "timeframe": "H1",
    }
    journal.log_trade_closed(closed_trade)

    df = pd.read_csv(j_path)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["action"] == "trade_closed"
    assert row["symbol"] == "USDJPY"
    assert row["signal"] == "SHORT"
    assert float(row["pnl"]) == 4.50
    assert row["outcome"] == "win"


def test_risk_manager_symbol_cooldown():
    """RiskManager must block orders on a symbol within its cooldown window."""
    settings = MagicMock()
    settings.max_daily_loss_pct = 3.0
    settings.max_open_positions = 5
    settings.symbol_cooldown_seconds = 600  # 10 min

    rm = RiskManager(settings, symbol_cooldown_seconds=600)
    sig = MagicMock()
    sig.signal.value = "LONG"

    # Initially allowed
    allowed, reason = rm.check_can_trade("EURUSD", sig, [])
    assert allowed is True

    # Record order opened
    rm.record_order_opened("EURUSD")

    # Immediate second check must be blocked by cooldown
    allowed, reason = rm.check_can_trade("EURUSD", sig, [])
    assert allowed is False
    assert "Symbol cooldown active" in reason

    # Different symbol should not be blocked
    allowed_other, _ = rm.check_can_trade("USDJPY", sig, [])
    assert allowed_other is True


def test_stop_target_min_sl_floor():
    """StopTargetCalculator must clamp stop loss distance to min_stop_loss_pips."""
    calc = StopTargetCalculator(atr_sl_multiplier=1.5, min_stop_loss_pips=15.0)

    sig = TradeSignal(
        signal=SignalType.LONG,
        symbol="EURUSD",
        timeframe="H1",
        confidence=0.75,
    )
    df = pd.DataFrame({"close": [1.16000]})

    # With tiny ATR (0.0001 = 1 pip), calculated ATR SL would be 1.5 pips
    res = calc.calculate_sl_tp(
        signal=sig,
        df=df,
        atr=0.0001,
        sr_levels=[],
        current_price=1.16000,
    )

    # Floor is 15 pips = 0.00150
    assert res["sl_distance"] == pytest.approx(0.00150, abs=1e-5)
    assert res["stop_loss"] == pytest.approx(1.16000 - 0.00150, abs=1e-5)


def test_signal_engine_online_learner_veto():
    """SignalEngine must return NO_TRADE when OnlineAdaptiveLearner vetoes a setup."""
    settings = MagicMock()
    settings.min_signal_confidence = 0.60
    settings.min_timeframe_agreement = 2

    regime_det = MagicMock()
    regime_det.detect.return_value.regime.value = "uptrend"
    regime_det.detect.return_value.confidence = 0.90
    regime_det.detect.return_value.reasoning = "test"

    tf_analyzer = MagicMock()
    tf_analyzer.align_signals.return_value = {
        "dominant_direction": "bullish",
        "agreement_count": 3,
        "agreement_score": 1.0,
    }

    session_ctx = MagicMock()
    session_ctx.is_avoided_window.return_value = False
    session_ctx.get_current_session.return_value.active_sessions = ["london"]
    session_ctx.get_current_session.return_value.overlap_name = None
    session_ctx.get_current_session.return_value.volatility_profile = "high"

    scorer = MagicMock()
    scorer.score.return_value = 0.85

    # Mock online learner that vetoes the trade
    online_learner = MagicMock()
    online_learner.evaluate_setup.return_value = SetupEvaluation(
        is_allowed=False,
        action="VETO",
        confidence_multiplier=0.0,
        bayesian_win_prob=0.20,
        ml_win_prob=0.25,
        is_quarantined=True,
        quarantine_reason="Quarantined due to losses",
        reason="Quarantined due to losses",
    )

    engine = SignalEngine(
        settings=settings,
        regime_detector=regime_det,
        sr_detector=MagicMock(),
        candle_detector=MagicMock(),
        chart_detector=MagicMock(),
        tf_analyzer=tf_analyzer,
        session_ctx=session_ctx,
        model_inference=None,
        confidence_scorer=scorer,
        online_learner=online_learner,
    )

    df_dummy = pd.DataFrame({"close": [1.16, 1.161]})
    ohlcv = {"H4": df_dummy, "H1": df_dummy, "M15": df_dummy}

    sig = engine.generate_signal("EURUSD", ohlcv)
    assert sig.signal == SignalType.NO_TRADE
    assert "Online learner veto" in sig.reasoning.get("no_trade_reason", "")
