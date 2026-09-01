"""
live/run_bot.py
----------------
Main entry point for the trading bot.

⚠️  FINANCIAL RISK DISCLAIMER ⚠️
This bot executes real financial trades and CAN LOSE MONEY.
Past backtest performance does NOT guarantee future results.
Test extensively on a DEMO account before enabling live trading.
Never risk funds you cannot afford to lose.

Usage:
    python -m live.run_bot           # runs with demo account (default)
    ENABLE_LIVE_TRADING=true python -m live.run_bot  # live mode (requires explicit flag)
"""

import sys
import time
from datetime import datetime, timezone

import yaml
from loguru import logger

from config.settings import get_settings
from data.mt5_connector import MT5Connector
from data.data_fetcher import DataFetcher
from analysis.regime_detection import RegimeDetector
from analysis.timeframes import TimeframeAnalyzer
from analysis.session_context import SessionContext
from analysis.patterns.candlestick_patterns import CandlestickDetector
from analysis.patterns.chart_patterns import ChartPatternDetector
from analysis.patterns.support_resistance import SupportResistanceDetector
from signals.signal_engine import SignalEngine
from signals.confidence_scoring import ConfidenceScorer
from ml.inference import ModelInference
from risk.position_sizing import PositionSizer
from risk.risk_manager import RiskManager
from risk.stop_target_logic import StopTargetCalculator
from execution.order_manager import OrderManager
from execution.trade_journal import TradeJournal

# Timeframes to fetch for each symbol on every cycle
ACTIVE_TIMEFRAMES = ["M15", "H1", "H4", "D1"]
LOOP_SLEEP_SECONDS = 60


def print_disclaimer() -> None:
    """Print risk disclaimer. Required by AGENT.md Rule 5."""
    print("\n" + "=" * 70)
    print("  ⚠️   TRADING BOT RISK DISCLAIMER  ⚠️")
    print("=" * 70)
    print("  This bot executes real financial trades and can lose money.")
    print("  Past backtest performance does not guarantee future results.")
    print("  You are responsible for testing thoroughly on a demo account")
    print("  before enabling live trading.")
    print("  Never risk funds you cannot afford to lose.")
    print("=" * 70 + "\n")


def setup_logging(settings) -> None:
    """Configure loguru logging to file and stderr."""
    import os
    os.makedirs(settings.log_dir, exist_ok=True)
    logger.remove()  # remove default handler
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
    )
    log_file = f"{settings.log_dir}/bot_{datetime.now().strftime('%Y%m%d')}.log"
    logger.add(log_file, level="DEBUG", rotation="1 day", retention="30 days")
    logger.info("Logging initialised. Log file: {}", log_file)


def load_instruments(path: str = "config/instruments.yaml") -> list[dict]:
    """Load and filter enabled instruments from YAML config."""
    with open(path) as f:
        config = yaml.safe_load(f)
    enabled = [i for i in config.get("instruments", []) if i.get("enabled", True)]
    logger.info("Loaded {} enabled instruments.", len(enabled))
    return enabled


def load_sessions(path: str = "config/sessions.yaml") -> dict:
    """Load session windows from YAML config."""
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    """Main bot loop."""
    print_disclaimer()

    settings = get_settings()
    setup_logging(settings)

    # --- Live trading guard ---
    if settings.enable_live_trading:
        logger.warning(
            "⚠️  LIVE TRADING ENABLED. Real money is at risk. "
            "Ensure you have thoroughly tested on DEMO first."
        )
        print("\n⚠️  LIVE TRADING IS ENABLED — REAL MONEY AT RISK.\n")
    else:
        logger.info("Running in DEMO mode (ENABLE_LIVE_TRADING=false).")

    instruments   = load_instruments()
    sessions_cfg  = load_sessions()
    instrument_map = {i["symbol"]: i for i in instruments}

    # --- Instantiate all components ---
    connector    = MT5Connector(settings)
    fetcher      = DataFetcher(connector)

    regime_detector = RegimeDetector()
    sr_detector     = SupportResistanceDetector()
    candle_detector = CandlestickDetector()
    chart_detector  = ChartPatternDetector()
    tf_analyzer     = TimeframeAnalyzer()
    session_ctx     = SessionContext(sessions_cfg)
    model_inference = ModelInference()
    confidence_scorer = ConfidenceScorer()

    signal_engine = SignalEngine(
        settings, regime_detector, sr_detector, candle_detector,
        chart_detector, tf_analyzer, session_ctx, model_inference, confidence_scorer,
    )

    position_sizer = PositionSizer(settings.risk_per_trade_pct)
    stop_calculator = StopTargetCalculator()
    risk_manager    = RiskManager(settings)
    order_manager   = OrderManager(connector, settings)
    journal         = TradeJournal()

    # Register kill switch to close all positions
    risk_manager.register_kill_switch_callback(order_manager.close_all_positions)

    logger.info("All components initialised. Starting trading loop...")

    # --- Main loop ---
    with connector:
        account = order_manager.get_account_info()
        risk_manager.reset_daily_stats(account.get("balance", 0.0))
        logger.info("Account balance: {:.2f} {}", account.get("balance", 0.0), account.get("currency", "DEMO"))

        try:
            while True:
                loop_start = datetime.now(timezone.utc)
                logger.info("--- Cycle start: {} ---", loop_start.strftime("%Y-%m-%d %H:%M:%S UTC"))

                if risk_manager.is_kill_switch_active:
                    logger.critical("Kill switch active — loop halted. Restart bot to resume.")
                    break

                open_positions = order_manager.get_open_positions()

                for instr in instruments:
                    symbol = instr["symbol"]
                    try:
                        # Fetch multi-timeframe data
                        ohlcv = fetcher.fetch_multi_timeframe(symbol, ACTIVE_TIMEFRAMES)
                        if not ohlcv:
                            logger.warning("No data fetched for {}. Skipping.", symbol)
                            continue

                        # Generate signal
                        signal = signal_engine.generate_signal(symbol, ohlcv)

                        if signal.signal.value == "NO_TRADE":
                            journal.log_no_trade(
                                symbol,
                                reason=signal.reasoning.get("no_trade_reason", "No signal"),
                                confidence=signal.confidence,
                            )
                            continue

                        # Risk check
                        allowed, reason = risk_manager.check_can_trade(symbol, signal, open_positions)
                        if not allowed:
                            journal.log_signal(signal, was_traded=False, skip_reason=reason)
                            continue

                        # Position sizing and SL/TP
                        from analysis.indicators import add_atr
                        h1_df   = ohlcv.get("H1")
                        atr_val = float(add_atr(h1_df, 14).iloc[-1]) if h1_df is not None else 0.001
                        price   = float(h1_df["close"].iloc[-1]) if h1_df is not None else 0.0

                        sl_tp = stop_calculator.calculate_sl_tp(
                            signal, h1_df, atr_val,
                            sr_detector.detect(h1_df) if h1_df is not None else [],
                            price,
                        )

                        sl_pips = sl_tp["sl_distance"] / instr["pip_size"]
                        pip_val = instr.get("pip_value", 10.0)  # default $10/pip/lot for majors
                        lot     = position_sizer.calculate_lot_size(
                            account.get("balance", 0.0), sl_pips, pip_val, instr
                        )

                        # Place order
                        order_type = "buy" if signal.signal.value == "LONG" else "sell"
                        result = order_manager.place_market_order(
                            symbol, order_type, lot, sl_tp["stop_loss"], sl_tp["take_profit"]
                        )

                        journal.log_signal(signal, was_traded=result["success"])
                        journal.log_order(result, signal)

                        if result["success"]:
                            open_positions = order_manager.get_open_positions()  # refresh

                    except Exception as exc:
                        logger.error("Error processing {}: {}", symbol, exc)
                        continue

                elapsed = (datetime.now(timezone.utc) - loop_start).total_seconds()
                sleep_time = max(0, LOOP_SLEEP_SECONDS - elapsed)
                logger.info("Cycle complete in {:.1f}s. Sleeping {:.0f}s...", elapsed, sleep_time)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Shutting down cleanly...")
        finally:
            logger.info("Bot stopped. Journal saved to: {}", journal.export_csv())
            print("\nBot stopped. See logs/ for full session log.")


if __name__ == "__main__":
    main()
