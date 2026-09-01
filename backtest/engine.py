"""
backtest/engine.py
------------------
Simulates the trading bot on historical OHLCV data.

⚠️ IMPORTANT: Backtesting runs in SIMULATION ONLY — no real MT5 orders
are placed. This is purely a historical performance evaluation tool.

Uses the same signal engine and risk manager as live trading to ensure
backtest results reflect real bot behaviour as closely as possible.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class BacktestEngine:
    """Bar-by-bar historical simulation of the trading bot.

    For each bar in the historical dataset:
      1. Feed data up to that bar into the signal engine.
      2. If a signal is generated and risk checks pass, simulate a trade.
      3. Scan future bars to determine trade outcome (TP/SL/timeout).
      4. Record result in the trade ledger.

    Limitations vs live trading:
      - No slippage simulation (can be added as a config parameter).
      - Uses close price as entry (real trades use next open or bid/ask).
      - No overnight swap/financing costs.
    """

    def __init__(
        self,
        signal_engine,
        risk_manager,
        position_sizer,
        stop_calculator,
        initial_balance: float = 10_000.0,
        pip_value:       float = 10.0,
        horizon_bars:    int   = 50,
    ):
        """
        Args:
            signal_engine:   SignalEngine instance.
            risk_manager:    RiskManager instance.
            position_sizer:  PositionSizer instance.
            stop_calculator: StopTargetCalculator instance.
            initial_balance: Starting virtual balance.
            pip_value:       Pip value per lot (simplification — for multi-symbol
                             use this should be per-instrument from instruments.yaml).
            horizon_bars:    Max bars to wait for trade resolution.
        """
        self._signal_engine   = signal_engine
        self._risk_manager    = risk_manager
        self._position_sizer  = position_sizer
        self._stop_calc       = stop_calculator
        self._initial_balance = initial_balance
        self._pip_value       = pip_value
        self._horizon         = horizon_bars
        self._trades: list[dict] = []

    def run(
        self,
        historical_data: dict[str, dict[str, pd.DataFrame]],
        symbols: list[str],
        start_date: Optional[datetime] = None,
        end_date:   Optional[datetime] = None,
    ) -> dict:
        """Run the backtest over all symbols and dates.

        Args:
            historical_data: {symbol: {timeframe: df}}
            symbols:         List of symbols to backtest.
            start_date:      Backtest start (inclusive).
            end_date:        Backtest end (inclusive).

        Returns:
            Dict with 'trades' (list), 'final_balance', 'total_return_pct'.
        """
        balance = self._initial_balance
        self._risk_manager.reset_daily_stats(balance)
        self._trades = []

        for symbol in symbols:
            symbol_data = historical_data.get(symbol, {})
            h1_df = symbol_data.get("H1")
            if h1_df is None or len(h1_df) < 100:
                logger.warning("Skipping {} — insufficient H1 data.", symbol)
                continue

            # Filter by date range
            if start_date:
                h1_df = h1_df[h1_df.index >= pd.Timestamp(start_date, tz="UTC")]
            if end_date:
                h1_df = h1_df[h1_df.index <= pd.Timestamp(end_date, tz="UTC")]

            logger.info("Backtesting {} | {} bars", symbol, len(h1_df))
            balance = self._run_symbol(symbol, symbol_data, h1_df, balance)

        total_return = (balance - self._initial_balance) / self._initial_balance * 100
        logger.info(
            "Backtest complete | trades={} | final_balance={:.2f} | return={:.2f}%",
            len(self._trades), balance, total_return,
        )

        return {
            "trades":          self._trades,
            "final_balance":   balance,
            "initial_balance": self._initial_balance,
            "total_return_pct": round(total_return, 2),
        }

    def get_results(self) -> pd.DataFrame:
        """Return all backtest trades as a DataFrame."""
        if not self._trades:
            return pd.DataFrame()
        return pd.DataFrame(self._trades)

    # ------------------------------------------------------------------
    # Internal simulation logic
    # ------------------------------------------------------------------

    def _run_symbol(
        self,
        symbol:      str,
        symbol_data: dict[str, pd.DataFrame],
        h1_df:       pd.DataFrame,
        balance:     float,
    ) -> float:
        """Run bar-by-bar simulation for one symbol."""
        open_positions: list[dict] = []

        for i in range(50, len(h1_df) - self._horizon):
            # Slice data up to current bar (no look-ahead)
            ohlcv_slice = {tf: df.iloc[:i] for tf, df in symbol_data.items()}

            signal = self._signal_engine.generate_signal(symbol, ohlcv_slice)

            if signal.signal.value == "NO_TRADE":
                continue

            allowed, reason = self._risk_manager.check_can_trade(symbol, signal, open_positions)
            if not allowed:
                continue

            # Use close of current bar as entry price
            current_price = float(h1_df["close"].iloc[i])
            from analysis.indicators import add_atr
            atr_series    = add_atr(ohlcv_slice.get("H1", h1_df), 14)
            atr_val       = float(atr_series.iloc[-1]) if not atr_series.empty else 0.001

            sl_tp = self._stop_calc.calculate_sl_tp(signal, h1_df.iloc[:i], atr_val, [], current_price)
            sl_pips = sl_tp["sl_distance"] / 0.0001  # simplified for forex

            lot = self._position_sizer.calculate_lot_size(
                balance, sl_pips, self._pip_value,
                {"min_lot": 0.01, "max_lot": 5.0, "lot_step": 0.01, "risk_multiplier": 1.0},
            )

            outcome = self._simulate_trade(signal, current_price, sl_tp, h1_df.iloc[i:i + self._horizon])
            pnl = outcome["pnl"] * lot

            balance += pnl
            self._risk_manager.update_daily_pnl(pnl)

            self._trades.append({
                "symbol":      symbol,
                "timestamp":   str(h1_df.index[i]),
                "signal":      signal.signal.value,
                "entry_price": current_price,
                "stop_loss":   sl_tp["stop_loss"],
                "take_profit": sl_tp["take_profit"],
                "lot":         lot,
                "outcome":     outcome["outcome"],
                "pnl":         round(pnl, 2),
                "balance":     round(balance, 2),
                "confidence":  signal.confidence,
                "rr":          sl_tp["risk_reward"],
            })

        return balance

    def _simulate_trade(
        self,
        signal,
        entry_price: float,
        sl_tp:       dict,
        future_bars: pd.DataFrame,
    ) -> dict:
        """Determine if price hits TP or SL in the future bars.

        Args:
            signal:      TradeSignal.
            entry_price: Trade entry price.
            sl_tp:       Dict with stop_loss, take_profit.
            future_bars: DataFrame of bars after entry.

        Returns:
            Dict with 'outcome' ('win'/'loss'/'timeout') and 'pnl' in price points.
        """
        sl = sl_tp["stop_loss"]
        tp = sl_tp["take_profit"]
        is_long = signal.signal.value == "LONG"

        for _, bar in future_bars.iterrows():
            if is_long:
                if bar["high"] >= tp:
                    return {"outcome": "win",  "pnl": tp - entry_price}
                if bar["low"]  <= sl:
                    return {"outcome": "loss", "pnl": sl - entry_price}
            else:
                if bar["low"]  <= tp:
                    return {"outcome": "win",  "pnl": entry_price - tp}
                if bar["high"] >= sl:
                    return {"outcome": "loss", "pnl": entry_price - sl}

        # Time expiry — close at last bar's close
        final_price = float(future_bars["close"].iloc[-1])
        pnl = (final_price - entry_price) if is_long else (entry_price - final_price)
        return {"outcome": "timeout", "pnl": pnl}
