"""
execution/trade_journal.py
---------------------------
Logs every trade decision — whether a trade was taken or skipped —
to a queryable CSV file for post-hoc review and performance analysis.

No trade decision should ever be invisible: every signal, every skip,
every order outcome is recorded here.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


JOURNAL_COLUMNS = [
    "timestamp", "symbol", "timeframe", "regime", "patterns",
    "confidence", "signal", "action", "order_ticket", "entry_price",
    "stop_loss", "take_profit", "risk_reward", "outcome", "pnl",
    "session", "skip_reason", "reasoning_json",
]


class TradeJournal:
    """Append-only CSV trade journal with summary stats.

    Every entry in the journal is one decision point:
      - A trade that was taken (action='order_placed')
      - A trade that was skipped (action='skipped')
      - A NO_TRADE signal (action='no_signal')

    This makes it possible to audit any bot decision after the fact.
    """

    def __init__(self, journal_path: str = "logs/trade_journal.csv"):
        self._path = Path(journal_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------

    def log_signal(
        self,
        signal,
        was_traded:  bool,
        skip_reason: str = "",
    ) -> None:
        """Log a generated signal, whether traded or not.

        Args:
            signal:      TradeSignal from the signal engine.
            was_traded:  True if an order was actually placed.
            skip_reason: Why the trade was skipped (if not traded).
        """
        action = "order_placed" if was_traded else "skipped"
        reasoning = signal.reasoning if hasattr(signal, "reasoning") else {}

        self._append_row({
            "timestamp":    signal.timestamp.isoformat(),
            "symbol":       signal.symbol,
            "timeframe":    signal.timeframe,
            "regime":       reasoning.get("steps", {}).get("regime_h1", {}).get("regime", ""),
            "patterns":     str([p["pattern"] for p in reasoning.get("steps", {}).get("patterns", {}).get("candlestick", [])]),
            "confidence":   f"{signal.confidence:.4f}",
            "signal":       signal.signal.value,
            "action":       action,
            "skip_reason":  skip_reason,
            "session":      str(reasoning.get("steps", {}).get("session", {}).get("active", [])),
            "reasoning_json": json.dumps(reasoning, default=str),
        })

    def log_order(self, order_result: dict, signal) -> None:
        """Update the journal row with the actual order outcome.

        Args:
            order_result: Dict from OrderManager.place_market_order().
            signal:       The TradeSignal that generated the order.
        """
        row = {
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "symbol":       signal.symbol,
            "timeframe":    signal.timeframe,
            "signal":       signal.signal.value,
            "action":       "order_result",
            "order_ticket": str(order_result.get("ticket", "")),
            "entry_price":  str(order_result.get("entry_price", "")),
            "stop_loss":    str(order_result.get("sl", "")),
            "take_profit":  str(order_result.get("tp", "")),
            "outcome":      "placed" if order_result.get("success") else "failed",
            "reasoning_json": json.dumps(order_result, default=str),
        }
        self._append_row(row)

    def log_no_trade(
        self,
        symbol:     str,
        reason:     str,
        regime:     str = "",
        confidence: float = 0.0,
        timestamp:  Optional[datetime] = None,
    ) -> None:
        """Log a decision not to trade (for audit trail).

        Args:
            symbol:     Instrument that was evaluated.
            reason:     Why no trade was taken.
            regime:     Detected regime at decision time.
            confidence: Signal confidence score (may be 0 if signal not generated).
            timestamp:  Override timestamp (defaults to now).
        """
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        self._append_row({
            "timestamp":   ts,
            "symbol":      symbol,
            "regime":      regime,
            "confidence":  f"{confidence:.4f}",
            "signal":      "NO_TRADE",
            "action":      "no_signal",
            "skip_reason": reason,
        })

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def get_summary_stats(self) -> dict:
        """Return basic performance stats from the journal.

        Returns:
            Dict with win_rate, avg_rr, total_trades, profit_factor.
        """
        try:
            df = pd.read_csv(self._path)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return {}

        trades = df[df["action"] == "order_placed"].copy()
        total  = len(trades)
        if total == 0:
            return {"total_trades": 0}

        wins   = trades[trades["outcome"] == "win"] if "outcome" in trades.columns else pd.DataFrame()
        losses = trades[trades["outcome"] == "loss"] if "outcome" in trades.columns else pd.DataFrame()

        win_rate = len(wins) / total if total > 0 else 0.0

        pnl_col = pd.to_numeric(trades.get("pnl", pd.Series()), errors="coerce")
        gross_profit = pnl_col[pnl_col > 0].sum()
        gross_loss   = abs(pnl_col[pnl_col < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        return {
            "total_trades":  total,
            "win_rate":      round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "net_pnl":       round(float(pnl_col.sum()), 2),
        }

    def export_csv(self) -> str:
        """Return the path to the journal CSV file."""
        return str(self._path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_header(self) -> None:
        """Write CSV header if file doesn't exist."""
        if not self._path.exists():
            with open(self._path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=JOURNAL_COLUMNS)
                writer.writeheader()

    def _append_row(self, row: dict) -> None:
        """Append one row to the CSV, filling missing columns with empty strings."""
        full_row = {col: row.get(col, "") for col in JOURNAL_COLUMNS}
        with open(self._path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=JOURNAL_COLUMNS)
            writer.writerow(full_row)
        logger.debug("Journal: {} | {} | {}", full_row.get("timestamp"), full_row.get("symbol"), full_row.get("action"))
