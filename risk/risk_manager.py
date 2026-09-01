"""
risk/risk_manager.py
---------------------
Central risk gatekeeper. Enforces all hard limits before any order is placed:
  - Kill switch (manual or auto-triggered)
  - Daily loss limit
  - Max concurrent open positions
  - Max correlated exposure

If the kill switch fires, all new order placement halts and open positions
are optionally flattened via a registered callback.
"""

from __future__ import annotations

from datetime import date
from typing import Callable, Optional

from loguru import logger


class RiskManager:
    """Enforces risk rules and manages the kill switch.

    Usage:
        rm = RiskManager(settings)
        rm.register_kill_switch_callback(order_manager.close_all_positions)
        allowed, reason = rm.check_can_trade(symbol, signal, open_positions)
        if not allowed:
            journal.log_no_trade(symbol, reason, ...)
    """

    # Pairs considered correlated — don't open all in same direction
    CORRELATED_GROUPS: list[list[str]] = [
        ["EURUSD", "GBPUSD", "AUDUSD"],  # USD as quote
        ["USDJPY", "USDCHF"],            # USD as base
        ["BTCUSD", "ETHUSD"],            # crypto
    ]

    def __init__(
        self,
        settings,
        max_daily_loss_pct:       Optional[float] = None,
        max_open_positions:       Optional[int]   = None,
        max_correlated_positions: int             = 3,
    ):
        """
        Args:
            settings:                  Settings instance.
            max_daily_loss_pct:        Override settings value if provided.
            max_open_positions:        Override settings value if provided.
            max_correlated_positions:  Max positions in same correlated group.
        """
        self._settings         = settings
        self._max_daily_loss   = max_daily_loss_pct or settings.max_daily_loss_pct
        self._max_positions    = max_open_positions or settings.max_open_positions
        self._max_correlated   = max_correlated_positions

        self._kill_switch_active  = False
        self._kill_switch_reason: Optional[str] = None
        self._daily_pnl:          float = 0.0
        self._day_start_balance:  float = 0.0
        self._today:              date  = date.today()
        self._kill_callbacks:     list[Callable] = []

    # ------------------------------------------------------------------
    # Core gate
    # ------------------------------------------------------------------

    def check_can_trade(
        self,
        symbol:         str,
        signal,
        open_positions: list[dict],
    ) -> tuple[bool, str]:
        """Check all risk rules before allowing a trade.

        Args:
            symbol:          Symbol being considered.
            signal:          TradeSignal (used to check direction for correlation).
            open_positions:  List of currently open position dicts (from MT5).

        Returns:
            (True, "") if trade is allowed.
            (False, reason_string) if trade is blocked.
        """
        # 1. Kill switch — hardest gate
        if self._kill_switch_active:
            return False, f"Kill switch active: {self._kill_switch_reason}"

        # 2. Daily loss limit
        if self._day_start_balance > 0:
            loss_pct = (-self._daily_pnl / self._day_start_balance) * 100
            if loss_pct >= self._max_daily_loss:
                self.trigger_kill_switch(
                    f"Daily loss limit reached: {loss_pct:.2f}% >= {self._max_daily_loss}%"
                )
                return False, f"Daily loss limit exceeded ({loss_pct:.2f}%)"

        # 3. Max concurrent positions
        if len(open_positions) >= self._max_positions:
            return False, (
                f"Max open positions reached: {len(open_positions)}/{self._max_positions}"
            )

        # 4. Correlated exposure
        correlated_count = self._count_correlated(symbol, open_positions)
        if correlated_count >= self._max_correlated:
            return False, (
                f"Max correlated positions reached for group containing {symbol}: "
                f"{correlated_count}/{self._max_correlated}"
            )

        return True, ""

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def trigger_kill_switch(self, reason: str) -> None:
        """Activate the kill switch: halt all new orders, flatten positions.

        This is irreversible within the current session — requires a manual
        restart to clear the kill switch after investigation.
        """
        self._kill_switch_active = True
        self._kill_switch_reason = reason
        logger.critical("⚠️  KILL SWITCH ACTIVATED: {}", reason)

        for callback in self._kill_callbacks:
            try:
                callback()
                logger.info("Kill switch callback executed: {}", callback.__name__)
            except Exception as exc:
                logger.error("Kill switch callback failed: {}", exc)

    def register_kill_switch_callback(self, fn: Callable) -> None:
        """Register a function to call when the kill switch fires.

        Typically: order_manager.close_all_positions
        """
        self._kill_callbacks.append(fn)
        logger.debug("Kill switch callback registered: {}", fn.__name__)

    @property
    def is_kill_switch_active(self) -> bool:
        """True if the kill switch has been triggered."""
        return self._kill_switch_active

    # ------------------------------------------------------------------
    # PnL tracking
    # ------------------------------------------------------------------

    def update_daily_pnl(self, realized_pnl: float) -> None:
        """Accumulate today's realised PnL.

        Call this after each trade closes with the trade's PnL.
        """
        self._daily_pnl += realized_pnl
        logger.debug("Daily PnL updated: {:.2f}", self._daily_pnl)

    def reset_daily_stats(self, current_balance: float) -> None:
        """Reset daily counters at the start of each trading day.

        Args:
            current_balance: Account balance at day start (used for loss % calc).
        """
        self._daily_pnl        = 0.0
        self._day_start_balance = current_balance
        self._today            = date.today()
        logger.info("Daily stats reset. Start balance: {:.2f}", current_balance)

    @property
    def daily_loss_pct(self) -> float:
        """Current day's loss as a percentage of starting balance."""
        if self._day_start_balance == 0:
            return 0.0
        return max(0.0, (-self._daily_pnl / self._day_start_balance) * 100)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_correlated(self, symbol: str, open_positions: list[dict]) -> int:
        """Count how many open positions are in the same correlated group as symbol."""
        for group in self.CORRELATED_GROUPS:
            if symbol in group:
                return sum(1 for p in open_positions if p.get("symbol") in group)
        return 0
