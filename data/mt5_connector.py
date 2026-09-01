"""
data/mt5_connector.py
---------------------
Handles connection, login, and reconnection logic to the MetaTrader 5 terminal.

On Linux, MT5 runs inside Wine and the Python API is accessed via an RPyC bridge:

  Architecture:
    [Linux Python] ──RPyC──> [Wine Python (rpyc_classic server)]
                                       │
                               [MetaTrader5 Windows DLL]
                                       │
                               [MT5 terminal (Wine)]

  Before using this class you must:
    1. Run:  bash scripts/install_wine_python.sh   (once)
    2. Run:  bash scripts/start_mt5_bridge.sh      (before each bot session)

The class is a drop-in replacement for the original direct-import approach —
all callers interact with the same connect/disconnect/is_connected API.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from loguru import logger

from data.mt5_bridge import MT5Bridge


class MT5Connector:
    """Manages the lifecycle of a MetaTrader 5 terminal connection.

    On Linux the connection goes through an RPyC bridge to Wine.
    The class exposes connect/disconnect/reconnect with exponential backoff,
    and a context manager interface for use in 'with' blocks.

    Example:
        with MT5Connector(settings) as conn:
            if conn.is_connected():
                mt5 = conn.mt5  # use the MT5 API
                rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 100)
    """

    def __init__(self, settings) -> None:
        """
        Args:
            settings: A Settings instance with mt5_login, mt5_password,
                      mt5_server, mt5_path, mt5_bridge_host, mt5_bridge_port.
        """
        self._settings = settings
        self._connected: bool = False
        self._symbol_cache: dict[str, str] = {}
        self._bridge = MT5Bridge(
            host=getattr(settings, "mt5_bridge_host", "127.0.0.1"),
            port=getattr(settings, "mt5_bridge_port", 18812),
            timeout=getattr(settings, "mt5_bridge_timeout", 60),
        )
        self._mt5: Optional[Any] = None  # remote MetaTrader5 module proxy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def mt5(self) -> Any:
        """Return the MetaTrader5 API proxy (remote module via RPyC)."""
        if self._mt5 is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._mt5

    def connect(self) -> bool:
        """Connect to the RPyC bridge and log in to the MT5 terminal.

        Returns:
            True if connection and login succeeded, False otherwise.
        """
        # ── Step 1: Connect RPyC bridge ───────────────────────────────
        if not self._bridge.connect():
            logger.error(
                "Cannot reach MT5 bridge. Start it first:\n"
                "  bash scripts/start_mt5_bridge.sh"
            )
            return False

        self._mt5 = self._bridge.get_api()

        # ── Step 2: Initialize the MT5 terminal ───────────────────────
        mt5_path = self._settings.mt5_path or None

        # mt5_path in settings is the *Linux* path to terminal64.exe;
        # we need to convert to the Windows (Wine C:\) path for the API call.
        win_path = _linux_path_to_wine(mt5_path) if mt5_path else None

        initialized = (
            self._mt5.initialize(win_path) if win_path else self._mt5.initialize()
        )

        if not initialized:
            logger.error(
                "MT5 initialize() failed: {} — Is the terminal running?",
                self._mt5.last_error(),
            )
            self._bridge.disconnect()
            self._mt5 = None
            return False

        # ── Step 3: Log in ────────────────────────────────────────────
        login_ok = self._mt5.login(
            login=self._settings.mt5_login,
            password=self._settings.mt5_password.get_secret_value(),
            server=self._settings.mt5_server,
        )

        if not login_ok:
            logger.error(
                "MT5 login failed for account {} on {}. Error: {}",
                self._settings.mt5_login,
                self._settings.mt5_server,
                self._mt5.last_error(),
            )
            self._mt5.shutdown()
            self._bridge.disconnect()
            self._mt5 = None
            return False

        info = self._mt5.account_info()
        logger.info(
            "MT5 connected | Account: {} | Server: {} | Balance: {:.2f} {}",
            info.login,
            info.server,
            info.balance,
            info.currency,
        )

        term_info = self._mt5.terminal_info()
        if term_info is not None and not getattr(term_info, "trade_allowed", True):
            logger.warning(
                "⚠️ Algo Trading is disabled in the MT5 terminal! "
                "Please enable 'Algo Trading' in MT5 (click the Algo Trading button in the top toolbar or press Ctrl+E)."
            )

        self._connected = True
        return True

    def disconnect(self) -> None:
        """Cleanly shut down the MT5 connection and RPyC bridge."""
        if self._mt5 is not None:
            try:
                self._mt5.shutdown()
            except Exception:
                pass
        self._bridge.disconnect()
        self._mt5 = None
        self._connected = False
        logger.info("MT5 disconnected.")

    def is_connected(self) -> bool:
        """Return True if the bridge is alive and MT5 terminal is reachable."""
        if not self._bridge.is_connected() or self._mt5 is None:
            return False
        try:
            return self._mt5.terminal_info() is not None
        except Exception:
            return False

    def ensure_connected(self, max_retries: int = 3, backoff_seconds: float = 5.0) -> bool:
        """Reconnect if the connection has been lost.

        Uses exponential backoff between retries. Never places orders
        while disconnected — callers must check the return value.

        Args:
            max_retries:      How many reconnect attempts to make.
            backoff_seconds:  Base wait time; doubles on each retry.

        Returns:
            True if connected after retries, False if all attempts fail.
        """
        if self.is_connected():
            return True

        logger.warning("MT5 connection lost. Attempting to reconnect...")
        wait = backoff_seconds
        for attempt in range(1, max_retries + 1):
            logger.info("Reconnect attempt {}/{}", attempt, max_retries)
            if self.connect():
                logger.info("Reconnected on attempt {}.", attempt)
                return True
            logger.warning("Attempt {} failed. Waiting {:.0f}s...", attempt, wait)
            time.sleep(wait)
            wait *= 2

        logger.critical(
            "Failed to reconnect to MT5 after {} attempts.", max_retries
        )
        return False

    def resolve_symbol(self, symbol: str) -> str:
        """Resolve a base symbol (e.g. 'EURUSD') to the broker-specific symbol (e.g. 'EURUSDm').

        Also ensures the symbol is selected in MT5 Market Watch. Caches the result.

        Args:
            symbol: Generic symbol name (e.g. 'EURUSD', 'BTCUSD').

        Returns:
            The matched broker symbol name (or original symbol as fallback).
        """
        if symbol in self._symbol_cache:
            resolved = self._symbol_cache[symbol]
            if self.is_connected():
                try:
                    self.mt5.symbol_select(resolved, True)
                except Exception:
                    pass
            return resolved

        if not self.ensure_connected():
            return symbol

        mt5 = self.mt5

        # 1. Try exact symbol match
        try:
            info = mt5.symbol_info(symbol)
            if info is not None:
                mt5.symbol_select(symbol, True)
                self._symbol_cache[symbol] = symbol
                return symbol
        except Exception:
            pass

        # 2. Try common broker suffixes (Exness mini 'm', cent 'c', raw '.r', etc.)
        common_suffixes = ["m", "c", ".r", "_i", ".", "+", ".a", "pro", "raw", "m.ecn", ".ecn"]
        for suffix in common_suffixes:
            candidate = f"{symbol}{suffix}"
            try:
                info = mt5.symbol_info(candidate)
                if info is not None:
                    mt5.symbol_select(candidate, True)
                    self._symbol_cache[symbol] = candidate
                    logger.info("Resolved symbol '{}' -> broker symbol '{}'", symbol, candidate)
                    return candidate
            except Exception:
                pass

        # 3. Fallback: Search all symbols in broker for matching pattern
        try:
            matches = mt5.symbols_get(f"*{symbol}*")
            if matches:
                matched_name = matches[0].name
                mt5.symbol_select(matched_name, True)
                self._symbol_cache[symbol] = matched_name
                logger.info("Resolved symbol '{}' -> broker symbol '{}'", symbol, matched_name)
                return matched_name
        except Exception:
            pass

        # 4. Fallback to original symbol
        return symbol

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "MT5Connector":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.disconnect()
        return False  # do not suppress exceptions


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _linux_path_to_wine(linux_path: str) -> str:
    """Convert a Linux filesystem path inside a Wine prefix to a Windows path.

    Example:
        "/home/omara/.mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe"
        → "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
    """
    import re
    # Strip the WINEPREFIX/drive_X/ prefix and convert slashes
    match = re.match(r".*/drive_([a-zA-Z])/(.*)", linux_path)
    if match:
        drive = match.group(1).upper()
        rest = match.group(2).replace("/", "\\")
        return f"{drive}:\\{rest}"
    # Fallback: return as-is (may already be a Windows path)
    return linux_path
