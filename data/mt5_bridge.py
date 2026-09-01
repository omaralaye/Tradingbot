"""
data/mt5_bridge.py
------------------
RPyC-based bridge that connects Linux Python to the MetaTrader 5 Windows API
running inside Wine.

Architecture:
  [Linux Python bot] ──RPyC──> [Wine Python rpyc_classic server]
                                        │
                                 [MetaTrader5 Windows DLL]
                                        │
                                 [MT5 terminal (Wine)]

Usage:
    from data.mt5_bridge import MT5Bridge
    bridge = MT5Bridge()          # connects to localhost:18812
    mt5 = bridge.get_api()        # returns proxied MetaTrader5 module
    mt5.initialize(...)

The bridge exposes the same API surface as the official MetaTrader5 package
so all existing code that calls mt5.* works unchanged.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from loguru import logger

try:
    import rpyc
    _RPYC_AVAILABLE = True
except ImportError:
    _RPYC_AVAILABLE = False
    logger.warning("rpyc not installed. Install it: pip install rpyc")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# These mirror the MetaTrader5 timeframe constants so code that references
# mt5.TIMEFRAME_* works even without a live connection.
TIMEFRAME_M1  = 1
TIMEFRAME_M2  = 2
TIMEFRAME_M3  = 3
TIMEFRAME_M4  = 4
TIMEFRAME_M5  = 5
TIMEFRAME_M6  = 6
TIMEFRAME_M10 = 10
TIMEFRAME_M12 = 12
TIMEFRAME_M15 = 15
TIMEFRAME_M20 = 20
TIMEFRAME_M30 = 30
TIMEFRAME_H1  = 16385
TIMEFRAME_H2  = 16386
TIMEFRAME_H3  = 16387
TIMEFRAME_H4  = 16388
TIMEFRAME_H6  = 16390
TIMEFRAME_H8  = 16392
TIMEFRAME_H12 = 16396
TIMEFRAME_D1  = 16408
TIMEFRAME_W1  = 32769
TIMEFRAME_MN1 = 49153

ORDER_TYPE_BUY        = 0
ORDER_TYPE_SELL       = 1
ORDER_TYPE_BUY_LIMIT  = 2
ORDER_TYPE_SELL_LIMIT = 3
ORDER_TYPE_BUY_STOP   = 4
ORDER_TYPE_SELL_STOP  = 5

TRADE_ACTION_DEAL    = 1
TRADE_ACTION_PENDING = 5
TRADE_ACTION_SLTP    = 6
TRADE_ACTION_MODIFY  = 7
TRADE_ACTION_REMOVE  = 8
TRADE_ACTION_CLOSE_BY = 10

ORDER_FILLING_FOK = 0
ORDER_FILLING_IOC = 1
ORDER_FILLING_RETURN = 2

ORDER_TIME_GTC           = 0
ORDER_TIME_DAY           = 1
ORDER_TIME_SPECIFIED     = 2
ORDER_TIME_SPECIFIED_DAY = 3

TRADE_RETCODE_DONE    = 10009
TRADE_RETCODE_REQUOTE = 10004

POSITION_TYPE_BUY  = 0
POSITION_TYPE_SELL = 1

COPY_TICKS_ALL  = -1
COPY_TICKS_INFO = 1
COPY_TICKS_TRADE = 2

RES_S_OK = 1


# ─────────────────────────────────────────────────────────────────────────────
# Proxy Wrapper for MT5 C Extension Compatibility over RPyC
# ─────────────────────────────────────────────────────────────────────────────

class _MT5Proxy:
    """Wrapper around the remote MetaTrader5 module proxy to ensure argument compatibility.

    MetaTrader 5's C-extension rejects RPyC netref dictionary proxy objects when passed
    as positional arguments, returning (-2, 'Unnamed arguments not allowed').
    This proxy automatically unpacks dict arguments into keyword arguments for
    order_send and order_check calls.
    """

    def __init__(self, remote_module: Any) -> None:
        self._remote = remote_module

    def order_send(self, *args: Any, **kwargs: Any) -> Any:
        """Forward order_send to MT5, unpacking dict requests into kwargs."""
        if args and isinstance(args[0], dict):
            combined = dict(args[0])
            combined.update(kwargs)
            return self._remote.order_send(**combined)
        return self._remote.order_send(*args, **kwargs)

    def order_check(self, *args: Any, **kwargs: Any) -> Any:
        """Forward order_check to MT5, unpacking dict requests into kwargs."""
        if args and isinstance(args[0], dict):
            combined = dict(args[0])
            combined.update(kwargs)
            return self._remote.order_check(**combined)
        return self._remote.order_check(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._remote, name)


# ─────────────────────────────────────────────────────────────────────────────
# Bridge class
# ─────────────────────────────────────────────────────────────────────────────

class MT5Bridge:
    """RPyC connection to the Wine-hosted MetaTrader5 Python API.

    The bridge keeps a single persistent RPyC connection open.  Call
    get_api() to obtain a proxy object with the same interface as the
    native MetaTrader5 module.

    Example:
        bridge = MT5Bridge(host="127.0.0.1", port=18812)
        mt5 = bridge.get_api()
        ok = mt5.initialize(
            path=r"C:\\Program Files\\MetaTrader 5\\terminal64.exe"
        )
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 18812,
        timeout: int = 60,
    ) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout
        self._conn: Optional[Any] = None
        self._mt5_module: Optional[Any] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self, max_retries: int = 3, backoff: float = 3.0) -> bool:
        """Establish the RPyC connection to the Wine bridge server.

        Returns:
            True if connected, False if all retries exhausted.
        """
        if not _RPYC_AVAILABLE:
            logger.error("rpyc is not installed — cannot connect to bridge.")
            return False

        wait = backoff
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "Connecting to MT5 RPyC bridge at {}:{} (attempt {}/{})...",
                    self._host, self._port, attempt, max_retries,
                )
                self._conn = rpyc.classic.connect(
                    self._host,
                    self._port,
                )
                # Set socket timeout manually (rpyc v6 removed the config= kwarg)
                self._conn._channel.stream.sock.settimeout(self._timeout)
                # Import MetaTrader5 on the remote (Wine) side
                self._conn.execute("import MetaTrader5 as _mt5")
                raw_module = self._conn.modules["MetaTrader5"]
                self._mt5_module = _MT5Proxy(raw_module)
                logger.info("✓ RPyC bridge connected.")
                return True
            except Exception as exc:
                logger.warning(
                    "RPyC connect attempt {}/{} failed: {}", attempt, max_retries, exc
                )
                if attempt < max_retries:
                    logger.info("Retrying in {:.0f}s...", wait)
                    time.sleep(wait)
                    wait *= 2
        logger.error(
            "Could not connect to MT5 RPyC bridge after {} attempts. "
            "Is the bridge running? Run: bash scripts/start_mt5_bridge.sh",
            max_retries,
        )
        return False

    def disconnect(self) -> None:
        """Close the RPyC connection."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            self._mt5_module = None
            logger.info("MT5 RPyC bridge disconnected.")

    def is_connected(self) -> bool:
        """Return True if the RPyC connection is alive."""
        if self._conn is None or self._mt5_module is None:
            return False
        try:
            self._conn.ping()
            return True
        except Exception:
            return False

    def get_api(self) -> Any:
        """Return the remote MetaTrader5 module proxy.

        The returned object has the same method signatures as the native
        MetaTrader5 package (initialize, login, copy_rates_from_pos, etc.).

        Raises:
            RuntimeError: If not connected.
        """
        if not self.is_connected():
            raise RuntimeError(
                "MT5 bridge not connected. Call connect() first, or use as context manager."
            )
        return self._mt5_module

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "MT5Bridge":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.disconnect()
        return False  # never suppress exceptions
