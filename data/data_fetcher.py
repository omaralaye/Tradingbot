"""
data/data_fetcher.py
--------------------
Fetches OHLCV data from MetaTrader 5 across multiple timeframes,
with a local parquet cache to avoid re-fetching for backtests.

On Linux this module talks to MT5 via the RPyC bridge (MT5Connector).
All mt5.* calls are proxied to the Wine-hosted MetaTrader5 package.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from data.mt5_connector import MT5Connector
# Import timeframe constants from the bridge (available without a live connection)
from data.mt5_bridge import (
    TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15, TIMEFRAME_M30,
    TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1, TIMEFRAME_W1, TIMEFRAME_MN1,
)


# Maps human-readable timeframe strings to MT5 integer constants
_TF_MAP: dict[str, int] = {
    "M1":  TIMEFRAME_M1,
    "M5":  TIMEFRAME_M5,
    "M15": TIMEFRAME_M15,
    "M30": TIMEFRAME_M30,
    "H1":  TIMEFRAME_H1,
    "H4":  TIMEFRAME_H4,
    "D1":  TIMEFRAME_D1,
    "W1":  TIMEFRAME_W1,
    "MN1": TIMEFRAME_MN1,
}

OHLCV_COLUMNS = ["time", "open", "high", "low", "close", "volume"]


class DataFetcher:
    """Fetches and caches OHLCV data from the MT5 terminal.

    Separates concerns cleanly: MT5Connector owns the connection,
    DataFetcher owns data retrieval and caching logic.
    """

    def __init__(self, connector: MT5Connector, cache_dir: str = "data/cache"):
        """
        Args:
            connector: An active (or activatable) MT5Connector instance.
            cache_dir: Directory for local parquet cache files.
        """
        self._connector = connector
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        n_bars: int = 500,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars from MT5 for a single symbol/timeframe.

        Args:
            symbol:    e.g. 'EURUSD'
            timeframe: e.g. 'H1', 'M15', 'D1'
            n_bars:    Number of bars to fetch (most recent first).

        Returns:
            DataFrame with columns [open, high, low, close, volume],
            indexed by UTC datetime.

        Raises:
            RuntimeError: If MT5 returns no data.
        """
        if not self._connector.ensure_connected():
            raise RuntimeError("MT5 not connected — cannot fetch data.")

        mt5 = self._connector.mt5
        resolved_symbol = self._connector.resolve_symbol(symbol) or symbol
        tf_const = self._to_mt5_timeframe(timeframe)
        rates = mt5.copy_rates_from_pos(resolved_symbol, tf_const, 0, n_bars)

        if rates is None or len(rates) == 0:
            try:
                error = mt5.last_error()
            except Exception:
                error = "unknown"
            raise RuntimeError(
                f"No data returned for {symbol} (broker symbol: {resolved_symbol})/{timeframe}. MT5 error: {error}"
            )

        names = list(rates.dtype.names) if hasattr(rates, "dtype") and rates.dtype.names else []
        if names:
            col_dict = {name: np.array(rates[name]) for name in names}
            df = pd.DataFrame(col_dict)
        else:
            df = pd.DataFrame(rates)

        if "tick_volume" in df.columns and "volume" not in df.columns:
            df["volume"] = df["tick_volume"]

        df = df[OHLCV_COLUMNS].copy()
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df = df.astype({
            "open": float, "high": float, "low": float,
            "close": float, "volume": float,
        })

        logger.debug("Fetched {} bars for {}/{} (resolved: {})", len(df), symbol, timeframe, resolved_symbol)
        return df

    def fetch_multi_timeframe(
        self,
        symbol: str,
        timeframes: list[str],
        n_bars: int = 500,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV for one symbol across multiple timeframes.

        Returns:
            Dict mapping timeframe string -> DataFrame, e.g.
            {'H1': df_h1, 'H4': df_h4, 'D1': df_d1}
        """
        result: dict[str, pd.DataFrame] = {}
        for tf in timeframes:
            try:
                result[tf] = self.fetch_ohlcv(symbol, tf, n_bars)
            except Exception as exc:
                logger.error("Failed to fetch {}/{}: {}", symbol, tf, exc)
        return result

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def save_to_cache(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """Save a DataFrame to local parquet cache."""
        path = self._cache_path(symbol, timeframe)
        df.to_parquet(path)
        logger.debug("Cached {}/{} -> {}", symbol, timeframe, path)

    def load_from_cache(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load a cached DataFrame if it exists."""
        path = self._cache_path(symbol, timeframe)
        if path.exists():
            df = pd.read_parquet(path)
            logger.debug("Loaded cache {}/{} ({} bars)", symbol, timeframe, len(df))
            return df
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_mt5_timeframe(self, tf_str: str) -> int:
        """Convert a timeframe string to its MT5 integer constant."""
        if tf_str not in _TF_MAP:
            raise ValueError(
                f"Unknown timeframe '{tf_str}'. Valid options: {list(_TF_MAP.keys())}"
            )
        return _TF_MAP[tf_str]

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        """Return the parquet cache path for a symbol/timeframe pair."""
        return self._cache_dir / f"{symbol}_{timeframe}.parquet"
