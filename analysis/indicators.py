"""
analysis/indicators.py
-----------------------
Technical indicator calculations. All functions accept a DataFrame with
OHLCV columns and return a Series or DataFrame of indicator values.

Uses pandas-ta where available; falls back to manual numpy/pandas math.
No future-leaking logic — indicators are computed only from data available
at or before the current bar.
"""

import pandas as pd
import numpy as np
from loguru import logger

try:
    import pandas_ta as ta
    _HAS_PANDAS_TA = True
except ImportError:
    _HAS_PANDAS_TA = False
    logger.warning("pandas-ta not installed; using manual indicator calculations.")


def add_ema(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
    """Exponential Moving Average.

    Args:
        df:     OHLCV DataFrame.
        period: EMA lookback period.
        col:    Column to compute EMA on (default: 'close').

    Returns:
        pd.Series of EMA values, same index as df.
    """
    return df[col].ewm(span=period, adjust=False).mean()


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — measures market volatility.

    ATR = rolling mean of True Range, where:
    TR = max(high-low, |high-prev_close|, |low-prev_close|)

    Args:
        df:     OHLCV DataFrame.
        period: ATR lookback period.

    Returns:
        pd.Series of ATR values.
    """
    if _HAS_PANDAS_TA:
        return df.ta.atr(length=period)

    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period).mean()


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index + Directional Movement indicators.

    Returns a DataFrame with columns:
        ADX      — trend strength (0-100; >25 = trending)
        DI_plus  — positive directional indicator (uptrend pressure)
        DI_minus — negative directional indicator (downtrend pressure)
    """
    if _HAS_PANDAS_TA:
        result = df.ta.adx(length=period)
        # pandas-ta column names vary by version; normalise them
        cols = result.columns.tolist()
        adx_col = [c for c in cols if c.startswith("ADX_")][0]
        dmp_col  = [c for c in cols if c.startswith("DMP_")][0]
        dmn_col  = [c for c in cols if c.startswith("DMN_")][0]
        return result[[adx_col, dmp_col, dmn_col]].rename(
            columns={adx_col: "ADX", dmp_col: "DI_plus", dmn_col: "DI_minus"}
        )

    # Manual calculation
    high, low, close = df["high"], df["low"], df["close"]
    prev_high = high.shift(1)
    prev_low  = low.shift(1)
    prev_close = close.shift(1)

    up_move   = high - prev_high
    down_move = prev_low - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = add_atr(df, period)
    di_plus  = 100 * pd.Series(plus_dm,  index=df.index).rolling(period).mean() / atr
    di_minus = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr
    dx = (abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, np.nan)) * 100
    adx = dx.rolling(period).mean()

    return pd.DataFrame({"ADX": adx, "DI_plus": di_plus, "DI_minus": di_minus})


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Relative Strength Index (0-100).

    Values above 70 suggest overbought; below 30 suggest oversold.
    """
    if _HAS_PANDAS_TA:
        return df.ta.rsi(length=period)

    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram.

    Returns DataFrame with columns: MACD, signal, histogram.
    """
    if _HAS_PANDAS_TA:
        result = df.ta.macd(fast=fast, slow=slow, signal=signal)
        cols = result.columns.tolist()
        macd_col = [c for c in cols if c.startswith("MACD_")][0]
        sig_col  = [c for c in cols if c.startswith("MACDs_")][0]
        hist_col = [c for c in cols if c.startswith("MACDh_")][0]
        return result[[macd_col, sig_col, hist_col]].rename(
            columns={macd_col: "MACD", sig_col: "signal", hist_col: "histogram"}
        )

    ema_fast = add_ema(df, fast)
    ema_slow = add_ema(df, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({"MACD": macd_line, "signal": signal_line, "histogram": histogram})


def add_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands — upper, middle (SMA), and lower bands.

    Returns DataFrame with columns: BB_upper, BB_mid, BB_lower, BB_pct_b.
    BB %B = (price - lower) / (upper - lower), useful as a 0-1 oscillator.
    """
    mid   = df["close"].rolling(period).mean()
    std   = df["close"].rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    pct_b = (df["close"] - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({"BB_upper": upper, "BB_mid": mid, "BB_lower": lower, "BB_pct_b": pct_b})


def add_all_indicators(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Add all standard indicators to a copy of the OHLCV DataFrame.

    Args:
        df:     OHLCV DataFrame.
        config: Optional dict to override default periods, e.g.
                {'ema_fast': 9, 'ema_slow': 21, 'atr_period': 14, ...}

    Returns:
        New DataFrame with all indicator columns appended.
    """
    cfg = config or {}
    out = df.copy()

    out["EMA_fast"] = add_ema(df, cfg.get("ema_fast", 9))
    out["EMA_slow"] = add_ema(df, cfg.get("ema_slow", 21))
    out["EMA_50"]   = add_ema(df, cfg.get("ema_50", 50))
    out["EMA_200"]  = add_ema(df, cfg.get("ema_200", 200))
    out["ATR"]      = add_atr(df, cfg.get("atr_period", 14))
    out["RSI"]      = add_rsi(df, cfg.get("rsi_period", 14))

    adx_df = add_adx(df, cfg.get("adx_period", 14))
    out[["ADX", "DI_plus", "DI_minus"]] = adx_df

    macd_df = add_macd(df)
    out[["MACD", "MACD_signal", "MACD_hist"]] = macd_df[["MACD", "signal", "histogram"]]

    bb_df = add_bollinger_bands(df, cfg.get("bb_period", 20), cfg.get("bb_std", 2.0))
    out[["BB_upper", "BB_mid", "BB_lower", "BB_pct_b"]] = bb_df

    return out
