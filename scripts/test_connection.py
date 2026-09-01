#!/usr/bin/env python3
"""
scripts/test_connection.py
--------------------------
Smoke-test for the MT5 Wine RPyC bridge.

Run after starting the bridge:
    bash scripts/start_mt5_bridge.sh
    source .venv/bin/activate
    python scripts/test_connection.py

Expected output on success:
    ✓ RPyC bridge connected
    ✓ MT5 initialized
    ✓ Logged in — Account: 5055253531 | Balance: XXXX.XX USD
    ✓ EURUSD H1: 500 bars fetched
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
logger.remove()
logger.add(sys.stdout, format="<level>{level:<8}</level> {message}", colorize=True)


def test_bridge_connection():
    from data.mt5_bridge import MT5Bridge

    print("\n── Test 1: RPyC bridge connection ──────────────────────")
    bridge = MT5Bridge(host="127.0.0.1", port=18812)
    ok = bridge.connect(max_retries=2, backoff=2.0)
    if not ok:
        print("✗ FAILED: Could not connect to RPyC bridge.")
        print("  Make sure the bridge is running:")
        print("    bash scripts/start_mt5_bridge.sh")
        return False
    print("✓ RPyC bridge connected")
    bridge.disconnect()
    return True


def test_mt5_initialize():
    from config.settings import get_settings
    from data.mt5_connector import MT5Connector

    print("\n── Test 2: MT5 initialize + login ──────────────────────")
    settings = get_settings()
    connector = MT5Connector(settings)

    ok = connector.connect()
    if not ok:
        print("✗ FAILED: MT5 connect/login failed.")
        return False, None

    info = connector.mt5.account_info()
    print(f"✓ Logged in — Account: {info.login} | Balance: {info.balance:.2f} {info.currency}")
    print(f"  Server: {info.server} | Leverage: 1:{info.leverage}")
    return True, connector


def test_fetch_data(connector):
    from data.data_fetcher import DataFetcher
    import pandas as pd

    print("\n── Test 3: Fetch EURUSD H1 data ────────────────────────")
    fetcher = DataFetcher(connector)
    try:
        df = fetcher.fetch_ohlcv("EURUSD", "H1", n_bars=500)
        print(f"✓ EURUSD H1: {len(df)} bars fetched")
        print(f"  From: {df.index[0]}  To: {df.index[-1]}")
        print(f"  Columns: {list(df.columns)}")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def main():
    print("═" * 55)
    print("  MT5 Wine Bridge — Connection Test")
    print("═" * 55)

    # Test 1: Bridge connectivity
    if not test_bridge_connection():
        sys.exit(1)

    # Test 2: MT5 initialize + login
    ok, connector = test_mt5_initialize()
    if not ok:
        sys.exit(1)

    try:
        # Test 3: Fetch data
        ok = test_fetch_data(connector)
    finally:
        connector.disconnect()

    print("\n" + "═" * 55)
    if ok:
        print("  ✓ All tests passed — MT5 bridge is working!")
    else:
        print("  ✗ Some tests failed — check logs above")
        sys.exit(1)
    print("═" * 55)


if __name__ == "__main__":
    main()
