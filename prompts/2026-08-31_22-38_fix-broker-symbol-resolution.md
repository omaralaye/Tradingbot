# Task: Fix Broker Symbol Suffix Resolution & MT5 Data Fetching

## Original User Prompt
```
omara@omara-HP-EliteBook-840-G2:~/Downloads/Bot$  source /home/omara/Downloads/Bot/.venv/bin/activate
(.venv) omara@omara-HP-EliteBook-840-G2:~/Downloads/Bot$  source .venv/bin/activate
    python scripts/test_connection.py
═══════════════════════════════════════════════════════
  MT5 Wine Bridge — Connection Test
═══════════════════════════════════════════════════════

── Test 1: RPyC bridge connection ──────────────────────
INFO     Connecting to MT5 RPyC bridge at 127.0.0.1:18812 (attempt 1/2)...
INFO     ✓ RPyC bridge connected.
✓ RPyC bridge connected
INFO     MT5 RPyC bridge disconnected.

── Test 2: MT5 initialize + login ──────────────────────
INFO     Connecting to MT5 RPyC bridge at 127.0.0.1:18812 (attempt 1/3)...
INFO     ✓ RPyC bridge connected.
INFO     MT5 connected | Account: 476821569 | Server: Exness-MT5Trial9 | Balance: 10.00 USD
✓ Logged in — Account: 476821569 | Balance: 10.00 USD
  Server: Exness-MT5Trial9 | Leverage: 1:2000

── Test 3: Fetch EURUSD H1 data ────────────────────────
✗ FAILED: No data returned for EURUSD/H1. MT5 error: (-2, 'Terminal: Invalid params')
INFO     MT5 RPyC bridge disconnected.
INFO     MT5 disconnected.

═══════════════════════════════════════════════════════
  ✗ Some tests failed — check logs above
```

## Context Gathered
- The user ran `python scripts/test_connection.py`.
- Test 1 (RPyC Bridge connection) and Test 2 (MT5 Login to Exness-MT5Trial9) passed successfully.
- Test 3 failed with `No data returned for EURUSD/H1. MT5 error: (-2, 'Terminal: Invalid params')`.
- Root cause: Exness accounts use broker symbol suffixes (e.g., `EURUSDm`, `GBPUSDm`, `BTCUSDm`). In addition, MT5 requires `symbol_select(symbol, True)` to enable symbols in the Market Watch.
- Also, `execution/order_manager.py` referenced top-level `mt5` module rather than `self._connector.mt5`, which would fail on Linux where `MetaTrader5` is bridged through RPyC.

## Clarifying Questions Asked
None needed — broker symbol inspection revealed `EURUSDm` on the active Exness demo account.

## Implementation Plan
1. Add automatic symbol resolution and `symbol_select(..., True)` in `MT5Connector` / `DataFetcher` so base symbols like `EURUSD` automatically resolve to broker-specific symbols like `EURUSDm`.
2. Update `DataFetcher` to use resolved symbols and handle symbol selection gracefully.
3. Update `execution/order_manager.py` to use `self._connector.mt5` instead of the top-level `mt5` import and resolve symbols before placing orders.
4. Update `scripts/test_connection.py` to test and report symbol resolution.
5. Run tests to verify all unit tests and connection smoke tests pass.
