# Task: Set up Bot project to work with MT5 (Wine/Linux)

## Original User Prompt
"Set up the Bot project in /home/omara/Downloads/Bot to work with MT5"

## Context Gathered
- OS: Linux (Ubuntu), Wine 10.0
- MT5 terminal: `/home/omara/.mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe`
- WINEPREFIX: `/home/omara/.mt5`
- Demo account: login `5055253531`, server `MetaQuotes-Demo`
- `mt5linux` v1.1.1 installed but requires Docker/udocker (not available)
- No Windows Python in the Wine prefix yet
- All other Python deps installed in `.venv` (pandas, xgboost, lightgbm, etc.)

## Clarifying Questions Asked
None needed — environment was fully probed via shell commands.

## What Was Implemented
1. `scripts/install_wine_python.sh` — installs Windows Python 3.11 + MetaTrader5 + rpyc inside Wine prefix
2. `scripts/start_mt5_bridge.sh` — launches MT5 terminal + RPyC server bridge
3. `data/mt5_connector.py` — rewritten to connect via RPyC to Wine bridge
4. `data/mt5_bridge.py` — RPyC client that mirrors the MetaTrader5 module API
5. `.env` — created from .env.example with correct Linux/Wine paths
6. `scripts/test_connection.py` — smoke-test to verify the bridge works
7. `config/settings.py` — updated mt5_path default to Wine terminal path
