# Task: Write Comprehensive README.md Documentation

## Original User Prompt (verbatim)

> "write the readme.md file on what the project does and how to use it in detail"

## Clarifications & Scope

- **Repository**: Multi-Market Trading Bot (Forex + Crypto) using MetaTrader 5 (MT5).
- **Scope**: Write a complete, comprehensive, and production-grade `README.md` explaining:
  1. What the project does (architecture, multi-market support, multi-timeframe analysis, session awareness, regime & pattern recognition, ML model ensemble, risk engine, execution).
  2. Linux/Wine bridge architecture vs Windows native.
  3. Prerequisites and step-by-step installation instructions.
  4. Configuration reference (`.env`, `instruments.yaml`, `sessions.yaml`, `settings.py`).
  5. Detailed usage instructions:
     - Starting the MT5 Wine bridge (`scripts/start_mt5_bridge.sh`).
     - Verifying connectivity with broker symbol resolution (`scripts/test_connection.py`).
     - Running the live/demo trading engine (`live/run_bot.py`).
     - Running historical backtests and generating reports (`backtest/engine.py`, `backtest/reports.py`).
     - Training and evaluating ML models with walk-forward validation (`ml/train.py`).
     - Running unit tests (`pytest`, `make test`).
  6. Safety rules, hard kill switch, position sizing, and trade journaling.
  7. Prominent financial risk disclaimer as required by `AGENT.md` rules 5 and 15.

## Implementation Intent

1. Record this task prompt log in `prompts/2026-09-01_23-25_write-readme-documentation.md` per `AGENT.md` Rule 6.
2. Draft a well-structured, clear, in-depth `README.md` covering all aspects of the system with CLI examples, configuration tables, architecture diagrams, and workflow guides.
3. Validate that markdown links and code paths are exact.
