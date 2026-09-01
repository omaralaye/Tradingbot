# Task: Implement Full Project Architecture from AGENT.md

## Original Prompt (verbatim)

> "implement the project architecture in agent.md"

## Clarifications

None required — AGENT.md § 3 defines the full directory and file structure.

## Implementation Intent

Create the complete scaffold of all folders and Python/YAML/TOML files
defined in AGENT.md § 3, including:

- Project root: `.env.example`, `pyproject.toml`
- `config/`: `settings.py` (pydantic), `instruments.yaml`, `sessions.yaml`
- `data/`: `mt5_connector.py`, `data_fetcher.py`, `cache/`
- `analysis/`: `timeframes.py`, `regime_detection.py`, `indicators.py`,
  `session_context.py`, `patterns/` (candlestick, chart, support_resistance)
- `ml/`: `features.py`, `labeling.py`, `train.py`, `inference.py`,
  `models/` (direction, regime, confidence), `registry/`
- `signals/`: `signal_engine.py`, `confidence_scoring.py`
- `risk/`: `position_sizing.py`, `risk_manager.py`, `stop_target_logic.py`
- `execution/`: `order_manager.py`, `trade_journal.py`
- `backtest/`: `engine.py`, `reports.py`
- `live/`: `run_bot.py`
- `tests/`, `logs/`

Each file will contain meaningful starter code (not empty stubs), docstrings,
and inline comments explaining how each piece contributes to the system.
