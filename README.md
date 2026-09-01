# Automated Multi-Market Trading Bot (Forex & Crypto)

An enterprise-grade, multi-timeframe algorithmic trading bot designed for **Forex** and **Crypto CFDs** via **MetaTrader 5 (MT5)**. Built with Python 3.11+, it combines quantitative technical analysis (regimes, multi-timeframe confluence, chart/candlestick patterns, and dynamic support/resistance) with an interpretable Machine Learning ensemble (gradient boosted trees), an institutional-grade risk management engine, and cross-platform Linux/Wine MT5 execution.

---

> [!CAUTION]
> ### ⚠️ Financial Risk & Safety Notice
> Automated algorithmic trading involves substantial risk of financial loss. Past performance (including backtests) does not guarantee future results.
> - **Demo by default:** The bot defaults strictly to **Demo/Paper trading**. Real money execution requires an explicit, deliberate configuration flag (`ENABLE_LIVE_TRADING=true`).
> - **Mandatory risk management:** Every order is hard-gated by strict stop-losses, volatility-adjusted position sizing, daily loss caps, and a hardware/software kill-switch.
> - **Test extensively:** Never risk capital you cannot afford to lose. Run extensive paper-trading and out-of-sample backtests before considering live trading.

---

## Table of Contents

- [Overview & Architecture](#overview--architecture)
- [Key Features](#key-features)
  - [1. Data Layer & Linux/Wine Bridge](#1-data-layer--linuxwine-bridge)
  - [2. Multi-Timeframe Confluence](#2-multi-timeframe-confluence)
  - [3. Market Regime & Pattern Detection](#3-market-regime--pattern-detection)
  - [4. Machine Learning Ensemble](#4-machine-learning-ensemble)
  - [5. Signal Engine & Confidence Scoring](#5-signal-engine--confidence-scoring)
  - [6. Risk Management & Position Sizing](#6-risk-management--position-sizing)
  - [7. Order Execution & Journaling](#7-order-execution--journaling)
  - [8. Backtesting & Reporting Engine](#8-backtesting--reporting-engine)
- [Project Directory Structure](#project-directory-structure)
- [Installation & Setup](#installation--setup)
  - [Prerequisites](#prerequisites)
  - [Step 1: Clone and Setup Python Environment](#step-1-clone-and-setup-python-environment)
  - [Step 2: Linux Wine Bridge Setup (Linux Users Only)](#step-2-linux-wine-bridge-setup-linux-users-only)
  - [Step 3: Environment Configuration](#step-3-environment-configuration)
- [How to Use](#how-to-use)
  - [1. Verify MT5 Connectivity & Symbol Resolution](#1-verify-mt5-connectivity--symbol-resolution)
  - [2. Run in Demo Mode (Default)](#2-run-in-demo-mode-default)
  - [3. Enabling Live Trading](#3-enabling-live-trading)
  - [4. Running Historical Backtests](#4-running-historical-backtests)
  - [5. Training & Evaluating ML Models](#5-training--evaluating-ml-models)
  - [6. Running Unit & Integration Tests](#6-running-unit--integration-tests)
- [Configuration Reference](#configuration-reference)
  - [Environment Variables (.env)](#environment-variables-env)
  - [Instruments Configuration (config/instruments.yaml)](#instruments-configuration-configinstrumentsyaml)
  - [Trading Sessions (config/sessions.yaml)](#trading-sessions-configsessionsyaml)
- [Monitoring & Trade Journal](#monitoring--trade-journal)
- [License](#license)

---

## Overview & Architecture

The bot uses a multi-layered quantitative pipeline where technical indicators, geometric chart patterns, and market regime classifications serve as explainable features for an ensemble of Machine Learning models. The resulting confluence feeds into a signal engine, passing through strict risk filters before execution.

```
                                  [MetaTrader 5 Terminal]
                                   (Native Windows or Wine)
                                             ▲
                                             │ RPyC (Port 18812)
                                             ▼
                                    [MT5 Connector]
                                (data/mt5_connector.py)
                                             │
                        ┌────────────────────┴────────────────────┐
                        ▼                                         ▼
                [Data Fetcher]                             [Order Manager]
             (OHLCV Multi-TF Cache)                      (Execution & Retries)
                        │                                         ▲
                        ▼                                         │
               [Market Analysis]                                  │
      ┌─────────────────┼─────────────────┐                       │
      ▼                 ▼                 ▼                       │
 [Regime Detect]  [TF Confluence]   [Patterns & S/R]              │
 (Trend/Range)    (M1-D1 Align)     (Chart/Candles)               │
      └─────────────────┬─────────────────┘                       │
                        ▼                                         │
                [ML Feature Store]                                │
             (ml/features.py No-Leak)                             │
                        │                                         │
                        ▼                                         │
              [ML Model Ensemble]                                 │
           (Direction / Regime / Win Prob)                        │
                        │                                         │
                        ▼                                         │
                 [Signal Engine]                                  │
              (LONG / SHORT / NO_TRADE)                           │
                        │                                         │
                        ▼                                         │
                 [Risk Manager] ──────────────────────────────────┘
            (Daily Loss, Kill-Switch,
             Position Sizer, ATR SL/TP)
                        │
                        ▼
                 [Trade Journal]
             (logs/trade_journal.csv)
```

---

## Key Features

### 1. Data Layer & Linux/Wine Bridge
- **Linux Wine Support:** MetaTrader 5's official Python library requires Windows C bindings. The bot includes a high-performance **RPyC bridge** (`data/mt5_bridge.py`) allowing a native Linux Python environment to seamlessly control a 64-bit MT5 terminal running inside Wine.
- **Auto Broker Suffix Resolution:** Automatically resolves base symbols (e.g., `EURUSD`, `BTCUSD`) to broker-specific suffixes (e.g., `EURUSDm`, `EURUSD.r`, `BTCUSDm` on Exness/IC Markets) and activates them in MT5 Market Watch.
- **Resilient Connectivity:** Automatic reconnection with exponential backoff if the terminal connection drops.
- **Local Parquet Caching:** Historical rates are cached in `data/cache/` to accelerate ML feature computation and backtests.

### 2. Multi-Timeframe Confluence
- Analyzes multiple timeframes simultaneously: **M1, M5, M15, M30, H1, H4, D1**.
- **Macro Bias:** D1 and H4 define overall market trend and macro momentum.
- **Structure:** H1 and M30 map key swing highs/lows, trendlines, and support/resistance zones.
- **Timing:** M15, M5, and M1 are reserved strictly for precision entry timing.
- Requires configurable confluence across at least two timeframes before a signal is triggered.

### 3. Market Regime & Pattern Detection
- **Regime Classifier (`analysis/regime_detection.py`):** Classifies price action into `UPTREND`, `DOWNTREND`, `RANGING`, or `CHOPPY` using ADX, directional movement (+DI/-DI), moving average slopes, and structural higher-high/lower-low analysis.
- **Geometric Chart Patterns (`analysis/patterns/chart_patterns.py`):** Detects Ascending/Descending/Symmetrical Triangles, Double Tops/Bottoms, Head & Shoulders (and Inverse), Flags, Pennants, and Channels with probabilistic confidence scores.
- **Candlestick Patterns (`analysis/patterns/candlestick_patterns.py`):** Recognizes Bullish/Bearish Engulfing, Hammer/Pin Bar, Shooting Star, Morning/Evening Star, and Doji candles.
- **Dynamic S/R Levels (`analysis/patterns/support_resistance.py`):** Clusters swing points to identify high-probability support and resistance zones and tracks historical retests.
- **Session Awareness (`analysis/session_context.py`):** Aware of Sydney, Tokyo, London, and New York sessions and volatile overlaps (e.g., London/NY overlap). Enforces configurable dead-zone avoidance.

### 4. Machine Learning Ensemble
Instead of a black box, rule-based indicators and pattern outputs serve as engineered features for three gradient-boosted models (XGBoost / LightGBM):
1. **Direction Model (`ml/models/direction_model.py`):** Predicts directional probability over a forward horizon.
2. **Regime Model (`ml/models/regime_model.py`):** Classifies trending vs ranging vs choppy conditions.
3. **Confidence Model (`ml/models/confidence_model.py`):** Predicts the probability of a candidate setup reaching Take-Profit before Stop-Loss (R-multiple success).
- **Leakage Prevention:** Built with Marcos López de Prado's **Walk-Forward Cross-Validation** featuring purge and embargo gaps (`ml/train.py`) and triple-barrier labeling (`ml/labeling.py`).

### 5. Signal Engine & Confidence Scoring
- Combines regime, pattern confidence, timeframe confluence, session context, and ML probabilities into unified `LONG`, `SHORT`, or `NO_TRADE` decisions (`signals/signal_engine.py`).
- Produces an auditable breakdown of reasons for every decision. Trades below `MIN_SIGNAL_CONFIDENCE` (default `0.60`) are rejected.

### 6. Risk Management & Position Sizing
- **Dynamic Volatility Position Sizing (`risk/position_sizing.py`):** Automatically calculates exact lot sizes based on account balance, user risk percentage (default 1.0%), stop-loss distance, and instrument pip/point values. Clamps to broker min/max lot and step constraints.
- **Structure & ATR Stops (`risk/stop_target_logic.py`):** Mandatory Stop Loss (SL) and Take Profit (TP) computed using Average True Range (ATR) multipliers and swing structures. No naked positions.
- **Account Protection & Kill-Switch (`risk/risk_manager.py`):**
  - **Max Daily Loss Limit:** Halts trading if daily realized/unrealized losses exceed the threshold (default 3.0%).
  - **Max Open Positions:** Enforces maximum concurrent open trades (default 5).
  - **Correlation Limiter:** Prevents stacking risk across heavily correlated pairs (e.g., EURUSD, GBPUSD, AUDUSD).
  - **Global Kill-Switch:** Immediately shuts down trading and optionally flattens all open positions.

### 7. Order Execution & Journaling
- **Order Manager (`execution/order_manager.py`):** Handles market and pending orders via MT5 with automatic retries for requotes (up to 3 attempts), IOC/FOK filling policy selection, and slippage monitoring.
- **Trade Journal (`execution/trade_journal.py`):** Every signal, skip reason, order execution, ticket, and PnL outcome is recorded to an append-only CSV file (`logs/trade_journal.csv`).

### 8. Backtesting & Reporting Engine
- **Bar-by-Bar Simulator (`backtest/engine.py`):** Replays historical OHLCV data using identical signal and risk logic as live execution.
- **Performance Analytics (`backtest/reports.py`):** Calculates Win Rate, Profit Factor, Maximum Drawdown %, Annualized Sharpe Ratio, Sortino Ratio, and breakdown reports segmented by market regime and trading session.

---

## Project Directory Structure

```
trading_bot/
├── AGENT.md                 # Core AI assistant instructions & operating rules
├── README.md                # Comprehensive project documentation
├── Makefile                 # Shortcuts for testing and dependency installation
├── setup.sh                 # One-click environment bootstrap script
├── pyproject.toml           # Project dependencies, build config, and pytest rules
├── .env.example             # Template for credentials and risk parameters
├── prompts/                 # Auditable task prompt logs (recorded before changes)
│
├── config/                  # Configuration system
│   ├── settings.py          # Pydantic BaseSettings loaded from environment/.env
│   ├── instruments.yaml     # Traded symbols (Forex & Crypto), pips, lot limits
│   └── sessions.yaml        # Market session hours (UTC) and avoidance windows
│
├── data/                    # Market data acquisition & connectivity
│   ├── mt5_bridge.py        # RPyC bridge client/server protocol (Linux/Wine)
│   ├── mt5_connector.py     # MT5 terminal connection & broker symbol resolution
│   ├── data_fetcher.py      # Multi-timeframe OHLCV fetcher with local caching
│   └── cache/               # Parquet cache storage for historical bars
│
├── analysis/                # Quantitative technical analysis
│   ├── indicators.py        # EMA, ATR, ADX, RSI, MACD, Bollinger Bands
│   ├── regime_detection.py  # Trend, range, and choppy market classification
│   ├── timeframes.py        # Multi-timeframe resampling and confluence
│   ├── session_context.py   # Active session tracking & volatility profile
│   └── patterns/
│       ├── candlestick_patterns.py  # Engulfing, Pin Bar, Hammer, Doji, Stars
│       ├── chart_patterns.py        # Triangles, H&S, Double Tops, Flags
│       └── support_resistance.py    # Swing point clustering for S/R zones
│
├── ml/                      # Machine Learning pipeline
│   ├── features.py          # Lookahead-free feature engineering
│   ├── labeling.py          # Triple-barrier and forward return labeling
│   ├── train.py             # Purged/embargoed walk-forward cross-validation
│   ├── inference.py         # Model loading and real-time inference
│   ├── models/
│   │   ├── direction_model.py   # Gradient-boosted directional classifier
│   │   ├── regime_model.py      # ML market regime classifier
│   │   └── confidence_model.py  # Trade setup quality & win probability scorer
│   └── registry/            # Serialized models (.joblib) and metadata JSONs
│
├── signals/                 # Signal synthesis
│   ├── signal_engine.py     # Multi-factor confluence signal generator
│   └── confidence_scoring.py# Composite confidence calculation
│
├── risk/                    # Institutional risk management
│   ├── position_sizing.py   # Balance-based % risk lot sizing calculator
│   ├── risk_manager.py      # Daily loss limit, max positions, kill-switch
│   └── stop_target_logic.py # ATR and structural SL/TP calculator
│
├── execution/               # MT5 order management
│   ├── order_manager.py     # Order placement, modification, retries, closing
│   └── trade_journal.py     # Comprehensive CSV decision journal
│
├── backtest/                # Historical simulation
│   ├── engine.py            # Event-driven backtesting engine
│   └── reports.py           # Equity curve, Sharpe, Sortino, regime breakdowns
│
├── live/                    # Live & Demo trading runner
│   └── run_bot.py           # Continuous orchestrator loop
│
├── scripts/                 # Operational & setup utilities
│   ├── install_wine_python.sh # Configures Windows Python 3.11 inside Wine
│   ├── start_mt5_bridge.sh    # Launches MT5 terminal + RPyC bridge server
│   └── test_connection.py     # Validates bridge, MT5 login, and symbol quotes
│
├── tests/                   # Automated unit & integration tests
│   ├── test_order_manager.py
│   ├── test_position_sizing.py
│   ├── test_regime_detection.py
│   └── test_risk_manager.py
│
└── logs/                    # Runtime logs and exported trade journals
```

---

## Installation & Setup

### Prerequisites

- **Operating System:** Linux (Ubuntu 22.04+ recommended) or Windows 10/11.
- **Python:** Python 3.11 or higher.
- **MetaTrader 5:**
  - **Windows:** MetaTrader 5 desktop terminal installed.
  - **Linux:** MetaTrader 5 installed via Wine (`wine-stable`).
- **Broker Account:** A Demo account from any MT5 broker (e.g., Exness, IC Markets, MetaQuotes Demo).

---

### Step 1: Clone and Setup Python Environment

Run the automated setup script or configure manually:

```bash
# Clone the repository
git clone https://github.com/omaralaye/Tradingbot.git
cd Tradingbot

# Option A: One-step bootstrap (Linux)
bash setup.sh

# Option B: Manual setup
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

---

### Step 2: Linux Wine Bridge Setup (Linux Users Only)

Because `MetaTrader5` is a Windows-only Python wheel, Linux users bridge calls into Wine:

1. **Verify Wine is installed:**
   ```bash
   wine --version  # Should be wine-stable 8.0+ or wine-staging
   ```

2. **Run the Wine Python setup script (one-time setup):**
   This downloads official Windows Python 3.11 AMD64, installs it silently into your Wine prefix (`~/.mt5`), and installs `MetaTrader5` and `rpyc`:
   ```bash
   bash scripts/install_wine_python.sh
   ```

3. **Start the MT5 Wine Bridge:**
   This starts the MT5 terminal and launches the RPyC server on port `18812`:
   ```bash
   bash scripts/start_mt5_bridge.sh
   ```
   *(To stop the bridge later: `bash scripts/start_mt5_bridge.sh --stop`)*

---

### Step 3: Environment Configuration

Copy the example `.env` file and fill in your broker credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
# Broker credentials
MT5_LOGIN=476821569
MT5_PASSWORD=your_demo_password
MT5_SERVER=Exness-MT5Trial9
MT5_PATH=/home/omara/.mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe

# RPyC Bridge (for Linux/Wine)
MT5_BRIDGE_HOST=127.0.0.1
MT5_BRIDGE_PORT=18812

# Trading Mode (Keep false until tested!)
ENABLE_LIVE_TRADING=false

# Risk Controls
RISK_PER_TRADE_PCT=1.0
MAX_DAILY_LOSS_PCT=3.0
MAX_OPEN_POSITIONS=5

# Logging
LOG_LEVEL=INFO
LOG_DIR=logs
```

---

## How to Use

### 1. Verify MT5 Connectivity & Symbol Resolution

Before launching the bot, run the connection diagnostic script. This verifies the RPyC bridge, authenticates with your broker, tests symbol resolution (e.g. matching `EURUSD` to `EURUSDm`), and fetches sample candle bars:

```bash
source .venv/bin/activate
python scripts/test_connection.py
```

**Expected Output:**
```
═══════════════════════════════════════════════════════
  MT5 Wine Bridge — Connection Test
═══════════════════════════════════════════════════════

── Test 1: RPyC bridge connection ──────────────────────
✓ RPyC bridge connected

── Test 2: MT5 initialize + login ──────────────────────
✓ Logged in — Account: 476821569 | Balance: 10000.00 USD
  Server: Exness-MT5Trial9 | Leverage: 1:2000

── Test 3: Fetch EURUSD H1 data ────────────────────────
✓ EURUSD H1: 500 bars fetched
  From: 2026-08-01 10:00:00+00:00  To: 2026-09-01 18:00:00+00:00
  Columns: ['open', 'high', 'low', 'close', 'volume']

═══════════════════════════════════════════════════════
  ✓ All tests passed — MT5 bridge is working!
═══════════════════════════════════════════════════════
```

---

### 2. Run in Demo Mode (Default)

Launch the trading loop in safe Demo mode:

```bash
source .venv/bin/activate
python -m live.run_bot
```

**Loop Cycle Workflow:**
1. Verifies connection to MT5 and checks account equity.
2. Checks kill-switch and active daily drawdown limits.
3. Iterates across all active instruments defined in `config/instruments.yaml`.
4. Fetches multi-timeframe OHLCV data (`M15`, `H1`, `H4`, `D1`).
5. Analyzes market regime, S/R zones, chart/candlestick patterns, and session windows.
6. Evaluates ML model probabilities.
7. Evaluates confidence score against `MIN_SIGNAL_CONFIDENCE`.
8. Applies risk gatekeeper rules (position limits, correlation limits).
9. Calculates ATR-based SL/TP and exact lot size based on `RISK_PER_TRADE_PCT`.
10. Transmits market order with retry logic and logs full details to `logs/trade_journal.csv`.
11. Sleeps until next cycle (default 60 seconds).

---

### 3. Enabling Live Trading

> [!WARNING]
> Only switch to live trading after extensive demo evaluation. Real capital will be committed.

To enable live trading:
1. Update `.env`:
   ```ini
   ENABLE_LIVE_TRADING=true
   ```
2. Or pass as an environment variable when invoking:
   ```bash
   ENABLE_LIVE_TRADING=true python -m live.run_bot
   ```

The bot will print a prominent high-visibility warning in the terminal and logs acknowledging live execution.

---

### 4. Running Historical Backtests

The backtesting engine (`backtest/engine.py`) simulates execution bar-by-bar using historical data without placing live orders:

```python
from datetime import datetime
from backtest.engine import BacktestEngine
from backtest.reports import BacktestReporter
from config.settings import get_settings
from data.mt5_connector import MT5Connector
from data.data_fetcher import DataFetcher
from signals.signal_engine import SignalEngine
from risk.risk_manager import RiskManager
from risk.position_sizing import PositionSizer
from risk.stop_target_logic import StopTargetCalculator

settings = get_settings()
connector = MT5Connector(settings)

with connector:
    fetcher = DataFetcher(connector)
    # Fetch historical data for EURUSD across timeframes
    data = {
        "EURUSD": {
            "H1": fetcher.fetch_ohlcv("EURUSD", "H1", n_bars=2000),
            "H4": fetcher.fetch_ohlcv("EURUSD", "H4", n_bars=1000),
            "M15": fetcher.fetch_ohlcv("EURUSD", "M15", n_bars=4000),
        }
    }

# Run backtest
engine = BacktestEngine(
    signal_engine=signal_engine,
    risk_manager=RiskManager(settings),
    position_sizer=PositionSizer(settings.risk_per_trade_pct),
    stop_calculator=StopTargetCalculator(),
    initial_balance=10000.0,
)
results = engine.run(data, symbols=["EURUSD"])

# Generate performance report
reporter = BacktestReporter(engine.get_results(), initial_balance=10000.0)
print(reporter.text_report())
```

**Metrics Provided:**
- Total Trades, Wins, Losses, Win Rate %
- Gross Profit, Gross Loss, Profit Factor
- Maximum Drawdown %
- Annualized Sharpe Ratio & Sortino Ratio
- Breakdown by market regime (Uptrend vs Range vs Choppy)
- Breakdown by trading session (London, New York, Asian)

---

### 5. Training & Evaluating ML Models

The machine learning pipeline implements purged and embargoed Walk-Forward Cross-Validation to eliminate lookahead bias:

```python
from ml.train import WalkForwardTrainer
from ml.features import FeatureEngineer
from ml.labeling import TripleBarrierLabeler
from ml.models.direction_model import DirectionModel

# 1. Engineer features from OHLCV and technical detectors
fe = FeatureEngineer()
X = fe.build_features(ohlcv_dict)

# 2. Label data using the triple-barrier method (TP vs SL vs time expiry)
labeler = TripleBarrierLabeler(take_profit_r=1.5, stop_loss_r=1.0, max_bars=24)
y = labeler.label(ohlcv_dict["H1"])

# 3. Train model with walk-forward CV
trainer = WalkForwardTrainer(
    model_class=DirectionModel,
    n_splits=5,
    purge_bars=10,
    embargo_bars=5,
    registry_dir="ml/registry",
)
metrics = trainer.train(X, y, model_name="direction_eurusd")
print("Validation Metrics:", metrics)
```

Approved models and their metadata (training dates, feature names, out-of-sample accuracy) are stored in `ml/registry/`.

---

### 6. Running Unit & Integration Tests

The test suite validates position sizing formulas, regime detection thresholds, order proxying, and risk management limits:

```bash
# Run all tests
make test
# Or directly via pytest:
.venv/bin/python -m pytest

# Run tests with code coverage report
make test-cov
```

---

## Configuration Reference

### Environment Variables (`.env`)

| Variable | Type | Default | Description |
|---|---|---|---|
| `MT5_LOGIN` | `int` | `0` | Broker account number |
| `MT5_PASSWORD` | `str` | `""` | Broker account trading password |
| `MT5_SERVER` | `str` | `"MetaQuotes-Demo"` | Broker MT5 server name |
| `MT5_PATH` | `str` | Auto | Linux path to `terminal64.exe` inside Wine prefix |
| `MT5_BRIDGE_HOST` | `str` | `"127.0.0.1"` | RPyC bridge server host IP |
| `MT5_BRIDGE_PORT` | `int` | `18812` | RPyC bridge server port |
| `MT5_BRIDGE_TIMEOUT`| `int` | `60` | RPC request timeout in seconds |
| `ENABLE_LIVE_TRADING` | `bool` | `false` | Must be `true` to place live market orders |
| `RISK_PER_TRADE_PCT` | `float` | `1.0` | Maximum account percentage risked per trade |
| `MAX_DAILY_LOSS_PCT` | `float` | `3.0` | Daily drawdown threshold triggering kill-switch |
| `MAX_OPEN_POSITIONS` | `int` | `5` | Maximum concurrent open positions |
| `MAX_CORRELATED_POSITIONS`| `int` | `3` | Maximum open trades in correlated pairs |
| `MIN_SIGNAL_CONFIDENCE` | `float` | `0.60` | Minimum composite score to take a trade |
| `MIN_TIMEFRAME_AGREEMENT` | `int` | `2` | Number of timeframes that must align |
| `LOG_LEVEL` | `str` | `"INFO"` | Console log verbosity (`DEBUG`, `INFO`, `WARNING`) |
| `LOG_DIR` | `str` | `"logs"` | Directory for log files and trade journals |
| `TELEGRAM_TOKEN` | `str` | `""` | *(Optional)* Telegram bot token for trade alerts |
| `TELEGRAM_CHAT_ID` | `str` | `""` | *(Optional)* Telegram chat ID for trade alerts |

---

### Instruments Configuration (`config/instruments.yaml`)

Defines active symbols and asset-specific rules:

```yaml
instruments:
  - symbol: EURUSD
    type: forex
    pip_size: 0.0001
    min_lot: 0.01
    max_lot: 10.0
    lot_step: 0.01
    enabled: true
    risk_multiplier: 1.0

  - symbol: BTCUSD
    type: crypto
    pip_size: 1.0
    min_lot: 0.01
    max_lot: 1.0
    lot_step: 0.01
    enabled: false         # Enable after demo testing
    risk_multiplier: 0.5   # Conservative sizing for volatile crypto
```

---

### Trading Sessions (`config/sessions.yaml`)

Configures UTC trading windows, overlaps, and periods to avoid (e.g. rollover or news releases):

```yaml
sessions:
  sydney:   { open: "22:00", close: "07:00", volatility: low }
  tokyo:    { open: "00:00", close: "09:00", volatility: medium }
  london:   { open: "08:00", close: "17:00", volatility: high }
  new_york: { open: "13:00", close: "22:00", volatility: high }

overlaps:
  london_new_york: { open: "13:00", close: "17:00", volatility: very_high }

avoid_trading_windows:
  - "21:30-22:30"   # NY close to Sydney open low-liquidity rollover
```

---

## Monitoring & Trade Journal

### Runtime Logs
Logs are rotated daily and stored in `logs/bot_YYYYMMDD.log`. Output includes cycle execution duration, signal scores, risk checks, and MT5 order confirmations.

### Trade Journal (`logs/trade_journal.csv`)
Every cycle logs an auditable entry with the following schema:
- `timestamp`: UTC execution time.
- `symbol`: Generic traded symbol (e.g. `EURUSD`).
- `timeframe`: Entry timeframe (e.g. `H1`).
- `regime`: Market regime detected (`UPTREND`, `DOWNTREND`, `RANGING`, `CHOPPY`).
- `patterns`: Detected chart/candlestick patterns.
- `confidence`: Composite confidence score (0.0 to 1.0).
- `signal`: `LONG`, `SHORT`, or `NO_TRADE`.
- `action`: `order_placed`, `skipped`, or `no_signal`.
- `order_ticket`: MT5 position ticket ID.
- `entry_price`: Actual filled execution price.
- `stop_loss`: Calculated stop loss level.
- `take_profit`: Calculated take profit level.
- `risk_reward`: Planned Risk-to-Reward ratio.
- `skip_reason`: Explicit reason if a trade was rejected by risk controls.
- `reasoning_json`: Detailed JSON payload with indicator values and ML scores.

---

## License

This project is proprietary and intended for private automated trading and research. Unauthorized copying, distribution, or public deployment is strictly prohibited.
