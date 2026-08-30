# AGENT.md — Multi-Market Trading Bot (Forex + Crypto, MetaTrader 5, Python)

This file tells an AI coding agent (e.g. Claude Code) how to build, extend, and
maintain this project. Read this fully before writing any code. If something
here is ambiguous or a decision would change the architecture, **ask a
clarifying question before proceeding** rather than guessing.

---

## 1. Project Summary

Build an automated trading bot that:

- Trades **Forex and Crypto CFDs/pairs** through **MetaTrader 5 (MT5)**.
- Analyzes **multiple timeframes** simultaneously: M1, M5, M15, M30, H1, H4, D1
  (and optionally W1/MN1 for higher-timeframe bias).
- Is aware of **trading sessions**: Sydney, Tokyo, London, New York, and
  session overlaps (London/NY overlap especially).
- Detects **market regime**: uptrend, downtrend, ranging/sideways, and
  transitional/choppy conditions.
- Detects **chart patterns**: support/resistance, trendlines, triangles,
  flags/pennants, head & shoulders, double top/bottom, wedges, channels,
  candlestick patterns (engulfing, pin bar, doji, etc.).
- Combines the above into a **signal engine** that produces trade decisions.
- **Executes trades automatically** via the MT5 Python API, with risk
  management, position sizing, and a hard kill-switch.
- Logs everything and supports **backtesting** and **paper/demo trading**
  before any live execution is enabled.

**Language/Stack:** Python 3.11+, `MetaTrader5` package, `pandas`, `numpy`,
`ta`/`pandas-ta` or custom indicators, `pytest` for tests, `pydantic` for
config validation, `loguru` or standard `logging` for logs.

---

## 2. Non-Negotiable Ground Rules

1. **Demo/paper account by default.** The bot must default to a demo MT5
   account. Live trading only activates when a config flag
   (`ENABLE_LIVE_TRADING=true`) is explicitly set **and** the agent has
   confirmed this with the user in writing.
2. **No hardcoded credentials.** MT5 login, password, server, and any API
   keys must come from environment variables or a `.env` file that is
   git-ignored. Never commit secrets.
3. **Risk management is mandatory, not optional.** Every trade must have:
   - a defined stop loss,
   - a position size calculated from account risk % (never a fixed lot size
     hardcoded without justification),
   - a max daily loss limit and max open positions limit,
   - a global kill-switch that flattens positions and halts new orders if
     triggered (e.g. daily drawdown exceeded, connection anomaly, repeated
     order errors).
4. **No silent scope changes.** If a task implies a design decision not
   covered in this file (e.g. which broker's swap/spread model to assume,
   which risk % to default to, whether to hedge or net positions), stop and
   ask the user instead of assuming.
5. **This is not financial advice and carries real financial risk.** The
   agent should remind the user, at minimum once during setup and once
   before enabling live trading, that automated trading can result in real
   monetary losses and that the user is responsible for testing thoroughly
   on demo/backtest before going live.

---

## 3. Architecture Overview

```
trading_bot/
├── AGENT.md
├── README.md
├── .env.example
├── pyproject.toml / requirements.txt
├── config/
│   ├── settings.py          # pydantic-based config loader
│   ├── instruments.yaml     # symbols traded (forex pairs, crypto pairs), per-symbol overrides
│   └── sessions.yaml         # session times (UTC), overlap windows
├── data/
│   ├── mt5_connector.py     # connect/login/reconnect logic to MT5 terminal
│   ├── data_fetcher.py      # pulls OHLCV per symbol/timeframe
│   └── cache/                # local parquet/csv cache for backtests
├── analysis/
│   ├── timeframes.py        # multi-timeframe alignment/resampling
│   ├── regime_detection.py  # trend / range / choppy classification
│   ├── indicators.py        # EMA, ATR, ADX, RSI, MACD, BB, etc.
│   ├── patterns/
│   │   ├── candlestick_patterns.py
│   │   ├── chart_patterns.py     # triangles, H&S, double top/bottom, flags
│   │   └── support_resistance.py
│   └── session_context.py   # which session(s) are active, volatility profile
├── ml/
│   ├── features.py           # feature engineering from indicators/patterns/regime/session
│   ├── labeling.py           # triple-barrier / forward-return labeling, regime labels
│   ├── train.py               # training pipeline (walk-forward CV)
│   ├── models/
│   │   ├── direction_model.py    # predicts next-move direction
│   │   ├── regime_model.py       # predicts trend/range/choppy
│   │   └── confidence_model.py   # scores/ranks candidate setups
│   ├── registry/              # saved model versions + metadata (metrics, train date, feature set)
│   └── inference.py           # loads latest approved model(s) for live use
├── signals/
│   ├── signal_engine.py     # combines regime + pattern + timeframe confluence + ML output
│   └── confidence_scoring.py
├── risk/
│   ├── position_sizing.py
│   ├── risk_manager.py      # daily loss limit, max positions, kill switch
│   └── stop_target_logic.py # SL/TP placement (ATR-based, structure-based)
├── execution/
│   ├── order_manager.py     # places/modifies/closes orders via MT5
│   └── trade_journal.py     # logs every decision + trade with reasoning
├── backtest/
│   ├── engine.py
│   └── reports.py
├── live/
│   └── run_bot.py           # main loop: fetch -> analyze -> signal -> risk -> execute
├── tests/
│   └── ...
└── logs/
```

The agent should propose this structure at project start and confirm with the
user before generating a large number of files, since exact folder naming may
need to adapt to whatever exists already in the repo.

---

## 4. Data Layer

- Use the official `MetaTrader5` Python package to connect to a running MT5
  terminal (Windows, or Wine/Linux bridge — **ask the user which OS the bot
  will run on**, since MT5's Python API requires a Windows-based MT5
  terminal or a compatibility layer on Linux/Mac).
- Fetch OHLCV data for each configured symbol across all configured
  timeframes (`M1, M5, M15, M30, H1, H4, D1`).
- Handle reconnects gracefully — MT5 terminal connections can drop; the bot
  must detect this and retry with backoff, not silently stop trading (but
  also must not place blind orders while disconnected).
- Cache historical data locally to avoid re-fetching for backtests.
- Normalize crypto vs forex symbol quirks (different pip/point sizes,
  different trading hours — crypto is ~24/7, forex has weekend gaps).

---

## 5. Multi-Timeframe Analysis

- Higher timeframes (D1, H4) establish **bias/context** (overall trend
  direction).
- Mid timeframes (H1, M30, M15) establish **structure** (swing highs/lows,
  key levels).
- Lower timeframes (M5, M1) are used for **entry timing** only, never as the
  sole basis for a trade decision.
- The signal engine should require **confluence across at least two
  timeframes** before generating a trade signal (exact rule set should be
  configurable, not hardcoded — expose thresholds in config).
- Document, in code comments and README, exactly how each timeframe
  contributes to the final decision so the logic isn't a black box.

---

## 6. Session Awareness

- Define session windows in UTC (configurable, since DST shifts them):
  - Sydney, Tokyo, London, New York.
  - London/New York overlap (typically highest volatility for majors).
- Adjust behavior by session:
  - Widen/tighten stop distances based on typical session volatility (e.g.
    ATR at that time of day historically).
  - Optionally avoid trading in known low-liquidity windows (e.g. right
    before/after major news, or the dead zone between NY close and Sydney
    open) — make this configurable, not a hard rule the agent invents.
- Crypto trades 24/7 — session logic mainly affects **volatility
  expectations**, not whether the market is "open."

---

## 7. Regime & Pattern Detection

**Regime classification** (per symbol, per relevant timeframe):
- Uptrend / downtrend: e.g. via moving average slope/stack, ADX + directional
  movement, higher-highs/higher-lows structure.
- Ranging: e.g. ADX below threshold, price oscillating between defined
  support/resistance bands.
- Choppy/transitional: flag when signals conflict across timeframes or
  volatility is erratic — the bot should be more conservative here, not
  force a trade.

**Chart pattern detection:**
- Support/resistance via swing point clustering.
- Trendlines and channels.
- Classic patterns: triangles (ascending/descending/symmetrical), flags,
  pennants, wedges, double top/bottom, head & shoulders (and inverse).
- Candlestick patterns: engulfing, pin bar/hammer, doji, morning/evening
  star, etc.
- Each detector should output a **confidence score**, not just a boolean —
  pattern recognition is inherently fuzzy and this should be reflected in
  the signal engine rather than treated as certain.

---

## 8. Machine Learning Model

The bot's decisions should be informed by ML models trained on price and
pattern data, not just hardcoded rules. The rule-based detectors in Section
7 (regime, patterns, S/R) still run — they become **feature inputs** to the
model layer, not the final decision-maker. This keeps the system explainable
(you can always see *why* the model leans a certain way) instead of being a
pure black box.

### 8.1 What the model(s) predict
Three separate, purpose-built models rather than one model trying to do
everything (easier to validate, debug, and retrain independently):

1. **Direction model** — predicts probability of up-move vs down-move over a
   defined forward horizon (e.g. next N candles on a given timeframe).
2. **Regime model** — predicts trend / range / choppy classification,
   complementing (or replacing over time) the rule-based regime detector.
3. **Confidence/ranking model** — given a candidate setup (rule-based signal
   + direction/regime model outputs), predicts probability the trade hits
   target before stop — i.e. "is this setup actually worth taking."

### 8.2 Features
Engineered from data already produced elsewhere in the pipeline — no
duplicate logic:
- Technical indicators across multiple timeframes (from `analysis/indicators.py`).
- Pattern-detector confidence scores (from `analysis/patterns/`).
- Regime classification and its confidence (from `analysis/regime_detection.py`).
- Session context (which session, historical volatility for that session).
- Price-derived features: returns, volatility (ATR-normalized), distance to
  key S/R levels, candle structure stats.
- Avoid features that leak future information (e.g. anything computed using
  data after the prediction point) — this is the single most common cause of
  misleadingly good backtests.

### 8.3 Labeling
- Direction model: label using forward returns over a fixed horizon, or a
  **triple-barrier method** (label = which is hit first: take-profit level,
  stop-loss level, or time expiry) — triple-barrier is generally more
  realistic for trading than fixed-horizon return classification.
- Regime model: label historical windows using a clear, reproducible rule
  (e.g. ADX + MA slope thresholds) so training labels are consistent, then
  let the model generalize beyond the rule's edge cases.
- Confidence model: label = actual outcome (win/loss, or R-multiple) of
  each historical rule-based signal, so it learns to filter good setups from
  bad ones.

### 8.4 Model choice
- Start with **gradient-boosted trees** (XGBoost or LightGBM) on the
  engineered tabular features — strong baseline, fast to train, interpretable
  via feature importance/SHAP, doesn't need a GPU.
- Sequence models (LSTM/GRU/Transformer) on raw OHLCV can be explored later
  as an upgrade path, but add real complexity (more data, more compute,
  harder to validate) — don't start there. **Ask the user** before investing
  in this path, since it changes infra requirements (GPU, more data).

### 8.5 Validation (critical for trading models)
- Use **walk-forward validation**, not random k-fold — trading data is
  time-ordered, and random splits leak future information into training.
- Use a **purge/embargo gap** between train and test windows so overlapping
  labels (e.g. triple-barrier windows) don't leak across the split.
- Report performance **per regime and per session**, not just in aggregate —
  a model can look good overall while being unprofitable in the exact
  conditions it'll actually be used in live.
- Treat any backtest Sharpe/win-rate that looks "too good" as a signal to
  check for leakage before celebrating it.

### 8.6 Retraining & model management
- Models are versioned in `ml/registry/` with metadata: training date, date
  range of training data, feature set version, validation metrics.
- Define a retraining cadence (e.g. weekly/monthly) — **ask the user** for
  their preference, since this affects how "adaptive" vs "stable" the bot's
  behavior is.
- New model versions must pass validation thresholds (defined in config)
  before being promoted to live inference — never auto-promote a model that
  underperforms the currently live one without a human checking it first.
- Monitor for **drift**: track live prediction confidence and realized
  outcomes over time; alert if the model's live performance diverges
  meaningfully from its validation performance.

### 8.7 Open questions to ask the user before building this layer
- Approximate size/history of data available to train on (affects which
  models are even viable).
- Whether they have (or want) GPU compute, or should everything stay
  CPU-friendly (gradient boosting).
- Preferred retraining cadence.
- Whether they want the ML layer to be a **hard gate** (no trade unless
  model agrees) or a **confidence weight** (model output nudges position
  size/confidence but rule-based signals can still fire alone).

---

## 9. Signal Engine

- Combines regime + pattern + timeframe confluence + session context +
  **ML model outputs** (direction, regime, confidence probabilities) into a
  single trade signal: `LONG`, `SHORT`, or `NO_TRADE`, with a confidence
  score and the reasoning that produced it (for the trade journal/logs).
- Signal thresholds, weightings, and which detectors/models are "required"
  vs "supporting" must be configurable — do not hardcode magic numbers deep
  in logic; put them in `config/`.
- No trade should ever be placed without a traceable reason logged
  (which regime, which pattern, which timeframes agreed, what the model(s)
  predicted and with what confidence).

---

## 10. Risk Management & Execution

- **Position sizing:** calculate lot size from account balance, a
  configurable risk-per-trade % (e.g. default 1%, but must be user-set, not
  assumed), and stop-loss distance.
- **Stop loss / take profit:** structure-based (beyond recent
  swing/support-resistance) or ATR-based; always required, never optional.
- **Max concurrent positions** and **max correlated exposure** (e.g. don't
  open five correlated forex pairs in the same direction without the user
  explicitly allowing it).
- **Daily/weekly loss limits** that halt new trades once breached.
- **Kill switch:** a manual and automatic mechanism to immediately stop all
  new order placement and optionally flatten open positions.
- **Order execution:** market or limit orders via MT5, with retry/error
  handling for requotes, slippage, and rejected orders. Log every order
  attempt and result.

---

## 11. Backtesting & Validation

- Every strategy change must be backtestable on historical data before
  being considered for demo, and demo-tested before being considered for
  live.
- Backtest engine should report: win rate, profit factor, max drawdown,
  Sharpe/Sortino if feasible, and per-regime/session performance breakdown
  (e.g. "this strategy performs well in trending sessions but poorly while
  ranging").
- Flag overfitting risk explicitly if a strategy is tuned tightly to
  historical data — suggest out-of-sample/walk-forward validation.

---

## 12. Logging, Monitoring, Journaling

- Every decision (trade or no-trade) should be logged with: timestamp,
  symbol, timeframe(s), regime, pattern(s) detected, confidence score,
  action taken, resulting order ID (if any).
- Trade journal should be queryable/exportable (CSV) for post-hoc review.
- Alerting: on connection loss, kill-switch trigger, repeated order
  failures, or daily loss limit breach, the bot should notify the user
  (log at minimum; email/Telegram/webhook alerting can be added later —
  ask the user if they want this before building it).

---

## 13. When the Agent Should Ask Questions

Stop and ask the user (don't assume) whenever a task touches:

- Which specific forex pairs / crypto symbols to trade.
- What OS/environment the bot will run in (MT5 Python API needs Windows or
  a Windows-compatible bridge).
- Default risk-per-trade %, max daily loss %, max open positions.
- Whether to net or hedge positions (MT5 supports both depending on
  broker/account mode).
- Whether news-event filtering (e.g. avoiding trades around high-impact
  news releases) is in scope.
- Whether alerting/notifications (Telegram, email, etc.) are wanted.
- Any time a requested change would bypass or weaken risk management or the
  kill-switch — flag this explicitly and confirm before implementing.
- How much historical data is available/desired for model training, and
  retraining cadence (see Section 8.7).
- Whether the ML layer should be a hard gate on trades or a confidence
  weight alongside rule-based signals (see Section 8.7).
- Whether GPU/sequence models are worth the added complexity, or whether to
  stay with CPU-friendly gradient boosting.

---

## 14. Definition of Done (per milestone)

A feature/module is not "done" until:
1. It has unit tests covering the core logic (especially regime/pattern
   detection and risk/position-sizing math).
2. It runs successfully against demo data or a demo MT5 account.
3. Any ML model involved has passed walk-forward validation with a
   purge/embargo gap, and its metrics (per Section 8.5) are recorded in the
   model registry before being used for a live/demo decision.
4. It logs its decisions in a way a human can audit after the fact.
5. Config values it depends on are documented in `README.md`/`.env.example`,
   not buried in code.

---

## 15. Disclaimer (keep visible in README)

This bot executes real financial trades and can lose money. Past
performance in backtests does not guarantee future results. Test
extensively on a demo account before enabling live trading, and never risk
funds you cannot afford to lose.