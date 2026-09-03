# Task: Implement Online Continuous Learning Loop for Real-Time Trade Adaptation

## Original User Prompt
```
study the data in the logs folder and explain the wins nad losses without modifying or changing any file
```
Followed by:
```
i want the bot to train on the data to make sure that the losses are not repeated again and the wins to be mained, is that possible?
```
Followed by clarifying interaction:
- User selected: `I want an online / continuous learning loop where the bot adapts dynamically after every live trade outcome.`
```
proceed with the implementation
```
Followed by continuation prompt:
```
continue with the previous task
```

## Context Gathered
- The user studied the log and MT5 deal history (32 closed trades: 12 wins, 20 losses; 6 open positions; account balance grew from $10.00 to $16.27).
- Analysis revealed that the bot was profitable due to a 2.11:1 win/loss payoff ratio, but suffered from:
  1. No post-order tracking: `live/run_bot.py` never monitored deals when positions closed, leaving `trade_journal.csv` PnL empty.
  2. Immediate stop-outs (1–2 minutes) from tight ATR stops during low-volatility chop.
  3. Counter-trend losses (e.g. EURUSD Long against strong H4 downtrend).
  4. Rapid duplicate orders on the same symbol 60 seconds apart.
- Rather than a static offline retrain or manual rules alone, the user chose an **online continuous learning loop** where the bot dynamically updates its state, setup scores, and incremental ML weights after every live trade outcome.

## Clarifying Questions Asked & Answers
- **Question:** What specific approach did you have in mind when you said 'none of the above'?
- **Answer:** `I want an online / continuous learning loop where the bot adapts dynamically after every live trade outcome.`

## Implementation Plan
1. **Create `ml/online_learner.py`**:
   - `OnlineAdaptiveLearner` class managing dynamic setup memory, Bayesian/Bandit setup scoring, quarantine cooldowns, and incremental ML model updates (`SGDClassifier` with `partial_fit`).
   - Persistent state in `logs/online_memory.json` so learning survives restarts.
   - Initial bootstrap/replay method to learn from all 32 past closed trades from MT5 deal history upon startup.
   - Filter/scoring API: `evaluate_setup()` to approve, penalize, or quarantine proposed signals.
2. **Implement Closed-Trade Listener in `execution/order_manager.py` & `live/run_bot.py`**:
   - Add `sync_closed_positions()` in `OrderManager` to detect when tracked open positions close via MT5 history deals (`history_deals_get`).
   - Extract exit price, exit time, net PnL, pip return, and exit reason (`[tp ...]` or `[sl ...]`).
   - Update `TradeJournal` with the realized PnL and outcome.
   - Feed the closed trade outcome directly into `OnlineAdaptiveLearner.on_trade_closed()`.
3. **Integrate Online Learner into `signals/signal_engine.py`**:
   - Query `OnlineAdaptiveLearner.evaluate_setup()` before generating `LONG`/`SHORT` signals.
   - Quarantine or reject setups with a decaying/penalized win rate or consecutive losses.
   - Boost confidence for high-performing setups.
4. **Implement Algorithmic Guards into `risk/risk_manager.py` / `risk/stop_target_logic.py`**:
   - Symbol cooldown to prevent duplicate entries on the same symbol within 30 minutes.
   - Minimum stop-loss distance floor (e.g., minimum 12–15 pips) to prevent 1-minute spread stop-outs.
5. **Add Comprehensive Tests**:
   - Unit tests for `OnlineAdaptiveLearner`, `sync_closed_positions`, and signal integration in `tests/test_online_learner.py`.
   - Run `pytest` to ensure all existing tests and new tests pass.
