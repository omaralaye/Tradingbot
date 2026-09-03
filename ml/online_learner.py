"""
ml/online_learner.py
--------------------
Online Continuous Learning & Real-Time Adaptive Feedback Brain.

Dynamically adapts trading decision boundaries after every live trade outcome:
  1. Setup Memory & Dynamic Scoring: Tracks performance per setup fingerprint
     [Symbol + Direction + H4/H1 Regime + Session + Patterns] using Bayesian Beta
     priors and multi-armed bandit logic.
  2. Incremental Machine Learning: Uses an online SGD classifier with partial_fit()
     to adjust decision weights in real time without waiting for batch retrains.
  3. Adaptive Cooldown & Circuit Breaker: Automatically quarantines underperforming
     setups or symbols that suffer repeated losses (e.g. counter-trend chop).
  4. State Persistence & Bootstrap: Stores memory in logs/online_memory.json and
     can replay historical closed trades from MT5 deal history to start warm.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger
from sklearn.linear_model import SGDClassifier


@dataclass
class SetupEvaluation:
    """The result of evaluating a proposed trade setup against online learning memory."""
    is_allowed:            bool
    action:                str    # "ALLOW", "BOOST", "PENALIZE", "VETO"
    confidence_multiplier: float  # 0.0 to 1.30
    bayesian_win_prob:     float
    ml_win_prob:           float
    is_quarantined:        bool
    quarantine_reason:     str    = ""
    setup_key:             str    = ""
    consecutive_losses:    int    = 0
    reason:                str    = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OnlineAdaptiveLearner:
    """Real-time online adaptive learner for live and demo trading.

    Maintains dynamic setup stats, Bayesian win rate estimations, quarantine
    cooldowns, and an incremental linear model (SGDClassifier with log_loss)
    that learns after every closed trade.
    """

    FEATURE_NAMES = [
        "h4_trend",          # +1: uptrend, -1: downtrend, 0: ranging/other
        "h1_trend",          # +1: uptrend, -1: downtrend, 0: ranging/other
        "direction",         # +1: LONG, -1: SHORT
        "trend_alignment",   # direction * h4_trend (+1: trend-following, -1: counter-trend)
        "base_confidence",   # 0.0 - 1.0
        "tf_agreement",      # 0.0 - 1.0
        "session_volatility",# 1.0: high, 0.5: medium, 0.0: low
        "has_pattern",       # 1.0: pattern present, 0.0: none
    ]

    def __init__(
        self,
        memory_path: str = "logs/online_memory.json",
        quarantine_loss_streak: int = 2,
        quarantine_duration_seconds: int = 14400,  # 4 hours
        min_trades_for_stats: int = 2,
    ):
        """
        Args:
            memory_path: File path to save/load persistent online memory.
            quarantine_loss_streak: Number of consecutive losses to trigger setup quarantine.
            quarantine_duration_seconds: How long a setup is quarantined (seconds).
            min_trades_for_stats: Minimum trade samples before applying heavy Bayesian penalties.
        """
        self._memory_path = Path(memory_path)
        self._quarantine_loss_streak = quarantine_loss_streak
        self._quarantine_duration = quarantine_duration_seconds
        self._min_trades = min_trades_for_stats

        # Setup stats: key -> {wins, losses, consecutive_losses, consecutive_wins, net_pnl, ...}
        self._setup_stats: dict[str, dict[str, Any]] = {}
        # Quarantined keys: key -> {quarantined_at, quarantine_until, reason, loss_streak}
        self._quarantines: dict[str, dict[str, Any]] = {}
        # List of processed trade IDs to prevent duplicate learning
        self._processed_trade_ids: set[int] = set()

        # Incremental online classifier
        self._clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            learning_rate="optimal",
            random_state=42,
        )
        self._is_clf_fitted = False

        # Try to load existing memory
        self.load()

    # ------------------------------------------------------------------
    # Feature Extraction for Online Model
    # ------------------------------------------------------------------

    def extract_features(
        self,
        direction: str,
        h4_regime: str,
        h1_regime: str,
        confidence: float,
        tf_agreement: float = 0.67,
        session_volatility: str = "medium",
        has_pattern: bool = False,
    ) -> np.ndarray:
        """Extract normalised feature vector for the online classifier."""
        dir_val = 1.0 if str(direction).upper() in ("LONG", "BUY") else -1.0

        def encode_regime(reg: str) -> float:
            reg_lower = str(reg).lower()
            if "up" in reg_lower:
                return 1.0
            if "down" in reg_lower:
                return -1.0
            return 0.0

        h4_val = encode_regime(h4_regime)
        h1_val = encode_regime(h1_regime)
        trend_alignment = dir_val * h4_val  # +1 if trend-following, -1 if counter-trend

        vol_map = {"high": 1.0, "medium": 0.5, "low": 0.0}
        vol_val = vol_map.get(str(session_volatility).lower(), 0.5)
        pat_val = 1.0 if has_pattern else 0.0

        return np.array([
            h4_val,
            h1_val,
            dir_val,
            trend_alignment,
            float(np.clip(confidence, 0.0, 1.0)),
            float(np.clip(tf_agreement, 0.0, 1.0)),
            vol_val,
            pat_val,
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    # Setup Fingerprints
    # ------------------------------------------------------------------

    @staticmethod
    def get_setup_keys(
        symbol: str,
        direction: str,
        h4_regime: str = "",
        session: str = "",
        pattern: str = "",
    ) -> list[str]:
        """Generate hierarchical setup keys from general to specific.

        Hierarchy:
          1. symbol:direction (e.g. 'EURUSD:LONG')
          2. symbol:direction:h4_regime (e.g. 'EURUSD:LONG:downtrend') -> catches counter-trend traps
          3. symbol:pattern (e.g. 'EURUSD:pin_bar')
          4. symbol:direction:session (e.g. 'EURUSD:LONG:tokyo')
        """
        sym = symbol.upper()
        d = "LONG" if str(direction).upper() in ("LONG", "BUY") else "SHORT"
        keys = [f"{sym}:{d}"]

        if h4_regime:
            reg_clean = "uptrend" if "up" in h4_regime.lower() else ("downtrend" if "down" in h4_regime.lower() else "ranging")
            keys.append(f"{sym}:{d}:{reg_clean}")

        if pattern and pattern.lower() not in ("none", "[]", ""):
            keys.append(f"{sym}:{pattern.lower()}")

        if session:
            sess_clean = session.lower().replace("[", "").replace("]", "").replace("'", "").strip()
            if sess_clean:
                keys.append(f"{sym}:{d}:{sess_clean}")

        return keys

    # ------------------------------------------------------------------
    # Setup Evaluation & Gating
    # ------------------------------------------------------------------

    def evaluate_setup(
        self,
        symbol: str,
        direction: str,
        h4_regime: str = "unknown",
        h1_regime: str = "unknown",
        session: str = "",
        session_volatility: str = "medium",
        patterns: list | None = None,
        confidence: float = 0.65,
        tf_agreement: float = 0.67,
    ) -> SetupEvaluation:
        """Evaluate a proposed trade setup against dynamic memory.

        Returns SetupEvaluation with decision, confidence multiplier, and reason.
        """
        now = time.time()
        patterns = patterns or []
        pat_name = patterns[0] if patterns else ""
        keys = self.get_setup_keys(symbol, direction, h4_regime, session, pat_name)
        primary_key = keys[1] if len(keys) > 1 else keys[0]

        # 1. Check for active quarantines
        for k in keys:
            if k in self._quarantines:
                q = self._quarantines[k]
                if now < q["quarantine_until"]:
                    remaining_min = (q["quarantine_until"] - now) / 60
                    reason = f"Setup '{k}' quarantined for {remaining_min:.0f}m more. Reason: {q['reason']}"
                    logger.warning("OnlineLearner VETO: {}", reason)
                    return SetupEvaluation(
                        is_allowed=False,
                        action="VETO",
                        confidence_multiplier=0.0,
                        bayesian_win_prob=0.0,
                        ml_win_prob=0.0,
                        is_quarantined=True,
                        quarantine_reason=reason,
                        setup_key=k,
                        consecutive_losses=q.get("loss_streak", 0),
                        reason=reason,
                    )
                else:
                    # Quarantine expired! Remove it cleanly
                    logger.info("OnlineLearner: Quarantine expired for '{}'. Lifting quarantine.", k)
                    del self._quarantines[k]

        # 2. Bayesian win rate estimation across matching keys
        # Beta prior: alpha=2, beta=2 (prior win prob = 50%)
        alpha_prior = 2.0
        beta_prior = 2.0
        tot_wins = 0
        tot_losses = 0
        max_loss_streak = 0

        for k in keys:
            stats = self._setup_stats.get(k)
            if stats:
                tot_wins += stats.get("wins", 0)
                tot_losses += stats.get("losses", 0)
                max_loss_streak = max(max_loss_streak, stats.get("consecutive_losses", 0))

        bayesian_win_prob = (alpha_prior + tot_wins) / (alpha_prior + beta_prior + tot_wins + tot_losses)

        # 3. Incremental ML Online Classifier prediction
        features = self.extract_features(
            direction=direction,
            h4_regime=h4_regime,
            h1_regime=h1_regime,
            confidence=confidence,
            tf_agreement=tf_agreement,
            session_volatility=session_volatility,
            has_pattern=len(patterns) > 0,
        )

        ml_win_prob = 0.50
        if self._is_clf_fitted:
            try:
                probs = self._clf.predict_proba([features])[0]
                # classes are [0, 1] -> index 1 is win prob
                ml_win_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
            except Exception as exc:
                logger.debug("Online classifier proba error: {}", exc)
                ml_win_prob = 0.50

        # 4. Check Counter-Trend Warning & Gating
        dir_val = 1.0 if str(direction).upper() in ("LONG", "BUY") else -1.0
        h4_val = 1.0 if "up" in h4_regime.lower() else (-1.0 if "down" in h4_regime.lower() else 0.0)
        is_counter_trend = (dir_val * h4_val) < 0

        # 5. Composite Action Determination
        total_samples = tot_wins + tot_losses
        multiplier = 1.0
        action = "ALLOW"
        decision_reason = f"Normal evaluation. Bayesian win prob: {bayesian_win_prob:.2f}, ML win prob: {ml_win_prob:.2f}."

        # Hard guard against counter-trend setups with poor track record
        if is_counter_trend and total_samples >= 2 and bayesian_win_prob < 0.40:
            reason = f"Counter-trend setup {primary_key} rejected (H4 bias conflicts and historical win rate is {bayesian_win_prob:.1%})."
            logger.warning("OnlineLearner VETO: {}", reason)
            return SetupEvaluation(
                is_allowed=False,
                action="VETO",
                confidence_multiplier=0.0,
                bayesian_win_prob=bayesian_win_prob,
                ml_win_prob=ml_win_prob,
                is_quarantined=False,
                setup_key=primary_key,
                consecutive_losses=max_loss_streak,
                reason=reason,
            )

        if total_samples >= self._min_trades:
            if bayesian_win_prob < 0.38 or ml_win_prob < 0.38:
                multiplier = 0.65
                action = "PENALIZE"
                decision_reason = (
                    f"Setup '{primary_key}' penalized (win prob: {bayesian_win_prob:.1%}, "
                    f"ML: {ml_win_prob:.1%}, loss streak: {max_loss_streak})."
                )
            elif bayesian_win_prob >= 0.62 and ml_win_prob >= 0.55:
                multiplier = 1.15
                action = "BOOST"
                decision_reason = (
                    f"Setup '{primary_key}' boosted (proven win rate: {bayesian_win_prob:.1%}, "
                    f"ML: {ml_win_prob:.1%})."
                )

        logger.debug(
            "OnlineLearner: {} -> action={}, mult={:.2f}, bayes={:.2f}, ml={:.2f}",
            primary_key, action, multiplier, bayesian_win_prob, ml_win_prob,
        )

        return SetupEvaluation(
            is_allowed=True,
            action=action,
            confidence_multiplier=multiplier,
            bayesian_win_prob=bayesian_win_prob,
            ml_win_prob=ml_win_prob,
            is_quarantined=False,
            setup_key=primary_key,
            consecutive_losses=max_loss_streak,
            reason=decision_reason,
        )

    # ------------------------------------------------------------------
    # Trade Closed Callback & Incremental Learning
    # ------------------------------------------------------------------

    def on_trade_closed(self, trade: dict[str, Any]) -> dict[str, Any]:
        """Called immediately whenever a trade is closed.

        Updates setup memory, adjusts Bayesian stats, updates incremental ML model,
        and triggers quarantine if necessary.

        Args:
            trade: Dict containing:
              - ticket or position_id (int)
              - symbol (str)
              - direction ('buy' or 'sell' / 'LONG' or 'SHORT')
              - net_pnl (float)
              - h4_regime (str, optional)
              - h1_regime (str, optional)
              - session (str, optional)
              - patterns (list/str, optional)
              - confidence (float, optional)
              - exit_reason (str, optional)
        """
        ticket = int(trade.get("ticket") or trade.get("position_id", 0))
        if ticket > 0 and ticket in self._processed_trade_ids:
            logger.debug("OnlineLearner: Trade {} already processed. Skipping duplicate learning.", ticket)
            return {"status": "skipped", "reason": "duplicate_trade"}

        symbol = str(trade.get("symbol", "")).upper().replace("M", "")  # normalize broker suffix
        direction = str(trade.get("direction", "buy")).upper()
        dir_label = "LONG" if direction in ("BUY", "LONG") else "SHORT"
        net_pnl = float(trade.get("net_pnl", trade.get("profit", 0.0)))
        is_win = net_pnl > 0.0
        y = 1 if is_win else 0

        h4_regime = str(trade.get("h4_regime", "unknown"))
        h1_regime = str(trade.get("h1_regime", "unknown"))
        session = str(trade.get("session", ""))
        patterns = trade.get("patterns", [])
        if isinstance(patterns, str):
            patterns = [p.strip().strip("'\"") for p in patterns.replace("[", "").replace("]", "").split(",") if p.strip()]
        pat_name = patterns[0] if patterns else ""
        confidence = float(trade.get("confidence", 0.65))

        keys = self.get_setup_keys(symbol, dir_label, h4_regime, session, pat_name)
        now = time.time()
        quarantine_triggered = False

        for k in keys:
            if k not in self._setup_stats:
                self._setup_stats[k] = {
                    "wins": 0,
                    "losses": 0,
                    "consecutive_wins": 0,
                    "consecutive_losses": 0,
                    "net_pnl": 0.0,
                    "last_updated": now,
                }
            st = self._setup_stats[k]
            st["net_pnl"] = round(st["net_pnl"] + net_pnl, 2)
            st["last_updated"] = now

            if is_win:
                st["wins"] += 1
                st["consecutive_wins"] += 1
                st["consecutive_losses"] = 0
                # Clear quarantine if active (redemption)
                if k in self._quarantines:
                    del self._quarantines[k]
                    logger.info("OnlineLearner: Quarantine lifted on '{}' due to winning resolution.", k)
            else:
                st["losses"] += 1
                st["consecutive_losses"] += 1
                st["consecutive_wins"] = 0

                # Check quarantine condition: 2 or more consecutive losses on this setup
                if st["consecutive_losses"] >= self._quarantine_loss_streak:
                    quarantine_until = now + self._quarantine_duration
                    self._quarantines[k] = {
                        "quarantined_at": now,
                        "quarantine_until": quarantine_until,
                        "reason": f"{st['consecutive_losses']} consecutive losses (Net PnL: ${st['net_pnl']:.2f})",
                        "loss_streak": st["consecutive_losses"],
                    }
                    quarantine_triggered = True
                    logger.warning(
                        "🚨 OnlineLearner QUARANTINE: Setup '{}' quarantined for {:.1f}h! "
                        "Reason: {} consecutive losses.",
                        k, self._quarantine_duration / 3600, st["consecutive_losses"],
                    )

        # Update online incremental classifier
        features = self.extract_features(
            direction=dir_label,
            h4_regime=h4_regime,
            h1_regime=h1_regime,
            confidence=confidence,
            tf_agreement=0.67,
            session_volatility="medium",
            has_pattern=len(patterns) > 0,
        )

        try:
            self._clf.partial_fit([features], [y], classes=[0, 1])
            self._is_clf_fitted = True
        except Exception as exc:
            logger.error("OnlineLearner: Error during partial_fit: {}", exc)

        if ticket > 0:
            self._processed_trade_ids.add(ticket)

        self.save()

        logger.info(
            "OnlineLearner updated: {} {} | PnL: ${:.2f} ({}) | Quarantine active: {}",
            symbol, dir_label, net_pnl, "WIN" if is_win else "LOSS", quarantine_triggered,
        )

        return {
            "status": "updated",
            "symbol": symbol,
            "direction": dir_label,
            "net_pnl": net_pnl,
            "outcome": "win" if is_win else "loss",
            "quarantine_triggered": quarantine_triggered,
        }

    # ------------------------------------------------------------------
    # Bootstrap / Replay History
    # ------------------------------------------------------------------

    def bootstrap_from_history(self, trades: list[dict[str, Any]]) -> int:
        """Replay historical closed trades chronologically to warm up memory.

        Returns count of newly learned trades.
        """
        count = 0
        logger.info("OnlineLearner: Bootstrapping from {} historical trades...", len(trades))
        for t in trades:
            res = self.on_trade_closed(t)
            if res.get("status") == "updated":
                count += 1
        logger.info("OnlineLearner: Bootstrap complete. Learned from {} trades.", count)
        return count

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """Save learner memory to JSON."""
        target = Path(path) if path else self._memory_path
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "setup_stats": self._setup_stats,
            "quarantines": self._quarantines,
            "processed_trade_ids": list(self._processed_trade_ids),
            "is_clf_fitted": self._is_clf_fitted,
            "clf_coef": self._clf.coef_.tolist() if self._is_clf_fitted else None,
            "clf_intercept": self._clf.intercept_.tolist() if self._is_clf_fitted else None,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(target, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug("OnlineLearner: Memory saved to {}", target)
        except Exception as exc:
            logger.error("OnlineLearner: Failed to save memory: {}", exc)

    def load(self, path: Optional[str] = None) -> bool:
        """Load learner memory from JSON if present."""
        target = Path(path) if path else self._memory_path
        if not target.exists():
            logger.info("OnlineLearner: No existing memory file at {}. Starting fresh.", target)
            return False

        try:
            with open(target) as f:
                data = json.load(f)
            self._setup_stats = data.get("setup_stats", {})
            self._quarantines = data.get("quarantines", {})
            self._processed_trade_ids = set(data.get("processed_trade_ids", []))
            self._is_clf_fitted = data.get("is_clf_fitted", False)

            if self._is_clf_fitted and data.get("clf_coef") is not None:
                # Re-initialize classifier and load weights
                self._clf.classes_ = np.array([0, 1])
                self._clf.coef_ = np.array(data["clf_coef"], dtype=np.float64)
                self._clf.intercept_ = np.array(data["clf_intercept"], dtype=np.float64)

            logger.info(
                "OnlineLearner: Loaded memory with {} setups, {} quarantines, {} processed trades.",
                len(self._setup_stats), len(self._quarantines), len(self._processed_trade_ids),
            )
            return True
        except Exception as exc:
            logger.warning("OnlineLearner: Could not load memory from {}: {}. Starting fresh.", target, exc)
            return False

    def get_summary(self) -> dict[str, Any]:
        """Return human-readable summary of learning memory."""
        active_quarantines = {}
        now = time.time()
        for k, v in self._quarantines.items():
            if now < v["quarantine_until"]:
                active_quarantines[k] = {
                    "remaining_minutes": round((v["quarantine_until"] - now) / 60, 1),
                    "reason": v["reason"],
                }

        top_winning_setups = sorted(
            [
                (k, v["wins"], v["losses"], v["net_pnl"])
                for k, v in self._setup_stats.items()
                if v["wins"] > 0
            ],
            key=lambda x: x[3],
            reverse=True,
        )[:5]

        top_losing_setups = sorted(
            [
                (k, v["wins"], v["losses"], v["net_pnl"])
                for k, v in self._setup_stats.items()
                if v["losses"] > 0
            ],
            key=lambda x: x[3],
        )[:5]

        return {
            "total_tracked_setups": len(self._setup_stats),
            "total_processed_trades": len(self._processed_trade_ids),
            "active_quarantines": active_quarantines,
            "top_winning_setups": top_winning_setups,
            "top_losing_setups": top_losing_setups,
            "is_clf_fitted": self._is_clf_fitted,
        }


def bootstrap_from_deals_and_journal(
    learner: OnlineAdaptiveLearner,
    connector,
    journal_path: str = "logs/trade_journal.csv",
    lookback_days: int = 30,
) -> int:
    """Extract closed deals from MT5 and enrich them with journal context to bootstrap learner.

    Returns:
        Number of historical closed trades learned.
    """
    if not connector.ensure_connected():
        logger.warning("OnlineLearner bootstrap: MT5 not connected.")
        return 0

    try:
        mt5 = connector.mt5
        now = time.time()
        from_date = now - lookback_days * 86400
        to_date = now + 86400
        deals = mt5.history_deals_get(from_date, to_date)
        if not deals:
            logger.info("OnlineLearner bootstrap: No historical deals found.")
            return 0

        deals_data = [
            d._asdict() if hasattr(d, "_asdict") else {a: getattr(d, a) for a in dir(d) if not a.startswith("_")}
            for d in deals
        ]

        # Group by position_id
        by_pos: dict[int, list[dict]] = {}
        for d in deals_data:
            pid = int(d.get("position_id", 0))
            if pid > 0 and d.get("type") in (0, 1):  # 0=BUY, 1=SELL
                by_pos.setdefault(pid, []).append(d)

        # Load journal to match tickets to signal context
        ticket_to_signal: dict[int, dict] = {}
        if os.path.exists(journal_path):
            try:
                import pandas as pd
                df_j = pd.read_csv(journal_path)
                for i in range(len(df_j)):
                    row = df_j.iloc[i]
                    if row["action"] == "order_result":
                        t_val = row.get("order_ticket")
                        try:
                            t_num = int(float(t_val))
                        except (ValueError, TypeError):
                            continue
                        for j in range(i - 1, max(-1, i - 10), -1):
                            prev = df_j.iloc[j]
                            if prev.get("symbol") == row.get("symbol") and prev.get("action") in ("order_placed", "skipped"):
                                ticket_to_signal[t_num] = prev.to_dict()
                                break
            except Exception as exc:
                logger.warning("OnlineLearner bootstrap: Error reading journal {}: {}", journal_path, exc)

        # Build trade dicts
        closed_trades = []
        for pid, p_deals in by_pos.items():
            in_deals = [d for d in p_deals if d.get("entry") == 0]
            out_deals = [d for d in p_deals if d.get("entry") == 1]
            if not in_deals or not out_deals:
                continue

            in_d = in_deals[0]
            out_d = out_deals[-1]
            net_pnl = float(sum(d.get("profit", 0.0) + d.get("commission", 0.0) + d.get("swap", 0.0) for d in out_deals))

            sig = ticket_to_signal.get(pid, {})
            broker_sym = in_d.get("symbol", "")
            base_sym = broker_sym.rstrip("m").rstrip(".raw").rstrip(".pro") if broker_sym else ""

            trade_dict = {
                "ticket": pid,
                "position_id": pid,
                "symbol": base_sym or broker_sym,
                "direction": "LONG" if in_d.get("type") == 0 else "SHORT",
                "net_pnl": net_pnl,
                "h4_regime": sig.get("regime", "unknown"),
                "h1_regime": sig.get("regime", "unknown"),
                "session": str(sig.get("session", "")),
                "patterns": sig.get("patterns", []),
                "confidence": float(sig.get("confidence", 0.65)) if sig.get("confidence") else 0.65,
                "exit_time": out_d.get("time", 0),
            }

            # Parse reasoning_json if present for richer H4 regime
            if "reasoning_json" in sig and isinstance(sig["reasoning_json"], str):
                try:
                    r_json = json.loads(sig["reasoning_json"])
                    steps = r_json.get("steps", {})
                    if "regime_h4" in steps:
                        trade_dict["h4_regime"] = steps["regime_h4"].get("regime", trade_dict["h4_regime"])
                    if "regime_h1" in steps:
                        trade_dict["h1_regime"] = steps["regime_h1"].get("regime", trade_dict["h1_regime"])
                except Exception:
                    pass

            closed_trades.append(trade_dict)

        # Sort chronologically by exit time
        closed_trades.sort(key=lambda x: x.get("exit_time", 0))
        return learner.bootstrap_from_history(closed_trades)
    except Exception as exc:
        logger.error("OnlineLearner bootstrap error: {}", exc)
        return 0

